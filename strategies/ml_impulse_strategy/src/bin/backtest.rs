// strategies/ml_impulse_strategy/src/bin/backtest.rs
//
// Impulse Absorption & Engulfing — Walk-Forward Trade Simulation
//
// NO LOOK-AHEAD BIAS:
//   - Walks bar-by-bar on target TFs (15m, 1h, 4h)
//   - Heuristic detects engulfing pattern using ONLY current + previous bar
//   - Features extracted from PAST candles only (lookback window BEFORE signal)
//   - ML model predicts probability of success
//   - If pred >= threshold → open virtual trade
//   - SL/TP checked on future bars within max_hold=3 window
//   - Conservative: if SL & TP both hit on same bar → SL wins
//
// PERFORMANCE:
//   - Batch model predictions (256 rows at once via XGBoost)
//   - Sequential symbol processing (memory-safe)
//   - sqlx slow statement warnings suppressed
//
// USAGE:
//   cargo build --release -p ml_impulse_strategy --bin impulse_backtest
//   ./target/release/impulse_backtest
//
// ENV VARS:
//   DATABASE_URL              — postgres connection
//   IAE_HEURISTIC_ONLY=1       — skip ML, trade ALL heuristic signals (pred=1.0)
//   IAE_LONG_MODEL_PATH       — long model (default: models/impulse_long_v1.ubj)
//   IAE_SHORT_MODEL_PATH      — short model (default: models/impulse_short_v1.ubj)
//   IAE_MODEL_TYPE             — "long", "short", or "both" (default: "both")
//   IAE_MAX_HOLD               — max candles to hold (default: 8)
//   IAE_PRED_STEP              — predict every N bars (default: 1)
//   IAE_COOLDOWN               — bars cooldown after trade (default: max_hold)
//   IAE_MIN_SIGNAL             — minimum pred to record (default: 0.50)
//   WFO_MIN_DATE               — only test after this date (YYYY-MM-DD)
//   DIRECTION_GPU=1            — use GPU for inference

use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use dotenvy::dotenv;
use sqlx::PgPool;
use std::collections::HashMap;
use tracing::info;

use ml_impulse_strategy::impulse::{
    CandleInd, SignalDir, ImpulseConfig, EngulfingSignal,
    ANALYSIS_TIMEFRAMES, FULL_FEATURES_PER_CANDLE,
    ENGULFING_FEATURE_COUNT,
    extract_candle_features, extract_engulfing_features,
    check_engulfing_at_bar,
};
use ml_impulse_strategy::dataset::fetch_all_candles_for_tf;

use predictors::ml::xgb_runtime::{Booster, Device, ModelKind};

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

const PRED_BATCH_SIZE: usize = 256;
const PRED_THRESHOLDS: &[f32] = &[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90];

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TradeOutcome {
    TpHit,
    SlHit,
    MaxHoldExpired,
}

impl std::fmt::Display for TradeOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TpHit => write!(f, "✅TP"),
            Self::SlHit => write!(f, "❌SL"),
            Self::MaxHoldExpired => write!(f, "⏰EXP"),
        }
    }
}

#[derive(Debug, Clone)]
struct SimTrade {
    symbol: String,
    tf_minutes: i32,
    direction: SignalDir,
    entry_time: DateTime<Utc>,
    entry_price: f64,
    exit_price: f64,
    pred: f32,
    outcome: TradeOutcome,
    hold_candles: usize,
    pnl_pct: f64,
    impulse_pct: f64,
}

// ─────────────────────────────────────────────────────────────────────
// Heuristic check: uses check_engulfing_at_bar from impulse.rs
// v3: ATR-based threshold, multi-bar absorption, micro-gap leniency,
//     volume on EITHER impulse OR absorption candle
// ─────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────
// Feature extraction (no look-ahead)
// ─────────────────────────────────────────────────────────────────────

/// Extract multi-TF features anchored at `bar_idx` on `target_tf`.
/// Features from lookback candles BEFORE bar_idx + engulfing meta.
fn extract_features_at_bar(
    all_tf_candles: &HashMap<i32, Vec<CandleInd>>,
    target_tf: i32,
    signal: &EngulfingSignal,
    lookback: usize,
) -> Option<Vec<f64>> {
    let n_tfs = ANALYSIS_TIMEFRAMES.len();
    let total_features = n_tfs * lookback * FULL_FEATURES_PER_CANDLE + ENGULFING_FEATURE_COUNT;

    let target_candles = all_tf_candles.get(&target_tf)?;
    let bar_idx = signal.bar_idx;
    if bar_idx >= target_candles.len() || bar_idx < lookback {
        return None;
    }

    let anchor_time = target_candles[bar_idx].time;
    let mut features = vec![0.0f64; total_features];
    let mut valid_tfs = 0u8;

    for (tf_idx, &tf) in ANALYSIS_TIMEFRAMES.iter().enumerate() {
        let candles = match all_tf_candles.get(&tf) {
            Some(c) if c.len() >= lookback + 10 => c,
            _ => continue,
        };

        let idx = if tf == target_tf {
            bar_idx
        } else {
            let pp = candles.partition_point(|c| c.time <= anchor_time);
            if pp == 0 { continue; }
            pp - 1
        };

        if idx < lookback { continue; }

        valid_tfs += 1;
        let feature_offset = tf_idx * lookback * FULL_FEATURES_PER_CANDLE;

        for c_off in 0..lookback {
            let candle_idx = idx - lookback + c_off;
            let candle_feats = extract_candle_features(candles, candle_idx);
            let start = feature_offset + c_off * FULL_FEATURES_PER_CANDLE;
            let end = start + FULL_FEATURES_PER_CANDLE;
            if end <= features.len() {
                features[start..end].copy_from_slice(&candle_feats);
            }
        }
    }

    if valid_tfs < 2 { return None; }

    // Append engulfing meta-features
    let eng_feats = extract_engulfing_features(target_candles, signal);
    let eng_start = n_tfs * lookback * FULL_FEATURES_PER_CANDLE;
    features[eng_start..eng_start + ENGULFING_FEATURE_COUNT]
        .copy_from_slice(&eng_feats);

    Some(features)
}

// ─────────────────────────────────────────────────────────────────────
// Trade simulation (no look-ahead)
// ─────────────────────────────────────────────────────────────────────

/// Simulate a single trade forward from entry_bar.
///
/// Entry at candles[entry_bar].close.
/// Walk forward up to max_hold bars, checking SL/TP.
/// Conservative: SL wins on same bar.
fn simulate_trade(
    candles: &[CandleInd],
    entry_bar: usize,
    direction: SignalDir,
    entry_price: f64,
    tp_price: f64,
    sl_price: f64,
    max_hold: usize,
) -> Option<(TradeOutcome, usize, f64, f64)> {
    let last_bar = (entry_bar + max_hold).min(candles.len().saturating_sub(1));
    if entry_bar + 1 > last_bar {
        return None;
    }

    for bar in (entry_bar + 1)..=last_bar {
        let c = &candles[bar];
        let hold = bar - entry_bar;

        match direction {
            SignalDir::Long => {
                if c.low <= sl_price {
                    let pnl = (sl_price - entry_price) / entry_price * 100.0;
                    return Some((TradeOutcome::SlHit, hold, sl_price, pnl));
                }
                if c.high >= tp_price {
                    let pnl = (tp_price - entry_price) / entry_price * 100.0;
                    return Some((TradeOutcome::TpHit, hold, tp_price, pnl));
                }
            }
            SignalDir::Short => {
                if c.high >= sl_price {
                    let pnl = (entry_price - sl_price) / entry_price * 100.0;
                    return Some((TradeOutcome::SlHit, hold, sl_price, pnl));
                }
                if c.low <= tp_price {
                    let pnl = (entry_price - tp_price) / entry_price * 100.0;
                    return Some((TradeOutcome::TpHit, hold, tp_price, pnl));
                }
            }
        }

        if hold >= max_hold {
            let exit = c.close;
            let pnl = match direction {
                SignalDir::Long  => (exit - entry_price) / entry_price * 100.0,
                SignalDir::Short => (entry_price - exit) / entry_price * 100.0,
            };
            return Some((TradeOutcome::MaxHoldExpired, hold, exit, pnl));
        }
    }

    None
}

// ─────────────────────────────────────────────────────────────────────
// Batch prediction
// ─────────────────────────────────────────────────────────────────────

/// A pending signal that passed heuristic but awaits ML scoring.
struct PendingSignal {
    bar_idx: usize,
    direction: SignalDir,
    impulse_pct: f64,
}

/// Predict a batch and collect signals above min_signal.
fn flush_predictions(
    batch_features: &[f32],
    batch_pending: &[PendingSignal],
    long_model: Option<&Booster>,
    short_model: Option<&Booster>,
    n_features: usize,
    min_signal: f32,
    signals_out: &mut Vec<(usize, SignalDir, f32, f64)>, // (bar_idx, dir, pred, impulse_pct)
) -> Result<u64> {
    let rows = batch_pending.len();
    if rows == 0 { return Ok(0); }

    // Separate long and short pending signals
    let mut long_indices: Vec<usize> = Vec::new();
    let mut short_indices: Vec<usize> = Vec::new();
    for (i, ps) in batch_pending.iter().enumerate() {
        match ps.direction {
            SignalDir::Long => long_indices.push(i),
            SignalDir::Short => short_indices.push(i),
        }
    }

    // Predict long signals
    if let Some(model) = long_model {
        if !long_indices.is_empty() {
            let long_features: Vec<f32> = long_indices.iter()
                .flat_map(|&i| {
                    let start = i * n_features;
                    let end = start + n_features;
                    batch_features[start..end].iter().copied()
                })
                .collect();

            let preds = model.predict_dense_cpu(
                &long_features, long_indices.len(), n_features, ModelKind::Regressor1,
            )?;

            for (j, &idx) in long_indices.iter().enumerate() {
                let pred = preds[j].clamp(0.0, 1.0);
                if pred >= min_signal {
                    let ps = &batch_pending[idx];
                    signals_out.push((ps.bar_idx, SignalDir::Long, pred, ps.impulse_pct));
                }
            }
        }
    }

    // Predict short signals
    if let Some(model) = short_model {
        if !short_indices.is_empty() {
            let short_features: Vec<f32> = short_indices.iter()
                .flat_map(|&i| {
                    let start = i * n_features;
                    let end = start + n_features;
                    batch_features[start..end].iter().copied()
                })
                .collect();

            let preds = model.predict_dense_cpu(
                &short_features, short_indices.len(), n_features, ModelKind::Regressor1,
            )?;

            for (j, &idx) in short_indices.iter().enumerate() {
                let pred = preds[j].clamp(0.0, 1.0);
                if pred >= min_signal {
                    let ps = &batch_pending[idx];
                    signals_out.push((ps.bar_idx, SignalDir::Short, pred, ps.impulse_pct));
                }
            }
        }
    }

    Ok(rows as u64)
}

// ─────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────

fn print_trade_stats(label: &str, trades: &[&SimTrade], max_hold: usize) {
    if trades.is_empty() {
        info!("    📊 {} — no trades", label);
        return;
    }

    let total = trades.len();
    let wins = trades.iter().filter(|t| t.outcome == TradeOutcome::TpHit).count();
    let losses = trades.iter().filter(|t| t.outcome == TradeOutcome::SlHit).count();
    let expired = trades.iter().filter(|t| t.outcome == TradeOutcome::MaxHoldExpired).count();
    let expired_pos = trades.iter()
        .filter(|t| t.outcome == TradeOutcome::MaxHoldExpired && t.pnl_pct > 0.0)
        .count();
    let win_rate = wins as f64 / total as f64 * 100.0;
    let total_pnl: f64 = trades.iter().map(|t| t.pnl_pct).sum();
    let avg_pnl = total_pnl / total as f64;
    let avg_hold: f64 = trades.iter().map(|t| t.hold_candles as f64).sum::<f64>() / total as f64;

    info!("    📊 {} ({} trades):", label, total);
    info!("      WinRate: {:.1}%  (TP:{} SL:{} Expired:{} [{}+/{}−])",
          win_rate, wins, losses, expired, expired_pos, expired - expired_pos);
    info!("      Avg P&L: {:.2}%,  Total P&L: {:.2}%", avg_pnl, total_pnl);
    info!("      Avg Hold: {:.1} candles (max {})", avg_hold, max_hold);

    // Hold time distribution
    info!("      ─── Hold Time Distribution ───");
    for hold in 1..=max_hold {
        let h_trades: Vec<_> = trades.iter().filter(|t| t.hold_candles == hold).collect();
        if h_trades.is_empty() { continue; }
        let h_count = h_trades.len();
        let h_wins = h_trades.iter().filter(|t| t.outcome == TradeOutcome::TpHit).count();
        let h_avg_pnl: f64 = h_trades.iter().map(|t| t.pnl_pct).sum::<f64>() / h_count as f64;
        info!("        hold={}: {} trades, WR: {:.1}%, AvgPnL: {:.2}%",
              hold, h_count, h_wins as f64 / h_count as f64 * 100.0, h_avg_pnl);
    }

    // Per prediction bucket
    info!("      ─── By Prediction Bucket ───");
    for i in 0..PRED_THRESHOLDS.len() {
        let lo = PRED_THRESHOLDS[i];
        let hi = if i + 1 < PRED_THRESHOLDS.len() { PRED_THRESHOLDS[i + 1] } else { 1.01 };

        let bucket: Vec<_> = trades.iter().filter(|t| t.pred >= lo && t.pred < hi).collect();
        if bucket.is_empty() { continue; }

        let bt = bucket.len();
        let bw = bucket.iter().filter(|t| t.outcome == TradeOutcome::TpHit).count();
        let bl = bucket.iter().filter(|t| t.outcome == TradeOutcome::SlHit).count();
        let bwr = bw as f64 / bt as f64 * 100.0;
        let bavg: f64 = bucket.iter().map(|t| t.pnl_pct).sum::<f64>() / bt as f64;

        info!("        [{:.2}-{:.2}): {} trades, WR: {:.1}% (TP:{} SL:{}), AvgPnL: {:.2}%",
              lo, hi, bt, bwr, bw, bl, bavg);
    }

    // Cumulative thresholds
    info!("      ─── Cumulative Thresholds (pred ≥ X) ───");
    for &th in PRED_THRESHOLDS {
        let above: Vec<_> = trades.iter().filter(|t| t.pred >= th).collect();
        if above.is_empty() { continue; }

        let at = above.len();
        let aw = above.iter().filter(|t| t.outcome == TradeOutcome::TpHit).count();
        let al = above.iter().filter(|t| t.outcome == TradeOutcome::SlHit).count();
        let awr = aw as f64 / at as f64 * 100.0;
        let aavg: f64 = above.iter().map(|t| t.pnl_pct).sum::<f64>() / at as f64;

        info!("        pred ≥ {:.2}: {} trades, WR: {:.1}% (TP:{} SL:{}), AvgPnL: {:.2}%",
              th, at, awr, aw, al, aavg);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // ─── Logging ───
    let log_path = "logs/impulse_backtest.log";
    std::fs::create_dir_all("logs")?;
    let log_file = std::fs::OpenOptions::new()
        .create(true).append(true).open(log_path)?;

    use tracing_subscriber::fmt;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let rust_log = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    let filter = if !rust_log.contains("sqlx") {
        format!("{},sqlx::query=error", rust_log)
    } else {
        rust_log
    };

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(fmt::layer().with_target(false).with_ansi(false)
            .with_writer(std::sync::Mutex::new(log_file)))
        .with(tracing_subscriber::EnvFilter::new(filter))
        .init();

    // ─── Configuration ───
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let config = ImpulseConfig::from_env();
    let max_hold = config.max_hold;

    let min_signal: f32 = std::env::var("IAE_MIN_SIGNAL")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0.50_f32).max(0.01);
    let cooldown: usize = std::env::var("IAE_COOLDOWN")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(max_hold);
    let pred_step: usize = std::env::var("IAE_PRED_STEP")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);

    let heuristic_only = std::env::var("IAE_HEURISTIC_ONLY")
        .map_or(false, |v| v == "1" || v.to_lowercase() == "true");

    let use_gpu = std::env::var("DIRECTION_GPU")
        .map_or(false, |v| v == "1" || v.to_lowercase() == "true");
    let device = if use_gpu { Device::Cuda } else { Device::Cpu };

    let model_type = std::env::var("IAE_MODEL_TYPE")
        .unwrap_or_else(|_| "both".to_string());

    let wfo_min_date: Option<DateTime<Utc>> = std::env::var("WFO_MIN_DATE").ok()
        .and_then(|s| NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()
            .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc()));

    info!("╔═══════════════════════════════════════════════════════════╗");
    if heuristic_only {
        info!("║  Impulse v3 — HEURISTIC-ONLY Backtest (no ML)             ║");
    } else {
        info!("║  Impulse v3 — Walk-Forward Backtest (ML-filtered)         ║");
    }
    info!("║  Realistic Trade Simulation — No Look-Ahead Bias          ║");
    info!("║  Max Hold: {} candles | SL-first conservative rule        ", max_hold);
    info!("╚═══════════════════════════════════════════════════════════╝");
    config.log_summary();
    if heuristic_only {
        info!("  🔥 HEURISTIC-ONLY MODE: all signals → trades (no ML filter)");
    }
    info!("  Min signal: {:.2}", min_signal);
    info!("  Cooldown: {} bars", cooldown);
    info!("  Pred step: {}", pred_step);
    info!("  Device: {:?}", device);
    info!("  Model type: {}", if heuristic_only { "HEURISTIC_ONLY" } else { &model_type });
    if let Some(d) = &wfo_min_date {
        info!("  WFO OOS filter: signals after {}", d.format("%Y-%m-%d"));
    }

    // ─── Load Models (skip in heuristic-only mode) ───
    let (long_model, short_model) = if heuristic_only {
        info!("  ⚡ Skipping model loading (heuristic-only mode)");
        (None, None)
    } else {
        let long_model_path = std::env::var("IAE_LONG_MODEL_PATH")
            .unwrap_or_else(|_| "models/impulse_long_v1.ubj".to_string());
        let short_model_path = std::env::var("IAE_SHORT_MODEL_PATH")
            .unwrap_or_else(|_| "models/impulse_short_v1.ubj".to_string());

        let lm = if model_type == "long" || model_type == "both" {
            match Booster::load(&long_model_path, device) {
                Ok(b) => { info!("  ✅ Long model: {}", long_model_path); Some(b) }
                Err(e) => { info!("  ❌ No long model: {} — {}", long_model_path, e); None }
            }
        } else { None };

        let sm = if model_type == "short" || model_type == "both" {
            match Booster::load(&short_model_path, device) {
                Ok(b) => { info!("  ✅ Short model: {}", short_model_path); Some(b) }
                Err(e) => { info!("  ❌ No short model: {} — {}", short_model_path, e); None }
            }
        } else { None };

        if lm.is_none() && sm.is_none() {
            anyhow::bail!("No models loaded. Set IAE_HEURISTIC_ONLY=1 to run without ML.");
        }
        (lm, sm)
    };

    // ─── Database ───
    let pool = PgPool::connect(&db_url).await?;
    let total_start = std::time::Instant::now();

    let lookback = config.pre_signal_lookback;
    let n_features = ANALYSIS_TIMEFRAMES.len() * lookback * FULL_FEATURES_PER_CANDLE
                     + ENGULFING_FEATURE_COUNT;
    info!("  Feature vector: {} features ({} TFs × {} lookback × {} per candle + {} engulfing meta)",
          n_features, ANALYSIS_TIMEFRAMES.len(), lookback, FULL_FEATURES_PER_CANDLE,
          ENGULFING_FEATURE_COUNT);

    // ─── Bulk Load ALL data (5 queries total) ───
    let tf_limits: HashMap<i32, usize> = vec![
        (1440, 3700), (240, 12000), (60, 12000),
        (15, 12000), (5, 12000),
    ].into_iter().collect();

    let mut bulk_data: HashMap<i32, HashMap<String, Vec<CandleInd>>> = HashMap::new();
    let mut all_symbols: Vec<String> = Vec::new();

    for &tf in ANALYSIS_TIMEFRAMES {
        let limit = tf_limits.get(&tf).copied().unwrap_or(5000);
        info!("  Bulk loading TF {}m (limit {} per symbol)...", tf, limit);
        let t0 = std::time::Instant::now();
        let grouped = fetch_all_candles_for_tf(&pool, tf, limit).await?;
        let n_syms = grouped.len();
        let n_candles: usize = grouped.values().map(|v| v.len()).sum();
        info!("    TF {}m: {} symbols, {} candles in {:.1}s",
              tf, n_syms, n_candles, t0.elapsed().as_secs_f64());

        if all_symbols.is_empty() {
            all_symbols = grouped.keys().cloned().collect();
            all_symbols.sort();
        }

        bulk_data.insert(tf, grouped);
    }

    info!("  {} total symbols loaded", all_symbols.len());
    info!("  Bulk data load complete in {:.1}s — starting simulation",
          total_start.elapsed().as_secs_f64());

    let min_history = lookback + 50;

    // ─── Walk-Forward Simulation ───
    let mut all_trades: Vec<SimTrade> = Vec::new();
    let mut total_predictions = 0u64;
    let mut total_bars_scanned = 0u64;
    let mut total_heuristic_signals = 0u64;
    let mut symbols_processed = 0u32;
    let mut symbols_skipped = 0u32;

    for (si, symbol) in all_symbols.iter().enumerate() {
        // Build per-symbol TF map from bulk data (no DB queries)
        let mut all_tf_candles: HashMap<i32, Vec<CandleInd>> = HashMap::new();
        let mut has_critical = true;

        for &tf in ANALYSIS_TIMEFRAMES {
            if let Some(tf_data) = bulk_data.get(&tf) {
                if let Some(candles) = tf_data.get(symbol) {
                    if candles.len() >= min_history + max_hold + 10 {
                        all_tf_candles.insert(tf, candles.clone());
                    } else if tf == 1440 || tf == 60 {
                        has_critical = false;
                        break;
                    }
                } else if tf == 1440 || tf == 60 {
                    has_critical = false;
                    break;
                }
            }
        }

        if !has_critical {
            symbols_skipped += 1;
            continue;
        }
        symbols_processed += 1;
        let mut sym_trades = 0u32;

        // ─── Process each target TF ───
        for tf_p in &config.tf_params {
            let target_tf = tf_p.tf_minutes;
            let target_candles = match all_tf_candles.get(&target_tf) {
                Some(c) if c.len() >= min_history + max_hold + 10 => c,
                _ => continue,
            };

            let n_candles = target_candles.len();
            let start_bar = min_history;
            let end_bar = n_candles.saturating_sub(max_hold + 1);
            if start_bar >= end_bar { continue; }

            // Phase 1: Walk bar-by-bar, detect heuristic signals, batch predict
            let mut signals: Vec<(usize, SignalDir, f32, f64)> = Vec::new();
            let mut batch_features: Vec<f32> = Vec::with_capacity(PRED_BATCH_SIZE * n_features);
            let mut batch_pending: Vec<PendingSignal> = Vec::with_capacity(PRED_BATCH_SIZE);

            let mut bar_idx = start_bar;
            while bar_idx < end_bar {
                // WFO date filter
                if let Some(min_d) = wfo_min_date {
                    if target_candles[bar_idx].time < min_d {
                        bar_idx += pred_step;
                        continue;
                    }
                }

                total_bars_scanned += 1;

                // Heuristic engulfing check (v3: ATR-based + multi-bar, no look-ahead)
                if let Some(signal) = check_engulfing_at_bar(
                    target_candles, bar_idx, &config, tf_p,
                ) {
                    total_heuristic_signals += 1;

                    if heuristic_only {
                        // Heuristic-only: pass all signals with pred=1.0
                        signals.push((signal.bar_idx, signal.direction, 1.0_f32, signal.impulse_pct));
                        total_predictions += 1;
                    } else {
                        // ML mode: batch features for prediction
                        let has_model = match signal.direction {
                            SignalDir::Long => long_model.is_some(),
                            SignalDir::Short => short_model.is_some(),
                        };

                        if has_model {
                            if let Some(features) = extract_features_at_bar(
                                &all_tf_candles, target_tf, &signal, lookback,
                            ) {
                                batch_features.extend(features.iter().map(|&v| v as f32));
                                batch_pending.push(PendingSignal {
                                    bar_idx: signal.bar_idx,
                                    direction: signal.direction,
                                    impulse_pct: signal.impulse_pct,
                                });
                            }
                        }
                    }
                }

                // Flush batch when full (ML mode only)
                if !heuristic_only && batch_pending.len() >= PRED_BATCH_SIZE {
                    total_predictions += flush_predictions(
                        &batch_features, &batch_pending,
                        long_model.as_ref(), short_model.as_ref(),
                        n_features, min_signal, &mut signals,
                    )?;
                    batch_features.clear();
                    batch_pending.clear();
                }

                bar_idx += pred_step;
            }

            // Flush remaining (ML mode only)
            if !heuristic_only && !batch_pending.is_empty() {
                total_predictions += flush_predictions(
                    &batch_features, &batch_pending,
                    long_model.as_ref(), short_model.as_ref(),
                    n_features, min_signal, &mut signals,
                )?;
                batch_features.clear();
                batch_pending.clear();
            }

            // Phase 2: Simulate trades with cooldown
            let mut last_long_bar: i64 = -(cooldown as i64) - 1;
            let mut last_short_bar: i64 = -(cooldown as i64) - 1;

            for &(sig_bar, direction, pred, impulse_pct) in &signals {
                let entry_price = target_candles[sig_bar].close;
                if entry_price < 1e-12 { continue; }

                let cooldown_bar = match direction {
                    SignalDir::Long => &mut last_long_bar,
                    SignalDir::Short => &mut last_short_bar,
                };

                if (sig_bar as i64 - *cooldown_bar) < cooldown as i64 {
                    continue;
                }

                let tp_price = match direction {
                    SignalDir::Long  => entry_price * (1.0 + tf_p.tp_pct / 100.0),
                    SignalDir::Short => entry_price * (1.0 - tf_p.tp_pct / 100.0),
                };
                let sl_price = match direction {
                    SignalDir::Long  => entry_price * (1.0 - tf_p.sl_pct / 100.0),
                    SignalDir::Short => entry_price * (1.0 + tf_p.sl_pct / 100.0),
                };

                if let Some((outcome, hold, exit, pnl)) = simulate_trade(
                    target_candles, sig_bar, direction,
                    entry_price, tp_price, sl_price, max_hold,
                ) {
                    all_trades.push(SimTrade {
                        symbol: symbol.clone(),
                        tf_minutes: target_tf,
                        direction,
                        entry_time: target_candles[sig_bar].time,
                        entry_price,
                        exit_price: exit,
                        pred,
                        outcome,
                        hold_candles: hold,
                        pnl_pct: pnl,
                        impulse_pct,
                    });
                    sym_trades += 1;
                    *cooldown_bar = sig_bar as i64;
                }
            }
        }

        // Memory cleanup
        drop(all_tf_candles);

        if (si + 1) % 50 == 0 || si == 0 || si + 1 == all_symbols.len() {
            info!("  [{}/{}] {} — {} sym trades (total: {}, heuristic: {}, ml_pred: {}, {:.1}s)",
                  si + 1, all_symbols.len(), symbol, sym_trades,
                  all_trades.len(), total_heuristic_signals, total_predictions,
                  total_start.elapsed().as_secs_f64());
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // RESULTS
    // ═════════════════════════════════════════════════════════════════
    let elapsed = total_start.elapsed();

    info!("");
    info!("╔════════════════════════════════════════════════════════════════╗");
    info!("║  IMPULSE ENGULFING — BACKTEST RESULTS                          ║");
    info!("╚════════════════════════════════════════════════════════════════╝");
    info!("  Symbols processed: {} (skipped: {})", symbols_processed, symbols_skipped);
    info!("  Bars scanned: {}", total_bars_scanned);
    info!("  Heuristic signals: {}", total_heuristic_signals);
    info!("  ML predictions: {}", total_predictions);
    info!("  Trades (pred ≥ {:.2}): {}", min_signal, all_trades.len());
    if total_bars_scanned > 0 {
        info!("  Heuristic signal density: {:.4}% ({} / {} bars)",
              total_heuristic_signals as f64 / total_bars_scanned as f64 * 100.0,
              total_heuristic_signals, total_bars_scanned);
    }
    info!("  Time: {:.1}s ({:.1}min)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);
    info!("");

    // ─── Per-TF Results ───
    for tf_p in &config.tf_params {
        let tf = tf_p.tf_minutes;
        let tf_trades: Vec<&SimTrade> = all_trades.iter()
            .filter(|t| t.tf_minutes == tf)
            .collect();

        if tf_trades.is_empty() {
            info!("  ═══ TF: {}m ═══  (no trades)", tf);
            info!("");
            continue;
        }

        info!("  ═══════════════════════════════════════════════════");
        info!("  ═══ TF: {}m — {} trades (impulse≥{:.1}%, TP={:.1}%, SL={:.1}%) ═══",
              tf, tf_trades.len(), tf_p.impulse_pct, tf_p.tp_pct, tf_p.sl_pct);
        info!("  ═══════════════════════════════════════════════════");

        let long_trades: Vec<&SimTrade> = tf_trades.iter()
            .filter(|t| t.direction == SignalDir::Long).copied().collect();
        print_trade_stats("LONG", &long_trades, max_hold);
        info!("");

        let short_trades: Vec<&SimTrade> = tf_trades.iter()
            .filter(|t| t.direction == SignalDir::Short).copied().collect();
        print_trade_stats("SHORT", &short_trades, max_hold);

        // Sample trades
        let samples: Vec<&SimTrade> = tf_trades.iter().take(5).copied().collect();
        if !samples.is_empty() {
            info!("");
            info!("    ─── Sample Trades (first 5) ───");
            for t in &samples {
                info!("      {} {} {} {} pred={:.3} imp={:.1}% entry={:.4} exit={:.4} hold={} P&L={:.2}% {}",
                      t.symbol, t.entry_time.format("%Y-%m-%d %H:%M"),
                      t.direction, t.tf_minutes, t.pred, t.impulse_pct,
                      t.entry_price, t.exit_price, t.hold_candles, t.pnl_pct, t.outcome);
            }
        }
        info!("");
    }

    // ─── Overall Summary ───
    if !all_trades.is_empty() {
        info!("  ═══════════════════════════════════════════════════");
        info!("  ═══ OVERALL SUMMARY ═══");
        info!("  ═══════════════════════════════════════════════════");

        let all_refs: Vec<&SimTrade> = all_trades.iter().collect();
        print_trade_stats("ALL TRADES", &all_refs, max_hold);

        // Top symbols
        info!("");
        info!("    ─── Top 10 Symbols by Trade Count ───");
        let mut sym_counts: HashMap<&str, (usize, usize)> = HashMap::new();
        for t in &all_trades {
            let e = sym_counts.entry(&t.symbol).or_default();
            e.0 += 1;
            if t.outcome == TradeOutcome::TpHit { e.1 += 1; }
        }
        let mut sym_list: Vec<_> = sym_counts.iter().collect();
        sym_list.sort_by(|a, b| b.1.0.cmp(&a.1.0));
        for (sym, (count, wins)) in sym_list.iter().take(10) {
            let sym_pnl: f64 = all_trades.iter()
                .filter(|t| &t.symbol == *sym)
                .map(|t| t.pnl_pct).sum();
            info!("      {}: {} trades, WR: {:.1}%, TotalPnL: {:.2}%",
                  sym, count, *wins as f64 / *count as f64 * 100.0, sym_pnl);
        }

        // Top 10 best
        info!("");
        info!("    ─── Top 10 Best Trades ───");
        let mut sorted = all_trades.clone();
        sorted.sort_by(|a, b| b.pnl_pct.partial_cmp(&a.pnl_pct).unwrap_or(std::cmp::Ordering::Equal));
        for (i, t) in sorted.iter().take(10).enumerate() {
            info!("      #{}: {} {} {} TF={}m pred={:.3} imp={:.1}% entry={:.4} P&L={:.2}% {}",
                  i + 1, t.symbol, t.entry_time.format("%Y-%m-%d %H:%M"),
                  t.direction, t.tf_minutes, t.pred, t.impulse_pct,
                  t.entry_price, t.pnl_pct, t.outcome);
        }

        // Top 10 worst
        info!("");
        info!("    ─── Top 10 Worst Trades ───");
        for (i, t) in sorted.iter().rev().take(10).enumerate() {
            info!("      #{}: {} {} {} TF={}m pred={:.3} imp={:.1}% entry={:.4} P&L={:.2}% {}",
                  i + 1, t.symbol, t.entry_time.format("%Y-%m-%d %H:%M"),
                  t.direction, t.tf_minutes, t.pred, t.impulse_pct,
                  t.entry_price, t.pnl_pct, t.outcome);
        }

        // Verdict — account for asymmetric R:R (breakeven WR is NOT 50%)
        info!("");
        let total = all_trades.len();
        let wins = all_trades.iter().filter(|t| t.outcome == TradeOutcome::TpHit).count();
        let wr = wins as f64 / total as f64 * 100.0;
        let avg_pnl: f64 = all_trades.iter().map(|t| t.pnl_pct).sum::<f64>() / total as f64;
        let total_pnl_final: f64 = all_trades.iter().map(|t| t.pnl_pct).sum();

        // Compute breakeven WR for asymmetric TP/SL
        // For R:R = TP:SL = 1.5:1 → breakeven WR = SL/(TP+SL) = 1.0/(1.5+1.0) = 40%
        // Using average TP/SL across all TFs
        let avg_tp: f64 = config.tf_params.iter().map(|p| p.tp_pct).sum::<f64>() / config.tf_params.len() as f64;
        let avg_sl: f64 = config.tf_params.iter().map(|p| p.sl_pct).sum::<f64>() / config.tf_params.len() as f64;
        let breakeven_wr = avg_sl / (avg_tp + avg_sl) * 100.0;

        info!("  Breakeven WR (with R:R={:.1}:{:.1}): {:.1}%", avg_tp, avg_sl, breakeven_wr);

        if avg_pnl > 0.1 && wr > breakeven_wr {
            info!("  🟢 VERDICT: Strategy shows STRONG edge (WR={:.1}% > BE={:.1}%, E[PnL]={:.2}%, Total={:.1}%)",
                  wr, breakeven_wr, avg_pnl, total_pnl_final);
        } else if avg_pnl > 0.0 {
            info!("  🟡 VERDICT: Strategy shows MARGINAL edge (WR={:.1}%, E[PnL]={:.2}%, Total={:.1}%)",
                  wr, avg_pnl, total_pnl_final);
        } else {
            info!("  🔴 VERDICT: Strategy shows NO edge (WR={:.1}%, E[PnL]={:.2}%, Total={:.1}%)",
                  wr, avg_pnl, total_pnl_final);
        }
        if heuristic_only {
            info!("  📌 This is HEURISTIC-ONLY baseline. Retrain ML models for filtered results.");
            info!("  📌 Next: ./target/release/impulse_dataset && python trainer/src/train_impulse_wfo.py");
        } else {
            info!("  NOTE: Check cumulative thresholds — higher pred cutoff may yield positive edge");
        }
    } else {
        info!("  ⚠️  No trades generated. No engulfing signals passed ML filter (min_pred={:.2})", min_signal);
    }

    info!("");
    info!("  Total time: {:.1}s ({:.1}min)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);

    Ok(())
}
