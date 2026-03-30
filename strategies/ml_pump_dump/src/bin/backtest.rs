// strategies/ml_pump_dump/src/bin/backtest.rs
//
// Pump/Dump Walk-Forward Trade Simulation v3
//
// NO LOOK-AHEAD BIAS:
//   - Walks bar-by-bar on target TFs (5m, 15m, 1h, 4h)
//   - Features extracted from PAST candles only (before signal bar)
//   - When model predicts pump/dump (pred >= threshold), opens a virtual trade
//   - Checks SL/TP hit on future candles within max_hold window
//   - Comprehensive statistics: win rate, P&L, hold times, per-bucket analysis
//
// PERFORMANCE:
//   - Batch model predictions (256 rows at once via XGBoost)
//   - sqlx slow statement warnings suppressed via tracing filter
//   - Sequential symbol processing (memory-safe)
//
// USAGE:
//   cargo build --release -p ml_pump_dump --bin pump_dump_backtest
//   ./target/release/pump_dump_backtest
//
// ENV VARS:
//   DATABASE_URL           — postgres connection
//   PD_MODEL_TYPE          — "pump", "dump", or "both" (default: "both")
//   PD_MAX_HOLD_CANDLES    — max candles to hold (default: 6)
//   PD_TARGET_PCT          — target move % (default: 15.0)
//   PD_SL_FRACTION         — SL as fraction of target (default: 0.65 → SL=9.75%)
//   PD_PRED_STEP           — predict every N bars (default: 1 = every bar)
//   PD_COOLDOWN            — bars cooldown after trade (default: max_hold)
//   PD_MIN_SIGNAL          — minimum pred to record signal (default: 0.50)
//   WFO_MIN_DATE           — only test signals after this date (YYYY-MM-DD)
//   DIRECTION_GPU=1        — use GPU for inference

use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use dotenvy::dotenv;
use sqlx::PgPool;
use std::collections::HashMap;
use tracing::info;

use ml_pump_dump::pump_dump::{
    CandleInd, EventType, PumpDumpConfig,
    ANALYSIS_TIMEFRAMES, FULL_FEATURES_PER_CANDLE,
    extract_candle_features,
};
use ml_pump_dump::dataset::{
    fetch_candles_with_indicators, fetch_active_symbols,
};

use predictors::ml::xgb_runtime::{Booster, Device, ModelKind};

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

/// Target timeframes for trade simulation
const TARGET_TFS: &[i32] = &[5, 15, 60, 240];

/// Batch size for XGBoost predictions (amortises DMatrix creation overhead)
const PRED_BATCH_SIZE: usize = 256;

/// Prediction threshold boundaries for analysis buckets
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
    event_type: EventType,
    entry_time: DateTime<Utc>,
    entry_price: f64,
    exit_price: f64,
    pred: f32,
    outcome: TradeOutcome,
    hold_candles: usize,
    pnl_pct: f64,
}

// ─────────────────────────────────────────────────────────────────────
// Feature extraction (no look-ahead)
// ─────────────────────────────────────────────────────────────────────

/// Extract multi-TF features anchored at `bar_idx` on `target_tf`.
///
/// Features come from the lookback window BEFORE bar_idx on each analysis TF.
/// NO look-ahead: only uses candle data with close_time <= anchor bar's time.
///
/// Feature layout matches training: for each TF in ANALYSIS_TIMEFRAMES order,
/// for each of `lookback` candles (oldest first), emit FULL_FEATURES_PER_CANDLE.
fn extract_features_at_bar(
    all_tf_candles: &HashMap<i32, Vec<CandleInd>>,
    target_tf: i32,
    bar_idx: usize,
    lookback: usize,
) -> Option<Vec<f64>> {
    let n_tfs = ANALYSIS_TIMEFRAMES.len();
    let total_features = n_tfs * lookback * FULL_FEATURES_PER_CANDLE;

    let target_candles = all_tf_candles.get(&target_tf)?;
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

        // Find reference index: latest candle with time <= anchor_time
        let idx = if tf == target_tf {
            // Same TF — use bar_idx directly
            bar_idx
        } else {
            // Binary search for latest candle at or before anchor_time
            let pp = candles.partition_point(|c| c.time <= anchor_time);
            if pp == 0 { continue; }
            pp - 1
        };

        if idx < lookback { continue; }

        valid_tfs += 1;
        let feature_offset = tf_idx * lookback * FULL_FEATURES_PER_CANDLE;

        // Extract features from candles [idx - lookback .. idx - 1]
        // These are the lookback candles BEFORE the signal bar (no look-ahead)
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

    // Need at least 2 TFs for a meaningful prediction
    if valid_tfs >= 2 { Some(features) } else { None }
}

// ─────────────────────────────────────────────────────────────────────
// Trade simulation
// ─────────────────────────────────────────────────────────────────────

/// Simulate a single trade forward from entry_bar.
///
/// Entry at `candles[entry_bar].close`. Walk forward up to `max_hold` bars,
/// checking SL/TP on each bar's high/low.
///
/// Conservative rule: if SL and TP both hit on the same bar, SL wins.
///
/// Returns `(outcome, hold_candles, exit_price, pnl_pct)` or `None`
/// if there aren't enough future candles.
fn simulate_trade(
    candles: &[CandleInd],
    entry_bar: usize,
    event_type: EventType,
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

        match event_type {
            EventType::Pump => {
                // LONG: SL if low <= sl_price, TP if high >= tp_price
                if c.low <= sl_price {
                    let pnl = (sl_price - entry_price) / entry_price * 100.0;
                    return Some((TradeOutcome::SlHit, hold, sl_price, pnl));
                }
                if c.high >= tp_price {
                    let pnl = (tp_price - entry_price) / entry_price * 100.0;
                    return Some((TradeOutcome::TpHit, hold, tp_price, pnl));
                }
            }
            EventType::Dump => {
                // SHORT: SL if high >= sl_price, TP if low <= tp_price
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

        // Max hold — close at this bar's close
        if hold >= max_hold {
            let exit = c.close;
            let pnl = match event_type {
                EventType::Pump => (exit - entry_price) / entry_price * 100.0,
                EventType::Dump => (entry_price - exit) / entry_price * 100.0,
            };
            return Some((TradeOutcome::MaxHoldExpired, hold, exit, pnl));
        }
    }

    None
}

// ─────────────────────────────────────────────────────────────────────
// Batch prediction helper
// ─────────────────────────────────────────────────────────────────────

/// Predict a batch of feature vectors and collect signals above min_signal.
///
/// Returns the number of predictions made.
fn flush_predictions(
    batch_features: &[f32],
    batch_indices: &[usize],
    pump_model: Option<&Booster>,
    dump_model: Option<&Booster>,
    n_features: usize,
    min_signal: f32,
    signals: &mut Vec<(usize, Option<f32>, Option<f32>)>,
) -> Result<u64> {
    let rows = batch_indices.len();
    if rows == 0 {
        return Ok(0);
    }

    let pump_preds = if let Some(m) = pump_model {
        Some(m.predict_dense_cpu(batch_features, rows, n_features, ModelKind::Regressor1)?)
    } else {
        None
    };

    let dump_preds = if let Some(m) = dump_model {
        Some(m.predict_dense_cpu(batch_features, rows, n_features, ModelKind::Regressor1)?)
    } else {
        None
    };

    for (i, &bidx) in batch_indices.iter().enumerate() {
        let pp = pump_preds.as_ref().map(|p| p[i].clamp(0.0, 1.0));
        let dp = dump_preds.as_ref().map(|p| p[i].clamp(0.0, 1.0));

        let has_pump = pp.map_or(false, |v| v >= min_signal);
        let has_dump = dp.map_or(false, |v| v >= min_signal);

        if has_pump || has_dump {
            signals.push((
                bidx,
                if has_pump { pp } else { None },
                if has_dump { dp } else { None },
            ));
        }
    }

    Ok(rows as u64)
}

// ─────────────────────────────────────────────────────────────────────
// Statistics output
// ─────────────────────────────────────────────────────────────────────

/// Print detailed statistics for a set of trades.
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
    info!("      Avg Hold: {:.1} candles", avg_hold);

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
        let bhold: f64 = bucket.iter().map(|t| t.hold_candles as f64).sum::<f64>() / bt as f64;

        info!("        [{:.2}-{:.2}): {} trades, WR: {:.1}% (TP:{} SL:{}), AvgPnL: {:.2}%, AvgHold: {:.1}",
              lo, hi, bt, bwr, bw, bl, bavg, bhold);
    }

    // Cumulative thresholds — key for choosing the right min_pred
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
    // Suppress sqlx slow statement warnings via tracing filter
    let log_path = "logs/pump_dump_backtest.log";
    std::fs::create_dir_all("logs")?;
    let log_file = std::fs::OpenOptions::new()
        .create(true).append(true).open(log_path)?;

    use tracing_subscriber::fmt;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let rust_log = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    // Suppress sqlx::query WARN (slow statement) unless user explicitly configured sqlx
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

    let config = PumpDumpConfig::from_env();

    let max_hold: usize = std::env::var("PD_MAX_HOLD_CANDLES")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let target_pct: f64 = std::env::var("PD_TARGET_PCT")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(15.0);
    let sl_fraction: f64 = std::env::var("PD_SL_FRACTION")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0.65);
    let sl_pct = target_pct * sl_fraction;
    let pred_step: usize = std::env::var("PD_PRED_STEP")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let cooldown: usize = std::env::var("PD_COOLDOWN")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(max_hold);
    let min_signal: f32 = std::env::var("PD_MIN_SIGNAL")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0.50_f32).max(0.01);

    let use_gpu = std::env::var("DIRECTION_GPU")
        .map_or(false, |v| v == "1" || v.to_lowercase() == "true");
    let device = if use_gpu { Device::Cuda } else { Device::Cpu };

    let model_type = std::env::var("PD_MODEL_TYPE")
        .unwrap_or_else(|_| "both".to_string());

    let wfo_min_date: Option<DateTime<Utc>> = std::env::var("WFO_MIN_DATE").ok()
        .and_then(|s| NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()
            .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc()));

    info!("╔═══════════════════════════════════════════════════════════╗");
    info!("║  Pump/Dump Walk-Forward Backtest v3                        ║");
    info!("║  True Trade Simulation — No Look-Ahead Bias                ║");
    info!("╚═══════════════════════════════════════════════════════════╝");
    info!("  Target TFs: {:?}", TARGET_TFS);
    info!("  Max hold: {} candles", max_hold);
    info!("  Target: {:.1}%, SL: {:.1}% (fraction: {:.2})", target_pct, sl_pct, sl_fraction);
    info!("  Pred step: {} (predict every {} bars)", pred_step, pred_step);
    info!("  Cooldown: {} bars between same-type signals", cooldown);
    info!("  Min signal: {:.2}", min_signal);
    info!("  Device: {:?}", device);
    info!("  Model type: {}", model_type);
    if let Some(d) = &wfo_min_date {
        info!("  WFO OOS filter: signals after {}", d.format("%Y-%m-%d"));
    }
    config.log_summary();

    // ─── Load Models ───
    let pump_model_path = std::env::var("PD_PUMP_MODEL_PATH")
        .unwrap_or_else(|_| "models/pump_dump_pump_v1.ubj".to_string());
    let dump_model_path = std::env::var("PD_DUMP_MODEL_PATH")
        .unwrap_or_else(|_| "models/pump_dump_dump_v1.ubj".to_string());

    let pump_model = if model_type == "pump" || model_type == "both" {
        match Booster::load(&pump_model_path, device) {
            Ok(b) => { info!("  ✅ Pump model: {}", pump_model_path); Some(b) }
            Err(e) => { info!("  ❌ No pump model: {} — {}", pump_model_path, e); None }
        }
    } else { None };

    let dump_model = if model_type == "dump" || model_type == "both" {
        match Booster::load(&dump_model_path, device) {
            Ok(b) => { info!("  ✅ Dump model: {}", dump_model_path); Some(b) }
            Err(e) => { info!("  ❌ No dump model: {} — {}", dump_model_path, e); None }
        }
    } else { None };

    if pump_model.is_none() && dump_model.is_none() {
        anyhow::bail!("No models loaded. Need at least one model to run backtest.");
    }

    // ─── Database ───
    let pool = PgPool::connect(&db_url).await?;
    let total_start = std::time::Instant::now();

    let lookback = config.pre_event_lookback;
    let n_features = ANALYSIS_TIMEFRAMES.len() * lookback * FULL_FEATURES_PER_CANDLE;
    info!("  Feature vector: {} features ({} TFs × {} lookback × {} per candle)",
          n_features, ANALYSIS_TIMEFRAMES.len(), lookback, FULL_FEATURES_PER_CANDLE);

    let symbols = fetch_active_symbols(&pool).await?;
    info!("  {} active symbols", symbols.len());

    let tf_limits: HashMap<i32, usize> = vec![
        (1440, 3700), (240, 12000), (60, 12000),
        (15, 12000), (5, 12000),
    ].into_iter().collect();

    // Minimum history needed: lookback + 50 bars (for temporal features lookback)
    let min_history = lookback + 50;

    // ─── Walk-Forward Simulation ───
    let mut all_trades: Vec<SimTrade> = Vec::new();
    let mut total_predictions = 0u64;
    let mut total_bars_scanned = 0u64;
    let mut symbols_processed = 0u32;
    let mut symbols_skipped = 0u32;

    for (si, symbol) in symbols.iter().enumerate() {
        // Load all TF data for this symbol
        let mut all_tf_candles: HashMap<i32, Vec<CandleInd>> = HashMap::new();
        let mut has_critical = true;

        for &tf in ANALYSIS_TIMEFRAMES {
            let limit = tf_limits.get(&tf).copied().unwrap_or(5000);
            match fetch_candles_with_indicators(&pool, symbol, tf, limit).await {
                Ok(c) if c.len() >= min_history + max_hold + 10 => {
                    all_tf_candles.insert(tf, c);
                }
                _ => {
                    // Daily and hourly are critical for multi-TF features
                    if tf == 1440 || tf == 60 {
                        has_critical = false;
                        break;
                    }
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
        for &target_tf in TARGET_TFS {
            let target_candles = match all_tf_candles.get(&target_tf) {
                Some(c) if c.len() >= min_history + max_hold + 10 => c,
                _ => continue,
            };

            let n_candles = target_candles.len();
            let start_bar = min_history;
            let end_bar = n_candles.saturating_sub(max_hold + 1);
            if start_bar >= end_bar { continue; }

            // Phase 1: Batch-predict all bars and collect signals
            let mut signals: Vec<(usize, Option<f32>, Option<f32>)> = Vec::new();
            let mut batch_features: Vec<f32> = Vec::with_capacity(PRED_BATCH_SIZE * n_features);
            let mut batch_indices: Vec<usize> = Vec::with_capacity(PRED_BATCH_SIZE);

            let mut bar_idx = start_bar;
            while bar_idx < end_bar {
                // WFO date filter
                if let Some(min_d) = wfo_min_date {
                    if target_candles[bar_idx].time < min_d {
                        bar_idx += pred_step;
                        continue;
                    }
                }

                if let Some(features) = extract_features_at_bar(
                    &all_tf_candles, target_tf, bar_idx, lookback,
                ) {
                    batch_features.extend(features.iter().map(|&v| v as f32));
                    batch_indices.push(bar_idx);
                }

                total_bars_scanned += 1;

                // Flush batch when full
                if batch_indices.len() >= PRED_BATCH_SIZE {
                    total_predictions += flush_predictions(
                        &batch_features, &batch_indices,
                        pump_model.as_ref(), dump_model.as_ref(),
                        n_features, min_signal, &mut signals,
                    )?;
                    batch_features.clear();
                    batch_indices.clear();
                }

                bar_idx += pred_step;
            }

            // Flush remaining
            if !batch_indices.is_empty() {
                total_predictions += flush_predictions(
                    &batch_features, &batch_indices,
                    pump_model.as_ref(), dump_model.as_ref(),
                    n_features, min_signal, &mut signals,
                )?;
                batch_features.clear();
                batch_indices.clear();
            }

            // Phase 2: Simulate trades with cooldown
            let mut last_pump_bar: i64 = -(cooldown as i64) - 1;
            let mut last_dump_bar: i64 = -(cooldown as i64) - 1;

            for &(sig_bar, pump_pred, dump_pred) in &signals {
                let entry_price = target_candles[sig_bar].close;
                if entry_price < 1e-12 { continue; }

                // ── Pump signal ──
                if let Some(pred) = pump_pred {
                    if (sig_bar as i64 - last_pump_bar) >= cooldown as i64 {
                        let tp = entry_price * (1.0 + target_pct / 100.0);
                        let sl = entry_price * (1.0 - sl_pct / 100.0);

                        if let Some((outcome, hold, exit, pnl)) = simulate_trade(
                            target_candles, sig_bar, EventType::Pump,
                            entry_price, tp, sl, max_hold,
                        ) {
                            all_trades.push(SimTrade {
                                symbol: symbol.clone(),
                                tf_minutes: target_tf,
                                event_type: EventType::Pump,
                                entry_time: target_candles[sig_bar].time,
                                entry_price,
                                exit_price: exit,
                                pred,
                                outcome,
                                hold_candles: hold,
                                pnl_pct: pnl,
                            });
                            sym_trades += 1;
                            last_pump_bar = sig_bar as i64;
                        }
                    }
                }

                // ── Dump signal ──
                if let Some(pred) = dump_pred {
                    if (sig_bar as i64 - last_dump_bar) >= cooldown as i64 {
                        let tp = entry_price * (1.0 - target_pct / 100.0);
                        let sl = entry_price * (1.0 + sl_pct / 100.0);

                        if let Some((outcome, hold, exit, pnl)) = simulate_trade(
                            target_candles, sig_bar, EventType::Dump,
                            entry_price, tp, sl, max_hold,
                        ) {
                            all_trades.push(SimTrade {
                                symbol: symbol.clone(),
                                tf_minutes: target_tf,
                                event_type: EventType::Dump,
                                entry_time: target_candles[sig_bar].time,
                                entry_price,
                                exit_price: exit,
                                pred,
                                outcome,
                                hold_candles: hold,
                                pnl_pct: pnl,
                            });
                            sym_trades += 1;
                            last_dump_bar = sig_bar as i64;
                        }
                    }
                }
            }
        }

        // Memory cleanup
        drop(all_tf_candles);

        if (si + 1) % 20 == 0 || si == 0 {
            info!("  [{}/{}] {} — {} sym trades (total: {}, predictions: {}, {:.1}s)",
                  si + 1, symbols.len(), symbol, sym_trades,
                  all_trades.len(), total_predictions,
                  total_start.elapsed().as_secs_f64());
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // RESULTS
    // ═════════════════════════════════════════════════════════════════
    let elapsed = total_start.elapsed();

    info!("");
    info!("╔════════════════════════════════════════════════════════════════╗");
    info!("║  PUMP/DUMP WALK-FORWARD BACKTEST RESULTS v3                    ║");
    info!("╚════════════════════════════════════════════════════════════════╝");
    info!("  Symbols processed: {} (skipped: {})", symbols_processed, symbols_skipped);
    info!("  Bars scanned: {}", total_bars_scanned);
    info!("  Model predictions: {}", total_predictions);
    info!("  Total signals (pred ≥ {:.2}): {}", min_signal, all_trades.len());
    info!("  Signal density: {:.4} signals/bar ({:.2}%)",
          if total_bars_scanned > 0 { all_trades.len() as f64 / total_bars_scanned as f64 } else { 0.0 },
          if total_bars_scanned > 0 { all_trades.len() as f64 / total_bars_scanned as f64 * 100.0 } else { 0.0 });
    info!("  Time: {:.1}s ({:.1}min)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);
    info!("");

    // ─── Per-TF Results ───
    for &tf in TARGET_TFS {
        let tf_trades: Vec<&SimTrade> = all_trades.iter()
            .filter(|t| t.tf_minutes == tf)
            .collect();
        if tf_trades.is_empty() {
            info!("  ═══ TF: {}m ═══  (no trades)", tf);
            info!("");
            continue;
        }

        let _tf_bars = total_bars_scanned / TARGET_TFS.len() as u64; // approximate
        info!("  ═══════════════════════════════════════════════════");
        info!("  ═══ TF: {}m — {} trades total ═══", tf, tf_trades.len());
        info!("  ═══════════════════════════════════════════════════");

        // ── Pump stats ──
        let pump_trades: Vec<&SimTrade> = tf_trades.iter()
            .filter(|t| t.event_type == EventType::Pump)
            .copied()
            .collect();
        print_trade_stats("PUMP", &pump_trades, max_hold);

        info!("");

        // ── Dump stats ──
        let dump_trades: Vec<&SimTrade> = tf_trades.iter()
            .filter(|t| t.event_type == EventType::Dump)
            .copied()
            .collect();
        print_trade_stats("DUMP", &dump_trades, max_hold);

        // Log first 5 sample trades for this TF
        let samples: Vec<&SimTrade> = tf_trades.iter().take(5).copied().collect();
        if !samples.is_empty() {
            info!("");
            info!("    ─── Sample Trades (first 5) ───");
            for t in &samples {
                info!("      {} {} {} pred={:.3} entry={:.4} exit={:.4} hold={} P&L={:.2}% {}",
                      t.symbol, t.entry_time.format("%Y-%m-%d %H:%M"),
                      t.event_type, t.pred, t.entry_price, t.exit_price,
                      t.hold_candles, t.pnl_pct, t.outcome);
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

        // ── Top symbols by trade count ──
        info!("");
        info!("    ─── Top 10 Symbols by Trade Count ───");
        let mut sym_counts: HashMap<&str, (usize, usize)> = HashMap::new(); // (total, wins)
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
                .map(|t| t.pnl_pct)
                .sum();
            info!("      {}: {} trades, WR: {:.1}%, TotalPnL: {:.2}%",
                  sym, count, *wins as f64 / *count as f64 * 100.0, sym_pnl);
        }

        // ── Top 10 best trades ──
        info!("");
        info!("    ─── Top 10 Best Trades ───");
        let mut sorted = all_trades.clone();
        sorted.sort_by(|a, b| b.pnl_pct.partial_cmp(&a.pnl_pct).unwrap_or(std::cmp::Ordering::Equal));
        for (i, t) in sorted.iter().take(10).enumerate() {
            info!("      #{}: {} {} {} TF={}m pred={:.3} entry={:.4} P&L={:.2}% hold={} {}",
                  i + 1, t.symbol, t.entry_time.format("%Y-%m-%d %H:%M"),
                  t.event_type, t.tf_minutes, t.pred, t.entry_price,
                  t.pnl_pct, t.hold_candles, t.outcome);
        }

        // ── Top 10 worst trades ──
        info!("");
        info!("    ─── Top 10 Worst Trades ───");
        for (i, t) in sorted.iter().rev().take(10).enumerate() {
            info!("      #{}: {} {} {} TF={}m pred={:.3} entry={:.4} P&L={:.2}% hold={} {}",
                  i + 1, t.symbol, t.entry_time.format("%Y-%m-%d %H:%M"),
                  t.event_type, t.tf_minutes, t.pred, t.entry_price,
                  t.pnl_pct, t.hold_candles, t.outcome);
        }

        // ── Quick verdict ──
        info!("");
        let total = all_trades.len();
        let wins = all_trades.iter().filter(|t| t.outcome == TradeOutcome::TpHit).count();
        let wr = wins as f64 / total as f64 * 100.0;
        let avg_pnl: f64 = all_trades.iter().map(|t| t.pnl_pct).sum::<f64>() / total as f64;
        let expectancy = avg_pnl; // already per-trade

        if wr > 50.0 && avg_pnl > 0.0 {
            info!("  🟢 VERDICT: Model shows POSITIVE edge (WR={:.1}%, E[PnL]={:.2}%)", wr, expectancy);
        } else if wr > 30.0 && avg_pnl > -2.0 {
            info!("  🟡 VERDICT: Model shows MARGINAL edge (WR={:.1}%, E[PnL]={:.2}%)", wr, expectancy);
        } else {
            info!("  🔴 VERDICT: Model shows NO edge (WR={:.1}%, E[PnL]={:.2}%)", wr, expectancy);
        }
        info!("  NOTE: Check cumulative thresholds — higher pred cutoff may yield positive edge");
    } else {
        info!("  ⚠️  No trades generated. Model never predicted above min_signal={:.2}", min_signal);
    }

    info!("");
    info!("  Total time: {:.1}s ({:.1}min)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);

    Ok(())
}
