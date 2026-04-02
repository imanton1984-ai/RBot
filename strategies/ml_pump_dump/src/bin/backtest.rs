// strategies/ml_pump_dump/src/bin/backtest.rs
//
// Pump/Dump Walk-Forward Trade Simulation v4
//
// CHANGES v4 (bulk-load optimization + readable logs):
//   - Bulk-loads ALL candles per TF in 5 SQL queries (instead of 332×5 = 1660)
//   - In-memory symbol processing (no per-symbol DB queries)
//   - Progress bar with ETA
//   - Clean, structured final report
//
// NO LOOK-AHEAD BIAS:
//   - Walks bar-by-bar on target TFs (5m, 15m, 1h, 4h)
//   - Features extracted from PAST candles only (before signal bar)
//   - When model predicts pump/dump (pred >= threshold), opens a virtual trade
//   - Checks SL/TP hit on future candles within max_hold window
//   - Comprehensive statistics: win rate, P&L, hold times, per-bucket analysis
//
// PERFORMANCE:
//   - Bulk data loading (5 queries total, ~20-40s)
//   - Batch model predictions (256 rows at once via XGBoost)
//   - Sequential symbol processing from in-memory data (no DB round-trips)
//   - Expected runtime: ~2-3 min for 300+ symbols (was ~60+ min)
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
use ml_pump_dump::dataset::fetch_all_candles_for_tf;

use predictors::ml::xgb_runtime::{Booster, Device, ModelKind};

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

/// Target timeframes for trade simulation
/// Trade execution TFs. 240m excluded as it's only a feature TF (drill-down from daily).
const TARGET_TFS: &[i32] = &[5, 15, 60];

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
                // (tp above entry, sl below entry)
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
                // (tp below entry, sl above entry)
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

/// Compact one-line summary for a trade group
fn one_line_summary(label: &str, trades: &[&SimTrade]) -> String {
    if trades.is_empty() {
        return format!("{}: — no trades —", label);
    }
    let total = trades.len();
    let wins = trades.iter().filter(|t| t.outcome == TradeOutcome::TpHit).count();
    let losses = trades.iter().filter(|t| t.outcome == TradeOutcome::SlHit).count();
    let wr = wins as f64 / total as f64 * 100.0;
    let total_pnl: f64 = trades.iter().map(|t| t.pnl_pct).sum();
    let avg_pnl = total_pnl / total as f64;
    format!("{}: {} trades | WR {:.1}% (TP:{} SL:{}) | AvgPnL {:.2}% | TotalPnL {:.1}%",
            label, total, wr, wins, losses, avg_pnl, total_pnl)
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
        .ok().and_then(|v| v.parse().ok()).unwrap_or(3.0);
    let sl_fraction: f64 = std::env::var("PD_SL_FRACTION")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0.3);
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

    // PD_REVERSE_MODELS=1 → use pump_model for SHORT signals, dump_model for LONG signals
    let reverse_models = std::env::var("PD_REVERSE_MODELS")
        .map_or(false, |v| v == "1" || v.to_lowercase() == "true");

    let wfo_min_date: Option<DateTime<Utc>> = std::env::var("WFO_MIN_DATE").ok()
        .and_then(|s| NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()
            .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc()));

    info!("╔═══════════════════════════════════════════════════════════╗");
    info!("║  Pump/Dump Walk-Forward Backtest v4 (bulk-load)           ║");
    info!("║  True Trade Simulation — No Look-Ahead Bias               ║");
    info!("╚═══════════════════════════════════════════════════════════╝");
    info!("  Target TFs: {:?}", TARGET_TFS);
    info!("  Max hold: {} candles", max_hold);
    info!("  Target: {:.1}%, SL: {:.1}% (fraction: {:.2}), R:R = {:.1}:{:.1}",
          target_pct, sl_pct, sl_fraction, target_pct, sl_pct);
    info!("  Pred step: {} | Cooldown: {} bars | Min signal: {:.2}", pred_step, cooldown, min_signal);
    info!("  Device: {:?} | Model type: {}", device, model_type);
    if let Some(d) = &wfo_min_date {
        info!("  WFO OOS filter: signals after {}", d.format("%Y-%m-%d"));
    }
    if reverse_models {
        info!("  ⚠️  REVERSE mode: pump_model → DUMP/SHORT, dump_model → PUMP/LONG (contrarian)");
    } else {
        info!("  ℹ️  Direct mode: pump_model → PUMP/LONG, dump_model → DUMP/SHORT");
    }
    info!("  ℹ️  Conflict resolution: if both fire on same bar, take strongest direction only");
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

    // ─── Phase 1: Bulk Load ALL data (5 queries total) ───
    info!("");
    info!("  ⏳ Bulk loading all candle data...");

    let tf_limits: HashMap<i32, usize> = vec![
        (1440, 3700), (240, 12000), (60, 12000),
        (15, 12000), (5, 12000),
    ].into_iter().collect();

    let mut bulk_data: HashMap<i32, HashMap<String, Vec<CandleInd>>> = HashMap::new();
    let mut all_symbols: Vec<String> = Vec::new();

    for &tf in ANALYSIS_TIMEFRAMES {
        let limit = tf_limits.get(&tf).copied().unwrap_or(5000);
        let t0 = std::time::Instant::now();
        let grouped = fetch_all_candles_for_tf(&pool, tf, limit).await?;
        let n_syms = grouped.len();
        let n_candles: usize = grouped.values().map(|v| v.len()).sum();
        info!("    TF {:>4}m: {:>3} symbols, {:>8} candles  ({:.1}s)",
              tf, n_syms, n_candles, t0.elapsed().as_secs_f64());

        if all_symbols.is_empty() {
            all_symbols = grouped.keys().cloned().collect();
            all_symbols.sort();
        }

        bulk_data.insert(tf, grouped);
    }

    let load_elapsed = total_start.elapsed().as_secs_f64();
    info!("  ✅ Bulk load complete: {} symbols in {:.1}s", all_symbols.len(), load_elapsed);
    info!("");

    // Minimum history needed: lookback + 50 bars (for temporal features lookback)
    let min_history = lookback + 50;

    // ─── Phase 2: Walk-Forward Simulation (in-memory, no DB queries) ───
    info!("  🚀 Starting walk-forward simulation...");

    let mut all_trades: Vec<SimTrade> = Vec::new();
    let mut total_predictions = 0u64;
    let mut total_bars_scanned = 0u64;
    let mut symbols_processed = 0u32;
    let mut symbols_skipped = 0u32;

    let n_symbols = all_symbols.len();
    let progress_interval = (n_symbols / 20).max(1); // ~20 progress updates

    for (si, symbol) in all_symbols.iter().enumerate() {
        // Build per-symbol TF map from bulk data (no DB queries!)
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
                    // In reverse mode: dump_model → pump signals, pump_model → dump signals
                    let (pm, dm) = if reverse_models {
                        (dump_model.as_ref(), pump_model.as_ref())
                    } else {
                        (pump_model.as_ref(), dump_model.as_ref())
                    };
                    total_predictions += flush_predictions(
                        &batch_features, &batch_indices,
                        pm, dm,
                        n_features, min_signal, &mut signals,
                    )?;
                    batch_features.clear();
                    batch_indices.clear();
                }

                bar_idx += pred_step;
            }

            // Flush remaining
            if !batch_indices.is_empty() {
                let (pm, dm) = if reverse_models {
                    (dump_model.as_ref(), pump_model.as_ref())
                } else {
                    (pump_model.as_ref(), dump_model.as_ref())
                };
                total_predictions += flush_predictions(
                    &batch_features, &batch_indices,
                    pm, dm,
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

                // ── Conflict resolution: if both directions fire, take only the stronger one ──
                let (eff_pump, eff_dump) = match (pump_pred, dump_pred) {
                    (Some(pp), Some(dp)) if pp > dp => (Some(pp), None),
                    (Some(pp), Some(dp)) if dp > pp => (None, Some(dp)),
                    (Some(_), Some(_)) => (None, None), // equal scores — skip both
                    _ => (pump_pred, dump_pred),
                };

                // ── Pump signal (LONG) ──
                if let Some(pred) = eff_pump {
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

                // ── Dump signal (SHORT) ──
                if let Some(pred) = eff_dump {
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

        // Progress with ETA
        if (si + 1) % progress_interval == 0 || si == 0 || si + 1 == n_symbols {
            let elapsed = total_start.elapsed().as_secs_f64();
            let pct = (si + 1) as f64 / n_symbols as f64 * 100.0;
            let rate = (si + 1) as f64 / elapsed;
            let remaining = (n_symbols - si - 1) as f64 / rate;
            info!("  [{:>3}/{}] {:>6.1}% | {} — {} trades | total: {} | ETA: {:.0}s",
                  si + 1, n_symbols, pct, symbol, sym_trades,
                  all_trades.len(), remaining);
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // RESULTS — Clear, Structured Final Report
    // ═════════════════════════════════════════════════════════════════
    let elapsed = total_start.elapsed();

    info!("");
    info!("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
    info!("┃          PUMP/DUMP BACKTEST — FINAL REPORT (v4)                  ┃");
    info!("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
    info!("");

    // ─── Quick Stats ───
    info!("  ┌─────────── EXECUTION ───────────┐");
    info!("  │ Symbols: {} processed, {} skipped │", symbols_processed, symbols_skipped);
    info!("  │ Bars scanned: {:>12}       │", total_bars_scanned);
    info!("  │ ML predictions: {:>10}       │", total_predictions);
    info!("  │ Total trades: {:>12}       │", all_trades.len());
    info!("  │ Time: {:.1}s ({:.1}min)              │", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);
    info!("  │ Data load: {:.1}s | Sim: {:.1}s      │", load_elapsed, elapsed.as_secs_f64() - load_elapsed);
    info!("  └─────────────────────────────────┘");
    info!("");

    // ─── Quick Overview Table ───
    if !all_trades.is_empty() {
        info!("  ┌─────────── QUICK OVERVIEW ─────────────────────────────────────┐");

        // Overall
        let all_refs: Vec<&SimTrade> = all_trades.iter().collect();
        info!("  │ {}", one_line_summary("ALL", &all_refs));

        // Pump vs Dump
        let pump_all: Vec<&SimTrade> = all_trades.iter().filter(|t| t.event_type == EventType::Pump).collect();
        let dump_all: Vec<&SimTrade> = all_trades.iter().filter(|t| t.event_type == EventType::Dump).collect();
        info!("  │ {}", one_line_summary("PUMP", &pump_all));
        info!("  │ {}", one_line_summary("DUMP", &dump_all));
        info!("  │");

        // Per-TF one-liner
        for &tf in TARGET_TFS {
            let tf_trades: Vec<&SimTrade> = all_trades.iter().filter(|t| t.tf_minutes == tf).collect();
            if !tf_trades.is_empty() {
                info!("  │ {}", one_line_summary(&format!("TF {:>3}m", tf), &tf_trades));
            }
        }

        // Breakeven WR
        let breakeven_wr = sl_pct / (target_pct + sl_pct) * 100.0;
        info!("  │");
        info!("  │ Breakeven WR (R:R = {:.1}:{:.1}): {:.1}%", target_pct, sl_pct, breakeven_wr);
        info!("  └───────────────────────────────────────────────────────────────┘");
    }
    info!("");

    // ─── Per-TF Detailed Results ───
    for &tf in TARGET_TFS {
        let tf_trades: Vec<&SimTrade> = all_trades.iter()
            .filter(|t| t.tf_minutes == tf)
            .collect();
        if tf_trades.is_empty() {
            info!("  ═══ TF: {}m ═══  (no trades)", tf);
            info!("");
            continue;
        }

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

    // ─── Overall Summary (Detailed) ───
    if !all_trades.is_empty() {
        info!("  ═══════════════════════════════════════════════════");
        info!("  ═══ OVERALL DETAILED SUMMARY ═══");
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
            info!("      {:>14}: {:>5} trades, WR: {:>5.1}%, TotalPnL: {:>8.2}%",
                  sym, count, *wins as f64 / *count as f64 * 100.0, sym_pnl);
        }

        // ── Top 10 best/worst trades ──
        info!("");
        info!("    ─── Top 10 Best Trades ───");
        let mut sorted = all_trades.clone();
        sorted.sort_by(|a, b| b.pnl_pct.partial_cmp(&a.pnl_pct).unwrap_or(std::cmp::Ordering::Equal));
        for (i, t) in sorted.iter().take(10).enumerate() {
            info!("      #{:>2}: {:>14} {} {} TF={}m pred={:.3} entry={:.4} P&L={:>+.2}% hold={} {}",
                  i + 1, t.symbol, t.entry_time.format("%Y-%m-%d %H:%M"),
                  t.event_type, t.tf_minutes, t.pred, t.entry_price,
                  t.pnl_pct, t.hold_candles, t.outcome);
        }

        info!("");
        info!("    ─── Top 10 Worst Trades ───");
        for (i, t) in sorted.iter().rev().take(10).enumerate() {
            info!("      #{:>2}: {:>14} {} {} TF={}m pred={:.3} entry={:.4} P&L={:>+.2}% hold={} {}",
                  i + 1, t.symbol, t.entry_time.format("%Y-%m-%d %H:%M"),
                  t.event_type, t.tf_minutes, t.pred, t.entry_price,
                  t.pnl_pct, t.hold_candles, t.outcome);
        }

        // ═════════════════════════════════════════════════════════
        //  FINAL VERDICT
        // ═════════════════════════════════════════════════════════
        info!("");
        info!("  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
        info!("  ┃                        VERDICT                              ┃");
        info!("  ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫");

        let total = all_trades.len();
        let wins = all_trades.iter().filter(|t| t.outcome == TradeOutcome::TpHit).count();
        let losses = all_trades.iter().filter(|t| t.outcome == TradeOutcome::SlHit).count();
        let wr = wins as f64 / total as f64 * 100.0;
        let avg_pnl: f64 = all_trades.iter().map(|t| t.pnl_pct).sum::<f64>() / total as f64;
        let total_pnl: f64 = all_trades.iter().map(|t| t.pnl_pct).sum();

        // Compute breakeven WR for asymmetric TP/SL
        let breakeven_wr = sl_pct / (target_pct + sl_pct) * 100.0;

        info!("  ┃  Total Trades:  {:>8}                                    ┃", total);
        info!("  ┃  Win Rate:      {:>7.1}%  (TP:{} SL:{})             ┃", wr, wins, losses);
        info!("  ┃  Avg P&L:       {:>+7.2}%                                   ┃", avg_pnl);
        info!("  ┃  Total P&L:     {:>+8.1}%                                  ┃", total_pnl);
        info!("  ┃  Breakeven WR:  {:>7.1}%  (R:R = {:.1}:{:.1})            ┃", breakeven_wr, target_pct, sl_pct);
        info!("  ┃                                                           ┃");

        if avg_pnl > 0.1 && wr > breakeven_wr {
            info!("  ┃  🟢 STRONG EDGE — strategy is profitable                 ┃");
        } else if avg_pnl > 0.0 || wr > breakeven_wr {
            info!("  ┃  🟡 MARGINAL EDGE — needs higher pred threshold           ┃");
        } else {
            info!("  ┃  🔴 NO EDGE — model not profitable at pred ≥ {:.2}        ┃", min_signal);
        }

        // Find best threshold
        let mut best_th = 0.50f32;
        let mut best_avg = f64::NEG_INFINITY;
        for &th in PRED_THRESHOLDS {
            let above: Vec<&SimTrade> = all_trades.iter().filter(|t| t.pred >= th).collect();
            if above.len() < 100 { continue; }
            let avg: f64 = above.iter().map(|t| t.pnl_pct).sum::<f64>() / above.len() as f64;
            if avg > best_avg {
                best_avg = avg;
                best_th = th;
            }
        }
        if best_avg > f64::NEG_INFINITY {
            let best_trades: Vec<&SimTrade> = all_trades.iter().filter(|t| t.pred >= best_th).collect();
            let best_wr = best_trades.iter().filter(|t| t.outcome == TradeOutcome::TpHit).count() as f64
                / best_trades.len() as f64 * 100.0;
            info!("  ┃                                                           ┃");
            info!("  ┃  Best threshold: pred ≥ {:.2} → {} trades, WR={:.1}%, AvgPnL={:+.2}%", 
                  best_th, best_trades.len(), best_wr, best_avg);
        }

        info!("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
    } else {
        info!("  ⚠️  No trades generated. Model never predicted above min_signal={:.2}", min_signal);
    }

    info!("");
    info!("  Total time: {:.1}s ({:.1}min)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);

    Ok(())
}
