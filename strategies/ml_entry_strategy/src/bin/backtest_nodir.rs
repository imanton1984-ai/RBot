// strategies/ml_entry_strategy/src/bin/backtest_nodir.rs
//
// NoDir Backtest — P(super_long) + P(super_short) approach
//
// Instead of P(super) + P(direction), we use two directional models:
//   - super_long_v1_tf{X}.ubj  → P(strong upward move)
//   - super_short_v1_tf{X}.ubj → P(strong downward move)
//
// Signal logic:
//   - If P(super_long) >= threshold  → LONG
//   - If P(super_short) >= threshold → SHORT
//   - If both >= threshold → pick higher probability
//   - If neither → skip
//
// No direction model, no heuristic filter, no pipeline dependency.
// Direct Booster loading + batch feature building + batch inference.
//
// USAGE:
//   cargo run --release -p ml_entry_strategy --bin super_entry_backtest_nodir
//
// ENV VARS:
//   DATABASE_URL                     — postgres connection
//   SUPER_ENTRY_P_THRESHOLD=0.55    — min probability for signal
//   WFO_MIN_DATE=2026-01-13         — only count trades after this date
//   SUPER_ENTRY_USE_GPU=true        — use GPU inference

use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use dotenvy::dotenv;
use sqlx::PgPool;
use std::collections::HashMap;

use ml_entry_strategy::config::SuperEntryConfig;
use ml_entry_strategy::dataset::{
    fetch_all_candles_for_tf, CandleWithIndicators,
    compute_dynamic_features_with_htf,
};
use ml_entry_strategy::heuristic::get_higher_tf;

use predictors::ml::xgb_runtime::{Booster, Device, ModelKind};

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

// Batch inference: features are built per-symbol because XGBoost predict_dense_cpu
// handles the full batch internally. No manual batching needed.

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum TradeOutcome {
    Win,
    Loss,
    Expired,
}

#[derive(Debug, Clone)]
struct TradeResult {
    symbol: String,
    tf_minutes: i32,
    direction: i8,
    entry_price: f64,
    exit_price: f64,
    pnl_pct: f64,
    outcome: TradeOutcome,
    bars_to_outcome: usize,
    p_long: f32,
    p_short: f32,
    entry_time: DateTime<Utc>,
}

// ─────────────────────────────────────────────────────────────────────
// Feature building (batch, zero-copy f32)
// ─────────────────────────────────────────────────────────────────────

/// Build feature vector for candle at index `i` in f32 directly.
/// Matches the 128-feature layout from pipeline.rs exactly.
fn build_features_f32(
    candles: &[CandleWithIndicators],
    i: usize,
    htf_candles: Option<&[CandleWithIndicators]>,
    features_out: &mut Vec<f32>,
) {
    let c = &candles[i];
    let close = c.close;
    let safe_div = |a: f64, b: f64| -> f32 {
        if b.abs() > 1e-12 { (a / b) as f32 } else { 0.0f32 }
    };

    // Raw indicators (33 features)
    features_out.push(c.rsi as f32);
    features_out.push(c.cci as f32);
    features_out.push(c.stoch_k as f32);
    features_out.push(c.stoch_d as f32);
    features_out.push(c.williams as f32);
    features_out.push(c.macd as f32);
    features_out.push(c.macd_signal as f32);
    features_out.push(c.macd_hist as f32);
    features_out.push(c.adx as f32);
    features_out.push(c.sma as f32);
    features_out.push(c.ema_20 as f32);
    features_out.push(c.ema_50 as f32);
    features_out.push(c.ema_200 as f32);
    features_out.push(c.bb_upper as f32);
    features_out.push(c.bb_mid as f32);
    features_out.push(c.bb_lower as f32);
    features_out.push(c.atr as f32);
    features_out.push(c.obv as f32);
    features_out.push(c.vwap as f32);
    features_out.push(c.volume_spike as f32);
    features_out.push(c.trend as f32);
    features_out.push(c.trend_short as f32);
    features_out.push(c.poc as f32);
    features_out.push(c.alligator_jaw as f32);
    features_out.push(c.alligator_teeth as f32);
    features_out.push(c.alligator_lips as f32);
    features_out.push(c.mfi as f32);
    features_out.push(c.fibo_pivot as f32);
    features_out.push(c.fibo_r1 as f32);
    features_out.push(c.fibo_s1 as f32);
    features_out.push(c.supertrend as f32);
    features_out.push(c.supertrend_dir as f32);
    features_out.push(c.cmf as f32);

    // Derived features (19 features)
    features_out.push((c.rsi / 100.0) as f32);
    features_out.push((c.cci / 200.0) as f32);
    features_out.push((c.stoch_k / 100.0) as f32);
    features_out.push(((c.williams + 100.0) / 100.0) as f32);
    let bb_range = c.bb_upper - c.bb_lower;
    features_out.push(if bb_range.abs() > 1e-12 {
        ((close - c.bb_lower) / bb_range) as f32
    } else { 0.5f32 });
    features_out.push(safe_div(bb_range, close) * 100.0);
    features_out.push(safe_div(c.atr, close) * 100.0);
    features_out.push(safe_div(close - c.sma, close) * 100.0);
    features_out.push(safe_div(close - c.ema_20, close) * 100.0);
    features_out.push(safe_div(close - c.ema_50, close) * 100.0);
    features_out.push(safe_div(close - c.ema_200, close) * 100.0);
    features_out.push(safe_div(close - c.vwap, close) * 100.0);
    features_out.push(safe_div(c.macd_hist, close) * 1000.0);
    features_out.push(0.0f32); // obv_change_pct (N/A)
    features_out.push(if c.volume_spike > 2.0 { 1.0f32 } else { 0.0f32 });
    features_out.push((c.mfi / 100.0) as f32);
    features_out.push(safe_div(close - c.fibo_pivot, close) * 100.0);
    features_out.push(safe_div(close - c.supertrend, close) * 100.0);
    features_out.push(safe_div(c.alligator_jaw - c.alligator_lips, close) * 100.0);

    // Dynamic temporal features (76 features)
    let htf_candle_ref = htf_candles.and_then(|htf| {
        let target_time = candles[i].time;
        let idx = htf.partition_point(|c| c.time <= target_time);
        if idx > 0 { Some(&htf[idx - 1]) } else { None }
    });
    let dyn_feats = compute_dynamic_features_with_htf(candles, i, htf_candle_ref);
    for v in &dyn_feats {
        features_out.push(*v as f32);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Trade Simulation
// ─────────────────────────────────────────────────────────────────────

fn simulate_trade(
    candles: &[CandleWithIndicators],
    signal_idx: usize,
    direction: i8,
    entry_price: f64,
    tp_pct: f64,
    sl_pct: f64,
    max_hold: usize,
) -> TradeResult {
    let tp_price = if direction == 1 {
        entry_price * (1.0 + tp_pct / 100.0)
    } else {
        entry_price * (1.0 - tp_pct / 100.0)
    };
    let sl_price = if direction == 1 {
        entry_price * (1.0 - sl_pct / 100.0)
    } else {
        entry_price * (1.0 + sl_pct / 100.0)
    };

    let end_idx = (signal_idx + max_hold).min(candles.len() - 1);

    for k in (signal_idx + 1)..=end_idx {
        let high = candles[k].high;
        let low = candles[k].low;

        if direction == 1 {
            // LONG: SL first (conservative)
            if low <= sl_price {
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0, direction, entry_price,
                    exit_price: sl_price,
                    pnl_pct: (sl_price - entry_price) / entry_price * 100.0,
                    outcome: TradeOutcome::Loss,
                    bars_to_outcome: k - signal_idx,
                    p_long: 0.0, p_short: 0.0,
                    entry_time: candles[signal_idx].time,
                };
            }
            if high >= tp_price {
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0, direction, entry_price,
                    exit_price: tp_price,
                    pnl_pct: (tp_price - entry_price) / entry_price * 100.0,
                    outcome: TradeOutcome::Win,
                    bars_to_outcome: k - signal_idx,
                    p_long: 0.0, p_short: 0.0,
                    entry_time: candles[signal_idx].time,
                };
            }
        } else {
            // SHORT: SL first (conservative)
            if high >= sl_price {
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0, direction, entry_price,
                    exit_price: sl_price,
                    pnl_pct: (entry_price - sl_price) / entry_price * 100.0,
                    outcome: TradeOutcome::Loss,
                    bars_to_outcome: k - signal_idx,
                    p_long: 0.0, p_short: 0.0,
                    entry_time: candles[signal_idx].time,
                };
            }
            if low <= tp_price {
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0, direction, entry_price,
                    exit_price: tp_price,
                    pnl_pct: (entry_price - tp_price) / entry_price * 100.0,
                    outcome: TradeOutcome::Win,
                    bars_to_outcome: k - signal_idx,
                    p_long: 0.0, p_short: 0.0,
                    entry_time: candles[signal_idx].time,
                };
            }
        }
    }

    // Expired
    let last_close = candles[end_idx].close;
    let pnl = if direction == 1 {
        (last_close - entry_price) / entry_price * 100.0
    } else {
        (entry_price - last_close) / entry_price * 100.0
    };

    TradeResult {
        symbol: candles[signal_idx].symbol.clone(),
        tf_minutes: 0, direction, entry_price,
        exit_price: last_close,
        pnl_pct: pnl,
        outcome: TradeOutcome::Expired,
        bars_to_outcome: end_idx - signal_idx,
        p_long: 0.0, p_short: 0.0,
        entry_time: candles[signal_idx].time,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct TfMetrics {
    total: usize,
    wins: usize,
    losses: usize,
    expired: usize,
    total_pnl: f64,
    pnl_values: Vec<f64>,
    total_candles: usize,
}

impl TfMetrics {
    fn win_rate(&self) -> f64 {
        if self.total > 0 { self.wins as f64 / self.total as f64 * 100.0 } else { 0.0 }
    }
    fn avg_pnl(&self) -> f64 {
        if self.total > 0 { self.total_pnl / self.total as f64 } else { 0.0 }
    }
    fn sharpe(&self) -> f64 {
        if self.total < 2 { return 0.0; }
        let mean = self.avg_pnl();
        let var = self.pnl_values.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / self.total as f64;
        if var > 0.0 { mean / var.sqrt() } else { 0.0 }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // Logging
    let log_path = "logs/super_entry_backtest_nodir.log";
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

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let config = SuperEntryConfig::from_env();

    let p_threshold: f32 = std::env::var("SUPER_ENTRY_P_THRESHOLD")
        .ok().and_then(|v| v.parse().ok())
        .unwrap_or(config.p_threshold as f32);

    let use_gpu = std::env::var("SUPER_ENTRY_USE_GPU")
        .unwrap_or_default().parse::<bool>().unwrap_or(false);
    let device = if use_gpu { Device::Cuda } else { Device::Cpu };

    let wfo_min_date: Option<DateTime<Utc>> = std::env::var("WFO_MIN_DATE").ok()
        .and_then(|s| NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()
            .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc()));

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   SUPER ENTRY BACKTEST — NoDir (P(super_long)+P(super_short)) ║");
    println!("║   No direction model — direction embedded in super labels     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  P threshold:     {}", p_threshold);
    println!("  Warmup bars:     {}", config.warmup_bars);
    println!("  Lookahead:       {}", config.lookahead_bars);
    println!("  Max hold:        {}", config.effective_max_hold());
    println!("  SL fraction:     {}", config.sl_fraction);
    println!("  GPU:             {}", use_gpu);
    if let Some(ref d) = wfo_min_date {
        println!("  WFO OOS from:    {}", d.format("%Y-%m-%d"));
    }
    println!();

    // ── Load Models ──
    let active_tfs = SuperEntryConfig::timeframes();
    let mut long_models: HashMap<i32, Booster> = HashMap::new();
    let mut short_models: HashMap<i32, Booster> = HashMap::new();

    for &tf in active_tfs {
        let long_path = format!("models/super_long_v1_tf{}.ubj", tf);
        let short_path = format!("models/super_short_v1_tf{}.ubj", tf);

        if std::path::Path::new(&long_path).exists() {
            match Booster::load(&long_path, device) {
                Ok(b) => {
                    println!("  ✅ Loaded super_long  TF {}m: {}", tf, long_path);
                    long_models.insert(tf, b);
                }
                Err(e) => println!("  ❌ Failed super_long  TF {}m: {}", tf, e),
            }
        } else {
            println!("  ⚠️  Missing super_long  TF {}m: {}", tf, long_path);
        }

        if std::path::Path::new(&short_path).exists() {
            match Booster::load(&short_path, device) {
                Ok(b) => {
                    println!("  ✅ Loaded super_short TF {}m: {}", tf, short_path);
                    short_models.insert(tf, b);
                }
                Err(e) => println!("  ❌ Failed super_short TF {}m: {}", tf, e),
            }
        } else {
            println!("  ⚠️  Missing super_short TF {}m: {}", tf, short_path);
        }
    }

    if long_models.is_empty() && short_models.is_empty() {
        eprintln!("No models found! Train first with:");
        eprintln!("  python trainer/src/train_super_entry_nodir.py --gpu");
        return Ok(());
    }
    println!();

    let pool = PgPool::connect(&db_url).await?;

    // ── PHASE 1: Pre-load ALL TF data ──
    println!("  Loading candle data...");
    let t0_load = std::time::Instant::now();

    // Collect all TFs needed (active + HTF for features)
    let mut all_tfs_needed: Vec<i32> = active_tfs.to_vec();
    for &tf in active_tfs {
        if let Some(h) = get_higher_tf(tf) {
            if !all_tfs_needed.contains(&h) { all_tfs_needed.push(h); }
        }
    }
    all_tfs_needed.sort_unstable();
    all_tfs_needed.dedup();

    let mut tf_store: HashMap<i32, HashMap<String, Vec<CandleWithIndicators>>> = HashMap::new();
    for &tf in &all_tfs_needed {
        let limit = match tf {
            1 => 5000, 5 => 12000, 15 => 12000, 60 => 12000,
            240 => 12000, 1440 => 3700, _ => 5000,
        };
        let t0 = std::time::Instant::now();
        let grouped = fetch_all_candles_for_tf(&pool, tf, limit).await?;
        let n_symbols = grouped.len();
        let n_candles: usize = grouped.values().map(|v| v.len()).sum();
        println!("    TF {:>5}m: {} symbols, {} candles ({:.1}s)",
            tf, n_symbols, n_candles, t0.elapsed().as_secs_f64());
        tf_store.insert(tf, grouped);
    }
    println!("  All data loaded in {:.1}s\n", t0_load.elapsed().as_secs_f64());

    // ── PHASE 2: Process each active TF ──
    let ncol = ml_entry_strategy::config::total_feature_count();
    let mut metrics_by_tf: HashMap<i32, TfMetrics> = HashMap::new();
    let mut all_trades: Vec<TradeResult> = Vec::new();
    let t0_total = std::time::Instant::now();

    for &tf in active_tfs {
        let has_long = long_models.contains_key(&tf);
        let has_short = short_models.contains_key(&tf);
        if !has_long && !has_short { continue; }

        let t0_tf = std::time::Instant::now();
        let target_pct = config.target_pct_for_tf(tf);
        let sl_pct = config.sl_pct_for_tf(tf);
        let max_hold = config.effective_max_hold();
        let mut tf_metrics = TfMetrics::default();

        let htf_tf = get_higher_tf(tf);

        let tf_data = match tf_store.get(&tf) {
            Some(d) => d,
            None => continue,
        };

        for (_symbol, candles) in tf_data {
            if candles.len() < config.warmup_bars + config.lookahead_bars + 1 {
                continue;
            }

            let symbol = &candles[0].symbol;
            let htf_candles_ref: Option<&[CandleWithIndicators]> = htf_tf.and_then(|htf| {
                tf_store.get(&htf)
                    .and_then(|tf_map| tf_map.get(symbol))
                    .map(|v| v.as_slice())
            });

            let warmup = config.warmup_bars;
            let n = candles.len();
            let batch_size = n - warmup;

            // Build feature matrix for batch inference
            let mut features_flat: Vec<f32> = Vec::with_capacity(batch_size * ncol);
            for i in warmup..n {
                build_features_f32(candles, i, htf_candles_ref, &mut features_flat);
            }

            // Batch predict P(super_long)
            let p_long_vec: Vec<f32> = if has_long {
                let model = &long_models[&tf];
                model.predict_dense_cpu(&features_flat, batch_size, ncol, ModelKind::Regressor1)?
                    .iter().map(|&v| v.clamp(0.0, 1.0)).collect()
            } else {
                vec![0.0f32; batch_size]
            };

            // Batch predict P(super_short)
            let p_short_vec: Vec<f32> = if has_short {
                let model = &short_models[&tf];
                model.predict_dense_cpu(&features_flat, batch_size, ncol, ModelKind::Regressor1)?
                    .iter().map(|&v| v.clamp(0.0, 1.0)).collect()
            } else {
                vec![0.0f32; batch_size]
            };

            // Free features immediately
            drop(features_flat);

            tf_metrics.total_candles += batch_size;

            // Generate signals and simulate trades
            for idx in 0..batch_size {
                let candle_idx = warmup + idx;
                let entry_price = candles[candle_idx].close;
                let entry_time = candles[candle_idx].time;

                // WFO date filter
                if let Some(min_date) = wfo_min_date {
                    if entry_time < min_date { continue; }
                }

                let pl = p_long_vec[idx];
                let ps = p_short_vec[idx];

                // Decision logic
                let direction: i8 = if pl >= p_threshold && ps >= p_threshold {
                    // Both above threshold → pick stronger
                    if pl >= ps { 1 } else { -1 }
                } else if pl >= p_threshold {
                    1 // LONG
                } else if ps >= p_threshold {
                    -1 // SHORT
                } else {
                    continue; // No signal
                };

                // Simulate trade
                let mut trade = simulate_trade(
                    candles, candle_idx, direction,
                    entry_price, target_pct, sl_pct, max_hold,
                );
                trade.tf_minutes = tf;
                trade.p_long = pl;
                trade.p_short = ps;

                tf_metrics.total += 1;
                tf_metrics.total_pnl += trade.pnl_pct;
                tf_metrics.pnl_values.push(trade.pnl_pct);
                match trade.outcome {
                    TradeOutcome::Win => tf_metrics.wins += 1,
                    TradeOutcome::Loss => tf_metrics.losses += 1,
                    TradeOutcome::Expired => tf_metrics.expired += 1,
                }
                all_trades.push(trade);
            }
        }

        println!("  TF {:>5}m: {} trades in {:.1}s",
            tf, tf_metrics.total, t0_tf.elapsed().as_secs_f64());
        metrics_by_tf.insert(tf, tf_metrics);
    }

    println!("\n  Total backtest time: {:.1}s\n", t0_total.elapsed().as_secs_f64());

    // ── RESULTS ──
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║     SUPER ENTRY BACKTEST — NoDir (P(super_long) + P(super_short))    ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    if let Some(ref d) = wfo_min_date {
        println!("  ⚠️  WFO OOS filter: only trades after {}", d.format("%Y-%m-%d"));
    }
    println!("  P threshold: {:.2}\n", p_threshold);

    println!(
        "{:<6} {:>7} {:>6} {:>6} {:>7} {:>8} {:>9} {:>8}",
        "TF", "Trades", "Wins", "Loss", "Exprd", "WinRate", "AvgPnL%", "Sharpe"
    );
    println!("{}", "-".repeat(75));

    let mut tfs: Vec<i32> = metrics_by_tf.keys().copied().collect();
    tfs.sort();

    for tf in &tfs {
        let m = &metrics_by_tf[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
        println!(
            "{:<6} {:>7} {:>6} {:>6} {:>7} {:>7.1}% {:>8.4}% {:>8.3}",
            tf_name, m.total, m.wins, m.losses, m.expired,
            m.win_rate(), m.avg_pnl(), m.sharpe(),
        );
    }

    // Overall
    let total_trades = all_trades.len();
    let total_wins = all_trades.iter().filter(|t| t.outcome == TradeOutcome::Win).count();
    let total_losses = all_trades.iter().filter(|t| t.outcome == TradeOutcome::Loss).count();
    let total_expired = all_trades.iter().filter(|t| t.outcome == TradeOutcome::Expired).count();
    let total_pnl: f64 = all_trades.iter().map(|t| t.pnl_pct).sum();

    println!("{}", "-".repeat(75));
    println!(
        "TOTAL  {:>7} {:>6} {:>6} {:>7} {:>7.1}% {:>8.4}%",
        total_trades, total_wins, total_losses, total_expired,
        if total_trades > 0 { total_wins as f64 / total_trades as f64 * 100.0 } else { 0.0 },
        if total_trades > 0 { total_pnl / total_trades as f64 } else { 0.0 }
    );
    println!();

    // ── P BUCKET ANALYSIS ──
    println!("======== P(super) BUCKET ANALYSIS ========");
    let p_buckets: Vec<(f32, f32)> = vec![
        (0.55, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01),
    ];

    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "TF", "0.55-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90+"
    );

    let mut by_tf_trades: HashMap<i32, Vec<&TradeResult>> = HashMap::new();
    for t in &all_trades {
        by_tf_trades.entry(t.tf_minutes).or_default().push(t);
    }
    let mut sorted_tfs: Vec<i32> = by_tf_trades.keys().copied().collect();
    sorted_tfs.sort();

    for tf in &sorted_tfs {
        let group = &by_tf_trades[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
        let mut cells: Vec<String> = Vec::new();
        for (lo, hi) in &p_buckets {
            let bucket: Vec<&&TradeResult> = group.iter()
                .filter(|t| {
                    let p = if t.direction == 1 { t.p_long } else { t.p_short };
                    p >= *lo && p < *hi
                })
                .collect();
            let n = bucket.len();
            if n == 0 { cells.push("  -  ".to_string()); }
            else {
                let w = bucket.iter().filter(|t| t.outcome == TradeOutcome::Win).count();
                cells.push(format!("{:.0}%({}) ", w as f64 / n as f64 * 100.0, n));
            }
        }
        println!("{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}",
            tf_name, cells[0], cells[1], cells[2], cells[3], cells[4]);
    }
    println!();

    // ── LONG vs SHORT breakdown ──
    println!("======== DIRECTION BREAKDOWN (LONG vs SHORT) ========");
    println!(
        "{:<6} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "TF", "LONG", "SHORT", "LongWR%", "ShortWR%", "LongPnL%", "ShortPnL%"
    );
    for tf in &sorted_tfs {
        let group = &by_tf_trades[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
        let longs: Vec<&&TradeResult> = group.iter().filter(|t| t.direction == 1).collect();
        let shorts: Vec<&&TradeResult> = group.iter().filter(|t| t.direction == -1).collect();
        let wr_l = if !longs.is_empty() {
            longs.iter().filter(|t| t.outcome == TradeOutcome::Win).count() as f64 / longs.len() as f64 * 100.0
        } else { 0.0 };
        let wr_s = if !shorts.is_empty() {
            shorts.iter().filter(|t| t.outcome == TradeOutcome::Win).count() as f64 / shorts.len() as f64 * 100.0
        } else { 0.0 };
        let pnl_l = if !longs.is_empty() { longs.iter().map(|t| t.pnl_pct).sum::<f64>() / longs.len() as f64 } else { 0.0 };
        let pnl_s = if !shorts.is_empty() { shorts.iter().map(|t| t.pnl_pct).sum::<f64>() / shorts.len() as f64 } else { 0.0 };
        println!("{:<6} {:>8} {:>8} {:>9.1}% {:>9.1}% {:>9.4}% {:>9.4}%",
            tf_name, longs.len(), shorts.len(), wr_l, wr_s, pnl_l, pnl_s);
    }
    println!();

    // ── Conflict analysis: how often both models fire ──
    println!("======== CONFLICT ANALYSIS (both P >= threshold) ========");
    let mut conflict_count = 0usize;
    let mut long_only = 0usize;
    let mut short_only = 0usize;
    for t in &all_trades {
        if t.p_long >= p_threshold && t.p_short >= p_threshold {
            conflict_count += 1;
        } else if t.p_long >= p_threshold {
            long_only += 1;
        } else {
            short_only += 1;
        }
    }
    println!("  Long-only signals:  {}", long_only);
    println!("  Short-only signals: {}", short_only);
    println!("  Conflict (both):    {} ({:.1}%)",
        conflict_count,
        if total_trades > 0 { conflict_count as f64 / total_trades as f64 * 100.0 } else { 0.0 }
    );
    println!();

    // Export CSV
    let csv_output = std::env::var("SUPER_ENTRY_BACKTEST_CSV").unwrap_or_default();
    if !csv_output.is_empty() {
        use std::io::Write;
        let mut file = std::fs::File::create(&csv_output)?;
        writeln!(file,
            "symbol,tf_minutes,direction,entry_price,exit_price,pnl_pct,outcome,\
             bars_to_outcome,p_long,p_short,entry_time"
        )?;
        for t in &all_trades {
            writeln!(file,
                "{},{},{},{:.6},{:.6},{:.6},{},{},{:.4},{:.4},{}",
                t.symbol, t.tf_minutes, t.direction, t.entry_price, t.exit_price,
                t.pnl_pct,
                match t.outcome { TradeOutcome::Win => "win", TradeOutcome::Loss => "loss", TradeOutcome::Expired => "expired" },
                t.bars_to_outcome, t.p_long, t.p_short,
                t.entry_time.to_rfc3339(),
            )?;
        }
        println!("  Backtest CSV: {}", csv_output);
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         NODIR BACKTEST COMPLETE                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
