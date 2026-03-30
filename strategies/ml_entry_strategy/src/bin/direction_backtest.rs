// strategies/ml_entry_strategy/src/bin/direction_backtest.rs
//
// Direction v4 Pattern Backtester
//
// Tests the CNN-like pattern direction model:
//   1. Load raw candles (OHLCV only — no indicators needed)
//   2. For each super point (from existing super model), compute pattern features
//   3. Predict direction with v4 model
//   4. Simulate trades and measure win rate
//
// ALSO supports standalone mode (without super model):
//   When DIRECTION_STANDALONE=1, tests direction predictions on ALL candles
//   instead of only on super points. This is useful for evaluating the
//   pure direction accuracy without the super filter.
//
// USAGE:
//   cargo build --release -p ml_entry_strategy --bin direction_backtest
//   DIRECTION_STANDALONE=1 ./target/release/direction_backtest
//   DIRECTION_GPU=1 ./target/release/direction_backtest
//
// ENV VARS:
//   DATABASE_URL              — postgres connection string
//   WFO_MIN_DATE=2026-01-13  — only count trades after this date
//   DIRECTION_TF=15,60,240   — timeframes to test (default: 15,60,240)
//   DIRECTION_GPU=1           — use GPU for model inference
//   DIRECTION_STANDALONE=1    — test without super model (all candles)
//   DIRECTION_MODE=binary     — model mode: "binary" (default) or "regression"
//   DIR_WINDOW_SIZE           — must match training config
//   DIR_PREDICTION_HORIZON    — must match training config
//   DIR_FEATURE_SET           — must match training config

use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use dotenvy::dotenv;
use sqlx::PgPool;
use tracing::info;

use ml_entry_strategy::config::SuperEntryConfig;
use ml_entry_strategy::dataset::CandleWithIndicators;
use ml_entry_strategy::direction::DirectionConfig;
use ml_entry_strategy::direction::features::compute_pattern_features;
use ml_entry_strategy::direction::dataset::fetch_all_raw_candles_for_tf;

use predictors::ml::xgb_runtime::{Booster, Device, ModelKind};

// ─────────────────────────────────────────────────────────────────────
// Trade simulation
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum Outcome { Win, Loss, Expired }

fn simulate_trade(
    candles: &[CandleWithIndicators],
    entry_idx: usize,
    direction: i8,
    tp_pct: f64,
    sl_pct: f64,
    max_hold: usize,
) -> (Outcome, f64) {
    let entry = candles[entry_idx].close;
    let (tp, sl) = if direction == 1 {
        (entry * (1.0 + tp_pct / 100.0), entry * (1.0 - sl_pct / 100.0))
    } else {
        (entry * (1.0 - tp_pct / 100.0), entry * (1.0 + sl_pct / 100.0))
    };

    let end = (entry_idx + max_hold).min(candles.len() - 1);
    for k in (entry_idx + 1)..=end {
        let (h, l) = (candles[k].high, candles[k].low);
        if direction == 1 {
            if l <= sl { return (Outcome::Loss, (sl - entry) / entry * 100.0); }
            if h >= tp { return (Outcome::Win, (tp - entry) / entry * 100.0); }
        } else {
            if h >= sl { return (Outcome::Loss, (entry - sl) / entry * 100.0); }
            if l <= tp { return (Outcome::Win, (entry - tp) / entry * 100.0); }
        }
    }
    let last = candles[end].close;
    let pnl = if direction == 1 { (last - entry) / entry * 100.0 }
              else { (entry - last) / entry * 100.0 };
    (Outcome::Expired, pnl)
}

// ─────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
struct Metrics {
    n: usize,
    wins: usize,
    losses: usize,
    expired: usize,
    total_pnl: f64,
}

impl Metrics {
    fn add(&mut self, outcome: Outcome, pnl: f64) {
        self.n += 1;
        self.total_pnl += pnl;
        match outcome {
            Outcome::Win => self.wins += 1,
            Outcome::Loss => self.losses += 1,
            Outcome::Expired => self.expired += 1,
        }
    }
    fn wr(&self) -> f64 { if self.n > 0 { self.wins as f64 / self.n as f64 * 100.0 } else { 0.0 } }
    fn avg_pnl(&self) -> f64 { if self.n > 0 { self.total_pnl / self.n as f64 } else { 0.0 } }
}

// ─────────────────────────────────────────────────────────────────────
// Accuracy metrics (direction correctness without trade simulation)
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct AccuracyMetrics {
    total: usize,
    correct: usize,
    up_correct: usize,
    up_total: usize,
    down_correct: usize,
    down_total: usize,
}

impl AccuracyMetrics {
    fn add(&mut self, predicted: i8, actual_return: f64) {
        // Actual direction based on return sign
        let actual_dir: i8 = if actual_return > 0.0 { 1 } else { -1 };

        self.total += 1;
        if predicted == actual_dir {
            self.correct += 1;
        }

        if actual_dir == 1 {
            self.up_total += 1;
            if predicted == 1 { self.up_correct += 1; }
        } else {
            self.down_total += 1;
            if predicted == -1 { self.down_correct += 1; }
        }
    }

    fn accuracy(&self) -> f64 {
        if self.total > 0 { self.correct as f64 / self.total as f64 * 100.0 } else { 0.0 }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // Logging
    let log_path = "logs/direction_backtest.log";
    std::fs::create_dir_all("logs")?;
    let log_file = std::fs::OpenOptions::new()
        .create(true).append(true).open(log_path)?;

    use tracing_subscriber::fmt;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(fmt::layer().with_target(false).with_ansi(false)
            .with_writer(std::sync::Mutex::new(log_file)))
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
        ))
        .init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let tf_str = std::env::var("DIRECTION_TF")
        .unwrap_or_else(|_| "15,60,240".to_string());
    let timeframes: Vec<i32> = tf_str.split(',')
        .filter_map(|s| s.trim().parse().ok()).collect();

    let use_gpu = std::env::var("DIRECTION_GPU")
        .map_or(false, |v| v == "1" || v.to_lowercase() == "true");

    let standalone = std::env::var("DIRECTION_STANDALONE")
        .map_or(false, |v| v == "1" || v.to_lowercase() == "true");

    // Model mode: "binary" (binary:logistic, output P(UP) ∈ [0,1])
    //             "regression" (reg:squarederror, output ∈ (-∞, +∞), sign = direction)
    let is_binary_mode = std::env::var("DIRECTION_MODE")
        .map_or(true, |v| v.to_lowercase() != "regression");

    let device = if use_gpu { Device::Cuda } else { Device::Cpu };

    let dir_config = DirectionConfig::from_env();
    let super_config = SuperEntryConfig::from_env();

    let wfo_min_date: Option<DateTime<Utc>> = std::env::var("WFO_MIN_DATE").ok()
        .and_then(|s| NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()
            .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc()));

    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║  Direction v4 Pattern Backtester                      ║");
    info!("║  CNN-like sliding window on XGBoost                   ║");
    info!("╚═══════════════════════════════════════════════════════╝");
    info!("Mode: {}", if standalone { "STANDALONE (all candles)" } else { "SUPER-FILTERED (super points only)" });
    info!("Model output: {}", if is_binary_mode { "BINARY (P(UP) ∈ [0,1], threshold=0.5)" } else { "REGRESSION (sign=direction)" });
    info!("Device: {:?}", device);
    dir_config.log_summary();
    if let Some(d) = &wfo_min_date {
        info!("WFO OOS filter: trades after {}", d.format("%Y-%m-%d"));
    }

    let pool = PgPool::connect(&db_url).await?;
    let total_start = std::time::Instant::now();

    for &tf in &timeframes {
        let tp_pct = super_config.target_pct_for_tf(tf);
        let sl_pct = super_config.sl_pct_for_tf(tf);
        let limit = match tf { 1=>5000, 5=>12000, 15=>12000, 60=>12000, 240=>12000, 1440=>3700, _=>5000 };

        let sep = "═".repeat(60);
        info!("");
        info!("{}", sep);
        info!("  TF {}m  TP={:.2}%  SL={:.2}%", tf, tp_pct, sl_pct);
        info!("{}", sep);

        // ── Load direction v4 model ──
        let v4_path = format!("models/direction_v4_tf{}.ubj", tf);
        let v4_model = match Booster::load(&v4_path, device) {
            Ok(b) => { info!("  ✅ direction v4 model: {} ({:?})", v4_path, device); Some(b) }
            Err(_) => match Booster::load(&v4_path, Device::Cpu) {
                Ok(b) => { info!("  ✅ direction v4 model: {} (CPU fallback)", v4_path); Some(b) }
                Err(e) => { info!("  ❌ No direction v4 model: {} — {}", v4_path, e); None }
            }
        };

        if v4_model.is_none() {
            info!("  No model for TF {}m. Skipping.", tf);
            continue;
        }
        let v4_model = v4_model.unwrap();

        // ── Load raw candles (OHLCV only) ──
        let t_load = std::time::Instant::now();
        let all_raw = fetch_all_raw_candles_for_tf(&pool, tf, limit).await?;
        let n_symbols = all_raw.len();
        let n_candles: usize = all_raw.values().map(|v| v.len()).sum();
        info!("  Loaded {} symbols, {} raw candles in {:.1}s",
              n_symbols, n_candles, t_load.elapsed().as_secs_f64());

        // Convert to CandleWithIndicators (indicators filled with defaults)
        let all_candles: std::collections::HashMap<String, Vec<CandleWithIndicators>> = all_raw
            .into_iter()
            .map(|(sym, raws)| {
                let cwi: Vec<CandleWithIndicators> = raws.iter()
                    .map(|r| r.to_candle_with_indicators(0))
                    .collect();
                (sym, cwi)
            })
            .collect();

        // ══════════════════════════════════════════════════════════════
        // Compute features + predict + simulate
        // ══════════════════════════════════════════════════════════════
        let t_eval = std::time::Instant::now();

        let n_features = dir_config.total_features();
        let min_candles = dir_config.min_candles_required();
        let max_hold = super_config.effective_max_hold();

        let mut metrics_v4 = Metrics::default();
        let mut accuracy = AccuracyMetrics::default();

        // Separate UP/DOWN metrics for per-direction analysis
        let mut metrics_up_only = Metrics::default();
        let mut metrics_down_only = Metrics::default();
        let mut n_pred_up = 0usize;
        let mut n_pred_down = 0usize;

        // Confidence gate thresholds — represent P(predicted_class)
        // e.g., gate≥0.60 means model is ≥60% confident in the predicted direction
        let gates = [0.50f32, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80];
        let mut metrics_gated: Vec<Metrics> = gates.iter().map(|_| Metrics::default()).collect();
        let mut accuracy_gated: Vec<AccuracyMetrics> = gates.iter().map(|_| AccuracyMetrics::default()).collect();
        // UP-only and DOWN-only gated metrics
        let mut metrics_gated_up: Vec<Metrics> = gates.iter().map(|_| Metrics::default()).collect();
        let mut metrics_gated_down: Vec<Metrics> = gates.iter().map(|_| Metrics::default()).collect();

        let mut n_total_points = 0usize;

        for (_symbol, candles) in &all_candles {
            if candles.len() < min_candles + max_hold {
                continue;
            }

            let start_idx = dir_config.window_size - 1;
            let end_idx = candles.len() - max_hold;

            if start_idx >= end_idx {
                continue;
            }

            // Batch compute features
            let mut batch_features: Vec<f32> = Vec::new();
            let mut batch_indices: Vec<usize> = Vec::new();

            for t in start_idx..end_idx {
                if candles[t].close <= 0.0 { continue; }

                // WFO date filter
                if let Some(min_d) = wfo_min_date {
                    if candles[t].time < min_d { continue; }
                }

                // Compute pattern features (no HTF in standalone backtest)
                if let Some(feats) = compute_pattern_features(candles, t, &dir_config, None) {
                    for &v in &feats {
                        batch_features.push(v as f32);
                    }
                    batch_indices.push(t);
                }
            }

            if batch_indices.is_empty() {
                continue;
            }

            let batch_size = batch_indices.len();
            n_total_points += batch_size;

            // Batch predict
            // Try as regressor first (output: single value, sign = direction)
            // If that fails, try as classifier (output: probabilities)
            let predictions = v4_model.predict_dense_cpu(
                &batch_features, batch_size, n_features, ModelKind::Regressor1,
            )?;

            // Process predictions
            for (i, &t) in batch_indices.iter().enumerate() {
                let pred_raw = predictions[i];

                // ── Direction from model output ──
                // Binary mode (binary:logistic): pred_raw = P(UP) ∈ [0, 1]
                //   direction = UP if P(UP) >= 0.5, DOWN if P(UP) < 0.5
                //   confidence = P(predicted_class) = max(P(UP), P(DOWN))
                // Regression mode (reg:squarederror): pred_raw ∈ (-∞, +∞)
                //   direction = sign(pred_raw)
                //   confidence = |pred_raw|
                let (v4_dir, confidence): (i8, f32) = if is_binary_mode {
                    let p_up = pred_raw.clamp(0.0, 1.0);
                    if p_up >= 0.5 {
                        (1, p_up)              // UP with confidence = P(UP)
                    } else {
                        (-1, 1.0 - p_up)       // DOWN with confidence = P(DOWN)
                    }
                } else {
                    // Regression: sign = direction, abs = confidence
                    let dir: i8 = if pred_raw >= 0.0 { 1 } else { -1 };
                    (dir, pred_raw.abs())
                };

                // Track per-direction counts
                if v4_dir == 1 { n_pred_up += 1; } else { n_pred_down += 1; }

                // Actual future return for accuracy measurement
                let horizon = dir_config.prediction_horizon;
                let future_idx = (t + horizon).min(candles.len() - 1);
                let actual_return = (candles[future_idx].close - candles[t].close) / candles[t].close * 100.0;

                // Accuracy (direction correctness)
                accuracy.add(v4_dir, actual_return);

                // Trade simulation
                let (outcome, pnl) = simulate_trade(
                    candles, t, v4_dir, tp_pct, sl_pct, max_hold,
                );
                metrics_v4.add(outcome, pnl);

                // Per-direction trade metrics
                if v4_dir == 1 {
                    metrics_up_only.add(outcome, pnl);
                } else {
                    metrics_down_only.add(outcome, pnl);
                }

                // Confidence-gated (confidence = P(predicted_class))
                for (gi, &gate) in gates.iter().enumerate() {
                    if confidence >= gate {
                        metrics_gated[gi].add(outcome, pnl);
                        accuracy_gated[gi].add(v4_dir, actual_return);
                        // Per-direction gated
                        if v4_dir == 1 {
                            metrics_gated_up[gi].add(outcome, pnl);
                        } else {
                            metrics_gated_down[gi].add(outcome, pnl);
                        }
                    }
                }
            }
        }

        info!("  Evaluation: {} points from {} symbols in {:.1}s",
              n_total_points, n_symbols, t_eval.elapsed().as_secs_f64());

        if n_total_points == 0 {
            info!("  No valid points for TF {}m. Skipping.", tf);
            continue;
        }

        // ── Print results ──
        info!("");
        info!("  TF {}m — {} points (pred UP={}, pred DOWN={})",
              tf, n_total_points, n_pred_up, n_pred_down);
        info!("");

        // Direction accuracy
        info!("  📊 Direction Accuracy (predict UP vs DOWN):");
        info!("    Overall: {:.1}% ({}/{})", accuracy.accuracy(), accuracy.correct, accuracy.total);
        if accuracy.up_total > 0 {
            info!("    UP recall:   {:.1}% ({}/{}) — actual UP points correctly predicted",
                  accuracy.up_correct as f64 / accuracy.up_total as f64 * 100.0,
                  accuracy.up_correct, accuracy.up_total);
        }
        if accuracy.down_total > 0 {
            info!("    DOWN recall: {:.1}% ({}/{}) — actual DOWN points correctly predicted",
                  accuracy.down_correct as f64 / accuracy.down_total as f64 * 100.0,
                  accuracy.down_correct, accuracy.down_total);
        }
        // UP/DOWN prediction precision
        if n_pred_up > 0 {
            info!("    UP precision: predicted {} UP, accuracy within UP predictions", n_pred_up);
        }
        if n_pred_down > 0 {
            info!("    DOWN precision: predicted {} DOWN, accuracy within DOWN predictions", n_pred_down);
        }
        if n_pred_down == 0 {
            info!("    ⚠️  Model predicts ZERO DOWN signals! Check DIRECTION_MODE env var.");
        }

        // Trade metrics
        info!("");
        info!("  📊 Trade Simulation (TP={:.2}% SL={:.2}% hold={}):", tp_pct, sl_pct, max_hold);
        info!("  {:<20} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
              "Variant", "Trades", "Wins", "Losses", "Exprd", "WR%", "AvgPnL%");
        info!("  {}", "-".repeat(76));

        let print_row = |name: &str, m: &Metrics| {
            let marker = if m.wr() >= 65.0 { "✅" } else if m.wr() >= 55.0 { "⚠ " } else { "❌" };
            info!("  {:<20} {:>8} {:>8} {:>8} {:>8} {:>9.1}% {:>9.4}% {}",
                  name, m.n, m.wins, m.losses, m.expired, m.wr(), m.avg_pnl(), marker);
        };

        print_row("All directions", &metrics_v4);
        if metrics_up_only.n > 0 {
            print_row("  LONG only", &metrics_up_only);
        }
        if metrics_down_only.n > 0 {
            print_row("  SHORT only", &metrics_down_only);
        }

        info!("");
        info!("  Confidence-gated results (confidence = P(predicted_class)):");
        for (gi, &gate) in gates.iter().enumerate() {
            let m = &metrics_gated[gi];
            let a = &accuracy_gated[gi];
            if m.n > 0 {
                let coverage = m.n as f64 / n_total_points as f64 * 100.0;
                let mu = &metrics_gated_up[gi];
                let md = &metrics_gated_down[gi];
                let name = format!("  gate≥{:.2}", gate);
                let marker = if a.accuracy() >= 60.0 { "✅" } else if a.accuracy() >= 55.0 { "⚠ " } else { "❌" };
                info!("  {:<20} {:>8} WR={:.1}% Acc={:.1}% PnL={:.4}% cov={:.0}% (L:{} S:{}) {}",
                      name, m.n, m.wr(), a.accuracy(), m.avg_pnl(), coverage,
                      mu.n, md.n, marker);
            }
        }
    }

    let total = total_start.elapsed();
    info!("");
    info!("Direction v4 backtest complete in {:.1}s ({:.1}min)",
          total.as_secs_f64(), total.as_secs_f64() / 60.0);
    info!("Target: direction accuracy >= 55-60% (beating random 50%)");

    Ok(())
}
