// strategies/ml_pump_dump/src/bin/backtest.rs
//
// Pump/Dump Backtest CLI
//
// Tests the trained pump/dump prediction model:
//   1. Load daily candles → detect sharp pump/dump events
//   2. For confirmed events, extract multi-TF features from BEFORE the event
//   3. Use the trained model to predict: would it have caught this?
//   4. Measure precision, recall, accuracy
//
// USAGE:
//   cargo build --release -p ml_pump_dump --bin pump_dump_backtest
//   ./target/release/pump_dump_backtest
//   PD_DAILY_THRESHOLD=10 ./target/release/pump_dump_backtest
//
// ENV VARS:
//   DATABASE_URL           — postgres connection
//   PD_DAILY_THRESHOLD     — min daily move % (default: 15)
//   PD_MODEL_PATH          — model file (default: models/pump_dump_v1.ubj)
//   PD_MODEL_TYPE          — "pump", "dump", or "both" (default: "both")
//   DIRECTION_GPU=1        — use GPU for inference

use anyhow::Result;
use chrono::{DateTime, Duration, NaiveDate, Utc};
use dotenvy::dotenv;
use sqlx::PgPool;
use std::collections::HashMap;
use tracing::info;

use ml_pump_dump::pump_dump::{
    CandleInd, EventType, PumpDumpConfig,
    ANALYSIS_TIMEFRAMES, FULL_FEATURES_PER_CANDLE,
    detect_anomalous_candles, validate_sharp_move,
};
use ml_pump_dump::dataset::{
    fetch_candles_with_indicators, fetch_active_symbols,
    drill_down_and_extract,
};

use predictors::ml::xgb_runtime::{Booster, Device, ModelKind};

// ─────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
#[allow(dead_code)]
struct BacktestMetrics {
    total_events: usize,
    true_positives: usize,
    false_negatives: usize,
    false_positives: usize,
    true_negatives: usize,
}

#[allow(dead_code)]
impl BacktestMetrics {
    fn add_positive(&mut self, predicted: bool) {
        self.total_events += 1;
        if predicted {
            self.true_positives += 1;
        } else {
            self.false_negatives += 1;
        }
    }

    fn add_negative(&mut self, predicted: bool) {
        if predicted {
            self.false_positives += 1;
        } else {
            self.true_negatives += 1;
        }
    }

    fn precision(&self) -> f64 {
        let denom = self.true_positives + self.false_positives;
        if denom > 0 { self.true_positives as f64 / denom as f64 } else { 0.0 }
    }

    fn recall(&self) -> f64 {
        let denom = self.true_positives + self.false_negatives;
        if denom > 0 { self.true_positives as f64 / denom as f64 } else { 0.0 }
    }

    fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 }
    }

    fn accuracy(&self) -> f64 {
        let total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives;
        if total > 0 { (self.true_positives + self.true_negatives) as f64 / total as f64 } else { 0.0 }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // Logging
    let log_path = "logs/pump_dump_backtest.log";
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

    let config = PumpDumpConfig::from_env();

    let use_gpu = std::env::var("DIRECTION_GPU")
        .map_or(false, |v| v == "1" || v.to_lowercase() == "true");
    let device = if use_gpu { Device::Cuda } else { Device::Cpu };

    let model_type = std::env::var("PD_MODEL_TYPE")
        .unwrap_or_else(|_| "both".to_string());

    let wfo_min_date: Option<DateTime<Utc>> = std::env::var("WFO_MIN_DATE").ok()
        .and_then(|s| NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()
            .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc()));

    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║  Pump/Dump Backtester                                 ║");
    info!("║  Sharp move detection + multi-TF indicator patterns   ║");
    info!("╚═══════════════════════════════════════════════════════╝");
    config.log_summary();
    info!("Device: {:?}", device);
    info!("Model type: {}", model_type);
    if let Some(d) = &wfo_min_date {
        info!("WFO OOS filter: events after {}", d.format("%Y-%m-%d"));
    }

    // Load model(s)
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
        info!("No models loaded. Running in DETECTION-ONLY mode (no predictions).");
    }

    let pool = PgPool::connect(&db_url).await?;
    let total_start = std::time::Instant::now();

    // TF limits for per-symbol data loading
    let tf_limits: HashMap<i32, usize> = vec![
        (1440, 3700), (240, 12000), (60, 12000),
        (15, 12000), (5, 12000), (1, 5000),
    ].into_iter().collect();

    let n_features = ANALYSIS_TIMEFRAMES.len() * config.pre_event_lookback * FULL_FEATURES_PER_CANDLE;
    info!("  Feature vector: {} features", n_features);

    let symbols = fetch_active_symbols(&pool).await?;
    info!("  {} active symbols", symbols.len());

    let mut pump_metrics = BacktestMetrics::default();
    let mut dump_metrics = BacktestMetrics::default();
    let mut total_daily_candidates = 0usize;
    let mut total_sharp_events = 0usize;
    let mut total_rejected = 0usize;

    // Process symbols one at a time (memory safe)
    for (si, symbol) in symbols.iter().enumerate() {
        // Load daily + hourly (required)
        let daily_candles = match fetch_candles_with_indicators(&pool, symbol, 1440, 3700).await {
            Ok(c) if c.len() >= config.pre_event_lookback + 10 => c,
            _ => continue,
        };
        let hourly_candles = match fetch_candles_with_indicators(&pool, symbol, 60, 12000).await {
            Ok(c) if c.len() >= config.pre_event_lookback + 10 => c,
            _ => continue,
        };

        // Detect daily events
        let daily_events = detect_anomalous_candles(&daily_candles, config.daily_threshold_pct);
        total_daily_candidates += daily_events.len();

        if daily_events.is_empty() { continue; }

        // Validate sharpness
        let mut validated: Vec<(usize, EventType, f64)> = Vec::new();
        for &(idx, event_type, move_pct) in &daily_events {
            let daily_time = daily_candles[idx].time;
            let daily_end = daily_time + Duration::days(1);

            // WFO date filter
            if let Some(min_d) = wfo_min_date {
                if daily_time < min_d { continue; }
            }

            match validate_sharp_move(
                &hourly_candles, daily_time, daily_end,
                event_type, move_pct, config.concentration_pct,
            ) {
                Some(_) => {
                    validated.push((idx, event_type, move_pct));
                    total_sharp_events += 1;
                }
                None => {
                    total_rejected += 1;
                }
            }
        }

        if validated.is_empty() { continue; }

        // Load remaining TFs for feature extraction
        let mut all_tf_candles: HashMap<i32, Vec<CandleInd>> = HashMap::new();
        all_tf_candles.insert(1440, daily_candles);
        all_tf_candles.insert(60, hourly_candles);

        for &tf in &[240i32, 15, 5, 1] {
            let limit = tf_limits.get(&tf).copied().unwrap_or(5000);
            if let Ok(c) = fetch_candles_with_indicators(&pool, symbol, tf, limit).await {
                if c.len() >= config.pre_event_lookback + 10 {
                    all_tf_candles.insert(tf, c);
                }
            }
        }

        // For each validated event, extract features and predict
        for &(daily_idx, event_type, move_pct) in &validated {
            let daily_time = all_tf_candles.get(&1440).unwrap()[daily_idx].time;

            let result = drill_down_and_extract(
                &all_tf_candles, daily_time, event_type, &config,
            );

            if let Some((features, finest_tf, onset_time)) = result {
                let feats_f32: Vec<f32> = features.iter().map(|&v| v as f32).collect();

                // Predict with the appropriate model
                let model = match event_type {
                    EventType::Pump => pump_model.as_ref(),
                    EventType::Dump => dump_model.as_ref(),
                };

                if let Some(m) = model {
                    let pred = m.predict_dense_cpu(&feats_f32, 1, n_features, ModelKind::Regressor1)?;
                    let prob = pred[0].clamp(0.0, 1.0);
                    let predicted = prob >= 0.5;

                    match event_type {
                        EventType::Pump => pump_metrics.add_positive(predicted),
                        EventType::Dump => dump_metrics.add_positive(predicted),
                    }

                    info!("  {} {} {} {:.1}% finest={}m pred={:.3} {}",
                          symbol, onset_time.format("%Y-%m-%d %H:%M"),
                          event_type, move_pct, finest_tf, prob,
                          if predicted { "✅" } else { "❌" });
                } else {
                    info!("  {} {} {} {:.1}% finest={}m (no model)",
                          symbol, onset_time.format("%Y-%m-%d %H:%M"),
                          event_type, move_pct, finest_tf);
                }
            }
        }

        // Memory: all_tf_candles dropped here
        drop(all_tf_candles);

        if (si + 1) % 20 == 0 {
            info!("  Progress: {}/{} symbols", si + 1, symbols.len());
        }
    }

    // Results
    let elapsed = total_start.elapsed();
    info!("");
    info!("╔═══════════════════════════════════════════════════════════╗");
    info!("║  PUMP/DUMP BACKTEST RESULTS                               ║");
    info!("╚═══════════════════════════════════════════════════════════╝");
    info!("  Daily event candidates: {}", total_daily_candidates);
    info!("  Sharp events (passed validation): {}", total_sharp_events);
    info!("  Rejected (gradual drift): {}", total_rejected);
    info!("");

    if pump_model.is_some() {
        info!("  📊 PUMP Model:");
        info!("    Events: {}", pump_metrics.total_events);
        info!("    TP={} FN={} FP={} TN={}",
              pump_metrics.true_positives, pump_metrics.false_negatives,
              pump_metrics.false_positives, pump_metrics.true_negatives);
        info!("    Precision: {:.1}%", pump_metrics.precision() * 100.0);
        info!("    Recall:    {:.1}%", pump_metrics.recall() * 100.0);
        info!("    F1:        {:.3}", pump_metrics.f1());
    }

    if dump_model.is_some() {
        info!("  📊 DUMP Model:");
        info!("    Events: {}", dump_metrics.total_events);
        info!("    TP={} FN={} FP={} TN={}",
              dump_metrics.true_positives, dump_metrics.false_negatives,
              dump_metrics.false_positives, dump_metrics.true_negatives);
        info!("    Precision: {:.1}%", dump_metrics.precision() * 100.0);
        info!("    Recall:    {:.1}%", dump_metrics.recall() * 100.0);
        info!("    F1:        {:.3}", dump_metrics.f1());
    }

    info!("");
    info!("  Total time: {:.1}s ({:.1}min)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);

    Ok(())
}
