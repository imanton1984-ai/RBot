// strategies/ml_impulse_strategy/src/bin/dataset_builder.rs
//
// Impulse Absorption & Engulfing Dataset Builder CLI
//
// Scans all active symbols for engulfing patterns on target TFs (15m, 1h, 4h),
// labels each by simulating SL/TP forward, extracts multi-TF indicator features,
// and exports a CSV dataset for the Python WFO trainer.
//
// USAGE:
//   cargo build --release -p ml_impulse_strategy --bin impulse_dataset
//   ./target/release/impulse_dataset
//
// ENV VARS:
//   DATABASE_URL            — postgres connection (default: postgres://postgres:postgres@localhost:5433/timescaledb_binance)
//   IAE_MIN_BODY_RATIO      — min body-to-candle ratio (default: 0.7)
//   IAE_VOLUME_SPIKE_MULT   — volume spike multiplier (default: 2.0)
//   IAE_PRE_SIGNAL_LOOKBACK — candles before signal (default: 10)
//   IAE_MAX_HOLD            — max candles to hold (default: 3)
//   IAE_NEGATIVE_RATIO      — neg:pos ratio (default: 3)
//   IAE_TF15_IMPULSE_PCT    — 15m impulse threshold (default: 4.5)
//   IAE_TF60_IMPULSE_PCT    — 1h impulse threshold (default: 7.0)
//   IAE_TF240_IMPULSE_PCT   — 4h impulse threshold (default: 10.0)
//   IAE_TF15_TP_PCT / IAE_TF15_SL_PCT — per-TF TP/SL overrides
//   IAE_OUTPUT_DIR          — output dir (default: dataset)
//   IAE_CONCURRENCY         — parallel symbol processing (default: 4)

use anyhow::Result;
use dotenvy::dotenv;
use tracing::info;

use ml_impulse_strategy::impulse::{ImpulseConfig, ANALYSIS_TIMEFRAMES, FULL_FEATURES_PER_CANDLE, ENGULFING_FEATURE_COUNT, all_feature_names};
use ml_impulse_strategy::dataset::{build_impulse_dataset, export_dataset_csv};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // Logging
    let log_path = "logs/impulse_dataset.log";
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

    let output_dir = std::env::var("IAE_OUTPUT_DIR")
        .unwrap_or_else(|_| "dataset".to_string());

    let config = ImpulseConfig::from_env();

    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║  Impulse Absorption & Engulfing — Dataset Builder     ║");
    info!("║  Multi-TF indicator snapshot + engulfing detection    ║");
    info!("╚═══════════════════════════════════════════════════════╝");

    let pool = sqlx::PgPool::connect(&db_url).await?;
    let t_start = std::time::Instant::now();

    // Build dataset
    let examples = build_impulse_dataset(&pool, &config).await?;

    if examples.is_empty() {
        info!("No engulfing signals found. Try lowering impulse thresholds.");
        return Ok(());
    }

    // Generate feature names
    let feature_names = all_feature_names(ANALYSIS_TIMEFRAMES, config.pre_signal_lookback);
    let n_tf_features = ANALYSIS_TIMEFRAMES.len() * config.pre_signal_lookback * FULL_FEATURES_PER_CANDLE;
    info!("  Feature vector size: {} ({} TF features + {} engulfing meta)",
          feature_names.len(), n_tf_features, ENGULFING_FEATURE_COUNT);

    // Export
    std::fs::create_dir_all(&output_dir)?;
    let csv_path = format!("{}/impulse_dataset.csv", output_dir);
    export_dataset_csv(&examples, &csv_path, &feature_names)?;

    let elapsed = t_start.elapsed();
    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║  Dataset build complete                               ║");
    info!("║  Total: {} examples                                  ", examples.len());
    info!("║  Output: {}                                          ", csv_path);
    info!("║  Time: {:.1}s                                        ", elapsed.as_secs_f64());
    info!("╚═══════════════════════════════════════════════════════╝");
    info!("");
    info!("Next step: train the model");
    info!("  python trainer/src/train_impulse_wfo.py --csv {} --gpu", csv_path);

    Ok(())
}
