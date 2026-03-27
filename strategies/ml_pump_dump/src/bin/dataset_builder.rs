// strategies/ml_pump_dump/src/bin/dataset_builder.rs
//
// Pump/Dump Dataset Builder CLI
//
// Scans all active symbols for pump/dump events on the daily chart,
// drills down to lower TFs, extracts multi-TF indicator features,
// and exports a CSV dataset for the Python WFO trainer.
//
// USAGE:
//   cargo build --release -p ml_pump_dump --bin pump_dump_dataset
//   ./target/release/pump_dump_dataset
//
// ENV VARS:
//   DATABASE_URL         — postgres connection (default: postgres://postgres:postgres@localhost:5433/timescaledb_binance)
//   PD_DAILY_THRESHOLD   — min daily move % (default: 15)
//   PD_PRE_EVENT_LOOKBACK — candles before event (default: 10)
//   PD_NEGATIVE_RATIO    — neg:pos ratio (default: 3)
//   PD_OUTPUT_DIR        — output dir (default: dataset)
//   PD_CONCURRENCY       — parallel symbol processing (default: 8)

use anyhow::Result;
use dotenvy::dotenv;
use tracing::info;

use ml_pump_dump::pump_dump::{PumpDumpConfig, ANALYSIS_TIMEFRAMES, all_feature_names};
use ml_pump_dump::dataset::{build_pump_dump_dataset, export_dataset_csv};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // Logging
    let log_path = "logs/pump_dump_dataset.log";
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

    let output_dir = std::env::var("PD_OUTPUT_DIR")
        .unwrap_or_else(|_| "dataset".to_string());

    let config = PumpDumpConfig::from_env();

    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║  Pump/Dump Dataset Builder                            ║");
    info!("║  Multi-TF indicator snapshot + anomalous move detect  ║");
    info!("╚═══════════════════════════════════════════════════════╝");

    let pool = sqlx::PgPool::connect(&db_url).await?;
    let t_start = std::time::Instant::now();

    // Build dataset
    let examples = build_pump_dump_dataset(&pool, &config).await?;

    if examples.is_empty() {
        info!("No pump/dump events found. Try lowering PD_DAILY_THRESHOLD.");
        return Ok(());
    }

    // Generate feature names
    let feature_names = all_feature_names(ANALYSIS_TIMEFRAMES, config.pre_event_lookback);
    info!("  Feature vector size: {} ({} TFs × {} candles × {} features/candle)",
          feature_names.len(),
          ANALYSIS_TIMEFRAMES.len(),
          config.pre_event_lookback,
          ml_pump_dump::pump_dump::FULL_FEATURES_PER_CANDLE);

    // Export
    std::fs::create_dir_all(&output_dir)?;
    let csv_path = format!("{}/pump_dump_dataset.csv", output_dir);
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
    info!("  python trainer/src/train_pump_dump_wfo.py --csv {} --gpu", csv_path);

    Ok(())
}
