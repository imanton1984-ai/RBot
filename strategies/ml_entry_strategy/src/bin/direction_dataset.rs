// strategies/ml_entry_strategy/src/bin/direction_dataset.rs
//
// Direction v4 Pattern Dataset Builder
//
// Generates the training dataset for Direction Model v4 (CNN-like pattern recognition).
// Uses only raw OHLCV candles — NO indicators needed → very fast data loading.
//
// USAGE:
//   cargo build --release -p ml_entry_strategy --bin direction_dataset
//   ./target/release/direction_dataset
//
// ENV VARS:
//   DATABASE_URL                  — postgres connection string
//   DIRECTION_DATASET_OUTPUT      — CSV output path (default: dataset/direction_v4_dataset.csv)
//   DIRECTION_TF                  — comma-separated TFs (default: "5,15,60,240,1440")
//   DIR_WINDOW_SIZE               — sliding window size (default: 30)
//   DIR_PREDICTION_HORIZON        — prediction horizon in bars (default: 10)
//   DIR_FEATURE_SET               — feature set: ohlc/ohlcv/ohlcvb/full (default: ohlcv)
//   DIR_UP_THRESHOLD              — up threshold % (default: 0.3)
//   DIR_DOWN_THRESHOLD            — down threshold % (default: 0.3)
//   DIR_LABEL_METHOD              — final_return or max_excursion (default: final_return)
//   DIR_EXCLUDE_FLAT              — exclude FLAT from CSV (default: false)
//
// OUTPUT:
//   dataset/direction_v4_dataset.csv — pattern features + labels for all (symbol, tf) pairs

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::postgres::PgPoolOptions;
use tracing::info;

use ml_entry_strategy::direction::DirectionConfig;
use ml_entry_strategy::direction::dataset::{
    build_direction_dataset_for_tf, export_direction_csv, DirectionPatternExample,
};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // ── Setup logging ──
    let log_path = "logs/direction_dataset.log";
    std::fs::create_dir_all("logs")?;
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    use tracing_subscriber::fmt;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false).with_thread_ids(false))
        .with(fmt::layer().with_target(false).with_thread_ids(false)
            .with_ansi(false).with_writer(std::sync::Mutex::new(log_file)))
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
        ))
        .init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let output_path = std::env::var("DIRECTION_DATASET_OUTPUT")
        .unwrap_or_else(|_| "dataset/direction_v4_dataset.csv".to_string());

    let tf_str = std::env::var("DIRECTION_TF")
        .unwrap_or_else(|_| "5,15,60,240,1440".to_string());

    let timeframes: Vec<i32> = tf_str
        .split(',')
        .filter_map(|s| s.trim().parse::<i32>().ok())
        .collect();

    // Load direction config from env
    let config = DirectionConfig::from_env();

    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║  Direction v4 Pattern Dataset Builder                 ║");
    info!("║  CNN-like sliding window — pure OHLCV, no indicators  ║");
    info!("╚═══════════════════════════════════════════════════════╝");
    info!("Timeframes: {:?}", timeframes);
    config.log_summary();
    info!("Output: {}", output_path);
    info!("Log: {}", log_path);

    // Pool
    let pool = PgPoolOptions::new()
        .max_connections(40)
        .connect(&db_url)
        .await?;
    info!("Connected to database (pool max_connections=40)");

    // Per-TF candle limits
    let limit_for_tf = |tf: i32| -> usize {
        match tf {
            1 => 5000,
            5 => 12000,
            15 => 12000,
            60 => 12000,
            240 => 12000,
            1440 => 3700,
            _ => 5000,
        }
    };

    let mut all_examples: Vec<DirectionPatternExample> = Vec::new();
    let mut stats: Vec<(i32, usize, usize, usize, usize)> = Vec::new(); // (tf, total, up, flat, down)

    let total_start = std::time::Instant::now();

    for &tf in &timeframes {
        let limit = limit_for_tf(tf);
        // Use TF-specific thresholds from P(super) targets
        let tf_config = config.for_tf(tf);
        info!("━━━ Processing TF {}m (limit={}, threshold={:.1}%) ━━━", tf, limit, tf_config.up_threshold_pct);

        let tf_start = std::time::Instant::now();

        let examples = build_direction_dataset_for_tf(&pool, tf, &tf_config, limit).await?;

        let n_up = examples.iter().filter(|e| e.label == 1).count();
        let n_flat = examples.iter().filter(|e| e.label == 0).count();
        let n_down = examples.iter().filter(|e| e.label == -1).count();

        let elapsed = tf_start.elapsed();

        info!("  TF {}m results:", tf);
        info!("    Total examples: {}", examples.len());
        info!("    UP: {} ({:.1}%)", n_up, pct(n_up, examples.len()));
        info!("    FLAT: {} ({:.1}%)", n_flat, pct(n_flat, examples.len()));
        info!("    DOWN: {} ({:.1}%)", n_down, pct(n_down, examples.len()));
        info!("    Elapsed: {:.1}s", elapsed.as_secs_f64());

        stats.push((tf, examples.len(), n_up, n_flat, n_down));
        all_examples.extend(examples);
    }

    // ── Export CSV ──
    std::fs::create_dir_all(
        std::path::Path::new(&output_path)
            .parent()
            .unwrap_or(std::path::Path::new(".")),
    )?;
    export_direction_csv(&all_examples, &output_path, &config)?;

    let total_elapsed = total_start.elapsed();

    // ── Summary ──
    info!("");
    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║  Direction v4 Pattern Dataset Summary                 ║");
    info!("╚═══════════════════════════════════════════════════════╝");
    info!("Total examples: {}", all_examples.len());
    info!("Features per row: {} (window={} × {}={:?})",
          config.total_features(),
          config.window_size,
          config.feature_set.features_per_candle(),
          config.feature_set);
    info!("Output: {} ({:.1} MB)",
          output_path,
          std::fs::metadata(&output_path).map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0));

    for (tf, total, n_up, n_flat, n_down) in &stats {
        info!("  TF {:>5}m: {:>9} total, UP={:>7} ({:.1}%), FLAT={:>7} ({:.1}%), DOWN={:>7} ({:.1}%)",
              tf, total,
              n_up, pct(*n_up, *total),
              n_flat, pct(*n_flat, *total),
              n_down, pct(*n_down, *total));
    }

    info!("Total time: {:.1}s ({:.1}min)", total_elapsed.as_secs_f64(), total_elapsed.as_secs_f64() / 60.0);
    info!("");
    info!("Next step: train models with:");
    info!("  python trainer/src/train_direction_wfo.py --csv {} --gpu", output_path);
    info!("Done ✅");

    Ok(())
}

fn pct(num: usize, total: usize) -> f64 {
    if total > 0 { num as f64 / total as f64 * 100.0 } else { 0.0 }
}
