// strategies/ml_entry_strategy/src/bin/direction_dataset.rs
//
// Direction v3 Dataset Builder
//
// Generates the training dataset for Direction Model v3.
// Computes only 32 curated features (vs 128 for super_entry) → ~4x faster.
//
// USAGE:
//   cargo build --release -p ml_entry_strategy --bin direction_dataset
//   ./target/release/direction_dataset
//
// ENV VARS:
//   DATABASE_URL — postgres connection string (default: localhost:5433)
//   DIRECTION_DATASET_OUTPUT — CSV output path (default: dataset/direction_v3_dataset.csv)
//   DIRECTION_TF — comma-separated TFs to process (default: "5,15,60,240,1440")
//
// OUTPUT:
//   dataset/direction_v3_dataset.csv — 32 features + labels for all (symbol, tf) pairs
//   logs/direction_dataset.log — structured log output
//
// IMPORTANT: This binary is SAFE to run while the bot is live.
//   It only performs SELECT queries and writes to separate output files.
//   Does NOT modify any tables, models, or running services.

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::postgres::PgPoolOptions;
use tracing::info;

use ml_entry_strategy::config::SuperEntryConfig;
use ml_entry_strategy::direction::dataset::{
    build_direction_dataset_for_tf, export_direction_csv, DirectionExample,
};
use ml_entry_strategy::direction::features::DIRECTION_V3_FEATURES;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // ── Setup logging to both stdout and file ──
    let log_path = "logs/direction_dataset.log";
    std::fs::create_dir_all("logs")?;
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    // Use tracing-subscriber with both stdout and file output
    use tracing_subscriber::fmt;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let stdout_layer = fmt::layer()
        .with_target(false)
        .with_thread_ids(false);

    let file_layer = fmt::layer()
        .with_target(false)
        .with_thread_ids(false)
        .with_ansi(false)
        .with_writer(std::sync::Mutex::new(log_file));

    tracing_subscriber::registry()
        .with(stdout_layer)
        .with(file_layer)
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
        ))
        .init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let output_path = std::env::var("DIRECTION_DATASET_OUTPUT")
        .unwrap_or_else(|_| "dataset/direction_v3_dataset.csv".to_string());

    let tf_str = std::env::var("DIRECTION_TF")
        .unwrap_or_else(|_| "5,15,60,240,1440".to_string());

    let timeframes: Vec<i32> = tf_str
        .split(',')
        .filter_map(|s| s.trim().parse::<i32>().ok())
        .collect();

    let config = SuperEntryConfig::from_env();

    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║  Direction v3 Dataset Builder                        ║");
    info!("║  32 curated features — 6 domains, no duplicates      ║");
    info!("╚═══════════════════════════════════════════════════════╝");
    info!("Timeframes: {:?}", timeframes);
    info!("Warmup bars: {}", config.warmup_bars);
    info!("Lookahead: {} bars", config.lookahead_bars);
    info!("SL fraction: {:.2}", config.sl_fraction);
    info!("Features: {} (direction v3)", DIRECTION_V3_FEATURES.len());
    info!("Output: {}", output_path);
    info!("Log: {}", log_path);

    // Pool with enough connections for concurrent symbol fetching (default concurrency=16,
    // each symbol needs up to 2 queries in flight → 32 connections minimum)
    let pool = PgPoolOptions::new()
        .max_connections(40)
        .connect(&db_url)
        .await?;
    info!("Connected to database (pool max_connections=40)");

    // Per-TF candle limits (same as super_entry dataset builder)
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

    let mut all_examples: Vec<DirectionExample> = Vec::new();
    let mut stats: Vec<(i32, usize, usize, usize)> = Vec::new(); // (tf, total, super, whipsaw)

    let total_start = std::time::Instant::now();

    for &tf in &timeframes {
        let limit = limit_for_tf(tf);
        info!("━━━ Processing TF {}m (limit={}) ━━━", tf, limit);

        let tf_start = std::time::Instant::now();

        let examples = build_direction_dataset_for_tf(&pool, tf, &config, limit).await?;

        let n_super = examples.iter().filter(|e| e.is_super).count();
        let n_whipsaw = examples.iter().filter(|e| e.direction == 0).count();
        let n_with_tp = examples.iter().filter(|e| e.bars_to_tp.is_some()).count();

        // Direction balance
        let n_long = examples.iter().filter(|e| e.direction == 1).count();
        let n_short = examples.iter().filter(|e| e.direction == -1).count();

        // Direction quality stats
        let super_with_tp: Vec<f64> = examples.iter()
            .filter(|e| e.is_super && e.bars_to_tp.is_some() && e.direction != 0)
            .map(|e| e.direction_quality)
            .collect();
        let mean_quality = if !super_with_tp.is_empty() {
            super_with_tp.iter().sum::<f64>() / super_with_tp.len() as f64
        } else {
            0.0
        };

        let elapsed = tf_start.elapsed();

        info!("  TF {}m results:", tf);
        info!("    Total examples: {}", examples.len());
        info!("    Super: {} ({:.1}%)",
              n_super,
              if examples.is_empty() { 0.0 } else { n_super as f64 / examples.len() as f64 * 100.0 });
        info!("    Whipsaw (dir=0): {} ({:.1}%)",
              n_whipsaw,
              if examples.is_empty() { 0.0 } else { n_whipsaw as f64 / examples.len() as f64 * 100.0 });
        info!("    With TP hit: {} ({:.1}%)",
              n_with_tp,
              if examples.is_empty() { 0.0 } else { n_with_tp as f64 / examples.len() as f64 * 100.0 });
        info!("    Direction balance: LONG={} SHORT={} (ratio={:.2})",
              n_long, n_short,
              if n_short > 0 { n_long as f64 / n_short as f64 } else { 999.0 });
        info!("    Mean direction_quality (super+TP): {:.4}", mean_quality);
        info!("    Elapsed: {:.1}s", elapsed.as_secs_f64());

        stats.push((tf, examples.len(), n_super, n_whipsaw));
        all_examples.extend(examples);
    }

    // ── Export CSV ──
    std::fs::create_dir_all(std::path::Path::new(&output_path).parent().unwrap_or(std::path::Path::new(".")))?;
    export_direction_csv(&all_examples, &output_path)?;

    let total_elapsed = total_start.elapsed();

    // ── Summary ──
    info!("");
    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║  Direction v3 Dataset Summary                        ║");
    info!("╚═══════════════════════════════════════════════════════╝");
    info!("Total examples: {}", all_examples.len());
    info!("Features: {} (v3 — optimized)", DIRECTION_V3_FEATURES.len());
    info!("Output: {} ({:.1} MB)",
          output_path,
          std::fs::metadata(&output_path).map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0));

    for (tf, total, n_super, n_whipsaw) in &stats {
        let super_rate = if *total > 0 { *n_super as f64 / *total as f64 * 100.0 } else { 0.0 };
        info!("  TF {:>5}m: {:>9} total, {:>7} super ({:.1}%), {:>5} whipsaw, target={:.1}%",
              tf, total, n_super, super_rate, n_whipsaw,
              config.target_pct_for_tf(*tf));
    }

    info!("Total time: {:.1}s ({:.1}min)", total_elapsed.as_secs_f64(), total_elapsed.as_secs_f64() / 60.0);
    info!("");
    info!("Next step: train models with:");
    info!("  python trainer/src/train_direction_wfo.py --csv {} --gpu", output_path);
    info!("Done ✅");

    Ok(())
}
