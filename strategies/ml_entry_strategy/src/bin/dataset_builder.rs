// strategies/ml_entry_strategy/src/bin/dataset_builder.rs
//
// Super Entry Dataset Builder
//
// Generates the training dataset for the Super Entry Model.
//
// USAGE:
//   cargo run --release -p ml_entry_strategy --bin super_entry_dataset
//
// OUTPUT:
//   super_entry_dataset.csv — features + labels for all (symbol, tf) pairs
//
// WORKFLOW:
//   1. Fetch candles per (symbol, tf) from DB (up to 12000 for key TFs)
//   2. Use first 300 as warmup (for indicator computation)
//   3. Label remaining candles with lookahead=20
//   4. Export to CSV for Python trainer

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::PgPool;
use tracing::info;

use ml_entry_strategy::config::SuperEntryConfig;
use ml_entry_strategy::dataset::{
    build_labels, fetch_candles_with_indicators, fetch_active_symbols,
    export_dataset_csv, all_feature_names, SuperEntryExample,
};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let output_path = std::env::var("SUPER_ENTRY_DATASET_OUTPUT")
        .unwrap_or_else(|_| "dataset/super_entry_dataset.csv".to_string());

    let config = SuperEntryConfig::from_env();

    info!("=== Super Entry Dataset Builder ===");
    info!("Warmup bars: {}", config.warmup_bars);
    info!("Lookahead: {}", config.lookahead_bars);
    info!("Output: {}", output_path);

    let pool = PgPool::connect(&db_url).await?;

    // Fetch active symbols
    let symbols = fetch_active_symbols(&pool).await?;
    info!("Found {} active symbols", symbols.len());

    if symbols.is_empty() {
        tracing::warn!("No active symbols found. Ensure market.pairs has data.");
        return Ok(());
    }

    // Use ALL timeframes for dataset building (including 1d for training)
    let timeframes = SuperEntryConfig::all_timeframes();
    let mut all_examples: Vec<SuperEntryExample> = Vec::new();

    let mut stats_by_tf: std::collections::HashMap<i32, (usize, usize)> = std::collections::HashMap::new();

    // Per-TF candle limits for dataset building — more candles = better model.
    // Must match what's available in DB (runtime.toml backfill_candles_per_tf).
    // After warmup (300) + lookahead (20), effective training examples = limit - 320.
    //   1m:  5000  → 4680 examples/symbol  (~3.5 days)
    //   5m:  12000 → 11680 examples/symbol (~41 days)
    //   15m: 12000 → 11680 examples/symbol (~125 days)
    //   1h:  12000 → 11680 examples/symbol (~500 days)
    //   4h:  12000 → 11680 examples/symbol (~5.5 years)
    //   1d:  3700  → 3380 examples/symbol  (~10 years)
    let dataset_limit_per_tf = |tf_minutes: i32| -> usize {
        match tf_minutes {
            1 => 5000,     // 1m: all available from DB
            5 => 12000,    // 5m: deep history
            15 => 12000,   // 15m: deep history
            60 => 12000,   // 1h: deep history (target TF)
            240 => 12000,  // 4h: deep history
            1440 => 3700,  // 1d: 10+ years
            _ => 5000,
        }
    };

    for &tf in timeframes {
        let target_pct = config.target_pct_for_tf(tf);
        let limit = dataset_limit_per_tf(tf);
        info!("Processing TF {}m (target_move={}%, limit={})", tf, target_pct, limit);

        let mut tf_total = 0;
        let mut tf_super = 0;

        for symbol in &symbols {
            let candles = fetch_candles_with_indicators(&pool, symbol, tf, limit).await?;

            if candles.len() < config.warmup_bars + config.lookahead_bars + 1 {
                continue;
            }

            let examples = build_labels(
                &candles,
                config.warmup_bars,
                config.lookahead_bars,
                target_pct,
                tf,
            );

            for ex in &examples {
                tf_total += 1;
                if ex.is_super {
                    tf_super += 1;
                }
            }

            all_examples.extend(examples);
        }

        let super_rate = if tf_total > 0 {
            tf_super as f64 / tf_total as f64 * 100.0
        } else {
            0.0
        };

        info!(
            "  TF {}m: {} examples, {} super ({:.1}%)",
            tf, tf_total, tf_super, super_rate
        );

        stats_by_tf.insert(tf, (tf_total, tf_super));
    }

    // Export
    let feature_names = all_feature_names();
    export_dataset_csv(&all_examples, &output_path, &feature_names)?;

    info!("\n=== Dataset Summary ===");
    info!("Total examples: {}", all_examples.len());
    info!("Output: {}", output_path);

    for &tf in timeframes {
        if let Some((total, super_count)) = stats_by_tf.get(&tf) {
            let rate = if *total > 0 {
                *super_count as f64 / *total as f64 * 100.0
            } else {
                0.0
            };
            info!(
                "  TF {}m: {} total, {} super ({:.1}%), target={:.2}%",
                tf, total, super_count, rate,
                config.target_pct_for_tf(tf)
            );
        }
    }

    info!("=== Done ===");
    Ok(())
}
