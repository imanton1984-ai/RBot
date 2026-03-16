// strategies/super_level_strategy/src/bin/dataset_builder.rs
//
// Super Level Dataset Builder
//
// Generates training dataset for 5 Super Level models.
//
// USAGE:
//   cargo run --release -p super_level_strategy --bin super_level_dataset
//
// OUTPUT:
//   dataset/super_level_dataset.csv

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::PgPool;
use tracing::info;

use super_level_strategy::config::SuperLevelConfig;
use super_level_strategy::dataset::{
    build_labels, fetch_candles_with_indicators, fetch_active_symbols,
    export_dataset_csv, SuperLevelExample,
};
use super_level_strategy::config::all_feature_names;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let output_path = std::env::var("SUPER_LEVEL_DATASET_OUTPUT")
        .unwrap_or_else(|_| "dataset/super_level_dataset.csv".to_string());

    let config = SuperLevelConfig::from_env();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         SUPER LEVEL DATASET BUILDER                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Warmup bars:      {}", config.warmup_bars);
    println!("  Lookahead bars:   {}", config.lookahead_bars);
    println!("  Formation bars:   {}", config.level_params.formation_bars);
    println!("  Sensitivity %:    {}", config.level_params.sensitivity_pct);
    println!("  Touch zone %:     {}", config.level_params.touch_zone_pct);
    println!("  Strong touches:   ≥{}", config.level_params.strong_touches);
    println!("  Medium touches:   ≥{}", config.level_params.medium_touches);
    println!("  Output:           {}", output_path);
    println!();

    let pool = PgPool::connect(&db_url).await?;

    let symbols = fetch_active_symbols(&pool).await?;
    info!("Found {} active symbols", symbols.len());

    if symbols.is_empty() {
        tracing::warn!("No active symbols found.");
        return Ok(());
    }

    // Ensure output directory exists
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let timeframes = SuperLevelConfig::all_timeframes();
    let mut all_examples: Vec<SuperLevelExample> = Vec::new();
    let mut stats_by_tf: std::collections::HashMap<i32, (usize, usize, usize, usize)> =
        std::collections::HashMap::new();

    let dataset_limit_per_tf = |tf_minutes: i32| -> usize {
        match tf_minutes {
            1 => 5000,
            5 => 12000,
            15 => 12000,
            60 => 12000,
            240 => 12000,
            1440 => 3700,
            _ => 5000,
        }
    };

    for &tf in timeframes {
        let target_pct = config.target_pct_for_tf(tf);
        let limit = dataset_limit_per_tf(tf);
        println!("  Processing TF {}m (target={:.2}%, limit={})...", tf, target_pct, limit);

        let mut tf_total = 0usize;
        let mut tf_wins = 0usize;
        let mut tf_near_level = 0usize;
        let mut tf_bounces = 0usize;

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
                config.sl_fraction,
                tf,
                &config.level_params,
            );

            for ex in &examples {
                tf_total += 1;
                if ex.is_win { tf_wins += 1; }
                if ex.nearest_level_dist_atr < 1.0 { tf_near_level += 1; }
                if ex.is_bounce { tf_bounces += 1; }
            }

            all_examples.extend(examples);
        }

        let win_rate = if tf_total > 0 { tf_wins as f64 / tf_total as f64 * 100.0 } else { 0.0 };
        let near_pct = if tf_total > 0 { tf_near_level as f64 / tf_total as f64 * 100.0 } else { 0.0 };
        let bounce_pct = if tf_total > 0 { tf_bounces as f64 / tf_total as f64 * 100.0 } else { 0.0 };

        println!(
            "    TF {}m: {} examples, {} wins ({:.1}%), {} near level ({:.1}%), {} bounces ({:.1}%)",
            tf, tf_total, tf_wins, win_rate, tf_near_level, near_pct, tf_bounces, bounce_pct,
        );

        stats_by_tf.insert(tf, (tf_total, tf_wins, tf_near_level, tf_bounces));
    }

    // Export
    let feature_names = all_feature_names();
    export_dataset_csv(&all_examples, &output_path, &feature_names)?;

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  Total examples:  {}", all_examples.len());
    println!("  Features:        {}", feature_names.len());
    println!("  Output:          {}", output_path);
    println!();
    println!("  Per-TF summary:");
    for &tf in timeframes {
        if let Some(&(total, wins, near, bounces)) = stats_by_tf.get(&tf) {
            println!(
                "    TF {:>5}m: {} total, {:.1}% win, {:.1}% near_level, {:.1}% bounce",
                tf, total,
                if total > 0 { wins as f64 / total as f64 * 100.0 } else { 0.0 },
                if total > 0 { near as f64 / total as f64 * 100.0 } else { 0.0 },
                if total > 0 { bounces as f64 / total as f64 * 100.0 } else { 0.0 },
            );
        }
    }
    println!("═══════════════════════════════════════════════════════");

    Ok(())
}
