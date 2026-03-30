// strategies/ml_entry_strategy/src/bin/dataset_builder.rs
//
// Super Entry Dataset Builder — OPTIMIZED
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
//   1. Bulk-fetch candles+indicators per TF (single query per TF, not per symbol!)
//   2. Bulk-fetch HTF candles (single query per HTF)
//   3. Process labels in parallel (per symbol)
//   4. Export to CSV with BufWriter for Python trainer
//
// PERFORMANCE vs OLD VERSION:
//   OLD: ~2200 sequential DB queries with heavy JOINs (N+1 anti-pattern)
//   NEW: ~11 bulk queries total (1 per TF + 1 per HTF)
//   Expected speedup: 10-50x depending on symbol count

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::postgres::PgPoolOptions;
use std::collections::HashMap;
use tracing::info;

use ml_entry_strategy::config::SuperEntryConfig;
use ml_entry_strategy::dataset::{
    build_labels_with_htf, fetch_all_candles_for_tf, CandleWithIndicators,
    export_dataset_csv, all_feature_names, SuperEntryExample,
};
use ml_entry_strategy::heuristic::get_higher_tf;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // ── Setup logging ──
    let log_path = "logs/super_entry_dataset.log";
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
        .with(
            fmt::layer()
                .with_target(false)
                .with_thread_ids(false)
                .with_ansi(false)
                .with_writer(std::sync::Mutex::new(log_file)),
        )
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
        ))
        .init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let output_path = std::env::var("SUPER_ENTRY_DATASET_OUTPUT")
        .unwrap_or_else(|_| "dataset/super_entry_dataset.csv".to_string());

    let config = SuperEntryConfig::from_env();

    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║  Super Entry Dataset Builder (OPTIMIZED)               ║");
    info!("║  Bulk queries + parallel processing                    ║");
    info!("╚═══════════════════════════════════════════════════════╝");
    info!("Warmup bars: {}", config.warmup_bars);
    info!("Lookahead: {}", config.lookahead_bars);
    info!("Output: {}", output_path);
    info!("Log: {}", log_path);

    // ── DB pool with many connections for parallel queries ──
    let pool = PgPoolOptions::new()
        .max_connections(40)
        .connect(&db_url)
        .await?;
    info!("Connected to database (pool max_connections=40)");

    // Use ALL timeframes for dataset building
    let timeframes = SuperEntryConfig::all_timeframes();
    let mut all_examples: Vec<SuperEntryExample> = Vec::new();
    let mut stats_by_tf: HashMap<i32, (usize, usize)> = HashMap::new();

    // Per-TF candle limits
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

    let total_start = std::time::Instant::now();

    // ── Pre-load ALL HTF data (bulk, one query per HTF) ──
    // Collect unique HTFs needed
    let mut htf_set: Vec<i32> = Vec::new();
    for &tf in timeframes {
        if let Some(htf) = get_higher_tf(tf) {
            if !htf_set.contains(&htf) {
                htf_set.push(htf);
            }
        }
    }

    info!("Pre-loading HTF data for {:?}...", htf_set);
    let htf_load_start = std::time::Instant::now();

    let mut htf_cache: HashMap<i32, HashMap<String, Vec<CandleWithIndicators>>> = HashMap::new();
    for &htf_tf in &htf_set {
        let htf_limit = dataset_limit_per_tf(htf_tf);
        info!("  Loading HTF {}m (limit={})...", htf_tf, htf_limit);
        let htf_data = fetch_all_candles_for_tf(&pool, htf_tf, htf_limit).await?;
        info!("  HTF {}m: {} symbols loaded", htf_tf, htf_data.len());
        htf_cache.insert(htf_tf, htf_data);
    }
    info!(
        "HTF pre-load complete in {:.1}s",
        htf_load_start.elapsed().as_secs_f64()
    );

    // ── Process each TF ──
    for &tf in timeframes {
        let target_pct = config.target_pct_for_tf(tf);
        let limit = dataset_limit_per_tf(tf);
        let htf = get_higher_tf(tf);

        info!("━━━ Processing TF {}m (target_move={}%, limit={}) ━━━", tf, target_pct, limit);
        if let Some(h) = htf {
            info!("  HTF for {}m = {}m", tf, h);
        } else {
            info!("  No HTF for {}m (highest TF)", tf);
        }

        let tf_start = std::time::Instant::now();

        // ── OPTIMIZATION #1: Single bulk query per TF ──
        // Instead of ~200 individual queries (one per symbol), we do ONE query
        // with ROW_NUMBER() OVER (PARTITION BY symbol) that fetches ALL symbols.
        let grouped = fetch_all_candles_for_tf(&pool, tf, limit).await?;
        let n_symbols = grouped.len();

        let db_elapsed = tf_start.elapsed();
        info!(
            "  Fetched {} symbols in {:.1}s (single bulk query)",
            n_symbols,
            db_elapsed.as_secs_f64()
        );

        // ── Process each symbol's candles ──
        let mut tf_total = 0usize;
        let mut tf_super = 0usize;

        // Get pre-loaded HTF data for this TF's higher timeframe
        let htf_data_for_tf: Option<&HashMap<String, Vec<CandleWithIndicators>>> =
            htf.and_then(|h| htf_cache.get(&h));

        for (symbol, candles) in &grouped {
            if candles.len() < config.warmup_bars + config.lookahead_bars + 1 {
                continue;
            }

            // ── OPTIMIZATION #2: HTF from pre-loaded cache, not per-symbol query ──
            let htf_candles: Option<&[CandleWithIndicators]> = htf_data_for_tf
                .and_then(|htf_map| htf_map.get(symbol))
                .filter(|c| c.len() >= 50)
                .map(|c| c.as_slice());

            let examples = build_labels_with_htf(
                candles,
                config.warmup_bars,
                config.lookahead_bars,
                target_pct,
                config.sl_fraction,
                tf,
                htf_candles,
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

        let tf_elapsed = tf_start.elapsed();
        info!(
            "  TF {}m: {} examples, {} super ({:.1}%), elapsed: {:.1}s",
            tf, tf_total, tf_super, super_rate, tf_elapsed.as_secs_f64()
        );

        stats_by_tf.insert(tf, (tf_total, tf_super));
    }

    // ── Export with BufWriter ──
    std::fs::create_dir_all(
        std::path::Path::new(&output_path)
            .parent()
            .unwrap_or(std::path::Path::new(".")),
    )?;

    let export_start = std::time::Instant::now();
    let feature_names = all_feature_names();
    export_dataset_csv(&all_examples, &output_path, &feature_names)?;
    info!(
        "CSV export: {:.1}s",
        export_start.elapsed().as_secs_f64()
    );

    let total_elapsed = total_start.elapsed();

    // ── Summary ──
    info!("");
    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║  Super Entry Dataset Summary                          ║");
    info!("╚═══════════════════════════════════════════════════════╝");
    info!("Total examples: {}", all_examples.len());
    info!(
        "Output: {} ({:.1} MB)",
        output_path,
        std::fs::metadata(&output_path)
            .map(|m| m.len() as f64 / 1_048_576.0)
            .unwrap_or(0.0)
    );

    for &tf in timeframes {
        if let Some((total, super_count)) = stats_by_tf.get(&tf) {
            let rate = if *total > 0 {
                *super_count as f64 / *total as f64 * 100.0
            } else {
                0.0
            };
            info!(
                "  TF {:>5}m: {:>9} total, {:>7} super ({:.1}%), target={:.2}%",
                tf,
                total,
                super_count,
                rate,
                config.target_pct_for_tf(tf)
            );
        }
    }

    info!(
        "Total time: {:.1}s ({:.1}min)",
        total_elapsed.as_secs_f64(),
        total_elapsed.as_secs_f64() / 60.0
    );
    info!("");
    info!("Next step: train models with:");
    info!(
        "  ./scripts/super_entry_backtester.sh --train"
    );
    info!("Done ✅");

    Ok(())
}
