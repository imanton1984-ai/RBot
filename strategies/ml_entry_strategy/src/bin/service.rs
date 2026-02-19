// strategies/ml_entry_strategy/src/bin/service.rs
//
// Super Entry Service — Production signal generator
//
// Stages:
//   1. INIT:     Connect to DB, ensure tables, check existing data
//   2. MODELS:   Load XGBoost models (10 boosters: 5 TF × 2 models)
//   3. BACKFILL: Generate signals for historical data (skip if already done)
//   4. REALTIME: Poll for new indicators → inference → write signals
//
// Logs written to: logs/super_entry_service.out

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::PgPool;
use std::time::{Duration, Instant};
use tracing::{info, warn, error};

use ml_entry_strategy::config::SuperEntryConfig;
use ml_entry_strategy::dataset::{fetch_candles_with_indicators, fetch_active_symbols};
use ml_entry_strategy::db_writer::{insert_signals_batch, ensure_table_exists, count_signals};
use ml_entry_strategy::pipeline::SuperEntryPipeline;
use ml_entry_strategy::signal_generator::SuperEntrySignal;

fn _stage(name: &str) {
    let bar = "═".repeat(60);
    println!("╔{}╗", bar);
    println!("║  {:<56}  ║", name);
    println!("╚{}╝", bar);
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let config = SuperEntryConfig::from_env();
    let use_gpu = std::env::var("SUPER_ENTRY_USE_GPU")
        .unwrap_or_default()
        .parse::<bool>()
        .unwrap_or(false);

    // ═══════════════════════════════════════════
    // Stage 1: INIT
    // ═══════════════════════════════════════════
    println!("\n=== [STAGE 1/4] INIT ===");
    info!(target: "super_entry", "Connecting to database...");
    
    let pool = PgPool::connect(&db_url).await?;
    info!(target: "super_entry", "✅ DB connected");

    ensure_table_exists(&pool).await?;
    info!(target: "super_entry", "✅ trade.super_entry_signals table ready");

    let initial_count = count_signals(&pool).await.unwrap_or(0);
    info!(target: "super_entry", "Existing signals in DB: {}", initial_count);

    // Wait for base pipeline to populate market.pairs (like compute_history does)
    info!(target: "super_entry", "Waiting for active symbols in market.pairs...");
    let symbols = loop {
        match fetch_active_symbols(&pool).await {
            Ok(s) if !s.is_empty() => break s,
            Ok(_) => {
                info!(target: "super_entry", "No active symbols yet. Waiting 10s for base pipeline...");
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
            Err(_) => {
                info!(target: "super_entry", "market.pairs not ready. Waiting 10s for DB init...");
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
        }
    };
    info!(target: "super_entry", "✅ Active symbols: {}", symbols.len());

    info!(target: "super_entry", "Config: p_threshold={}, warmup={}, lookahead={}, sl_fraction={}, gpu={}",
        config.p_threshold, config.warmup_bars, config.lookahead_bars, config.sl_fraction, use_gpu);

    for &tf in SuperEntryConfig::timeframes() {
        info!(target: "super_entry", "  TF {:>5}m: TP={:.2}% SL={:.2}%",
            tf, config.target_pct_for_tf(tf), config.sl_pct_for_tf(tf));
    }

    // ═══════════════════════════════════════════
    // Stage 2: MODELS
    // ═══════════════════════════════════════════
    println!("\n=== [STAGE 2/4] LOADING MODELS ===");
    let model_t0 = Instant::now();

    let pipeline = match SuperEntryPipeline::new(config.clone(), use_gpu) {
        Ok(p) => p,
        Err(e) => {
            error!(target: "super_entry", "❌ Failed to load models: {}", e);
            error!(target: "super_entry", "Train first: ./scripts/super_entry.sh --train");
            return Err(e);
        }
    };

    if !pipeline.has_models() {
        error!(target: "super_entry", "❌ No models found!");
        return Ok(());
    }

    info!(target: "super_entry", "✅ Models loaded in {}ms", model_t0.elapsed().as_millis());

    // ═══════════════════════════════════════════
    // Stage 3: BACKFILL
    // ═══════════════════════════════════════════
    let force_backfill = std::env::var("SUPER_ENTRY_FORCE_BACKFILL")
        .unwrap_or_default() == "true";

    if initial_count == 0 || force_backfill {
        println!("\n=== [STAGE 3/4] HISTORY BACKFILL ===");
        info!(target: "super_entry", "Starting backfill ({})",
            if force_backfill { "forced" } else { "DB empty" });

        let t0 = Instant::now();
        let mut total_signals = 0usize;

        for &tf in SuperEntryConfig::timeframes() {
            let tf_t0 = Instant::now();
            let mut tf_signals: Vec<SuperEntrySignal> = Vec::new();
            let mut tf_candles = 0usize;
            let mut tf_symbols_ok = 0usize;

            // Concurrent fetch in batches of 20 symbols
            for chunk in symbols.chunks(20) {
                let mut handles = Vec::new();
                for symbol in chunk {
                    let pool_c = pool.clone();
                    let sym = symbol.clone();
                    handles.push(tokio::spawn(async move {
                        ml_entry_strategy::dataset::fetch_candles_with_indicators(
                            &pool_c, &sym, tf, 1000
                        ).await
                    }));
                }

                for handle in handles {
                    if let Ok(Ok(candles)) = handle.await {
                        if candles.len() < config.warmup_bars + 1 { continue; }
                        tf_symbols_ok += 1;
                        let results = match pipeline.process_candles(&candles, tf, use_gpu) {
                            Ok(r) => r,
                            Err(_) => continue,
                        };
                        tf_candles += results.len();
                        for r in results {
                            if let Some(s) = r.signal { tf_signals.push(s); }
                        }
                    }
                }
            }

            if !tf_signals.is_empty() {
                let count = tf_signals.len();
                match insert_signals_batch(&pool, &tf_signals).await {
                    Ok(n) => {
                        info!(target: "super_entry", 
                            "  TF {:>5}m: {} symbols → {} candles → {} signals → {} written ({}ms)",
                            tf, tf_symbols_ok, tf_candles, count, n, tf_t0.elapsed().as_millis());
                        total_signals += count;
                    }
                    Err(e) => error!(target: "super_entry", "  TF {}m INSERT failed: {}", tf, e),
                }
            } else {
                info!(target: "super_entry", "  TF {:>5}m: {} symbols → 0 signals ({}ms)",
                    tf, tf_symbols_ok, tf_t0.elapsed().as_millis());
            }
        }

        let final_count = count_signals(&pool).await.unwrap_or(0);
        info!(target: "super_entry", "✅ Backfill complete: {} signals in {:.1}s. DB total: {}",
            total_signals, t0.elapsed().as_secs_f64(), final_count);
    } else {
        println!("\n=== [STAGE 3/4] BACKFILL SKIPPED ===");
        info!(target: "super_entry", "Skip: {} signals exist. SUPER_ENTRY_FORCE_BACKFILL=true to force.", initial_count);
    }

    // ═══════════════════════════════════════════
    // Stage 4: REALTIME
    // ═══════════════════════════════════════════
    println!("\n=== [STAGE 4/4] REALTIME SIGNAL GENERATION ===");

    let poll_interval = Duration::from_secs(
        std::env::var("SUPER_ENTRY_POLL_SECS")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(60)
    );
    let max_idle: u32 = std::env::var("SUPER_ENTRY_MAX_IDLE_CYCLES")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0);

    info!(target: "super_entry", "Polling every {:?}, max_idle={} (0=forever)", poll_interval, max_idle);

    let mut idle_cycles = 0u32;
    let mut total_rt_signals = 0u64;
    let mut cycle_num = 0u64;

    loop {
        tokio::time::sleep(poll_interval).await;
        cycle_num += 1;

        let rt_t0 = Instant::now();
        let mut cycle_signals = 0usize;

        for &tf in SuperEntryConfig::timeframes() {
            let mut tf_signals: Vec<SuperEntrySignal> = Vec::new();

            // Concurrent fetch for last candle (warmup+10)
            for chunk in symbols.chunks(20) {
                let mut handles = Vec::new();
                for symbol in chunk {
                    let pool_c = pool.clone();
                    let sym = symbol.clone();
                    let limit = config.warmup_bars + 10;
                    handles.push(tokio::spawn(async move {
                        ml_entry_strategy::dataset::fetch_candles_with_indicators(
                            &pool_c, &sym, tf, limit
                        ).await
                    }));
                }

                for handle in handles {
                    if let Ok(Ok(candles)) = handle.await {
                        if candles.len() < config.warmup_bars + 1 { continue; }
                        let last = candles.last().unwrap();
                        if let Ok(r) = pipeline.process_single(last, tf, use_gpu) {
                            if let Some(s) = r.signal { tf_signals.push(s); }
                        }
                    }
                }
            }

            if !tf_signals.is_empty() {
                match insert_signals_batch(&pool, &tf_signals).await {
                    Ok(n) => cycle_signals += n,
                    Err(e) => warn!(target: "super_entry", "Insert tf{}m: {}", tf, e),
                }
            }
        }

        let rt_ms = rt_t0.elapsed().as_millis();
        total_rt_signals += cycle_signals as u64;

        if cycle_signals > 0 {
            idle_cycles = 0;
            info!(target: "super_entry", 
                "[RT cycle #{}] {} signals written in {}ms (total RT: {})",
                cycle_num, cycle_signals, rt_ms, total_rt_signals);
        } else {
            idle_cycles += 1;
            if idle_cycles % 5 == 0 {
                info!(target: "super_entry", "[RT cycle #{}] idle {}/{} ({}ms)", 
                    cycle_num, idle_cycles, max_idle, rt_ms);
            }
        }

        if max_idle > 0 && idle_cycles >= max_idle {
            info!(target: "super_entry", "Max idle reached ({}), shutting down", max_idle);
            break;
        }
    }

    info!(target: "super_entry", "Service stopped. Total RT signals: {}", total_rt_signals);
    Ok(())
}
