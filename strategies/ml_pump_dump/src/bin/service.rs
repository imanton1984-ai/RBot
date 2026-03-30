// strategies/ml_pump_dump/src/bin/service.rs
//
// Pump/Dump Service — Production signal generator
//
// Runs alongside ml_entry_strategy (super_entry) to detect pump/dump patterns.
//
// Stages:
//   1. INIT:     Connect to DB, ensure table, check existing data
//   2. MODELS:   Load XGBoost models (pump + dump)
//   3. BACKFILL: Generate signals for historical data (skip if already done)
//   4. REALTIME: Poll for new indicators → inference → write signals
//
// ENV VARS:
//   DATABASE_URL            — postgres connection
//   PD_DAILY_THRESHOLD      — min daily move % for detection (default: 15)
//   PD_MIN_PRED             — min prediction to generate signal (default: 0.65)
//   PD_TARGET_PCT           — target move % (default: 15.0)
//   PD_MAX_HOLD_BARS        — max bars to hold (default: 10)
//   PD_MIN_FINEST_TF        — min finest TF to accept (default: 1)
//   PD_MAX_FINEST_TF        — max finest TF to accept (default: 60)
//   PD_PUMP_MODEL_PATH      — pump model file (default: models/pump_dump_pump_v1.ubj)
//   PD_DUMP_MODEL_PATH      — dump model file (default: models/pump_dump_dump_v1.ubj)
//   PD_POLL_SECS            — realtime poll interval (default: 60)
//   PD_FORCE_BACKFILL       — force backfill even if signals exist (default: false)
//   DIRECTION_GPU           — use GPU for inference (default: false)
//
// Logs written to: logs/pump_dump_service.out

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::PgPool;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{info, warn, error};

use ml_pump_dump::pump_dump::{PumpDumpConfig, ANALYSIS_TIMEFRAMES};
use ml_pump_dump::dataset::fetch_active_symbols;
use ml_pump_dump::signal_generator::{PumpDumpSignalConfig, PumpDumpSignal};
use ml_pump_dump::db_writer::{ensure_table_exists, insert_signals_batch, count_signals};
use ml_pump_dump::pipeline::PumpDumpPipeline;

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

    let config = PumpDumpConfig::from_env();
    let signal_config = PumpDumpSignalConfig::from_env();
    let use_gpu = std::env::var("DIRECTION_GPU")
        .map_or(false, |v| v == "1" || v.to_lowercase() == "true");

    // ═══════════════════════════════════════════
    // Stage 1: INIT
    // ═══════════════════════════════════════════
    println!("\n=== [STAGE 1/4] INIT ===");
    info!(target: "pump_dump", "Connecting to database...");

    let pool = PgPool::connect(&db_url).await?;
    info!(target: "pump_dump", "✅ DB connected");

    ensure_table_exists(&pool).await?;
    info!(target: "pump_dump", "✅ trade.pump_dump_signals table ready");

    let initial_count = count_signals(&pool).await.unwrap_or(0);
    info!(target: "pump_dump", "Existing signals in DB: {}", initial_count);

    // Wait for base pipeline to populate market.pairs
    info!(target: "pump_dump", "Waiting for active symbols in market.pairs...");
    let symbols = loop {
        match fetch_active_symbols(&pool).await {
            Ok(s) if !s.is_empty() => break s,
            Ok(_) => {
                info!(target: "pump_dump", "No active symbols yet. Waiting 10s...");
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
            Err(_) => {
                info!(target: "pump_dump", "market.pairs not ready. Waiting 10s...");
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
        }
    };
    info!(target: "pump_dump", "✅ Active symbols: {}", symbols.len());

    config.log_summary();
    signal_config.log_summary();
    info!(target: "pump_dump", "GPU: {}", use_gpu);

    // ═══════════════════════════════════════════
    // Stage 2: MODELS
    // ═══════════════════════════════════════════
    println!("\n=== [STAGE 2/4] LOADING MODELS ===");
    let model_t0 = Instant::now();

    let pipeline = match PumpDumpPipeline::new(config.clone(), signal_config.clone(), use_gpu) {
        Ok(p) => p,
        Err(e) => {
            error!(target: "pump_dump", "❌ Failed to load models: {}", e);
            error!(target: "pump_dump", "Train first with pump_dump_dataset + train_pump_dump_wfo.py");
            return Err(e);
        }
    };

    if !pipeline.has_models() {
        error!(target: "pump_dump", "❌ No models found! Train first.");
        return Ok(());
    }

    info!(target: "pump_dump", "✅ Models loaded in {}ms", model_t0.elapsed().as_millis());

    // ═══════════════════════════════════════════
    // Stage 3: BACKFILL
    // ═══════════════════════════════════════════
    let force_backfill = std::env::var("PD_FORCE_BACKFILL")
        .unwrap_or_default() == "true";

    if initial_count == 0 || force_backfill {
        println!("\n=== [STAGE 3/4] HISTORY BACKFILL ===");
        info!(target: "pump_dump", "Starting backfill ({})",
            if force_backfill { "forced" } else { "DB empty" });

        let t0 = Instant::now();
        let mut total_signals = 0usize;
        let mut total_symbols_processed = 0usize;

        // Process symbols in batches (memory-safe)
        for (si, symbol) in symbols.iter().enumerate() {
            let all_tf_candles = match PumpDumpPipeline::load_symbol_data(
                &pool, symbol, config.pre_event_lookback
            ).await {
                Ok(d) => d,
                Err(e) => {
                    warn!(target: "pump_dump", "  {} fetch failed: {}", symbol, e);
                    continue;
                }
            };

            if all_tf_candles.is_empty() { continue; }
            total_symbols_processed += 1;

            // Resolve symbol_id from loaded data
            let symbol_id = all_tf_candles.values()
                .flat_map(|v| v.first())
                .next()
                .map(|c| c.symbol_id)
                .unwrap_or(0);

            let sigs = pipeline.process_symbol_candles(&all_tf_candles, symbol, symbol_id);

            if !sigs.is_empty() {
                let count = sigs.len();
                match insert_signals_batch(&pool, &sigs).await {
                    Ok(n) => {
                        info!(target: "pump_dump",
                            "  {} → {} signals ({} written)", symbol, count, n);
                        total_signals += count;
                    }
                    Err(e) => error!(target: "pump_dump", "  {} INSERT failed: {}", symbol, e),
                }
            }

            // Memory: all_tf_candles dropped here
            if (si + 1) % 50 == 0 {
                info!(target: "pump_dump", "  Progress: {}/{} symbols, {} signals",
                    si + 1, symbols.len(), total_signals);
            }
        }

        let final_count = count_signals(&pool).await.unwrap_or(0);
        info!(target: "pump_dump",
            "✅ Backfill complete: {} signals from {} symbols in {:.1}s. DB total: {}",
            total_signals, total_symbols_processed, t0.elapsed().as_secs_f64(), final_count);
    } else {
        println!("\n=== [STAGE 3/4] BACKFILL SKIPPED ===");
        info!(target: "pump_dump", "Skip: {} signals exist. PD_FORCE_BACKFILL=true to force.",
            initial_count);
    }

    // ═══════════════════════════════════════════
    // Stage 4: REALTIME
    // ═══════════════════════════════════════════
    println!("\n=== [STAGE 4/4] REALTIME SIGNAL GENERATION ===");

    let poll_interval = Duration::from_secs(
        std::env::var("PD_POLL_SECS")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(60)
    );
    let max_idle: u32 = std::env::var("PD_MAX_IDLE_CYCLES")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0);

    info!(target: "pump_dump", "Polling every {:?}, max_idle={} (0=forever)",
        poll_interval, max_idle);

    let mut idle_cycles = 0u32;
    let mut total_rt_signals = 0u64;
    let mut cycle_num = 0u64;

    // For realtime, use a smaller candle window (just enough for features)
    let rt_lookback = config.pre_event_lookback;
    let rt_limit = rt_lookback + 100; // extra margin for temporal features

    loop {
        tokio::time::sleep(poll_interval).await;
        cycle_num += 1;

        let rt_t0 = Instant::now();
        let mut cycle_signals: Vec<PumpDumpSignal> = Vec::new();

        // Process all symbols — concurrent batches of 10
        for chunk in symbols.chunks(10) {
            let mut handles = Vec::new();
            for symbol in chunk {
                let pool_c = pool.clone();
                let sym = symbol.clone();
                handles.push(tokio::spawn(async move {
                    let mut all_tf_candles: HashMap<i32, Vec<_>> = HashMap::new();
                    for &tf in ANALYSIS_TIMEFRAMES {
                        if let Ok(c) = ml_pump_dump::dataset::fetch_candles_with_indicators(
                            &pool_c, &sym, tf, rt_limit
                        ).await {
                            if c.len() >= rt_lookback + 10 {
                                all_tf_candles.insert(tf, c);
                            }
                        }
                    }
                    (sym, all_tf_candles)
                }));
            }

            for handle in handles {
                if let Ok((symbol, all_tf_candles)) = handle.await {
                    if all_tf_candles.is_empty() { continue; }

                    let symbol_id = all_tf_candles.values()
                        .flat_map(|v| v.first())
                        .next()
                        .map(|c| c.symbol_id)
                        .unwrap_or(0);

                    let sigs = pipeline.process_symbol_candles(
                        &all_tf_candles, &symbol, symbol_id
                    );
                    cycle_signals.extend(sigs);
                }
            }
        }

        // Write all signals from this cycle
        if !cycle_signals.is_empty() {
            match insert_signals_batch(&pool, &cycle_signals).await {
                Ok(_n) => {
                    let count = cycle_signals.len();
                    total_rt_signals += count as u64;
                    idle_cycles = 0;
                    info!(target: "pump_dump",
                        "[RT cycle #{}] {} signals written in {}ms (total RT: {})",
                        cycle_num, count, rt_t0.elapsed().as_millis(), total_rt_signals);
                }
                Err(e) => warn!(target: "pump_dump", "Insert failed: {}", e),
            }
        } else {
            idle_cycles += 1;
            if idle_cycles % 5 == 0 {
                info!(target: "pump_dump", "[RT cycle #{}] idle {}/{} ({}ms)",
                    cycle_num, idle_cycles, max_idle, rt_t0.elapsed().as_millis());
            }
        }

        if max_idle > 0 && idle_cycles >= max_idle {
            info!(target: "pump_dump", "Max idle reached ({}), shutting down", max_idle);
            break;
        }
    }

    info!(target: "pump_dump", "Service stopped. Total RT signals: {}", total_rt_signals);
    Ok(())
}
