// strategies/ewmac_strategy/src/bin/service.rs
//
// EWMAC Service — Production signal generator
//
// Stages:
//   1. INIT:     Connect to DB, ensure tables, check existing data
//   2. BACKFILL: Generate signals for historical data (skip if already done)
//   3. REALTIME: Poll for new candles → EWMAC computation → write signals
//
// USAGE:
//   cargo run --release -p ewmac_strategy --bin ewmac_service

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::PgPool;
use std::time::{Duration, Instant};
use tracing::{info, warn, error};

use ewmac_strategy::config::EwmacConfig;
use ewmac_strategy::dataset::{fetch_candles, fetch_active_symbols};
use ewmac_strategy::db_writer::{insert_signals_batch, ensure_table_exists, count_signals};
use ewmac_strategy::pipeline::EwmacPipeline;
use ewmac_strategy::signal_generator::EwmacSignal;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let config = EwmacConfig::from_env();

    // ═══════════════════════════════════════════
    // Stage 1: INIT
    // ═══════════════════════════════════════════
    println!("\n=== [STAGE 1/3] INIT ===");
    info!(target: "ewmac", "Connecting to database...");

    let pool = PgPool::connect(&db_url).await?;
    info!(target: "ewmac", "✅ DB connected");

    ensure_table_exists(&pool).await?;
    info!(target: "ewmac", "✅ trade.ewmac_signals table ready");

    let initial_count = count_signals(&pool).await.unwrap_or(0);
    info!(target: "ewmac", "Existing signals in DB: {}", initial_count);

    // Wait for base pipeline to populate market.pairs
    info!(target: "ewmac", "Waiting for active symbols in market.pairs...");
    let symbols = loop {
        match fetch_active_symbols(&pool).await {
            Ok(s) if !s.is_empty() => break s,
            Ok(_) => {
                info!(target: "ewmac", "No active symbols yet. Waiting 10s...");
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
            Err(_) => {
                info!(target: "ewmac", "market.pairs not ready. Waiting 10s...");
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
        }
    };
    info!(target: "ewmac", "✅ Active symbols: {}", symbols.len());

    info!(target: "ewmac", "Config: min_forecast={}, warmup={}, max_hold={}, fdm={}, pairs={}",
        config.min_forecast, config.warmup_bars, config.max_hold_bars, config.fdm, config.pairs.len());

    for &tf in EwmacConfig::timeframes() {
        let (sl_m, tp_m) = config.atr_mults_for_tf(tf);
        info!(target: "ewmac", "  TF {:>5}m: SL=ATR*{:.1} TP=ATR*{:.1}", tf, sl_m, tp_m);
    }

    // Initialize pipeline (instant — no models)
    let pipeline = EwmacPipeline::new(config.clone());

    // ═══════════════════════════════════════════
    // Stage 2: BACKFILL
    // ═══════════════════════════════════════════
    let force_backfill = std::env::var("EWMAC_FORCE_BACKFILL")
        .unwrap_or_default() == "true";

    if initial_count == 0 || force_backfill {
        println!("\n=== [STAGE 2/3] HISTORY BACKFILL ===");
        info!(target: "ewmac", "Starting backfill ({})",
            if force_backfill { "forced" } else { "DB empty" });

        let t0 = Instant::now();
        let mut total_signals = 0usize;

        for &tf in EwmacConfig::timeframes() {
            let tf_t0 = Instant::now();
            let mut tf_signals: Vec<EwmacSignal> = Vec::new();
            let mut tf_symbols_ok = 0usize;

            // Concurrent fetch in batches of 20 symbols
            for chunk in symbols.chunks(20) {
                let mut handles = Vec::new();
                for symbol in chunk {
                    let pool_c = pool.clone();
                    let sym = symbol.clone();
                    handles.push(tokio::spawn(async move {
                        fetch_candles(&pool_c, &sym, tf, 1000).await
                    }));
                }

                for handle in handles {
                    match handle.await {
                        Ok(Ok(candles)) => {
                            if candles.len() < config.warmup_bars + 1 {
                                continue;
                            }
                            tf_symbols_ok += 1;

                            let results = match pipeline.process_candles(&candles, tf) {
                                Ok(r) => r,
                                Err(e) => {
                                    warn!(target: "ewmac",
                                        "  {} {}m: process failed: {}",
                                        candles.first().map(|c| c.symbol.as_str()).unwrap_or("?"), tf, e);
                                    continue;
                                }
                            };
                            for r in results {
                                if let Some(s) = r.signal { tf_signals.push(s); }
                            }
                        }
                        Ok(Err(e)) => {
                            warn!(target: "ewmac", "  Fetch failed: {}", e);
                        }
                        Err(e) => {
                            warn!(target: "ewmac", "  Task join failed: {:?}", e);
                        }
                    }
                }
            }

            if !tf_signals.is_empty() {
                let count = tf_signals.len();
                match insert_signals_batch(&pool, &tf_signals).await {
                    Ok(n) => {
                        info!(target: "ewmac",
                            "  TF {:>5}m: {} symbols → {} signals → {} written ({}ms)",
                            tf, tf_symbols_ok, count, n, tf_t0.elapsed().as_millis());
                        total_signals += count;
                    }
                    Err(e) => error!(target: "ewmac", "  TF {}m INSERT failed: {}", tf, e),
                }
            } else {
                info!(target: "ewmac", "  TF {:>5}m: {} symbols → 0 signals ({}ms)",
                    tf, tf_symbols_ok, tf_t0.elapsed().as_millis());
            }
        }

        let final_count = count_signals(&pool).await.unwrap_or(0);
        info!(target: "ewmac", "✅ Backfill complete: {} signals in {:.1}s. DB total: {}",
            total_signals, t0.elapsed().as_secs_f64(), final_count);
    } else {
        println!("\n=== [STAGE 2/3] BACKFILL SKIPPED ===");
        info!(target: "ewmac", "Skip: {} signals exist. EWMAC_FORCE_BACKFILL=true to force.", initial_count);
    }

    // ═══════════════════════════════════════════
    // Stage 3: REALTIME
    // ═══════════════════════════════════════════
    println!("\n=== [STAGE 3/3] REALTIME SIGNAL GENERATION ===");

    let poll_interval = Duration::from_secs(
        std::env::var("EWMAC_POLL_SECS")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(60)
    );
    let max_idle: u32 = std::env::var("EWMAC_MAX_IDLE_CYCLES")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0);

    info!(target: "ewmac", "Polling every {:?}, max_idle={} (0=forever)", poll_interval, max_idle);

    let mut idle_cycles = 0u32;
    let mut total_rt_signals = 0u64;
    let mut cycle_num = 0u64;

    loop {
        tokio::time::sleep(poll_interval).await;
        cycle_num += 1;

        let rt_t0 = Instant::now();
        let mut cycle_signals = 0usize;

        for &tf in EwmacConfig::timeframes() {
            let mut tf_signals: Vec<EwmacSignal> = Vec::new();

            // Fetch enough candles for EWMAC warmup + a small buffer
            for chunk in symbols.chunks(20) {
                let mut handles = Vec::new();
                for symbol in chunk {
                    let pool_c = pool.clone();
                    let sym = symbol.clone();
                    let limit = config.warmup_bars + 10;
                    handles.push(tokio::spawn(async move {
                        fetch_candles(&pool_c, &sym, tf, limit).await
                    }));
                }

                for handle in handles {
                    if let Ok(Ok(candles)) = handle.await {
                        if candles.len() < config.warmup_bars + 1 { continue; }

                        // Process all candles to build up EMA state, take only the last signal
                        let results = match pipeline.process_candles(&candles, tf) {
                            Ok(r) => r,
                            Err(_) => continue,
                        };

                        // Take only the last result (most recent candle)
                        if let Some(last) = results.last() {
                            if let Some(ref s) = last.signal {
                                tf_signals.push(s.clone());
                            }
                        }
                    }
                }
            }

            if !tf_signals.is_empty() {
                match insert_signals_batch(&pool, &tf_signals).await {
                    Ok(n) => cycle_signals += n,
                    Err(e) => warn!(target: "ewmac", "Insert tf{}m: {}", tf, e),
                }
            }
        }

        let rt_ms = rt_t0.elapsed().as_millis();
        total_rt_signals += cycle_signals as u64;

        if cycle_signals > 0 {
            idle_cycles = 0;
            info!(target: "ewmac",
                "[RT cycle #{}] {} signals written in {}ms (total RT: {})",
                cycle_num, cycle_signals, rt_ms, total_rt_signals);
        } else {
            idle_cycles += 1;
            if idle_cycles % 5 == 0 {
                info!(target: "ewmac", "[RT cycle #{}] idle {}/{} ({}ms)",
                    cycle_num, idle_cycles, max_idle, rt_ms);
            }
        }

        if max_idle > 0 && idle_cycles >= max_idle {
            info!(target: "ewmac", "Max idle reached ({}), shutting down", max_idle);
            break;
        }
    }

    info!(target: "ewmac", "Service stopped. Total RT signals: {}", total_rt_signals);
    Ok(())
}
