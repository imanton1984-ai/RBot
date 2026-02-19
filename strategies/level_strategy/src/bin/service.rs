// strategies/level_strategy/src/bin/service.rs
//
// Level Strategy Service — Production signal generator
//
// Stages:
//   1. INIT:     Connect to DB, ensure tables, check existing data
//   2. MODELS:   Load XGBoost models (price + levels predictors)
//   3. BACKFILL: Generate signals for historical data
//   4. REALTIME: Poll for new indicators → predictions → write signals
//
// Logs written to: logs/level_strategy_service.out

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::{PgPool, Row};
use std::time::{Duration, Instant};
use tracing::{info, error};

use level_strategy::config::LevelStrategyConfig;
use level_strategy::pipeline::LevelStrategyPipeline;

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

    let config = LevelStrategyConfig::from_env();

    // ═══════════════════════════════════════════
    // Stage 1: INIT
    // ═══════════════════════════════════════════
    println!("\n=== [STAGE 1/4] INIT ===");
    info!(target: "level_strategy", "Connecting to database...");

    let pool = PgPool::connect(&db_url).await?;
    info!(target: "level_strategy", "✅ DB connected");

    // Wait for base pipeline to populate market.pairs
    info!(target: "level_strategy", "Waiting for active symbols in market.pairs...");
    let symbols = loop {
        match fetch_active_symbols(&pool).await {
            Ok(s) if !s.is_empty() => break s,
            Ok(_) => {
                info!(target: "level_strategy", "No active symbols yet. Waiting 10s for base pipeline...");
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
            Err(_) => {
                info!(target: "level_strategy", "market.pairs not ready. Waiting 10s for DB init...");
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
        }
    };
    info!(target: "level_strategy", "✅ Active symbols: {}", symbols.len());

    info!(target: "level_strategy", "Config: horizon={}, min_score={}, min_final={}, gpu={}",
        config.horizon_bars, config.min_store_score, config.min_final_score, config.use_gpu_history);

    // ═══════════════════════════════════════════
    // Stage 2: MODELS
    // ═══════════════════════════════════════════
    println!("\n=== [STAGE 2/4] LOADING MODELS ===");
    let model_t0 = Instant::now();

    let pipeline = LevelStrategyPipeline::new(config.clone());

    if !pipeline.has_models() {
        error!(target: "level_strategy", "❌ No models found!");
        error!(target: "level_strategy", "Train models first via trainer/teacher.sh");
        return Ok(());
    }

    info!(target: "level_strategy", "✅ Models loaded in {}ms", model_t0.elapsed().as_millis());

    // ═══════════════════════════════════════════
    // Stage 3: BACKFILL
    // ═══════════════════════════════════════════
    let force_backfill = std::env::var("LEVEL_FORCE_BACKFILL")
        .unwrap_or_default() == "true";

    if force_backfill {
        println!("\n=== [STAGE 3/4] HISTORY BACKFILL ===");
        info!(target: "level_strategy", "Starting backfill (forced)");

        let t0 = Instant::now();
        match pipeline.run_history(&pool).await {
            Ok(count) => {
                info!(target: "level_strategy", "✅ Backfill complete: {} signals in {:.1}s",
                    count, t0.elapsed().as_secs_f64());
            }
            Err(e) => {
                error!(target: "level_strategy", "Backfill error: {}", e);
            }
        }
    } else {
        println!("\n=== [STAGE 3/4] BACKFILL SKIPPED ===");
        info!(target: "level_strategy", "Skip: use LEVEL_FORCE_BACKFILL=true to force");
    }

    // ═══════════════════════════════════════════
    // Stage 4: REALTIME
    // ═══════════════════════════════════════════
    println!("\n=== [STAGE 4/4] REALTIME SIGNAL GENERATION ===");

    let poll_interval = Duration::from_secs(
        std::env::var("LEVEL_POLL_SECS")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(60)
    );
    let max_idle: u32 = std::env::var("LEVEL_MAX_IDLE_CYCLES")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0);

    info!(target: "level_strategy", "Polling every {:?}, max_idle={} (0=forever)", poll_interval, max_idle);

    let mut idle_cycles = 0u32;
    let mut total_rt_signals = 0u64;
    let mut cycle_num = 0u64;

    loop {
        tokio::time::sleep(poll_interval).await;
        cycle_num += 1;

        let rt_t0 = Instant::now();

        match pipeline.run_realtime(&pool).await {
            Ok(cycle_signals) => {
                total_rt_signals += cycle_signals as u64;

                if cycle_signals > 0 {
                    idle_cycles = 0;
                    info!(target: "level_strategy",
                        "[RT cycle #{}] {} signals written in {}ms (total RT: {})",
                        cycle_num, cycle_signals, rt_t0.elapsed().as_millis(), total_rt_signals);
                } else {
                    idle_cycles += 1;
                    if idle_cycles % 5 == 0 {
                        info!(target: "level_strategy", "[RT cycle #{}] idle {}/{} ({}ms)",
                            cycle_num, idle_cycles, max_idle, rt_t0.elapsed().as_millis());
                    }
                }
            }
            Err(e) => {
                error!(target: "level_strategy", "Realtime error: {}", e);
                idle_cycles += 1;
            }
        }

        if max_idle > 0 && idle_cycles >= max_idle {
            info!(target: "level_strategy", "Max idle reached ({}), shutting down", max_idle);
            break;
        }
    }

    info!(target: "level_strategy", "Service stopped. Total RT signals: {}", total_rt_signals);
    Ok(())
}

/// Fetch active symbols from database
async fn fetch_active_symbols(pool: &PgPool) -> Result<Vec<String>> {
    let rows = sqlx::query("SELECT symbol FROM market.pairs WHERE is_active = true")
        .fetch_all(pool)
        .await?;

    Ok(rows.into_iter().map(|r| r.get::<String, _>("symbol")).collect())
}
