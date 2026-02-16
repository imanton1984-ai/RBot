use std::sync::Arc;
use anyhow::Result;
use common::{Symbol, Timeframe, MessageBus};
use compute_lib::{*, IndicatorPersistor, RawSignalPersistor, RawSignalProcessor, ResultProcessor};
use database_lib;
use dotenvy::dotenv;
use sqlx::{PgPool, Row};
use std::time::Duration;
use raw_signals::thresholds::SignalConfig;
use compute_lib::predictors::config::PredictorsConfig;
use compute_lib::predictors::pipeline::{PredictorsPipeline, FeatureSnapshot};
use compute_lib::scoring::trade_signal_processor::{TradeSignalStage, TradeSignalInput};
use compute_lib::scoring::market_params_calculator::MarketParamsCalculator;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());
    
    database_lib::init_db::initialize_database(&db_url)
        .await
        .map_err(|e| anyhow::anyhow!("Database init failed: {}", e))?;

    // Backend (CUDA preferred for history)
    let backend_type = if cfg!(feature = "cuda") {
        ComputeBackendType::Cuda
    } else {
        tracing::warn!("CUDA feature not enabled, falling back to CPU for history");
        ComputeBackendType::Cpu
    };
    
    let compute_backend_manager = ComputeBackendManager::new(backend_type);
    let compute_backend = compute_backend_manager.get_backend();

    // Config
    let config = ComputeConfig {
        batch_size: 2000,
        max_concurrent_jobs: 2,
        use_cuda: cfg!(feature = "cuda"),
        cuda_device_id: Some(0),
    };

    let db_pool = sqlx::PgPool::connect(&db_url).await?;
    let candle_fetcher = Arc::new(CandleWindowFetcher::new(db_pool.clone()));

    // --- PIPELINE SETUP START ---

    // 1. Initialize DB Persistor (HISTORY Mode)
    std::env::set_var("DB_PERSIST_MODE", "history");
    // Optimization for bulk loading (reduced batch sizes for faster commits)
    std::env::set_var("DB_PERSIST_CHUNK_SIZE_HISTORY", "20000");
    std::env::set_var("DB_PERSIST_FLUSH_MS", "500");
    std::env::set_var("DB_PERSIST_HISTORY_SKIP_JSON", "1"); 
    std::env::set_var("DB_PERSIST_HISTORY_UPSERT", "0"); 

    let bulk_persistor = database_lib::bulk_persistor::BulkPersistor::new_from_env_mode(
        database_lib::bulk_persistor::PersistMode::History
    ).await?;
    let bulk_sender = bulk_persistor.sender();

    // 2. Initialize Indicator & RawSignal Persistors
    let (indicator_persistor, _ind_tx) = IndicatorPersistor::new(bulk_sender.clone());
    let raw_signal_persistor = RawSignalPersistor::new(bulk_sender.clone());
    
    // 3. Raw Signal Processor
    let raw_cfg = SignalConfig {
        enable_filtering: true,
        min_interesting_score: 0.60,
        ..Default::default()
    };
    let raw_processor = Arc::new(RawSignalProcessor::new(raw_cfg));

    // Scheduler
    let (job_scheduler, result_receiver) = JobScheduler::new(
        compute_backend,
        config.clone(),
        candle_fetcher.clone(),
    );
    let job_scheduler = Arc::new(job_scheduler);

    // 4. Predictors Pipeline (XGBoost)
    let message_bus = MessageBus::new_from_env()?;
    let (_shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel::<bool>(1);
    
    // min_final_score: configurable via env var.
    // Default 0.55 is achievable with heuristic-only predictors.
    let min_final_score: f64 = std::env::var("MIN_FINAL_SCORE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.55);

    tracing::info!("Using min_final_score = {}", min_final_score);

    let pred_config = PredictorsConfig {
        enabled: true,
        horizon_bars: 10,
        min_store_score: 0.50,
        min_final_score,
        prefer_ml: true,
        max_levels_per_side: 2,
        use_cuda: config.use_cuda,
        use_gpu_history: config.use_cuda,
        use_gpu_realtime: false,
        model_path_price: "models/price_v1_tf{tf}.ubj".to_string(),
        model_path_levels: "models/levels_v1_tf{tf}.ubj".to_string(),
        ml_batch_size: 4096,
    };

    let (feature_tx, feature_rx) = tokio::sync::mpsc::unbounded_channel::<FeatureSnapshot>();
    
    let mut predictors_pipeline = PredictorsPipeline::new(
        pred_config.clone(),
        db_pool.clone(),
        message_bus,
        shutdown_rx.resubscribe(),
    );
    predictors_pipeline.set_input_receiver(feature_rx);
    predictors_pipeline.set_bulk_sender(bulk_sender.clone());

    // --- TradeSignalStage setup ---
    let (trade_signal_tx, trade_signal_rx) = tokio::sync::mpsc::unbounded_channel::<TradeSignalInput>();
    predictors_pipeline.set_trade_signal_sender(trade_signal_tx);

    let market_params_calc = MarketParamsCalculator::new(common::Symbol::from("BTCUSDT"));
    let trade_signal_stage = TradeSignalStage::new(
        db_pool.clone(),
        market_params_calc,
        bulk_sender.clone(),
        trade_signal_rx,
        pred_config.min_final_score,
    );

    tokio::spawn(async move {
        if let Err(e) = trade_signal_stage.run().await {
            tracing::error!(target: "trade_signal_stage", "TradeSignalStage error: {}", e);
        }
    });

    // Spawn Predictors Pipeline
    tokio::spawn(async move {
        if let Err(e) = predictors_pipeline.run().await {
            tracing::error!(target: "compute_predictors", "Predictors pipeline error: {}", e);
        }
    });

    // 5. Spawn Result Processor
    let result_processor = ResultProcessor::new(
        Arc::new(indicator_persistor),
        Arc::new(raw_signal_persistor),
        raw_processor,
        feature_tx,
    );
    
    tokio::spawn(async move {
        result_processor.run(result_receiver).await;
    });

    // --- PIPELINE SETUP END ---

    // =====================================================================
    // INCREMENTAL GAP-AWARE PROCESSING LOOP
    // Instead of processing all history once and exiting,
    // we poll for uncomputed candle gaps and process only what's missing.
    // This handles the race condition where ingestor is still backfilling
    // candles via REST while we're already running.
    // =====================================================================

    // Config for the polling loop
    let max_idle_cycles: u32 = std::env::var("HISTORY_MAX_IDLE_CYCLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10); // Exit after N consecutive cycles with no new work

    let poll_interval_secs: u64 = std::env::var("HISTORY_POLL_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30); // Check for new gaps every 30s

    let initial_wait_secs: u64 = std::env::var("HISTORY_INITIAL_WAIT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15); // Wait for ingestor to populate initial data

    // 1) Wait for active pairs to appear
    let symbols = wait_for_active_symbols(&db_pool).await?;
    tracing::info!("Found {} active symbols", symbols.len());

    let timeframes = [
        Timeframe::M1,
        Timeframe::M5,
        Timeframe::M15,
        Timeframe::H1,
        Timeframe::H4,
        Timeframe::D1,
    ];

    // Initial wait to let ingestor start backfilling
    tracing::info!(
        "Waiting {}s for ingestor to populate candle data...",
        initial_wait_secs
    );
    tokio::time::sleep(Duration::from_secs(initial_wait_secs)).await;

    let mut idle_cycles: u32 = 0;
    let mut total_jobs_submitted: u64 = 0;
    let mut cycle_count: u64 = 0;

    loop {
        cycle_count += 1;
        let mut jobs_this_cycle: u64 = 0;

        for timeframe in &timeframes {
            let candle_table = format!("market.candles_{}", timeframe.as_str());
            let tf_minutes = timeframe.to_minutes() as i16;

            // Check if candle table has any data
            if !table_has_data(&db_pool, &candle_table).await {
                continue;
            }

            for symbol in &symbols {
                // Find the gap: latest candle time vs latest indicator time
                let gap = find_uncomputed_gap(
                    &db_pool,
                    symbol.as_str(),
                    &candle_table,
                    tf_minutes,
                )
                .await;

                match gap {
                    Ok(Some(gap_info)) => {
                        if gap_info.candle_count < 15 {
                            // Not enough data for ATR etc.
                            continue;
                        }

                        // We need warmup candles before the gap start for indicator calculations
                        // Fetch enough history: gap + 200 warmup bars
                        let length = (gap_info.candle_count as usize + 200).min(10_000);

                        tracing::info!(
                            "Gap detected: {} {} gap_candles={} total_fetch={}",
                            symbol.as_str(),
                            timeframe.as_str(),
                            gap_info.candle_count,
                            length
                        );

                        let window_spec = WindowSpec {
                            length,
                            warmup: 100,
                        };

                        if let Err(e) = job_scheduler
                            .submit_batch(*timeframe, vec![symbol.clone()], window_spec)
                            .await
                        {
                            tracing::error!("Submit batch failed for {} {}: {}", symbol.as_str(), timeframe.as_str(), e);
                        } else {
                            jobs_this_cycle += 1;
                        }
                    }
                    Ok(None) => {
                        // No gap — this symbol/tf is up to date
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Error checking gap for {} {}: {}",
                            symbol.as_str(),
                            timeframe.as_str(),
                            e
                        );
                    }
                }
            }
        }

        total_jobs_submitted += jobs_this_cycle;

        if jobs_this_cycle > 0 {
            idle_cycles = 0;
            tracing::info!(
                "Cycle {}: submitted {} jobs (total: {}). Waiting for processing...",
                cycle_count,
                jobs_this_cycle,
                total_jobs_submitted
            );
            // Give time for computation to complete before next cycle
            // Longer wait when processing lots of jobs
            let wait = if jobs_this_cycle > 100 {
                Duration::from_secs(120)
            } else if jobs_this_cycle > 20 {
                Duration::from_secs(60)
            } else {
                Duration::from_secs(poll_interval_secs)
            };
            tokio::time::sleep(wait).await;
        } else {
            idle_cycles += 1;
            tracing::info!(
                "Cycle {}: no gaps found (idle {}/{}). Total jobs submitted: {}",
                cycle_count,
                idle_cycles,
                max_idle_cycles,
                total_jobs_submitted
            );

            if idle_cycles >= max_idle_cycles {
                tracing::info!(
                    "No new data for {} consecutive cycles. History processing complete. \
                     Total cycles: {}, total jobs: {}",
                    max_idle_cycles,
                    cycle_count,
                    total_jobs_submitted
                );
                break;
            }

            tokio::time::sleep(Duration::from_secs(poll_interval_secs)).await;
        }
    }

    // Give final batch time to flush
    tracing::info!("Waiting 30s for final flush...");
    tokio::time::sleep(Duration::from_secs(30)).await;

    tracing::info!("compute_history exiting gracefully");
    Ok(())
}

/// Information about an uncomputed gap for a symbol/timeframe pair
struct GapInfo {
    /// Number of candles that don't have corresponding indicators
    candle_count: i64,
}

/// Find candles that exist but don't have computed indicators yet.
/// Returns None if everything is up to date.
async fn find_uncomputed_gap(
    pool: &PgPool,
    symbol: &str,
    candle_table: &str,
    tf_minutes: i16,
) -> Result<Option<GapInfo>> {
    // Strategy: Compare max(time) in candle table vs max(time) in indicators_wide.
    // If indicators lag behind candles, there's a gap to process.
    //
    // We also check if there are ANY indicators for this symbol/tf.
    // If none exist at all, that's a full gap.

    let query = format!(
        r#"
        WITH candle_range AS (
            SELECT 
                MIN(time) as first_candle,
                MAX(time) as last_candle,
                COUNT(*) as total_candles
            FROM {} 
            WHERE symbol = $1
        ),
        indicator_range AS (
            SELECT MAX(time) as last_indicator
            FROM market.indicators_wide 
            WHERE symbol = $1 AND tf_minutes = $2
        )
        SELECT 
            cr.total_candles,
            cr.first_candle,
            cr.last_candle,
            ir.last_indicator,
            CASE
                WHEN cr.total_candles = 0 THEN 0
                WHEN ir.last_indicator IS NULL THEN cr.total_candles
                ELSE (
                    SELECT COUNT(*) FROM {} c
                    WHERE c.symbol = $1 AND c.time > ir.last_indicator
                )
            END as gap_candles
        FROM candle_range cr, indicator_range ir
        "#,
        candle_table, candle_table
    );

    let row = sqlx::query(&query)
        .bind(symbol)
        .bind(tf_minutes)
        .fetch_optional(pool)
        .await?;

    let row = match row {
        Some(r) => r,
        None => return Ok(None),
    };

    let total_candles: i64 = row.get("total_candles");
    if total_candles == 0 {
        return Ok(None);
    }

    let gap_candles: i64 = row.get("gap_candles");
    if gap_candles <= 0 {
        return Ok(None);
    }

    Ok(Some(GapInfo {
        candle_count: gap_candles,
    }))
}

/// Check if table has any data (fast: LIMIT 1)
async fn table_has_data(pool: &PgPool, table_name: &str) -> bool {
    let query = format!("SELECT 1 FROM {} LIMIT 1", table_name);
    sqlx::query(&query)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .is_some()
}

/// Wait for active symbols in market.pairs
async fn wait_for_active_symbols(pool: &PgPool) -> Result<Vec<Symbol>> {
    let start = std::time::Instant::now();
    loop {
        let symbols = fetch_active_symbols_from_db(pool).await?;
        if !symbols.is_empty() {
            return Ok(symbols);
        }

        if start.elapsed() > Duration::from_secs(300) {
            anyhow::bail!("Timeout waiting for market.pairs");
        }

        eprintln!("Waiting for active symbols in market.pairs...");
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

async fn fetch_active_symbols_from_db(pool: &PgPool) -> Result<Vec<Symbol>> {
    let table_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema='market' AND table_name='pairs')"
    ).fetch_one(pool).await.unwrap_or(false);

    if !table_exists { return Ok(vec![]); }

    let rows = sqlx::query("SELECT symbol FROM market.pairs WHERE is_active = true")
        .fetch_all(pool).await?;

    Ok(rows.into_iter().map(|r| Symbol::from(r.get::<String, _>("symbol"))).collect())
}
