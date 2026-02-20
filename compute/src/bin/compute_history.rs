use std::sync::Arc;
use anyhow::Result;
use common::{Symbol, Timeframe};
use compute_lib::{*, IndicatorPersistor, RawSignalPersistor, RawSignalProcessor, ResultProcessor};
use compute_lib::predictors::pipeline::FeatureSnapshot;
use database_lib;
use dotenvy::dotenv;
use sqlx::{PgPool, Row};
use std::time::Duration;
use raw_signals::thresholds::SignalConfig;

/// Determine which strategy is active
fn get_active_strategy() -> String {
    std::env::var("ACTIVE_STRATEGY")
        .unwrap_or_else(|_| "level".to_string())
}

/// Check if we should run predictors pipeline (only for level strategy)
fn should_run_predictors() -> bool {
    let strategy = get_active_strategy();
    strategy == "level" || strategy == "default"
}

/// Check if super_entry strategy is enabled
fn is_super_entry_enabled() -> bool {
    let strategy = get_active_strategy();
    let explicit = std::env::var("SUPER_ENTRY_ENABLED")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);
    strategy == "super_entry" || strategy == "combined" || explicit
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let active_strategy = get_active_strategy();
    tracing::info!("compute_history: ACTIVE_STRATEGY = {}", active_strategy);

    let run_predictors = should_run_predictors();
    let is_super = is_super_entry_enabled();

    if run_predictors {
        tracing::info!("compute_history: Running predictors + trade_signals pipeline (level strategy)");
    } else if is_super {
        tracing::info!("compute_history: MODE: indicators + raw_signals + super_entry_signals (ZERO-COPY)");
        tracing::info!("compute_history: SUPER_ENTRY_ENABLED={}", 
            std::env::var("SUPER_ENTRY_ENABLED").unwrap_or_else(|_| "false".to_string()));
    } else {
        tracing::info!("compute_history: Skipping predictors pipeline (strategy: {})", active_strategy);
    }

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

    // 4. Feature Channel: GPU indicators → downstream consumer
    let (feature_tx, feature_rx) = tokio::sync::mpsc::unbounded_channel::<FeatureSnapshot>();

    // 5. Connect feature_rx to the appropriate consumer:
    //    - Level strategy → PredictorsPipeline
    //    - Super Entry strategy → SuperEntryStage (ZERO-COPY!)
    //    - Neither → drop receiver
    let super_entry_handle: Option<tokio::task::JoinHandle<Result<()>>> = if run_predictors {
        // Level strategy: use predictors pipeline
        setup_predictors_pipeline(
            &db_pool,
            &bulk_sender,
            feature_rx,
            config.use_cuda,
        ).await;
        None
    } else if is_super {
        // ═══════════════════════════════════════════════════════════════════
        // SUPER ENTRY ZERO-COPY PATH:
        // GPU indicators → FeatureSnapshot → SuperEntryStage → XGBoost → DB
        // No DB roundtrip for indicator data — direct in-memory processing.
        // ═══════════════════════════════════════════════════════════════════
        tracing::info!("compute_history: Setting up SuperEntryStage for ZERO-COPY history pipeline");
        match compute_lib::super_entry_stage::setup_super_entry_stage(
            &db_pool,
            feature_rx,
            config.use_cuda,
        ).await {
            Some(handle) => {
                tracing::info!("compute_history: ✅ SuperEntryStage spawned — ZERO-COPY pipeline active");
                Some(handle)
            }
            None => {
                // Stage failed to initialize (no models? table error?)
                // feature_rx was already consumed by setup_super_entry_stage
                tracing::warn!("compute_history: SuperEntryStage not available — will use DB backfill fallback");
                None
            }
        }
    } else {
        drop(feature_rx);
        None
    };

    // 6. Spawn Result Processor
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
    // =====================================================================

    // For super_entry mode: use longer wait for candles to load (ingestor backfill takes 2+ minutes)
    let max_idle_cycles: u32 = if is_super {
        6  // Super entry: allow more idle cycles (candles take time to load)
    } else {
        std::env::var("HISTORY_MAX_IDLE_CYCLES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10)
    };

    let poll_interval_secs: u64 = if is_super { 30 } else { 30 };
    let initial_wait_secs: u64 = if is_super { 120 } else { 15 };

    // 1) Wait for active pairs to appear
    let symbols = wait_for_active_symbols(&db_pool).await?;
    tracing::info!("Found {} active symbols", symbols.len());

    if is_super {
        tracing::info!("compute_history (super_entry): will persist indicators for: {:?}", symbols);
        tracing::info!("compute_history (super_entry): fast mode enabled (idle_cycles={}, poll={}s, initial_wait={}s)", 
            max_idle_cycles, poll_interval_secs, initial_wait_secs);
    }

    let timeframes = [
        Timeframe::M1,
        Timeframe::M5,
        Timeframe::M15,
        Timeframe::H1,
        Timeframe::H4,
        Timeframe::D1,
    ];

    tracing::info!(
        "Waiting {}s for ingestor to populate candle data...",
        initial_wait_secs
    );
    tokio::time::sleep(Duration::from_secs(initial_wait_secs)).await;
    
    // Log current state before processing
    if is_super {
        for tf in &timeframes {
            let candle_table = format!("market.candles_{}", tf.as_str());
            for symbol in &symbols {
                let candle_count: Option<i64> = sqlx::query_scalar(&format!(
                    "SELECT COUNT(*) FROM {} WHERE symbol = $1", candle_table
                ))
                .bind(symbol.as_str())
                .fetch_optional(&db_pool)
                .await
                .ok()
                .flatten();
                
                let indicator_count: Option<i64> = sqlx::query_scalar(
                    "SELECT COUNT(*) FROM market.indicators_wide WHERE symbol = $1 AND tf_minutes = $2"
                )
                .bind(symbol.as_str())
                .bind(tf.to_minutes() as i16)
                .fetch_optional(&db_pool)
                .await
                .ok()
                .flatten();
                
                tracing::info!(
                    "compute_history (super_entry): {} {} - candles: {:?}, indicators: {:?}",
                    symbol, tf.as_str(), candle_count, indicator_count
                );
            }
        }
    }

    let mut idle_cycles: u32 = 0;
    let mut total_jobs_submitted: u64 = 0;
    let mut cycle_count: u64 = 0;

    loop {
        cycle_count += 1;
        let mut jobs_this_cycle: u64 = 0;
        let mut tables_with_data = 0u64;
        let mut tables_empty = 0u64;

        for timeframe in &timeframes {
            let candle_table = format!("market.candles_{}", timeframe.as_str());
            let tf_minutes = timeframe.to_minutes() as i16;

            // Check if candle table has any data
            if !table_has_data(&db_pool, &candle_table).await {
                tables_empty += 1;
                continue;
            }
            tables_with_data += 1;

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
                "Cycle {}: submitted {} jobs (total: {}). Tables with data: {}/{}. Waiting for processing...",
                cycle_count,
                jobs_this_cycle,
                total_jobs_submitted,
                tables_with_data,
                timeframes.len()
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
            // Only count idle cycles if ALL tables have data
            // If some tables are still empty, ingestor is still loading candles
            let all_tables_ready = tables_empty == 0;
            
            if all_tables_ready {
                idle_cycles += 1;
            } else {
                tracing::info!(
                    "Cycle {}: waiting for ingestor (tables_empty={}/{}, will not count as idle)",
                    cycle_count, tables_empty, timeframes.len()
                );
            }
            
            tracing::info!(
                "Cycle {}: no gaps found (idle {}/{}, tables_with_data={}/{}). Total jobs: {}",
                cycle_count,
                idle_cycles,
                max_idle_cycles,
                tables_with_data,
                timeframes.len(),
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

    // =====================================================================
    // SHUTDOWN CHAIN — ensure all data is flushed
    // =====================================================================
    //
    // The chain is:
    //   1. Drop job_scheduler → closes result_sender
    //   2. ResultProcessor sees channel closed → exits → drops feature_tx
    //   3. SuperEntryStage sees feature_rx closed → flushes remaining buffers → exits
    //   4. BulkPersistor flushes remaining data to DB
    //
    // This ensures true ZERO-COPY: GPU → indicators → channel → XGBoost → DB

    tracing::info!("compute_history: Main loop complete. Starting shutdown chain...");

    // Step 1: Drop job_scheduler to close the result_sender channel.
    // This triggers ResultProcessor to exit, which in turn drops feature_tx,
    // which triggers SuperEntryStage to flush remaining buffers.
    drop(job_scheduler);
    tracing::info!("compute_history: JobScheduler dropped → result_sender closed");

    // Step 2: Wait for SuperEntryStage to finish processing
    if let Some(handle) = super_entry_handle {
        tracing::info!("compute_history: Waiting for SuperEntryStage to finish (timeout: 300s)...");
        match tokio::time::timeout(Duration::from_secs(300), handle).await {
            Ok(Ok(Ok(()))) => {
                tracing::info!("compute_history: ✅ SuperEntryStage completed successfully");
            }
            Ok(Ok(Err(e))) => {
                tracing::error!("compute_history: ❌ SuperEntryStage error: {}", e);
            }
            Ok(Err(e)) => {
                tracing::error!("compute_history: ❌ SuperEntryStage panicked: {}", e);
            }
            Err(_) => {
                tracing::error!("compute_history: ❌ SuperEntryStage timed out after 300s");
            }
        }
    } else if is_super {
        // ═══════════════════════════════════════════════════════════════════
        // FALLBACK: DB-based backfill (when SuperEntryStage was not available)
        // This is slower — reads indicators from DB instead of in-memory.
        // ═══════════════════════════════════════════════════════════════════
        tracing::info!("compute_history: Using DB-based backfill fallback for super_entry signals");
        
        // Give final batch time to flush to DB before reading back
        tracing::info!("Waiting 30s for final flush to DB...");
        tokio::time::sleep(Duration::from_secs(30)).await;

        tracing::info!("compute_history: === SUPER ENTRY BACKFILL (DB fallback) ===");
        
        let ind_count: Option<i64> = sqlx::query_scalar("SELECT COUNT(*) FROM market.indicators_wide")
            .fetch_optional(&db_pool)
            .await
            .ok()
            .flatten();
        tracing::info!("compute_history: indicators_wide has {} records", ind_count.unwrap_or(0));
        
        let sig_count: Option<i64> = sqlx::query_scalar("SELECT COUNT(*) FROM trade.super_entry_signals")
            .fetch_optional(&db_pool)
            .await
            .ok()
            .flatten();
        tracing::info!("compute_history: super_entry_signals has {} records (before backfill)", sig_count.unwrap_or(0));

        match run_super_entry_backfill(&db_pool, config.use_cuda).await {
            Ok(total_signals) => {
                tracing::info!(
                    "compute_history: ✅ Super Entry backfill complete: {} signals written",
                    total_signals
                );
            }
            Err(e) => {
                tracing::error!("compute_history: ❌ Super Entry backfill failed: {}", e);
            }
        }
    }

    // Step 3: Wait for BulkPersistor to flush remaining data
    tracing::info!("compute_history: Waiting 15s for BulkPersistor final flush...");
    tokio::time::sleep(Duration::from_secs(15)).await;

    // Final summary for super_entry mode
    if is_super {
        tracing::info!("compute_history: === FINAL SUMMARY ===");
        for tf in &timeframes {
            for symbol in &symbols {
                let indicator_count: Option<i64> = sqlx::query_scalar(
                    "SELECT COUNT(*) FROM market.indicators_wide WHERE symbol = $1 AND tf_minutes = $2"
                )
                .bind(symbol.as_str())
                .bind(tf.to_minutes() as i16)
                .fetch_optional(&db_pool)
                .await
                .ok()
                .flatten();
                
                let raw_signal_count: Option<i64> = sqlx::query_scalar(
                    "SELECT COUNT(*) FROM market.raw_signals WHERE symbol = $1 AND tf_minutes = $2"
                )
                .bind(symbol.as_str())
                .bind(tf.to_minutes() as i16)
                .fetch_optional(&db_pool)
                .await
                .ok()
                .flatten();
                
                let super_entry_count: Option<i64> = sqlx::query_scalar(
                    "SELECT COUNT(*) FROM trade.super_entry_signals WHERE symbol = $1 AND tf_minutes = $2"
                )
                .bind(symbol.as_str())
                .bind(tf.to_minutes() as i16)
                .fetch_optional(&db_pool)
                .await
                .ok()
                .flatten();
                
                tracing::info!(
                    "RESULT: {} {} - indicators: {:?}, raw_signals: {:?}, super_entry_signals: {:?}",
                    symbol, tf.as_str(), indicator_count, raw_signal_count, super_entry_count
                );
            }
        }
    }

    tracing::info!("compute_history exiting gracefully");
    Ok(())
}

/// Run Super Entry backfill — reads candles+indicators from DB, runs XGBoost inference,
/// writes signals to trade.super_entry_signals.
///
/// This is the FALLBACK path when SuperEntryStage couldn't be initialized.
/// The preferred path is ZERO-COPY via SuperEntryStage in the main pipeline.
///
/// Optimized for throughput:
///   - Uses fetch_all_candles_for_tf() — single SQL query per TF (all symbols at once)
///   - Batch XGBoost inference per (symbol, tf) — processes all candles in one predict_batch call
///   - UNNEST batch INSERT for signals (single RTT per TF)
///   - GPU inference when available (use_gpu flag)
async fn run_super_entry_backfill(pool: &PgPool, use_cuda: bool) -> Result<usize> {
    use ml_entry_strategy::config::SuperEntryConfig;
    use ml_entry_strategy::dataset::fetch_all_candles_for_tf;
    use ml_entry_strategy::db_writer::{insert_signals_batch, ensure_table_exists};
    use ml_entry_strategy::pipeline::SuperEntryPipeline;
    use ml_entry_strategy::signal_generator::SuperEntrySignal;

    let config = SuperEntryConfig::from_env();
    let use_gpu = std::env::var("SUPER_ENTRY_USE_GPU")
        .unwrap_or_default()
        .parse::<bool>()
        .unwrap_or(false) || use_cuda;

    tracing::info!("Super Entry backfill: config loaded (warmup={}, lookahead={}, p_threshold={})", 
        config.warmup_bars, config.lookahead_bars, config.p_threshold);
    tracing::info!("Super Entry backfill: GPU={}", use_gpu);

    // Ensure table exists
    ensure_table_exists(pool).await?;
    tracing::info!("Super Entry backfill: table ensured");

    // Load models
    tracing::info!("Super Entry backfill: loading models...");
    let pipeline = SuperEntryPipeline::new(config.clone(), use_gpu)?;
    if !pipeline.has_models() {
        tracing::warn!("Super Entry backfill: no models found, skipping");
        return Ok(0);
    }
    tracing::info!("Super Entry backfill: models loaded, has_models={}", pipeline.has_models());

    tracing::info!(
        "Super Entry backfill: {} TFs, warmup={}, gpu={}",
        SuperEntryConfig::timeframes().len(),
        config.warmup_bars,
        use_gpu
    );

    let t0 = std::time::Instant::now();
    let mut total_signals = 0usize;
    let mut total_candles_processed = 0usize;

    for &tf in SuperEntryConfig::timeframes() {
        let tf_t0 = std::time::Instant::now();

        // Single bulk query: fetch ALL symbols' candles+indicators for this TF
        // Uses window function (ROW_NUMBER) — much faster than N per-symbol queries
        let grouped = match fetch_all_candles_for_tf(pool, tf, 1000).await {
            Ok(g) => g,
            Err(e) => {
                tracing::warn!("Super Entry backfill: fetch_all TF {}m failed: {}", tf, e);
                continue;
            }
        };

        let mut tf_signals: Vec<SuperEntrySignal> = Vec::new();
        let mut tf_candles = 0usize;
        let mut tf_symbols_ok = 0usize;

        tracing::info!(
            "Super Entry backfill: TF {}m — fetched {} symbols in {}ms",
            tf, grouped.len(), tf_t0.elapsed().as_millis()
        );

        // Process each symbol's candles through the pipeline
        // process_candles does batch XGBoost inference internally
        for (symbol, candles) in &grouped {
            if candles.len() < config.warmup_bars + 1 {
                continue;
            }
            tf_symbols_ok += 1;

            let results = match pipeline.process_candles(candles, tf, use_gpu) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(
                        "Super Entry backfill: {} {}m process_candles failed: {}",
                        symbol, tf, e
                    );
                    continue;
                }
            };
            tf_candles += results.len();
            for r in results {
                if let Some(s) = r.signal {
                    tf_signals.push(s);
                }
            }
        }

        total_candles_processed += tf_candles;

        // Batch INSERT all signals for this TF in one go (UNNEST)
        if !tf_signals.is_empty() {
            let count = tf_signals.len();
            match insert_signals_batch(pool, &tf_signals).await {
                Ok(n) => {
                    tracing::info!(
                        "Super Entry backfill: TF {:>5}m: {} symbols → {} candles → {} signals → {} written ({}ms)",
                        tf, tf_symbols_ok, tf_candles, count, n, tf_t0.elapsed().as_millis()
                    );
                    total_signals += count;
                }
                Err(e) => {
                    tracing::error!("Super Entry backfill: TF {}m INSERT failed: {}", tf, e);
                }
            }
        } else {
            tracing::info!(
                "Super Entry backfill: TF {:>5}m: {} symbols → {} candles → 0 signals ({}ms)",
                tf, tf_symbols_ok, tf_candles, tf_t0.elapsed().as_millis()
            );
        }
    }

    tracing::info!(
        "Super Entry backfill complete: {} signals from {} candles in {:.1}s",
        total_signals, total_candles_processed, t0.elapsed().as_secs_f64()
    );

    Ok(total_signals)
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

/// Setup predictors pipeline (only for level strategy)
async fn setup_predictors_pipeline(
    db_pool: &PgPool,
    bulk_sender: &tokio::sync::mpsc::Sender<database_lib::PersistRecord>,
    feature_rx: tokio::sync::mpsc::UnboundedReceiver<FeatureSnapshot>,
    use_cuda: bool,
) {
    use compute_lib::predictors::pipeline::PredictorsPipeline;
    use compute_lib::predictors::config::PredictorsConfig;
    use compute_lib::scoring::trade_signal_processor::{TradeSignalStage, TradeSignalInput};
    use compute_lib::scoring::market_params_calculator::MarketParamsCalculator;
    use common::MessageBus;

    let message_bus = MessageBus::new_from_env().expect("Failed to create message bus");
    let (_shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel::<bool>(1);

    // min_final_score: configurable via env var.
    let min_final_score: f64 = std::env::var("MIN_FINAL_SCORE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.60);

    tracing::info!("Using min_final_score = {}", min_final_score);

    let pred_config = PredictorsConfig {
        enabled: true,
        horizon_bars: 10,
        min_store_score: 0.50,
        min_final_score,
        prefer_ml: true,
        max_levels_per_side: 2,
        use_cuda,
        use_gpu_history: use_cuda,
        use_gpu_realtime: false,
        model_path_price: "models/price_v1_tf{tf}.ubj".to_string(),
        model_path_levels: "models/levels_v1_tf{tf}.ubj".to_string(),
        ml_batch_size: 4096,
    };

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

    tracing::info!("Predictors pipeline initialized (level strategy mode)");
}
