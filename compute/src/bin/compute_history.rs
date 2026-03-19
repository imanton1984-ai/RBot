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
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn,compute_history=info"))
        .init();

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
    // Reduced chunk size: 20 new pairs × deep history = massive batches
    // that hit TimescaleDB "tuple decompression limit exceeded" on compressed chunks.
    // 2000 rows per chunk keeps each INSERT well within the decompression limit.
    std::env::set_var("DB_PERSIST_CHUNK_SIZE_HISTORY", "2000");
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
    // INCREMENTAL GAP-AWARE PROCESSING LOOP (v2 — optimized)
    // =====================================================================
    //
    // Key optimizations vs v1:
    //   1. Smart polling instead of hardcoded sleep(120s)
    //   2. Bulk gap detection: ONE SQL query for ALL symbols×TFs
    //   3. Ingestor-aware exit: tracks candle growth to know when ingestor is done
    //   4. No N+1 diagnostic queries

    let poll_interval_secs: u64 = 15;  // faster polling for gap check

    // 1) Wait for active pairs to appear
    let symbols = wait_for_active_symbols(&db_pool).await?;
    tracing::info!("Found {} active symbols", symbols.len());

    // ─── Timeframe selection ───────────────────────────────────────────
    let all_timeframes = [
        Timeframe::M1,
        Timeframe::M5,
        Timeframe::M15,
        Timeframe::H1,
        Timeframe::H4,
        Timeframe::D1,
    ];

    let timeframes: Vec<Timeframe> = if is_super {
        use ml_entry_strategy::config::SuperEntryConfig;
        let active_tfs = SuperEntryConfig::timeframes();
        let filtered: Vec<Timeframe> = all_timeframes.iter()
            .filter(|tf| active_tfs.contains(&(tf.to_minutes() as i32)))
            .copied()
            .collect();
        tracing::info!(
            "compute_history (super_entry): FILTERED timeframes: {:?}",
            filtered.iter().map(|tf| tf.as_str()).collect::<Vec<_>>()
        );
        filtered
    } else {
        all_timeframes.to_vec()
    };

    // ─── Decompress ALL target hypertable chunks ────────────────────
    // TimescaleDB compression causes "tuple decompression limit exceeded by operation"
    // on bulk INSERT into compressed chunks (especially with 20+ pairs × deep history).
    // Decompress ALL tables that BulkPersistor writes to before starting.
    {
        let tables_to_decompress = [
            "market.indicators_wide",
            "market.raw_signals",
            "trade.predictors",
            "trade.final_signals",
            "trade.super_entry_signals",
        ];
        for table in &tables_to_decompress {
            tracing::info!("Decompressing compressed chunks in {} ...", table);
            let sql = format!(
                "SELECT decompress_chunk(c, true) \
                 FROM show_chunks('{}', older_than => INTERVAL '0 seconds') c \
                 WHERE is_compressed",
                table
            );
            match sqlx::raw_sql(&sql).execute(&db_pool).await {
                Ok(_) => tracing::info!("{} chunks decompressed OK", table),
                Err(e) => tracing::warn!(
                    "{} decompress skipped (no compressed chunks or not a hypertable): {}",
                    table, e
                ),
            }
        }
        tracing::info!("All target tables decompressed — safe for bulk INSERT");
    }

    // ─── Smart wait for ingestor (replaces hardcoded sleep 120s) ──────
    // Poll every 5s, start as soon as ANY candle data appears in required TFs.
    // Max wait 180s to handle cold-start.
    {
        let wait_start = std::time::Instant::now();
        let max_wait = Duration::from_secs(180);
        let poll = Duration::from_secs(5);
        tracing::info!("Waiting for ingestor to populate candle data (poll={}s, max={}s)...",
            poll.as_secs(), max_wait.as_secs());
        
        loop {
            let mut ready_tfs = 0u32;
            for tf in &timeframes {
                let table = format!("market.candles_{}", tf.as_str());
                if table_has_data(&db_pool, &table).await {
                    ready_tfs += 1;
                }
            }
            if ready_tfs > 0 {
                tracing::info!(
                    "Candle data detected in {}/{} TFs after {:.1}s — starting processing",
                    ready_tfs, timeframes.len(), wait_start.elapsed().as_secs_f64()
                );
                break;
            }
            if wait_start.elapsed() > max_wait {
                tracing::warn!("Max wait {}s exceeded, starting with whatever data is available", max_wait.as_secs());
                break;
            }
            tokio::time::sleep(poll).await;
        }
    }

    // ─── Bulk diagnostic (ONE query instead of 724) ───────────────────
    if is_super {
        let tf_list: Vec<i16> = timeframes.iter().map(|tf| tf.to_minutes() as i16).collect();
        let diag = bulk_gap_summary(&db_pool, &timeframes).await;
        tracing::info!(
            "compute_history: {} symbols, {} TFs ({:?}). Gaps: {}/{} symbol×TF pairs need indicators",
            symbols.len(), tf_list.len(), tf_list, diag.gaps_with_work, diag.total_pairs
        );
    }

    // ─── Main processing loop ─────────────────────────────────────────
    // Ingestor-aware: we track total candle count across all TFs.
    // If candle count grows between cycles → ingestor still loading → don't exit.
    // If candle count stable AND no gaps → truly done.
    let mut prev_total_candles: i64 = 0;
    let mut stable_cycles: u32 = 0;          // cycles where candle count didn't grow AND no gaps
    let max_stable_cycles: u32 = 2;          // exit after 2 consecutive stable+idle cycles
    let mut total_jobs_submitted: u64 = 0;
    let mut cycle_count: u64 = 0;

    loop {
        cycle_count += 1;
        let mut jobs_this_cycle: u64 = 0;

        // ── BULK gap detection: ONE SQL per TF (instead of 724 individual queries) ──
        let all_gaps = find_all_gaps_bulk(&db_pool, &timeframes).await;

        for gap in &all_gaps {
            if gap.gap_candles < 25 {
                continue;
            }

            let symbol = Symbol::from(gap.symbol.clone());
            let timeframe = match gap.tf_str.parse::<Timeframe>() {
                Ok(tf) => tf,
                Err(_) => continue,
            };

            let length = (gap.gap_candles as usize + 200).min(10_000);

            tracing::info!(
                "Gap detected: {} {} gap_candles={} total_fetch={}",
                gap.symbol, gap.tf_str, gap.gap_candles, length
            );

            let window_spec = WindowSpec {
                length,
                warmup: 100,
            };

            if let Err(e) = job_scheduler
                .submit_batch(timeframe, vec![symbol], window_spec)
                .await
            {
                tracing::error!("Submit batch failed for {} {}: {}", gap.symbol, gap.tf_str, e);
            } else {
                jobs_this_cycle += 1;
            }
        }

        total_jobs_submitted += jobs_this_cycle;

        // ── Check ingestor progress: are candles still growing? ──
        let current_total_candles = count_total_candles(&db_pool, &timeframes).await;
        let candles_growing = current_total_candles > prev_total_candles;
        prev_total_candles = current_total_candles;

        if jobs_this_cycle > 0 {
            stable_cycles = 0;
            tracing::info!(
                "Cycle {}: submitted {} jobs (total: {}). Candles: {} ({}). Waiting for processing...",
                cycle_count, jobs_this_cycle, total_jobs_submitted,
                current_total_candles,
                if candles_growing { "growing" } else { "stable" }
            );
            // Scale wait by job count — but not excessively
            let wait = if jobs_this_cycle > 100 {
                Duration::from_secs(60)
            } else if jobs_this_cycle > 20 {
                Duration::from_secs(30)
            } else {
                Duration::from_secs(poll_interval_secs)
            };
            tokio::time::sleep(wait).await;
        } else {
            // No gaps found this cycle
            if candles_growing {
                // Ingestor still loading — new candles appeared, wait and recheck
                stable_cycles = 0;
                tracing::info!(
                    "Cycle {}: no gaps but candles still growing ({} total, +{}). Ingestor active — waiting...",
                    cycle_count, current_total_candles,
                    current_total_candles - prev_total_candles + (current_total_candles - prev_total_candles).abs()
                );
                tokio::time::sleep(Duration::from_secs(poll_interval_secs)).await;
            } else {
                // Candles stable AND no gaps → possible completion
                stable_cycles += 1;
                tracing::info!(
                    "Cycle {}: no gaps, candles stable ({} total). Stable cycle {}/{}. Total jobs: {}",
                    cycle_count, current_total_candles, stable_cycles, max_stable_cycles, total_jobs_submitted
                );
                if stable_cycles >= max_stable_cycles {
                    tracing::info!(
                        "Ingestor done + no gaps for {} cycles. History processing complete. \
                         Total cycles: {}, total jobs: {}, total candles: {}",
                        max_stable_cycles, cycle_count, total_jobs_submitted, current_total_candles
                    );
                    break;
                }
                tokio::time::sleep(Duration::from_secs(poll_interval_secs)).await;
            }
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

    // Final summary (BULK — 2 queries instead of 2172 individual queries)
    if is_super {
        tracing::info!("compute_history: === FINAL SUMMARY ===");
        let summary_rows = sqlx::query_as::<_, (String, i16, i64)>(
            "SELECT symbol, tf_minutes, COUNT(*) as cnt \
             FROM market.indicators_wide \
             GROUP BY symbol, tf_minutes \
             ORDER BY symbol, tf_minutes"
        )
        .fetch_all(&db_pool)
        .await
        .unwrap_or_default();

        let se_rows = sqlx::query_as::<_, (String, i16, i64)>(
            "SELECT symbol, tf_minutes, COUNT(*) as cnt \
             FROM trade.super_entry_signals \
             GROUP BY symbol, tf_minutes \
             ORDER BY symbol, tf_minutes"
        )
        .fetch_all(&db_pool)
        .await
        .unwrap_or_default();

        let mut se_map: std::collections::HashMap<(String, i16), i64> = std::collections::HashMap::new();
        for (sym, tf, cnt) in &se_rows {
            se_map.insert((sym.clone(), *tf), *cnt);
        }

        for (sym, tf, ind_cnt) in &summary_rows {
            let se_cnt = se_map.get(&(sym.clone(), *tf)).copied().unwrap_or(0);
            tracing::info!(
                "RESULT: {} {}m - indicators: {}, super_entry_signals: {}",
                sym, tf, ind_cnt, se_cnt
            );
        }

        let total_ind: i64 = summary_rows.iter().map(|(_, _, c)| c).sum();
        let total_se: i64 = se_rows.iter().map(|(_, _, c)| c).sum();
        tracing::info!(
            "TOTAL: {} indicator rows, {} super_entry_signal rows across {} symbol×TF pairs",
            total_ind, total_se, summary_rows.len()
        );
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

    // Per-TF candle limits — must match dataset_builder and backtest limits.
    // More data = more signals = better coverage.
    let backfill_limit_per_tf = |tf_minutes: i32| -> usize {
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

    let t0 = std::time::Instant::now();
    let mut total_signals = 0usize;
    let mut total_candles_processed = 0usize;

    for &tf in SuperEntryConfig::timeframes() {
        let tf_t0 = std::time::Instant::now();
        let limit = backfill_limit_per_tf(tf);

        // Single bulk query: fetch ALL symbols' candles+indicators for this TF
        // Uses window function (ROW_NUMBER) — much faster than N per-symbol queries
        let grouped = match fetch_all_candles_for_tf(pool, tf, limit).await {
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

// ═══════════════════════════════════════════════════════════════════════════
// BULK GAP DETECTION (v2) — replaces per-symbol find_uncomputed_gap
// ═══════════════════════════════════════════════════════════════════════════

/// A gap found for a specific symbol × timeframe pair
struct BulkGapInfo {
    symbol: String,
    tf_str: String,
    gap_candles: i64,
}

/// Summary of gap detection across all symbols×TFs
struct GapSummary {
    total_pairs: u64,
    gaps_with_work: u64,
}

/// Find ALL uncomputed gaps across ALL symbols and TFs in bulk.
/// Uses ONE SQL query per TF — NO correlated subqueries (fast: <100ms).
///
/// Strategy: Compare MAX(time_ms) in candles vs MAX(time_ms) in indicators.
/// When last_indicator IS NULL → all candles are gaps (total_candles).
/// When last_candle > last_indicator → estimate gap from time difference.
/// This avoids the slow `COUNT(*) WHERE time > X` correlated subquery
/// that was taking 3-5 seconds per TF on compressed TimescaleDB tables.
async fn find_all_gaps_bulk(pool: &PgPool, timeframes: &[Timeframe]) -> Vec<BulkGapInfo> {
    let mut all_gaps = Vec::new();

    for tf in timeframes {
        let candle_table = format!("market.candles_{}", tf.as_str());
        let tf_minutes = tf.to_minutes() as i16;
        let tf_str = tf.as_str().to_string();
        let interval_ms = tf.to_minutes() as i64 * 60 * 1000;

        // Fast query: NO correlated subqueries.
        // Uses MAX(time_ms) comparison and estimates gap size from time delta.
        let query = format!(
            r#"
            WITH candle_stats AS (
                SELECT symbol,
                       COUNT(*) as total_candles,
                       MAX(time_ms) as last_candle_ms
                FROM {}
                GROUP BY symbol
                HAVING COUNT(*) > 0
            ),
            indicator_stats AS (
                SELECT symbol,
                       MAX(time_ms) as last_indicator_ms
                FROM market.indicators_wide
                WHERE tf_minutes = $1
                GROUP BY symbol
            )
            SELECT
                c.symbol,
                CASE
                    WHEN i.last_indicator_ms IS NULL THEN c.total_candles
                    WHEN c.last_candle_ms > i.last_indicator_ms THEN
                        GREATEST(1, (c.last_candle_ms - i.last_indicator_ms) / $2)
                    ELSE 0
                END as gap_candles
            FROM candle_stats c
            LEFT JOIN indicator_stats i ON c.symbol = i.symbol
            WHERE i.last_indicator_ms IS NULL
               OR c.last_candle_ms > i.last_indicator_ms
            "#,
            candle_table
        );

        match sqlx::query(&query)
            .bind(tf_minutes)
            .bind(interval_ms)
            .fetch_all(pool)
            .await
        {
            Ok(rows) => {
                for row in rows {
                    let symbol: String = row.get("symbol");
                    let gap_candles: i64 = row.get("gap_candles");
                    all_gaps.push(BulkGapInfo {
                        symbol,
                        tf_str: tf_str.clone(),
                        gap_candles,
                    });
                }
            }
            Err(e) => {
                tracing::warn!("Bulk gap detection failed for TF {}: {}", tf.as_str(), e);
            }
        }
    }

    all_gaps
}

/// Quick summary of gaps (for diagnostic logging)
async fn bulk_gap_summary(pool: &PgPool, timeframes: &[Timeframe]) -> GapSummary {
    let gaps = find_all_gaps_bulk(pool, timeframes).await;
    let gaps_with_work = gaps.iter().filter(|g| g.gap_candles >= 25).count() as u64;
    let total_pairs = gaps.len() as u64;
    GapSummary {
        total_pairs,
        gaps_with_work,
    }
}

/// Count total candles across all required TFs (for ingestor progress tracking).
/// Returns a single number — if it grows between cycles, ingestor is still active.
async fn count_total_candles(pool: &PgPool, timeframes: &[Timeframe]) -> i64 {
    let mut total: i64 = 0;
    for tf in timeframes {
        let table = format!("market.candles_{}", tf.as_str());
        let query = format!("SELECT COUNT(*) FROM {}", table);
        let count: Option<i64> = sqlx::query_scalar(&query)
            .fetch_optional(pool)
            .await
            .ok()
            .flatten();
        total += count.unwrap_or(0);
    }
    total
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
