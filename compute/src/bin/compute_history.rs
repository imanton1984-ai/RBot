use std::sync::Arc;
use anyhow::{Context, Result};
use common::{Symbol, Timeframe, MessageBus};
use compute_lib::{*, IndicatorPersistor, RawSignalPersistor, RawSignalProcessor, ResultProcessor};
use database_lib;
use dotenvy::dotenv;
use sqlx::{PgPool, Row};
use std::time::Duration;
use raw_signals::thresholds::SignalConfig;
use compute_lib::predictors::config::PredictorsConfig;
use compute_lib::predictors::pipeline::{PredictorsPipeline, FeatureSnapshot};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());
    
    database_lib::init_db::initialize_database(&db_url)
        .await
        .map_err(|e| anyhow::anyhow!("Database init failed: {}", e))?;

    // фиксируем CUDA backend (если бинарь собран с --features cuda)
    let backend_type = if cfg!(feature = "cuda") {
        ComputeBackendType::Cuda
    } else {
        eprintln!("CUDA feature not enabled, falling back to CPU for history");
        ComputeBackendType::Cpu
    };
    
    let compute_backend_manager = ComputeBackendManager::new(backend_type);
    let compute_backend = compute_backend_manager.get_backend();

    // load config
    let config = ComputeConfig {
        batch_size: 100,
        max_concurrent_jobs: 4,
        use_cuda: cfg!(feature = "cuda"),
        cuda_device_id: Some(0),
    };

    let db_pool = sqlx::PgPool::connect(&db_url).await?;
    let candle_fetcher = Arc::new(CandleWindowFetcher::new(db_pool.clone()));

    // Set environment for history mode
    std::env::set_var("DB_PERSIST_MODE", "history");
    std::env::set_var("DB_PERSIST_CHUNK_SIZE_HISTORY", "50000"); // Large chunks for history
    std::env::set_var("DB_PERSIST_FLUSH_MS", "150"); // Slower flushing for history
    std::env::set_var("DB_PERSIST_HISTORY_SKIP_JSON", "1"); // Skip heavy JSON for history
    std::env::set_var("DB_PERSIST_HISTORY_UPSERT", "0"); // Append mode for history

    // Initialize DB Persistor (HISTORY Mode)
    let bulk_persistor = database_lib::bulk_persistor::BulkPersistor::new_from_env_mode(
        database_lib::bulk_persistor::PersistMode::History
    ).await?;
    let bulk_sender = bulk_persistor.sender();

    // Initialize Indicator & RawSignal Persistors
    let (indicator_persistor, _ind_tx) = IndicatorPersistor::new(bulk_sender.clone());
    let raw_signal_persistor = RawSignalPersistor::new(bulk_sender.clone());
    
    // Raw Signal Processor
    let raw_cfg = SignalConfig {
        enable_filtering: true,
        min_interesting_score: 0.60, // Can be configured via env
        ..Default::default()
    };
    let raw_processor = Arc::new(RawSignalProcessor::new(raw_cfg));

    let (job_scheduler, result_receiver) = JobScheduler::new(
        compute_backend,
        config.clone(),
        candle_fetcher.clone(),
    );
    let job_scheduler = Arc::new(job_scheduler);

    // Predictors Pipeline (XGBoost)
    let message_bus = MessageBus::new_from_env()?;
    let (_shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel::<bool>(1);
    
    let pred_config = PredictorsConfig {
        enabled: true,
        horizon_bars: 10,
        min_store_score: 0.60,
        min_final_score: 0.70,
        prefer_ml: true,
        max_levels_per_side: 2,
        use_cuda: config.use_cuda,
        use_gpu_history: config.use_cuda, // Important for batch processing
        use_gpu_realtime: false,
        model_path_price: "models/price_v1_tf{tf}.ubj".to_string(),
        model_path_levels: "models/levels_v1_tf{tf}.ubj".to_string(),
        ml_batch_size: 4096,
    };

    let (feature_tx, feature_rx) = tokio::sync::mpsc::unbounded_channel::<FeatureSnapshot>();
    
    let mut predictors_pipeline = PredictorsPipeline::new(
        pred_config,
        db_pool.clone(),
        message_bus,
        shutdown_rx.resubscribe(), // Create a new subscription for the pipeline
    );
    predictors_pipeline.set_input_receiver(feature_rx);

    // Spawn Predictors Pipeline
    tokio::spawn(async move {
        if let Err(e) = predictors_pipeline.run().await {
            tracing::error!(target: "compute_predictors", "Predictors pipeline error: {}", e);
        }
    });

    // Spawn Result Processor (The missing link!)
    let result_processor = ResultProcessor::new(
        Arc::new(indicator_persistor),
        Arc::new(raw_signal_persistor),
        raw_processor,
        feature_tx,
    );
    
    tokio::spawn(async move {
        result_processor.run(result_receiver).await;
    });

    // 1) получаем список активных пар из БД (market.pairs)
    let symbols = fetch_active_symbols_from_db(&db_pool).await?;

    // 2) для каждого TF запускаем submit_batch(…, window_spec)
    let timeframes = [
        Timeframe::M1,
        Timeframe::M5,
        Timeframe::M15,
        Timeframe::H1,
        Timeframe::H4,
        Timeframe::D1,
    ];

    for timeframe in &timeframes {
        println!("Processing historical data for timeframe: {:?}", timeframe);

        for symbol in &symbols {
            // Determine the historical range to process
            let table_name = format!("market.candles_{}", timeframe.as_str());
            let query = format!(
                "SELECT MIN(time) as first_time, MAX(time) as last_time
                 FROM {} c
                 JOIN market.pairs p ON c.symbol_id = p.symbol_id
                 WHERE p.symbol = ",
                table_name
            );

            let row = sqlx::query(&query)
                .bind(symbol.as_str())
                .fetch_optional(&db_pool)
                .await?;

            if let Some(row) = row {
                let first_time_opt: Option<chrono::DateTime<chrono::Utc>> = row.get("first_time");
                let last_time_opt: Option<chrono::DateTime<chrono::Utc>> = row.get("last_time");

                if let (Some(start_time), Some(end_time)) = (first_time_opt, last_time_opt) {
                    println!("Processing {} for {} from {} to {}", symbol.as_str(), timeframe.as_str(), start_time, end_time);

                    let duration_in_minutes = (end_time - start_time).num_minutes();
                    let timeframe_in_minutes = timeframe.to_minutes() as i64;
                    let length = if timeframe_in_minutes > 0 {
                        (duration_in_minutes / timeframe_in_minutes) as usize
                    } else {
                        0
                    };

                    let window_spec = WindowSpec {
                        length,
                        warmup: 0,
                    };

                    if let Err(e) = job_scheduler.submit_batch(*timeframe, vec![symbol.clone()], window_spec).await {
                        eprintln!("Error submitting batch job for {} {}: {}", symbol.as_str(), timeframe.as_str(), e);
                    } else {
                        println!("Submitted batch job for {} {}", symbol.as_str(), timeframe.as_str());
                    }
                }
            }
        }
    }

    // Wait a bit to allow jobs to complete
    tokio::time::sleep(Duration::from_secs(10)).await;

    Ok(())
}

// Function to fetch active symbols from the database
async fn fetch_active_symbols_from_db(
    db_pool: &PgPool,
) -> Result<Vec<Symbol>> {
    // First check if the table exists
    let table_exists: bool = sqlx::query_scalar(
        r#"SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'market' AND table_name = 'pairs'
        )"#,
    )
    .fetch_one(db_pool)
    .await
    .context("Failed to check for 'market.pairs' table existence")?;

    if !table_exists {
        return Ok(Vec::new()); // Return empty vector if table doesn't exist
    }

    // If table exists, try to fetch symbols
    let rows = sqlx::query("SELECT symbol FROM market.pairs WHERE is_active = true")
        .fetch_all(db_pool)
        .await
        .context("Failed to fetch active symbols from 'market.pairs'")?;

    let symbols: Vec<Symbol> = rows
        .into_iter()
        .map(|row| {
            let symbol: String = row.get("symbol");
            Symbol::from(symbol)
        })
        .collect();
    Ok(symbols)
}