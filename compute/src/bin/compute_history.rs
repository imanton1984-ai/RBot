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
        batch_size: 5000,
        max_concurrent_jobs: 2,
        use_cuda: cfg!(feature = "cuda"),
        cuda_device_id: Some(0),
    };

    let db_pool = sqlx::PgPool::connect(&db_url).await?;
    let candle_fetcher = Arc::new(CandleWindowFetcher::new(db_pool.clone()));

    // --- PIPELINE SETUP START ---

    // 1. Initialize DB Persistor (HISTORY Mode)
    std::env::set_var("DB_PERSIST_MODE", "history");
    // Optimization for bulk loading
    std::env::set_var("DB_PERSIST_CHUNK_SIZE_HISTORY", "50000"); 
    std::env::set_var("DB_PERSIST_FLUSH_MS", "150"); 
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
    
    let pred_config = PredictorsConfig {
        enabled: true,
        horizon_bars: 10,
        min_store_score: 0.60,
        min_final_score: 0.70,
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
        pred_config,
        db_pool.clone(),
        message_bus,
        shutdown_rx.resubscribe(),
    );
    predictors_pipeline.set_input_receiver(feature_rx);
    predictors_pipeline.set_bulk_sender(bulk_sender.clone());

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

    // 1) Ждем и получаем список активных пар
    let symbols = wait_for_active_symbols(&db_pool).await?;

    // 2) Обрабатываем таймфреймы
    let timeframes = [
        Timeframe::M1,
        Timeframe::M5,
        Timeframe::M15,
        Timeframe::H1,
        Timeframe::H4,
        Timeframe::D1,
    ];

    for timeframe in &timeframes {
        let table_name = format!("market.candles_{}", timeframe.as_str());

        // ВАЖНО: Ждем данные вместо пропуска
        if !wait_for_table_data(&db_pool, &table_name).await {
            eprintln!("Timeout waiting for data in {}, skipping TF", table_name);
            continue;
        }

        println!("Processing historical data for timeframe: {:?}", timeframe);

        for symbol in &symbols {
            // Check if candles exist for this specific symbol
            let symbol_query = format!(
                "SELECT MIN(time) as first_time, MAX(time) as last_time, COUNT(*) as total_count
                 FROM {} WHERE symbol = $1",
                table_name
            );

            match sqlx::query(&symbol_query)
                .bind(symbol.as_str())
                .fetch_optional(&db_pool)
                .await {

                Ok(Some(row)) => {
                    let total_count: i64 = row.get("total_count");
                    if total_count == 0 {
                        continue;
                    }

                    let first_time: Option<chrono::DateTime<chrono::Utc>> = row.get("first_time");
                    let last_time: Option<chrono::DateTime<chrono::Utc>> = row.get("last_time");

                    if let (Some(start), Some(end)) = (first_time, last_time) {
                        let duration_min = (end - start).num_minutes();
                        let tf_min = timeframe.to_minutes() as i64;
                        let length = if tf_min > 0 { (duration_min / tf_min) as usize } else { 0 };

                        if length > 0 {
                            // Minimum required length for ATR and other indicators to work properly
                            // ATR typically needs at least period (14) + 1 data points
                            let min_required_length = 15; // 14 + 1 for ATR calculation
                            
                            if length < min_required_length {
                                tracing::warn!(
                                    "Skipping symbol {} on timeframe {} due to insufficient data: {} < {}", 
                                    symbol.as_str(), 
                                    timeframe.as_str(), 
                                    length, 
                                    min_required_length
                                );
                                continue; // Skip this symbol-timeframe combination
                            }

                            let window_spec = WindowSpec {
                                length: std::cmp::min(length + 100, 10_000), // +buffer
                                warmup: 100,
                            };

                            println!("Submitting batch: {} {}, len={}", symbol.as_str(), timeframe.as_str(), window_spec.length);

                            if let Err(e) = job_scheduler.submit_batch(*timeframe, vec![symbol.clone()], window_spec).await {
                                eprintln!("Submit batch failed: {}", e);
                            }
                        }
                    }
                }
                Err(e) => eprintln!("DB Error: {}", e),
                _ => {}
            }
        }
    }

    println!("Jobs submitted. Waiting for processing...");
    // Даем время на обработку задач (можно улучшить через счетчик задач)
    tokio::time::sleep(Duration::from_secs(300)).await;
    
    Ok(())
}

/// Helper: Ждет появления активных пар в БД
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

/// Helper: Ждет появления данных в таблице свечей
async fn wait_for_table_data(pool: &PgPool, table_name: &str) -> bool {
    let start = std::time::Instant::now();
    loop {
        // Проверяем наличие хотя бы одной записи
        let query = format!("SELECT 1 FROM {} LIMIT 1", table_name);
        match sqlx::query(&query).fetch_optional(pool).await {
            Ok(Some(_)) => return true,
            Ok(None) => {
                if start.elapsed() > Duration::from_secs(300) {
                    return false;
                }
                eprintln!("Waiting for data in {} (elapsed: {:.0}s)...", table_name, start.elapsed().as_secs_f64());
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            Err(e) => {
                // Если таблицы еще нет (например, миграция не прошла), тоже ждем
                eprintln!("Error checking {}: {}. Retrying...", table_name, e);
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
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