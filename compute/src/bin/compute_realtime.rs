use std::sync::Arc;
use dotenvy::dotenv;
use database_lib;
use common::{Symbol, Timeframe, MessageBus};
use sqlx::PgPool;
use compute_lib::{ComputeJob, JobScheduler, ComputeBackendManager, ComputeBackendType, CandleWindowFetcher, ComputeConfig, IndicatorPersistor, RawSignalPersistor, RawSignalProcessor, ResultProcessor};
use raw_signals::thresholds::SignalConfig;
use compute_lib::predictors::config::PredictorsConfig;
use compute_lib::predictors::pipeline::{PredictorsPipeline, FeatureSnapshot};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());
    database_lib::init_db::initialize_database(&db_url).await?;

    // фиксируем CPU backend
    let backend_type = ComputeBackendType::Cpu;
    let compute_backend_manager = ComputeBackendManager::new(backend_type);
    let compute_backend = compute_backend_manager.get_backend();

    // load config
    let config = ComputeConfig {
        batch_size: 100,
        max_concurrent_jobs: 4,
        use_cuda: false, // CPU only for realtime
        cuda_device_id: None,
    };

    let db_pool = sqlx::PgPool::connect(&db_url).await?;
    let candle_fetcher = Arc::new(CandleWindowFetcher::new(db_pool.clone()));

    // Initialize DB Persistor (REALTIME Mode)
    std::env::set_var("DB_PERSIST_MODE", "realtime");
    std::env::set_var("DB_PERSIST_CHUNK_SIZE_REALTIME", "2000"); // Small chunks for realtime
    std::env::set_var("DB_PERSIST_FLUSH_MS", "50"); // Faster flushing for realtime

    let bulk_persistor = database_lib::bulk_persistor::BulkPersistor::new_from_env_mode(
        database_lib::bulk_persistor::PersistMode::Realtime
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

    // Predictors Pipeline
    let message_bus = MessageBus::new_from_env()?;
    let (_shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel::<bool>(1);
    
    let pred_config = PredictorsConfig {
        enabled: true,
        horizon_bars: 10,
        min_store_score: 0.60,
        min_final_score: 0.70,
        prefer_ml: true,
        max_levels_per_side: 2,
        use_cuda: cfg!(feature = "cuda"), // Can use GPU for inference even in RT
        use_gpu_history: false,
        use_gpu_realtime: false, // Usually CPU is faster for single inferences
        model_path_price: "models/price_v1_tf{tf}.ubj".to_string(),
        model_path_levels: "models/levels_v1_tf{tf}.ubj".to_string(),
        ml_batch_size: 64,
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

    // Spawn Result Processor (Connecting the dots!)
    let result_processor = ResultProcessor::new(
        Arc::new(indicator_persistor),
        Arc::new(raw_signal_persistor),
        raw_processor,
        feature_tx,
    );
    
    tokio::spawn(async move {
        result_processor.run(result_receiver).await;
    });

    // realtime consumer loop (copy from your main.rs)
    run_realtime_consumer(job_scheduler, db_pool).await?;

    Ok(())
}

async fn run_realtime_consumer(
    job_scheduler: Arc<JobScheduler>,
    _db_pool: PgPool,
) -> Result<(), Box<dyn std::error::Error>> {
    use rdkafka::{
        config::ClientConfig,
        consumer::{CommitMode, Consumer, StreamConsumer, DefaultConsumerContext},
        message::Message,
        types::RDKafkaErrorCode,
        error::KafkaError,
    };
    use std::time::Duration;

    let brokers = std::env::var("KAFKA_BROKERS").unwrap_or_else(|_| "127.0.0.1:19092".to_string());
    let topic = std::env::var("KAFKA_CANDLES_CLOSE_TOPIC").unwrap_or_else(|_| "candles.close".to_string());
    let group_id = std::env::var("COMPUTE_CONSUMER_GROUP").unwrap_or_else(|_| "compute_group".to_string());

    let mut backoff = Duration::from_secs(1);
    const MAX_BACKOFF: Duration = Duration::from_secs(64);

    loop {
        let consumer: StreamConsumer<DefaultConsumerContext> = match ClientConfig::new()
            .set("bootstrap.servers", &brokers)
            .set("group.id", &group_id)
            .set("enable.partition.eof", "false")
            .set("session.timeout.ms", "6000")
            .set("enable.auto.commit", "false") // Disable auto-commit
            .set("auto.offset.reset", "latest")
            .create() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Failed to create Kafka consumer: {}. Retrying in {:?}...", e, backoff);
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(MAX_BACKOFF);
                    continue;
                }
            };

        match consumer.subscribe(&[&topic]) {
            Ok(_) => {
                println!("Started Kafka consumer, listening on topic: {}", topic);
                backoff = Duration::from_secs(1); // Reset backoff on successful connection

                loop {
                    match consumer.recv().await {
                        Err(e) => {
                            eprintln!("Kafka consumer error: {}", e);
                            if let KafkaError::MessageConsumption(RDKafkaErrorCode::AllBrokersDown) = e {
                                eprintln!("All brokers down. Breaking to reconnect...");
                                tokio::time::sleep(Duration::from_secs(5)).await; // Wait before reconnecting
                                break; // Break from inner loop to recreate consumer
                            }
                        }
                        Ok(msg) => {
                            match msg.payload() {
                                None => {
                                    println!("Received message with no payload");
                                }
                                Some(payload) => {
                                    match serde_json::from_slice::<CandleCloseEvent>(payload) {
                                        Ok(event) => {
                                            println!("Received candle close event: {} {} at {}", event.symbol, event.timeframe, event.close_time);

                                            let timeframe = match event.timeframe.as_str() {
                                                "1m" => Timeframe::M1,
                                                "5m" => Timeframe::M5,
                                                "15m" => Timeframe::M15,
                                                "1h" => Timeframe::H1,
                                                "4h" => Timeframe::H4,
                                                "1d" => Timeframe::D1,
                                                _ => {
                                                    eprintln!("Unknown timeframe: {}", event.timeframe);
                                                    continue;
                                                }
                                            };

                                            let tf_ms = timeframe.to_minutes() as i64 * 60_000;
                                            let window_end = event.close_time;
                                            let window_start = window_end - (120 as i64) * tf_ms; // Lookback 120 candles for indicators

                                            let job = ComputeJob {
                                                symbol: Symbol::from(event.symbol),
                                                timeframe,
                                                window_start,
                                                window_end,
                                                indicators: vec![
                                                    "adx".to_string(), "atr".to_string(), "bb".to_string(),
                                                    "cci".to_string(), "ema".to_string(), "macd".to_string(),
                                                    "obv".to_string(), "rsi".to_string(), "sma".to_string(),
                                                    "stoch".to_string(), "vwap".to_string(), "williams".to_string(),
                                                    "alligator".to_string(), "sr_levels".to_string(),
                                                ],
                                                candle_window: None,
                                                is_realtime: true, // Guarantee realtime = true
                                            };

                                            if let Err(e) = job_scheduler.process_single_job(job).await {
                                                eprintln!("Error processing job: {}", e);
                                                // Do not commit if processing fails
                                            } else {
                                                // Manually commit the offset after successful processing
                                                if let Err(e) = consumer.commit_message(&msg, CommitMode::Async) {
                                                    eprintln!("Failed to commit Kafka offset: {}", e);
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("Failed to deserialize candle close event: {}", e);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            Err(e) => {
                eprintln!("Failed to subscribe to topic '{}': {}. Retrying in {:?}...", topic, e, backoff);
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }
        };
    }
}

// Define the structure for candle close events
#[derive(serde::Deserialize, Debug)]
struct CandleCloseEvent {
    symbol: String,
    timeframe: String,  // This will need to be parsed to Timeframe
    close_time: i64,
}