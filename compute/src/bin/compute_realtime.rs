use std::sync::Arc;
use dotenvy::dotenv;
use database_lib;
use common::{Symbol, Timeframe, MessageBus, CandleCloseEvent};
use sqlx::PgPool;
use compute_lib::{ComputeJob, JobScheduler, ComputeBackendManager, ComputeBackendType, CandleWindowFetcher, ComputeConfig, IndicatorPersistor, RawSignalPersistor, RawSignalProcessor, ResultProcessor};
use raw_signals::thresholds::SignalConfig;
use compute_lib::predictors::config::PredictorsConfig;
use compute_lib::predictors::pipeline::{PredictorsPipeline, FeatureSnapshot, TradeSignalInput};
use compute_lib::scoring::trade_signal_processor::TradeSignalStage;
use compute_lib::scoring::market_params_calculator::MarketParamsCalculator;

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let active_strategy = get_active_strategy();
    tracing::info!("compute_realtime: ACTIVE_STRATEGY = {}", active_strategy);

    let run_predictors = should_run_predictors();
    if run_predictors {
        tracing::info!("compute_realtime: Running predictors + trade_signals pipeline (level strategy)");
    } else {
        tracing::info!("compute_realtime: Skipping predictors pipeline (strategy: {})", active_strategy);
    }

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

    let (feature_tx, feature_rx) = tokio::sync::mpsc::unbounded_channel::<FeatureSnapshot>();

    if run_predictors {
        // Level strategy: use predictors pipeline
        let message_bus = MessageBus::new_from_env()?;
        let (_shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel::<bool>(1);

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
            use_cuda: cfg!(feature = "cuda"),
            use_gpu_history: false,
            use_gpu_realtime: false,
            model_path_price: "models/price_v1_tf{tf}.ubj".to_string(),
            model_path_levels: "models/levels_v1_tf{tf}.ubj".to_string(),
            ml_batch_size: 64,
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
    } else {
        // Super Entry strategy: use Super Entry stage for realtime signals
        tracing::info!("compute_realtime: Setting up Super Entry stage for realtime signals");
        if let Some(_handle) = compute_lib::super_entry_stage::setup_super_entry_stage(
            &db_pool,
            feature_rx,
            config.use_cuda,
        ).await {
            tracing::info!("compute_realtime: ✅ Super Entry realtime stage spawned");
        } else {
            tracing::warn!("compute_realtime: Super Entry stage not available — signals will not be generated in realtime");
        }
    }

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
                                            let lookback: i64 = std::env::var("BACKFILL_CANDLES")
                                                .ok().and_then(|v| v.parse().ok()).unwrap_or(500);
                                            let window_start = window_end - lookback * tf_ms; // Lookback from BACKFILL_CANDLES env var

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

// CandleCloseEvent is imported from common::CandleCloseEvent