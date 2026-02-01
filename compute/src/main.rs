use std::sync::Arc;
use common::{Symbol, Timeframe};
use compute_lib::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Compute Service...");

    // Initialize configuration
    let config = ComputeConfig {
        batch_size: 100,
        max_concurrent_jobs: 4,
        use_cuda: cfg!(feature = "cuda"), // Use CUDA if feature is enabled
        cuda_device_id: Some(0), // Use first CUDA device if available
    };

    // Determine backend type based on configuration
    let backend_type = if config.use_cuda {
        ComputeBackendType::Cuda
    } else {
        ComputeBackendType::Cpu
    };

    // Initialize compute backend
    let compute_backend_manager = ComputeBackendManager::new(backend_type);
    let compute_backend = compute_backend_manager.get_backend();

    // Initialize job scheduler
    let (job_scheduler, mut result_receiver) = JobScheduler::new(
        compute_backend,
        config.clone(),
    );
    let job_scheduler = Arc::new(job_scheduler);

    // Initialize candle window fetcher
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());
    let _candle_fetcher = Arc::new(CandleWindowFetcher::new(db_url.clone()));

    // Initialize database connection pool
    let db_pool = sqlx::PgPool::connect(&db_url).await?;

    // Initialize indicator persistor
    let (persistor, _persist_sender) = IndicatorPersistor::new(
        db_pool,
        1000, // batch size
        5000, // flush every 5 seconds
    );
    let persistor = Arc::new(persistor);

    // Initialize bootstrap coordinator
    let feature_store = Arc::new(compute_indicators::FeatureStore::new());
    let _bootstrap_coordinator = Arc::new(BootstrapCoordinator::new(
        feature_store,
        job_scheduler.clone(),
        1000, // required lookback
        100,  // warmup bars
    ));

    // Spawn persistence loop
    let persistor_clone = persistor.clone();
    tokio::spawn(async move {
        if let Err(e) = persistor_clone.start_persistence_loop().await {
            eprintln!("Persistence loop error: {}", e);
        }
    });

    // Spawn result processor to handle computed indicators
    let persistor_clone2 = persistor.clone();
    tokio::spawn(async move {
        while let Some(feature_window) = result_receiver.recv().await {
            println!("Processing {} computed features", feature_window.features.len());

            // Convert features to indicator records and queue for persistence
            let mut records = Vec::new();
            for result in &feature_window.features {
                for (indicator_name, value) in &result.features {
                    records.push(IndicatorRecord {
                        symbol: result.symbol.clone(),
                        timeframe: result.timeframe,
                        timestamp: result.timestamp,
                        indicator_name: indicator_name.clone(),
                        value: *value,
                    });
                }
            }

            if !records.is_empty() {
                if let Err(e) = persistor_clone2.queue_records(records).await {
                    eprintln!("Error queuing records for persistence: {}", e);
                }
            }
        }
    });

    // Start Kafka consumer to listen for candle close events
    let job_scheduler_clone = job_scheduler.clone();

    // Spawn Kafka consumer task
    tokio::spawn(async move {
        if let Err(e) = start_kafka_consumer(job_scheduler_clone).await {
            eprintln!("Kafka consumer error: {}", e);
        }
    });

    // Simulate getting historical data and starting calculations
    println!("Fetching historical data and starting calculations...");

    // Example: Submit a job for processing
    let symbols = vec![Symbol::from("BTCUSDT"), Symbol::from("ETHUSDT")];
    let window_spec = WindowSpec {
        length: 1000,
        warmup: 100,
    };

    // Submit batch jobs for computation
    if let Err(e) = job_scheduler.submit_batch(Timeframe::M1, symbols, window_spec).await {
        eprintln!("Error submitting batch job: {}", e);
    }

    // Keep the service running
    println!("Compute service started successfully. Processing indicators...");

    // In a real implementation, this would listen for incoming requests
    // For now, we'll just keep it running
    tokio::signal::ctrl_c().await?;
    println!("Received shutdown signal");

    Ok(())
}

// Define the structure for candle close events
#[derive(serde::Deserialize, Debug)]
struct CandleCloseEvent {
    symbol: String,
    timeframe: String,  // This will need to be parsed to Timeframe
    close_time: i64,
}

async fn start_kafka_consumer(
    job_scheduler: Arc<JobScheduler>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use rdkafka::{
        config::ClientConfig,
        consumer::{Consumer, StreamConsumer, DefaultConsumerContext},
        message::Message,
    };

    let brokers = std::env::var("KAFKA_BROKERS").unwrap_or_else(|_| "127.0.0.1:19092".to_string());
    let topic = std::env::var("KAFKA_CANDLES_CLOSE_TOPIC").unwrap_or_else(|_| "candles.close".to_string());
    let group_id = std::env::var("COMPUTE_CONSUMER_GROUP").unwrap_or_else(|_| "compute_group".to_string());

    let consumer: StreamConsumer<DefaultConsumerContext> = ClientConfig::new()
        .set("bootstrap.servers", &brokers)
        .set("group.id", &group_id)
        .set("enable.partition.eof", "false")
        .set("session.timeout.ms", "6000")
        .set("enable.auto.commit", "true")
        .set("auto.commit.interval.ms", "1000")
        .set("auto.offset.reset", "latest")
        .create()
        .expect("Consumer creation failed");

    consumer.subscribe(&[&topic]).expect("Can't subscribe to topic");

    println!("Started Kafka consumer, listening on topic: {}", topic);

    loop {
        match consumer.recv().await {
            Err(e) => {
                eprintln!("Kafka consumer error: {}", e);
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

                                // Convert timeframe string to Timeframe enum
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

                                // Create a compute job based on the received candle close event
                                let job = ComputeJob {
                                    symbol: Symbol::from(event.symbol),
                                    timeframe,
                                    window_end: event.close_time,
                                    window_start: event.close_time - (1000 * 60000), // 1000 minutes lookback
                                    indicators: vec!["rsi".to_string(), "sr_levels".to_string(), "trend".to_string()],
                                };

                                // Submit the job to the scheduler
                                if let Err(e) = job_scheduler.process_single_job(job).await {
                                    eprintln!("Error processing job: {}", e);
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
}
