use std::sync::Arc;
use common::{Symbol, Timeframe};
use compute_lib::*;
use sqlx::Row;

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

    // Initialize database connection pool
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());
    let db_pool = sqlx::PgPool::connect(&db_url).await?;

    // Initialize candle window fetcher
    let candle_fetcher = Arc::new(CandleWindowFetcher::new(db_pool.clone()));

    // Initialize job scheduler
    let (job_scheduler, mut result_receiver) = JobScheduler::new(
        compute_backend,
        config.clone(),
        candle_fetcher.clone(),
    );
    let job_scheduler = Arc::new(job_scheduler);

    // Initialize indicator persistor
    let (persistor, _persist_sender) = IndicatorPersistor::new(
        db_pool.clone(), // Clone the pool to use in persistor
        1000, // batch size
        5000, // flush every 5 seconds
    );
    let persistor = Arc::new(persistor);

    // Initialize bootstrap coordinator
    let feature_store = Arc::new(compute_indicators::FeatureStore::new());
    let bootstrap_coordinator = Arc::new(BootstrapCoordinator::new(
        feature_store,
        job_scheduler.clone(),
        1000, // required lookback
        100,  // warmup bars
    ));

    // Trigger bootstrap computation for historical data
    let bootstrap_coordinator_clone = bootstrap_coordinator.clone();
    let db_pool_clone = db_pool.clone();
    
    tokio::spawn(async move {
        // IMPROVED: Wait loop logic
        println!("Waiting for historical data to be ingested...");
        let max_retries = 30; // Try for ~60 seconds
        let mut data_found = false;
        
        for i in 0..max_retries {
            // Check if we have any candles in M1 table
            let row = sqlx::query("SELECT 1 FROM market.candles_1m LIMIT 1")
                .fetch_optional(&db_pool_clone)
                .await;
                
            if let Ok(Some(_)) = row {
                println!("Data detected in DB. Starting historical compute...");
                data_found = true;
                // Give a little more grace time for bulk copy to fully commit/index
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                break;
            }
            
            if i % 5 == 0 {
                println!("Waiting for ingestor... (attempt {}/{})", i+1, max_retries);
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        }

        if data_found {
            if let Err(e) = trigger_historical_compute(&bootstrap_coordinator_clone, &db_pool_clone).await {
                eprintln!("Error triggering historical compute: {}", e);
            }
        } else {
            eprintln!("TIMEOUT: No historical data found after waiting. Historical compute skipped.");
        }
    });

    // Spawn persistence loop
    let persistor_clone = persistor.clone();
    tokio::spawn(async move {
        if let Err(e) = persistor_clone.start_persistence_loop().await {
            eprintln!("Persistence loop error: {}", e);
        }
    });

    // Initialize RawSignal persistor
    let (raw_signal_persistor, raw_signal_sender) = RawSignalPersistor::new(
        db_pool.clone(),
        1000,
        5000,
    );
    let raw_signal_persistor = Arc::new(raw_signal_persistor);

    // Initialize RawSignal processor
    let raw_signal_processor = Arc::new(RawSignalProcessor::new(Default::default()));

    // Spawn RawSignal persistence loop
    let raw_signal_persistor_clone = raw_signal_persistor.clone();
    tokio::spawn(async move {
        if let Err(e) = raw_signal_persistor_clone.start_persistence_loop().await {
            eprintln!("RawSignal persistence loop error: {}", e);
        }
    });

    // Spawn result processor to handle computed indicators and raw signals
    let persistor_clone2 = persistor.clone();
    let raw_signal_processor_clone = raw_signal_processor.clone();
    tokio::spawn(async move {
        while let Some(feature_window) = result_receiver.recv().await {
            println!("Processing {} computed features", feature_window.features.len());

            // Handle indicators
            let mut records = Vec::new();
            for result in &feature_window.features {
                for (indicator_name, value) in &result.features {
                    records.push(IndicatorRecord {
                        symbol: result.symbol.clone(),
                        timeframe: result.timeframe,
                        timestamp: result.timestamp,
                        indicator_name: indicator_name.clone(),
                        value: value.clone(),
                    });
                }
            }

            if !records.is_empty() {
                if let Err(e) = persistor_clone2.queue_records(records).await {
                    eprintln!("Error queuing records for persistence: {}", e);
                }
            }
            
            // Handle raw signals
            let raw_signals = raw_signal_processor_clone.process_feature_window(&feature_window);
            for signal in raw_signals {
                if let Err(e) = raw_signal_sender.send(signal) {
                    eprintln!("Error sending raw signal for processing: {}", e);
                }
            }
        }
    });

    // Start Kafka consumer to listen for candle close events
    // Wait for historical compute to complete before starting real-time processing
    let bootstrap_coordinator_clone2 = bootstrap_coordinator.clone();
    let kafka_job_scheduler = job_scheduler.clone();

    tokio::spawn(async move {
        // Wait for historical data to be processed (wait for at least some symbols to be ready)
        println!("Waiting for historical data processing to complete before starting real-time processing...");
        if let Err(e) = bootstrap_coordinator_clone2.wait_for_readiness_threshold(5).await { // Wait for at least 5 symbols
            eprintln!("Error waiting for historical data readiness: {}", e);
        }
        println!("Historical data processing completed. Starting real-time processing...");

        if let Err(e) = start_kafka_consumer(kafka_job_scheduler).await {
            eprintln!("Kafka consumer error: {}", e);
        }
    });

    // Fetch symbols from database and submit jobs
    println!("Fetching symbols from database and starting calculations...");

    // Fetch active symbols from the database
    let symbols = loop {
        let symbols_res = fetch_active_symbols_from_db(&db_pool).await;
        match symbols_res {
            Ok(s) if !s.is_empty() => {
                println!("Fetched {} symbols from database", s.len());
                break s;
            }
            _ => {
                println!("Waiting for market.pairs to be populated...");
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        }
    };

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

// Function to fetch active symbols from the database
async fn fetch_active_symbols_from_db(
    db_pool: &sqlx::PgPool,
) -> Result<Vec<Symbol>, Box<dyn std::error::Error + Send + Sync>> {
    // First check if the table exists
    let table_exists_result = sqlx::query_scalar::<_, bool>(
        r#"SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'market' AND table_name = 'pairs'
        )"#
    )
    .fetch_one(db_pool)
    .await;

    match table_exists_result {
        Ok(table_exists) => {
            if !table_exists {
                return Ok(Vec::new()); // Return empty vector if table doesn't exist
            }
        },
        Err(_) => {
            return Ok(Vec::new()); // Return empty vector if table check fails
        }
    }

    // If table exists, try to fetch symbols
    let rows_result = sqlx::query(
        r#"SELECT symbol FROM market.pairs WHERE is_active = true"#
    )
    .fetch_all(db_pool)
    .await;

    match rows_result {
        Ok(rows) => {
            let symbols: Vec<Symbol> = rows
                .into_iter()
                .map(|row| {
                    let symbol: String = row.get("symbol");
                    Symbol::from(symbol)
                })
                .collect();
            Ok(symbols)
        },
        Err(_) => Ok(Vec::new()), // Return empty vector if fetch fails
    }
}

// Function to trigger historical compute for existing data in the database
async fn trigger_historical_compute(
    bootstrap_coordinator: &BootstrapCoordinator,
    db_pool: &sqlx::PgPool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use common::Timeframe;

    println!("Triggering historical compute for existing data...");

    // Define the timeframes we want to process
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

        // Get symbols that have data for this timeframe
        let table_name = format!("market.candles_{}", timeframe.as_str());

        // Check if table exists and has data - extract just the table name part after the dot
        let table_name_part = &table_name[7..]; // Remove "market." prefix to get just the table name
        let table_exists = sqlx::query_scalar::<_, bool>(
            r#"SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'market' AND table_name = $1
            )"#
        )
        .bind(table_name_part)
        .fetch_one(db_pool)
        .await
        .unwrap_or(false);

        if !table_exists {
            println!("Table {} does not exist, skipping", table_name);
            continue;
        }

        // Get symbols with data in this timeframe
        let symbols_query = format!(
            "SELECT DISTINCT p.symbol
             FROM {} c
             JOIN market.pairs p ON c.symbol_id = p.symbol_id
             WHERE p.is_active = true
             LIMIT 10", // Limit to avoid overwhelming the system
            table_name
        );

        let symbol_rows = sqlx::query(&symbols_query)
            .fetch_all(db_pool)
            .await
            .unwrap_or_default();

        let symbols: Vec<Symbol> = symbol_rows
            .into_iter()
            .map(|row| {
                let symbol: String = row.get("symbol");
                Symbol::from(symbol)
            })
            .collect();

        if symbols.is_empty() {
            println!("No symbols found for timeframe: {:?}", timeframe);
            continue;
        }

        println!("Found {} symbols for timeframe {:?}, submitting compute jobs...", symbols.len(), timeframe);

        // Submit compute jobs for these symbols by updating history status to trigger bootstrap
        for symbol in &symbols {
            // Get the latest timestamp and count of candles for this symbol/timeframe
            let stats_query = format!(
                "SELECT MAX(time_ms) as latest_time, COUNT(*) as total_candles
                 FROM {} c
                 JOIN market.pairs p ON c.symbol_id = p.symbol_id
                 WHERE p.symbol = $1",
                table_name
            );

            if let Ok(row) = sqlx::query(&stats_query)
                .bind(symbol.as_str())
                .fetch_optional(db_pool)
                .await
            {
                if let Some(row) = row {
                    let latest_time: Option<i64> = row.get("latest_time");
                    let total_candles: i64 = row.get("total_candles");

                    let latest_time = latest_time.unwrap_or(0);
                    let bars_count = total_candles as usize;

                    // Update history status to mark this symbol/timeframe as ready
                    bootstrap_coordinator.update_history_status(
                        symbol.clone(),
                        *timeframe,
                        latest_time,
                        bars_count // Use actual count of bars
                    ).await;
                }
            }
        }

        // Now trigger bootstrap compute for all ready symbols
        if let Err(e) = bootstrap_coordinator.start_bootstrap_compute().await {
            eprintln!("Error submitting batch job for {:?}: {}", timeframe, e);
        } else {
            println!("Successfully submitted batch job for {:?} with {} symbols", timeframe, symbols.len());
        }
    }

    Ok(())
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

    let result = consumer.subscribe(&[&topic]);
    match result {
        Ok(_) => {
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
                                        // Fetch a window of historical data to calculate indicators properly
                                        let window_lookback_minutes = 1000; // Look back 1000 minutes
                                        let window_end = event.close_time;
                                        let window_start = event.close_time - (window_lookback_minutes * 60000); // Convert minutes to milliseconds

                                        let job = ComputeJob {
                                            symbol: Symbol::from(event.symbol),
                                            timeframe,
                                            window_start,
                                            window_end,
                                            indicators: vec![
                                                "adx".to_string(),
                                                "atr".to_string(),
                                                "bb".to_string(),
                                                "cci".to_string(),
                                                "ema".to_string(),
                                                "macd".to_string(),
                                                "obv".to_string(),
                                                "rsi".to_string(),
                                                "sma".to_string(),
                                                "stoch".to_string(),
                                                "vwap".to_string(),
                                                "williams".to_string(),
                                                "alligator".to_string(),
                                                "sr_levels".to_string(),
                                            ],
                                            candle_window: None, // Will be filled in by process_single_job
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
        },
        Err(e) => {
            eprintln!("Failed to subscribe to topic '{}': {}. This may happen if the topic doesn't exist yet. The service will continue running but won't process candle close events from Kafka.", topic, e);
            // Just return to continue with other functionality
            return Ok(());
        }
    };
}
