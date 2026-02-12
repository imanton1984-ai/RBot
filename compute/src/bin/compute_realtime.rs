use std::sync::Arc;
use dotenvy::dotenv;
use compute_lib::*;
use database_lib;
use common::{Symbol, Timeframe};
use tokio::sync::broadcast;
use rdkafka::{config::ClientConfig, consumer::{CommitMode, Consumer, StreamConsumer, DefaultConsumerContext}, message::Message, types::RDKafkaErrorCode, error::KafkaError};
use sqlx::PgPool;
use crate::compute_lib::{ComputeJob, FeatureWindow};

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

    let (job_scheduler, _result_receiver) = JobScheduler::new(
        compute_backend,
        config.clone(),
        candle_fetcher.clone(),
    );
    let job_scheduler = Arc::new(job_scheduler);

    // Set environment for realtime mode
    std::env::set_var("DB_PERSIST_MODE", "realtime");
    std::env::set_var("DB_PERSIST_CHUNK_SIZE_REALTIME", "2000"); // Small chunks for realtime
    std::env::set_var("DB_PERSIST_FLUSH_MS", "50"); // Faster flushing for realtime

    // realtime consumer loop (copy from your main.rs)
    run_realtime_consumer(job_scheduler, db_pool).await?;

    Ok(())
}

async fn run_realtime_consumer(
    job_scheduler: Arc<JobScheduler>,
    db_pool: PgPool,
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