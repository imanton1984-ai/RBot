use anyhow::{Context, Result};
use common::{load_config, AppConfig, Candle, TimeFrame};
use connections_lib::redpanda::RedpandaConfig;
use futures::StreamExt;
use rdkafka::{
    config::ClientConfig,
    consumer::{Consumer, StreamConsumer},
    message::Message,
};
use serde_json;
use sqlx::PgPool;
use std::sync::Arc;
use tokio::sync::Notify;
use tracing::{error, info};

pub struct CandleWriter {
    pool: PgPool,
    consumer: StreamConsumer,
    shutdown_notify: Arc<Notify>,
}

impl CandleWriter {
    pub async fn new() -> Result<Self> {
        let cfg: AppConfig = load_config().context("load_config() failed")?;
        let db_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| cfg.database.url());
        let pool = PgPool::connect(&db_url).await.context("connect DB failed")?;

        // Setup Redpanda consumer
        let redpanda_cfg = RedpandaConfig {
            brokers: cfg.rust_bot.redpanda_brokers.join(","),
            topic: cfg.rust_bot.topic_candles_close.clone(),
            group_id: "candle_writer_group".to_string(),
        };

        let consumer: StreamConsumer = ClientConfig::new()
            .set("bootstrap.servers", &redpanda_cfg.brokers)
            .set("group.id", &redpanda_cfg.group_id)
            .set("enable.partition.eof", "false")
            .set("session.timeout.ms", "6000")
            .set("auto.offset.reset", "earliest")
            .create()
            .context("Consumer creation failed")?;

        consumer.subscribe(&[&redpanda_cfg.topic]).context("Can't subscribe to topic")?;

        let shutdown_notify = Arc::new(Notify::new());

        Ok(CandleWriter {
            pool,
            consumer,
            shutdown_notify,
        })
    }

    pub async fn run(&self) -> Result<()> {
        info!("Starting candle writer...");

        let mut message_stream = self.consumer.stream();

        while let Some(message_result) = message_stream.next().await {
            match message_result {
                Ok(borrowed_message) => {
                    let payload: &[u8] = borrowed_message.payload().unwrap_or(b"");

                    match serde_json::from_slice::<Candle>(payload) {
                        Ok(candle) => {
                            // Determine the appropriate table based on the timeframe
                            // For now, we'll need to determine the timeframe from the time_ms
                            // A more robust solution would include the timeframe in the message

                            // Calculate the timeframe based on the time difference between candles
                            // For now, we'll just try to insert into the appropriate table based on common timeframes
                            if let Err(e) = self.write_candle_to_correct_table(&candle).await {
                                error!("Failed to write candle to database: {}", e);
                            }
                        }
                        Err(e) => {
                            error!("Failed to deserialize candle from message: {}", e);
                        }
                    }

                    // Commit message offset after processing
                    self.consumer.store_offset(
                        borrowed_message.topic(),
                        borrowed_message.partition(),
                        borrowed_message.offset()
                    ).context("Failed to store offset")?;
                }
                Err(e) => {
                    error!("Kafka error: {}", e);
                }
            }
        }

        Ok(())
    }

    async fn write_candle_to_correct_table(&self, candle: &Candle) -> Result<()> {
        // For now, we'll need to determine the timeframe from the time_ms
        // A better approach would be to include the timeframe in the message
        // For this implementation, we'll try to determine the most likely timeframe
        // based on the time interval of the candle

        // Since we don't have previous candle data here to calculate the interval,
        // we'll use a heuristic approach or assume it comes from the right topic

        // For now, let's just insert into the 1m table as a default
        // In a real implementation, we'd have the timeframe in the message or topic
        self.insert_single_candle(candle).await
    }

    async fn insert_single_candle(&self, candle: &Candle) -> Result<()> {
        // Default to 1m timeframe table - this should be determined from the message
        sqlx::query(
            "INSERT INTO market.candles_1m (time_ms, symbol_id, open, high, low, close, volume)
             VALUES ($1, $2, $3, $4, $5, $6, $7)
             ON CONFLICT (symbol_id, time) DO NOTHING"
        )
        .bind(candle.time_ms)
        .bind(candle.symbol_id)
        .bind(candle.open)
        .bind(candle.high)
        .bind(candle.low)
        .bind(candle.close)
        .bind(candle.volume)
        .execute(&self.pool)
        .await
        .context("Failed to insert candle")?;

        Ok(())
    }

    pub async fn start_listening_by_timeframe(&self) -> Result<()> {
        info!("Starting candle writer with timeframe-specific topics...");

        // Get configuration to determine which timeframes to listen to
        let cfg: AppConfig = load_config().context("load_config() failed")?;

        // Subscribe to multiple topics based on timeframes
        let mut topics = Vec::new();
        for timeframe in TimeFrame::all_timeframes() {
            let topic = match timeframe {
                TimeFrame::M1 => std::env::var("TOPIC_CANDLES_M1").unwrap_or_else(|_| format!("candles_{}", timeframe.as_str())),
                TimeFrame::M5 => std::env::var("TOPIC_CANDLES_M5").unwrap_or_else(|_| format!("candles_{}", timeframe.as_str())),
                TimeFrame::M15 => std::env::var("TOPIC_CANDLES_M15").unwrap_or_else(|_| format!("candles_{}", timeframe.as_str())),
                TimeFrame::H1 => std::env::var("TOPIC_CANDLES_H1").unwrap_or_else(|_| format!("candles_{}", timeframe.as_str())),
                TimeFrame::H4 => std::env::var("TOPIC_CANDLES_H4").unwrap_or_else(|_| format!("candles_{}", timeframe.as_str())),
                TimeFrame::D1 => std::env::var("TOPIC_CANDLES_D1").unwrap_or_else(|_| format!("candles_{}", timeframe.as_str())),
                _ => format!("candles_{}", timeframe.as_str()),
            };
            topics.push(topic);
        }

        // Subscribe to all timeframe-specific topics
        let topic_refs: Vec<&str> = topics.iter().map(|s| s.as_str()).collect();
        self.consumer.subscribe(&topic_refs).context("Can't subscribe to topics")?;

        info!("Subscribed to topics: {:?}", topic_refs);

        let mut message_stream = self.consumer.stream();

        while let Some(message_result) = message_stream.next().await {
            match message_result {
                Ok(borrowed_message) => {
                    let payload: &[u8] = borrowed_message.payload().unwrap_or(b"");
                    let topic = borrowed_message.topic();

                    match serde_json::from_slice::<Candle>(payload) {
                        Ok(candle) => {
                            // Determine the table based on the topic name
                            let table_name = self.get_table_name_from_topic(topic);
                            if let Err(e) = self.insert_candle_with_timeframe(&candle, &table_name).await {
                                error!("Failed to write candle to table {}: {}", table_name, e);
                            }
                        }
                        Err(e) => {
                            error!("Failed to deserialize candle from message: {}", e);
                        }
                    }

                    // Commit message offset after processing
                    self.consumer.store_offset(
                        borrowed_message.topic(),
                        borrowed_message.partition(),
                        borrowed_message.offset()
                    ).context("Failed to store offset")?;
                }
                Err(e) => {
                    error!("Kafka error: {}", e);
                }
            }
        }

        Ok(())
    }

    fn get_table_name_from_topic(&self, topic: &str) -> String {
        // Map topic names to table names
        if topic.contains("1m") {
            "market.candles_1m".to_string()
        } else if topic.contains("5m") {
            "market.candles_5m".to_string()
        } else if topic.contains("15m") {
            "market.candles_15m".to_string()
        } else if topic.contains("1h") {
            "market.candles_1h".to_string()
        } else if topic.contains("4h") {
            "market.candles_4h".to_string()
        } else if topic.contains("1d") {
            "market.candles_1d".to_string()
        } else {
            // Default to 1m if we can't determine the timeframe
            "market.candles_1m".to_string()
        }
    }

    async fn insert_candle_with_timeframe(&self, candle: &Candle, table_name: &str) -> Result<()> {
        let query = format!(
            "INSERT INTO {} (time_ms, symbol_id, open, high, low, close, volume)
             VALUES ($1, $2, $3, $4, $5, $6, $7)
             ON CONFLICT (symbol_id, time) DO NOTHING",
            table_name
        );

        sqlx::query(&query)
            .bind(candle.time_ms)
            .bind(candle.symbol_id)
            .bind(candle.open)
            .bind(candle.high)
            .bind(candle.low)
            .bind(candle.close)
            .bind(candle.volume)
            .execute(&self.pool)
            .await
            .context("Failed to insert candle")?;

        Ok(())
    }
}