use anyhow::Result;
use rdkafka::{
    config::ClientConfig,
    producer::{FutureProducer, FutureRecord},
};
use std::time::Duration;
use tracing::{error};

#[derive(Debug, Clone)]
pub struct RedpandaConfig {
    pub brokers: String,
    pub topic: String,
    pub group_id: String,
}

impl Default for RedpandaConfig {
    fn default() -> Self {
        RedpandaConfig {
            brokers: std::env::var("KAFKA_BROKERS").unwrap_or_else(|_| "127.0.0.1:19092".to_string()),
            topic: std::env::var("KAFKA_TOPIC_DEFAULT").unwrap_or_else(|_| "default_topic".to_string()),
            group_id: std::env::var("WRITER_GROUP").unwrap_or_else(|_| "default_group".to_string()),
        }
    }
}

pub struct RedpandaConnection {
    config: RedpandaConfig,
    producer: Option<FutureProducer>,
}

impl RedpandaConnection {
    pub fn new(config: RedpandaConfig) -> Result<Self> {
        Ok(RedpandaConnection { config, producer: None })
    }

    pub async fn new_from_env() -> Result<Self> {
        Self::new(RedpandaConfig::default())
    }

    pub async fn connect_producer(&mut self) -> Result<()> {
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", &self.config.brokers)
            .set("message.timeout.ms", "5000")
            .create()?;
        self.producer = Some(producer);
        Ok(())
    }

    pub async fn send_message(&self, topic: &str, key: &str, payload: impl AsRef<[u8]>) -> Result<()> {
        if let Some(ref producer) = self.producer {
            // В новых версиях rdkafka возвращает Result<Delivery, ...>
            let delivery = producer
                .send(
                    FutureRecord::to(topic)
                        .key(key)
                        .payload(payload.as_ref()),
                    Duration::from_secs(1),
                )
                .await;

            match delivery {
                Ok(_) => Ok(()),
                Err((e, _)) => {
                    error!("Failed to deliver message: {}", e);
                    anyhow::bail!(e)
                }
            }
        } else {
            anyhow::bail!("Producer not initialized")
        }
    }

    pub async fn ping(&self) -> Result<()> {
        self.send_message(&self.config.topic, "ping", b"test").await
    }
}

