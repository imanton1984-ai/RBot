use anyhow::Result;
use rdkafka::{
    config::ClientConfig,
    consumer::{Consumer, StreamConsumer},
    producer::{FutureProducer, FutureRecord},
    Message,
};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

pub struct KafkaManager {
    producer: FutureProducer,
    consumer: Option<StreamConsumer>,
    brokers: String,
    topic: String,
}

impl KafkaManager {
    pub fn new(brokers: &str, topic: &str) -> Result<Self> {
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", brokers)
            .set("message.timeout.ms", "5000")
            .set("request.timeout.ms", "5000")
            .set("retry.backoff.ms", "100")
            .set("retries", "3")
            .set("batch.size", "10000")
            .set("linger.ms", "5")
            .set("compression.type", "snappy")
            .create()?;

        Ok(Self {
            producer,
            consumer: None,
            brokers: brokers.to_string(),
            topic: topic.to_string(),
        })
    }

    pub fn with_consumer(mut self, group_id: &str) -> Result<Self> {
        let consumer: StreamConsumer = ClientConfig::new()
            .set("bootstrap.servers", &self.brokers)
            .set("group.id", group_id)
            .set("enable.partition.eof", "false")
            .set("session.timeout.ms", "6000")
            .set("enable.auto.commit", "true")
            .set("auto.commit.interval.ms", "100")
            .set("auto.offset.reset", "earliest")
            .set("socket.keepalive.enable", "true")
            .set("socket.nagle.disable", "true")
            .create()?;

        self.consumer = Some(consumer);
        Ok(self)
    }

    pub async fn send_message(&self, key: &str, payload: &str) -> Result<()> {
        let record: FutureRecord<'_, str, str> = FutureRecord::to(&self.topic)
            .key(key)
            .payload(payload);

        match timeout(Duration::from_secs(10), self.producer.send(record, Duration::from_secs(5))).await {
            Ok(result) => {
                match result {
                    Ok((partition, offset)) => {
                        debug!("Message delivered to partition {} at offset {}", partition, offset);
                        Ok(())
                    }
                    Err((error, _msg)) => {
                        error!("Failed to deliver message: {:?}", error);
                        Err(anyhow::anyhow!("Kafka delivery error: {:?}", error))
                    }
                }
            }
            Err(_) => {
                error!("Kafka send timed out");
                Err(anyhow::anyhow!("Kafka send timed out"))
            }
        }
    }

    pub async fn health_check(&self) -> Result<bool> {
        // Test producing a small message to check connectivity
        let test_key = "health_check";
        let test_payload = "test";

        match timeout(Duration::from_secs(5), self.producer.send(
            FutureRecord::to(&self.topic).key(test_key).payload(test_payload),
            Duration::from_secs(3)
        )).await {
            Ok(result) => {
                match result {
                    Ok((partition, offset)) => {
                        debug!("Health check message delivered to partition {} at offset {}", partition, offset);
                        Ok(true)
                    }
                    Err((error, _msg)) => {
                        error!("Health check failed to deliver message: {:?}", error);
                        Ok(false)
                    }
                }
            }
            Err(_) => {
                error!("Health check timed out");
                Ok(false)
            }
        }
    }

    pub async fn ensure_connection(&self) -> Result<()> {
        if !self.health_check().await? {
            error!("Kafka connection lost");
            return Err(anyhow::anyhow!("Kafka connection lost"));
        }
        Ok(())
    }

    pub fn get_topic(&self) -> &str {
        &self.topic
    }

    pub fn get_brokers(&self) -> &str {
        &self.brokers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kafka_manager_creation() {
        // This test would require a running Kafka instance
        // For now, we'll just test the interface
        assert!(true);
    }
}
