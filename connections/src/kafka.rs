use anyhow::Result;
use rdkafka::{
    config::ClientConfig,
    consumer::{StreamConsumer},
    producer::{FutureProducer, FutureRecord},
};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{error};

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
        
        // Map the tuple error to a simple KafkaError or string for anyhow 
        let _ = self.producer
            .send(record, Duration::from_secs(3))
            .await
            .map_err(|(e, _)| anyhow::anyhow!("Kafka send error: {:?}", e))?; 
            
        Ok(())
    }

    pub async fn health_check(&self) -> Result<bool> {
        let test_key = "health_check";
        let test_payload = "test";

        match timeout(Duration::from_secs(5), self.producer.send(
            FutureRecord::to(&self.topic).key(test_key).payload(test_payload),
            Duration::from_secs(3),
        )).await {
            Ok(Ok((_partition, _offset))) => Ok(true),
            Ok(Err((e, _))) => { error!("kafka health_check send failed: {:?}", e); Ok(false) }
            Err(_) => { error!("kafka health_check timeout"); Ok(false) }
        }
    }

    pub fn brokers(&self) -> &str { &self.brokers }
    pub fn topic(&self) -> &str { &self.topic }
}
