use anyhow::{anyhow, Result};
use futures::{Stream, StreamExt};
use rdkafka::{
    consumer::{Consumer, StreamConsumer},
    message::{OwnedMessage, Message},
    producer::{FutureProducer, FutureRecord},
    ClientConfig,
};
use serde::{de::DeserializeOwned, Serialize};
use std::{pin::Pin, sync::Arc, time::Duration};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Clone, Debug)]
pub struct MessageBusConfig {
    pub brokers: String,
    pub topic_prefix: String,

    // producer
    pub message_timeout_ms: u64,
    pub retry_backoff_ms: u64,
    pub max_in_flight: usize,
    pub flush_interval_ms: u64,
    pub batch_max_messages: usize,

    // consumer
    pub auto_offset_reset: String,
    pub enable_auto_commit: bool,
}

impl MessageBusConfig {
    pub fn from_env() -> Self {
        let brokers = std::env::var("KAFKA_BROKERS").unwrap_or_else(|_| "localhost:9092".to_string());
        let message_timeout_ms = std::env::var("KAFKA_MESSAGE_TIMEOUT_MS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(5_000);
        let retry_backoff_ms = std::env::var("KAFKA_RETRY_BACKOFF_MS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(100);
        let max_in_flight = std::env::var("KAFKA_MAX_IN_FLIGHT").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(10_000);
        let flush_interval_ms = std::env::var("KAFKA_FLUSH_INTERVAL_MS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(25);
        let batch_max_messages = std::env::var("KAFKA_BATCH_MAX_MESSAGES").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(5_000);

        let topic_prefix = std::env::var("KAFKA_TOPIC_PREFIX").unwrap_or_else(|_| "whitelist".to_string());

        let auto_offset_reset = std::env::var("KAFKA_AUTO_OFFSET_RESET").unwrap_or_else(|_| "latest".to_string());
        let enable_auto_commit = std::env::var("KAFKA_ENABLE_AUTO_COMMIT").ok()
            .map(|v| v == "true" || v == "1").unwrap_or(true);

        Self {
            brokers,
            topic_prefix,
            message_timeout_ms,
            retry_backoff_ms,
            max_in_flight,
            flush_interval_ms,
            batch_max_messages,
            auto_offset_reset,
            enable_auto_commit,
        }
    }
}

#[derive(Clone)]
pub struct MessageBus {
    config: Arc<MessageBusConfig>,
    publisher: MessagePublisher,
}

#[derive(Clone)]
struct MessagePublisher {
    tx: mpsc::Sender<OutboundMessage>,
}

struct OutboundMessage {
    topic: String,
    key: Vec<u8>,
    payload: Vec<u8>,
    partition: Option<i32>,
}

#[derive(Debug, Clone)]
pub struct IncomingMessage<T> {
    pub key: Vec<u8>,
    pub payload: T,
    pub topic: String,
    pub partition: i32,
    pub offset: i64,
    pub timestamp: Option<i64>,
}

impl MessageBus {
    pub fn new(config: MessageBusConfig) -> Result<Self> {
        let config = Arc::new(config);

        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", &config.brokers)
            .set("message.timeout.ms", &config.message_timeout_ms.to_string())
            .set("retry.backoff.ms", &config.retry_backoff_ms.to_string())
            // throughput knobs (без фанатизма)
            .set("linger.ms", "5")
            .set("batch.num.messages", "10000")
            .set("compression.type", "lz4")
            .create()
            .map_err(|e| anyhow!("Failed to create Kafka producer: {}", e))?;

        let (tx, mut rx) = mpsc::channel::<OutboundMessage>(config.max_in_flight);

        // publisher loop
        let cfg = config.clone();
        tokio::spawn(async move {
            let flush_every = Duration::from_millis(cfg.flush_interval_ms);
            let timeout = Duration::from_millis(cfg.message_timeout_ms);
            let mut tick = tokio::time::interval(flush_every);

            let mut batch: Vec<OutboundMessage> = Vec::with_capacity(cfg.batch_max_messages);

            loop {
                tokio::select! {
                    _ = tick.tick() => {
                        if !batch.is_empty() {
                            let _ = flush_batch(&producer, &mut batch, timeout).await;
                        }
                    }
                    msg = rx.recv() => {
                        match msg {
                            Some(m) => {
                                batch.push(m);
                                if batch.len() >= cfg.batch_max_messages {
                                    let _ = flush_batch(&producer, &mut batch, timeout).await;
                                }
                            }
                            None => {
                                if !batch.is_empty() {
                                    let _ = flush_batch(&producer, &mut batch, timeout).await;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        });

        Ok(Self {
            config,
            publisher: MessagePublisher { tx },
        })
    }

    pub fn new_from_env() -> Result<Self> {
        Self::new(MessageBusConfig::from_env())
    }

    #[inline]
    pub fn topic(&self, name: &str) -> String {
        format!("{}.{}", self.config.topic_prefix, name)
    }

    pub async fn publish<T: Serialize>(&self, topic: &str, key: &[u8], message: &T) -> Result<()> {
        self.publish_partitioned(topic, key, message, None).await
    }

    pub async fn publish_partitioned<T: Serialize>(
        &self,
        topic: &str,
        key: &[u8],
        message: &T,
        partition: Option<i32>,
    ) -> Result<()> {
        let payload = bincode::serialize(message)?;
        let msg = OutboundMessage {
            topic: self.topic(topic),
            key: key.to_vec(),
            payload,
            partition,
        };
        self.publisher.tx.send(msg).await.map_err(|e| anyhow!("Publish channel closed: {}", e))
    }

    pub async fn subscribe<T: DeserializeOwned + Send + 'static>(
        &self,
        topic: &str,
        group_id: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<IncomingMessage<T>>> + Send>>> {
        let consumer: StreamConsumer = ClientConfig::new()
            .set("bootstrap.servers", &self.config.brokers)
            .set("group.id", group_id)
            .set("enable.auto.commit", if self.config.enable_auto_commit { "true" } else { "false" })
            .set("auto.offset.reset", &self.config.auto_offset_reset)
            .create()
            .map_err(|e| anyhow!("Failed to create Kafka consumer: {}", e))?;

        consumer
            .subscribe(&[&self.topic(topic)])
            .map_err(|e| anyhow!("Failed to subscribe: {}", e))?;

        let (tx, rx) = mpsc::channel::<Result<IncomingMessage<T>>>(1024);

        tokio::spawn(async move {
            let mut stream = consumer.stream();
            while let Some(msg) = stream.next().await {
                match msg {
                    Ok(m) => {
                        let owned: OwnedMessage = m.detach();
                        let payload = match owned.payload() {
                            Some(p) => p,
                            None => continue,
                        };
                        let decoded: Result<T> = bincode::deserialize(payload).map_err(|e| anyhow!("Deserialize error: {}", e));
                        let item = decoded.map(|payload| IncomingMessage {
                            key: owned.key().unwrap_or(&[]).to_vec(),
                            payload,
                            topic: owned.topic().to_string(),
                            partition: owned.partition(),
                            offset: owned.offset(),
                            timestamp: owned.timestamp().to_millis(),
                        });

                        if tx.send(item).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(anyhow!("Kafka consume error: {}", e))).await;
                    }
                }
            }
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}

async fn flush_batch(producer: &FutureProducer, batch: &mut Vec<OutboundMessage>, timeout: Duration) -> Result<()> {
    // Process each message individually to avoid lifetime issues
    for m in batch.drain(..) {
        let topic = &m.topic;
        let payload = &m.payload;
        let key = &m.key;
        
        let mut rec = FutureRecord::to(topic).payload(payload).key(key);
        if let Some(p) = m.partition {
            rec = rec.partition(p);
        }
        
        match producer.send(rec, timeout).await {
            Ok(_) => {},
            Err((e, _)) => {
                // не паникуем: один fail не должен убить весь сервис
                return Err(anyhow!("Kafka produce error: {}", e));
            }
        }
    }
    Ok(())
}
