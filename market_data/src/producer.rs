use anyhow::{Context, Result};
use rdkafka::message::Message;
use rdkafka::{
    producer::{FutureProducer, FutureRecord},
    ClientConfig,
};
use std::time::Duration;

use crate::config::IngestConfig;

pub fn build_producer(cfg: &IngestConfig) -> Result<FutureProducer> {
    let producer: FutureProducer = ClientConfig::new()
        .set("bootstrap.servers", &cfg.redpanda_brokers)
        .set("client.id", &cfg.redpanda_client_id)
        .set("compression.type", "lz4")
        .set("linger.ms", "5")
        .set("batch.num.messages", "10000")
        .set("queue.buffering.max.kbytes", "1048576")
        .set("queue.buffering.max.messages", "1000000")
        .set("acks", "all")
        .set("enable.idempotence", "true")
        .set("message.timeout.ms", "10000")
        .create()
        .context("create kafka producer failed")?;

    Ok(producer)
}

pub async fn send_mp(
    producer: &FutureProducer,
    topic: &str,
    key: &str,
    payload: &[u8],
) -> Result<()> {
    let rec = FutureRecord::to(topic).key(key).payload(payload);
    let delivery_result = producer.send(rec, Duration::from_secs(3)).await;

    match delivery_result {
        Ok((_partition, _offset)) => Ok(()),
        Err((kafka_error, msg)) => {
            // Исправленная логика извлечения полезной нагрузки для лога:
            // payload_view возвращает Option<Result<&str, Utf8Error>>
            let payload_str = msg
                .payload_view::<str>()
                .and_then(|res| res.ok()) // берем &str только если парсинг успешен
                .unwrap_or("<invalid utf8 or empty>");

            anyhow::bail!(
                "Kafka delivery failed: {:?}, topic: {}, key: {}, payload: {}",
                kafka_error,
                topic,
                key,
                payload_str
            );
        }
    }
}

pub async fn send_close_mp<T: serde::Serialize>(
    producer: &FutureProducer,
    topic: &str,
    key: &str,
    msg: &T,
) -> Result<()> {
    let payload = rmp_serde::to_vec_named(msg).context("msgpack encode failed")?;
    send_mp(producer, topic, key, &payload).await
}
