use anyhow::{Context, Result};
use rdkafka::{
    producer::{FutureProducer, FutureRecord},
    ClientConfig,
};
use std::time::Duration;

use crate::config::IngestConfig;

pub fn build_producer(cfg: &IngestConfig) -> Result<FutureProducer> {
    let brokers = cfg.redpanda_brokers.join(",");

    let producer: FutureProducer = ClientConfig::new()
        .set("bootstrap.servers", &brokers)
        .set("client.id", &cfg.redpanda_client_id)
        // throughput
        .set("compression.type", "lz4")
        .set("linger.ms", "5")
        .set("batch.num.messages", "10000")
        .set("queue.buffering.max.kbytes", "1048576")
        .set("queue.buffering.max.messages", "1000000")
        // reliability baseline
        .set("acks", "all")
        .set("enable.idempotence", "true")
        .set("message.timeout.ms", "10000")
        .create()
        .context("create kafka producer failed")?;

    Ok(producer)
}

pub async fn send_mp(producer: &FutureProducer, topic: &str, key: &str, payload: &[u8]) -> Result<()> {
    let rec = FutureRecord::to(topic).key(key).payload(payload);
    let (_partition, delivery) = producer.send(rec, Duration::from_secs(3)).await;
    delivery.context("kafka delivery error")?;
    Ok(())
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
