use anyhow::Result;
use rdkafka::config::ClientConfig;
use rdkafka::producer::FutureProducer;

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
        .create()?;

    Ok(producer)
}
