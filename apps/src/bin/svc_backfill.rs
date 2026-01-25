use anyhow::{Context, Result};
use futures_util::stream::{FuturesUnordered, StreamExt};
use rdkafka::{producer::{FutureProducer, FutureRecord}, ClientConfig};
use serde::Serialize;
use std::{sync::Arc, time::Duration};
use tokio::sync::Semaphore;
use tokio_postgres::NoTls;
use tracing::{info, warn};

use apps::{build_ctx, init_tracing};

#[derive(Debug, Serialize)]
struct CandleCloseMsg {
    pub symbol: String,
    pub tf: String,
    pub close_time_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub source_event_time_ms: Option<i64>,
}

fn parse_f64(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::String(s) => s.parse().ok(),
        serde_json::Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

async fn load_active_symbols(db_url: &str) -> Result<Vec<String>> {
    let (client, conn) = tokio_postgres::connect(db_url, NoTls).await?;
    tokio::spawn(async move {
        let _ = conn.await;
    });

    let rows = client
        .query("SELECT symbol FROM market.pairs WHERE is_active = true ORDER BY symbol", &[])
        .await
        .context("query market.pairs failed")?;

    Ok(rows.into_iter().map(|r| r.get::<_, String>(0)).collect())
}

fn build_producer(brokers: &str) -> Result<FutureProducer> {
    let p: FutureProducer = ClientConfig::new()
        .set("bootstrap.servers", brokers)
        .set("message.timeout.ms", "5000")
        .set("queue.buffering.max.messages", "200000")
        .set("queue.buffering.max.kbytes", "1048576")
        .set("linger.ms", "2")
        .set("batch.num.messages", "10000")
        .create()
        .context("create kafka producer failed")?;
    Ok(p)
}

async fn send_mp(producer: &FutureProducer, topic: &str, key: &str, payload: &[u8]) -> Result<()> {
    // простая надежная отправка
    let rec = FutureRecord::to(topic).key(key).payload(payload);
    let (_partition, delivery) = producer.send(rec, Duration::from_secs(3)).await;
    delivery.context("kafka delivery error")?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    info!("svc_backfill: starting");

    let ctx = build_ctx().await?;
    let cfg = &ctx.cfg;

    let db_url = cfg.database.url();
    let brokers = cfg.rust_bot.redpanda_brokers.join(",");
    let topic = cfg.rust_bot.topic_candles_close.clone();

    let max_candles: usize = std::env::var("BACKFILL_CANDLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(cfg.runtime.backfill_candles as usize);

    let concurrency: usize = std::env::var("BACKFILL_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);

    let tfs: Vec<String> = cfg.runtime.timeframes.clone();

    info!("svc_backfill: brokers={brokers} topic={topic} max_candles={max_candles} concurrency={concurrency}");
    info!("svc_backfill: tfs={:?}", tfs);

    let symbols = if let Ok(s) = std::env::var("BACKFILL_SYMBOLS") {
        s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()
    } else {
        load_active_symbols(&db_url).await?
    };
    info!("svc_backfill: symbols={}", symbols.len());

    let producer = build_producer(&brokers)?;
    let sem = Arc::new(Semaphore::new(concurrency));

    let mut futs = FuturesUnordered::new();

    for sym in symbols {
        for tf in &tfs {
            let permit = sem.clone().acquire_owned().await.unwrap();
            let rest = ctx.rest.clone();
            let producer = producer.clone();
            let topic = topic.clone();
            let sym2 = sym.clone();
            let tf2 = tf.clone();

            futs.push(tokio::spawn(async move {
                let _p = permit;

                // Binance: берем последние max_candles баров
                let kl = rest
                    .futures_klines(&sym2, &tf2, max_candles as i32, None, None)
                    .await
                    .with_context(|| format!("klines failed {sym2} {tf2}"))?;

                // ожидаем формат как Vec<Vec<Value>> (как в большинстве бинанс-оберток)
                // [open_time, open, high, low, close, volume, close_time, ...]
                let arr = kl.as_array().context("klines: not array")?;

                let mut sent = 0usize;
                for row in arr {
                    let row = row.as_array().context("kline row: not array")?;
                    if row.len() < 7 { continue; }

                    let close_time_ms = row[6].as_i64().or_else(|| row[6].as_u64().map(|x| x as i64));
                    let open = parse_f64(&row[1]);
                    let high = parse_f64(&row[2]);
                    let low = parse_f64(&row[3]);
                    let close = parse_f64(&row[4]);
                    let volume = parse_f64(&row[5]);

                    let (Some(close_time_ms), Some(open), Some(high), Some(low), Some(close), Some(volume)) =
                        (close_time_ms, open, high, low, close, volume) else { continue; };

                    let msg = CandleCloseMsg {
                        symbol: sym2.clone(),
                        tf: tf2.clone(),
                        close_time_ms,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        source_event_time_ms: None,
                    };

                    let payload = rmp_serde::to_vec_named(&msg)?;
                    let key = format!("{}|{}|{}", msg.symbol, msg.tf, msg.close_time_ms);
                    send_mp(&producer, &topic, &key, &payload).await?;
                    sent += 1;
                }

                Ok::<usize, anyhow::Error>(sent)
            }));
        }
    }

    let mut total = 0usize;
    while let Some(res) = futs.next().await {
        match res {
            Ok(Ok(n)) => total += n,
            Ok(Err(e)) => warn!("svc_backfill: task failed: {e:#}"),
            Err(e) => warn!("svc_backfill: join failed: {e:#}"),
        }
    }

    info!("svc_backfill: done, published {total} candle close events");
    Ok(())
}

