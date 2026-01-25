use anyhow::{Context, Result};
use futures_util::StreamExt;
use rdkafka::{producer::{FutureProducer, FutureRecord}, ClientConfig};
use serde::Serialize;
use serde_json::Value;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::{Semaphore, RwLock};
use tokio_postgres::NoTls;
use tracing::{info, warn};

use apps::{build_ctx, init_tracing};
use connections::{BinanceWsEvent, preflight};

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

fn parse_f64(v: &Value) -> Option<f64> {
    match v {
        Value::String(s) => s.parse().ok(),
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

async fn load_active_symbols(db_url: &str) -> Result<Vec<String>> {
    let (client, conn) = tokio_postgres::connect(db_url, NoTls).await?;
    tokio::spawn(async move { let _ = conn.await; });

    let rows = client
        .query("SELECT symbol FROM market.pairs WHERE is_active = true ORDER BY symbol", &[])
        .await?;
    Ok(rows.into_iter().map(|r| r.get::<_, String>(0)).collect())
}

fn candles_table(tf: &str) -> Result<&'static str> {
    Ok(match tf {
        "1m" => "market.candles_1m",
        "5m" => "market.candles_5m",
        "15m" => "market.candles_15m",
        "1h" => "market.candles_1h",
        "4h" => "market.candles_4h",
        "1d" => "market.candles_1d",
        _ => anyhow::bail!("unknown tf: {tf}"),
    })
}

async fn load_last_close_map(db_url: &str, tfs: &[String]) -> Result<HashMap<(String, String), i64>> {
    let (client, conn) = tokio_postgres::connect(db_url, NoTls).await?;
    tokio::spawn(async move { let _ = conn.await; });

    // map[(symbol, tf)] = max(close_time_ms)
    let mut out: HashMap<(String, String), i64> = HashMap::new();

    for tf in tfs {
        let table = candles_table(tf)?;
        // join pairs to filter active and map symbol->id
        let q = format!(
            "SELECT p.symbol, COALESCE(MAX(c.close_time_ms), 0) AS mx
             FROM market.pairs p
             LEFT JOIN {table} c ON c.symbol_id = p.symbol_id
             WHERE p.is_active = true
             GROUP BY p.symbol
             ORDER BY p.symbol"
        );

        let rows = client.query(&q, &[]).await
            .with_context(|| format!("load_last_close_map query failed for tf={tf}"))?;

        for r in rows {
            let sym: String = r.get(0);
            let mx: i64 = r.get(1);
            out.insert((sym, tf.clone()), mx);
        }
    }

    Ok(out)
}

fn build_producer(brokers: &str) -> Result<FutureProducer> {
    Ok(ClientConfig::new()
        .set("bootstrap.servers", brokers)
        .set("message.timeout.ms", "5000")
        .set("queue.buffering.max.messages", "200000")
        .set("queue.buffering.max.kbytes", "1048576")
        .set("linger.ms", "2")
        .set("batch.num.messages", "10000")
        .create()
        .context("create kafka producer failed")?)
}

async fn send_mp(producer: &FutureProducer, topic: &str, key: &str, payload: &[u8]) -> Result<()> {
    let rec = FutureRecord::to(topic).key(key).payload(payload);
    let (_p, delivery) = producer.send(rec, Duration::from_secs(3)).await;
    delivery.context("kafka delivery error")?;
    Ok(())
}

fn parse_kline_close(sym: &str, tf: &str, data: &Value) -> Option<CandleCloseMsg> {
    // ожидаем структуру binance kline event: data["k"]
    let k = data.get("k")?;
    let is_closed = k.get("x")?.as_bool()?;
    if !is_closed {
        return None;
    }

    let close_time_ms = k.get("T")?.as_i64().or_else(|| k.get("T")?.as_u64().map(|x| x as i64))?;
    let open = parse_f64(k.get("o")?)?;
    let high = parse_f64(k.get("h")?)?;
    let low = parse_f64(k.get("l")?)?;
    let close = parse_f64(k.get("c")?)?;
    let volume = parse_f64(k.get("v")?)?;
    let evt_time = data.get("E").and_then(|x| x.as_i64().or_else(|| x.as_u64().map(|u| u as i64)));

    Some(CandleCloseMsg {
        symbol: sym.to_string(),
        tf: tf.to_string(),
        close_time_ms,
        open,
        high,
        low,
        close,
        volume,
        source_event_time_ms: evt_time,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    info!("svc_live: starting");

    let ctx = build_ctx().await?;
    preflight(&ctx.db, &ctx.kafka, &ctx.rest).await?;

    let cfg = &ctx.cfg;
    let db_url = cfg.database.url();
    let brokers = cfg.rust_bot.redpanda_brokers.join(",");
    let topic = cfg.rust_bot.topic_candles_close.clone();

    let tfs: Vec<String> = cfg.runtime.timeframes.clone();
    let symbols = load_active_symbols(&db_url).await?;
    info!("svc_live: symbols={} tfs={:?}", symbols.len(), tfs);

    let producer = build_producer(&brokers)?;
    let last_map = load_last_close_map(&db_url, &tfs).await?;
    let last_map = Arc::new(RwLock::new(last_map));

    // Binance WS streams: "<symbol>@kline_<tf>"
    let mut streams = Vec::new();
    for sym in &symbols {
        for tf in &tfs {
            streams.push(format!("{}@kline_{}", sym.to_lowercase(), tf));
        }
    }
    info!("svc_live: streams={}", streams.len());

    // sharded connect in ws client (у тебя уже есть логика шардирования в connections/ws)
    let (mut rx, _handles) = ctx.ws.connect_sharded(streams, 200).await?;

    // ограничим publish-конкурентность
    let sem = Arc::new(Semaphore::new(
        std::env::var("LIVE_PUB_CONCURRENCY").ok().and_then(|v| v.parse().ok()).unwrap_or(64),
    ));

    loop {
        let evt = rx.recv().await.context("ws channel closed")?;
        match evt {
            BinanceWsEvent::Message { stream, data } => {
                // stream: "btcusdt@kline_1m" → достаем tf
                let tf = stream.split("@kline_").nth(1).unwrap_or("");
                if tf.is_empty() { continue; }

                // symbol из payload обычно в data["s"]
                let sym = data.get("s").and_then(|x| x.as_str()).map(|s| s.to_string())
                    .unwrap_or_else(|| stream.split('@').next().unwrap_or("").to_uppercase());

                if let Some(msg) = parse_kline_close(&sym, tf, &data) {
                    let key = (msg.symbol.clone(), msg.tf.clone());
                    let mut lm = last_map.write().await;
                    let last = *lm.get(&key).unwrap_or(&0);

                    if msg.close_time_ms <= last {
                        // дубль/старое
                        continue;
                    }
                    lm.insert(key, msg.close_time_ms);
                    drop(lm);

                    let payload = rmp_serde::to_vec_named(&msg)?;
                    let k = format!("{}|{}|{}", msg.symbol, msg.tf, msg.close_time_ms);

                    let permit = sem.clone().acquire_owned().await.unwrap();
                    let producer = producer.clone();
                    let topic = topic.clone();

                    tokio::spawn(async move {
                        let _p = permit;
                        if let Err(e) = send_mp(&producer, &topic, &k, &payload).await {
                            warn!("svc_live: publish failed: {e:#}");
                        }
                    });
                }
            }
            BinanceWsEvent::Closed { reason } => {
                warn!("svc_live: ws closed: {reason}");
            }
