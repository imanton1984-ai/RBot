use anyhow::{Context, Result};
use rdkafka::error::KafkaError;
use rdkafka::message::OwnedMessage;
use rdkafka::{
    producer::{FutureProducer, FutureRecord},
    ClientConfig,
};
use serde::Serialize;
use serde_json::Value;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::{RwLock, Semaphore};
use tokio_postgres::NoTls;
use tracing::{info, warn};

use apps::{build_ctx, init_tracing};
use common::config::load_config;
use connections::binance_rest::BinanceRestClient;
use connections::preflight::preflight;
use connections::{BinanceWsClient, BinanceWsEvent, WsCfg};

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

async fn load_active_symbols(db_url: &str) -> Result<Vec<String>> {
    let (client, conn) = tokio_postgres::connect(db_url, NoTls).await?;
    tokio::spawn(async move {
        let _ = conn.await;
    });

    let rows = client
        .query(
            "SELECT symbol FROM market.pairs WHERE is_active = true ORDER BY symbol",
            &[],
        )
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

async fn load_last_close_map(
    db_url: &str,
    tfs: &[String],
) -> Result<HashMap<(String, String), i64>> {
    let (client, conn) = tokio_postgres::connect(db_url, NoTls).await?;
    tokio::spawn(async move {
        let _ = conn.await;
    });

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

        let rows = client
            .query(&q, &[])
            .await
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
    let record: FutureRecord<'_, str, [u8]> = FutureRecord::to(topic).key(key).payload(payload);

    // producer.send возвращает Future, результат которого при .await дает Result<(Partition, Offset), (KafkaError, OwnedMessage)>
    // Partition - i32, Offset - i64
    let delivery_result: Result<(i32, i64), (KafkaError, OwnedMessage)> =
        producer.send(record, Duration::from_secs(3)).await;

    // Извлекаем результат. Если delivery_result.Ok, сообщение успешно доставлено.
    // Если Err, произошла ошибка Kafka или при обработке сообщения (например, таймаут).
    match delivery_result {
        Ok((_partition, _offset)) => {
            // Сообщение успешно отправлено и подтверждено брокером.
            // _partition и _offset можно использовать для логирования, если нужно.
            Ok(())
        }
        Err((kafka_error, _failed_message)) => {
            // Произошла ошибка при отправке/доставке.
            // _failed_message содержит данные неудачного сообщения.
            Err(anyhow::Error::from(kafka_error).context("kafka delivery error"))
            // Или используйте ваш подход с .context и ?:
            // Err(anyhow::anyhow!("Kafka delivery failed: {}", kafka_error).context("kafka delivery error"))
        }
    }
}

fn parse_kline_close(sym: &str, tf: &str, data: &Value) -> Option<CandleCloseMsg> {
    // ожидаем структуру binance kline event: data["k"]
    let k = data.get("k")?;
    let is_closed = k.get("x")?.as_bool()?;
    if !is_closed {
        return None;
    }

    let close_time_ms = k
        .get("T")?
        .as_i64()
        .or_else(|| k.get("T")?.as_u64().map(|x| x as i64))?;
    let open = parse_f64(k.get("o")?)?;
    let high = parse_f64(k.get("h")?)?;
    let low = parse_f64(k.get("l")?)?;
    let close = parse_f64(k.get("c")?)?;
    let volume = parse_f64(k.get("v")?)?;
    let evt_time = data
        .get("E")
        .and_then(|x| x.as_i64().or_else(|| x.as_u64().map(|u| u as i64)));

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

    let cfg = load_config()?;
    let ctx = build_ctx().await?;

    // Create REST client for preflight check
    let rest_client = BinanceRestClient::new(&cfg.binance.rest_base_url, Default::default())?;
    preflight(&ctx.db, &ctx.kafka, &rest_client).await?;

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

    // Create WebSocket client
    let ws_cfg = WsCfg {
        reconnect_delay: Duration::from_secs(cfg.binance.ws_reconnect_backoff_ms as u64 / 1000),
        ping_interval: Duration::from_secs(cfg.binance.ws_ping_interval_sec),
        max_streams_per_connection: 200,
    };
    let ws_client = BinanceWsClient::new(&cfg.binance.ws_base_url, ws_cfg)?;

    // Subscribe to streams
    for stream in &streams {
        ws_client.subscribe(stream.clone())?;
    }

    // Start WebSocket connection
    ws_client.connect_sharded().await?;

    // Get the event receiver
    let mut rx = ws_client.events();

    // ограничим publish-конкурентность
    let sem = Arc::new(Semaphore::new(
        std::env::var("LIVE_PUB_CONCURRENCY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64),
    ));

    loop {
        let evt = rx.recv().await.context("ws channel closed")?;
        match evt {
            // <-- Открывающая скобка для всего match
            BinanceWsEvent::Message { stream, data } => {
                // stream: "btcusdt@kline_1m" → достаем tf
                let tf = stream.split("@kline_").nth(1).unwrap_or("");
                if tf.is_empty() {
                    continue;
                }

                // symbol из payload обычно в data["s"]
                let sym = data
                    .get("s")
                    .and_then(|x| x.as_str())
                    .map(|s| s.to_string())
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
            } // <--- Запятая или точка с запятой после блока Message, внутри match
            BinanceWsEvent::Connected => {
                info!("svc_live: ws connected");
            }
            BinanceWsEvent::Disconnected => {
                // <-- Второй вариант match
                warn!("svc_live: ws disconnected");
            }
            BinanceWsEvent::Error(err) => {
                warn!("svc_live: ws error: {err}");
            } // <-- Блок Disconnected закончен
        } // <-- Закрывающая скобка для ВСЕГО match выражения
    } // <-- Конец loop
}
