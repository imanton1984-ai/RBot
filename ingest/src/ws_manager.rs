use anyhow::{Context, Result};
use common::timeframe::Timeframe;
use futures::StreamExt;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use rmp_serde;

use crate::backfill::LastCloseMap;
use crate::candle_builder::{fetch_klines, now_ms, tf_ms};
use crate::config::IngestConfig;
use crate::gap_fill::gap_fill_between;
use crate::producer::build_producer;
use db_writer::messages::CandleCloseMsg;

#[derive(serde::Deserialize)]
struct WsCombined<T> {
    stream: String,
    data: T,
}

#[derive(serde::Deserialize)]
struct KlineEvent {
    #[serde(rename = "E")]
    event_time_ms: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "k")]
    k: KlineData,
}

#[derive(serde::Deserialize)]
struct KlineData {
    #[serde(rename = "i")]
    interval: String,
    #[serde(rename = "t")]
    open_time_ms: i64,
    #[serde(rename = "T")]
    close_time_ms: i64,
    #[serde(rename = "o")]
    open: String,
    #[serde(rename = "h")]
    high: String,
    #[serde(rename = "l")]
    low: String,
    #[serde(rename = "c")]
    close: String,
    #[serde(rename = "v")]
    volume: String,
    #[serde(rename = "x")]
    is_closed: bool,
}

fn pf64(s: &str) -> Result<f64> {
    Ok(s.parse::<f64>()?)
}

fn make_close(evt: &KlineEvent) -> Result<CandleCloseMsg> {
    Ok(CandleCloseMsg {
        symbol: evt.symbol.clone(),
        tf: evt.k.interval.clone(),
        close_time_ms: evt.k.close_time_ms,
        open: pf64(&evt.k.open)?,
        high: pf64(&evt.k.high)?,
        low: pf64(&evt.k.low)?,
        close: pf64(&evt.k.close)?,
        volume: pf64(&evt.k.volume)?,
        source_event_time_ms: Some(evt.event_time_ms),
    })
}

async fn send_close(producer: &FutureProducer, topic: &str, key: &str, msg: &CandleCloseMsg) -> Result<()> {
    let payload = rmp_serde::to_vec_named(msg)?;
    loop {
        match producer.send(FutureRecord::to(topic).key(key).payload(&payload), Timeout::Never) {
            Ok(_) => return Ok(()),
            Err((e, _)) => {
                if e.to_string().contains("Queue full") {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                    continue;
                }
                return Err(e.into());
            }
        }
    }
}

// делим symbols на чанки так, чтобы (symbols_in_conn * ws_timeframes.len()) <= max_streams
fn shard_symbols(symbols: &[String], ws_tfs: usize, max_streams: usize) -> Vec<Vec<String>> {
    let per_conn = (max_streams / ws_tfs).max(1);
    symbols
        .chunks(per_conn)
        .map(|c| c.to_vec())
        .collect()
}

fn build_streams_chunk(symbols: &[String], tfs: &[Timeframe]) -> String {
    let mut parts = Vec::with_capacity(symbols.len() * tfs.len());
    for s in symbols {
        let sl = s.to_lowercase();
        for tf in tfs {
            parts.push(format!("{sl}@kline_{}", tf.as_binance_interval()));
        }
    }
    parts.join("/")
}

pub async fn run_ws_and_poll(
    cfg: Arc<IngestConfig>,
    symbols: Vec<String>,
    last_map: LastCloseMap,
) -> Result<()> {
    let http = reqwest::Client::builder()
        .tcp_nodelay(true)
        .pool_max_idle_per_host(8)
        .timeout(std::time::Duration::from_millis(10_000))
        .build()?;

    let producer = build_producer(&cfg)?;

    // shared last-close map (для gap fill + poll)
    let last = Arc::new(RwLock::new(last_map));

    // WS realtime для ws_timeframes :contentReference[oaicite:11]{index=11}
    let shards = shard_symbols(&symbols, cfg.ws_timeframes.len().max(1), cfg.ws_max_streams_per_conn);

    for (idx, chunk) in shards.into_iter().enumerate() {
        let cfg = cfg.clone();
        let http = http.clone();
        let producer = producer.clone();
        let last = last.clone();

        tokio::spawn(async move {
            loop {
                let streams = build_streams_chunk(&chunk, &cfg.ws_timeframes);
                let url = format!("{}/stream?streams={}", cfg.ws_base_url.trim_end_matches('/'), streams);

                let conn = tokio_tungstenite::connect_async(url).await;
                let (mut ws, _) = match conn {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[ws#{idx}] connect error: {e:?}");
                        tokio::time::sleep(std::time::Duration::from_millis(800)).await;
                        continue;
                    }
                };

                while let Some(msg) = ws.next().await {
                    let msg = match msg {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("[ws#{idx}] read error: {e:?}");
                            break;
                        }
                    };

                    let text = match msg {
                        WsMessage::Text(t) => t,
                        WsMessage::Binary(b) => String::from_utf8_lossy(&b).to_string(),
                        _ => continue,
                    };

                    let combined: WsCombined<KlineEvent> = match serde_json::from_str(&text) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let ev = combined.data;
                    if !ev.k.is_closed {
                        // intra-candle updates можно добавить позже (topic_candles_update),
                        // но db_writer пока принимает только close.
                        continue;
                    }

                    // tf parse
                    let tf = match Timeframe::parse(&ev.k.interval) {
                        Some(t) => t,
                        None => continue,
                    };

                    let close_evt = make_close(&ev).context("make_close")?;
                    let key = format!("{}|{}", close_evt.symbol, close_evt.tf);

                    // gap fill
                    let mut guard = last.write().await;
                    let prev = guard.get(&(close_evt.symbol.clone(), close_evt.tf.clone())).copied();

                    if let Some(prev_close) = prev {
                        let step = tf_ms(tf);
                        if close_evt.close_time_ms > prev_close + step {
                            // догружаем между prev_close и current_close (exclusive current)
                            let filled_last = gap_fill_between(
                                &cfg,
                                &http,
                                &producer,
                                &close_evt.symbol,
                                tf,
                                prev_close,
                                close_evt.close_time_ms,
                                ev.event_time_ms,
                            )
                            .await
                            .unwrap_or(prev_close);
                            guard.insert((close_evt.symbol.clone(), close_evt.tf.clone()), filled_last);
                        }
                    }

                    // publish current close
                    send_close(&producer, &cfg.topic_candles_close, &key, &close_evt).await?;
                    guard.insert((close_evt.symbol.clone(), close_evt.tf.clone()), close_evt.close_time_ms);
                }

                tokio::time::sleep(std::time::Duration::from_millis(800)).await;
            }

            #[allow(unreachable_code)]
            Ok::<(), anyhow::Error>(())
        });
    }

    // poll-on-close для старших TF :contentReference[oaicite:12]{index=12}
    if !cfg.poll_timeframes.is_empty() {
        let cfg2 = cfg.clone();
        let http2 = http.clone();
        let producer2 = producer.clone();
        let last2 = last.clone();
        let symbols2 = symbols.clone();

        tokio::spawn(async move {
            let tick = std::time::Duration::from_secs(cfg2.poll_on_close_interval_sec.max(5));
            loop {
                tokio::time::sleep(tick).await;

                // параллелим умеренно (не надо DDOS’ить REST)
                let sem = Arc::new(tokio::sync::Semaphore::new(cfg2.http_concurrency));

                let mut futs = futures::stream::FuturesUnordered::new();

                for sym in symbols2.iter().cloned() {
                    for &tf in cfg2.poll_timeframes.iter() {
                        let permit = sem.clone().acquire_owned().await.unwrap();
                        let cfg2 = cfg2.clone();
                        let http2 = http2.clone();
                        let producer2 = producer2.clone();
                        let last2 = last2.clone();

                        futs.push(tokio::spawn(async move {
                            let _permit = permit;

                            // берём последние 2 свечи и публикуем последнюю закрытую
                            let ks = fetch_klines(&http2, &cfg2.rest_base_url, &sym, tf, 2, None).await?;
                            if ks.is_empty() {
                                return Ok::<(), anyhow::Error>(());
                            }
                            let k = ks.last().unwrap().clone();
                            let evt = CandleCloseMsg {
                                symbol: sym.clone(),
                                tf: tf.as_binance_interval().to_string(),
                                close_time_ms: k.close_time_ms,
                                open: k.open,
                                high: k.high,
                                low: k.low,
                                close: k.close,
                                volume: k.volume,
                                source_event_time_ms: Some(now_ms()),
                            };

                            let key = format!("{}|{}", evt.symbol, evt.tf);

                            let mut guard = last2.write().await;
                            let prev = guard.get(&(evt.symbol.clone(), evt.tf.clone())).copied();
                            if prev.map(|p| evt.close_time_ms > p).unwrap_or(true) {
                                send_close(&producer2, &cfg2.topic_candles_close, &key, &evt).await?;
                                guard.insert((evt.symbol.clone(), evt.tf.clone()), evt.close_time_ms);
                            }

                            Ok::<(), anyhow::Error>(())
                        }));
                    }
                }

                while let Some(r) = futs.next().await {
                    if let Err(e) = r {
                        eprintln!("poll task join error: {e:?}");
                    }
                }
            }
        });
    }

    // держим функцию “живой”
    futures::future::pending::<()>().await;
    Ok(())
}
