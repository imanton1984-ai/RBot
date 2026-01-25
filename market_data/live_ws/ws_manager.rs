use anyhow::{Context, Result};
use common::timeframe::Timeframe;

use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tracing::{debug, info, warn};
use futures::{SinkExt, stream::FuturesUnordered, StreamExt};

use crate::backfill::LastCloseMap;
use crate::candle_builder::{fetch_klines, now_ms, tf_ms, KlineRow};
use crate::config::IngestConfig;
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
    Ok(s.parse::<f64>().with_context(|| format!("parse f64: {}", s))?)
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

async fn send_close_mp(
    producer: &FutureProducer,
    topic: &str,
    key: &str,
    msg: &CandleCloseMsg,
) -> Result<()> {
    // ⚡ MessagePack (быстрее JSON, меньше трафика)
    let payload = rmp_serde::to_vec_named(msg)?;
    loop {
        match producer
            .send(
                FutureRecord::to(topic).key(key).payload(&payload),
                Timeout::Never,
            )
            .await
        {
            Ok(_) => return Ok(()),
            Err((e, _)) => {
                // мягкий backpressure на переполнение очереди
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
    let per_conn = (max_streams / ws_tfs.max(1)).max(1);
    symbols.chunks(per_conn).map(|c| c.to_vec()).collect()
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

/// Gap-fill между prev_close_ms и new_close_ms (исключая new_close_ms),
/// публикуя закрытые свечи через тот же MessagePack.
async fn gap_fill_between_mp(
    cfg: &IngestConfig,
    http: &reqwest::Client,
    producer: &FutureProducer,
    symbol: &str,
    tf: Timeframe,
    prev_close_ms: i64,
    new_close_ms: i64,
    src_event_ms: i64,
) -> Result<i64> {
    let step = tf_ms(tf);
    if new_close_ms <= prev_close_ms + step {
        return Ok(prev_close_ms);
    }

    // Для Binance closeTime = openTime + step - 1
    // Следующий бар начинается с open = prev_close + 1
    let mut cursor_open = prev_close_ms + 1;
    let mut last_published = prev_close_ms;

    while cursor_open + step < new_close_ms {
        let klines = fetch_klines(
            http,
            &cfg.rest_base_url,
            symbol,
            tf,
            1000,
            Some(cursor_open),
        )
        .await
        .with_context(|| format!("fetch_klines gapfill {} {}", symbol, tf.as_binance_interval()))?;

        if klines.is_empty() {
            break;
        }

        let mut advanced = false;
        for k in klines {
            if k.close_time_ms >= new_close_ms {
                break;
            }
            // publish only strictly after prev
            if k.close_time_ms <= last_published {
                continue;
            }

            let evt = CandleCloseMsg {
                symbol: symbol.to_string(),
                tf: tf.as_binance_interval().to_string(),
                close_time_ms: k.close_time_ms,
                open: k.open,
                high: k.high,
                low: k.low,
                close: k.close,
                volume: k.volume,
                source_event_time_ms: Some(src_event_ms),
            };

            let key = format!("{}|{}", symbol, evt.tf);
            send_close_mp(producer, &cfg.topic_candles_close, &key, &evt).await?;
            last_published = k.close_time_ms;
            cursor_open = k.close_time_ms + 1;
            advanced = true;
        }

        if !advanced {
            break;
        }

        if last_published + step >= new_close_ms {
            break;
        }
    }

    Ok(last_published)
}

pub async fn run_ws_and_poll(cfg: Arc<IngestConfig>, symbols: Vec<String>, last_map: LastCloseMap) -> Result<()> {
    let http = reqwest::Client::builder()
        .tcp_nodelay(true)
        .pool_max_idle_per_host(32)
        .timeout(std::time::Duration::from_millis(10_000))
        .build()
        .context("build reqwest client")?;

    let producer = build_producer(&cfg).context("build_producer")?;

    // shared last-close map (для gap fill + poll)
    let last = Arc::new(RwLock::new(last_map));

    // ---- WS realtime (для cfg.ws_timeframes) ----
    let shards = shard_symbols(
        &symbols,
        cfg.ws_timeframes.len().max(1),
        cfg.ws_max_streams_per_conn,
    );

    for (idx, chunk) in shards.into_iter().enumerate() {
        let cfg = cfg.clone();
        let http = http.clone();
        let producer = producer.clone();
        let last = last.clone();

        tokio::spawn(async move {
            // backoff при реконнекте
            let mut backoff_ms: u64 = 300;

            loop {
                let streams = build_streams_chunk(&chunk, &cfg.ws_timeframes);
                let url = format!(
                    "{}/stream?streams={}",
                    cfg.ws_base_url.trim_end_matches('/'),
                    streams
                );

                info!("[ws#{idx}] connecting ({} symbols, {} tfs)", chunk.len(), cfg.ws_timeframes.len());

                let conn = tokio_tungstenite::connect_async(url).await;
                let (mut ws, _) = match conn {
                    Ok(v) => {
                        backoff_ms = 300; // reset on success
                        v
                    }
                    Err(e) => {
                        warn!("[ws#{idx}] connect error: {e:?}, sleep {}ms", backoff_ms);
                        tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                        backoff_ms = (backoff_ms * 2).min(10_000);
                        continue;
                    }
                };

                // keepalive ping
                let mut ping_tick = tokio::time::interval(std::time::Duration::from_secs(30));

                loop {
                    tokio::select! {
                        _ = ping_tick.tick() => {
                            // tungstenite ping
                            let _ = ws.send(WsMessage::Ping(Vec::new())).await;
                        }

                        maybe = ws.next() => {
                            let msg = match maybe {
                                Some(Ok(m)) => m,
                                Some(Err(e)) => {
                                    warn!("[ws#{idx}] read error: {e:?}");
                                    break;
                                }
                                None => {
                                    warn!("[ws#{idx}] ws ended (None)");
                                    break;
                                }
                            };

                            let bytes: Vec<u8> = match msg {
                                WsMessage::Text(t) => t.into_bytes(),
                                WsMessage::Binary(b) => b,
                                WsMessage::Ping(p) => {
                                    // auto-pong
                                    let _ = ws.send(WsMessage::Pong(p)).await;
                                    continue;
                                }
                                WsMessage::Pong(_) => continue,
                                WsMessage::Close(c) => {
                                    debug!("[ws#{idx}] close frame: {c:?}");
                                    break;
                                }
                                _ => continue,
                            };

                            let combined: WsCombined<KlineEvent> = match serde_json::from_slice(&bytes) {
                                Ok(v) => v,
                                Err(_) => continue, // шум/служебка
                            };

                            let ev = combined.data;

                            // сейчас пишем только закрытые бары (db_writer так ожидает)
                            if !ev.k.is_closed {
                                continue;
                            }

                            let tf = match Timeframe::parse(&ev.k.interval) {
                                Ok(t) => t,
                                Err(_) => continue,
                            };

                            let close_evt = match make_close(&ev) {
                                Ok(v) => v,
                                Err(e) => {
                                    warn!("[ws#{idx}] make_close err: {e:#}");
                                    continue;
                                }
                            };

                            let key = format!("{}|{}", close_evt.symbol, close_evt.tf);

                            // --- gap fill + publish ---
                            // 1) read prev without holding write lock долго
                            let prev = {
                                let guard = last.read().await;
                                guard.get(&(close_evt.symbol.clone(), close_evt.tf.clone())).copied()
                            };

                            if let Some(prev_close) = prev {
                                let step = tf_ms(tf);
                                if close_evt.close_time_ms > prev_close + step {
                                    match gap_fill_between_mp(
                                        &cfg,
                                        &http,
                                        &producer,
                                        &close_evt.symbol,
                                        tf,
                                        prev_close,
                                        close_evt.close_time_ms,
                                        ev.event_time_ms,
                                    ).await {
                                        Ok(filled_last) => {
                                            let mut guard = last.write().await;
                                            guard.insert((close_evt.symbol.clone(), close_evt.tf.clone()), filled_last);
                                        }
                                        Err(e) => {
                                            warn!("[ws#{idx}] gap_fill failed {} {}: {e:#}",
                                                close_evt.symbol, close_evt.tf
                                            );
                                        }
                                    }
                                }
                            }

                            // 2) publish current close
                            if let Err(e) = send_close_mp(&producer, &cfg.topic_candles_close, &key, &close_evt).await {
                                warn!("[ws#{idx}] send_close failed: {e:#}");
                                break; // вероятно продюсер/брокер в плохом состоянии → реконнект
                            }

                            // 3) update last
                            {
                                let mut guard = last.write().await;
                                guard.insert((close_evt.symbol.clone(), close_evt.tf.clone()), close_evt.close_time_ms);
                            }
                        }
                    }
                }

                warn!("[ws#{idx}] reconnect in {}ms", backoff_ms);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(10_000);
            }
        });
    }

    // ---- Poll-on-close (для старших TF) ----
    if !cfg.poll_timeframes.is_empty() {
        let cfg2 = cfg.clone();
        let http2 = http.clone();
        let producer2 = producer.clone();
        let last2 = last.clone();
        let symbols2 = symbols.clone();

        tokio::spawn(async move {
            let tick = std::time::Duration::from_secs(cfg2.poll_on_close_interval_sec.max(5));
            let sem = Arc::new(tokio::sync::Semaphore::new(cfg2.http_concurrency.max(1)));

            loop {
                tokio::time::sleep(tick).await;

                let mut futs: FuturesUnordered<_> = FuturesUnordered::new();

                for sym in symbols2.iter() {
                    for &tf in cfg2.poll_timeframes.iter() {
                        let cfg2 = cfg2.clone();
                        let http2 = http2.clone();
                        let producer2 = producer2.clone();
                        let last2 = last2.clone();
                        let sem = sem.clone();
                        let sym = sym.clone();

                        futs.push(async move {
                            let _permit = sem.acquire_owned().await.context("semaphore closed")?;

                            // последние 2 свечи
                            let ks = fetch_klines(&http2, &cfg2.rest_base_url, &sym, tf, 2, None).await?;
                            if ks.is_empty() {
                                return Ok::<(), anyhow::Error>(());
                            }
                            let k: KlineRow = ks.last().unwrap().clone();

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

                            // publish only if newer than last
                            let prev = {
                                let guard = last2.read().await;
                                guard.get(&(evt.symbol.clone(), evt.tf.clone())).copied()
                            };

                            if prev.map(|p| evt.close_time_ms > p).unwrap_or(true) {
                                let key = format!("{}|{}", evt.symbol, evt.tf);
                                send_close_mp(&producer2, &cfg2.topic_candles_close, &key, &evt).await?;

                                let mut guard = last2.write().await;
                                guard.insert((evt.symbol.clone(), evt.tf.clone()), evt.close_time_ms);
                            }

                            Ok::<(), anyhow::Error>(())
                        });
                    }
                }

                while let Some(res) = futs.next().await {
                    if let Err(e) = res {
                        warn!("poll task error: {e:#}");
                    }
                }
            }
        });
    }

    // держим функцию “живой”
    futures::future::pending::<()>().await;
    Ok(())
}

