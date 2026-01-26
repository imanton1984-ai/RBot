use anyhow::{Context, Result};
use common::timeframe::Timeframe;
use connections::BinanceRestClient;
use futures_util::{stream::FuturesUnordered, SinkExt, StreamExt};
use rdkafka::producer::FutureProducer;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tracing::{debug, info, warn};

use crate::backfill::LastCloseMap;
use crate::candle_builder::{fetch_klines, tf_ms, KlineRow};
use crate::config::IngestConfig;
use crate::gap_fill::gap_fill_between_mp;
use crate::producer::send_close_mp;
use data_writer::messages::CandleCloseMsg;

#[derive(serde::Deserialize)]
#[allow(dead_code)]
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
#[allow(dead_code)]
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
    Ok(s.parse::<f64>()
        .with_context(|| format!("parse f64: {}", s))?)
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

fn shard_symbols(
    symbols: &[String],
    tfs_count: usize,
    max_streams_per_conn: usize,
) -> Vec<Vec<String>> {
    let per_conn = (max_streams_per_conn / tfs_count.max(1)).max(1);
    symbols
        .chunks(per_conn)
        .map(|c| c.to_vec())
        .collect::<Vec<_>>()
}

fn build_streams_chunk(symbols: &[String], tfs: &[Timeframe]) -> String {
    // btcusdt@kline_1m/ethusdt@kline_1m/...
    let mut out = String::new();
    for (i, s) in symbols.iter().enumerate() {
        let lower = s.to_lowercase();
        for tf in tfs {
            if !out.is_empty() {
                out.push('/');
            }
            out.push_str(&lower);
            out.push_str("@kline_");
            out.push_str(tf.as_binance_interval());
        }
        if i > 10_000 {
            break; // safety
        }
    }
    out
}

pub async fn run_ws_and_poll(
    cfg: Arc<IngestConfig>,
    rest: BinanceRestClient,
    producer: FutureProducer,
    symbols: Vec<String>,
    last_map: LastCloseMap,
) -> Result<()> {
    // shared last-close map (для gap fill + poll)
    let last = Arc::new(RwLock::new(last_map));

    // ---- WS realtime (для cfg.ws_timeframes) ----
    let shards = shard_symbols(
        &symbols,
        cfg.realtime_ws_timeframes.len().max(1),
        cfg.ws_max_streams_per_conn,
    );

    for (idx, chunk) in shards.into_iter().enumerate() {
        let cfg = cfg.clone();
        let rest = rest.clone();
        let producer = producer.clone();
        let last = last.clone();

        tokio::spawn(async move {
            // backoff при реконнекте
            let mut backoff_ms: u64 = 300;

            loop {
                let streams = build_streams_chunk(&chunk, &cfg.realtime_ws_timeframes);
                let url = format!(
                    "{}/stream?streams={}",
                    cfg.ws_base_url.trim_end_matches('/'),
                    streams
                );

                info!(
                    "[ws#{idx}] connecting ({} symbols, {} tfs)",
                    chunk.len(),
                    cfg.realtime_ws_timeframes.len()
                );

                let conn = tokio_tungstenite::connect_async(url).await;
                let (mut ws, _) = match conn {
                    Ok(v) => {
                        backoff_ms = 300;
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
                                Err(_) => continue,
                            };

                            let ev = combined.data;

                            // пишем только закрытые бары
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
                            let prev = {
                                let guard = last.read().await;
                                guard.get(&(close_evt.symbol.clone(), close_evt.tf.clone())).copied()
                            };

                            if let Some(prev_close) = prev {
                                let step = tf_ms(tf);
                                if close_evt.close_time_ms > prev_close + step {
                                    match gap_fill_between_mp(
                                        &cfg,
                                        &rest,
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

                            if let Err(e) = send_close_mp(&producer, &cfg.topic_candles_close, &key, &close_evt).await {
                                warn!("[ws#{idx}] send_close failed: {e:#}");
                                break;
                            }

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
        let rest2 = rest.clone();
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
                        let rest2 = rest2.clone();
                        let producer2 = producer2.clone();
                        let last2 = last2.clone();
                        let sem = sem.clone();
                        let sym = sym.clone();

                        futs.push(async move {
                            let _permit = sem.acquire_owned().await.context("semaphore closed")?;

                            let ks = fetch_klines(&rest2, &sym, tf, 2, None).await?;
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
                                source_event_time_ms: Some(connections::now_ms()),
                            };

                            let prev = {
                                let guard = last2.read().await;
                                guard.get(&(evt.symbol.clone(), evt.tf.clone())).copied()
                            };

                            if prev.map(|p| evt.close_time_ms > p).unwrap_or(true) {
                                let key = format!("{}|{}", evt.symbol, evt.tf);
                                send_close_mp(&producer2, &cfg2.topic_candles_close, &key, &evt)
                                    .await?;

                                let mut guard = last2.write().await;
                                guard.insert(
                                    (evt.symbol.clone(), evt.tf.clone()),
                                    evt.close_time_ms,
                                );
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

    futures_util::future::pending::<()>().await;
    Ok(())
}
