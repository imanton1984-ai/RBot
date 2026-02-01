/*
 * FILE: candle_ws.rs
 *
 * PURPOSE:
 * This module handles real-time candle data retrieval from WebSocket streams.
 * It manages the connection to Binance WebSocket API and processes live candle updates.
 *
 * RESPONSIBILITIES:
 * - Establishes and maintains WebSocket connections
 * - Processes real-time candle data from WebSocket streams
 * - Filters and validates incoming candle data
 * - Forwards processed data to appropriate writers
 * - Handles reconnection logic for robust operation
 * - Sends live candle updates to the LiveWriter
 *
 * WORKFLOW:
 * 1. ws_worker() establishes WebSocket connection to Binance
 * 2. Receives WebSocket messages continuously
 * 3. Parses incoming kline/candle data
 * 4. Validates that candles are closed before processing
 * 5. Routes data to appropriate timeframe writers
 * 6. Handles connection failures and reconnection
 * 7. Forwards live updates to LiveWriter for market.candles_live table
 */
use crate::market::candles::candle_common::*;
use crate::market::candles::candle_writer::{TypedWriterMsg, DataSource, LiveCandle};
use anyhow::{Context, Result};
use common::timeframe::TimeFrame;
use common::AppConfig;
use futures::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tokio_tungstenite::tungstenite::Message;
use tracing::info;
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use std::time::Duration;

// Define the candle close event structure that matches what the compute service expects
#[derive(serde::Serialize)]
struct CandleCloseEvent {
    symbol: String,
    timeframe: String,
    close_time: i64,
}

// Create a Kafka producer for publishing candle close events
async fn create_kafka_producer() -> Result<FutureProducer> {
    let brokers = std::env::var("KAFKA_BROKERS").unwrap_or_else(|_| "127.0.0.1:19092".to_string());
    let producer: FutureProducer = ClientConfig::new()
        .set("bootstrap.servers", &brokers)
        .set("message.timeout.ms", "5000")
        .create()
        .context("Producer creation failed")?;

    Ok(producer)
}

// Publish a candle close event to the Kafka topic
async fn publish_candle_close_event(producer: &FutureProducer, event: &CandleCloseEvent) -> Result<()> {
    let topic = std::env::var("KAFKA_CANDLES_CLOSE_TOPIC").unwrap_or_else(|_| "candles.close".to_string());
    let payload = serde_json::to_string(event).context("Failed to serialize candle close event")?;

    // Send the message and handle the result
    match producer
        .send(
            FutureRecord::to(&topic)
                .key(&event.symbol)  // Use symbol as key for partitioning
                .payload(&payload),
            Duration::from_secs(1),
        )
        .await
    {
        Ok(_) => Ok(()),  // Message sent successfully
        Err((err, _)) => Err(anyhow::anyhow!("Kafka send error: {}", err)),
    }
}



pub async fn run_kline_ws(
    ws_url: String,
    streams: Vec<String>,
    tx_live: mpsc::Sender<LiveCandle>,
) -> Result<()> {
    // Binance combined stream: wss://.../stream?streams=a/b/c
    let full = format!("{}?streams={}", ws_url.trim_end_matches('/'), streams.join("/"));
    info!("WS connect: streams={}", streams.len());

    let (ws, _resp) = tokio_tungstenite::connect_async(&full).await.context("ws connect failed")?;
    let (_, mut read) = ws.split(); // Split the WebSocket into sender/receiver parts

    while let Some(msg) = read.next().await {
        let msg = msg.context("WebSocket message error")?;
        if !msg.is_text() { continue; }
        let txt = msg.into_text().context("Failed to convert message to text")?;

        // envelope with {stream,data}
        let env: BinanceWsEnvelope<serde_json::Value> = match serde_json::from_str(&txt) {
            Ok(v) => v,
            Err(e) => { tracing::debug!("ws json parse envelope err: {}", e); continue; }
        };

        // live routing: kline updates → tx_live (без фильтра по close)
        if env.data.get("e").and_then(|v| v.as_str()) == Some("kline") {
            let ev: KlineEvent = match serde_json::from_value(env.data) {
                Ok(v) => v,
                Err(e) => { tracing::debug!("ws kline parse err: {}", e); continue; }
            };

            let k = ev.k;
            let lc = LiveCandle {
                symbol: ev.symbol,
                timeframe: k.interval,
                open_time_ms: k.open_time,
                close_time_ms: k.close_time,
                open: str_f64(&k.open),
                high: str_f64(&k.high),
                low: str_f64(&k.low),
                close: str_f64(&k.close),
                volume: str_f64(&k.volume),
                trades: k.trades,
                is_final: k.is_final,
                event_time_ms: ev.event_time,
                source: "ws",
            };

            // если канал переполнен — лучше дропнуть, чем убить систему
            if tx_live.try_send(lc).is_err() {
                // drop
            }
        }
    }

    Ok(())
}

pub async fn ws_worker(
    cfg: Arc<AppConfig>,
    url: String,
    symbol_to_id: Arc<HashMap<String, i64>>,
    last_seen_per_symbol: Arc<HashMap<i64, i64>>, // Last seen timestamps per symbol_id
    writers: Arc<HashMap<TimeFrame, mpsc::UnboundedSender<TypedWriterMsg>>>,
    mut shutdown: watch::Receiver<bool>,
) -> Result<()> {
    let mut backoff = cfg.binance.ws_reconnect_backoff_ms.max(200);
    let ping_every = tokio::time::Duration::from_secs(cfg.binance.ws_ping_interval_sec.max(5) as u64);

    // Create Kafka producer for publishing candle close events
    let kafka_producer = create_kafka_producer().await.context("Failed to create Kafka producer")?;

    loop {
        if *shutdown.borrow() {
            return Ok(());
        }

        info!("WS connect: {}", url);
        let conn = tokio_tungstenite::connect_async(&url).await;
        let (mut ws, _) = match conn {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("WS connect failed: {} err={}", url, e);
                tokio::time::sleep(tokio::time::Duration::from_millis(backoff)).await;
                backoff = (backoff * 2).min(cfg.binance.ws_reconnect_backoff_max_ms);
                continue;
            }
        };

        backoff = cfg.binance.ws_reconnect_backoff_ms.max(200);
        let mut ping = tokio::time::interval(ping_every);

        loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        let _ = ws.send(Message::Close(None)).await;
                        return Ok(());
                    }
                }

                _ = ping.tick() => {
                    // tungstenite 0.28: Ping(Bytes)
                    let _ = ws.send(Message::Ping(bytes::Bytes::new())).await;
                }

                msg = ws.next() => {
                    let msg = match msg {
                        None => { tracing::warn!("WS closed by server: {}", url); break; }
                        Some(Err(e)) => { tracing::warn!("WS read error: {} err={}", url, e); break; }
                        Some(Ok(m)) => m,
                    };

                    tracing::trace!("Received raw ws message: {:?}", msg);

                    let data_slice: &[u8] = match &msg {
                        Message::Text(t) => t.as_bytes(),
                        Message::Binary(b) => b.as_ref(),
                        Message::Ping(_) | Message::Pong(_) => continue,
                        Message::Close(_) => { tracing::warn!("WS close frame: {}", url); break; }
                        _ => continue,
                    };

                    // Use the correct parsing logic
                    let env: BinanceWsEnvelope<serde_json::Value> = match serde_json::from_slice(data_slice) {
                        Ok(v) => v,
                        Err(e) => {
                            tracing::debug!("ws json parse envelope err: {}", e);
                            continue;
                        }
                    };

                    if env.data.get("e").and_then(|v| v.as_str()) != Some("kline") {
                        continue;
                    }

                    let ev: KlineEvent = match serde_json::from_value(env.data) {
                        Ok(v) => v,
                        Err(e) => {
                            tracing::debug!("ws kline parse err: {}", e);
                            continue;
                        }
                    };

                    let k = ev.k;
                    if !k.is_final {
                        tracing::trace!("Skipping non-closed candle: symbol={}, tf={}, time={}", ev.symbol, k.interval, k.close_time);
                        continue;
                    }

                    tracing::debug!("Received closed candle: symbol={}, tf={}, time={}", ev.symbol, k.interval, k.close_time);

                    let tf = match parse_tf(&k.interval) {
                        Some(t) => t,
                        None => continue,
                    };

                    let sid = match symbol_to_id.get(&ev.symbol) {
                        Some(v) => *v,
                        None => continue,
                    };

                    if let Some(last_time) = last_seen_per_symbol.get(&sid) {
                        if k.close_time <= *last_time {
                            continue;
                        }
                    }

                    let row = CandleRow {
                        time_ms: k.close_time,
                        symbol_id: sid,
                        open: str_f64(&k.open),
                        high: str_f64(&k.high),
                        low: str_f64(&k.low),
                        close: str_f64(&k.close),
                        volume: str_f64(&k.volume),
                    };

                    if let Some(tx) = writers.get(&tf) {
                        tracing::trace!("writer({}) received message: {:?}", tf.as_str(), row);
                        let _ = tx.send(TypedWriterMsg { rows: vec![row], source: DataSource::WebSocket });
                    }

                    // Publish candle close event to Kafka topic
                    let close_event = CandleCloseEvent {
                        symbol: ev.symbol.clone(),
                        timeframe: k.interval.clone(),
                        close_time: k.close_time,
                    };

                    // Publish the event asynchronously without blocking the main loop
                    let producer_clone = kafka_producer.clone();
                    let _ = tokio::spawn(async move {
                        if let Err(e) = publish_candle_close_event(&producer_clone, &close_event).await {
                            tracing::error!("Failed to publish candle close event to Kafka: {}", e);
                        } else {
                            tracing::debug!("Published candle close event: {} {} at {}", close_event.symbol, close_event.timeframe, close_event.close_time);
                        }
                    });
                }
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(backoff)).await;
        backoff = (backoff * 2).min(cfg.binance.ws_reconnect_backoff_max_ms);
    }
}