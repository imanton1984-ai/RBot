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
use common::{timeframe::TimeFrame, MessageBus};
use common::AppConfig;
use futures::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tokio_tungstenite::tungstenite::Message;
use tracing::info;




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

    // Create message bus for publishing raw candles
    let message_bus = MessageBus::new_from_env().context("Failed to create message bus")?;

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

                    // Create the specific event structure Compute expects
                    #[derive(serde::Serialize)]
                    struct CandleCloseEvent {
                        symbol: String,
                        timeframe: String,
                        close_time: i64,
                        open: f64,
                        high: f64,
                        low: f64,
                        close: f64,
                        volume: f64,
                    }

                    let close_event = CandleCloseEvent {
                        symbol: ev.symbol.clone(),
                        timeframe: k.interval.clone(),
                        close_time: k.close_time,
                        open: str_f64(&k.open),
                        high: str_f64(&k.high),
                        low: str_f64(&k.low),
                        close: str_f64(&k.close),
                        volume: str_f64(&k.volume),
                    };

                    // Publish to the correct topic that Compute service listens to
                    let topic_name = std::env::var("TOPIC_CANDLES_CLOSE").unwrap_or_else(|_| "candles.close".to_string());
                    let message_bus_clone = message_bus.clone();
                    let symbol_key = ev.symbol.clone();

                    let _ = tokio::spawn(async move {
                        // Note: We use the topic name directly, assuming message_bus handles prefixes if configured
                        // or pass the raw topic name found in config/rust_bot.toml
                        if let Err(e) = message_bus_clone.publish(&topic_name, symbol_key.as_bytes(), &close_event).await {
                            tracing::error!("Failed to publish candle close event: {}", e);
                        } else {
                            tracing::debug!("Published close event: {} {}", symbol_key, k.interval);
                        }
                    });

                    // Optionally still send to legacy writers if needed for other purposes
                    if let Some(tx) = writers.get(&tf) {
                        let row = CandleRow {
                            time_ms: k.close_time,
                            symbol_id: sid,
                            symbol: ev.symbol.clone(),
                            open: str_f64(&k.open),
                            high: str_f64(&k.high),
                            low: str_f64(&k.low),
                            close: str_f64(&k.close),
                            volume: str_f64(&k.volume),
                        };
                        tracing::trace!("writer({}) received message: {:?}", tf.as_str(), row);
                        let _ = tx.send(TypedWriterMsg { rows: vec![row], source: DataSource::WebSocket });
                    }
                }
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(backoff)).await;
        backoff = (backoff * 2).min(cfg.binance.ws_reconnect_backoff_max_ms);
    }
}