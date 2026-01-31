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
 *
 * WORKFLOW:
 * 1. ws_worker() establishes WebSocket connection to Binance
 * 2. Receives WebSocket messages continuously
 * 3. Parses incoming kline/candle data
 * 4. Validates that candles are closed before processing
 * 5. Routes data to appropriate timeframe writers
 * 6. Handles connection failures and reconnection
 */
use crate::market::candles::candle_common::*;
use crate::market::candles::candle_writer::{TypedWriterMsg, DataSource};
use anyhow::Result;
use common::timeframe::TimeFrame;
use common::AppConfig;
use futures::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tokio_tungstenite::tungstenite::Message;
use tracing::info;

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

                    let data_slice: &[u8] = match &msg {
                        Message::Text(t) => t.as_bytes(),   // Utf8Bytes
                        Message::Binary(b) => b.as_ref(),   // Bytes
                        Message::Ping(_) | Message::Pong(_) => continue,
                        Message::Close(_) => { tracing::warn!("WS close frame: {}", url); break; }
                        _ => continue,
                    };

                    let env: WsEnvelope = match serde_json::from_slice(data_slice) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let k = env.data.k;
                    if !k.is_closed {
                        continue;
                    }

                    tracing::debug!("Received closed candle: symbol={}, tf={}, time={}", k.symbol, k.interval, k.close_time);

                    let tf = match parse_tf(k.interval.as_ref()) {
                        Some(t) => t,
                        None => continue,
                    };

                    let sid = match symbol_to_id.get(k.symbol.as_ref()) {
                        Some(v) => *v,
                        None => continue,
                    };

                    // Skip WebSocket data that is older than or equal to the last loaded timestamp for this symbol
                    // This prevents duplicates between historical and real-time data
                    if let Some(last_time) = last_seen_per_symbol.get(&sid) {
                        if k.close_time <= *last_time {
                            continue; // Skip this candle as it's already in the database
                        }
                    }

                    let row = CandleRow {
                        time_ms: k.close_time,
                        symbol_id: sid,
                        open: str_f64(k.open.as_ref()),
                        high: str_f64(k.high.as_ref()),
                        low: str_f64(k.low.as_ref()),
                        close: str_f64(k.close.as_ref()),
                        volume: str_f64(k.volume.as_ref()),
                    };

                    if let Some(tx) = writers.get(&tf) {
                        let _ = tx.send(TypedWriterMsg { rows: vec![row], source: DataSource::WebSocket });
                    }
                }
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(backoff)).await;
        backoff = (backoff * 2).min(cfg.binance.ws_reconnect_backoff_max_ms);
    }
}