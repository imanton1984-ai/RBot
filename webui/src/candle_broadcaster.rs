// webui/src/candle_broadcaster.rs
//
// Background task that polls market.candles_live every 5 seconds
// and broadcasts CandleUpdate messages via WebSocket to all connected clients.
//
// This provides near-real-time chart updates without the client needing to
// poll the HTTP /api/candles endpoint every second.
//
// Architecture:
//   1. Try to consume from Redpanda topic "market.candles.live" (if available)
//   2. Fallback: poll market.candles_live table every 5 seconds
//   3. Broadcast CandleUpdate WsMessage for each changed candle

use sqlx::PgPool;
use tokio::sync::broadcast;
use crate::state::{WsMessage, CandleUpdate};
use tracing::{info, warn, debug};

/// Map timeframe string to minutes
fn tf_label_to_minutes(tf: &str) -> i32 {
    match tf {
        "1m"  => 1,
        "5m"  => 5,
        "15m" => 15,
        "1h"  => 60,
        "4h"  => 240,
        "1d"  => 1440,
        _     => 1,
    }
}

/// Spawn the candle broadcaster background task.
/// Polls candles_live every 5 seconds and broadcasts updates via WebSocket.
pub fn spawn_candle_broadcaster(pool: PgPool, ws_sender: broadcast::Sender<WsMessage>) {
    tokio::spawn(async move {
        info!("📡 Candle broadcaster started (5s poll interval)");

        // Try Redpanda consumer first, fallback to DB polling
        let use_kafka = try_kafka_consumer(pool.clone(), ws_sender.clone()).await;

        if !use_kafka {
            info!("📡 Candle broadcaster: using DB polling fallback (Kafka unavailable)");
            run_db_poll_loop(pool, ws_sender).await;
        }
    });
}

/// Try to start Kafka consumer for candle updates.
/// Returns true if Kafka consumer was started successfully (blocks forever).
/// Returns false immediately if Kafka is unavailable or topics don't exist.
async fn try_kafka_consumer(_pool: PgPool, ws_sender: broadcast::Sender<WsMessage>) -> bool {
    use rdkafka::config::ClientConfig;
    use rdkafka::consumer::{Consumer, StreamConsumer};
    use rdkafka::Message as KafkaMessage;
    use futures::StreamExt;

    let brokers = std::env::var("KAFKA_BROKERS")
        .unwrap_or_else(|_| "127.0.0.1:19092".to_string());

    // Quick TCP check: if Kafka is not reachable, skip immediately
    let addr = brokers.split(',').next().unwrap_or("127.0.0.1:19092");
    if tokio::net::TcpStream::connect(addr).await.is_err() {
        debug!("📡 Kafka not reachable at {}, using DB polling", addr);
        return false;
    }

    // Try to create consumer for candle topics
    let consumer: Result<StreamConsumer, _> = ClientConfig::new()
        .set("bootstrap.servers", &brokers)
        .set("group.id", "webui-candle-broadcaster")
        .set("auto.offset.reset", "latest")
        .set("enable.auto.commit", "true")
        .set("session.timeout.ms", "6000")
        .set("fetch.wait.max.ms", "500")
        .create();

    let consumer = match consumer {
        Ok(c) => c,
        Err(e) => {
            debug!("📡 Cannot create Kafka consumer: {}", e);
            return false;
        }
    };

    // Check if our candle topic actually exists via metadata
    let topic = "market.candles.live";
    let metadata = consumer.fetch_metadata(Some(topic), std::time::Duration::from_secs(3));
    match metadata {
        Ok(md) => {
            let topics = md.topics();
            if topics.is_empty() || topics.iter().all(|t| t.partitions().is_empty()) {
                debug!("📡 Kafka topic '{}' does not exist, using DB polling", topic);
                return false;
            }
        }
        Err(e) => {
            debug!("📡 Cannot fetch Kafka metadata: {}, using DB polling", e);
            return false;
        }
    }

    // Subscribe to the verified topic
    if let Err(e) = consumer.subscribe(&[topic]) {
        debug!("📡 Cannot subscribe to {}: {}", topic, e);
        return false;
    }

    info!("📡 Candle broadcaster: consuming from Kafka topic '{}'", topic);

    // Process Kafka messages; on repeated errors, fall back to DB polling
    let mut consecutive_errors = 0u32;
    let mut stream = consumer.stream();
    while let Some(result) = stream.next().await {
        match result {
            Ok(msg) => {
                consecutive_errors = 0;
                if let Some(payload) = msg.payload() {
                    if let Ok(update) = serde_json::from_slice::<CandleLiveMessage>(payload) {
                        let tf_minutes = tf_label_to_minutes(&update.timeframe);
                        let tf_ms = tf_minutes as i64 * 60_000;
                        let close_time_ms = update.open_time_ms + tf_ms;

                        let candle_update = CandleUpdate {
                            pair: update.symbol.clone(),
                            tf: tf_minutes,
                            t: close_time_ms,
                            o: update.open,
                            h: update.high,
                            l: update.low,
                            c: update.close,
                            v: update.volume,
                            is_closed: false,
                        };

                        let _ = ws_sender.send(WsMessage::CandleUpdate(candle_update));
                    }
                }
            }
            Err(e) => {
                consecutive_errors += 1;
                if consecutive_errors <= 2 {
                    debug!("📡 Kafka consumer error ({}): {}", consecutive_errors, e);
                }
                if consecutive_errors >= 5 {
                    warn!("📡 Too many Kafka errors, switching to DB polling");
                    return false;
                }
            }
        }
    }

    true
}

/// Kafka message format for candle live data
#[derive(serde::Deserialize)]
struct CandleLiveMessage {
    symbol: String,
    timeframe: String,
    open_time_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

/// Fallback: poll candles_live table every 5 seconds and broadcast updates.
async fn run_db_poll_loop(pool: PgPool, ws_sender: broadcast::Sender<WsMessage>) {
    use sqlx::Row;
    use std::collections::HashMap;

    // Track last known candle state to avoid broadcasting unchanged data
    let mut last_state: HashMap<(String, String), (f64, f64, f64, f64)> = HashMap::new();

    loop {
        // Query all recently updated candles from candles_live
        // Only fetch candles updated in the last 30 seconds to limit scope
        let rows = sqlx::query(
            "SELECT symbol, timeframe, open_time_ms,
                    open, high, low, close, volume
             FROM market.candles_live
             WHERE open_time_ms > (EXTRACT(EPOCH FROM now()) * 1000 - 120000)::bigint
             ORDER BY open_time_ms DESC"
        )
        .fetch_all(&pool)
        .await;

        match rows {
            Ok(rows) => {
                for row in &rows {
                    let symbol: String = row.get("symbol");
                    let timeframe: String = row.get("timeframe");
                    let open_time_ms: i64 = row.get("open_time_ms");
                    let open: f64 = row.get("open");
                    let high: f64 = row.get("high");
                    let low: f64 = row.get("low");
                    let close: f64 = row.get("close");
                    let volume: f64 = row.get("volume");

                    // Check if candle actually changed
                    let key = (symbol.clone(), timeframe.clone());
                    let new_state = (open, high, low, close);

                    if let Some(prev) = last_state.get(&key) {
                        if *prev == new_state {
                            continue; // No change, skip broadcast
                        }
                    }

                    last_state.insert(key, new_state);

                    let tf_minutes = tf_label_to_minutes(&timeframe);
                    let tf_ms = tf_minutes as i64 * 60_000;
                    let close_time_ms = open_time_ms + tf_ms;

                    let candle_update = CandleUpdate {
                        pair: symbol,
                        tf: tf_minutes,
                        t: close_time_ms,
                        o: open,
                        h: high,
                        l: low,
                        c: close,
                        v: volume,
                        is_closed: false,
                    };

                    let _ = ws_sender.send(WsMessage::CandleUpdate(candle_update));
                }

                // Clean up old entries from tracking map (keep it bounded)
                if last_state.len() > 5000 {
                    last_state.clear();
                }
            }
            Err(e) => {
                debug!("📡 Candle broadcaster poll error: {}", e);
            }
        }

        // Sleep 5 seconds before next poll
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    }
}
