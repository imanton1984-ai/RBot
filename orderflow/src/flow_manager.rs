/// Central manager for order-flow data.
///
/// - Fetches active symbols from the database (same approach as
///   [`connections::binance_websocket`]).
/// - Connects to Binance Futures combined WS stream
///   (`fstream.binance.com`) for `@depth20@100ms` + `@aggTrade`.
/// - Dispatches messages to per-symbol [`BookTracker`] / [`TradeTracker`].
/// - Assembles [`OrderFlowSnapshot`]s behind `Arc<RwLock<..>>` for
///   lock-free reads from the compute pipeline.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::StreamExt;
use tokio::sync::RwLock;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

use crate::book_tracker::BookTracker;
use crate::trade_tracker::TradeTracker;
use crate::types::{AggTrade, BookLevel, OrderFlowSnapshot};

// ── Configuration ───────────────────────────────────────────────────

/// Read `ORDERFLOW_WINDOW_SECS` from environment (default 60).
fn window_secs() -> u64 {
    std::env::var("ORDERFLOW_WINDOW_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60)
}

// ── Public API ──────────────────────────────────────────────────────

/// Thread-safe handle to live order-flow snapshots for every tracked
/// symbol.
#[derive(Clone)]
pub struct OrderFlowManager {
    snapshots: Arc<HashMap<String, Arc<RwLock<OrderFlowSnapshot>>>>,
}

impl OrderFlowManager {
    // ── Construction ────────────────────────────────────────────────

    /// Build the manager, populate the symbol list from the database,
    /// and spawn the WebSocket ingestion loop.
    ///
    /// Returns immediately — the WS loop runs in the background.
    pub async fn start(symbols: Vec<String>) -> Result<Self> {
        let window = window_secs();
        info!(
            symbols = symbols.len(),
            window_secs = window,
            "OrderFlowManager: initialising"
        );

        // Pre-allocate per-symbol state.
        let mut snapshot_map: HashMap<String, Arc<RwLock<OrderFlowSnapshot>>> =
            HashMap::with_capacity(symbols.len());

        for sym in &symbols {
            let snap = OrderFlowSnapshot {
                symbol: sym.clone(),
                ..Default::default()
            };
            snapshot_map.insert(sym.clone(), Arc::new(RwLock::new(snap)));
        }

        let snapshots = Arc::new(snapshot_map);

        // Spawn the WebSocket ingestion loop.
        let ws_snapshots = Arc::clone(&snapshots);
        let ws_symbols = symbols.clone();
        tokio::spawn(async move {
            if let Err(e) = ws_loop(ws_symbols, ws_snapshots, window).await {
                error!("OrderFlowManager WS loop terminated: {e:#}");
            }
        });

        Ok(Self { snapshots })
    }

    // ── Queries ─────────────────────────────────────────────────────

    /// Async snapshot read (acquires the `RwLock` read guard).
    pub async fn get_snapshot(&self, symbol: &str) -> Option<OrderFlowSnapshot> {
        let lock = self.snapshots.get(symbol)?;
        Some(lock.read().await.clone())
    }

    /// Non-blocking snapshot read via `try_read`.
    /// Returns `None` if the symbol is unknown **or** the lock is
    /// currently held for writing.
    pub fn get_snapshot_sync(&self, symbol: &str) -> Option<OrderFlowSnapshot> {
        let lock = self.snapshots.get(symbol)?;
        lock.try_read().ok().map(|g| g.clone())
    }

    /// Return a list of all tracked symbols.
    pub fn symbols(&self) -> Vec<&String> {
        self.snapshots.keys().collect()
    }
}

// ── DB helper ───────────────────────────────────────────────────────

/// Fetch active symbols from `market.pairs` — identical logic to
/// `connections::BinanceWsConfig::fetch_active_pairs_from_db`.
pub async fn fetch_symbols_from_db() -> Result<Vec<String>> {
    use tokio_postgres::NoTls;

    let db_url = std::env::var("DATABASE_URL")
        .or_else(|_| std::env::var("POSTGRES_CONNECTION_STRING"))
        .or_else(|_| std::env::var("POSTGRES_URL"))
        .unwrap_or_else(|_| {
            "postgresql://postgres:@localhost:5433/timescaledb_binance".to_string()
        });

    let (client, connection) =
        tokio_postgres::connect(&db_url, NoTls)
            .await
            .context("orderflow: DB connect")?;

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            error!("orderflow DB connection error: {e}");
        }
    });

    let rows = client
        .query(
            "SELECT symbol FROM market.pairs WHERE is_active = TRUE ORDER BY symbol_id",
            &[],
        )
        .await
        .context("orderflow: query active pairs")?;

    let symbols: Vec<String> = rows.iter().map(|r| r.get::<_, String>(0)).collect();
    info!(count = symbols.len(), "orderflow: loaded symbols from DB");
    Ok(symbols)
}

// ── WebSocket loop ──────────────────────────────────────────────────

/// Build the combined stream URL and run the consume-reconnect loop.
async fn ws_loop(
    symbols: Vec<String>,
    snapshots: Arc<HashMap<String, Arc<RwLock<OrderFlowSnapshot>>>>,
    window_secs: u64,
) -> Result<()> {
    let base_url = std::env::var("BINANCE_WS_BASE_URL")
        .unwrap_or_else(|_| "wss://fstream.binance.com/stream".to_string());

    // Build stream names: <symbol_lower>@depth20@100ms + <symbol_lower>@aggTrade
    let mut stream_names: Vec<String> = Vec::with_capacity(symbols.len() * 2);
    for sym in &symbols {
        let lower = sym.to_lowercase();
        stream_names.push(format!("{lower}@depth20@100ms"));
        stream_names.push(format!("{lower}@aggTrade"));
    }
    let joined = stream_names.join("/");
    let url = format!("{base_url}?streams={joined}");

    // Per-symbol trackers (mutable, owned by WS task exclusively).
    let mut books: HashMap<String, BookTracker> = HashMap::with_capacity(symbols.len());
    let mut trades: HashMap<String, TradeTracker> = HashMap::with_capacity(symbols.len());
    for sym in &symbols {
        books.insert(sym.clone(), BookTracker::new());
        trades.insert(sym.clone(), TradeTracker::new(window_secs));
    }

    let reconnect_base = Duration::from_secs(1);
    let reconnect_max = Duration::from_secs(60);
    let mut backoff = reconnect_base;

    loop {
        info!("orderflow: connecting to {url}");
        match connect_async(&url).await {
            Ok((ws, _)) => {
                info!("orderflow: WS connected");
                backoff = reconnect_base;

                let (_write, mut read) = ws.split();
                while let Some(msg) = read.next().await {
                    match msg {
                        Ok(Message::Text(text)) => {
                            handle_message(
                                text.as_ref(),
                                &mut books,
                                &mut trades,
                                &snapshots,
                            )
                            .await;
                        }
                        Ok(Message::Binary(bin)) => {
                            if let Ok(text) = std::str::from_utf8(&bin) {
                                handle_message(
                                    text,
                                    &mut books,
                                    &mut trades,
                                    &snapshots,
                                )
                                .await;
                            }
                        }
                        Ok(Message::Ping(_) | Message::Pong(_)) => {}
                        Ok(Message::Close(_)) => {
                            warn!("orderflow: WS close frame received");
                            break;
                        }
                        Err(e) => {
                            warn!("orderflow: WS recv error: {e}");
                            break;
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                warn!("orderflow: WS connect failed: {e}");
            }
        }

        warn!(backoff_ms = backoff.as_millis(), "orderflow: reconnecting");
        tokio::time::sleep(backoff).await;
        backoff = std::cmp::min(backoff * 2, reconnect_max);
    }
}

// ── Message dispatch ────────────────────────────────────────────────

/// Route a single combined-stream JSON message to the appropriate
/// tracker, recompute metrics, and update the shared snapshot.
///
/// Binance combined stream format:
/// ```json
/// { "stream": "btcusdt@aggTrade", "data": { ... } }
/// ```
async fn handle_message(
    raw: &str,
    books: &mut HashMap<String, BookTracker>,
    trades: &mut HashMap<String, TradeTracker>,
    snapshots: &HashMap<String, Arc<RwLock<OrderFlowSnapshot>>>,
) {
    // Lightweight JSON parse — we only need "stream" and "data".
    let envelope: serde_json::Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return,
    };

    let stream = match envelope.get("stream").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return,
    };
    let data = match envelope.get("data") {
        Some(d) => d,
        None => return,
    };

    // Determine symbol from stream name (e.g. "btcusdt@depth20@100ms").
    let symbol_lower = match stream.split('@').next() {
        Some(s) => s,
        None => return,
    };
    let symbol = symbol_lower.to_uppercase();

    if stream.contains("@depth20") {
        if let Some(book) = books.get_mut(&symbol) {
            if let Some((bids, asks)) = parse_depth20(data) {
                book.update(bids, asks);
                let trade_ref = trades.get(&symbol).map(|t| t as &TradeTracker);
                update_snapshot(&symbol, Some(book as &BookTracker), trade_ref, snapshots).await;
            }
        }
    } else if stream.contains("@aggTrade") {
        if let Some(tracker) = trades.get_mut(&symbol) {
            if let Some(agg) = parse_agg_trade(data) {
                tracker.push(agg);
                let book_ref = books.get(&symbol).map(|b| b as &BookTracker);
                update_snapshot(&symbol, book_ref, Some(tracker as &TradeTracker), snapshots).await;
            }
        }
    }
}

/// Recompute the full snapshot from current tracker states.
async fn update_snapshot(
    symbol: &str,
    book: Option<&BookTracker>,
    trade: Option<&TradeTracker>,
    snapshots: &HashMap<String, Arc<RwLock<OrderFlowSnapshot>>>,
) {
    let lock = match snapshots.get(symbol) {
        Some(l) => l,
        None => return,
    };

    let now_ms = chrono::Utc::now().timestamp_millis();

    let mut snap = lock.write().await;
    snap.timestamp_ms = now_ms;

    if let Some(b) = book {
        let bm = b.compute_metrics();
        snap.bid_ask_imbalance = bm.bid_ask_imbalance;
        snap.bid_depth_usd = bm.bid_depth_usd;
        snap.ask_depth_usd = bm.ask_depth_usd;
        snap.spread_bps = bm.spread_bps;
        snap.bid_wall_price = bm.bid_wall_price;
        snap.ask_wall_price = bm.ask_wall_price;
    }

    if let Some(t) = trade {
        let tm = t.compute_metrics();
        snap.volume_delta = tm.volume_delta;
        snap.volume_delta_ratio = tm.volume_delta_ratio;
        snap.buy_volume = tm.buy_volume;
        snap.sell_volume = tm.sell_volume;
        snap.large_trade_count = tm.large_trade_count;
        snap.large_trade_bias = tm.large_trade_bias;
    }

    // pressure_score = 0.4 * bid_ask_imbalance + 0.6 * volume_delta_ratio
    snap.pressure_score = 0.4 * snap.bid_ask_imbalance + 0.6 * snap.volume_delta_ratio;
}

// ── JSON parsers ────────────────────────────────────────────────────

/// Parse the `data` object of a `@depth20@100ms` message.
///
/// Expected shape:
/// ```json
/// { "bids": [["price","qty"], ...], "asks": [["price","qty"], ...] }
/// ```
fn parse_depth20(data: &serde_json::Value) -> Option<(Vec<BookLevel>, Vec<BookLevel>)> {
    let bids_arr = data.get("b").or_else(|| data.get("bids"))?.as_array()?;
    let asks_arr = data.get("a").or_else(|| data.get("asks"))?.as_array()?;

    let parse_levels = |arr: &[serde_json::Value]| -> Vec<BookLevel> {
        arr.iter()
            .filter_map(|entry| {
                let pair = entry.as_array()?;
                let price: f64 = pair.first()?.as_str()?.parse().ok()?;
                let qty: f64 = pair.get(1)?.as_str()?.parse().ok()?;
                Some(BookLevel { price, qty })
            })
            .collect()
    };

    Some((parse_levels(bids_arr), parse_levels(asks_arr)))
}

/// Parse the `data` object of an `@aggTrade` message.
///
/// Expected fields: `"T"` (trade time), `"p"` (price), `"q"` (qty),
/// `"m"` (is buyer maker).
fn parse_agg_trade(data: &serde_json::Value) -> Option<AggTrade> {
    let timestamp_ms = data.get("T")?.as_i64()?;
    let price: f64 = data.get("p")?.as_str()?.parse().ok()?;
    let qty: f64 = data.get("q")?.as_str()?.parse().ok()?;
    let is_buyer_maker = data.get("m")?.as_bool()?;

    Some(AggTrade {
        timestamp_ms,
        price,
        qty,
        is_buyer_maker,
    })
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_depth20_valid() {
        let json = serde_json::json!({
            "b": [["100.0", "5.0"], ["99.5", "3.0"]],
            "a": [["100.5", "2.0"], ["101.0", "4.0"]]
        });
        let (bids, asks) = parse_depth20(&json).unwrap();
        assert_eq!(bids.len(), 2);
        assert_eq!(asks.len(), 2);
        assert!((bids[0].price - 100.0).abs() < 1e-8);
        assert!((asks[1].qty - 4.0).abs() < 1e-8);
    }

    #[test]
    fn parse_agg_trade_valid() {
        let json = serde_json::json!({
            "e": "aggTrade",
            "s": "BTCUSDT",
            "T": 1_700_000_000_000_i64,
            "p": "43210.50",
            "q": "0.123",
            "m": true
        });
        let t = parse_agg_trade(&json).unwrap();
        assert_eq!(t.timestamp_ms, 1_700_000_000_000);
        assert!((t.price - 43210.50).abs() < 1e-8);
        assert!((t.qty - 0.123).abs() < 1e-8);
        assert!(t.is_buyer_maker);
        assert!(!t.is_buy());
    }

    #[test]
    fn parse_depth20_missing_field() {
        let json = serde_json::json!({"bids": []});
        assert!(parse_depth20(&json).is_none());
    }
}
