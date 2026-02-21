// webui/src/api/handlers.rs — Full WebUI API Handlers
//
// Endpoints for:
//   - Market data (pairs, candles, indicators, signals)
//   - Balance (from Binance Futures account via settings)
//   - PnL (calculated without leverage = real PnL)
//   - Connections (DB, Redpanda, REST API, WebSocket, Account)
//   - Trading Options (order_settings.toml)
//   - Order Options (order_manager.toml + risk_manager.toml)
//   - Alerts (from risk.alerts Kafka topic / stub)
//   - Trading control (start/stop, emergency stop)
//   - Manual order placement with TP/SL

use axum::{extract::{Query, State}, Json, http::StatusCode};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use crate::state::{
    AppState, BalanceInfo, PnlOverview,
    ConnectionStatus, TradingOptionsPayload,
    AutoTradingState, ManualOrderRequest, CandlesLeftUpdateRequest,
};

// ─── Query parameters ─────────────────────────────────────────────────
#[derive(Debug, Deserialize)]
pub struct PairsQuery { pub search: Option<String>, pub limit: Option<i64> }

#[derive(Debug, Deserialize)]
pub struct CandlesQuery { pub pair: String, pub tf: i32, pub limit: Option<i64> }

#[derive(Debug, Deserialize)]
pub struct IndicatorsQuery {
    pub pair: String,
    pub tf: i32,
    #[serde(rename = "type")]
    pub indicator_type: String,
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct SignalsQuery { pub pair: Option<String>, pub tf: Option<i32>, pub limit: Option<i64> }

#[derive(Debug, Deserialize)]
pub struct AlertsQuery { pub limit: Option<i64> }

#[derive(Debug, Deserialize)]
pub struct PnlQuery { pub range: Option<String> }

// ─── Response types ────────────────────────────────────────────────────
#[derive(Debug, Serialize)]
pub struct PairInfo { pub symbol: String, pub symbol_id: i64 }

#[derive(Debug, Serialize)]
pub struct CandleResponse { pub t: i64, pub o: f64, pub h: f64, pub l: f64, pub c: f64, pub v: f64 }

#[derive(Debug, Serialize)]
pub struct IndicatorResponse { pub t: i64, pub value: Option<f64> }

#[derive(Debug, Serialize)]
pub struct MarketSummary {
    pub price: f64,
    pub volume_24h: f64,
    pub change_24h: f64,
    pub change_1h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
}

#[derive(Debug, Serialize)]
pub struct SignalRow {
    pub id: String,
    pub time: String,
    pub pair: String,
    pub tf: i32,
    pub side: i16,
    pub score: f32,
    pub entry_price: f64,
    pub sl_price: f64,
    pub tp_price: f64,
    pub strategy: String,
    pub status: String,
    pub p_super: f32,
}

#[derive(Debug, Serialize)]
pub struct PositionRow {
    pub id: i64,
    pub pair: String,
    pub side: String,
    pub qty: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub pnl_usdt: f64,
    pub pnl_pct: f64,
    pub status: String,
    pub open_time: String,
    pub candles_left: i16,
    pub leverage: u16,
    pub tf_minutes: i16,
}

#[derive(Debug, Serialize)]
pub struct HistoryRow {
    pub id: i64,
    pub pair: String,
    pub side: String,
    pub qty: f64,
    pub entry_price: f64,
    pub close_price: f64,
    pub close_type: String,
    pub pnl_usdt: f64,
    pub pnl_pct: f64,
    pub open_time: String,
    pub close_time: String,
}

#[derive(Debug, Serialize)]
pub struct AlertRow {
    pub id: i64,
    pub pair: String,
    pub alert_type: String,
    pub time_ago: String,
    pub message: String,
    pub severity: String,
    pub source: String,
    pub timestamp: String,
}

// ─── Helpers ───────────────────────────────────────────────────────────

/// Map tf_minutes → table name
fn candle_table(tf: i32) -> &'static str {
    match tf {
        1    => "market.candles_1m",
        5    => "market.candles_5m",
        15   => "market.candles_15m",
        60   => "market.candles_1h",
        240  => "market.candles_4h",
        1440 => "market.candles_1d",
        _    => "market.candles_1m",
    }
}

// ─── GET /api/pairs ────────────────────────────────────────────────────
pub async fn get_pairs(
    Query(query): Query<PairsQuery>,
    State(state): State<AppState>,
) -> Result<Json<Vec<PairInfo>>, StatusCode> {
    let limit = query.limit.unwrap_or(500); // Default 500 for full list
    let search = query.search.unwrap_or_default();

    let rows = sqlx::query(
        "SELECT symbol, symbol_id FROM market.pairs
         WHERE is_active = true AND ($1 = '' OR symbol ILIKE $1)
         ORDER BY volume_24h_usdt DESC NULLS LAST, symbol
         LIMIT $2"
    )
    .bind(format!("%{}%", search))
    .bind(limit)
    .fetch_all(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("Pairs error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let pairs = rows.iter().map(|r| PairInfo {
        symbol: r.get("symbol"),
        symbol_id: r.get("symbol_id"),
    }).collect();

    Ok(Json(pairs))
}

// ─── GET /api/market/summary ───────────────────────────────────────────
pub async fn get_market_summary(
    Query(query): Query<CandlesQuery>,
    State(state): State<AppState>,
) -> Result<Json<MarketSummary>, StatusCode> {
    let row = sqlx::query(
        "WITH latest AS (
            SELECT close FROM market.candles_1m
            WHERE symbol = $1 ORDER BY time DESC LIMIT 1
        ), stats AS (
            SELECT
                COALESCE(MAX(high), 0) as high_24h,
                COALESCE(MIN(low), 0) as low_24h,
                COALESCE(SUM(volume), 0) as volume_24h
            FROM market.candles_1m
            WHERE symbol = $1 AND time >= now() - INTERVAL '24 hours'
        ), price_24h_ago AS (
            SELECT close as price_24h FROM market.candles_1m
            WHERE symbol = $1 AND time <= now() - INTERVAL '24 hours'
            ORDER BY time DESC LIMIT 1
        ), price_1h_ago AS (
            SELECT close as price_1h FROM market.candles_1m
            WHERE symbol = $1 AND time <= now() - INTERVAL '1 hour'
            ORDER BY time DESC LIMIT 1
        )
        SELECT
            (SELECT close FROM latest) as last_price,
            (SELECT high_24h FROM stats),
            (SELECT low_24h FROM stats),
            (SELECT volume_24h FROM stats),
            (SELECT price_24h FROM price_24h_ago) as price_24h,
            (SELECT price_1h FROM price_1h_ago) as price_1h"
    )
    .bind(&query.pair)
    .fetch_optional(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("Market summary error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    match row {
        Some(r) => {
            let price: f64 = r.try_get("last_price").unwrap_or(0.0);
            let price_24h: f64 = r.try_get("price_24h").unwrap_or(price);
            let price_1h: f64 = r.try_get("price_1h").unwrap_or(price);
            let change_24h = if price_24h > 0.0 { (price - price_24h) / price_24h * 100.0 } else { 0.0 };
            let change_1h = if price_1h > 0.0 { (price - price_1h) / price_1h * 100.0 } else { 0.0 };
            Ok(Json(MarketSummary {
                price,
                volume_24h: r.try_get("volume_24h").unwrap_or(0.0),
                change_24h,
                change_1h,
                high_24h: r.try_get("high_24h").unwrap_or(0.0),
                low_24h: r.try_get("low_24h").unwrap_or(0.0),
            }))
        }
        None => Ok(Json(MarketSummary {
            price: 0.0, volume_24h: 0.0, change_24h: 0.0,
            change_1h: 0.0, high_24h: 0.0, low_24h: 0.0,
        })),
    }
}

// ─── GET /api/candles ──────────────────────────────────────────────────
pub async fn get_candles(
    Query(query): Query<CandlesQuery>,
    State(state): State<AppState>,
) -> Result<Json<Vec<CandleResponse>>, StatusCode> {
    let limit = query.limit.unwrap_or(500).min(5000);
    let table = candle_table(query.tf);

    let sql = format!(
        "SELECT c.time_ms as t, c.open as o, c.high as h, c.low as l, c.close as c, c.volume as v
         FROM {} c
         JOIN market.pairs p ON c.symbol_id = p.symbol_id
         WHERE p.symbol = $1
         ORDER BY c.time DESC
         LIMIT $2",
        table
    );

    let rows = sqlx::query(&sql)
        .bind(&query.pair)
        .bind(limit)
        .fetch_all(&state.db_pool)
        .await
        .map_err(|e| { tracing::error!("Candles error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let candles: Vec<CandleResponse> = rows.iter().rev().map(|r| CandleResponse {
        t: r.get("t"),
        o: r.get("o"),
        h: r.get("h"),
        l: r.get("l"),
        c: r.get("c"),
        v: r.get("v"),
    }).collect();

    Ok(Json(candles))
}

// ─── GET /api/indicators ──────────────────────────────────────────────
pub async fn get_indicators(
    Query(query): Query<IndicatorsQuery>,
    State(state): State<AppState>,
) -> Result<Json<Vec<IndicatorResponse>>, StatusCode> {
    let column = match query.indicator_type.as_str() {
        "ema" | "ema20" | "ema_20"   => "ema_20",
        "ema50" | "ema_50"           => "ema_50",
        "ema200" | "ema_200"         => "ema_200",
        "rsi"                        => "rsi",
        "macd"                       => "macd",
        "macd_signal"                => "macd_signal",
        "macd_hist"                  => "macd_hist",
        "adx"                        => "adx",
        "cci"                        => "cci",
        "atr"                        => "atr",
        "bb_upper"                   => "bb_upper",
        "bb_mid"                     => "bb_mid",
        "bb_lower"                   => "bb_lower",
        "stoch_k"                    => "stoch_k",
        "stoch_d"                    => "stoch_d",
        "obv"                        => "obv",
        "vwap"                       => "vwap",
        "volume_spike"               => "volume_spike",
        "sma"                        => "sma",
        _                            => "rsi",
    };

    let limit = query.limit.unwrap_or(500).min(5000);

    let sql = format!(
        "SELECT time_ms as t, {}::double precision as value
         FROM market.indicators_wide
         WHERE symbol = $1 AND tf_minutes = $2
         ORDER BY time DESC
         LIMIT $3",
        column
    );

    let rows = sqlx::query(&sql)
        .bind(&query.pair)
        .bind(query.tf as i16)
        .bind(limit)
        .fetch_all(&state.db_pool)
        .await
        .map_err(|e| { tracing::error!("Indicators error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let indicators: Vec<IndicatorResponse> = rows.iter().rev().map(|r| IndicatorResponse {
        t: r.get("t"),
        value: r.try_get("value").ok(),
    }).collect();

    Ok(Json(indicators))
}

// ─── GET /api/signals ─────────────────────────────────────────────────
pub async fn get_signals(
    Query(query): Query<SignalsQuery>,
    State(state): State<AppState>,
) -> Result<Json<Vec<SignalRow>>, StatusCode> {
    let limit = query.limit.unwrap_or(100);

    let rows = sqlx::query(
        "SELECT
            time, time_ms, symbol, tf_minutes, side, combined_score,
            entry_price, sl_price, tp_price, strategy, p_super
         FROM trade.super_entry_signals
         WHERE ($1::text IS NULL OR symbol = $1)
           AND ($2::int IS NULL OR tf_minutes = $2)
         ORDER BY time DESC
         LIMIT $3"
    )
    .bind(&query.pair)
    .bind(query.tf)
    .bind(limit)
    .fetch_all(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("Signals error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let signals: Vec<SignalRow> = rows.iter().map(|r| {
        let time: chrono::DateTime<chrono::Utc> = r.get("time");
        let time_ms: i64 = r.get("time_ms");
        SignalRow {
            id: format!("{}", time_ms),
            time: time.to_rfc3339(),
            pair: r.get("symbol"),
            tf: r.get::<i16, _>("tf_minutes") as i32,
            side: r.get("side"),
            score: r.get("combined_score"),
            entry_price: r.get("entry_price"),
            sl_price: r.get("sl_price"),
            tp_price: r.get("tp_price"),
            strategy: r.try_get("strategy").unwrap_or("super_entry_v1".to_string()),
            status: "active".to_string(),
            p_super: r.get("p_super"),
        }
    }).collect();

    Ok(Json(signals))
}

// ─── GET /api/positions/open ───────────────────────────────────────────
pub async fn get_open_positions(
    State(state): State<AppState>,
) -> Result<Json<Vec<PositionRow>>, StatusCode> {
    let rows = sqlx::query(
        "SELECT p.id,
                COALESCE(p.symbol, mp.symbol) as pair,
                CASE WHEN p.side = 1 THEN 'LONG' ELSE 'SHORT' END as side,
                p.qty,
                p.entry_price,
                COALESCE(p.entry_price, 0) as current_price,
                COALESCE(p.sl_price, 0) as stop_loss,
                COALESCE(p.tp_price, 0) as take_profit,
                COALESCE(p.unrealized_pnl, 0) as pnl_usdt,
                0::double precision as pnl_pct,
                COALESCE(p.candles_left, 0)::smallint as candles_left,
                COALESCE(p.leverage, 10)::smallint as leverage,
                COALESCE(p.tf_minutes, 60)::smallint as tf_minutes,
                p.opened_at as open_time
         FROM trade.positions p
         JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
         WHERE p.status = 1
         ORDER BY p.opened_at DESC"
    )
    .fetch_all(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("Positions error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let positions: Vec<PositionRow> = rows.iter().map(|r| {
        let open_time: chrono::DateTime<chrono::Utc> = r.get("open_time");
        let leverage: i16 = r.try_get("leverage").unwrap_or(10);
        PositionRow {
            id: r.get("id"),
            pair: r.get("pair"),
            side: r.get("side"),
            qty: r.get("qty"),
            entry_price: r.get("entry_price"),
            current_price: r.get("current_price"),
            stop_loss: r.get("stop_loss"),
            take_profit: r.get("take_profit"),
            pnl_usdt: r.get("pnl_usdt"),
            pnl_pct: r.get("pnl_pct"),
            status: "open".to_string(),
            open_time: open_time.to_rfc3339(),
            candles_left: r.get("candles_left"),
            leverage: leverage as u16,
            tf_minutes: r.get("tf_minutes"),
        }
    }).collect();

    Ok(Json(positions))
}

// ─── GET /api/positions/history ────────────────────────────────────────
pub async fn get_positions_history(
    State(state): State<AppState>,
) -> Result<Json<Vec<HistoryRow>>, StatusCode> {
    let rows = sqlx::query(
        "SELECT id, position_id, symbol, side, qty, entry_price,
                COALESCE(exit_price, entry_price) as close_price,
                COALESCE(close_reason, 'manual') as close_type,
                COALESCE(realized_pnl, 0) as pnl_usdt,
                COALESCE(realized_pnl_pct, 0) as pnl_pct,
                opened_at as open_time,
                closed_at as close_time
         FROM trade.position_history
         ORDER BY closed_at DESC
         LIMIT 500"
    )
    .fetch_all(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("History error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let history: Vec<HistoryRow> = rows.iter().map(|r| {
        let open_time: chrono::DateTime<chrono::Utc> = r.get("open_time");
        let close_time: chrono::DateTime<chrono::Utc> = r.get("close_time");
        let side_i16: i16 = r.get("side");
        HistoryRow {
            id: r.get("id"),
            pair: r.get("symbol"),
            side: if side_i16 == 1 { "LONG".to_string() } else { "SHORT".to_string() },
            qty: r.get("qty"),
            entry_price: r.get("entry_price"),
            close_price: r.get("close_price"),
            close_type: r.get("close_type"),
            pnl_usdt: r.get("pnl_usdt"),
            pnl_pct: r.get("pnl_pct"),
            open_time: open_time.to_rfc3339(),
            close_time: close_time.to_rfc3339(),
        }
    }).collect();

    Ok(Json(history))
}

// ─── GET /api/pnl/overview ─────────────────────────────────────────────
pub async fn get_pnl_overview(
    Query(_query): Query<PnlQuery>,
    State(state): State<AppState>,
) -> Result<Json<PnlOverview>, StatusCode> {
    // Closed PnL — real PnL without leverage effect
    // realized_pnl in position_history is already the real USDT PnL
    let row = sqlx::query(
        "SELECT
            COALESCE(SUM(realized_pnl), 0) as closed_pnl,
            COUNT(*) FILTER (WHERE realized_pnl > 0) as wins,
            COUNT(*) as total_trades
         FROM trade.position_history
         WHERE closed_at > NOW() - INTERVAL '30 days'"
    )
    .fetch_one(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("PnL error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let closed_pnl: f64 = row.get("closed_pnl");
    let wins: i64 = row.get("wins");
    let total_trades: i64 = row.get("total_trades");
    let win_rate = if total_trades > 0 { (wins as f64 / total_trades as f64) * 100.0 } else { 0.0 };

    // Unrealized PnL from open positions (real PnL, not leveraged)
    // unrealized_pnl = direction * (current_price - entry_price) * qty
    // This is already in USDT without leverage amplification
    let unrealized_row = sqlx::query(
        "SELECT COALESCE(SUM(unrealized_pnl), 0) as unrealized
         FROM trade.positions WHERE status = 1"
    )
    .fetch_one(&state.db_pool)
    .await
    .ok();

    let unrealized_pnl = unrealized_row
        .and_then(|r| r.try_get::<f64, _>("unrealized").ok())
        .unwrap_or(0.0);

    // Get overall balance from Binance account (wallet_balance)
    // This is the real balance without leverage
    let balance_info = get_binance_balance_internal().await;
    let overall_balance = balance_info.wallet_balance;

    Ok(Json(PnlOverview {
        closed_pnl,
        unrealized_pnl,
        win_rate,
        today_trades: total_trades,
        equity_points: vec![],
        overall_balance,
    }))
}

// ─── GET /api/balance ──────────────────────────────────────────────────
pub async fn get_balance(
    State(state): State<AppState>,
) -> Result<Json<BalanceInfo>, StatusCode> {
    // Read real balance from Binance Futures account
    let mut balance = get_binance_balance_internal().await;

    // Also include in_orders from our DB (real margin used, not leveraged notional)
    let in_orders_row = sqlx::query(
        "SELECT COALESCE(SUM(qty * entry_price / GREATEST(COALESCE(leverage, 10), 1)), 0) as in_orders
         FROM trade.positions WHERE status = 1"
    )
    .fetch_optional(&state.db_pool)
    .await
    .ok()
    .flatten();

    if let Some(row) = in_orders_row {
        let db_in_orders: f64 = row.try_get("in_orders").unwrap_or(0.0);
        if db_in_orders > 0.0 {
            balance.in_orders = db_in_orders;
            balance.available = balance.wallet_balance - balance.in_orders;
        }
    }

    // Overall = wallet_balance (real deposit + realized PnL, WITHOUT leverage)
    balance.overall = balance.wallet_balance;

    Ok(Json(balance))
}

/// Internal function to get Binance balance
async fn get_binance_balance_internal() -> BalanceInfo {
    // Try to load credentials and fetch real balance
    match settings::ExchangeSettings::load() {
        Ok(exchange_settings) => {
            if exchange_settings.has_credentials() {
                let testnet = std::env::var("BINANCE_TESTNET")
                    .map(|v| v == "true" || v == "1")
                    .unwrap_or(false);
                match connections_lib::BinanceFuturesClient::new(
                    exchange_settings.api_key(),
                    exchange_settings.api_secret(),
                    testnet,
                ) {
                    Ok(client) => {
                        match client.account_info().await {
                            Ok(info) => {
                                let wallet_balance = info.total_wallet_balance
                                    .parse::<f64>().unwrap_or(0.0);
                                let unrealized_pnl = info.total_unrealized_profit
                                    .parse::<f64>().unwrap_or(0.0);
                                let available = info.available_balance
                                    .parse::<f64>().unwrap_or(0.0);

                                // Calculate in_orders = wallet - available
                                let in_orders = (wallet_balance - available).max(0.0);

                                return BalanceInfo {
                                    overall: wallet_balance,
                                    in_orders,
                                    available,
                                    wallet_balance,
                                    unrealized_pnl,
                                };
                            }
                            Err(e) => {
                                tracing::debug!("Binance account unavailable: {:?}", e);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::debug!("Failed to create Binance client: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            tracing::debug!("No exchange settings: {}", e);
        }
    }

    // Fallback — return zeros
    BalanceInfo {
        overall: 0.0,
        in_orders: 0.0,
        available: 0.0,
        wallet_balance: 0.0,
        unrealized_pnl: 0.0,
    }
}

// ─── GET /api/connections ──────────────────────────────────────────────
pub async fn get_connections(
    State(state): State<AppState>,
) -> Result<Json<ConnectionStatus>, StatusCode> {
    // Check all connections in parallel
    let db_ok = check_database(&state.db_pool).await;
    let (api_ok, ws_ok, account_ok) = check_binance().await;
    let redpanda_ok = check_redpanda().await;

    Ok(Json(ConnectionStatus {
        database: db_ok,
        redpanda: redpanda_ok,
        rest_api: api_ok,
        websocket: ws_ok,
        account: account_ok,
    }))
}

async fn check_database(pool: &sqlx::PgPool) -> bool {
    sqlx::query("SELECT 1")
        .fetch_one(pool)
        .await
        .is_ok()
}

async fn check_redpanda() -> bool {
    // Try TCP connect to Kafka brokers
    let brokers = std::env::var("KAFKA_BROKERS")
        .unwrap_or_else(|_| "127.0.0.1:19092".to_string());
    let addr = brokers.split(',').next().unwrap_or("127.0.0.1:19092");
    tokio::net::TcpStream::connect(addr)
        .await
        .is_ok()
}

async fn check_binance() -> (bool, bool, bool) {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    // REST API ping
    let rest_ok = client.get("https://fapi.binance.com/fapi/v1/ping")
        .send().await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    // WebSocket check (quick TCP/TLS handshake to fstream)
    let ws_ok = tokio::net::TcpStream::connect("fstream.binance.com:443")
        .await
        .is_ok();

    // Account check
    let account_ok = match settings::ExchangeSettings::load() {
        Ok(es) if es.has_credentials() => {
            let testnet = std::env::var("BINANCE_TESTNET")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false);
            match connections_lib::BinanceFuturesClient::new(es.api_key(), es.api_secret(), testnet) {
                Ok(c) => c.ping().await.is_ok(),
                Err(_) => false,
            }
        }
        _ => false,
    };

    (rest_ok, ws_ok, account_ok)
}

// ─── GET /api/alerts ──────────────────────────────────────────────────
pub async fn get_alerts(
    Query(query): Query<AlertsQuery>,
    State(state): State<AppState>,
) -> Result<Json<Vec<AlertRow>>, StatusCode> {
    let limit = query.limit.unwrap_or(50);

    // Try to read from risk.alerts table if it exists
    let rows = sqlx::query(
        "SELECT id, symbol, source, severity, message, price, change_pct, timestamp
         FROM risk.alerts
         ORDER BY timestamp DESC
         LIMIT $1"
    )
    .bind(limit)
    .fetch_all(&state.db_pool)
    .await;

    match rows {
        Ok(rows) => {
            let alerts: Vec<AlertRow> = rows.iter().map(|r| {
                let ts: chrono::DateTime<chrono::Utc> = r.get("timestamp");
                let now = chrono::Utc::now();
                let diff = now - ts;
                let time_ago = if diff.num_hours() > 0 {
                    format!("{}h ago", diff.num_hours())
                } else if diff.num_minutes() > 0 {
                    format!("{}m ago", diff.num_minutes())
                } else {
                    "just now".to_string()
                };

                AlertRow {
                    id: r.get("id"),
                    pair: r.get("symbol"),
                    alert_type: r.try_get::<String, _>("source").unwrap_or("Info".to_string()),
                    time_ago,
                    message: r.get("message"),
                    severity: r.try_get::<String, _>("severity").unwrap_or("info".to_string()),
                    source: r.try_get::<String, _>("source").unwrap_or("system".to_string()),
                    timestamp: ts.to_rfc3339(),
                }
            }).collect();
            Ok(Json(alerts))
        }
        Err(_) => {
            // Table doesn't exist yet — return empty
            Ok(Json(vec![]))
        }
    }
}

// ─── GET /api/trading-options ─────────────────────────────────────────
pub async fn get_trading_options(
) -> Result<Json<TradingOptionsPayload>, StatusCode> {
    // Read from unified config/order_manager.toml
    let path = "config/order_manager.toml";
    if let Ok(content) = std::fs::read_to_string(path) {
        if let Ok(val) = content.parse::<toml::Value>() {
            return Ok(Json(TradingOptionsPayload {
                leverage: val.get("leverage").and_then(|v| v.as_integer()).unwrap_or(10) as u16,
                max_orders_at_a_time: val.get("max_orders_at_a_time").and_then(|v| v.as_integer()).unwrap_or(10) as u16,
                trade_size_type: val.get("trade_size_type").and_then(|v| v.as_str()).unwrap_or("fixed_usdt").to_string(),
                trade_size_value: val.get("trade_size_value").and_then(|v| v.as_float()).unwrap_or(100.0),
                strategy_type: val.get("strategy_type").and_then(|v| v.as_str()).unwrap_or("ml_super_entry").to_string(),
                order_type: val.get("order_type").and_then(|v| v.as_str()).unwrap_or("futures_oco").to_string(),
                trading_mode: val.get("trading_mode").and_then(|v| v.as_str()).unwrap_or("off").to_string(),
            }));
        }
    }
    // Fallback defaults
    Ok(Json(TradingOptionsPayload {
        leverage: 10, max_orders_at_a_time: 10,
        trade_size_type: "fixed_usdt".to_string(), trade_size_value: 100.0,
        strategy_type: "ml_super_entry".to_string(), order_type: "futures_oco".to_string(),
        trading_mode: "off".to_string(),
    }))
}

// ─── POST /api/trading-options ────────────────────────────────────────
pub async fn save_trading_options(
    State(state): State<AppState>,
    Json(options): Json<TradingOptionsPayload>,
) -> Result<StatusCode, StatusCode> {
    // Update trading fields in order_manager.toml (preserve other fields)
    let path = "config/order_manager.toml";
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut val: toml::Value = content.parse().unwrap_or(toml::Value::Table(Default::default()));

    if let Some(table) = val.as_table_mut() {
        table.insert("leverage".to_string(), toml::Value::Integer(options.leverage as i64));
        table.insert("max_orders_at_a_time".to_string(), toml::Value::Integer(options.max_orders_at_a_time as i64));
        table.insert("trade_size_type".to_string(), toml::Value::String(options.trade_size_type));
        table.insert("trade_size_value".to_string(), toml::Value::Float(options.trade_size_value));
        table.insert("strategy_type".to_string(), toml::Value::String(options.strategy_type));
        table.insert("order_type".to_string(), toml::Value::String(options.order_type));
        table.insert("trading_mode".to_string(), toml::Value::String(options.trading_mode));
    }

    std::fs::write(path, toml::to_string_pretty(&val).unwrap_or_default())
        .map_err(|e| {
            tracing::error!("Failed to save trading options: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    state.broadcast(crate::state::WsMessage::OptionsUpdated);
    Ok(StatusCode::OK)
}

// ─── GET /api/order-options ───────────────────────────────────────────
pub async fn get_order_options(
) -> Result<Json<crate::state::OrderOptionsPayload>, StatusCode> {
    // Load order_manager.toml
    let om = load_order_manager_toml();
    // Load risk_manager.toml
    let rm = load_risk_manager_toml();

    Ok(Json(crate::state::OrderOptionsPayload {
        order_manager: crate::state::OrderManagerOptionsPayload {
            signal_score_min: om.0,
            signal_score_max: om.1,
            max_hold_bars: om.2,
            tf_1h_pct: om.3,
            tf_4h_pct: om.4,
            tf_15m_pct: om.5,
        },
        risk_manager: crate::state::RiskManagerOptionsPayload {
            btc_alert_threshold_pct: rm.0,
            alt_alert_threshold_pct: rm.1,
            volume_spike_threshold: rm.2,
        },
    }))
}

fn load_order_manager_toml() -> (f64, f64, i32, u16, u16, u16) {
    let path = "config/order_manager.toml";
    if let Ok(content) = std::fs::read_to_string(path) {
        if let Ok(val) = content.parse::<toml::Value>() {
            let score_min = val.get("signal_score_min").and_then(|v| v.as_float()).unwrap_or(0.70);
            let score_max = val.get("signal_score_max").and_then(|v| v.as_float()).unwrap_or(0.80);
            let max_hold = val.get("max_hold_bars").and_then(|v| v.as_integer()).unwrap_or(25) as i32;
            let tf_1h = val.get("tf_1h_pct").and_then(|v| v.as_integer()).unwrap_or(70) as u16;
            let tf_4h = val.get("tf_4h_pct").and_then(|v| v.as_integer()).unwrap_or(20) as u16;
            let tf_15m = val.get("tf_15m_pct").and_then(|v| v.as_integer()).unwrap_or(10) as u16;
            return (score_min, score_max, max_hold, tf_1h, tf_4h, tf_15m);
        }
    }
    (0.70, 0.80, 25, 70, 20, 10)
}

fn load_risk_manager_toml() -> (f64, f64, f64) {
    let path = "config/risk_manager.toml";
    if let Ok(content) = std::fs::read_to_string(path) {
        if let Ok(val) = content.parse::<toml::Value>() {
            let btc = val.get("btc_alert_threshold_pct").and_then(|v| v.as_float()).unwrap_or(0.5);
            let alt = val.get("alt_alert_threshold_pct").and_then(|v| v.as_float()).unwrap_or(1.5);
            let vol = val.get("volume_spike_threshold").and_then(|v| v.as_float()).unwrap_or(2.0);
            return (btc, alt, vol);
        }
    }
    (0.5, 1.5, 2.0)
}

// ─── POST /api/order-options ──────────────────────────────────────────
pub async fn save_order_options(
    State(state): State<AppState>,
    Json(options): Json<crate::state::OrderOptionsPayload>,
) -> Result<StatusCode, StatusCode> {
    // Update order_manager.toml — preserve existing fields, update subset
    save_order_manager_subset(&options.order_manager)
        .map_err(|e| {
            tracing::error!("Failed to save order manager opts: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Update risk_manager.toml — preserve existing fields, update subset
    save_risk_manager_subset(&options.risk_manager)
        .map_err(|e| {
            tracing::error!("Failed to save risk manager opts: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    state.broadcast(crate::state::WsMessage::OptionsUpdated);
    tracing::info!("Order options saved");
    Ok(StatusCode::OK)
}

fn save_order_manager_subset(opts: &crate::state::OrderManagerOptionsPayload) -> anyhow::Result<()> {
    let path = "config/order_manager.toml";
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut val: toml::Value = content.parse().unwrap_or(toml::Value::Table(Default::default()));

    if let Some(table) = val.as_table_mut() {
        table.insert("signal_score_min".to_string(), toml::Value::Float(opts.signal_score_min));
        table.insert("signal_score_max".to_string(), toml::Value::Float(opts.signal_score_max));
        table.insert("max_hold_bars".to_string(), toml::Value::Integer(opts.max_hold_bars as i64));
        table.insert("tf_1h_pct".to_string(), toml::Value::Integer(opts.tf_1h_pct as i64));
        table.insert("tf_4h_pct".to_string(), toml::Value::Integer(opts.tf_4h_pct as i64));
        table.insert("tf_15m_pct".to_string(), toml::Value::Integer(opts.tf_15m_pct as i64));
    }

    std::fs::write(path, toml::to_string_pretty(&val)?)?;
    Ok(())
}

fn save_risk_manager_subset(opts: &crate::state::RiskManagerOptionsPayload) -> anyhow::Result<()> {
    let path = "config/risk_manager.toml";
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut val: toml::Value = content.parse().unwrap_or(toml::Value::Table(Default::default()));

    if let Some(table) = val.as_table_mut() {
        table.insert("btc_alert_threshold_pct".to_string(), toml::Value::Float(opts.btc_alert_threshold_pct));
        table.insert("alt_alert_threshold_pct".to_string(), toml::Value::Float(opts.alt_alert_threshold_pct));
        table.insert("volume_spike_threshold".to_string(), toml::Value::Float(opts.volume_spike_threshold));
    }

    std::fs::write(path, toml::to_string_pretty(&val)?)?;
    Ok(())
}

// ─── POST /api/trade/order (manual with TP/SL) ───────────────────────
pub async fn place_order(
    State(state): State<AppState>,
    Json(order): Json<ManualOrderRequest>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Manual order: {:?}", order);

    // Broadcast order event
    state.broadcast(crate::state::WsMessage::OrderEvent(crate::state::OrderEvent {
        order_id: uuid::Uuid::new_v4().to_string(),
        pair: order.pair.clone(),
        side: order.side.clone(),
        order_type: order.order_type.clone(),
        status: "created".into(),
        price: order.entry_price,
        qty: order.amount_usdt,
        ts: chrono::Utc::now().timestamp_millis(),
    }));

    // TODO: Actually place order via BinanceFuturesClient
    // 1. Set leverage
    // 2. Place market/limit entry
    // 3. Place stop-loss order
    // 4. Place take-profit order
    // 5. Record in trade.positions

    Ok(StatusCode::OK)
}

// ─── POST /api/trade/close ────────────────────────────────────────────
pub async fn close_position(
    _state: State<AppState>,
    Json(request): Json<crate::state::CandlesLeftUpdateRequest>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Close position: {}", request.position_id);
    // TODO: implement via BinanceFuturesClient
    Ok(StatusCode::OK)
}

// ─── POST /api/trade/update-candles-left ──────────────────────────────
pub async fn update_candles_left(
    State(state): State<AppState>,
    Json(request): Json<CandlesLeftUpdateRequest>,
) -> Result<StatusCode, StatusCode> {
    sqlx::query(
        "UPDATE trade.positions SET candles_left = $1 WHERE id = $2"
    )
    .bind(request.candles_left)
    .bind(request.position_id)
    .execute(&state.db_pool)
    .await
    .map_err(|e| {
        tracing::error!("Update candles_left error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    tracing::info!("Candles left updated: position={} candles_left={}", request.position_id, request.candles_left);
    Ok(StatusCode::OK)
}

// ─── POST /api/control/emergency_stop ─────────────────────────────────
pub async fn emergency_stop(
    State(state): State<AppState>,
) -> Result<StatusCode, StatusCode> {
    tracing::warn!("🚨 EMERGENCY STOP triggered from WebUI!");

    // 1. Stop auto trading in memory
    {
        let mut at = state.auto_trading.write().await;
        at.is_running = false;
        at.trading_mode = "off".to_string();
    }

    // 2. Persist trading_mode = "off" to config file
    update_trading_mode_in_config("off");

    // 3. Close ALL open positions on Binance
    let close_results = emergency_close_all_positions(&state.db_pool).await;
    match &close_results {
        Ok(closed) => {
            tracing::warn!("🚨 Emergency closed {} positions", closed);
        }
        Err(e) => {
            tracing::error!("🚨 Emergency close failed: {}", e);
        }
    }

    // 4. Send Kafka close_all command to order_manager (best-effort)
    if let Err(e) = send_kafka_close_all().await {
        tracing::warn!("Failed to send Kafka close_all (order_manager may not receive): {}", e);
    }

    state.broadcast(crate::state::WsMessage::TradingStateUpdate(
        state.auto_trading.read().await.clone()
    ));

    Ok(StatusCode::OK)
}

/// Close all open positions via Binance Futures API.
/// Returns the number of positions closed.
async fn emergency_close_all_positions(pool: &sqlx::PgPool) -> Result<usize, StatusCode> {
    // Load Binance client
    let exchange_settings = settings::ExchangeSettings::load()
        .map_err(|e| {
            tracing::error!("Cannot load exchange settings for emergency close: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if !exchange_settings.has_credentials() {
        tracing::error!("No API credentials for emergency close");
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    let testnet = std::env::var("BINANCE_TESTNET")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    let client = connections_lib::BinanceFuturesClient::new(
        exchange_settings.api_key(),
        exchange_settings.api_secret(),
        testnet,
    ).map_err(|e| {
        tracing::error!("Failed to create Binance client for emergency close: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Query open positions from DB
    let rows = sqlx::query(
        "SELECT p.id, COALESCE(p.symbol, mp.symbol) as pair,
                CASE WHEN p.side = 1 THEN 'LONG' ELSE 'SHORT' END as side,
                p.qty
         FROM trade.positions p
         JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
         WHERE p.status = 1"
    )
    .fetch_all(pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to query open positions: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let mut closed = 0usize;

    for row in &rows {
        let pair: String = row.get("pair");
        let side: String = row.get("side");
        let qty: f64 = row.get("qty");
        let position_id: i64 = row.get("id");

        // Cancel all open orders first (SL/TP)
        if let Err(e) = client.cancel_all_orders(&pair).await {
            tracing::warn!("Failed to cancel orders for {} (may be none): {}", pair, e);
        }

        // If qty=0 in DB (entry wasn't confirmed), skip closing via DB qty.
        // The safety net below will find and close the Binance position.
        if qty <= 0.0 {
            tracing::warn!(
                "🚨 Position #{} {} {} has qty=0 in DB (entry price bug). Skipping DB close, will use Binance safety net.",
                position_id, pair, side
            );
            // Mark as closed in DB so it doesn't show up anymore
            let _ = sqlx::query(
                "UPDATE trade.positions SET status = 2, close_reason = 'emergency', closed_at = now() WHERE id = $1"
            )
            .bind(position_id)
            .execute(pool)
            .await;
            continue;
        }

        tracing::warn!("🚨 Emergency closing: {} {} qty={:.8} (position #{})", pair, side, qty, position_id);

        // Close position via market order
        match client.close_position(&pair, &side, qty).await {
            Ok(order) => {
                tracing::warn!(
                    "🚨 Position #{} {} {} closed → orderId={}, status={}",
                    position_id, pair, side, order.order_id, order.status
                );

                // Update DB status
                let _ = sqlx::query(
                    "UPDATE trade.positions SET status = 2, close_reason = 'emergency', closed_at = now() WHERE id = $1"
                )
                .bind(position_id)
                .execute(pool)
                .await;

                closed += 1;
            }
            Err(e) => {
                tracing::error!("🚨 Failed to close position #{} {} {}: {}", position_id, pair, side, e);
            }
        }
    }

    // Also close any positions on Binance that aren't in our DB (safety net)
    if let Ok(binance_positions) = client.open_positions().await {
        for bp in &binance_positions {
            let amt: f64 = bp.position_amt.parse().unwrap_or(0.0);
            if amt.abs() < 0.001 { continue; }

            let side = if amt > 0.0 { "LONG" } else { "SHORT" };
            tracing::warn!(
                "🚨 Found untracked Binance position: {} {} qty={:.8}, closing...",
                bp.symbol, side, amt.abs()
            );

            match client.close_position(&bp.symbol, side, amt.abs()).await {
                Ok(order) => {
                    tracing::warn!("🚨 Untracked position {} closed → orderId={}", bp.symbol, order.order_id);
                    closed += 1;
                }
                Err(e) => {
                    tracing::error!("🚨 Failed to close untracked position {}: {}", bp.symbol, e);
                }
            }
        }
    }

    Ok(closed)
}

/// Send close_all command to order_manager via Kafka (best-effort).
async fn send_kafka_close_all() -> anyhow::Result<()> {
    let brokers = std::env::var("KAFKA_BROKERS")
        .unwrap_or_else(|_| "127.0.0.1:19092".to_string());

    let producer: rdkafka::producer::FutureProducer = rdkafka::config::ClientConfig::new()
        .set("bootstrap.servers", &brokers)
        .set("message.timeout.ms", "5000")
        .create()?;

    let payload = serde_json::json!({
        "cmd_type": "close_all",
        "source": "emergency_stop",
        "timestamp": chrono::Utc::now().timestamp_millis()
    });

    let payload_str = serde_json::to_string(&payload)?;
    let record = rdkafka::producer::FutureRecord::to("orders.cmd")
        .key("emergency")
        .payload(&payload_str);

    producer.send(record, std::time::Duration::from_secs(3)).await
        .map_err(|(e, _)| anyhow::anyhow!("Kafka send error: {}", e))?;

    tracing::info!("📤 Sent close_all command to orders.cmd topic");
    Ok(())
}

// ─── POST /api/control/start_trading ──────────────────────────────────
pub async fn start_trading(
    State(state): State<AppState>,
) -> Result<Json<AutoTradingState>, StatusCode> {
    tracing::info!("START TRADING triggered from WebUI");

    let new_state = {
        let mut at = state.auto_trading.write().await;
        at.is_running = true;
        at.trading_mode = "auto".to_string();
        at.clone()
    };

    // Update order_manager.toml trading_mode to auto
    update_trading_mode_in_config("auto");

    state.broadcast(crate::state::WsMessage::TradingStateUpdate(new_state.clone()));
    Ok(Json(new_state))
}

// ─── POST /api/control/stop_trading ───────────────────────────────────
pub async fn stop_trading(
    State(state): State<AppState>,
) -> Result<Json<AutoTradingState>, StatusCode> {
    tracing::info!("STOP TRADING triggered from WebUI");

    let new_state = {
        let mut at = state.auto_trading.write().await;
        at.is_running = false;
        at.trading_mode = "off".to_string();
        at.clone()
    };

    // Update order_manager.toml trading_mode to off
    update_trading_mode_in_config("off");

    state.broadcast(crate::state::WsMessage::TradingStateUpdate(new_state.clone()));
    Ok(Json(new_state))
}

// ─── GET /api/control/trading_state ───────────────────────────────────
pub async fn get_trading_state(
    State(state): State<AppState>,
) -> Result<Json<AutoTradingState>, StatusCode> {
    let at = state.auto_trading.read().await.clone();
    Ok(Json(at))
}

/// Helper: update trading_mode in order_manager.toml
fn update_trading_mode_in_config(mode: &str) {
    let path = "config/order_manager.toml";
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut val: toml::Value = content.parse().unwrap_or(toml::Value::Table(Default::default()));
    if let Some(table) = val.as_table_mut() {
        table.insert("trading_mode".to_string(), toml::Value::String(mode.to_string()));
    }
    let _ = std::fs::write(path, toml::to_string_pretty(&val).unwrap_or_default());
}

// ─── GET /api/strategies ──────────────────────────────────────────────
pub async fn get_strategies(
    _state: State<AppState>,
) -> Result<Json<Vec<crate::state::StrategyInfo>>, StatusCode> {
    // Read current strategy from order_manager.toml
    let current_strategy = std::fs::read_to_string("config/order_manager.toml")
        .ok()
        .and_then(|c| c.parse::<toml::Value>().ok())
        .and_then(|v| v.get("strategy_type").and_then(|s| s.as_str()).map(|s| s.to_string()))
        .unwrap_or_else(|| "ml_super_entry".to_string());

    Ok(Json(vec![
        crate::state::StrategyInfo {
            id: "ml_super_entry".into(),
            name: "ML Super Entry".into(),
            enabled: current_strategy == "ml_super_entry",
            priority: 0,
            description: "ML-модель поиска super moves с P(super) > threshold".into(),
        },
        crate::state::StrategyInfo {
            id: "level_strategy".into(),
            name: "Level Strategy".into(),
            enabled: current_strategy == "level_strategy",
            priority: 1,
            description: "ML predictors + trade signals (в разработке)".into(),
        },
    ]))
}

// ─── POST /api/strategy/toggle ────────────────────────────────────────
pub async fn toggle_strategy(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<StatusCode, StatusCode> {
    let id = payload.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let enabled = payload.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
    tracing::info!("Toggle strategy: {} -> {}", id, enabled);
    state.broadcast(crate::state::WsMessage::StrategyUpdated);
    Ok(StatusCode::OK)
}
