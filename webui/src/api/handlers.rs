// webui/src/api/handlers.rs — Fixed to match actual DB schema
//
// Key corrections:
//   - Candles: separate tables per timeframe (market.candles_1m, candles_5m, etc.)
//   - Indicators: market.indicators_wide with columns ema_20, ema_50, ema_200 (underscores)
//   - Signals: trade.super_entry_signals (main strategy)
//   - Positions: direct columns sl_price, tp_price, candles_left, close_reason, exit_price
//   - History: trade.position_history for closed trades
//   - Alerts: stub (no risk.alerts table yet)

use axum::{extract::{Query, State}, Json, http::StatusCode};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use crate::state::{AppState, WebUiSettings, BalanceInfo, PnlOverview};

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

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderRequest {
    pub pair: String,
    pub side: String,
    #[serde(rename = "type")]
    pub order_type: String,
    pub price: Option<f64>,
    pub amount_usdt: f64,
    pub leverage: u16,
    pub reduce_only: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClosePositionRequest { pub position_id: i64 }

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
    let limit = query.limit.unwrap_or(100);
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
    // Get latest price and 24h stats from candles_1m
    let row = sqlx::query(
        "SELECT
            (SELECT close FROM market.candles_1m
             WHERE symbol = $1 ORDER BY time DESC LIMIT 1) as last_price,
            COALESCE(MAX(high), 0) as high_24h,
            COALESCE(MIN(low), 0) as low_24h,
            COALESCE(SUM(volume), 0) as volume_24h
         FROM market.candles_1m
         WHERE symbol = $1 AND time >= now() - INTERVAL '24 hours'"
    )
    .bind(&query.pair)
    .fetch_optional(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("Market summary error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    match row {
        Some(r) => {
            let price: f64 = r.try_get("last_price").unwrap_or(0.0);
            Ok(Json(MarketSummary {
                price,
                volume_24h: r.try_get("volume_24h").unwrap_or(0.0),
                change_24h: 0.0, // TODO: calculate from 24h ago price
                change_1h: 0.0,
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

    // Dynamic SQL — table name is from our controlled enum, not user input
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

    // Rows come DESC, reverse for chronological order
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
    // Map indicator type to actual column name in market.indicators_wide
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

// ─── GET /api/signals (trade.super_entry_signals) ─────────────────────
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
    State(state): State<AppState>,
) -> Result<Json<PnlOverview>, StatusCode> {
    // Use position_history for closed PnL stats
    let row = sqlx::query(
        "SELECT
            COALESCE(SUM(realized_pnl), 0) as closed_pnl,
            COUNT(*) FILTER (WHERE realized_pnl > 0) as wins,
            COUNT(*) as total_trades
         FROM trade.position_history
         WHERE closed_at > NOW() - INTERVAL '1 day'"
    )
    .fetch_one(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("PnL error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let closed_pnl: f64 = row.get("closed_pnl");
    let wins: i64 = row.get("wins");
    let total_trades: i64 = row.get("total_trades");
    let win_rate = if total_trades > 0 { (wins as f64 / total_trades as f64) * 100.0 } else { 0.0 };

    // Unrealized PnL from open positions
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

    Ok(Json(PnlOverview {
        closed_pnl,
        unrealized_pnl,
        win_rate,
        today_trades: total_trades,
        equity_points: vec![],
    }))
}

// ─── GET /api/balance ──────────────────────────────────────────────────
pub async fn get_balance(
    _state: State<AppState>,
) -> Result<Json<BalanceInfo>, StatusCode> {
    // TODO: connect to actual Binance balance via exchange settings
    Ok(Json(BalanceInfo { overall: 1000.0, in_orders: 0.0, available: 1000.0 }))
}

// ─── GET /api/alerts ──────────────────────────────────────────────────
pub async fn get_alerts(
    Query(query): Query<AlertsQuery>,
    State(_state): State<AppState>,
) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
    // TODO: risk.alerts table doesn't exist yet — return empty for now
    let _limit = query.limit.unwrap_or(50);
    Ok(Json(vec![]))
}

// ─── GET /api/options ─────────────────────────────────────────────────
pub async fn get_options(
    State(state): State<AppState>,
) -> Result<Json<WebUiSettings>, StatusCode> {
    Ok(Json(state.settings.read().await.clone()))
}

// ─── POST /api/options ────────────────────────────────────────────────
pub async fn save_options(
    State(state): State<AppState>,
    Json(new_settings): Json<WebUiSettings>,
) -> Result<StatusCode, StatusCode> {
    *state.settings.write().await = new_settings;
    state.broadcast(crate::state::WsMessage::OptionsUpdated);
    Ok(StatusCode::OK)
}

// ─── POST /api/trade/order ────────────────────────────────────────────
pub async fn place_order(
    State(state): State<AppState>,
    Json(order): Json<OrderRequest>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Order: {:?}", order);
    state.broadcast(crate::state::WsMessage::OrderEvent(crate::state::OrderEvent {
        order_id: uuid::Uuid::new_v4().to_string(),
        pair: order.pair,
        side: order.side.clone(),
        order_type: order.order_type.clone(),
        status: "created".into(),
        price: order.price,
        qty: order.amount_usdt,
        ts: chrono::Utc::now().timestamp_millis(),
    }));
    Ok(StatusCode::OK)
}

// ─── POST /api/trade/close ────────────────────────────────────────────
pub async fn close_position(
    _state: State<AppState>,
    _request: Json<ClosePositionRequest>,
) -> Result<StatusCode, StatusCode> {
    // TODO: implement via order_manager
    Ok(StatusCode::OK)
}

// ─── POST /api/control/* ──────────────────────────────────────────────
pub async fn emergency_stop(
    _state: State<AppState>,
) -> Result<StatusCode, StatusCode> {
    tracing::warn!("EMERGENCY STOP triggered from WebUI!");
    // TODO: send Kafka command to order_manager
    Ok(StatusCode::OK)
}

pub async fn reload_base(
    _state: State<AppState>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Reload base triggered from WebUI");
    Ok(StatusCode::OK)
}

pub async fn start_trading(
    _state: State<AppState>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Start trading triggered from WebUI");
    Ok(StatusCode::OK)
}

// ─── GET /api/strategies ──────────────────────────────────────────────
pub async fn get_strategies(
    _state: State<AppState>,
) -> Result<Json<Vec<crate::state::StrategyInfo>>, StatusCode> {
    Ok(Json(vec![
        crate::state::StrategyInfo {
            id: "super_entry".into(),
            name: "Super Entry (ML)".into(),
            enabled: true,
            priority: 0,
            description: "ML-модель поиска super moves с P(super) > threshold".into(),
        },
        crate::state::StrategyInfo {
            id: "level".into(),
            name: "Level Strategy".into(),
            enabled: false,
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
