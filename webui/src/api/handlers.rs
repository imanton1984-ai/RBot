// webui/src/api/handlers.rs - Simplified version with proper types

use axum::{extract::{Query, State}, Json, http::StatusCode};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use crate::state::{AppState, WebUiSettings, BalanceInfo, PnlOverview};

#[derive(Debug, Deserialize)] pub struct PairsQuery { pub search: Option<String>, pub limit: Option<i64> }
#[derive(Debug, Deserialize)] pub struct CandlesQuery { pub pair: String, pub tf: i32, pub limit: Option<i64> }
#[derive(Debug, Deserialize)] pub struct IndicatorsQuery { pub pair: String, pub tf: i32, #[serde(rename = "type")] pub indicator_type: String, pub limit: Option<i64> }
#[derive(Debug, Deserialize)] pub struct SignalsQuery { pub pair: Option<String>, pub tf: Option<i32>, pub limit: Option<i64> }
#[derive(Debug, Serialize, sqlx::FromRow)] pub struct PairInfo { pub symbol: String, pub symbol_id: i64 }
#[derive(Debug, Serialize, sqlx::FromRow)] pub struct CandleRow { pub t: i64, pub o: f64, pub h: f64, pub l: f64, pub c: f64, pub v: f64 }
#[derive(Debug, Serialize, sqlx::FromRow)] pub struct IndicatorRow { pub t: i64, pub value: Option<f32> }
#[derive(Debug, Serialize, Deserialize)] pub struct OrderRequest { pub pair: String, pub side: String, #[serde(rename = "type")] pub order_type: String, pub price: Option<f64>, pub amount_usdt: f64, pub leverage: u16, pub reduce_only: bool }
#[derive(Debug, Serialize, Deserialize)] pub struct ClosePositionRequest { pub position_id: i64 }
#[derive(Debug, Serialize)] pub struct MarketSummary { pub price: f64, pub volume_24h: f64, pub change_24h: f64, pub change_1h: f64, pub high_24h: f64, pub low_24h: f64 }

#[derive(Debug, Serialize)] pub struct CandleResponse { pub t: i64, pub o: f64, pub h: f64, pub l: f64, pub c: f64, pub v: f64 }
#[derive(Debug, Serialize)] pub struct IndicatorResponse { pub t: i64, pub value: Option<f64> }

pub async fn get_pairs(Query(query): Query<PairsQuery>, State(state): State<AppState>) -> Result<Json<Vec<PairInfo>>, StatusCode> {
    let limit = query.limit.unwrap_or(100);
    let search = query.search.unwrap_or_default();
    let pairs = sqlx::query_as::<_, PairInfo>("SELECT symbol, symbol_id FROM market.pairs WHERE is_active = true AND ($1 = '' OR symbol ILIKE $1) ORDER BY symbol LIMIT $2").bind(format!("%{}%", search)).bind(limit).fetch_all(&state.db_pool).await.map_err(|e| { tracing::error!("Pairs error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;
    Ok(Json(pairs))
}

pub async fn get_market_summary(_query: Query<CandlesQuery>, _state: State<AppState>) -> Result<Json<MarketSummary>, StatusCode> {
    Ok(Json(MarketSummary { price: 0.0, volume_24h: 0.0, change_24h: 0.0, change_1h: 0.0, high_24h: 0.0, low_24h: 0.0 }))
}

pub async fn get_candles(Query(query): Query<CandlesQuery>, State(state): State<AppState>) -> Result<Json<Vec<CandleResponse>>, StatusCode> {
    let limit = query.limit.unwrap_or(1000);
    let candles = sqlx::query_as::<_, CandleRow>("SELECT time_ms as t, open as o, high as h, low as l, close as c, volume as v FROM market.candles WHERE symbol = $1 AND tf_minutes = $2 ORDER BY time_ms DESC LIMIT $3").bind(&query.pair).bind(query.tf).bind(limit).fetch_all(&state.db_pool).await.map_err(|e| { tracing::error!("Candles error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;
    Ok(Json(candles.into_iter().rev().map(|c| CandleResponse { t: c.t, o: c.o, h: c.h, l: c.l, c: c.c, v: c.v }).collect()))
}

pub async fn get_indicators(Query(query): Query<IndicatorsQuery>, State(state): State<AppState>) -> Result<Json<Vec<IndicatorResponse>>, StatusCode> {
    let column = match query.indicator_type.as_str() { "ema" => "ema20", "ema50" => "ema50", "ema200" => "ema200", "rsi" => "rsi", _ => "rsi" };
    let indicators = sqlx::query_as::<_, IndicatorRow>(&format!("SELECT time_ms as t, {} as value FROM market.indicators_wide WHERE symbol = $1 AND tf_minutes = $2 ORDER BY time_ms DESC LIMIT $3", column)).bind(&query.pair).bind(query.tf).bind(query.limit.unwrap_or(1000)).fetch_all(&state.db_pool).await.map_err(|e| { tracing::error!("Indicators error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;
    Ok(Json(indicators.into_iter().rev().map(|i| IndicatorResponse { t: i.t, value: i.value.map(|v| v as f64) }).collect()))
}

pub async fn get_signals(Query(query): Query<SignalsQuery>, State(state): State<AppState>) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
    let limit = query.limit.unwrap_or(100);
    let rows = sqlx::query("SELECT fs.id, fs.time, fs.symbol as pair, fs.tf_minutes as tf, fs.side, fs.final_score as score, fs.entry_price, fs.sl_price, fs.tp_price, fs.strategy, 'active' as status FROM trade.final_signals fs WHERE ($1::text IS NULL OR fs.symbol = $1) AND ($2::int IS NULL OR fs.tf_minutes = $2) ORDER BY fs.time DESC LIMIT $3").bind(query.pair).bind(query.tf).bind(limit).fetch_all(&state.db_pool).await.map_err(|e| { tracing::error!("Signals error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;
    Ok(Json(rows.into_iter().filter_map(|r| sqlx::Row::try_get::<serde_json::Value, _>(&r, 0).ok()).collect()))
}

pub async fn get_open_positions(State(state): State<AppState>) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
    let rows = sqlx::query("SELECT p.id, p.symbol as pair, CASE WHEN p.side = 1 THEN 'LONG' ELSE 'SHORT' END as side, p.quantity as qty, p.entry_price, p.entry_price as current_price, p.stop_loss, p.take_profit, 0.0 as pnl_usdt, 0.0 as pnl_pct, 'open' as status, p.open_time FROM trade.positions p WHERE p.close_time IS NULL ORDER BY p.open_time DESC").fetch_all(&state.db_pool).await.map_err(|e| { tracing::error!("Positions error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;
    Ok(Json(rows.into_iter().filter_map(|r| sqlx::Row::try_get::<serde_json::Value, _>(&r, 0).ok()).collect()))
}

pub async fn get_positions_history(State(state): State<AppState>) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
    let rows = sqlx::query("SELECT p.id, p.symbol as pair, p.quantity as qty, p.entry_price, p.close_price, p.close_type, p.pnl_usdt, p.pnl_pct, p.open_time, p.close_time FROM trade.positions p WHERE p.close_time IS NOT NULL ORDER BY p.close_time DESC LIMIT 500").fetch_all(&state.db_pool).await.map_err(|e| { tracing::error!("History error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;
    Ok(Json(rows.into_iter().filter_map(|r| sqlx::Row::try_get::<serde_json::Value, _>(&r, 0).ok()).collect()))
}

pub async fn get_pnl_overview(State(state): State<AppState>) -> Result<Json<PnlOverview>, StatusCode> {
    let row = sqlx::query("SELECT COALESCE(SUM(pnl_usdt), 0) as closed_pnl, COUNT(*) FILTER (WHERE pnl_usdt > 0) as wins, COUNT(*) as total_trades FROM trade.positions WHERE close_time IS NOT NULL AND close_time > NOW() - INTERVAL '1 day'").fetch_one(&state.db_pool).await.map_err(|e| { tracing::error!("PnL error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;
    let closed_pnl: f64 = row.get("closed_pnl");
    let wins: i64 = row.get("wins");
    let total_trades: i64 = row.get("total_trades");
    let win_rate = if total_trades > 0 { (wins as f64 / total_trades as f64) * 100.0 } else { 0.0 };
    Ok(Json(PnlOverview { closed_pnl, unrealized_pnl: 0.0, win_rate, today_trades: total_trades, equity_points: vec![] }))
}

pub async fn get_balance(_state: State<AppState>) -> Result<Json<BalanceInfo>, StatusCode> { Ok(Json(BalanceInfo { overall: 1000.0, in_orders: 0.0, available: 1000.0 })) }

pub async fn get_alerts(Query(limit): Query<Option<i64>>, State(state): State<AppState>) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
    let limit = limit.unwrap_or(50);
    let rows = sqlx::query("SELECT id, symbol as pair, 'Info' as alert_type, 'now' as time_ago, message, 'info' as severity FROM risk.alerts ORDER BY timestamp DESC LIMIT $1").bind(limit).fetch_all(&state.db_pool).await.map_err(|e| { tracing::error!("Alerts error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;
    Ok(Json(rows.into_iter().filter_map(|r| sqlx::Row::try_get::<serde_json::Value, _>(&r, 0).ok()).collect()))
}

pub async fn get_options(State(state): State<AppState>) -> Result<Json<WebUiSettings>, StatusCode> { Ok(Json(state.settings.read().await.clone())) }

pub async fn save_options(State(state): State<AppState>, Json(new_settings): Json<WebUiSettings>) -> Result<StatusCode, StatusCode> {
    *state.settings.write().await = new_settings;
    state.broadcast(crate::state::WsMessage::OptionsUpdated);
    Ok(StatusCode::OK)
}

pub async fn place_order(State(state): State<AppState>, Json(order): Json<OrderRequest>) -> Result<StatusCode, StatusCode> {
    tracing::info!("Order: {:?}", order);
    state.broadcast(crate::state::WsMessage::OrderEvent(crate::state::OrderEvent { order_id: uuid::Uuid::new_v4().to_string(), pair: order.pair, side: order.side.clone(), order_type: order.order_type.clone(), status: "created".into(), price: order.price, qty: order.amount_usdt, ts: chrono::Utc::now().timestamp_millis() }));
    Ok(StatusCode::OK)
}

pub async fn close_position(_state: State<AppState>, _request: Json<ClosePositionRequest>) -> Result<StatusCode, StatusCode> { Ok(StatusCode::OK) }
pub async fn emergency_stop(_state: State<AppState>) -> Result<StatusCode, StatusCode> { tracing::warn!("EMERGENCY STOP!"); Ok(StatusCode::OK) }
pub async fn reload_base(_state: State<AppState>) -> Result<StatusCode, StatusCode> { tracing::info!("Reload base"); Ok(StatusCode::OK) }
pub async fn start_trading(_state: State<AppState>) -> Result<StatusCode, StatusCode> { tracing::info!("Start trading"); Ok(StatusCode::OK) }

pub async fn get_strategies(_state: State<AppState>) -> Result<Json<Vec<crate::state::StrategyInfo>>, StatusCode> {
    Ok(Json(vec![
        crate::state::StrategyInfo { id: "level".into(), name: "Level Strategy".into(), enabled: true, priority: 0, description: "ML predictors + trade signals".into() },
        crate::state::StrategyInfo { id: "super_entry".into(), name: "Super Entry".into(), enabled: false, priority: 1, description: "ML-модель поиска точек с сильным движением".into() },
    ]))
}

pub async fn toggle_strategy(State(state): State<AppState>, Json(payload): Json<serde_json::Value>) -> Result<StatusCode, StatusCode> {
    let id = payload.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let enabled = payload.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
    tracing::info!("Toggle: {} -> {}", id, enabled);
    state.broadcast(crate::state::WsMessage::StrategyUpdated);
    Ok(StatusCode::OK)
}
