// webui/src/state.rs
//
// Shared application state for WebUI backend

use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use sqlx::PgPool;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub db_pool: PgPool,
    pub ws_sender: broadcast::Sender<WsMessage>,
    pub settings: Arc<RwLock<WebUiSettings>>,
    pub auto_trading: Arc<RwLock<AutoTradingState>>,
}

impl AppState {
    pub fn new(db_pool: PgPool, settings: WebUiSettings) -> Self {
        let (ws_sender, _) = broadcast::channel(1000);
        Self {
            db_pool,
            ws_sender,
            settings: Arc::new(RwLock::new(settings)),
            auto_trading: Arc::new(RwLock::new(AutoTradingState::default())),
        }
    }

    pub fn broadcast(&self, msg: WsMessage) {
        let _ = self.ws_sender.send(msg);
    }
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    #[serde(rename = "ticker_update")]
    TickerUpdate(TickerUpdate),
    #[serde(rename = "candle_update")]
    CandleUpdate(CandleUpdate),
    #[serde(rename = "positions_update")]
    PositionsUpdate(Vec<PositionInfo>),
    #[serde(rename = "order_event")]
    OrderEvent(OrderEvent),
    #[serde(rename = "balance_update")]
    BalanceUpdate(BalanceInfo),
    #[serde(rename = "alert_event")]
    AlertEvent(RiskAlertInfo),
    #[serde(rename = "pnl_update")]
    PnlUpdate(PnlOverview),
    #[serde(rename = "options_updated")]
    OptionsUpdated,
    #[serde(rename = "strategy_updated")]
    StrategyUpdated,
    #[serde(rename = "connections_update")]
    ConnectionsUpdate(ConnectionStatus),
    #[serde(rename = "trading_state_update")]
    TradingStateUpdate(AutoTradingState),
}

/// Ticker update for header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerUpdate {
    pub pair: String,
    pub price: f64,
    pub change_24h: f64,
    pub change_1h: f64,
    pub volume_24h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub ts: i64,
}

/// Candle update for chart
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleUpdate {
    pub pair: String,
    pub tf: i32,
    pub t: i64,
    pub o: f64,
    pub h: f64,
    pub l: f64,
    pub c: f64,
    pub v: f64,
    pub is_closed: bool,
}

/// Position info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionInfo {
    pub id: i64,
    pub pair: String,
    pub side: String, // "LONG" or "SHORT"
    pub qty: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub pnl_usdt: f64,
    pub pnl_pct: f64,
    pub status: String,
    pub open_time: DateTime<Utc>,
}

/// Order event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderEvent {
    pub order_id: String,
    pub pair: String,
    pub side: String,
    pub order_type: String,
    pub status: String, // created/filled/canceled
    pub price: Option<f64>,
    pub qty: f64,
    pub ts: i64,
}

/// Balance info — with proper breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceInfo {
    pub overall: f64,
    pub in_orders: f64,
    pub available: f64,
    pub wallet_balance: f64,
    pub unrealized_pnl: f64,
}

/// Risk alert info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAlertInfo {
    pub id: i64,
    pub pair: String,
    pub alert_type: String, // "Long", "Short", "Info"
    pub time_ago: String,
    pub message: String,
    pub severity: String,
    pub source: String,
    pub timestamp: String,
}

/// PnL overview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnlOverview {
    pub closed_pnl: f64,
    pub unrealized_pnl: f64,
    pub win_rate: f64,
    pub today_trades: i64,
    pub equity_points: Vec<PnlPoint>,
    pub overall_balance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnlPoint {
    pub t: i64,
    pub value: f64,
}

/// WebUI settings (legacy compat)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebUiSettings {
    pub leverage_default: u16,
    pub max_open_orders: u16,
    pub max_risk_pct: f64,
    pub order_timeout_bars: i32,
    pub ws_update_rate_ms: u64,
}

impl Default for WebUiSettings {
    fn default() -> Self {
        Self {
            leverage_default: 10,
            max_open_orders: 10,
            max_risk_pct: 2.0,
            order_timeout_bars: 10,
            ws_update_rate_ms: 100,
        }
    }
}

/// Connection status for all services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStatus {
    pub database: bool,
    pub redpanda: bool,
    pub rest_api: bool,
    pub websocket: bool,
    pub account: bool,
}

/// Trading Options (maps to config/order_settings.toml)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingOptionsPayload {
    pub leverage: u16,
    pub max_orders_at_a_time: u16,
    pub trade_size_type: String,
    pub trade_size_value: f64,
    pub strategy_type: String,
    pub order_type: String,
    pub trading_mode: String,
}

/// Order Manager Options (maps to subset of config/order_manager.toml)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderManagerOptionsPayload {
    pub signal_score_min: f64,
    pub signal_score_max: f64,
    pub max_hold_bars: i32,
    pub tf_1h_pct: u16,
    pub tf_4h_pct: u16,
    pub tf_15m_pct: u16,
}

/// Risk Manager Options (subset of config/risk_manager.toml)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskManagerOptionsPayload {
    pub btc_alert_threshold_pct: f64,
    pub alt_alert_threshold_pct: f64,
    pub volume_spike_threshold: f64,
}

/// Combined Order Options payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderOptionsPayload {
    pub order_manager: OrderManagerOptionsPayload,
    pub risk_manager: RiskManagerOptionsPayload,
}

/// Auto trading state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTradingState {
    pub is_running: bool,
    pub trading_mode: String,
}

impl Default for AutoTradingState {
    fn default() -> Self {
        Self {
            is_running: false,
            trading_mode: "off".to_string(),
        }
    }
}

/// Strategy summary for UI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyInfo {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub priority: u8,
    pub description: String,
}

/// Signal info for UI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalInfo {
    pub id: i64,
    pub time: DateTime<Utc>,
    pub pair: String,
    pub tf: i32,
    pub side: i16, // 1 = LONG, -1 = SHORT
    pub score: f32,
    pub entry_price: f64,
    pub sl_price: f64,
    pub tp_price: f64,
    pub strategy: String,
    pub status: String, // active/closed/timeout
}

/// Manual order request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualOrderRequest {
    pub pair: String,
    pub side: String,
    #[serde(rename = "type")]
    pub order_type: String,
    pub price: Option<f64>,
    pub amount_usdt: f64,
    pub leverage: u16,
    pub take_profit: f64,
    pub stop_loss: f64,
    pub entry_price: Option<f64>,
    pub reduce_only: bool,
}

/// Candles left update request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandlesLeftUpdateRequest {
    pub position_id: i64,
    pub candles_left: i16,
}

/// Close position request (from WebUI Action button)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosePositionRequest {
    pub position_id: i64,
}
