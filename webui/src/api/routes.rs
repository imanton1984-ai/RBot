// webui/src/api/routes.rs
//
// API route definitions

use axum::{
    routing::{get, post},
    Router,
};
use crate::state::AppState;
use super::handlers::*;

/// Create API router with all endpoints
pub fn create_api_router() -> Router<AppState> {
    Router::new()
        // Pairs and market data
        .route("/pairs", get(get_pairs))
        .route("/market/summary", get(get_market_summary))
        
        // Candles and indicators
        .route("/candles", get(get_candles))
        .route("/indicators", get(get_indicators))
        
        // Signals
        .route("/signals", get(get_signals))
        
        // Positions
        .route("/positions/open", get(get_open_positions))
        .route("/positions/history", get(get_positions_history))
        
        // PnL and balance
        .route("/pnl/overview", get(get_pnl_overview))
        .route("/balance", get(get_balance))
        
        // Alerts
        .route("/alerts", get(get_alerts))
        
        // Options/Settings
        .route("/options", get(get_options))
        .route("/options", post(save_options))
        
        // Strategies
        .route("/strategies", get(get_strategies))
        .route("/strategy/toggle", post(toggle_strategy))
        
        // Trading
        .route("/trade/order", post(place_order))
        .route("/trade/close", post(close_position))
        
        // Control endpoints
        .route("/control/emergency_stop", post(emergency_stop))
        .route("/control/reload_base", post(reload_base))
        .route("/control/start_trading", post(start_trading))
}
