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

        // Statistics
        .route("/statistics", get(get_statistics))

        // PnL and balance
        .route("/pnl/overview", get(get_pnl_overview))
        .route("/balance", get(get_balance))
        
        // Connections
        .route("/connections", get(get_connections))
        
        // Alerts
        .route("/alerts", get(get_alerts))
        
        // Trading Options (order_settings.toml)
        .route("/trading-options", get(get_trading_options))
        .route("/trading-options", post(save_trading_options))
        
        // Order Options (order_manager.toml + risk_manager.toml)
        .route("/order-options", get(get_order_options))
        .route("/order-options", post(save_order_options))
        
        // Strategies
        .route("/strategies", get(get_strategies))
        .route("/strategy/toggle", post(toggle_strategy))
        
        // Trading
        .route("/trade/order", post(place_order))
        .route("/trade/close", post(close_position))
        .route("/trade/update-candles-left", post(update_candles_left))
        
        // Control endpoints
        .route("/control/emergency_stop", post(emergency_stop))
        .route("/control/start_trading", post(start_trading))
        .route("/control/stop_trading", post(stop_trading))
        .route("/control/trading_state", get(get_trading_state))
}
