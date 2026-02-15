//! **orderflow** — real-time order-flow metrics for the level
//! bounce/breakout strategy.
//!
//! Connects to Binance Futures WebSocket streams (`@depth20@100ms` +
//! `@aggTrade`), computes per-symbol order-book imbalance, volume
//! delta, large-trade detection and a combined pressure score.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use orderflow_lib::{OrderFlowManager, fetch_symbols_from_db};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let symbols = fetch_symbols_from_db().await?;
//!     let mgr = OrderFlowManager::start(symbols).await?;
//!     // … later, in the compute pipeline:
//!     if let Some(snap) = mgr.get_snapshot("BTCUSDT").await {
//!         println!("pressure = {}", snap.pressure_score);
//!     }
//!     Ok(())
//! }
//! ```

pub mod types;
pub mod book_tracker;
pub mod trade_tracker;
pub mod flow_manager;

// ── Re-exports for ergonomic use ────────────────────────────────────
pub use flow_manager::{fetch_symbols_from_db, OrderFlowManager};
pub use types::OrderFlowSnapshot;
