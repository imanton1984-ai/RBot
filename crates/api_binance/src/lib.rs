//! Binance API integration for the trading bot
//!
//! This crate provides interfaces for interacting with Binance Futures API,
//! including both REST and WebSocket connections for market data and order management.

pub mod rest;
pub mod ws_market;
pub mod ws_user;
pub mod sign;
pub mod rate_limit;

pub use rest::*;
pub use ws_market::*;
pub use ws_user::*;
pub use sign::*;
pub use rate_limit::*;