//! Database writer service for the trading bot
//!
//! This crate handles batch writing of market data, indicators, signals,
//! orders, and positions to TimescaleDB.

pub mod batcher;
pub mod copy;
pub mod migrations;
pub mod idempotency;
pub mod retention;
pub mod health;

// pub use batcher::*;
// pub use copy::*;
// pub use migrations::*;
// pub use idempotency::*;
// pub use retention::*;
// pub use health::*;