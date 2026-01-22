//! Core computation engine for the trading bot
//!
//! This crate handles the calculation of technical indicators, generation of
//! raw signals, and final scoring for trade decisions.

pub mod state;
pub mod indicators;
pub mod raw_signals;
pub mod planner;
pub mod ml;
pub mod scorer;
pub mod health;

pub use state::*;
pub use indicators::*;
pub use raw_signals::*;
pub use planner::*;
pub use ml::*;
pub use scorer::*;
pub use health::*;