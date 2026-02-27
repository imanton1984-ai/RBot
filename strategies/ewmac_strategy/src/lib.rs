// strategies/ewmac_strategy/src/lib.rs
//
// EWMAC Strategy — Exponentially Weighted Moving Average Crossover
//
// A trend-following strategy based on Robert Carver's "Systematic Trading".
// Uses multiple EMA crossover pairs (fast/slow) normalized by ATR
// to generate directional forecasts.
//
// This strategy is purely rule-based (no ML models required).
// It can be run independently from the existing super_entry strategy.
//
// MODULES:
//   - config: Configuration (pairs, scalars, thresholds)
//   - ewmac: Core EWMAC calculation (EMA, ATR, crossover, forecast)
//   - dataset: Data fetching from TimescaleDB
//   - signal_generator: Trade signal generation (with ATR-based SL/TP)
//   - db_writer: Batch writing signals to trade.ewmac_signals
//   - pipeline: Full orchestration pipeline
//   - strategy: High-level strategy interface

pub mod config;
pub mod ewmac;
pub mod dataset;
pub mod signal_generator;
pub mod db_writer;
pub mod pipeline;
pub mod strategy;

// Re-exports for convenience
pub use config::EwmacConfig;
pub use ewmac::{EwmacCalculator, EwmacResult};
pub use signal_generator::{SignalGenerator, EwmacSignal};
pub use pipeline::EwmacPipeline;
pub use strategy::EwmacStrategy;
