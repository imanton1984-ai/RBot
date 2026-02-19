// strategies/level_strategy/src/lib.rs
//
// Level Strategy — Main Trading Strategy (formerly "default" strategy)
//
// This strategy uses:
//   - ML predictors (price targets, level bounce/breakout)
//   - Signal quality scoring
//   - Entry policy for timing
//
// The pipeline:
//   indicators_wide → predictors → trade_signals

pub mod config;
pub mod pipeline;
pub mod signal_generator;
pub mod strategy;

// Re-exports for convenience
pub use config::LevelStrategyConfig;
pub use pipeline::LevelStrategyPipeline;
pub use strategy::LevelStrategy;
