// strategies/ml_entry_strategy/src/lib.rs
//
// ML Entry Strategy — Super Entry Model
//
// A standalone strategy that uses XGBoost models to identify
// high-probability entry points ("super entries") that are likely
// to produce significant moves in the target direction.
//
// This strategy is independent from the existing entry_policy and
// signal quality models. It can be enabled/disabled via config flags.
//
// MODULES:
//   - config: Configuration and TF-specific target thresholds
//   - dataset: Training data preparation (labeling, features)
//   - model: XGBoost model loading and inference
//   - scorer: Decision logic (P(super), direction, overheated filter)
//   - signal_generator: Trade signal generation in system format
//   - pipeline: Full orchestration pipeline
//   - strategy: High-level strategy interface

pub mod config;
pub mod dataset;
pub mod model;
pub mod scorer;
pub mod signal_generator;
pub mod pipeline;
pub mod strategy;

// Re-exports for convenience
pub use config::SuperEntryConfig;
pub use model::SuperEntryModelManager;
pub use scorer::{SuperEntryScorer, SuperEntryDecision};
pub use signal_generator::{SignalGenerator, SuperEntrySignal};
pub use pipeline::SuperEntryPipeline;
pub use strategy::SuperEntryStrategy;
