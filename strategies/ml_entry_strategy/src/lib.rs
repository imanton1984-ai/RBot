// strategies/ml_entry_strategy/src/lib.rs
//
// ML Entry Strategy — Super Entry Model (NoDir)
//
// A standalone strategy that uses two XGBoost models to identify
// high-probability entry points:
//   - P(super_long):  strong upward move probability
//   - P(super_short): strong downward move probability
//
// Direction is embedded in the labels — NO separate direction model.
// Conflict filter handles cases where both models fire simultaneously.
//
// This strategy is independent from the existing entry_policy and
// signal quality models. It can be enabled/disabled via config flags.
//
// MODULES:
//   - config: Configuration and TF-specific target thresholds
//   - dataset: Training data preparation (labeling, features)
//   - model: XGBoost model loading and inference (NoDir: super_long + super_short)
//   - scorer: Decision logic (per-TF thresholds, conflict filter)
//   - signal_generator: Trade signal generation in system format
//   - pipeline: Full orchestration pipeline
//   - strategy: High-level strategy interface
//   - direction: DEPRECATED — kept for backward compat but not used in production

pub mod config;
pub mod dataset;
pub mod model;
pub mod scorer;
pub mod signal_generator;
pub mod pipeline;
pub mod strategy;
pub mod db_writer;
pub mod heuristic;
pub mod direction;

// Re-exports for convenience
pub use config::SuperEntryConfig;
pub use model::SuperEntryModelManager;
pub use scorer::{SuperEntryScorer, SuperEntryDecision};
pub use signal_generator::{SignalGenerator, SuperEntrySignal};
pub use pipeline::SuperEntryPipeline;
pub use strategy::SuperEntryStrategy;
pub use heuristic::{
    CrossTfStore, MultiTfStore, HeuristicMode, HeuristicResult,
    compute_heuristic_direction, compute_heuristic_direction_backtest,
    apply_heuristic_filter, get_higher_tf, get_lower_tf,
    heuristic_min_confidence_from_env,
};
