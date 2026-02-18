#![recursion_limit = "256"]
// lib.rs for the predictors crate
// Re-export all modules from the predictors directory

pub mod future_predictor;  // Contains ml_predictor and heuristic_predictor
pub mod level_predictor;
pub mod ml;
pub mod config;
pub mod types;
pub mod feature_view;
pub mod level_view;
pub mod scoring;
pub mod persistence;  // Fixed typo from persisence
pub mod feature_schema;
pub mod pipeline;
pub mod consensus;
pub mod signal_quality;  // Signal Quality meta-scorer (heuristic + ML)
pub mod entry_policy;    // Entry Agent for online entry timing (ENTER/WAIT/CANCEL)