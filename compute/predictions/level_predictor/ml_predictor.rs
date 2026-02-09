// compute/predictors/level_predictor/ml_predictor.rs

// Machine Learning predictor for bounce/break of support/resistance levels using LightGBM via ONNX.
// Calculations performed on CUDA if available, fallback to CPU.

use anyhow::Result;
use ort::{Session, GraphOptimizationLevel}; // TODO: Add 'ort' crate to compute/Cargo.toml
use crate::compute::compute_backend::ComputeBackend;
use crate::compute::types::Features;

pub struct LevelPredictorMl {
    model: Session,
    backend: ComputeBackend,
}

impl LevelPredictorMl {
    /// Creates a new ML predictor, loading the ONNX model (exported from LightGBM).
    pub fn new(model_path: &str, backend: ComputeBackend) -> Result<Self> {
        let model = ort::Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .commit_from_file(model_path)?;

        Ok(Self { model, backend })
    }

    /// Predicts bounce/break probabilities for a given level using features from indicators_wide and raw_signals.
    /// Returns (level, prob_bounce, prob_break, score) if score >= 0.80.
    pub fn predict(&self, symbol: &str, timeframe: &str, features: &Features) -> Result<Option<(f64, f64, f64, f64)>> {
        // TODO: Fetch and prepare features, identify potential level from sr_levels

        // Run inference using backend

        // Extract level, prob_bounce, prob_break, score

        // If score < 0.80, return None

        todo!("Implement prediction logic with CUDA/CPU fallback")
    }
}

// // TODO: Refactor for performance according to Manifesto v1.0
