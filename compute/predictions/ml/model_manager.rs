// compute/predictors/ml/model_manager.rs

// Central manager for loading and managing ONNX models for ML predictors.
// Supports XGBoost for future price and LightGBM for level predictor.

use anyhow::Result;
use ort::{Session, SessionBuilder, GraphOptimizationLevel, ExecutionProvider}; // TODO: Add 'ort' crate to compute/Cargo.toml

pub struct ModelManager {
    future_price_model: Session,
    level_predictor_model: Session,
}

impl ModelManager {
    /// Loads ONNX models for future price (XGBoost) and level predictor (LightGBM).
    /// Configures for CUDA if available.
    pub fn new(future_path: &str, level_path: &str, use_cuda: bool) -> Result<Self> {
        let mut builder = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::All)?;

        if use_cuda {
            builder = builder.use_cuda(0)?; // Use first GPU
        }

        let future_price_model = builder.commit_from_file(future_path)?;
        let level_predictor_model = builder.commit_from_file(level_path)?;

        Ok(Self {
            future_price_model,
            level_predictor_model,
        })
    }

    pub fn get_future_price_model(&self) -> &Session {
        &self.future_price_model
    }

    pub fn get_level_predictor_model(&self) -> &Session {
        &self.level_predictor_model
    }
}

// Note: Models are exported from Python trainer (XGBoost and LightGBM to ONNX).
// Integrate with compute_backend for CUDA detection.
// // TODO: Refactor for performance according to Manifesto v1.0
