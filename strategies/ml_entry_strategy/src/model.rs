// strategies/ml_entry_strategy/src/model.rs
//
// Model wrapper for Super Entry XGBoost models.
//
// Uses direct Booster::load() instead of ModelManager to avoid
// loading/parsing large XGBoost JSON model dumps as schema files,
// which causes memory pressure and abort() in release builds.
//
// Models:
//   - super_entry_v1_tf{X}.ubj — binary classifier: P(super move)
//   - super_dir_v1_tf{X}.ubj — binary classifier: P(direction=LONG)

use anyhow::Result;
use tracing::{info, warn};
use std::collections::HashMap;

use predictors::ml::xgb_runtime::{Booster, Device, ModelKind};
use crate::config::SuperEntryConfig;

/// Prediction output from the super entry model
#[derive(Debug, Clone, Copy)]
pub struct SuperEntryPrediction {
    /// Probability that this is a "super" move (P >= target move %)
    pub p_super: f32,
    /// Probability that the direction is LONG (>0.5 = LONG, <0.5 = SHORT)
    pub p_long: f32,
    /// Derived: direction (1 = LONG, -1 = SHORT)
    pub direction: i8,
    /// Derived: expected magnitude estimate (optional, from p_super * target)
    pub estimated_magnitude_pct: f64,
}

/// Loaded model pair for one TF
struct TfModels {
    super_model: Booster,
    dir_model: Option<Booster>,
}

/// Super Entry Model Manager
///
/// Loads and manages P(super) and P(direction) models per timeframe.
/// Uses direct Booster loading to avoid ModelManager schema parsing overhead.
pub struct SuperEntryModelManager {
    models: HashMap<i32, TfModels>,
    config: SuperEntryConfig,
}

impl SuperEntryModelManager {
    /// Create a new SuperEntryModelManager and load models from disk.
    ///
    /// Models are loaded one-by-one directly via Booster::load(),
    /// bypassing ModelManager's schema parsing which can crash on
    /// large XGBoost JSON model dumps.
    pub fn new(config: SuperEntryConfig, use_gpu: bool) -> Result<Self> {
        let mut models = HashMap::new();
        let timeframes = SuperEntryConfig::timeframes();

        // Select device: GPU for batch inference (history), CPU for single inference (realtime)
        let device = if use_gpu { Device::Cuda } else { Device::Cpu };
        info!("Loading super_entry models (device={:?})...", device);

        for &tf in timeframes {
            let super_path = config.model_path(tf);
            let dir_path = config.direction_model_path(tf);

            // Load P(super) model — required
            if !std::path::Path::new(&super_path).exists() {
                warn!("Model not found: {} — skipping TF {}m", super_path, tf);
                continue;
            }

            let super_model = match Booster::load(&super_path, device) {
                Ok(b) => {
                    info!("✅ Loaded super_entry TF {}m: {} (device={:?})", tf, super_path, device);
                    b
                }
                Err(e) => {
                    // Fallback to CPU if GPU load fails
                    if use_gpu {
                        warn!("GPU load failed for {}, falling back to CPU: {}", super_path, e);
                        match Booster::load(&super_path, Device::Cpu) {
                            Ok(b) => {
                                info!("✅ Loaded super_entry TF {}m: {} (CPU fallback)", tf, super_path);
                                b
                            }
                            Err(e2) => {
                                warn!("❌ Failed to load {} on CPU too: {}", super_path, e2);
                                continue;
                            }
                        }
                    } else {
                        warn!("❌ Failed to load {}: {}", super_path, e);
                        continue;
                    }
                }
            };

            // Load P(direction) model — optional
            let dir_model = if std::path::Path::new(&dir_path).exists() {
                match Booster::load(&dir_path, device) {
                    Ok(b) => {
                        info!("✅ Loaded super_dir TF {}m: {} (device={:?})", tf, dir_path, device);
                        Some(b)
                    }
                    Err(e) => {
                        // Fallback to CPU
                        if use_gpu {
                            Booster::load(&dir_path, Device::Cpu).ok()
                        } else {
                            warn!("⚠️  Failed to load dir model {}: {}, using neutral direction", dir_path, e);
                            None
                        }
                    }
                }
            } else {
                None
            };

            models.insert(tf, TfModels { super_model, dir_model });

            // Small yield between model loads
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        info!(
            "Models loaded: {} TFs with super_entry, {} with direction",
            models.len(),
            models.values().filter(|m| m.dir_model.is_some()).count()
        );

        Ok(Self { models, config })
    }

    /// Check if models are loaded for any timeframe
    pub fn has_models(&self) -> bool {
        !self.models.is_empty()
    }

    /// Check if a model is available for a specific timeframe
    pub fn has_model_for_tf(&self, tf_minutes: i32) -> bool {
        self.models.contains_key(&tf_minutes)
    }

    /// Get available timeframes with loaded models
    pub fn available_timeframes(&self) -> Vec<i32> {
        let mut tfs: Vec<i32> = self.models.keys().copied().collect();
        tfs.sort();
        tfs
    }

    /// Run inference for a single candle's features.
    pub fn predict(
        &self,
        tf_minutes: i32,
        features: &[f32],
        _use_gpu: bool,
    ) -> Result<Option<SuperEntryPrediction>> {
        let tf_models = match self.models.get(&tf_minutes) {
            Some(m) => m,
            None => return Ok(None),
        };

        let ncol = features.len();

        // P(super) prediction
        let p_super = tf_models.super_model
            .predict_dense_cpu(features, 1, ncol, ModelKind::Regressor1)?
            .first()
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        // P(direction=LONG) prediction
        let p_long = match &tf_models.dir_model {
            Some(dir) => dir
                .predict_dense_cpu(features, 1, ncol, ModelKind::Regressor1)?
                .first()
                .copied()
                .unwrap_or(0.5)
                .clamp(0.0, 1.0),
            None => 0.5, // Neutral if no direction model
        };

        let direction = if p_long >= 0.5 { 1 } else { -1 };
        let target = self.config.target_pct_for_tf(tf_minutes);

        Ok(Some(SuperEntryPrediction {
            p_super,
            p_long,
            direction,
            estimated_magnitude_pct: target * p_super as f64,
        }))
    }

    /// Batch inference for multiple feature rows.
    pub fn predict_batch(
        &self,
        tf_minutes: i32,
        features_batch: &[f32],
        nrow: usize,
        ncol: usize,
        _use_gpu: bool,
    ) -> Result<Vec<SuperEntryPrediction>> {
        let tf_models = match self.models.get(&tf_minutes) {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };

        // Batch P(super) prediction
        let p_super_vec = tf_models.super_model
            .predict_dense_cpu(features_batch, nrow, ncol, ModelKind::Regressor1)?;

        // Batch P(direction) prediction
        let p_long_vec = match &tf_models.dir_model {
            Some(dir) => dir.predict_dense_cpu(features_batch, nrow, ncol, ModelKind::Regressor1)?,
            None => vec![0.5f32; nrow],
        };

        let target = self.config.target_pct_for_tf(tf_minutes);

        let mut predictions = Vec::with_capacity(nrow);
        for i in 0..nrow {
            let p_super = p_super_vec.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let p_long = p_long_vec.get(i).copied().unwrap_or(0.5).clamp(0.0, 1.0);
            let direction = if p_long >= 0.5 { 1 } else { -1 };

            predictions.push(SuperEntryPrediction {
                p_super,
                p_long,
                direction,
                estimated_magnitude_pct: target * p_super as f64,
            });
        }

        Ok(predictions)
    }

    /// Get the underlying config
    pub fn config(&self) -> &SuperEntryConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_direction() {
        let pred = SuperEntryPrediction {
            p_super: 0.8,
            p_long: 0.7,
            direction: 1,
            estimated_magnitude_pct: 3.0,
        };
        assert_eq!(pred.direction, 1);
        assert!(pred.p_super > 0.5);
    }
}
