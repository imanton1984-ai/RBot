// strategies/ml_entry_strategy/src/model.rs
//
// Model wrapper for Super Entry XGBoost models.
//
// Wraps the existing ModelManager from compute/predictors/src/ml/
// to load and run inference on super_entry and direction models.
//
// Models:
//   - super_entry_v1_tf{X}.ubj — binary classifier: P(super move)
//   - super_dir_v1_tf{X}.ubj — binary classifier: P(direction=LONG)

use anyhow::Result;
use tracing::{info, warn};

use predictors::ml::model_manager::ModelManager;
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

/// Super Entry Model Manager
///
/// Loads and manages P(super) and P(direction) models per timeframe.
pub struct SuperEntryModelManager {
    model_manager: ModelManager,
    config: SuperEntryConfig,
    available_tfs: Vec<i32>,
}

impl SuperEntryModelManager {
    /// Create a new SuperEntryModelManager and load models from disk.
    ///
    /// # Arguments
    /// * `config` - Strategy configuration
    /// * `use_gpu` - Whether to attempt GPU loading for inference.
    ///
    /// NOTE: XGBoost GPU inference works by loading models on CPU first,
    /// then the Booster with device=cuda set uses GPU predictor automatically
    /// during inference. Loading separate GPU copies often causes CUDA OOM  
    /// when loading 12+ models simultaneously. We always load CPU models
    /// and let XGBoost handle GPU prediction internally.
    pub fn new(config: SuperEntryConfig, use_gpu: bool) -> Result<Self> {
        // Always load on CPU to avoid CUDA OOM from loading 12+ GPU model copies.
        // XGBoost's gpu_predictor handles GPU inference at predict time,
        // even with CPU-loaded models (it transfers DMatrix to GPU internally).
        let mut model_manager = ModelManager::new(false);

        let timeframes = SuperEntryConfig::timeframes();
        let mut available_tfs = Vec::new();

        if use_gpu {
            info!("GPU requested — models will be loaded on CPU, XGBoost uses GPU predictor at inference time");
        }

        // Load super_entry models (binary classifier: P(super move))
        info!("Loading super_entry models...");
        model_manager.load_models_for_timeframes(
            "super_entry",
            &config.model_path_template,
            timeframes,
            false, // CPU load only — GPU inference handled by XGBoost internally
        )?;

        // Load direction models (binary classifier: P(LONG))
        info!("Loading super_dir models...");
        model_manager.load_models_for_timeframes(
            "super_dir",
            &config.direction_model_path_template,
            timeframes,
            false, // CPU load only
        )?;

        // Check which TFs have models
        for &tf in timeframes {
            let key = format!("super_entry_tf{}", tf);
            if model_manager.has_model(&key) {
                available_tfs.push(tf);
                info!("✅ super_entry model available for TF {}m", tf);
            } else {
                warn!("⚠️  super_entry model NOT found for TF {}m", tf);
            }
        }

        if available_tfs.is_empty() {
            warn!("No super_entry models found! Strategy will not generate signals.");
        }

        Ok(Self {
            model_manager,
            config,
            available_tfs,
        })
    }

    /// Check if models are loaded for any timeframe
    pub fn has_models(&self) -> bool {
        !self.available_tfs.is_empty()
    }

    /// Check if a model is available for a specific timeframe
    pub fn has_model_for_tf(&self, tf_minutes: i32) -> bool {
        self.available_tfs.contains(&tf_minutes)
    }

    /// Get available timeframes with loaded models
    pub fn available_timeframes(&self) -> &[i32] {
        &self.available_tfs
    }

    /// Run inference for a single candle's features.
    ///
    /// # Arguments
    /// * `tf_minutes` - Timeframe in minutes
    /// * `features` - Feature vector (must match model schema)
    /// * `use_gpu` - Whether to use GPU for inference
    ///
    /// # Returns
    /// SuperEntryPrediction with p_super, p_long, direction
    pub fn predict(
        &self,
        tf_minutes: i32,
        features: &[f32],
        use_gpu: bool,
    ) -> Result<Option<SuperEntryPrediction>> {
        let key_super = format!("super_entry_tf{}", tf_minutes);
        let key_dir = format!("super_dir_tf{}", tf_minutes);

        // P(super) prediction
        let p_super = match self.model_manager.predict_one(&key_super, features, use_gpu)? {
            Some(v) => v.first().copied().unwrap_or(0.0).clamp(0.0, 1.0),
            None => return Ok(None), // Model not loaded
        };

        // P(direction=LONG) prediction
        let p_long = match self.model_manager.predict_one(&key_dir, features, use_gpu)? {
            Some(v) => v.first().copied().unwrap_or(0.5).clamp(0.0, 1.0),
            None => 0.5, // Default to neutral if direction model not available
        };

        let direction = if p_long >= 0.5 { 1 } else { -1 };
        let target = self.config.target_pct_for_tf(tf_minutes);
        let estimated_magnitude = target * p_super as f64;

        Ok(Some(SuperEntryPrediction {
            p_super,
            p_long,
            direction,
            estimated_magnitude_pct: estimated_magnitude,
        }))
    }

    /// Batch inference for multiple feature rows.
    ///
    /// # Arguments
    /// * `tf_minutes` - Timeframe in minutes
    /// * `features_batch` - Row-major flattened feature matrix [nrow * ncol]
    /// * `nrow` - Number of rows / examples
    /// * `ncol` - Number of features per row
    /// * `use_gpu` - Whether to use GPU
    ///
    /// # Returns
    /// Vec of SuperEntryPrediction, one per row
    pub fn predict_batch(
        &self,
        tf_minutes: i32,
        features_batch: &[f32],
        nrow: usize,
        ncol: usize,
        use_gpu: bool,
    ) -> Result<Vec<SuperEntryPrediction>> {
        let key_super = format!("super_entry_tf{}", tf_minutes);
        let key_dir = format!("super_dir_tf{}", tf_minutes);

        // Batch P(super) prediction
        let p_super_vec = match self.model_manager.predict_batch(
            &key_super, features_batch, nrow, ncol, use_gpu,
        )? {
            Some(v) => v,
            None => return Ok(Vec::new()),
        };

        // Batch P(direction) prediction
        let p_long_vec = match self.model_manager.predict_batch(
            &key_dir, features_batch, nrow, ncol, use_gpu,
        )? {
            Some(v) => v,
            None => vec![0.5f32; nrow], // Default neutral
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
