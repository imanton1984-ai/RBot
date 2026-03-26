// strategies/ml_entry_strategy/src/model.rs
//
// Model wrapper for Super Entry + Direction v3 XGBoost models.
//
// Models:
//   - super_entry_v1_tf{X}.ubj — binary classifier: P(super move) [128 features]
//   - direction_v3_tf{X}.ubj  — regressor: direction_quality [32 features]
//
// Direction v3 uses DIFFERENT features from super_entry:
//   - 26 features extracted from the 128-feature vector (subset)
//   - 5 BTC relative features (not in 128-feature set, need BTC candle data)
//   - 1 HTF feature (from 128-feature set)
//
// At inference:
//   - prediction = direction_v3_model.predict(32_features)
//   - direction = sign(prediction): +1 = LONG, -1 = SHORT
//   - confidence = abs(prediction): higher = more certain
//   - If confidence < threshold → skip (direction uncertain)
//
// Falls back to legacy super_dir_v1_tf{X}.ubj if direction_v3 not found.

use anyhow::Result;
use tracing::{info, warn};
use std::collections::HashMap;

use predictors::ml::xgb_runtime::{Booster, Device, ModelKind};
use crate::config::SuperEntryConfig;
use crate::direction::features::DIRECTION_V3_FEATURE_COUNT;

/// Prediction output from the super entry model
#[derive(Debug, Clone, Copy)]
pub struct SuperEntryPrediction {
    /// Probability that this is a "super" move (P >= target move %)
    pub p_super: f32,
    /// For v3: raw regression output (sign = direction, abs = confidence)
    /// For legacy: P(direction=LONG) in [0, 1]
    pub p_long: f32,
    /// Derived: direction (1 = LONG, -1 = SHORT)
    pub direction: i8,
    /// Derived: direction confidence (abs of regression prediction, or |p_long - 0.5| for legacy)
    pub dir_confidence: f32,
    /// Derived: expected magnitude estimate (optional, from p_super * target)
    pub estimated_magnitude_pct: f64,
    /// Whether direction model v3 was used (True) or legacy/none (False)
    pub direction_v3: bool,
}

/// Loaded model pair for one TF
struct TfModels {
    super_model: Booster,
    /// Direction v3 model (regression, 32 features) — preferred
    dir_v3_model: Option<Booster>,
    /// Legacy direction model (binary, 128 features) — fallback
    dir_legacy_model: Option<Booster>,
}

/// Super Entry Model Manager
///
/// Loads and manages P(super) and Direction models per timeframe.
pub struct SuperEntryModelManager {
    models: HashMap<i32, TfModels>,
    config: SuperEntryConfig,
}

impl SuperEntryModelManager {
    /// Create a new SuperEntryModelManager and load models from disk.
    pub fn new(config: SuperEntryConfig, use_gpu: bool) -> Result<Self> {
        let mut models = HashMap::new();
        let timeframes = SuperEntryConfig::timeframes();

        let device = if use_gpu { Device::Cuda } else { Device::Cpu };
        info!("Loading super_entry models (device={:?})...", device);

        for &tf in timeframes {
            let super_path = config.model_path(tf);
            let dir_v3_path = format!("models/direction_v3_tf{}.ubj", tf);
            let dir_legacy_path = config.direction_model_path(tf);

            // Load P(super) model — required
            if !std::path::Path::new(&super_path).exists() {
                warn!("Model not found: {} — skipping TF {}m", super_path, tf);
                continue;
            }

            let super_model = match load_booster(&super_path, device, use_gpu) {
                Some(b) => {
                    info!("✅ Loaded super_entry TF {}m: {}", tf, super_path);
                    b
                }
                None => continue,
            };

            // Try to load Direction v3 model first (preferred)
            let dir_v3_model = if std::path::Path::new(&dir_v3_path).exists() {
                match load_booster(&dir_v3_path, device, use_gpu) {
                    Some(b) => {
                        info!("✅ Loaded direction_v3 TF {}m: {} (32 features, regression)",
                              tf, dir_v3_path);
                        Some(b)
                    }
                    None => None,
                }
            } else {
                None
            };

            // Fallback to legacy direction model
            let dir_legacy_model = if dir_v3_model.is_none()
                && std::path::Path::new(&dir_legacy_path).exists()
            {
                match load_booster(&dir_legacy_path, device, use_gpu) {
                    Some(b) => {
                        info!("⚠️  Loaded legacy super_dir TF {}m: {} (128 features, binary)",
                              tf, dir_legacy_path);
                        Some(b)
                    }
                    None => None,
                }
            } else {
                None
            };

            if dir_v3_model.is_none() && dir_legacy_model.is_none() {
                warn!("⚠️  No direction model for TF {}m — using neutral direction", tf);
            }

            models.insert(tf, TfModels {
                super_model,
                dir_v3_model,
                dir_legacy_model,
            });

            // Small yield between model loads
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        info!(
            "Models loaded: {} TFs, {} with direction_v3, {} with legacy direction",
            models.len(),
            models.values().filter(|m| m.dir_v3_model.is_some()).count(),
            models.values().filter(|m| m.dir_legacy_model.is_some()).count(),
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

    /// Check if direction v3 model is available for a specific timeframe
    pub fn has_direction_v3_for_tf(&self, tf_minutes: i32) -> bool {
        self.models
            .get(&tf_minutes)
            .map_or(false, |m| m.dir_v3_model.is_some())
    }

    /// Get available timeframes with loaded models
    pub fn available_timeframes(&self) -> Vec<i32> {
        let mut tfs: Vec<i32> = self.models.keys().copied().collect();
        tfs.sort();
        tfs
    }

    /// Run inference for a single candle's features.
    ///
    /// # Arguments
    /// * `tf_minutes` — timeframe
    /// * `features` — 128 super_entry features (f32)
    /// * `dir_v3_features` — optional 32 direction v3 features (f64, will be converted)
    /// * `_use_gpu` — reserved
    pub fn predict(
        &self,
        tf_minutes: i32,
        features: &[f32],
        dir_v3_features: Option<&[f64]>,
        _use_gpu: bool,
    ) -> Result<Option<SuperEntryPrediction>> {
        let tf_models = match self.models.get(&tf_minutes) {
            Some(m) => m,
            None => return Ok(None),
        };

        let ncol = features.len();

        // P(super) prediction using 128 features
        let p_super = tf_models.super_model
            .predict_dense_cpu(features, 1, ncol, ModelKind::Regressor1)?
            .first()
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        // Direction prediction
        let (p_long, direction, dir_confidence, direction_v3) =
            self.predict_direction(tf_models, features, ncol, dir_v3_features)?;

        let target = self.config.target_pct_for_tf(tf_minutes);

        Ok(Some(SuperEntryPrediction {
            p_super,
            p_long,
            direction,
            dir_confidence,
            estimated_magnitude_pct: target * p_super as f64,
            direction_v3,
        }))
    }

    /// Batch inference for multiple feature rows.
    ///
    /// For direction v3 batch inference, pass `dir_v3_features_batch` with
    /// batch_size * 32 f32 values.
    pub fn predict_batch(
        &self,
        tf_minutes: i32,
        features_batch: &[f32],
        nrow: usize,
        ncol: usize,
        dir_v3_features_batch: Option<&[f32]>,
        _use_gpu: bool,
    ) -> Result<Vec<SuperEntryPrediction>> {
        let tf_models = match self.models.get(&tf_minutes) {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };

        // Batch P(super) prediction using 128 features
        let p_super_vec = tf_models.super_model
            .predict_dense_cpu(features_batch, nrow, ncol, ModelKind::Regressor1)?;

        // Batch direction prediction
        let (p_long_vec, directions, dir_confidences, used_v3) =
            self.predict_direction_batch(
                tf_models, features_batch, nrow, ncol, dir_v3_features_batch,
            )?;

        let target = self.config.target_pct_for_tf(tf_minutes);

        let mut predictions = Vec::with_capacity(nrow);
        for i in 0..nrow {
            let p_super = p_super_vec.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let p_long = p_long_vec.get(i).copied().unwrap_or(0.5);
            let direction = directions.get(i).copied().unwrap_or(1);
            let dir_confidence = dir_confidences.get(i).copied().unwrap_or(0.0);

            predictions.push(SuperEntryPrediction {
                p_super,
                p_long,
                direction,
                dir_confidence,
                estimated_magnitude_pct: target * p_super as f64,
                direction_v3: used_v3,
            });
        }

        Ok(predictions)
    }

    /// Predict direction using v3 model (preferred) or legacy model (fallback).
    fn predict_direction(
        &self,
        tf_models: &TfModels,
        features_128: &[f32],
        ncol_128: usize,
        dir_v3_features: Option<&[f64]>,
    ) -> Result<(f32, i8, f32, bool)> {
        // Try direction v3 model first
        if let (Some(dir_model), Some(v3_feats)) = (&tf_models.dir_v3_model, dir_v3_features) {
            if v3_feats.len() == DIRECTION_V3_FEATURE_COUNT {
                let v3_f32: Vec<f32> = v3_feats.iter().map(|&v| v as f32).collect();
                let raw_pred = dir_model
                    .predict_dense_cpu(&v3_f32, 1, DIRECTION_V3_FEATURE_COUNT, ModelKind::Regressor1)?
                    .first()
                    .copied()
                    .unwrap_or(0.0);

                let direction: i8 = if raw_pred >= 0.0 { 1 } else { -1 };
                let confidence = raw_pred.abs();
                // Store raw_pred as p_long for compatibility (rescaled to 0..1 range)
                let p_long = 0.5 + raw_pred.clamp(-0.5, 0.5);

                return Ok((p_long, direction, confidence, true));
            }
        }

        // Fallback to legacy direction model (128 features, binary)
        if let Some(dir_model) = &tf_models.dir_legacy_model {
            let p_long = dir_model
                .predict_dense_cpu(features_128, 1, ncol_128, ModelKind::Regressor1)?
                .first()
                .copied()
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);

            let direction: i8 = if p_long >= 0.5 { 1 } else { -1 };
            let confidence = (p_long - 0.5).abs();

            return Ok((p_long, direction, confidence, false));
        }

        // No direction model — neutral
        Ok((0.5, 1, 0.0, false))
    }

    /// Batch direction prediction.
    fn predict_direction_batch(
        &self,
        tf_models: &TfModels,
        features_128_batch: &[f32],
        nrow: usize,
        ncol_128: usize,
        dir_v3_features_batch: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<i8>, Vec<f32>, bool)> {
        // Try direction v3 model first
        if let (Some(dir_model), Some(v3_batch)) = (&tf_models.dir_v3_model, dir_v3_features_batch) {
            let expected_len = nrow * DIRECTION_V3_FEATURE_COUNT;
            if v3_batch.len() == expected_len {
                let raw_preds = dir_model.predict_dense_cpu(
                    v3_batch, nrow, DIRECTION_V3_FEATURE_COUNT, ModelKind::Regressor1,
                )?;

                let mut p_longs = Vec::with_capacity(nrow);
                let mut directions = Vec::with_capacity(nrow);
                let mut confidences = Vec::with_capacity(nrow);

                for &raw in &raw_preds {
                    let dir: i8 = if raw >= 0.0 { 1 } else { -1 };
                    let conf = raw.abs();
                    let p_long = 0.5 + raw.clamp(-0.5, 0.5);
                    p_longs.push(p_long);
                    directions.push(dir);
                    confidences.push(conf);
                }

                return Ok((p_longs, directions, confidences, true));
            }
        }

        // Fallback to legacy
        if let Some(dir_model) = &tf_models.dir_legacy_model {
            let p_long_vec = dir_model.predict_dense_cpu(
                features_128_batch, nrow, ncol_128, ModelKind::Regressor1,
            )?;

            let mut directions = Vec::with_capacity(nrow);
            let mut confidences = Vec::with_capacity(nrow);

            for &p_long in &p_long_vec {
                let p = p_long.clamp(0.0, 1.0);
                directions.push(if p >= 0.5 { 1 } else { -1 });
                confidences.push((p - 0.5).abs());
            }

            let p_longs: Vec<f32> = p_long_vec.iter().map(|&p| p.clamp(0.0, 1.0)).collect();
            return Ok((p_longs, directions, confidences, false));
        }

        // No direction model
        Ok((
            vec![0.5f32; nrow],
            vec![1i8; nrow],
            vec![0.0f32; nrow],
            false,
        ))
    }

    /// Get the underlying config
    pub fn config(&self) -> &SuperEntryConfig {
        &self.config
    }
}

/// Helper: load a Booster with GPU → CPU fallback
fn load_booster(path: &str, device: Device, use_gpu: bool) -> Option<Booster> {
    match Booster::load(path, device) {
        Ok(b) => Some(b),
        Err(e) => {
            if use_gpu {
                warn!("GPU load failed for {}, falling back to CPU: {}", path, e);
                Booster::load(path, Device::Cpu).ok()
            } else {
                warn!("❌ Failed to load {}: {}", path, e);
                None
            }
        }
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
            dir_confidence: 0.2,
            estimated_magnitude_pct: 3.0,
            direction_v3: false,
        };
        assert_eq!(pred.direction, 1);
        assert!(pred.p_super > 0.5);
    }

    #[test]
    fn test_prediction_v3_direction() {
        // Simulate v3 regression prediction
        let raw_pred = 0.15f32; // positive = LONG
        let direction: i8 = if raw_pred >= 0.0 { 1 } else { -1 };
        let confidence = raw_pred.abs();

        let pred = SuperEntryPrediction {
            p_super: 0.75,
            p_long: 0.5 + raw_pred.clamp(-0.5, 0.5),
            direction,
            dir_confidence: confidence,
            estimated_magnitude_pct: 3.5,
            direction_v3: true,
        };

        assert_eq!(pred.direction, 1);
        assert!(pred.dir_confidence > 0.1);
        assert!(pred.direction_v3);
    }
}
