// strategies/ml_entry_strategy/src/model.rs
//
// Model wrapper for Super Entry + Direction v4 XGBoost models.
//
// Models:
//   - super_entry_v1_tf{X}.ubj — binary classifier: P(super move) [128 features]
//   - direction_v4_tf{X}.ubj  — binary classifier: P(UP) [W*fpc pattern features]
//
// Direction v4 uses CNN-like sliding window OHLCV pattern features:
//   - compute_pattern_features() → window_size * features_per_candle values
//   - Output: P(UP) ∈ [0, 1] (binary:logistic)
//   - Direction: UP if P(UP) >= 0.5, DOWN otherwise
//   - Confidence: P(predicted_class) = max(P(UP), 1 - P(UP))
//
// Per-TF confidence thresholds (only ML models affect signal):
//   - 4h (240m): dir_confidence >= 0.65
//   - 1h (60m):  dir_confidence >= 0.70
//   - 15m:       dir_confidence >= 0.75
//
// Falls back to direction_v3_tf{X}.ubj → super_dir_v1_tf{X}.ubj if v4 not found.

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
    /// For v4: P(UP) ∈ [0, 1] directly from binary classifier
    /// For v3: raw regression output rescaled to [0, 1]
    /// For legacy: P(direction=LONG) in [0, 1]
    pub p_long: f32,
    /// Derived: direction (1 = LONG, -1 = SHORT)
    pub direction: i8,
    /// Derived: direction confidence
    /// For v4: P(predicted_class) = max(P(UP), 1-P(UP)) ∈ [0.5, 1.0]
    /// For v3: abs(regression prediction)
    /// For legacy: |p_long - 0.5|
    pub dir_confidence: f32,
    /// Derived: expected magnitude estimate (optional, from p_super * target)
    pub estimated_magnitude_pct: f64,
    /// Which direction model version was used
    pub direction_model_version: DirectionModelVersion,
}

/// Which direction model was used for prediction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DirectionModelVersion {
    /// Direction v4 — CNN-like pattern model (preferred)
    V4,
    /// Direction v3 — 32-feature regression model
    V3,
    /// Legacy binary classifier (128 features, same as super_entry)
    Legacy,
    /// No direction model available — neutral direction
    None,
}

/// Loaded model pair for one TF
struct TfModels {
    super_model: Booster,
    /// Direction v4 model (binary, pattern features) — preferred
    dir_v4_model: Option<Booster>,
    /// Direction v3 model (regression, 32 features) — fallback
    dir_v3_model: Option<Booster>,
    /// Legacy direction model (binary, 128 features) — last resort
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
            let dir_v4_path = format!("models/direction_v4_tf{}.ubj", tf);
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

            // ── Direction model loading: v4 → v3 → legacy ──

            // Try Direction v4 first (preferred — CNN-like pattern features)
            let dir_v4_model = if std::path::Path::new(&dir_v4_path).exists() {
                match load_booster(&dir_v4_path, device, use_gpu) {
                    Some(b) => {
                        info!("✅ Loaded direction_v4 TF {}m: {} (pattern features, binary)",
                              tf, dir_v4_path);
                        Some(b)
                    }
                    None => None,
                }
            } else {
                None
            };

            // Try Direction v3 if v4 not available
            let dir_v3_model = if dir_v4_model.is_none()
                && std::path::Path::new(&dir_v3_path).exists()
            {
                match load_booster(&dir_v3_path, device, use_gpu) {
                    Some(b) => {
                        info!("⚠️  Loaded direction_v3 TF {}m: {} (32 features, regression fallback)",
                              tf, dir_v3_path);
                        Some(b)
                    }
                    None => None,
                }
            } else {
                None
            };

            // Fallback to legacy direction model
            let dir_legacy_model = if dir_v4_model.is_none()
                && dir_v3_model.is_none()
                && std::path::Path::new(&dir_legacy_path).exists()
            {
                match load_booster(&dir_legacy_path, device, use_gpu) {
                    Some(b) => {
                        info!("⚠️  Loaded legacy super_dir TF {}m: {} (128 features, binary fallback)",
                              tf, dir_legacy_path);
                        Some(b)
                    }
                    None => None,
                }
            } else {
                None
            };

            if dir_v4_model.is_none() && dir_v3_model.is_none() && dir_legacy_model.is_none() {
                warn!("⚠️  No direction model for TF {}m — using neutral direction", tf);
            }

            models.insert(tf, TfModels {
                super_model,
                dir_v4_model,
                dir_v3_model,
                dir_legacy_model,
            });

            // Small yield between model loads
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        info!(
            "Models loaded: {} TFs, {} with direction_v4, {} with direction_v3, {} with legacy direction",
            models.len(),
            models.values().filter(|m| m.dir_v4_model.is_some()).count(),
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

    /// Check if direction v4 model is available for a specific timeframe
    pub fn has_direction_v4_for_tf(&self, tf_minutes: i32) -> bool {
        self.models
            .get(&tf_minutes)
            .map_or(false, |m| m.dir_v4_model.is_some())
    }

    /// Check if direction v3 model is available for a specific timeframe
    pub fn has_direction_v3_for_tf(&self, tf_minutes: i32) -> bool {
        self.models
            .get(&tf_minutes)
            .map_or(false, |m| m.dir_v3_model.is_some())
    }

    /// Check if any direction model (v4/v3/legacy) available for TF
    pub fn has_any_direction_for_tf(&self, tf_minutes: i32) -> bool {
        self.models.get(&tf_minutes).map_or(false, |m| {
            m.dir_v4_model.is_some() || m.dir_v3_model.is_some() || m.dir_legacy_model.is_some()
        })
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
    /// * `dir_v4_features` — optional v4 pattern features (f32)
    /// * `dir_v4_ncol` — number of v4 feature columns
    /// * `_use_gpu` — reserved
    pub fn predict(
        &self,
        tf_minutes: i32,
        features: &[f32],
        dir_v3_features: Option<&[f64]>,
        dir_v4_features: Option<&[f32]>,
        dir_v4_ncol: usize,
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

        // Direction prediction (v4 → v3 → legacy → neutral)
        let (p_long, direction, dir_confidence, version) =
            self.predict_direction(tf_models, features, ncol, dir_v3_features, dir_v4_features, dir_v4_ncol)?;

        let target = self.config.target_pct_for_tf(tf_minutes);

        Ok(Some(SuperEntryPrediction {
            p_super,
            p_long,
            direction,
            dir_confidence,
            estimated_magnitude_pct: target * p_super as f64,
            direction_model_version: version,
        }))
    }

    /// Batch inference for multiple feature rows.
    ///
    /// For direction v4 batch inference, pass `dir_v4_features_batch` with
    /// batch_size * v4_ncol f32 values.
    pub fn predict_batch(
        &self,
        tf_minutes: i32,
        features_batch: &[f32],
        nrow: usize,
        ncol: usize,
        dir_v3_features_batch: Option<&[f32]>,
        dir_v4_features_batch: Option<&[f32]>,
        dir_v4_ncol: usize,
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
        let (p_long_vec, directions, dir_confidences, version) =
            self.predict_direction_batch(
                tf_models, features_batch, nrow, ncol,
                dir_v3_features_batch, dir_v4_features_batch, dir_v4_ncol,
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
                direction_model_version: version,
            });
        }

        Ok(predictions)
    }

    /// Predict direction using v4 (preferred) → v3 → legacy → neutral.
    fn predict_direction(
        &self,
        tf_models: &TfModels,
        features_128: &[f32],
        ncol_128: usize,
        dir_v3_features: Option<&[f64]>,
        dir_v4_features: Option<&[f32]>,
        dir_v4_ncol: usize,
    ) -> Result<(f32, i8, f32, DirectionModelVersion)> {
        // ── Try Direction v4 first (CNN-like pattern model) ──
        if let (Some(dir_model), Some(v4_feats)) = (&tf_models.dir_v4_model, dir_v4_features) {
            if !v4_feats.is_empty() && v4_feats.len() == dir_v4_ncol {
                let raw_pred = dir_model
                    .predict_dense_cpu(v4_feats, 1, dir_v4_ncol, ModelKind::Regressor1)?
                    .first()
                    .copied()
                    .unwrap_or(0.5);

                // Binary model: P(UP) ∈ [0, 1]
                let p_up = raw_pred.clamp(0.0, 1.0);
                let (direction, confidence): (i8, f32) = if p_up >= 0.5 {
                    (1, p_up)          // UP with confidence = P(UP)
                } else {
                    (-1, 1.0 - p_up)   // DOWN with confidence = P(DOWN) = 1 - P(UP)
                };

                return Ok((p_up, direction, confidence, DirectionModelVersion::V4));
            }
        }

        // ── Try Direction v3 (32-feature regression) ──
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
                let p_long = 0.5 + raw_pred.clamp(-0.5, 0.5);

                return Ok((p_long, direction, confidence, DirectionModelVersion::V3));
            }
        }

        // ── Fallback to legacy direction model (128 features, binary) ──
        if let Some(dir_model) = &tf_models.dir_legacy_model {
            let p_long = dir_model
                .predict_dense_cpu(features_128, 1, ncol_128, ModelKind::Regressor1)?
                .first()
                .copied()
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);

            let direction: i8 = if p_long >= 0.5 { 1 } else { -1 };
            let confidence = (p_long - 0.5).abs();

            return Ok((p_long, direction, confidence, DirectionModelVersion::Legacy));
        }

        // No direction model — neutral
        Ok((0.5, 1, 0.0, DirectionModelVersion::None))
    }

    /// Batch direction prediction.
    fn predict_direction_batch(
        &self,
        tf_models: &TfModels,
        features_128_batch: &[f32],
        nrow: usize,
        ncol_128: usize,
        dir_v3_features_batch: Option<&[f32]>,
        dir_v4_features_batch: Option<&[f32]>,
        dir_v4_ncol: usize,
    ) -> Result<(Vec<f32>, Vec<i8>, Vec<f32>, DirectionModelVersion)> {
        // ── Try Direction v4 first ──
        if let (Some(dir_model), Some(v4_batch)) = (&tf_models.dir_v4_model, dir_v4_features_batch) {
            let expected_len = nrow * dir_v4_ncol;
            if v4_batch.len() == expected_len && dir_v4_ncol > 0 {
                let raw_preds = dir_model.predict_dense_cpu(
                    v4_batch, nrow, dir_v4_ncol, ModelKind::Regressor1,
                )?;

                let mut p_longs = Vec::with_capacity(nrow);
                let mut directions = Vec::with_capacity(nrow);
                let mut confidences = Vec::with_capacity(nrow);

                for &raw in &raw_preds {
                    let p_up = raw.clamp(0.0, 1.0);
                    let (dir, conf): (i8, f32) = if p_up >= 0.5 {
                        (1, p_up)
                    } else {
                        (-1, 1.0 - p_up)
                    };
                    p_longs.push(p_up);
                    directions.push(dir);
                    confidences.push(conf);
                }

                return Ok((p_longs, directions, confidences, DirectionModelVersion::V4));
            }
        }

        // ── Try Direction v3 ──
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

                return Ok((p_longs, directions, confidences, DirectionModelVersion::V3));
            }
        }

        // ── Fallback to legacy ──
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
            return Ok((p_longs, directions, confidences, DirectionModelVersion::Legacy));
        }

        // No direction model
        Ok((
            vec![0.5f32; nrow],
            vec![1i8; nrow],
            vec![0.0f32; nrow],
            DirectionModelVersion::None,
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
            dir_confidence: 0.7,
            estimated_magnitude_pct: 3.0,
            direction_model_version: DirectionModelVersion::V4,
        };
        assert_eq!(pred.direction, 1);
        assert!(pred.p_super > 0.5);
        assert_eq!(pred.direction_model_version, DirectionModelVersion::V4);
    }

    #[test]
    fn test_prediction_v4_direction() {
        // Simulate v4 binary prediction
        let p_up = 0.72f32; // P(UP) = 0.72 → LONG, confidence = 0.72
        let direction: i8 = if p_up >= 0.5 { 1 } else { -1 };
        let confidence = if p_up >= 0.5 { p_up } else { 1.0 - p_up };

        let pred = SuperEntryPrediction {
            p_super: 0.75,
            p_long: p_up,
            direction,
            dir_confidence: confidence,
            estimated_magnitude_pct: 3.5,
            direction_model_version: DirectionModelVersion::V4,
        };

        assert_eq!(pred.direction, 1);
        assert!((pred.dir_confidence - 0.72).abs() < 1e-6);
        assert_eq!(pred.direction_model_version, DirectionModelVersion::V4);
    }

    #[test]
    fn test_prediction_v4_short() {
        // P(UP) = 0.30 → SHORT, confidence = P(DOWN) = 0.70
        let p_up = 0.30f32;
        let direction: i8 = if p_up >= 0.5 { 1 } else { -1 };
        let confidence = if p_up >= 0.5 { p_up } else { 1.0 - p_up };

        assert_eq!(direction, -1);
        assert!((confidence - 0.70).abs() < 1e-6);
    }
}
