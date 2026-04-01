// strategies/ml_entry_strategy/src/model.rs
//
// Model wrapper for Super Entry NoDir XGBoost models.
//
// NoDir approach — NO separate direction model:
//   - super_long_v1_tf{X}.ubj  — binary classifier: P(strong upward move) [128 features]
//   - super_short_v1_tf{X}.ubj — binary classifier: P(strong downward move) [128 features]
//
// Both models use the same 128-feature set (ALL_FEATURES from config.rs).
// Direction is embedded in the label: P(super_long) fires → LONG, P(super_short) fires → SHORT.
//
// Conflict filter (both models fire):
//   - If margin = |P(super_long) - P(super_short)| < CONFLICT_MIN_MARGIN → skip (ambiguous)
//   - Otherwise, pick the direction with higher probability
//
// Per-TF probability thresholds (from config or order_manager.toml):
//   - 15m: p >= 0.85
//   - 1h:  p >= 0.75
//   - 4h:  p >= 0.75

use anyhow::Result;
use tracing::{info, warn};
use std::collections::HashMap;

use predictors::ml::xgb_runtime::{Booster, Device, ModelKind};
use crate::config::SuperEntryConfig;

/// Prediction output from the NoDir super entry models
#[derive(Debug, Clone, Copy)]
pub struct SuperEntryPrediction {
    /// P(strong upward move) from super_long model
    pub p_super_long: f32,
    /// P(strong downward move) from super_short model
    pub p_super_short: f32,
    /// Derived: direction (1 = LONG, -1 = SHORT, 0 = no signal)
    pub direction: i8,
    /// The probability that was used for the chosen direction
    /// LONG → p_super_long, SHORT → p_super_short
    pub p_super: f32,
    /// Margin between the two probabilities: |p_super_long - p_super_short|
    pub conflict_margin: f32,
    /// Whether both models fired above threshold (conflict state)
    pub is_conflict: bool,
    /// Derived: expected magnitude estimate (optional, from p_super * target)
    pub estimated_magnitude_pct: f64,
}

/// Backward-compatible: keep DirectionModelVersion for any code that references it
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DirectionModelVersion {
    /// NoDir — direction embedded in super_long/super_short labels
    NoDir,
    /// Direction v4 — CNN-like pattern model (DEPRECATED, not loaded)
    V4,
    /// Direction v3 — 32-feature regression model (DEPRECATED)
    V3,
    /// Legacy binary classifier (DEPRECATED)
    Legacy,
    /// No direction model available
    None,
}

/// Loaded model pair for one TF (NoDir architecture)
struct TfModels {
    /// P(strong upward move) model — binary classifier
    super_long_model: Booster,
    /// P(strong downward move) model — binary classifier
    super_short_model: Booster,
}

/// Super Entry Model Manager (NoDir)
///
/// Loads and manages P(super_long) and P(super_short) models per timeframe.
/// No separate direction model — direction is embedded in the labels.
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
        info!("Loading NoDir super_entry models (device={:?})...", device);

        for &tf in timeframes {
            let long_path = format!("models/super_long_v1_tf{}.ubj", tf);
            let short_path = format!("models/super_short_v1_tf{}.ubj", tf);

            // Load P(super_long) model — required
            if !std::path::Path::new(&long_path).exists() {
                warn!("super_long model not found: {} — skipping TF {}m", long_path, tf);
                continue;
            }

            let super_long_model = match load_booster(&long_path, device, use_gpu) {
                Some(b) => {
                    info!("✅ Loaded super_long  TF {}m: {}", tf, long_path);
                    b
                }
                None => continue,
            };

            // Load P(super_short) model — required
            if !std::path::Path::new(&short_path).exists() {
                warn!("super_short model not found: {} — skipping TF {}m", short_path, tf);
                continue;
            }

            let super_short_model = match load_booster(&short_path, device, use_gpu) {
                Some(b) => {
                    info!("✅ Loaded super_short TF {}m: {}", tf, short_path);
                    b
                }
                None => continue,
            };

            models.insert(tf, TfModels {
                super_long_model,
                super_short_model,
            });

            // Small yield between model loads
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        info!(
            "NoDir models loaded: {} TFs with super_long+super_short pairs",
            models.len(),
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

    /// DEPRECATED: Direction v4/v3 are removed. Always returns false.
    pub fn has_direction_v4_for_tf(&self, _tf_minutes: i32) -> bool {
        false
    }

    /// DEPRECATED: Direction v3 is removed. Always returns false.
    pub fn has_direction_v3_for_tf(&self, _tf_minutes: i32) -> bool {
        false
    }

    /// DEPRECATED: No direction models in NoDir. Always returns false.
    pub fn has_any_direction_for_tf(&self, _tf_minutes: i32) -> bool {
        false
    }

    /// Get available timeframes with loaded models
    pub fn available_timeframes(&self) -> Vec<i32> {
        let mut tfs: Vec<i32> = self.models.keys().copied().collect();
        tfs.sort();
        tfs
    }

    /// Run inference for a single candle's features (NoDir).
    ///
    /// # Arguments
    /// * `tf_minutes` — timeframe
    /// * `features` — 128 features (f32)
    /// * `dir_v3_features` — IGNORED (kept for backward compat)
    /// * `dir_v4_features` — IGNORED (kept for backward compat)
    /// * `dir_v4_ncol` — IGNORED
    /// * `_use_gpu` — reserved
    pub fn predict(
        &self,
        tf_minutes: i32,
        features: &[f32],
        _dir_v3_features: Option<&[f64]>,
        _dir_v4_features: Option<&[f32]>,
        _dir_v4_ncol: usize,
        _use_gpu: bool,
    ) -> Result<Option<SuperEntryPrediction>> {
        let tf_models = match self.models.get(&tf_minutes) {
            Some(m) => m,
            None => return Ok(None),
        };

        let ncol = features.len();

        // P(super_long) prediction using 128 features
        let p_super_long = tf_models.super_long_model
            .predict_dense_cpu(features, 1, ncol, ModelKind::Regressor1)?
            .first()
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        // P(super_short) prediction using same 128 features
        let p_super_short = tf_models.super_short_model
            .predict_dense_cpu(features, 1, ncol, ModelKind::Regressor1)?
            .first()
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        let target = self.config.target_pct_for_tf(tf_minutes);
        let pred = build_prediction(p_super_long, p_super_short, target);

        Ok(Some(pred))
    }

    /// Batch inference for multiple feature rows (NoDir).
    ///
    /// Direction-related params are IGNORED — kept for backward compat signature.
    pub fn predict_batch(
        &self,
        tf_minutes: i32,
        features_batch: &[f32],
        nrow: usize,
        ncol: usize,
        _dir_v3_features_batch: Option<&[f32]>,
        _dir_v4_features_batch: Option<&[f32]>,
        _dir_v4_ncol: usize,
        _use_gpu: bool,
    ) -> Result<Vec<SuperEntryPrediction>> {
        let tf_models = match self.models.get(&tf_minutes) {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };

        // Batch P(super_long) prediction
        let p_long_vec = tf_models.super_long_model
            .predict_dense_cpu(features_batch, nrow, ncol, ModelKind::Regressor1)?;

        // Batch P(super_short) prediction using same features
        let p_short_vec = tf_models.super_short_model
            .predict_dense_cpu(features_batch, nrow, ncol, ModelKind::Regressor1)?;

        let target = self.config.target_pct_for_tf(tf_minutes);

        let mut predictions = Vec::with_capacity(nrow);
        for i in 0..nrow {
            let p_super_long = p_long_vec.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let p_super_short = p_short_vec.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);

            predictions.push(build_prediction(p_super_long, p_super_short, target));
        }

        Ok(predictions)
    }

    /// Get the underlying config
    pub fn config(&self) -> &SuperEntryConfig {
        &self.config
    }
}

/// Build a prediction from raw P(super_long) and P(super_short) probabilities.
///
/// Direction logic:
///   - LONG if p_super_long > p_super_short
///   - SHORT if p_super_short > p_super_long
///   - LONG (default) if equal
///
/// Conflict detection: both probabilities above some minimal threshold.
/// The scorer will apply per-TF thresholds and conflict filtering.
fn build_prediction(p_super_long: f32, p_super_short: f32, target_pct: f64) -> SuperEntryPrediction {
    let margin = (p_super_long - p_super_short).abs();

    // Both are "active" if both > 0.5 (raw model threshold)
    let is_conflict = p_super_long > 0.5 && p_super_short > 0.5;

    let (direction, p_super) = if p_super_long >= p_super_short {
        (1i8, p_super_long)
    } else {
        (-1i8, p_super_short)
    };

    SuperEntryPrediction {
        p_super_long,
        p_super_short,
        direction,
        p_super,
        conflict_margin: margin,
        is_conflict,
        estimated_magnitude_pct: target_pct * p_super as f64,
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
    fn test_prediction_long() {
        let pred = build_prediction(0.85, 0.30, 5.0);
        assert_eq!(pred.direction, 1);
        assert!((pred.p_super - 0.85).abs() < 1e-6);
        assert!((pred.p_super_long - 0.85).abs() < 1e-6);
        assert!((pred.conflict_margin - 0.55).abs() < 1e-5);
        assert!(!pred.is_conflict);
    }

    #[test]
    fn test_prediction_short() {
        let pred = build_prediction(0.20, 0.90, 5.0);
        assert_eq!(pred.direction, -1);
        assert!((pred.p_super - 0.90).abs() < 1e-6);
        assert!((pred.p_super_short - 0.90).abs() < 1e-6);
        assert!(!pred.is_conflict);
    }

    #[test]
    fn test_prediction_conflict() {
        let pred = build_prediction(0.75, 0.72, 5.0);
        assert_eq!(pred.direction, 1); // Long wins by margin
        assert!(pred.is_conflict);
        assert!((pred.conflict_margin - 0.03).abs() < 1e-5);
    }

    #[test]
    fn test_prediction_no_conflict_one_low() {
        let pred = build_prediction(0.80, 0.40, 5.0);
        assert_eq!(pred.direction, 1);
        assert!(!pred.is_conflict); // 0.40 < 0.5, not a conflict
    }

    #[test]
    fn test_prediction_equal() {
        let pred = build_prediction(0.60, 0.60, 5.0);
        assert_eq!(pred.direction, 1); // Default to LONG on tie
        assert!(pred.is_conflict);
        assert!((pred.conflict_margin).abs() < 1e-6);
    }
}
