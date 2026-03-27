// strategies/ml_entry_strategy/src/direction/mod.rs
//
// Direction Model v4 — CNN-like Pattern Recognition on XGBoost
//
// PHILOSOPHY:
//   Stop feeding indicators. Feed only pure price action patterns.
//   XGBoost learns to recognize patterns in a sliding window of normalized
//   OHLCV data — effectively imitating a 1D-CNN on tabular data.
//
// HOW IT WORKS:
//   1. Take a sliding window of W candles (e.g., 30)
//   2. Normalize all prices relative to the first candle's open price
//      → makes patterns independent of absolute price level (stationarity!)
//   3. Flatten the window into a single wide row: W × features_per_candle columns
//   4. XGBoost builds trees that find combinations of features within the window
//      → effectively learning local patterns like a convolutional layer
//
// KEY INSIGHT:
//   Previous v1-v3 approaches with 20-128 indicator features achieved ~0.50 accuracy
//   (random). Indicators are lagging and noisy. Raw price structure contains
//   the actual predictive signal — if normalized properly for stationarity.
//
// TUNABLE PARAMETERS (via DirectionConfig):
//   - window_size: how many candles in the pattern (10-60, default 30)
//   - prediction_horizon: how far ahead to predict (3-50, default 10)
//   - feature_set: which features per candle (OHLC, +Volume, +Body, +InterBar)
//   - up_threshold_pct / down_threshold_pct: dead zone for FLAT classification
//   - label_method: FinalReturn or MaxExcursion
//
// TARGET:
//   3-class: UP (1) / FLAT (0) / DOWN (-1)
//   Binary variant: UP (1) / DOWN (0) — excludes FLAT from training

pub mod features;
pub mod dataset;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// P(super) target move thresholds per TF (from config.rs TF_TARGET_MOVE_PCT).
/// Used as default UP/DOWN thresholds — the model predicts whether price will
/// reach the SAME target as the super model within the prediction horizon.
fn tf_target_move_pct() -> HashMap<i32, f64> {
    let mut m = HashMap::new();
    m.insert(1, 1.2);
    m.insert(5, 2.8);
    m.insert(15, 3.5);
    m.insert(60, 5.0);
    m.insert(240, 7.5);
    m.insert(1440, 10.0);
    m
}

/// Which features to extract per candle in the sliding window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureSet {
    /// OHLC relative to ref price (4 per candle)
    OhlcOnly,
    /// OHLC + normalized volume (5 per candle)
    OhlcVolume,
    /// OHLC + volume + body/wick analysis (8 per candle)
    OhlcVolumeBody,
    /// OHLC + volume + body + inter-bar dynamics (10 per candle)
    Full,
}

impl FeatureSet {
    /// Number of features per candle for this feature set.
    pub fn features_per_candle(&self) -> usize {
        match self {
            FeatureSet::OhlcOnly => 4,
            FeatureSet::OhlcVolume => 5,
            FeatureSet::OhlcVolumeBody => 8,
            FeatureSet::Full => 10,
        }
    }

    pub fn from_str_name(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "ohlc" | "ohlc_only" => FeatureSet::OhlcOnly,
            "ohlcv" | "ohlc_volume" => FeatureSet::OhlcVolume,
            "ohlcvb" | "ohlc_volume_body" => FeatureSet::OhlcVolumeBody,
            "full" => FeatureSet::Full,
            _ => FeatureSet::OhlcVolume,
        }
    }
}

/// How to determine the label (UP/DOWN/FLAT) for training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LabelMethod {
    /// Label based on close[t+horizon] vs close[t]
    FinalReturn,
    /// Label based on max favorable excursion within horizon
    /// (did price EVER reach threshold, even if it came back?)
    MaxExcursion,
}

/// Full configuration for the Direction v4 Pattern Model.
///
/// All parameters are tunable via environment variables for quick experimentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionConfig {
    // ─── Pattern Window ───
    /// Number of candles in the sliding window (the "image width").
    /// More candles = broader context, but more features = potential overfitting.
    /// Env: DIR_WINDOW_SIZE (default: 30)
    pub window_size: usize,

    // ─── Prediction Target ───
    /// How many candles ahead to predict.
    /// Short horizon (3-5) = more noise, longer (15-25) = smoother but slower.
    /// Env: DIR_PREDICTION_HORIZON (default: 10)
    pub prediction_horizon: usize,

    // ─── Feature Configuration ───
    /// Which features to extract per candle.
    /// Env: DIR_FEATURE_SET (default: "ohlcv")
    pub feature_set: FeatureSet,

    // ─── Label Thresholds ───
    /// Minimum upward move (%) to label as UP.
    /// Set to 0.0 for pure binary (any positive return = UP).
    /// Env: DIR_UP_THRESHOLD (default: 0.3)
    pub up_threshold_pct: f64,

    /// Minimum downward move (%) to label as DOWN.
    /// Env: DIR_DOWN_THRESHOLD (default: 0.3)
    pub down_threshold_pct: f64,

    /// How to compute the label.
    /// Env: DIR_LABEL_METHOD ("final_return" or "max_excursion", default: "final_return")
    pub label_method: LabelMethod,

    // ─── Training Filters ───
    /// Exclude FLAT examples from training? (train only on UP/DOWN)
    /// Reduces noise but may create distribution mismatch at inference.
    /// Env: DIR_EXCLUDE_FLAT (default: false)
    pub exclude_flat_from_training: bool,

    /// Minimum volume filter: skip candles with volume below this percentile.
    /// Helps filter dead/illiquid periods.
    /// Env: DIR_MIN_VOLUME_PCTL (default: 0.0 = no filter)
    pub min_volume_percentile: f64,
}

impl Default for DirectionConfig {
    fn default() -> Self {
        Self {
            // Window=20: fewer features (100 vs 150), less overfitting.
            // 20 candles × 15m = 5 hours of context, enough for pattern recognition.
            window_size: 20,
            // Horizon=25: matches P(super) lookahead_bars.
            // Longer horizon = less noisy labels than horizon=10.
            prediction_horizon: 25,
            feature_set: FeatureSet::OhlcVolume,
            // Thresholds from P(super) targets — overridden per-TF.
            // Global fallback value; use threshold_for_tf() for TF-specific.
            up_threshold_pct: 3.5,
            down_threshold_pct: 3.5,
            // MaxExcursion: label based on "did price EVER reach threshold?" 
            // This matches P(super) logic exactly (TP hit within lookahead).
            label_method: LabelMethod::MaxExcursion,
            exclude_flat_from_training: false,
            min_volume_percentile: 0.0,
        }
    }
}

impl DirectionConfig {
    /// Load from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("DIR_WINDOW_SIZE") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.window_size = n.clamp(5, 120);
            }
        }
        if let Ok(v) = std::env::var("DIR_PREDICTION_HORIZON") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.prediction_horizon = n.clamp(1, 100);
            }
        }
        if let Ok(v) = std::env::var("DIR_FEATURE_SET") {
            cfg.feature_set = FeatureSet::from_str_name(&v);
        }
        if let Ok(v) = std::env::var("DIR_UP_THRESHOLD") {
            if let Ok(n) = v.parse::<f64>() {
                cfg.up_threshold_pct = n.max(0.0);
            }
        }
        if let Ok(v) = std::env::var("DIR_DOWN_THRESHOLD") {
            if let Ok(n) = v.parse::<f64>() {
                cfg.down_threshold_pct = n.max(0.0);
            }
        }
        if let Ok(v) = std::env::var("DIR_LABEL_METHOD") {
            cfg.label_method = match v.to_lowercase().as_str() {
                "max_excursion" | "excursion" => LabelMethod::MaxExcursion,
                _ => LabelMethod::FinalReturn,
            };
        }
        if let Ok(v) = std::env::var("DIR_EXCLUDE_FLAT") {
            cfg.exclude_flat_from_training = v == "1" || v == "true";
        }
        if let Ok(v) = std::env::var("DIR_MIN_VOLUME_PCTL") {
            if let Ok(n) = v.parse::<f64>() {
                cfg.min_volume_percentile = n.clamp(0.0, 1.0);
            }
        }

        cfg
    }

    /// Total number of features in a single flattened row.
    pub fn total_features(&self) -> usize {
        self.window_size * self.feature_set.features_per_candle()
    }

    /// Minimum number of candles required (window + prediction horizon).
    pub fn min_candles_required(&self) -> usize {
        self.window_size + self.prediction_horizon
    }

    /// Get TF-specific threshold (%) for UP/DOWN labeling.
    /// Uses P(super) target thresholds: 1m=1.2%, 5m=2.8%, 15m=3.5%, 60m=5.0%, 240m=7.5%, 1440m=10%.
    /// Falls back to self.up_threshold_pct if TF not found.
    ///
    /// ENV override: DIR_UP_THRESHOLD (applies to all TFs, overrides per-TF).
    pub fn threshold_for_tf(&self, tf_minutes: i32) -> f64 {
        // If user explicitly set threshold via env, use that for all TFs
        if std::env::var("DIR_UP_THRESHOLD").is_ok() {
            return self.up_threshold_pct;
        }
        // Otherwise use P(super) TF-specific targets
        tf_target_move_pct()
            .get(&tf_minutes)
            .copied()
            .unwrap_or(self.up_threshold_pct)
    }

    /// Create a TF-specific copy of this config with correct thresholds.
    pub fn for_tf(&self, tf_minutes: i32) -> Self {
        let threshold = self.threshold_for_tf(tf_minutes);
        let mut cfg = self.clone();
        cfg.up_threshold_pct = threshold;
        cfg.down_threshold_pct = threshold;
        cfg
    }

    /// Print config summary to log.
    pub fn log_summary(&self) {
        tracing::info!("Direction v4 Config:");
        tracing::info!("  window_size: {}", self.window_size);
        tracing::info!("  prediction_horizon: {}", self.prediction_horizon);
        tracing::info!("  feature_set: {:?} ({} per candle)", self.feature_set, self.feature_set.features_per_candle());
        tracing::info!("  total_features: {}", self.total_features());
        tracing::info!("  up_threshold: {:.2}% (default, per-TF from P(super))", self.up_threshold_pct);
        tracing::info!("  down_threshold: {:.2}%", self.down_threshold_pct);
        tracing::info!("  label_method: {:?}", self.label_method);
        tracing::info!("  exclude_flat: {}", self.exclude_flat_from_training);
        tracing::info!("  Per-TF thresholds (from P(super)):");
        for &tf in &[5, 15, 60, 240, 1440] {
            tracing::info!("    TF {:>5}m: {:.1}%", tf, self.threshold_for_tf(tf));
        }
    }
}
