// strategies/ml_entry_strategy/src/config.rs
//
// Configuration for ML Entry Strategy (Super Entry Model)
//
// Contains:
//   - TF_TARGET_MOVE_PCT: minimum % move per timeframe to qualify as "super"
//   - Lookahead window (default 20 bars)
//   - p_threshold: minimum P(super) to trigger entry
//   - SL/TP parameters
//   - Warmup bars (first 300 candles used for indicator context)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Target move percentage thresholds per timeframe (in %).
/// A "super move" is defined as price moving >= this threshold
/// within the lookahead window of 20 bars.
///
/// These are based on realistic TP3 targets adjusted per TF volatility.
pub fn tf_target_move_pct() -> HashMap<i32, f64> {
    let mut m = HashMap::new();
    m.insert(1, 1.0);     // 1m:  1.0%
    m.insert(5, 1.75);    // 5m:  1.75%
    m.insert(15, 2.75);   // 15m: 2.75%
    m.insert(60, 3.75);   // 1h:  3.75%
    m.insert(240, 4.75);  // 4h:  4.75%
    m.insert(1440, 5.75); // 1d:  5.75%
    m
}

/// Get target move % for a specific timeframe (minutes).
/// Returns None if TF is not configured.
pub fn get_target_move_pct(tf_minutes: i32) -> Option<f64> {
    tf_target_move_pct().get(&tf_minutes).copied()
}

/// Full configuration for the Super Entry Strategy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperEntryConfig {
    /// Number of warmup candles needed before generating training examples.
    /// First `warmup_bars` candles are used to compute indicator state.
    pub warmup_bars: usize,

    /// Number of future bars to look ahead for measuring max move.
    pub lookahead_bars: usize,

    /// Minimum P(super) probability threshold to trigger an entry signal.
    pub p_threshold: f64,

    /// Stop-loss as a fraction of the target move (e.g., 0.5 = SL = 50% of TP).
    pub sl_fraction: f64,

    /// Target move percentages per timeframe (override defaults).
    /// Key = tf_minutes, Value = target_pct.
    pub tf_targets: HashMap<i32, f64>,

    /// Model file template (e.g., "models/super_entry_v1_tf{tf}.ubj")
    pub model_path_template: String,

    /// Direction model file template
    pub direction_model_path_template: String,

    /// Minimum magnitude to include in training (filter noise)
    pub min_magnitude_pct: f64,

    /// Train/test split ratio (fraction of pairs for training)
    pub train_split_ratio: f64,
}

impl Default for SuperEntryConfig {
    fn default() -> Self {
        Self {
            warmup_bars: 300,
            lookahead_bars: 20,
            p_threshold: 0.55,
            sl_fraction: 0.5,
            tf_targets: tf_target_move_pct(),
            model_path_template: "models/super_entry_v1_tf{tf}.ubj".to_string(),
            direction_model_path_template: "models/super_dir_v1_tf{tf}.ubj".to_string(),
            min_magnitude_pct: 0.1,
            train_split_ratio: 0.8,
        }
    }
}

impl SuperEntryConfig {
    /// Load config from environment variables with defaults
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("SUPER_ENTRY_WARMUP_BARS") {
            if let Ok(n) = v.parse() { cfg.warmup_bars = n; }
        }
        if let Ok(v) = std::env::var("SUPER_ENTRY_LOOKAHEAD") {
            if let Ok(n) = v.parse() { cfg.lookahead_bars = n; }
        }
        if let Ok(v) = std::env::var("SUPER_ENTRY_P_THRESHOLD") {
            if let Ok(n) = v.parse() { cfg.p_threshold = n; }
        }
        if let Ok(v) = std::env::var("SUPER_ENTRY_SL_FRACTION") {
            if let Ok(n) = v.parse() { cfg.sl_fraction = n; }
        }
        if let Ok(v) = std::env::var("SUPER_ENTRY_TRAIN_SPLIT") {
            if let Ok(n) = v.parse() { cfg.train_split_ratio = n; }
        }

        cfg
    }

    /// Get target move % for a timeframe, using override or default.
    pub fn target_pct_for_tf(&self, tf_minutes: i32) -> f64 {
        self.tf_targets
            .get(&tf_minutes)
            .copied()
            .unwrap_or_else(|| get_target_move_pct(tf_minutes).unwrap_or(2.0))
    }

    /// Get the stop-loss % for a given TF (derived from target and sl_fraction).
    pub fn sl_pct_for_tf(&self, tf_minutes: i32) -> f64 {
        self.target_pct_for_tf(tf_minutes) * self.sl_fraction
    }

    /// All supported timeframes for inference.
    /// NOTE: 1440 (1d) disabled by default — 99.7% super rate makes model useless.
    /// XGBoost also has a limit of ~10 simultaneous Booster objects in release mode.
    /// Train 1d models but only use 1m-4h for production signals.
    pub fn timeframes() -> &'static [i32] {
        &[1, 5, 15, 60, 240]
    }

    /// All timeframes including 1d (for dataset building / training only)
    pub fn all_timeframes() -> &'static [i32] {
        &[1, 5, 15, 60, 240, 1440]
    }

    /// Resolve model path for a given timeframe
    pub fn model_path(&self, tf_minutes: i32) -> String {
        self.model_path_template
            .replace("{tf}", &tf_minutes.to_string())
    }

    /// Resolve direction model path for a given timeframe
    pub fn direction_model_path(&self, tf_minutes: i32) -> String {
        self.direction_model_path_template
            .replace("{tf}", &tf_minutes.to_string())
    }
}

/// Feature names used by the super_entry model.
/// These match the columns from market.indicators_wide.
/// Must be kept in sync with Python trainer.
pub const INDICATOR_FEATURES: &[&str] = &[
    "rsi", "cci", "stoch_k", "stoch_d", "williams",
    "macd", "macd_signal", "macd_hist",
    "adx", "sma", "ema_20", "ema_50", "ema_200",
    "bb_upper", "bb_mid", "bb_lower", "atr",
    "obv", "vwap", "volume_spike",
    "trend", "trend_short", "poc",
];

/// Derived features computed from raw indicators.
/// Keep in sync with Python trainer.
pub const DERIVED_FEATURES: &[&str] = &[
    "rsi_norm",        // RSI / 100 (normalized)
    "cci_norm",        // CCI / 200 (normalized)
    "stoch_norm",      // stoch_k / 100
    "williams_norm",   // (williams + 100) / 100
    "bb_position",     // (close - bb_lower) / (bb_upper - bb_lower)
    "bb_width_pct",    // (bb_upper - bb_lower) / close * 100
    "atr_pct",         // atr / close * 100
    "price_vs_sma",    // (close - sma) / close * 100
    "price_vs_ema20",  // (close - ema_20) / close * 100
    "price_vs_ema50",  // (close - ema_50) / close * 100
    "price_vs_ema200", // (close - ema_200) / close * 100
    "price_vs_vwap",   // (close - vwap) / close * 100
    "macd_norm",       // macd_hist / close * 1000
    "obv_change_pct",  // not available without history — set 0
    "volume_spike_flag", // volume_spike > 2.0
];

/// Total number of features = INDICATOR_FEATURES + DERIVED_FEATURES
pub fn total_feature_count() -> usize {
    INDICATOR_FEATURES.len() + DERIVED_FEATURES.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = SuperEntryConfig::default();
        assert_eq!(cfg.warmup_bars, 300);
        assert_eq!(cfg.lookahead_bars, 20);
        assert!((cfg.p_threshold - 0.55).abs() < 1e-6);
    }

    #[test]
    fn test_target_move_pct() {
        let cfg = SuperEntryConfig::default();
        assert!((cfg.target_pct_for_tf(1) - 1.0).abs() < 1e-6);
        assert!((cfg.target_pct_for_tf(60) - 3.75).abs() < 1e-6);
        assert!((cfg.target_pct_for_tf(1440) - 5.75).abs() < 1e-6);
    }

    #[test]
    fn test_sl_pct() {
        let cfg = SuperEntryConfig::default();
        // SL = 50% of TP target
        assert!((cfg.sl_pct_for_tf(1) - 0.5).abs() < 1e-6);
        assert!((cfg.sl_pct_for_tf(60) - 1.875).abs() < 1e-6);
    }

    #[test]
    fn test_model_paths() {
        let cfg = SuperEntryConfig::default();
        assert_eq!(cfg.model_path(60), "models/super_entry_v1_tf60.ubj");
        assert_eq!(cfg.direction_model_path(15), "models/super_dir_v1_tf15.ubj");
    }

    #[test]
    fn test_feature_count() {
        assert_eq!(INDICATOR_FEATURES.len(), 23);
        assert_eq!(DERIVED_FEATURES.len(), 15);
        assert_eq!(total_feature_count(), 38);
    }
}
