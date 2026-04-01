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
use std::sync::OnceLock;

/// Target move percentage thresholds per timeframe (in %).
/// A "super move" is defined as price moving >= this threshold
/// within the lookahead window of 20 bars.
///
/// These thresholds are set to achieve ~30% positive class balance.
/// Higher thresholds = fewer "super" labels = more selective model.
pub fn tf_target_move_pct() -> HashMap<i32, f64> {
    let mut m = HashMap::new();
    m.insert(1, 1.2);     // 1m:  1.2% (was 0.95)
    m.insert(5, 2.8);     // 5m:  2.8% (was 2.0)
    m.insert(15, 3.5);    // 15m: 3.5% (was 2.1)
    m.insert(60, 5.0);    // 1h:  5.0% (was 3.2)
    m.insert(240, 7.5);   // 4h:  7.5% (was 4.3)
    m.insert(1440, 10.0); // 1d:  10.0% (was 6.4)
    m
}

/// Get target move % for a specific timeframe (minutes).
/// Returns None if TF is not configured.
pub fn get_target_move_pct(tf_minutes: i32) -> Option<f64> {
    tf_target_move_pct().get(&tf_minutes).copied()
}

/// Full configuration for the Super Entry Strategy (NoDir)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperEntryConfig {
    /// Number of warmup candles needed before generating training examples.
    /// First `warmup_bars` candles are used to compute indicator state.
    pub warmup_bars: usize,

    /// Number of future bars to look ahead for measuring max move.
    pub lookahead_bars: usize,

    /// Minimum P(super) probability threshold to trigger an entry signal.
    /// This is the GLOBAL fallback. Per-TF thresholds are in the scorer.
    pub p_threshold: f64,

    /// Stop-loss as a fraction of the target move (e.g., 0.5 = SL = 50% of TP).
    pub sl_fraction: f64,

    /// Maximum bars to hold a trade before force-closing.
    /// If a trade doesn't hit TP or SL within max_hold_bars, close at market.
    /// Set to 0 to use lookahead_bars as max hold (original behavior).
    /// Env: SUPER_ENTRY_MAX_HOLD_BARS
    pub max_hold_bars: usize,

    /// Target move percentages per timeframe (override defaults).
    /// Key = tf_minutes, Value = target_pct.
    pub tf_targets: HashMap<i32, f64>,

    /// Model file template (e.g., "models/super_entry_v1_tf{tf}.ubj")
    /// DEPRECATED in NoDir: super_long/super_short use hardcoded paths.
    pub model_path_template: String,

    /// Direction model file template
    /// DEPRECATED in NoDir: no separate direction model.
    pub direction_model_path_template: String,

    /// Minimum magnitude to include in training (filter noise)
    pub min_magnitude_pct: f64,

    /// Train/test split ratio (fraction of pairs for training)
    pub train_split_ratio: f64,

    /// DEPRECATED: Danger zone filter is removed.
    pub enable_danger_zone_filter: bool,

    /// DEPRECATED in NoDir: direction confidence thresholds are no longer used.
    /// Per-TF thresholds are now in the scorer (ScorerConfig::p_thresholds).
    /// Kept for backward compat with existing code that references this field.
    pub direction_confidence_thresholds: HashMap<i32, f64>,
}

impl Default for SuperEntryConfig {
    fn default() -> Self {
        Self {
            warmup_bars: 300,
            lookahead_bars: 25, // was 20
            p_threshold: 0.55,  //was 0.55
            sl_fraction: 0.65, //was 0.5
            max_hold_bars: 25,  // Force-close after 25 bars (reduces expired trades)
            tf_targets: tf_target_move_pct(),
            model_path_template: "models/super_entry_v1_tf{tf}.ubj".to_string(),
            direction_model_path_template: "models/super_dir_v1_tf{tf}.ubj".to_string(),
            min_magnitude_pct: 0.5,
            train_split_ratio: 0.5,
            enable_danger_zone_filter: false, // DEPRECATED: only ML models affect signal
            direction_confidence_thresholds: Self::default_direction_confidence_thresholds(),
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
        if let Ok(v) = std::env::var("SUPER_ENTRY_MAX_HOLD_BARS") {
            if let Ok(n) = v.parse() { cfg.max_hold_bars = n; }
        }
        if let Ok(v) = std::env::var("SUPER_ENTRY_TRAIN_SPLIT") {
            if let Ok(n) = v.parse() { cfg.train_split_ratio = n; }
        }
        if let Ok(v) = std::env::var("SUPER_ENTRY_DANGER_ZONE_FILTER") {
            cfg.enable_danger_zone_filter = v == "true" || v == "1";
        }

        // Per-TF direction confidence thresholds:
        // SUPER_ENTRY_DIR_CONF_15=0.75, SUPER_ENTRY_DIR_CONF_60=0.70, etc.
        for &tf in &[5, 15, 60, 240, 1440] {
            let env_key = format!("SUPER_ENTRY_DIR_CONF_{}", tf);
            if let Ok(v) = std::env::var(&env_key) {
                if let Ok(n) = v.parse::<f64>() {
                    cfg.direction_confidence_thresholds.insert(tf, n.clamp(0.5, 1.0));
                }
            }
        }

        cfg
    }

    /// Default direction confidence thresholds per TF.
    /// Higher TFs are more reliable → lower confidence threshold.
    /// Lower TFs are noisy → require stronger model confidence.
    fn default_direction_confidence_thresholds() -> HashMap<i32, f64> {
        let mut m = HashMap::new();
        m.insert(5, 0.80);     // 5m: very noisy, require P(class) >= 0.80
        m.insert(15, 0.75);    // 15m: require P(class) >= 0.75
        m.insert(60, 0.70);    // 1h: require P(class) >= 0.70
        m.insert(240, 0.65);   // 4h: require P(class) >= 0.65
        m.insert(1440, 0.65);  // 1d: require P(class) >= 0.65
        m
    }

    /// Get direction confidence threshold for a specific TF.
    /// Returns the per-TF threshold, or 0.65 as default fallback.
    pub fn dir_confidence_for_tf(&self, tf_minutes: i32) -> f64 {
        self.direction_confidence_thresholds
            .get(&tf_minutes)
            .copied()
            .unwrap_or(0.65)
    }

    /// Effective max hold bars for backtesting.
    /// If max_hold_bars is 0, falls back to lookahead_bars.
    pub fn effective_max_hold(&self) -> usize {
        if self.max_hold_bars > 0 {
            self.max_hold_bars
        } else {
            self.lookahead_bars
        }
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

    /// Production timeframes for inference — configurable via env var.
    ///
    /// Env: SUPER_ENTRY_TIMEFRAMES (comma-separated minutes, e.g. "15,60,240,1440")
    /// Default: "15,60,240,1440" (15m, 1h, 4h, 1d)
    ///
    /// Controls which TFs get indicator computation, feature snapshots, and
    /// signal generation. Candle loading is NOT affected — all TFs still load.
    ///
    /// Previously hardcoded as [1, 5, 15, 60, 240, 1440]; 1m/5m removed by default
    /// because they are too noisy for the super_entry model.
    pub fn timeframes() -> &'static [i32] {
        static TIMEFRAMES: OnceLock<Vec<i32>> = OnceLock::new();
        TIMEFRAMES.get_or_init(|| {
            let raw = std::env::var("SUPER_ENTRY_TIMEFRAMES")
                .unwrap_or_else(|_| "5,15,60,240,1440".to_string());
            let mut tfs: Vec<i32> = raw
                .split(',')
                .filter_map(|s| s.trim().parse::<i32>().ok())
                .filter(|&m| [1, 5, 15, 60, 240, 1440].contains(&m))
                .collect();
            tfs.sort_unstable();
            tfs.dedup();
            if tfs.is_empty() {
                tracing::warn!(
                    "SUPER_ENTRY_TIMEFRAMES is empty or invalid ('{}'). Falling back to [15,60,240,1440]",
                    raw
                );
                tfs = vec![5, 15, 60, 240, 1440];
            }
            tracing::info!("SuperEntry compute timeframes: {:?} (from SUPER_ENTRY_TIMEFRAMES)", tfs);
            tfs
        })
    }

    /// All timeframes including 1d (for dataset building / training only).
    /// This is NOT affected by SUPER_ENTRY_TIMEFRAMES — always returns all TFs.
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

    /// Check if a given TF (in minutes) is in the active compute set.
    /// Use this to skip indicator computation for disabled TFs.
    pub fn is_tf_active(tf_minutes: i32) -> bool {
        Self::timeframes().contains(&tf_minutes)
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
    "alligator_jaw", "alligator_teeth", "alligator_lips",
    "mfi", "fibo_pivot", "fibo_r1", "fibo_s1",
    "supertrend", "supertrend_dir", "cmf",
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
    "mfi_norm",        // mfi / 100
    "price_vs_fibo_pivot", // (close - fibo_pivot) / close * 100
    "price_vs_supertrend", // (close - supertrend) / close * 100
    "alligator_spread",    // (jaw - lips) / close * 100
];

/// Lookback windows (in bars) for computing dynamic/temporal features.
/// Used by both dataset builder and inference pipeline.
/// v2: expanded from [3,5,10,15] to include 25 and 50 for broader trend context.
/// For 1h candles: lb25 = 1 day, lb50 = 2 days — critical for direction prediction.
pub const DYNAMIC_LOOKBACK_WINDOWS: &[usize] = &[3, 5, 10, 15, 25, 50];

/// Lookback window for BB Squeeze percentile calculation.
/// 100 bars captures enough history to determine if current BB width
/// is a squeeze (low percentile) or expansion (high percentile).
/// For 15m candles: 100 bars = 25 hours. For 1h: 100 bars = ~4 days.
pub const BB_SQUEEZE_LOOKBACK: usize = 100;

/// Dynamic (temporal) features — computed from lookback over candle history.
/// These capture HOW indicators are CHANGING, not just their current value.
/// Critical for direction prediction: a static snapshot doesn't tell if
/// the trend is accelerating or decelerating.
///
/// Naming: `{metric}_lb{N}` where N is the lookback window in bars.
///
/// Keep in sync with Python trainer (DYNAMIC_FEATURES list).
pub const DYNAMIC_FEATURES: &[&str] = &[
    // --- Per-window features (8 metrics × 6 windows = 48) ---
    // Window 3 bars
    "price_return_lb3",         // (close[t] - close[t-3]) / close[t] * 100
    "atr_ratio_lb3",            // atr[t] / atr[t-3] - 1 (vol expansion/contraction)
    "rsi_slope_lb3",            // (rsi[t] - rsi[t-3]) / 100
    "trend_persist_lb3",        // sum(trend[t-2..=t]) / 3 — trend consistency
    "trend_short_persist_lb3",  // sum(trend_short[t-2..=t]) / 3
    "adx_slope_lb3",            // (adx[t] - adx[t-3]) / 100
    "macd_hist_slope_lb3",      // (macd_hist[t] - macd_hist[t-3]) / close * 1000
    "ema20_direction_lb3",      // (ema20[t] - ema20[t-3]) / close * 100
    // Window 5 bars
    "price_return_lb5",
    "atr_ratio_lb5",
    "rsi_slope_lb5",
    "trend_persist_lb5",
    "trend_short_persist_lb5",
    "adx_slope_lb5",
    "macd_hist_slope_lb5",
    "ema20_direction_lb5",
    // Window 10 bars
    "price_return_lb10",
    "atr_ratio_lb10",
    "rsi_slope_lb10",
    "trend_persist_lb10",
    "trend_short_persist_lb10",
    "adx_slope_lb10",
    "macd_hist_slope_lb10",
    "ema20_direction_lb10",
    // Window 15 bars
    "price_return_lb15",
    "atr_ratio_lb15",
    "rsi_slope_lb15",
    "trend_persist_lb15",
    "trend_short_persist_lb15",
    "adx_slope_lb15",
    "macd_hist_slope_lb15",
    "ema20_direction_lb15",
    // Window 25 bars (v2: ~1 day for 1h candles)
    "price_return_lb25",
    "atr_ratio_lb25",
    "rsi_slope_lb25",
    "trend_persist_lb25",
    "trend_short_persist_lb25",
    "adx_slope_lb25",
    "macd_hist_slope_lb25",
    "ema20_direction_lb25",
    // Window 50 bars (v2: ~2 days for 1h candles)
    "price_return_lb50",
    "atr_ratio_lb50",
    "rsi_slope_lb50",
    "trend_persist_lb50",
    "trend_short_persist_lb50",
    "adx_slope_lb50",
    "macd_hist_slope_lb50",
    "ema20_direction_lb50",
    // --- Aggregate/cross-window features (6) ---
    "supertrend_consistency",   // sum(supertrend_dir[t-49..=t]) / 50 (v2: expanded from 15 to 50)
    "trend_alignment",          // trend[t] * trend_short[t]
    "price_accel",              // (ret_5 - ret_5_prev) normalized — momentum acceleration
    "volume_trend_ratio",       // mean_vol_recent_5 / mean_vol_prev_5
    "ema_convergence_change",   // (ema20-ema50 now) - (ema20-ema50 5 bars ago) / close * 100
    "high_low_pressure",        // bias of wicks over last 10 bars (buying/selling pressure)
    // --- v3: Rate-of-Change & Momentum Dynamics (14 features) ---
    // Price RoC — short-term momentum critical for direction
    "price_roc_lb1",            // 1-bar return: (close[t] - close[t-1]) / close[t-1] * 100
    "price_roc_lb2",            // 2-bar return: (close[t] - close[t-2]) / close[t-2] * 100
    "price_accel_1bar",         // 1-bar acceleration: roc_lb1[t] vs roc_lb1[t-1]
    "price_accel_3bar",         // 3-bar acceleration: return_lb3[t] vs return_lb3 at [t-3]
    // Volume RoC — volume dynamics (is volume BUILDING or fading?)
    "volume_roc_lb1",           // volume[t] / volume[t-1] - 1 (single bar change)
    "volume_roc_lb3",           // volume[t] / mean(volume[t-3..t-1]) - 1
    "volume_roc_lb5",           // volume[t] / mean(volume[t-5..t-1]) - 1
    "volume_roc_lb10",          // volume[t] / mean(volume[t-10..t-1]) - 1
    "volume_accel",             // volume momentum change: vol_roc_recent vs vol_roc_prev
    // BB Squeeze — low percentile of BB width = volatility compression = breakout imminent
    "bb_squeeze_pctl",          // percentile(bb_width_pct, lookback=100): 0=squeeze, 1=expansion
    // OBV Divergence — smart money detection
    "obv_price_divergence",     // sign mismatch between OBV slope and price slope over 10 bars
    // MACD Histogram Acceleration — momentum acceleration beyond simple slope
    "macd_hist_roc_lb1",        // macd_hist[t] - macd_hist[t-1], normalized
    "macd_hist_roc_lb3",        // macd_hist[t] - macd_hist[t-3], normalized
    "macd_hist_accel",          // second derivative: roc_lb1[t] - roc_lb1[t-1]
    // --- v4: HTF (Higher Timeframe) features (3) ---
    // Brings the heuristic filter's cross-TF intelligence directly into the model.
    // XGBoost can learn non-linear interactions (e.g., HTF bearish + BB squeeze → short).
    "htf_trend",                // trend direction from higher TF (-1, 0, +1)
    "htf_supertrend_dir",       // supertrend direction from higher TF (-1 or +1)
    "htf_ema20_slope",          // EMA20 slope from higher TF (normalized by HTF close)
    // --- v4: Killer features for direction prediction (5) ---
    // Structural metrics: WHERE is price relative to liquidity + volume distribution
    "dist_to_low_50",           // (close - min_low_50) / close * 100 — distance to liquidity below
    "dist_to_high_50",          // (max_high_50 - close) / close * 100 — distance to liquidity above
    "acute_wick_rejection_2bar",// wick bias over last 2 bars / ATR — immediate reaction signal
    "bb_squeeze_x_vwap",       // (1 - bb_squeeze_pctl) * price_vs_vwap — squeeze direction hint
    "volume_up_vs_down_lb10",  // sum(vol_up) / sum(vol_down) over 10 bars — buyer/seller pressure
];

/// Total number of features = INDICATOR(33) + DERIVED(19) + DYNAMIC(76) = 128
pub fn total_feature_count() -> usize {
    INDICATOR_FEATURES.len() + DERIVED_FEATURES.len() + DYNAMIC_FEATURES.len()
}

/// Number of static features (indicators + derived, no lookback needed)
pub fn static_feature_count() -> usize {
    INDICATOR_FEATURES.len() + DERIVED_FEATURES.len()
}

/// Number of dynamic features (require candle history lookback)
pub fn dynamic_feature_count() -> usize {
    DYNAMIC_FEATURES.len()
}

/// Maximum lookback needed for dynamic features.
/// Takes the max of the standard lookback windows and the BB squeeze window,
/// since BB squeeze percentile needs 100 bars of bb_width_pct history.
pub fn max_dynamic_lookback() -> usize {
    let window_max = *DYNAMIC_LOOKBACK_WINDOWS.last().unwrap_or(&15);
    window_max.max(BB_SQUEEZE_LOOKBACK)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = SuperEntryConfig::default();
        assert_eq!(cfg.warmup_bars, 300);
        assert_eq!(cfg.lookahead_bars, 25);
        assert!((cfg.p_threshold - 0.55).abs() < 1e-6);
        assert_eq!(cfg.max_hold_bars, 25);
        assert_eq!(cfg.effective_max_hold(), 25);
    }

    #[test]
    fn test_max_hold_zero_fallback() {
        let mut cfg = SuperEntryConfig::default();
        cfg.max_hold_bars = 0;
        assert_eq!(cfg.effective_max_hold(), cfg.lookahead_bars);
    }

    #[test]
    fn test_target_move_pct() {
        let cfg = SuperEntryConfig::default();
        assert!((cfg.target_pct_for_tf(1) - 1.2).abs() < 1e-6);
        assert!((cfg.target_pct_for_tf(60) - 5.0).abs() < 1e-6);
        assert!((cfg.target_pct_for_tf(1440) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_sl_pct() {
        let cfg = SuperEntryConfig::default();
        // SL = 65% of TP target
        let expected_1m = 1.2 * 0.65;
        assert!((cfg.sl_pct_for_tf(1) - expected_1m).abs() < 1e-6);
        let expected_1h = 5.0 * 0.65;
        assert!((cfg.sl_pct_for_tf(60) - expected_1h).abs() < 1e-6);
    }

    #[test]
    fn test_model_paths() {
        let cfg = SuperEntryConfig::default();
        assert_eq!(cfg.model_path(60), "models/super_entry_v1_tf60.ubj");
        assert_eq!(cfg.direction_model_path(15), "models/super_dir_v1_tf15.ubj");
    }

    #[test]
    fn test_feature_count() {
        assert_eq!(INDICATOR_FEATURES.len(), 33);
        assert_eq!(DERIVED_FEATURES.len(), 19);
        // 8 metrics × 6 windows + 6 aggregate + 14 v3 (RoC/squeeze/divergence/accel)
        // + 3 v4 HTF + 5 v4 killer = 76
        assert_eq!(DYNAMIC_FEATURES.len(), 76);
        assert_eq!(static_feature_count(), 52);
        assert_eq!(total_feature_count(), 128); // 52 + 76
        assert_eq!(dynamic_feature_count(), 76);
        // BB squeeze needs 100 bars, which is > 50 (max lookback window)
        assert_eq!(max_dynamic_lookback(), 100);
    }
}
