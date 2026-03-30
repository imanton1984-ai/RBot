// strategies/ml_entry_strategy/src/direction/features.rs
//
// Direction Model v4+ — CNN-like Sliding Window Feature Extraction
// with Summary Features and Cross-TF Context
//
// CORE IDEA:
//   Instead of computing 32 indicator-derived features for a single point in time,
//   we take a WINDOW of W raw candles and flatten them into a wide feature row.
//   XGBoost then builds decision trees that naturally find local patterns
//   within this temporal window — mimicking a 1D convolutional layer.
//
// v4+ ENHANCEMENTS:
//   After the raw sliding window features, we append:
//     1. SUMMARY features (11): slope, range_pos, momentum_diff, volatility_change,
//        volume_pressure, body_ratio×3, return_5bar, return_10bar, return_full_window
//     2. HTF context features (3): htf_trend, htf_range_pos, htf_vol_spike
//   Total = W × fpc + 11 + 3
//
// STATIONARITY:
//   Raw prices are non-stationary (BTC at 60K vs 100K).
//   We normalize ALL prices relative to the FIRST candle's OPEN in the window:
//     feature = (price - ref_open) / ref_open × 100 (% deviation from reference)
//   This makes the pattern invariant to absolute price level.
//
// FEATURE SETS (configurable):
//   OhlcOnly (4/candle):  open_rel, high_rel, low_rel, close_rel
//   OhlcVolume (5):       + volume_rel (normalized to window mean)
//   OhlcVolumeBody (8):   + body_pct, upper_wick_ratio, lower_wick_ratio
//   Full (10):            + bar_return, gap_pct
//
// COLUMN NAMING:
//   w{i}_{feature} — e.g. w0_open_rel, w0_high_rel, ..., w29_close_rel
//   summary_{name} — e.g. summary_slope, summary_range_pos
//   htf_{name}     — e.g. htf_trend, htf_range_pos, htf_vol_spike

use crate::dataset::CandleWithIndicators;
use super::{DirectionConfig, FeatureSet, SUMMARY_FEATURE_COUNT, HTF_FEATURE_COUNT};

// ═════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═════════════════════════════════════════════════════════════════════════════

/// Linear regression slope of a slice of values, normalized to [-1, +1].
///
/// Uses least-squares fit: slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
/// Then normalizes by dividing by the mean of y (so slope is "% change per bar").
fn linear_regression_slope(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }

    let n_f = n as f64;
    let x_mean = (n_f - 1.0) / 2.0;
    let y_mean: f64 = values.iter().sum::<f64>() / n_f;

    if y_mean.abs() < 1e-12 {
        return 0.0;
    }

    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &v) in values.iter().enumerate() {
        let xi = i as f64 - x_mean;
        num += xi * (v - y_mean);
        den += xi * xi;
    }

    if den.abs() < 1e-12 {
        return 0.0;
    }

    let raw_slope = num / den;
    // Normalize: slope per bar as fraction of mean price → clamp to [-1, +1]
    (raw_slope / y_mean * n_f).clamp(-1.0, 1.0)
}

/// Average True Range for a slice of candles.
fn avg_true_range(candles: &[CandleWithIndicators]) -> f64 {
    if candles.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for (i, c) in candles.iter().enumerate() {
        let tr = if i == 0 {
            c.high - c.low
        } else {
            let prev_close = candles[i - 1].close;
            (c.high - c.low)
                .max((c.high - prev_close).abs())
                .max((c.low - prev_close).abs())
        };
        sum += tr;
    }
    sum / candles.len() as f64
}

// ═════════════════════════════════════════════════════════════════════════════
// V4+ PATTERN FEATURES — CNN-like sliding window + summary + HTF
// ═════════════════════════════════════════════════════════════════════════════

/// Generate feature column names for the v4+ pattern model.
///
/// Returns a list of names:
///   ["w0_open_rel", ..., "w{W-1}_vol_rel",
///    "summary_slope", "summary_range_pos", ...,
///    "htf_trend", "htf_range_pos", "htf_vol_spike"]
pub fn direction_v4_feature_names(config: &DirectionConfig) -> Vec<String> {
    let mut names = Vec::with_capacity(config.total_features());

    // ── Sliding window features ──
    for i in 0..config.window_size {
        names.push(format!("w{}_open_rel", i));
        names.push(format!("w{}_high_rel", i));
        names.push(format!("w{}_low_rel", i));
        names.push(format!("w{}_close_rel", i));

        if matches!(config.feature_set, FeatureSet::OhlcVolume | FeatureSet::OhlcVolumeBody | FeatureSet::Full) {
            names.push(format!("w{}_vol_rel", i));
        }

        if matches!(config.feature_set, FeatureSet::OhlcVolumeBody | FeatureSet::Full) {
            names.push(format!("w{}_body_pct", i));
            names.push(format!("w{}_upper_wick", i));
            names.push(format!("w{}_lower_wick", i));
        }

        if matches!(config.feature_set, FeatureSet::Full) {
            names.push(format!("w{}_bar_return", i));
            names.push(format!("w{}_gap_pct", i));
        }
    }

    // ── Summary features (11) ──
    names.push("summary_slope".to_string());
    names.push("summary_range_pos".to_string());
    names.push("summary_momentum_diff".to_string());
    names.push("summary_volatility_change".to_string());
    names.push("summary_volume_pressure".to_string());
    names.push("summary_body_ratio_0".to_string());
    names.push("summary_body_ratio_1".to_string());
    names.push("summary_body_ratio_2".to_string());
    names.push("summary_return_5bar".to_string());
    names.push("summary_return_10bar".to_string());
    names.push("summary_return_full".to_string());

    // ── HTF context features (3) ──
    names.push("htf_trend".to_string());
    names.push("htf_range_pos".to_string());
    names.push("htf_vol_spike".to_string());

    debug_assert_eq!(names.len(), config.total_features(),
        "Feature name count mismatch: {} vs {}", names.len(), config.total_features());

    names
}

/// Compute CNN-like sliding window features + summary features + HTF context
/// for candle at index `t`.
///
/// The window covers candles [t - window_size + 1, ..., t].
/// All prices are normalized relative to window[0].open (the oldest candle's open).
///
/// # Arguments
/// * `candles` — full candle history (sorted by time ASC)
/// * `t` — index of the CURRENT (latest) candle in the window
/// * `config` — direction model configuration
/// * `htf_candles` — optional higher-timeframe candle data (sorted by time ASC)
///
/// # Returns
/// `None` if not enough history (t < window_size - 1).
/// `Some(Vec<f64>)` of exactly `config.total_features()` values.
pub fn compute_pattern_features(
    candles: &[CandleWithIndicators],
    t: usize,
    config: &DirectionConfig,
    htf_candles: Option<&[CandleWithIndicators]>,
) -> Option<Vec<f64>> {
    let w = config.window_size;

    // Guard: need at least W candles of history
    if t + 1 < w || t >= candles.len() {
        return None;
    }

    let window_start = t + 1 - w; // index of oldest candle in window
    let ref_open = candles[window_start].open;

    if ref_open.abs() < 1e-12 {
        return None; // degenerate: zero price
    }

    let total = config.total_features();
    let mut feats = Vec::with_capacity(total);

    // Pre-compute mean volume for normalization (avoid division by zero)
    let mean_volume = if matches!(config.feature_set, FeatureSet::OhlcVolume | FeatureSet::OhlcVolumeBody | FeatureSet::Full) {
        let vol_sum: f64 = (0..w).map(|i| candles[window_start + i].volume).sum();
        let mean = vol_sum / w as f64;
        if mean > 1e-12 { mean } else { 1.0 }
    } else {
        1.0
    };

    // ═══════════════════════════════════════════════════════════════════
    // Part 1: Sliding window features (same as v4)
    // ═══════════════════════════════════════════════════════════════════

    for i in 0..w {
        let c = &candles[window_start + i];

        // ── OHLC relative (always present) ──
        let open_rel = (c.open - ref_open) / ref_open * 100.0;
        let high_rel = (c.high - ref_open) / ref_open * 100.0;
        let low_rel = (c.low - ref_open) / ref_open * 100.0;
        let close_rel = (c.close - ref_open) / ref_open * 100.0;

        feats.push(open_rel);
        feats.push(high_rel);
        feats.push(low_rel);
        feats.push(close_rel);

        // ── Volume relative ──
        if matches!(config.feature_set, FeatureSet::OhlcVolume | FeatureSet::OhlcVolumeBody | FeatureSet::Full) {
            feats.push(c.volume / mean_volume);
        }

        // ── Body / Wick analysis ──
        if matches!(config.feature_set, FeatureSet::OhlcVolumeBody | FeatureSet::Full) {
            let range = c.high - c.low;
            let body = c.close - c.open;

            // body_pct: body / ref_open × 100 (positive = bullish, negative = bearish)
            feats.push(body / ref_open * 100.0);

            if range > 1e-12 {
                let upper_wick = c.high - c.close.max(c.open);
                feats.push(upper_wick / range);

                let lower_wick = c.close.min(c.open) - c.low;
                feats.push(lower_wick / range);
            } else {
                feats.push(0.0);
                feats.push(0.0);
            }
        }

        // ── Inter-bar dynamics ──
        if matches!(config.feature_set, FeatureSet::Full) {
            if i > 0 {
                let prev = &candles[window_start + i - 1];
                let bar_return = if prev.close.abs() > 1e-12 {
                    (c.close - prev.close) / prev.close * 100.0
                } else {
                    0.0
                };
                feats.push(bar_return);

                let gap = if prev.close.abs() > 1e-12 {
                    (c.open - prev.close) / prev.close * 100.0
                } else {
                    0.0
                };
                feats.push(gap);
            } else {
                feats.push(0.0); // bar_return
                feats.push(0.0); // gap_pct
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Part 2: Summary features (11)
    // ═══════════════════════════════════════════════════════════════════

    let window = &candles[window_start..window_start + w];
    let last_close = candles[t].close;

    // 1. SLOPE: linear regression slope of closes in window (normalized -1..+1)
    {
        let closes: Vec<f64> = window.iter().map(|c| c.close).collect();
        feats.push(linear_regression_slope(&closes));
    }

    // 2. RANGE POSITION: where current price is in the window's H-L range [0, 1]
    {
        let min_low = window.iter().map(|c| c.low).fold(f64::MAX, f64::min);
        let max_high = window.iter().map(|c| c.high).fold(f64::MIN, f64::max);
        let range = max_high - min_low;
        let range_pos = if range > 1e-12 {
            (last_close - min_low) / range
        } else {
            0.5
        };
        feats.push(range_pos);
    }

    // 3. MOMENTUM DIFF: return(last 5) − return(prev 5) → acceleration
    {
        let momentum_diff = if w >= 11 {
            let c_last = candles[t].close;
            let c_5ago = candles[t - 5].close;
            let c_10ago = candles[t - 10].close;
            if c_5ago.abs() > 1e-12 && c_10ago.abs() > 1e-12 {
                let ret_last5 = (c_last - c_5ago) / c_5ago * 100.0;
                let ret_prev5 = (c_5ago - c_10ago) / c_10ago * 100.0;
                ret_last5 - ret_prev5
            } else {
                0.0
            }
        } else {
            0.0
        };
        feats.push(momentum_diff);
    }

    // 4. VOLATILITY CHANGE: ATR(last 5) / ATR(first 5)
    {
        let volatility_change = if w >= 10 {
            let atr_recent = avg_true_range(&window[w - 5..]);
            let atr_old = avg_true_range(&window[..5]);
            if atr_old > 1e-12 {
                atr_recent / atr_old
            } else {
                1.0
            }
        } else {
            1.0
        };
        feats.push(volatility_change);
    }

    // 5. VOLUME PRESSURE: ln(vol_up / vol_down)
    {
        let (vol_up, vol_down) = window.iter().fold((0.0_f64, 0.0_f64), |(vu, vd), c| {
            if c.close >= c.open {
                (vu + c.volume, vd)
            } else {
                (vu, vd + c.volume)
            }
        });
        let vp = if vol_down > 1e-12 {
            (vol_up / vol_down).ln()
        } else {
            0.0
        };
        feats.push(vp);
    }

    // 6-8. BODY/WICK RATIO for last 3 bars (candle pattern)
    {
        let start_j = if w >= 3 { w - 3 } else { 0 };
        for j in start_j..w {
            let c = &window[j];
            let range = c.high - c.low;
            let body = (c.close - c.open).abs();
            feats.push(if range > 1e-12 { body / range } else { 0.0 });
        }
        // Pad if window < 3
        for _ in 0..(3usize.saturating_sub(w)) {
            feats.push(0.0);
        }
    }

    // 9-11. MULTI-SCALE RETURNS: return over 5, 10, full window
    {
        // 5-bar return
        let ret_5 = if w >= 6 {
            let c5 = candles[t - 5].close;
            if c5.abs() > 1e-12 { (last_close - c5) / c5 * 100.0 } else { 0.0 }
        } else {
            0.0
        };
        feats.push(ret_5);

        // 10-bar return
        let ret_10 = if w >= 11 {
            let c10 = candles[t - 10].close;
            if c10.abs() > 1e-12 { (last_close - c10) / c10 * 100.0 } else { 0.0 }
        } else {
            0.0
        };
        feats.push(ret_10);

        // Full window return
        let c0 = candles[window_start].close;
        let ret_full = if c0.abs() > 1e-12 {
            (last_close - c0) / c0 * 100.0
        } else {
            0.0
        };
        feats.push(ret_full);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Part 3: HTF (Higher Timeframe) context features (3)
    // ═══════════════════════════════════════════════════════════════════

    let target_time = candles[t].time;

    match htf_candles {
        Some(htf) if !htf.is_empty() => {
            // Find the most recent HTF candle at or before target_time
            let htf_idx = htf.partition_point(|c| c.time <= target_time);
            if htf_idx > 0 {
                let hi = htf_idx - 1; // index of current HTF candle

                // HTF_TREND: slope of closes over last 10 HTF bars (-1..+1)
                let htf_trend = if hi >= 9 {
                    let htf_closes: Vec<f64> = htf[hi - 9..=hi].iter().map(|c| c.close).collect();
                    linear_regression_slope(&htf_closes)
                } else if hi >= 1 {
                    let htf_closes: Vec<f64> = htf[..=hi].iter().map(|c| c.close).collect();
                    linear_regression_slope(&htf_closes)
                } else {
                    0.0
                };
                feats.push(htf_trend);

                // HTF_RANGE_POS: position of HTF close in its 50-bar range
                let htf_lb = 50.min(hi + 1);
                let htf_range_pos = if htf_lb >= 2 {
                    let htf_window = &htf[hi + 1 - htf_lb..=hi];
                    let min_l = htf_window.iter().map(|c| c.low).fold(f64::MAX, f64::min);
                    let max_h = htf_window.iter().map(|c| c.high).fold(f64::MIN, f64::max);
                    let r = max_h - min_l;
                    if r > 1e-12 {
                        (htf[hi].close - min_l) / r
                    } else {
                        0.5
                    }
                } else {
                    0.5
                };
                feats.push(htf_range_pos);

                // HTF_VOL_SPIKE: current HTF volume / average of last 20 bars
                let htf_vol_spike = if hi >= 1 {
                    let lb = 20.min(hi);
                    let mean_vol: f64 = htf[hi - lb..hi].iter().map(|c| c.volume).sum::<f64>()
                        / lb as f64;
                    if mean_vol > 1e-12 {
                        htf[hi].volume / mean_vol
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };
                feats.push(htf_vol_spike);
            } else {
                // No HTF candle before target time → pad with defaults
                feats.push(0.0); // htf_trend
                feats.push(0.5); // htf_range_pos
                feats.push(1.0); // htf_vol_spike
            }
        }
        _ => {
            // No HTF data available → pad with defaults
            feats.push(0.0); // htf_trend
            feats.push(0.5); // htf_range_pos
            feats.push(1.0); // htf_vol_spike
        }
    }

    debug_assert_eq!(feats.len(), total,
        "Pattern feature count mismatch: expected {}, got {} (window={}, summary={}, htf={})",
        total, feats.len(), w, SUMMARY_FEATURE_COUNT, HTF_FEATURE_COUNT);

    // ── Sanitize: replace inf/NaN with 0.0, clamp to f32-safe range ──
    // XGBoost rejects inf values; large f64 values overflow to f32::INFINITY on cast.
    // f32::MAX ≈ 3.4e38 — clamp well below that.
    const MAX_ABS: f64 = 1e30;
    for v in &mut feats {
        if !v.is_finite() {
            *v = 0.0;
        } else {
            *v = v.clamp(-MAX_ABS, MAX_ABS);
        }
    }

    Some(feats)
}

/// Compute labels for candle at index `t` with the given prediction horizon.
///
/// Returns (label, future_return_pct, max_up_pct, max_down_pct).
///
/// Label values:
///   1 = UP, 0 = FLAT, -1 = DOWN
///
/// # Arguments
/// * `candles` — full candle history
/// * `t` — current candle index
/// * `config` — direction configuration (horizon, thresholds, label_method)
pub fn compute_label(
    candles: &[CandleWithIndicators],
    t: usize,
    config: &DirectionConfig,
) -> Option<(i8, f64, f64, f64)> {
    let horizon = config.prediction_horizon;

    if t + horizon >= candles.len() {
        return None;
    }

    let entry_price = candles[t].close;
    if entry_price.abs() < 1e-12 {
        return None;
    }

    // Future return at exactly horizon bars
    let future_close = candles[t + horizon].close;
    let future_return = (future_close - entry_price) / entry_price * 100.0;

    // Max favorable excursion (up and down) within horizon
    let mut max_up: f64 = 0.0;
    let mut max_down: f64 = 0.0;

    for k in 1..=horizon {
        let idx = t + k;
        if idx >= candles.len() { break; }
        let high = candles[idx].high;
        let low = candles[idx].low;

        let up_move = (high - entry_price) / entry_price * 100.0;
        let down_move = (entry_price - low) / entry_price * 100.0;

        if up_move > max_up { max_up = up_move; }
        if down_move > max_down { max_down = down_move; }
    }

    // Determine label
    let label = match config.label_method {
        super::LabelMethod::FinalReturn => {
            if future_return > config.up_threshold_pct {
                1i8  // UP
            } else if future_return < -config.down_threshold_pct {
                -1i8 // DOWN
            } else {
                0i8  // FLAT
            }
        }
        super::LabelMethod::MaxExcursion => {
            if max_up > config.up_threshold_pct && max_up > max_down {
                1i8  // UP
            } else if max_down > config.down_threshold_pct && max_down > max_up {
                -1i8 // DOWN
            } else {
                0i8  // FLAT
            }
        }
    };

    Some((label, future_return, max_up, max_down))
}


// ═════════════════════════════════════════════════════════════════════════════
// HTF TIMEFRAME MAPPING
// ═════════════════════════════════════════════════════════════════════════════

/// Get the higher timeframe for cross-TF context.
/// Returns None if no higher TF available.
pub fn get_htf_minutes(tf_minutes: i32) -> Option<i32> {
    match tf_minutes {
        1 => Some(5),
        5 => Some(15),
        15 => Some(60),
        60 => Some(240),
        240 => Some(1440),
        _ => None,
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// BACKWARD COMPATIBILITY — Legacy v3 stubs for pipeline.rs / model.rs
// These functions are preserved so that the existing main pipeline compiles
// without changes. They are NOT used by the new v4+ pattern model.
// ═════════════════════════════════════════════════════════════════════════════

/// Legacy: v3 feature names (32 features). Kept for backward compat with pipeline.
pub const DIRECTION_V3_FEATURES: &[&str] = &[
    "supertrend_dir", "htf_supertrend_dir", "trend", "trend_short",
    "trend_alignment", "ema_convergence_change",
    "price_roc_lb1", "price_roc_lb2", "price_accel_1bar",
    "macd_hist", "macd_hist_roc_lb1", "rsi_slope_lb3",
    "dist_to_low_50", "dist_to_high_50", "bb_position",
    "price_vs_vwap", "price_vs_ema20", "high_low_pressure",
    "volume_up_vs_down_lb10", "obv_price_divergence",
    "acute_wick_rejection_2bar", "cmf",
    "btc_return_lb5", "btc_return_lb25", "btc_supertrend_dir",
    "alt_vs_btc_return_lb5", "alt_vs_btc_return_lb25",
    "bb_squeeze_pctl", "atr_ratio_lb5", "bb_width_pct",
    "supertrend_consistency", "volume_trend_ratio",
];

/// Legacy: Number of v3 features. Used by model.rs.
pub const DIRECTION_V3_FEATURE_COUNT: usize = 32;

/// Legacy: BTC context (kept for pipeline.rs backward compat).
#[derive(Debug, Clone)]
pub struct BtcContext {
    pub close: f64,
    pub supertrend_dir: f64,
    pub close_lb5: f64,
    pub close_lb25: f64,
}

impl BtcContext {
    pub fn return_lb5(&self) -> f64 {
        if self.close_lb5.abs() > 1e-12 {
            (self.close - self.close_lb5) / self.close_lb5 * 100.0
        } else { 0.0 }
    }
    pub fn return_lb25(&self) -> f64 {
        if self.close_lb25.abs() > 1e-12 {
            (self.close - self.close_lb25) / self.close_lb25 * 100.0
        } else { 0.0 }
    }
}

/// Legacy: HTF context.
#[derive(Debug, Clone)]
pub struct HtfContext {
    pub supertrend_dir: f64,
}

/// Legacy: Compute 32 v3 direction features.
/// Still functional for backward compat with pipeline.rs.
pub fn compute_direction_v3_features(
    candles: &[CandleWithIndicators],
    t: usize,
    btc_ctx: Option<&BtcContext>,
    htf_ctx: Option<&HtfContext>,
) -> Vec<f64> {
    if t >= candles.len() || t < 50 {
        return vec![0.0; DIRECTION_V3_FEATURE_COUNT];
    }

    let cur = &candles[t];
    let close = cur.close;
    let safe_div = |a: f64, b: f64| -> f64 {
        if b.abs() > 1e-12 { a / b } else { 0.0 }
    };

    let mut feats = Vec::with_capacity(DIRECTION_V3_FEATURE_COUNT);

    // GROUP 1: Trend alignment (6)
    feats.push(cur.supertrend_dir);
    feats.push(htf_ctx.map_or(0.0, |h| h.supertrend_dir));
    feats.push(cur.trend);
    feats.push(cur.trend_short);
    feats.push(cur.trend * cur.trend_short);
    let ema_conv = if t >= 5 {
        let cn = cur.ema_20 - cur.ema_50;
        let cp = candles[t-5].ema_20 - candles[t-5].ema_50;
        safe_div(cn - cp, close) * 100.0
    } else { 0.0 };
    feats.push(ema_conv);

    // GROUP 2: Momentum (6)
    let roc1 = if t >= 1 && candles[t-1].close.abs() > 1e-12 {
        (close - candles[t-1].close) / candles[t-1].close * 100.0
    } else { 0.0 };
    feats.push(roc1);
    let roc2 = if t >= 2 && candles[t-2].close.abs() > 1e-12 {
        (close - candles[t-2].close) / candles[t-2].close * 100.0
    } else { 0.0 };
    feats.push(roc2);
    let prev_roc1 = if t >= 2 && candles[t-2].close.abs() > 1e-12 {
        (candles[t-1].close - candles[t-2].close) / candles[t-2].close * 100.0
    } else { 0.0 };
    feats.push(roc1 - prev_roc1);
    feats.push(cur.macd_hist);
    let mhr = if t >= 1 { safe_div(cur.macd_hist - candles[t-1].macd_hist, close) * 1000.0 } else { 0.0 };
    feats.push(mhr);
    let rs = if t >= 3 { (cur.rsi - candles[t-3].rsi) / 100.0 } else { 0.0 };
    feats.push(rs);

    // GROUP 3: Market structure (6)
    let lb = 50.min(t+1);
    let min_low = (0..lb).map(|j| candles[t-j].low).fold(f64::MAX, f64::min);
    feats.push(safe_div(close - min_low, close) * 100.0);
    let max_high = (0..lb).map(|j| candles[t-j].high).fold(f64::MIN, f64::max);
    feats.push(safe_div(max_high - close, close) * 100.0);
    let bbr = cur.bb_upper - cur.bb_lower;
    feats.push(if bbr.abs() > 1e-12 { (close - cur.bb_lower)/bbr } else { 0.5 });
    feats.push(safe_div(close - cur.vwap, close) * 100.0);
    feats.push(safe_div(close - cur.ema_20, close) * 100.0);
    let hlp = if t >= 10 {
        let mut ps = 0.0; let mut cnt = 0;
        for j in 0..10.min(t+1) {
            let c = &candles[t-j]; let tr = c.high-c.low;
            if tr > 1e-12 {
                let uw = c.high - c.close.max(c.open);
                let lw = c.close.min(c.open) - c.low;
                ps += safe_div(lw-uw, tr); cnt += 1;
            }
        }
        if cnt > 0 { ps / cnt as f64 } else { 0.0 }
    } else { 0.0 };
    feats.push(hlp);

    // GROUP 4: Volume/Pressure (4)
    let vud = if t >= 10 {
        let (mut vu, mut vd) = (0.0f64, 0.0f64);
        for j in 0..10 { let c=&candles[t-j]; if c.close>=c.open { vu+=c.volume } else { vd+=c.volume } }
        if vd > 1e-12 { (vu/vd).clamp(0.1,10.0) } else if vu > 1e-12 { 10.0 } else { 1.0 }
    } else { 1.0 };
    feats.push(vud);
    let odlb = 10.min(t);
    let odiv = if odlb >= 3 {
        let ps = close - candles[t-odlb].close;
        let os = cur.obv - candles[t-odlb].obv;
        let psn = if ps > 0.01*close { 1.0 } else if ps < -0.01*close { -1.0 } else { 0.0 };
        let osn = if os.abs() > 1e-12 { os.signum() } else { 0.0 };
        if psn==0.0 && osn>0.0 { 1.0 } else if psn==0.0 && osn<0.0 { -1.0 }
        else if psn>0.0 && osn<0.0 { -0.5 } else if psn<0.0 && osn>0.0 { 0.5 }
        else { 0.0 }
    } else { 0.0 };
    feats.push(odiv);
    let aw = if t >= 1 {
        let mut ws = 0.0;
        for j in 0..2.min(t+1) {
            let c = &candles[t-j];
            let uw = c.high - c.close.max(c.open);
            let lw = c.close.min(c.open) - c.low;
            let a = if c.atr > 1e-12 { c.atr } else { 1.0 };
            ws += safe_div(lw-uw, a);
        }
        ws / 2.0
    } else { 0.0 };
    feats.push(aw);
    feats.push(cur.cmf);

    // GROUP 5: BTC relative (5)
    let ar5 = if t >= 5 && candles[t-5].close.abs() > 1e-12 {
        (close - candles[t-5].close) / candles[t-5].close * 100.0
    } else { 0.0 };
    let ar25 = if t >= 25 && candles[t-25].close.abs() > 1e-12 {
        (close - candles[t-25].close) / candles[t-25].close * 100.0
    } else { 0.0 };
    let (br5, br25, bsd) = match btc_ctx {
        Some(ctx) => (ctx.return_lb5(), ctx.return_lb25(), ctx.supertrend_dir),
        None => (0.0, 0.0, 0.0),
    };
    feats.push(br5);
    feats.push(br25);
    feats.push(bsd);
    feats.push(ar5 - br5);
    feats.push(ar25 - br25);

    // GROUP 6: Volatility (5)
    let bbs = if t >= 100 {
        let cw = safe_div(bbr, close) * 100.0;
        let mut cb = 0usize; let mut cv = 0usize;
        for j in 1..=100 {
            let c = &candles[t-j]; let w = c.bb_upper-c.bb_lower;
            if safe_div(w, c.close)*100.0 < cw { cb += 1; }
            cv += 1;
        }
        if cv > 0 { cb as f64 / cv as f64 } else { 0.5 }
    } else { 0.5 };
    feats.push(bbs);
    feats.push(if t >= 5 && candles[t-5].atr.abs() > 1e-12 { cur.atr/candles[t-5].atr - 1.0 } else { 0.0 });
    feats.push(safe_div(bbr, close) * 100.0);
    let stlb = 50.min(t+1);
    let stc = if stlb >= 5 { (0..stlb).map(|j| candles[t-j].supertrend_dir).sum::<f64>() / stlb as f64 } else { 0.0 };
    feats.push(stc);
    let vtr = if t >= 10 {
        let r: f64 = (0..5).map(|j| candles[t-j].volume).sum::<f64>() / 5.0;
        let p: f64 = (5..10).map(|j| candles[t-j].volume).sum::<f64>() / 5.0;
        if p > 1e-12 { (r/p).clamp(0.1, 10.0) } else { 1.0 }
    } else { 1.0 };
    feats.push(vtr);

    debug_assert_eq!(feats.len(), DIRECTION_V3_FEATURE_COUNT);
    feats
}

/// Legacy: Resolve BTC context for pipeline.rs.
pub fn resolve_btc_context(
    btc_candles: &[CandleWithIndicators],
    alt_time: chrono::DateTime<chrono::Utc>,
) -> Option<BtcContext> {
    if btc_candles.is_empty() { return None; }
    let idx = btc_candles.partition_point(|c| c.time <= alt_time);
    if idx == 0 { return None; }
    let bi = idx - 1;
    let bc = &btc_candles[bi];
    Some(BtcContext {
        close: bc.close,
        supertrend_dir: bc.supertrend_dir,
        close_lb5: if bi >= 5 { btc_candles[bi-5].close } else { bc.close },
        close_lb25: if bi >= 25 { btc_candles[bi-25].close } else { bc.close },
    })
}

/// Legacy: Resolve HTF context for pipeline.rs.
pub fn resolve_htf_context(
    htf_candles: &[CandleWithIndicators],
    target_time: chrono::DateTime<chrono::Utc>,
) -> Option<HtfContext> {
    if htf_candles.is_empty() { return None; }
    let idx = htf_candles.partition_point(|c| c.time <= target_time);
    if idx == 0 { return None; }
    Some(HtfContext { supertrend_dir: htf_candles[idx-1].supertrend_dir })
}


// ═════════════════════════════════════════════════════════════════════════════
// TESTS
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_candle(open: f64, high: f64, low: f64, close: f64, volume: f64) -> CandleWithIndicators {
        CandleWithIndicators {
            time: Utc::now(),
            symbol: "ETHUSDT".to_string(),
            symbol_id: 2,
            open, high, low, close, volume,
            rsi: 50.0, cci: 0.0, stoch_k: 50.0, stoch_d: 50.0, williams: -50.0,
            macd: 0.0, macd_signal: 0.0, macd_hist: 0.0,
            adx: 25.0, sma: close, ema_20: close, ema_50: close, ema_200: close,
            bb_upper: close * 1.02, bb_mid: close, bb_lower: close * 0.98,
            atr: close * 0.01, obv: 0.0, vwap: close, volume_spike: 1.0,
            trend: 0.0, trend_short: 0.0, poc: close,
            alligator_jaw: close, alligator_teeth: close, alligator_lips: close,
            mfi: 50.0, fibo_pivot: close, fibo_r1: close * 1.01, fibo_s1: close * 0.99,
            supertrend: close, supertrend_dir: 1.0, cmf: 0.0,
        }
    }

    #[test]
    fn test_linear_regression_slope() {
        // Perfect uptrend
        let up = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let slope = linear_regression_slope(&up);
        assert!(slope > 0.5, "Expected positive slope for uptrend, got {}", slope);

        // Perfect downtrend
        let down = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let slope = linear_regression_slope(&down);
        assert!(slope < -0.5, "Expected negative slope for downtrend, got {}", slope);

        // Flat
        let flat = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let slope = linear_regression_slope(&flat);
        assert!(slope.abs() < 1e-6, "Expected ~0 slope for flat, got {}", slope);
    }

    #[test]
    fn test_avg_true_range() {
        let candles: Vec<CandleWithIndicators> = (0..5)
            .map(|i| make_candle(100.0, 102.0, 98.0, 100.0 + i as f64, 1000.0))
            .collect();
        let atr = avg_true_range(&candles);
        assert!(atr > 0.0, "ATR should be positive");
    }

    #[test]
    fn test_pattern_features_ohlcv_with_summary_and_htf() {
        let config = DirectionConfig {
            window_size: 10,
            feature_set: FeatureSet::OhlcVolume,
            ..Default::default()
        };

        // Create 15 candles with slight uptrend
        let candles: Vec<CandleWithIndicators> = (0..15)
            .map(|i| {
                let p = 100.0 + i as f64 * 0.5;
                make_candle(p, p + 1.0, p - 0.5, p + 0.3, 1000.0 + i as f64 * 10.0)
            })
            .collect();

        // At t=9 (10 candles available), should produce features
        let feats = compute_pattern_features(&candles, 9, &config, None);
        assert!(feats.is_some());
        let feats = feats.unwrap();

        // Expected: 10*5 window + 11 summary + 3 htf = 64
        assert_eq!(feats.len(), config.total_features());
        assert_eq!(feats.len(), 10 * 5 + 11 + 3);

        // First candle (w0): open_rel should be 0.0 (reference point)
        assert!((feats[0] - 0.0).abs() < 1e-6, "w0_open_rel should be ~0.0, got {}", feats[0]);

        // Last candle should show the uptrend
        let last_close_idx = 9 * 5 + 3; // w9_close_rel
        assert!(feats[last_close_idx] > 0.0, "Last close should be above reference");

        // Summary slope should be positive for uptrend
        let slope_idx = 10 * 5; // first summary feature
        assert!(feats[slope_idx] > 0.0, "Slope should be positive for uptrend, got {}", feats[slope_idx]);

        // Range position should be near top (uptrend)
        let range_pos_idx = 10 * 5 + 1;
        assert!(feats[range_pos_idx] > 0.5, "Range pos should be >0.5 for uptrend, got {}", feats[range_pos_idx]);

        // HTF features should be defaults (no HTF data passed)
        let htf_start = feats.len() - 3;
        assert!((feats[htf_start] - 0.0).abs() < 1e-6, "htf_trend should be 0.0");
        assert!((feats[htf_start + 1] - 0.5).abs() < 1e-6, "htf_range_pos should be 0.5");
        assert!((feats[htf_start + 2] - 1.0).abs() < 1e-6, "htf_vol_spike should be 1.0");

        // All values should be finite
        for (i, &v) in feats.iter().enumerate() {
            assert!(v.is_finite(), "Feature {} is not finite: {}", i, v);
        }
    }

    #[test]
    fn test_pattern_features_with_htf() {
        use chrono::{Duration, TimeZone};

        let config = DirectionConfig {
            window_size: 10,
            feature_set: FeatureSet::OhlcVolume,
            ..Default::default()
        };

        let base_time = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();

        // Main TF candles: 15m intervals
        let candles: Vec<CandleWithIndicators> = (0..15)
            .map(|i| {
                let p = 100.0 + i as f64 * 0.5;
                let mut c = make_candle(p, p + 1.0, p - 0.5, p + 0.3, 1000.0 + i as f64 * 10.0);
                c.time = base_time + Duration::minutes(15 * i as i64);
                c
            })
            .collect();

        // HTF candles (1h): created BEFORE the main TF time range so they're findable
        let htf_candles: Vec<CandleWithIndicators> = (0..20)
            .map(|i| {
                let p = 100.0 + i as f64 * 2.0;
                let mut c = make_candle(p, p + 3.0, p - 1.0, p + 1.0, 5000.0 + i as f64 * 100.0);
                // HTF candles start well before main TF, spaced 1h apart
                c.time = base_time - Duration::hours(20 - i as i64);
                c
            })
            .collect();

        let feats = compute_pattern_features(&candles, 9, &config, Some(&htf_candles));
        assert!(feats.is_some());
        let feats = feats.unwrap();
        assert_eq!(feats.len(), config.total_features());

        // HTF trend should be positive (uptrend in HTF)
        let htf_start = feats.len() - 3;
        assert!(feats[htf_start] > 0.0, "htf_trend should be positive, got {}", feats[htf_start]);

        // HTF range_pos should be near top (uptrend)
        assert!(feats[htf_start + 1] > 0.5, "htf_range_pos should be >0.5, got {}", feats[htf_start + 1]);

        // All values should be finite
        for (i, &v) in feats.iter().enumerate() {
            assert!(v.is_finite(), "Feature {} is not finite: {}", i, v);
        }
    }

    #[test]
    fn test_pattern_features_not_enough_history() {
        let config = DirectionConfig {
            window_size: 10,
            ..Default::default()
        };

        let candles: Vec<CandleWithIndicators> = (0..8)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.5, 1000.0))
            .collect();

        // t=7, need 10 candles, only have 8 → None
        assert!(compute_pattern_features(&candles, 7, &config, None).is_none());
    }

    #[test]
    fn test_pattern_features_full_set() {
        let config = DirectionConfig {
            window_size: 5,
            feature_set: FeatureSet::Full,
            ..Default::default()
        };

        let candles: Vec<CandleWithIndicators> = (0..10)
            .map(|i| {
                let p = 100.0 + i as f64;
                make_candle(p, p + 2.0, p - 1.0, p + 0.5, 1000.0)
            })
            .collect();

        let feats = compute_pattern_features(&candles, 6, &config, None);
        assert!(feats.is_some());
        let feats = feats.unwrap();
        // 5 candles × 10 fpc + 11 summary + 3 htf = 64
        assert_eq!(feats.len(), 5 * 10 + 11 + 3);
    }

    #[test]
    fn test_label_computation() {
        let config = DirectionConfig {
            prediction_horizon: 5,
            up_threshold_pct: 0.5,
            down_threshold_pct: 0.5,
            ..Default::default()
        };

        // Create candles: flat then up
        let mut candles: Vec<CandleWithIndicators> = (0..20)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.0, 1000.0))
            .collect();

        // Make candle at t+5 go up by 2%
        candles[15].close = 102.0;
        candles[15].high = 102.5;

        let result = compute_label(&candles, 10, &config);
        assert!(result.is_some());
        let (label, future_return, max_up, _) = result.unwrap();
        assert_eq!(label, 1); // UP
        assert!((future_return - 2.0).abs() < 0.1);
        assert!(max_up >= 2.0);
    }

    #[test]
    fn test_feature_names() {
        let config = DirectionConfig {
            window_size: 3,
            feature_set: FeatureSet::OhlcVolume,
            ..Default::default()
        };

        let names = direction_v4_feature_names(&config);
        // 3*5 window + 11 summary + 3 htf = 29
        assert_eq!(names.len(), 29);
        assert_eq!(names[0], "w0_open_rel");
        assert_eq!(names[4], "w0_vol_rel");
        assert_eq!(names[5], "w1_open_rel");
        assert_eq!(names[14], "w2_vol_rel");
        // Summary features start at index 15
        assert_eq!(names[15], "summary_slope");
        assert_eq!(names[16], "summary_range_pos");
        // HTF features at the end
        assert_eq!(names[26], "htf_trend");
        assert_eq!(names[27], "htf_range_pos");
        assert_eq!(names[28], "htf_vol_spike");
    }

    #[test]
    fn test_legacy_v3_feature_count() {
        assert_eq!(DIRECTION_V3_FEATURES.len(), DIRECTION_V3_FEATURE_COUNT);
    }

    #[test]
    fn test_legacy_v3_features_still_work() {
        let candles: Vec<CandleWithIndicators> = (0..120)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.0, 1000.0))
            .collect();

        let feats = compute_direction_v3_features(&candles, 110, None, None);
        assert_eq!(feats.len(), DIRECTION_V3_FEATURE_COUNT);
        for (i, &v) in feats.iter().enumerate() {
            assert!(v.is_finite(), "Legacy feature {} is not finite", i);
        }
    }

    #[test]
    fn test_get_htf_minutes() {
        assert_eq!(get_htf_minutes(1), Some(5));
        assert_eq!(get_htf_minutes(5), Some(15));
        assert_eq!(get_htf_minutes(15), Some(60));
        assert_eq!(get_htf_minutes(60), Some(240));
        assert_eq!(get_htf_minutes(240), Some(1440));
        assert_eq!(get_htf_minutes(1440), None);
    }
}
