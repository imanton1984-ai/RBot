// strategies/ml_entry_strategy/src/direction/features.rs
//
// Direction Model v3 — Feature Extraction
//
// 32 curated, low-correlation features from 6 distinct domains:
//   - Trend alignment (6):  supertrend_dir, htf_supertrend_dir, trend, trend_short,
//                           trend_alignment, ema_convergence_change
//   - Momentum (6):         price_roc_lb1/lb2, price_accel_1bar, macd_hist,
//                           macd_hist_roc_lb1, rsi_slope_lb3
//   - Market structure (6): dist_to_low/high_50, bb_position, price_vs_vwap,
//                           price_vs_ema20, high_low_pressure
//   - Volume/Pressure (4):  volume_up_vs_down_lb10, obv_price_divergence,
//                           acute_wick_rejection_2bar, cmf
//   - BTC relative (5):     btc_return_lb5/lb25, btc_supertrend_dir,
//                           alt_vs_btc_return_lb5/lb25
//   - Volatility (5):       bb_squeeze_pctl, atr_ratio_lb5, bb_width_pct,
//                           supertrend_consistency, volume_trend_ratio
//
// KEY DIFFERENCES FROM v2 (20 features):
//   - Removed: funding_rate (placeholder=0), temporal sin/cos (noise),
//              sr_support/resistance_dist_pct (sparse)
//   - Added: 12 features from super_entry 128-feature set that had signal
//            (trend, macd_hist, bb_position, price_vs_ema20, momentum features)
//   - BTC features preserved — they were TOP-5 important on all TFs in v2 WFO
//
// KEY DIFFERENCES FROM super_entry 128-feature model:
//   - 128 → 32 features: removes 96 features that were noise/duplicates
//   - Removed: all raw indicator values (rsi, cci, stoch_k/d, williams, sma, ema_50/200...)
//   - Removed: all normalized variants (rsi_norm, cci_norm, stoch_norm...)
//   - Removed: per-window duplicate slopes (lb10/15/25/50 for most metrics)
//   - Removed: volume_roc variants, macd_hist_roc_lb3, price_accel_3bar...
//   - Added: 5 BTC relative features (completely missing from super_entry features)
//
// TRAINING: regression on direction_quality = direction * (1/bars_to_tp)
//   - Fast TP = high |quality| = clean signal
//   - Slow/no TP = low |quality| = noise → model ignores
//   - Inference: sign(prediction) = direction, abs(prediction) = confidence

use crate::dataset::CandleWithIndicators;

/// Names of the 32 direction v3 features. Must match Python trainer exactly.
pub const DIRECTION_V3_FEATURES: &[&str] = &[
    // ── GROUP 1: Trend alignment (6) ──
    "supertrend_dir",           // Current supertrend direction (-1/+1)
    "htf_supertrend_dir",       // HTF supertrend — strongest signal
    "trend",                    // Compute indicator trend
    "trend_short",              // Short-term trend
    "trend_alignment",          // trend * trend_short — multi-indicator consensus
    "ema_convergence_change",   // (ema20-ema50 now) - (ema20-ema50 5 bars ago) / close * 100

    // ── GROUP 2: Momentum (6) ──
    "price_roc_lb1",            // 1-bar return: (close[t] - close[t-1]) / close[t-1] * 100
    "price_roc_lb2",            // 2-bar return
    "price_accel_1bar",         // RoC acceleration: roc_lb1[t] - roc_lb1[t-1]
    "macd_hist",                // MACD histogram — classic momentum
    "macd_hist_roc_lb1",        // MACD hist change: (macd_hist[t] - macd_hist[t-1]) / close * 1000
    "rsi_slope_lb3",            // (rsi[t] - rsi[t-3]) / 100

    // ── GROUP 3: Market structure (6) ──
    "dist_to_low_50",           // (close - min_low_50) / close * 100
    "dist_to_high_50",          // (max_high_50 - close) / close * 100
    "bb_position",              // (close - bb_lower) / (bb_upper - bb_lower)
    "price_vs_vwap",            // (close - vwap) / close * 100
    "price_vs_ema20",           // (close - ema_20) / close * 100
    "high_low_pressure",        // Wick bias over 10 bars: buying/selling pressure

    // ── GROUP 4: Volume/Pressure (4) ──
    "volume_up_vs_down_lb10",   // sum(up_vol) / sum(down_vol) over 10 bars
    "obv_price_divergence",     // OBV slope vs price slope mismatch
    "acute_wick_rejection_2bar",// Wick rejection over 2 bars / ATR
    "cmf",                      // Chaikin Money Flow (raw indicator)

    // ── GROUP 5: BTC relative (5) ──
    "btc_return_lb5",           // BTC return over 5 bars (%)
    "btc_return_lb25",          // BTC return over 25 bars (%)
    "btc_supertrend_dir",       // BTC supertrend direction (-1/+1)
    "alt_vs_btc_return_lb5",    // alt_return_lb5 - btc_return_lb5
    "alt_vs_btc_return_lb25",   // alt_return_lb25 - btc_return_lb25

    // ── GROUP 6: Volatility context (5) ──
    "bb_squeeze_pctl",          // BB width percentile (0-1) over 100 bars
    "atr_ratio_lb5",            // atr[t] / atr[t-5] - 1 (expansion/contraction)
    "bb_width_pct",             // (bb_upper - bb_lower) / close * 100
    "supertrend_consistency",   // sum(supertrend_dir[t-49..=t]) / 50
    "volume_trend_ratio",       // mean_vol_recent_5 / mean_vol_prev_5
];

/// Number of direction v3 features (compile-time constant for assertions)
pub const DIRECTION_V3_FEATURE_COUNT: usize = 32;

/// SR levels parsed from JSONB in indicators_wide (kept for backward compat)
#[derive(Debug, Clone, Default)]
pub struct SrLevels {
    pub strong_support: f64,
    pub mid_support: f64,
    pub light_support: f64,
    pub strong_resistance: f64,
    pub mid_resistance: f64,
    pub light_resistance: f64,
}

/// BTC context for a specific candle timestamp.
/// Pre-loaded once and passed into feature computation via AS-OF lookup.
#[derive(Debug, Clone)]
pub struct BtcContext {
    pub close: f64,
    pub supertrend_dir: f64,
    /// BTC close 5 bars ago (for return calculation)
    pub close_lb5: f64,
    /// BTC close 25 bars ago (for return calculation)
    pub close_lb25: f64,
}

impl BtcContext {
    /// BTC return over last 5 bars (%).
    pub fn return_lb5(&self) -> f64 {
        if self.close_lb5.abs() > 1e-12 {
            (self.close - self.close_lb5) / self.close_lb5 * 100.0
        } else {
            0.0
        }
    }

    /// BTC return over last 25 bars (%).
    pub fn return_lb25(&self) -> f64 {
        if self.close_lb25.abs() > 1e-12 {
            (self.close - self.close_lb25) / self.close_lb25 * 100.0
        } else {
            0.0
        }
    }
}

/// HTF (Higher Timeframe) context for a specific candle.
/// Resolved via AS-OF binary search from pre-loaded HTF candles.
#[derive(Debug, Clone)]
pub struct HtfContext {
    pub supertrend_dir: f64,
}

/// Compute all 32 direction v3 features for candle at index `t`.
///
/// Computes only what's needed — no wasted work on 96 unused features.
/// All lookback-based features are computed inline from candle history.
///
/// # Arguments
/// * `candles` — full candle history for this (symbol, tf)
/// * `t` — index of current candle
/// * `btc_ctx` — BTC context at this timestamp (None = zeros for BTC features)
/// * `htf_ctx` — HTF context at this timestamp (None = 0 for htf_supertrend_dir)
///
/// # Returns
/// Vec of exactly `DIRECTION_V3_FEATURE_COUNT` values in `DIRECTION_V3_FEATURES` order.
pub fn compute_direction_v3_features(
    candles: &[CandleWithIndicators],
    t: usize,
    btc_ctx: Option<&BtcContext>,
    htf_ctx: Option<&HtfContext>,
) -> Vec<f64> {
    let mut feats = Vec::with_capacity(DIRECTION_V3_FEATURE_COUNT);

    // Guard: not enough history for lookback features
    if t >= candles.len() || t < 50 {
        return vec![0.0; DIRECTION_V3_FEATURE_COUNT];
    }

    let cur = &candles[t];
    let close = cur.close;

    let safe_div = |a: f64, b: f64| -> f64 {
        if b.abs() > 1e-12 { a / b } else { 0.0 }
    };

    // ═══════════════════════════════════════════════════════════════════════
    // GROUP 1: Trend alignment (6 features)
    // ═══════════════════════════════════════════════════════════════════════

    // supertrend_dir: current supertrend direction (-1 or +1)
    feats.push(cur.supertrend_dir);

    // htf_supertrend_dir: higher timeframe supertrend direction
    feats.push(htf_ctx.map_or(0.0, |h| h.supertrend_dir));

    // trend: compute indicator trend
    feats.push(cur.trend);

    // trend_short: short-term trend
    feats.push(cur.trend_short);

    // trend_alignment: trend * trend_short (consensus signal)
    feats.push(cur.trend * cur.trend_short);

    // ema_convergence_change: (ema20-ema50 now) - (ema20-ema50 5bars ago) / close * 100
    let ema_conv_change = if t >= 5 {
        let conv_now = cur.ema_20 - cur.ema_50;
        let conv_prev = candles[t - 5].ema_20 - candles[t - 5].ema_50;
        safe_div(conv_now - conv_prev, close) * 100.0
    } else {
        0.0
    };
    feats.push(ema_conv_change);

    // ═══════════════════════════════════════════════════════════════════════
    // GROUP 2: Momentum (6 features)
    // ═══════════════════════════════════════════════════════════════════════

    // price_roc_lb1: 1-bar return (%)
    let roc_lb1 = if t >= 1 && candles[t - 1].close.abs() > 1e-12 {
        (close - candles[t - 1].close) / candles[t - 1].close * 100.0
    } else {
        0.0
    };
    feats.push(roc_lb1);

    // price_roc_lb2: 2-bar return (%)
    let roc_lb2 = if t >= 2 && candles[t - 2].close.abs() > 1e-12 {
        (close - candles[t - 2].close) / candles[t - 2].close * 100.0
    } else {
        0.0
    };
    feats.push(roc_lb2);

    // price_accel_1bar: roc_lb1[t] - roc_lb1[t-1]
    let prev_roc_lb1 = if t >= 2 && candles[t - 2].close.abs() > 1e-12 {
        (candles[t - 1].close - candles[t - 2].close) / candles[t - 2].close * 100.0
    } else {
        0.0
    };
    feats.push(roc_lb1 - prev_roc_lb1);

    // macd_hist: raw MACD histogram
    feats.push(cur.macd_hist);

    // macd_hist_roc_lb1: (macd_hist[t] - macd_hist[t-1]) / close * 1000
    let macd_hist_roc = if t >= 1 {
        safe_div(cur.macd_hist - candles[t - 1].macd_hist, close) * 1000.0
    } else {
        0.0
    };
    feats.push(macd_hist_roc);

    // rsi_slope_lb3: (rsi[t] - rsi[t-3]) / 100
    let rsi_slope = if t >= 3 {
        (cur.rsi - candles[t - 3].rsi) / 100.0
    } else {
        0.0
    };
    feats.push(rsi_slope);

    // ═══════════════════════════════════════════════════════════════════════
    // GROUP 3: Market structure (6 features)
    // ═══════════════════════════════════════════════════════════════════════

    // dist_to_low_50: (close - min_low_50) / close * 100
    let dist_to_low_50 = {
        let lb = 50.min(t + 1);
        let min_low = (0..lb)
            .map(|j| candles[t - j].low)
            .fold(f64::MAX, f64::min);
        safe_div(close - min_low, close) * 100.0
    };
    feats.push(dist_to_low_50);

    // dist_to_high_50: (max_high_50 - close) / close * 100
    let dist_to_high_50 = {
        let lb = 50.min(t + 1);
        let max_high = (0..lb)
            .map(|j| candles[t - j].high)
            .fold(f64::MIN, f64::max);
        safe_div(max_high - close, close) * 100.0
    };
    feats.push(dist_to_high_50);

    // bb_position: (close - bb_lower) / (bb_upper - bb_lower)
    let bb_range = cur.bb_upper - cur.bb_lower;
    let bb_position = if bb_range.abs() > 1e-12 {
        (close - cur.bb_lower) / bb_range
    } else {
        0.5
    };
    feats.push(bb_position);

    // price_vs_vwap: (close - vwap) / close * 100
    feats.push(safe_div(close - cur.vwap, close) * 100.0);

    // price_vs_ema20: (close - ema_20) / close * 100
    feats.push(safe_div(close - cur.ema_20, close) * 100.0);

    // high_low_pressure: wick bias over 10 bars (buying/selling pressure)
    let high_low_pressure = if t >= 10 {
        let mut pressure_sum = 0.0;
        let mut count = 0;
        for j in 0..10.min(t + 1) {
            let c = &candles[t - j];
            let total_range = c.high - c.low;
            if total_range > 1e-12 {
                let upper_wick = c.high - c.close.max(c.open);
                let lower_wick = c.close.min(c.open) - c.low;
                // Positive = buying pressure (lower wicks dominate)
                pressure_sum += safe_div(lower_wick - upper_wick, total_range);
                count += 1;
            }
        }
        if count > 0 { pressure_sum / count as f64 } else { 0.0 }
    } else {
        0.0
    };
    feats.push(high_low_pressure);

    // ═══════════════════════════════════════════════════════════════════════
    // GROUP 4: Volume/Pressure (4 features)
    // ═══════════════════════════════════════════════════════════════════════

    // volume_up_vs_down_lb10: ratio of up-candle volume to down-candle volume
    let vol_up_down = if t >= 10 {
        let mut vol_up = 0.0f64;
        let mut vol_down = 0.0f64;
        for j in 0..10 {
            let c = &candles[t - j];
            if c.close >= c.open {
                vol_up += c.volume;
            } else {
                vol_down += c.volume;
            }
        }
        if vol_down > 1e-12 {
            (vol_up / vol_down).clamp(0.1, 10.0)
        } else if vol_up > 1e-12 {
            10.0
        } else {
            1.0
        }
    } else {
        1.0
    };
    feats.push(vol_up_down);

    // obv_price_divergence: sign mismatch between OBV slope and price slope
    let obv_div_lb = 10.min(t);
    let obv_divergence = if obv_div_lb >= 3 {
        let price_slope = close - candles[t - obv_div_lb].close;
        let obv_slope = cur.obv - candles[t - obv_div_lb].obv;
        let price_sign = if price_slope > 0.01 * close {
            1.0
        } else if price_slope < -0.01 * close {
            -1.0
        } else {
            0.0
        };
        let obv_sign = if obv_slope.abs() > 1e-12 {
            obv_slope.signum()
        } else {
            0.0
        };

        if price_sign == 0.0 && obv_sign > 0.0 {
            1.0 // Bullish divergence
        } else if price_sign == 0.0 && obv_sign < 0.0 {
            -1.0 // Bearish divergence
        } else if price_sign > 0.0 && obv_sign < 0.0 {
            -0.5 // Weak bearish
        } else if price_sign < 0.0 && obv_sign > 0.0 {
            0.5 // Weak bullish
        } else {
            0.0 // Agreement
        }
    } else {
        0.0
    };
    feats.push(obv_divergence);

    // acute_wick_rejection_2bar: wick bias over last 2 bars / ATR
    let acute_wick = if t >= 1 {
        let mut wick_sum = 0.0;
        for j in 0..2.min(t + 1) {
            let c = &candles[t - j];
            let upper_wick = c.high - c.close.max(c.open);
            let lower_wick = c.close.min(c.open) - c.low;
            let atr_safe = if c.atr > 1e-12 { c.atr } else { 1.0 };
            wick_sum += safe_div(lower_wick - upper_wick, atr_safe);
        }
        wick_sum / 2.0
    } else {
        0.0
    };
    feats.push(acute_wick);

    // cmf: Chaikin Money Flow (raw indicator value)
    feats.push(cur.cmf);

    // ═══════════════════════════════════════════════════════════════════════
    // GROUP 5: BTC relative (5 features)
    // ═══════════════════════════════════════════════════════════════════════

    let alt_return_lb5 = if t >= 5 && candles[t - 5].close.abs() > 1e-12 {
        (close - candles[t - 5].close) / candles[t - 5].close * 100.0
    } else {
        0.0
    };

    let alt_return_lb25 = if t >= 25 && candles[t - 25].close.abs() > 1e-12 {
        (close - candles[t - 25].close) / candles[t - 25].close * 100.0
    } else {
        0.0
    };

    let (btc_ret5, btc_ret25, btc_st_dir) = match btc_ctx {
        Some(ctx) => (ctx.return_lb5(), ctx.return_lb25(), ctx.supertrend_dir),
        None => (0.0, 0.0, 0.0),
    };

    feats.push(btc_ret5);                           // btc_return_lb5
    feats.push(btc_ret25);                           // btc_return_lb25
    feats.push(btc_st_dir);                          // btc_supertrend_dir
    feats.push(alt_return_lb5 - btc_ret5);           // alt_vs_btc_return_lb5
    feats.push(alt_return_lb25 - btc_ret25);         // alt_vs_btc_return_lb25

    // ═══════════════════════════════════════════════════════════════════════
    // GROUP 6: Volatility context (5 features)
    // ═══════════════════════════════════════════════════════════════════════

    // bb_squeeze_pctl: percentile of BB width within last 100 bars
    let bb_squeeze_lb = 100;
    let bb_squeeze_pctl = if t >= bb_squeeze_lb {
        let cur_bb_width_pct = safe_div(bb_range, close) * 100.0;
        let mut count_below = 0usize;
        let mut count_valid = 0usize;
        for j in 1..=bb_squeeze_lb {
            let c = &candles[t - j];
            let w = c.bb_upper - c.bb_lower;
            let w_pct = safe_div(w, c.close) * 100.0;
            if w_pct < cur_bb_width_pct {
                count_below += 1;
            }
            count_valid += 1;
        }
        if count_valid > 0 {
            count_below as f64 / count_valid as f64
        } else {
            0.5
        }
    } else {
        0.5
    };
    feats.push(bb_squeeze_pctl);

    // atr_ratio_lb5: atr[t] / atr[t-5] - 1
    let atr_ratio_lb5 = if t >= 5 && candles[t - 5].atr.abs() > 1e-12 {
        cur.atr / candles[t - 5].atr - 1.0
    } else {
        0.0
    };
    feats.push(atr_ratio_lb5);

    // bb_width_pct: (bb_upper - bb_lower) / close * 100
    feats.push(safe_div(bb_range, close) * 100.0);

    // supertrend_consistency: sum(supertrend_dir[t-49..=t]) / 50
    let st_lb = 50.min(t + 1);
    let st_consistency = if st_lb >= 5 {
        let sum: f64 = (0..st_lb)
            .map(|j| candles[t - j].supertrend_dir)
            .sum();
        sum / st_lb as f64
    } else {
        0.0
    };
    feats.push(st_consistency);

    // volume_trend_ratio: mean_vol_recent_5 / mean_vol_prev_5
    let vol_trend_ratio = if t >= 10 {
        let recent: f64 = (0..5).map(|j| candles[t - j].volume).sum::<f64>() / 5.0;
        let prev: f64 = (5..10).map(|j| candles[t - j].volume).sum::<f64>() / 5.0;
        if prev > 1e-12 {
            (recent / prev).clamp(0.1, 10.0)
        } else {
            1.0
        }
    } else {
        1.0
    };
    feats.push(vol_trend_ratio);

    debug_assert_eq!(
        feats.len(),
        DIRECTION_V3_FEATURE_COUNT,
        "Direction v3 feature count mismatch: expected {}, got {}",
        DIRECTION_V3_FEATURE_COUNT,
        feats.len()
    );

    feats
}

/// Resolve BTC context for a specific candle index using pre-loaded BTC candles.
///
/// Performs AS-OF lookup: finds the latest BTC candle at or before the alt candle time,
/// then looks back 5/25 bars in the BTC array for return calculations.
///
/// # Arguments
/// * `btc_candles` — sorted ASC by time
/// * `alt_time` — timestamp of the alt candle
pub fn resolve_btc_context(
    btc_candles: &[CandleWithIndicators],
    alt_time: chrono::DateTime<chrono::Utc>,
) -> Option<BtcContext> {
    if btc_candles.is_empty() {
        return None;
    }

    // Binary search for the latest BTC candle at or before alt_time
    let idx = btc_candles.partition_point(|c| c.time <= alt_time);
    if idx == 0 {
        return None;
    }
    let btc_idx = idx - 1;
    let btc_cur = &btc_candles[btc_idx];

    let close_lb5 = if btc_idx >= 5 {
        btc_candles[btc_idx - 5].close
    } else {
        btc_cur.close
    };

    let close_lb25 = if btc_idx >= 25 {
        btc_candles[btc_idx - 25].close
    } else {
        btc_cur.close
    };

    Some(BtcContext {
        close: btc_cur.close,
        supertrend_dir: btc_cur.supertrend_dir,
        close_lb5,
        close_lb25,
    })
}

/// Resolve HTF context for a specific candle using pre-loaded HTF candles.
///
/// # Arguments
/// * `htf_candles` — HTF candles sorted ASC by time
/// * `target_time` — timestamp of the current LTF candle
pub fn resolve_htf_context(
    htf_candles: &[CandleWithIndicators],
    target_time: chrono::DateTime<chrono::Utc>,
) -> Option<HtfContext> {
    if htf_candles.is_empty() {
        return None;
    }

    let idx = htf_candles.partition_point(|c| c.time <= target_time);
    if idx == 0 {
        return None;
    }
    let htf = &htf_candles[idx - 1];

    Some(HtfContext {
        supertrend_dir: htf.supertrend_dir,
    })
}

/// Parse SR levels from JSONB string (from sqlx).
/// Kept for backward compatibility with v2 dataset code.
pub fn parse_sr_levels(json_str: &str) -> SrLevels {
    match serde_json::from_str::<serde_json::Value>(json_str) {
        Ok(val) => SrLevels {
            strong_support: val
                .get("strong_support")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            mid_support: val
                .get("mid_support")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            light_support: val
                .get("light_support")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            strong_resistance: val
                .get("strong_resistance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            mid_resistance: val
                .get("mid_resistance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            light_resistance: val
                .get("light_resistance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
        },
        Err(_) => SrLevels::default(),
    }
}

/// Map from direction v3 feature name → index in the super_entry 128-feature vector.
///
/// Used for extracting direction features from already-computed 128-feature vectors
/// during real-time inference (avoids recomputing everything).
///
/// Returns (feature_indices_in_128, btc_features_needed)
/// where btc_features_needed = true if BTC features need separate computation.
pub fn direction_v3_feature_indices_in_128() -> Vec<Option<usize>> {
    // The 128 features are: INDICATOR(33) + DERIVED(19) + DYNAMIC(76)
    // We need to map each of our 32 features to its index in the 128 vector.
    //
    // Features that DON'T exist in the 128 set (BTC features) get None.
    use crate::config::{INDICATOR_FEATURES, DERIVED_FEATURES, DYNAMIC_FEATURES};

    let ind_offset = 0usize;
    let der_offset = INDICATOR_FEATURES.len(); // 33
    let dyn_offset = der_offset + DERIVED_FEATURES.len(); // 33 + 19 = 52

    let find_ind = |name: &str| -> Option<usize> {
        INDICATOR_FEATURES.iter().position(|&n| n == name).map(|i| ind_offset + i)
    };
    let find_der = |name: &str| -> Option<usize> {
        DERIVED_FEATURES.iter().position(|&n| n == name).map(|i| der_offset + i)
    };
    let find_dyn = |name: &str| -> Option<usize> {
        DYNAMIC_FEATURES.iter().position(|&n| n == name).map(|i| dyn_offset + i)
    };

    DIRECTION_V3_FEATURES
        .iter()
        .map(|&name| {
            find_ind(name)
                .or_else(|| find_der(name))
                .or_else(|| find_dyn(name))
            // BTC features won't be found → None
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_candle(close: f64, high: f64, low: f64) -> CandleWithIndicators {
        CandleWithIndicators {
            time: Utc::now(),
            symbol: "ETHUSDT".to_string(),
            symbol_id: 2,
            open: close,
            high,
            low,
            close,
            volume: 1000.0,
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
    fn test_feature_count() {
        assert_eq!(DIRECTION_V3_FEATURES.len(), DIRECTION_V3_FEATURE_COUNT);
    }

    #[test]
    fn test_compute_features_basic() {
        let candles: Vec<CandleWithIndicators> = (0..120)
            .map(|_| make_candle(100.0, 101.0, 99.0))
            .collect();

        let feats = compute_direction_v3_features(&candles, 110, None, None);
        assert_eq!(feats.len(), DIRECTION_V3_FEATURE_COUNT);

        for (i, &v) in feats.iter().enumerate() {
            assert!(v.is_finite(), "Feature {} ({}) is not finite: {}",
                    i, DIRECTION_V3_FEATURES[i], v);
        }
    }

    #[test]
    fn test_not_enough_history() {
        let candles: Vec<CandleWithIndicators> = (0..30)
            .map(|_| make_candle(100.0, 101.0, 99.0))
            .collect();

        // t=10 < 50 → should return zeros
        let feats = compute_direction_v3_features(&candles, 10, None, None);
        assert_eq!(feats.len(), DIRECTION_V3_FEATURE_COUNT);
        assert!(feats.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_feature_indices_mapping() {
        let indices = direction_v3_feature_indices_in_128();
        assert_eq!(indices.len(), DIRECTION_V3_FEATURE_COUNT);

        // BTC features (indices 22-26) should be None
        assert!(indices[22].is_none(), "btc_return_lb5 should not be in 128");
        assert!(indices[23].is_none(), "btc_return_lb25 should not be in 128");
        assert!(indices[24].is_none(), "btc_supertrend_dir should not be in 128");
        assert!(indices[25].is_none(), "alt_vs_btc_return_lb5 should not be in 128");
        assert!(indices[26].is_none(), "alt_vs_btc_return_lb25 should not be in 128");

        // supertrend_dir should be in INDICATOR_FEATURES
        assert!(indices[0].is_some(), "supertrend_dir should be in 128");

        // trend should be in INDICATOR_FEATURES
        assert!(indices[2].is_some(), "trend should be in 128");

        // bb_position should be in DERIVED_FEATURES
        assert!(indices[14].is_some(), "bb_position should be in 128");
    }

    #[test]
    fn test_parse_sr_levels() {
        let json = r#"{"mid_support": 70821.2, "light_support": 70899.9, "mid_resistance": 71536.0, "strong_support": 70456.0, "light_resistance": 71980.5, "strong_resistance": 71438.1}"#;
        let sr = parse_sr_levels(json);
        assert!((sr.strong_support - 70456.0).abs() < 1e-6);
        assert!((sr.strong_resistance - 71438.1).abs() < 1e-6);
    }
}
