// strategies/ml_entry_strategy/src/dataset.rs
//
// Dataset Builder for Super Entry Model
//
// For each (pair, timeframe):
//   - First 300 candles are used as warmup for indicator computation
//   - From candle t=301 to t=980, create training examples:
//     * Features: indicator values at candle t + dynamic temporal features
//     * Labels:
//       - max_up_move_pct, max_down_move_pct (over next 20 candles)
//       - direction (LONG if up >= down, else SHORT)
//       - magnitude_pct (max of up/down)
//       - is_super (magnitude >= TF_TARGET_MOVE_PCT)
//
// The dataset is saved as CSV for Python trainer consumption.
//
// DYNAMIC FEATURES (v2):
//   For direction prediction, static indicator snapshots are insufficient
//   (AUC ~0.50 = random). We add temporal/lookback features that capture
//   HOW indicators are changing over time:
//   - price_return, atr_ratio, rsi_slope, trend persistence, etc.
//   - Lookback windows: 3, 5, 10, 15 bars
//   - Plus aggregate features: supertrend consistency, trend alignment, etc.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::io::Write;

use crate::config::{INDICATOR_FEATURES, DYNAMIC_LOOKBACK_WINDOWS};

/// A single candle row with indicators from the database
#[derive(Debug, Clone)]
pub struct CandleWithIndicators {
    pub time: DateTime<Utc>,
    pub symbol: String,
    pub symbol_id: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    // Indicators (match INDICATOR_FEATURES order)
    pub rsi: f64,
    pub cci: f64,
    pub stoch_k: f64,
    pub stoch_d: f64,
    pub williams: f64,
    pub macd: f64,
    pub macd_signal: f64,
    pub macd_hist: f64,
    pub adx: f64,
    pub sma: f64,
    pub ema_20: f64,
    pub ema_50: f64,
    pub ema_200: f64,
    pub bb_upper: f64,
    pub bb_mid: f64,
    pub bb_lower: f64,
    pub atr: f64,
    pub obv: f64,
    pub vwap: f64,
    pub volume_spike: f64,
    pub trend: f64,
    pub trend_short: f64,
    pub poc: f64,
    // Alligator
    pub alligator_jaw: f64,
    pub alligator_teeth: f64,
    pub alligator_lips: f64,
    // New indicators (v2)
    pub mfi: f64,
    pub fibo_pivot: f64,
    pub fibo_r1: f64,
    pub fibo_s1: f64,
    pub supertrend: f64,
    pub supertrend_dir: f64,
    pub cmf: f64,
}

impl CandleWithIndicators {
    /// Extract raw indicator values in the order of INDICATOR_FEATURES
    pub fn indicator_values(&self) -> Vec<f64> {
        vec![
            self.rsi, self.cci, self.stoch_k, self.stoch_d, self.williams,
            self.macd, self.macd_signal, self.macd_hist,
            self.adx, self.sma, self.ema_20, self.ema_50, self.ema_200,
            self.bb_upper, self.bb_mid, self.bb_lower, self.atr,
            self.obv, self.vwap, self.volume_spike,
            self.trend, self.trend_short, self.poc,
            self.alligator_jaw, self.alligator_teeth, self.alligator_lips,
            self.mfi, self.fibo_pivot, self.fibo_r1, self.fibo_s1,
            self.supertrend, self.supertrend_dir, self.cmf,
        ]
    }

    /// Compute derived features from raw indicators
    pub fn derived_features(&self) -> Vec<f64> {
        let close = self.close;
        let safe_div = |a: f64, b: f64| if b.abs() > 1e-12 { a / b } else { 0.0 };

        let rsi_norm = self.rsi / 100.0;
        let cci_norm = self.cci / 200.0;
        let stoch_norm = self.stoch_k / 100.0;
        let williams_norm = (self.williams + 100.0) / 100.0;

        let bb_range = self.bb_upper - self.bb_lower;
        let bb_position = if bb_range.abs() > 1e-12 {
            (close - self.bb_lower) / bb_range
        } else {
            0.5
        };
        let bb_width_pct = safe_div(bb_range, close) * 100.0;

        let atr_pct = safe_div(self.atr, close) * 100.0;
        let price_vs_sma = safe_div(close - self.sma, close) * 100.0;
        let price_vs_ema20 = safe_div(close - self.ema_20, close) * 100.0;
        let price_vs_ema50 = safe_div(close - self.ema_50, close) * 100.0;
        let price_vs_ema200 = safe_div(close - self.ema_200, close) * 100.0;
        let price_vs_vwap = safe_div(close - self.vwap, close) * 100.0;
        let macd_norm = safe_div(self.macd_hist, close) * 1000.0;
        let obv_change_pct = 0.0; // Requires history; not available in single-row context
        let volume_spike_flag = if self.volume_spike > 2.0 { 1.0 } else { 0.0 };

        // New derived features
        let mfi_norm = self.mfi / 100.0;
        let price_vs_fibo_pivot = safe_div(close - self.fibo_pivot, close) * 100.0;
        let price_vs_supertrend = safe_div(close - self.supertrend, close) * 100.0;
        // Alligator spread: jaw-lips difference normalized by price
        let alligator_spread = safe_div(self.alligator_jaw - self.alligator_lips, close) * 100.0;

        vec![
            rsi_norm, cci_norm, stoch_norm, williams_norm,
            bb_position, bb_width_pct, atr_pct,
            price_vs_sma, price_vs_ema20, price_vs_ema50,
            price_vs_ema200, price_vs_vwap,
            macd_norm, obv_change_pct, volume_spike_flag,
            mfi_norm, price_vs_fibo_pivot, price_vs_supertrend,
            alligator_spread,
        ]
    }

    /// Get full feature vector (indicators + derived)
    pub fn full_features(&self) -> Vec<f64> {
        let mut feats = self.indicator_values();
        feats.extend(self.derived_features());
        feats
    }
}

/// A labeled training example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperEntryExample {
    pub symbol: String,
    pub tf_minutes: i32,
    pub timestamp: String, // ISO8601
    pub features: Vec<f64>,
    pub max_up_move_pct: f64,
    pub max_down_move_pct: f64,
    pub direction: i8,       // 1 = LONG, -1 = SHORT
    pub magnitude_pct: f64,
    pub is_super: bool,
    pub future_return_20: f64, // close[t+20] / close[t] - 1 (in %)
}

/// Compute dynamic/temporal features for candle at index `t` using lookback
/// over the candle history. These features capture HOW indicators are changing,
/// which is critical for direction prediction.
///
/// Returns a Vec<f64> of exactly `crate::config::dynamic_feature_count()` elements.
///
/// The lookback windows are defined in `DYNAMIC_LOOKBACK_WINDOWS` (3, 5, 10, 15).
/// If `t` < max_lookback, returns zeros (safe default for warmup candles).
///
/// # Feature groups (per window N):
/// 1. price_return — directional price momentum
/// 2. atr_ratio — volatility expansion/contraction
/// 3. rsi_slope — oscillator momentum
/// 4. trend_persist — trend direction consistency
/// 5. trend_short_persist — short-term trend consistency
/// 6. adx_slope — trend strength change
/// 7. macd_hist_slope — MACD momentum
/// 8. ema20_direction — moving average slope
///
/// # Aggregate features:
/// 9. supertrend_consistency — directional conviction over 15 bars
/// 10. trend_alignment — agreement between long/short trends
/// 11. price_accel — momentum acceleration (2nd derivative)
/// 12. volume_trend_ratio — recent vs older volume activity
/// 13. ema_convergence_change — EMA20/50 convergence/divergence shift
/// 14. high_low_pressure — wick bias (buying vs selling pressure)
pub fn compute_dynamic_features(candles: &[CandleWithIndicators], t: usize) -> Vec<f64> {
    let n_dynamic = crate::config::dynamic_feature_count();
    let max_lb = crate::config::max_dynamic_lookback();

    // Not enough history — return zeros
    if t < max_lb || t >= candles.len() {
        return vec![0.0; n_dynamic];
    }

    let cur = &candles[t];
    let close = cur.close;

    // Safe division helper
    let safe_div = |a: f64, b: f64| -> f64 {
        if b.abs() > 1e-12 { a / b } else { 0.0 }
    };

    let mut feats = Vec::with_capacity(n_dynamic);

    // --- Per-window features (8 per window × 4 windows = 32) ---
    for &lb in DYNAMIC_LOOKBACK_WINDOWS {
        let prev = &candles[t - lb];

        // 1. Price return (%)
        feats.push(safe_div(close - prev.close, close) * 100.0);

        // 2. ATR ratio (expansion/contraction)
        feats.push(safe_div(cur.atr, prev.atr) - 1.0);

        // 3. RSI slope (normalized)
        feats.push((cur.rsi - prev.rsi) / 100.0);

        // 4. Trend persistence: average of trend values over last lb bars
        let trend_sum: f64 = (0..lb)
            .map(|j| candles[t - j].trend)
            .sum();
        feats.push(trend_sum / lb as f64);

        // 5. Trend short persistence: average of trend_short over last lb bars
        let trend_short_sum: f64 = (0..lb)
            .map(|j| candles[t - j].trend_short)
            .sum();
        feats.push(trend_short_sum / lb as f64);

        // 6. ADX slope (normalized)
        feats.push((cur.adx - prev.adx) / 100.0);

        // 7. MACD histogram slope (normalized by price)
        feats.push(safe_div(cur.macd_hist - prev.macd_hist, close) * 1000.0);

        // 8. EMA20 direction (slope normalized by price)
        feats.push(safe_div(cur.ema_20 - prev.ema_20, close) * 100.0);
    }

    // --- Aggregate features (6) ---

    // 9. Supertrend consistency over 15 bars: how often supertrend_dir agrees
    let st_sum: f64 = (0..15.min(t + 1))
        .map(|j| candles[t - j].supertrend_dir)
        .sum();
    feats.push(st_sum / 15.0);

    // 10. Trend alignment: do long-term and short-term trends agree?
    feats.push(cur.trend * cur.trend_short);

    // 11. Price acceleration: momentum[0..5] vs momentum[5..10]
    //     = (ret_recent_5) - (ret_prev_5), normalized
    let ret_recent_5 = if t >= 5 {
        safe_div(close - candles[t - 5].close, close) * 100.0
    } else {
        0.0
    };
    let ret_prev_5 = if t >= 10 {
        safe_div(candles[t - 5].close - candles[t - 10].close, candles[t - 5].close) * 100.0
    } else {
        0.0
    };
    feats.push(ret_recent_5 - ret_prev_5);

    // 12. Volume trend ratio: average vol of recent 5 vs previous 5 bars
    let vol_recent: f64 = (0..5.min(t + 1))
        .map(|j| candles[t - j].volume)
        .sum::<f64>()
        / 5.0f64.min((t + 1) as f64);
    let vol_prev: f64 = if t >= 5 {
        (5..10.min(t + 1))
            .map(|j| candles[t - j].volume)
            .sum::<f64>()
            / 5.0f64.min((t - 4) as f64)
    } else {
        vol_recent
    };
    feats.push(safe_div(vol_recent, vol_prev));

    // 13. EMA convergence change: (ema20-ema50) change over 5 bars
    let ema_gap_now = cur.ema_20 - cur.ema_50;
    let ema_gap_prev = if t >= 5 {
        candles[t - 5].ema_20 - candles[t - 5].ema_50
    } else {
        ema_gap_now
    };
    feats.push(safe_div(ema_gap_now - ema_gap_prev, close) * 100.0);

    // 14. High-low pressure: are wicks biased up or down over last 10 bars?
    //     Positive = buying pressure (lower wicks larger), Negative = selling pressure
    let mut pressure_sum = 0.0;
    let pressure_window = 10.min(t + 1);
    for j in 0..pressure_window {
        let c = &candles[t - j];
        let upper_wick = c.high - c.close.max(c.open);
        let lower_wick = c.close.min(c.open) - c.low;
        let atr_safe = if c.atr > 1e-12 { c.atr } else { 1.0 };
        // Positive = lower wick bigger = buying support
        pressure_sum += safe_div(lower_wick - upper_wick, atr_safe);
    }
    feats.push(pressure_sum / pressure_window as f64);

    debug_assert_eq!(feats.len(), n_dynamic,
        "Dynamic feature count mismatch: expected {}, got {}", n_dynamic, feats.len());
    feats
}

/// Build labels for a time series of candles starting from `start_idx`
/// with `lookahead` bars look-forward.
///
/// This function simulates the exact trade path candle-by-candle to determine
/// if TP or SL is hit FIRST (First-Touch logic). If SL is hit before TP, the
/// trade is a loss even if price eventually reaches TP.
///
/// # Arguments
/// * `candles` - Price data with indicators
/// * `start_idx` - Starting index for labeling (after warmup)
/// * `lookahead` - Number of bars to look ahead
/// * `target_move_pct` - Target TP move percentage
/// * `sl_fraction` - SL as a fraction of target_move_pct (e.g., 0.5 = SL at 50% of TP)
/// * `tf_minutes` - Timeframe in minutes
///
/// Returns labeled examples for candles [start_idx .. end_idx].
pub fn build_labels(
    candles: &[CandleWithIndicators],
    start_idx: usize,
    lookahead: usize,
    target_move_pct: f64,
    sl_fraction: f64,
    tf_minutes: i32,
) -> Vec<SuperEntryExample> {
    let n = candles.len();
    if n < start_idx + lookahead + 1 {
        return Vec::new();
    }

    let end_idx = n - lookahead; // last valid index for labeling
    let mut examples = Vec::with_capacity(end_idx - start_idx);

    for t in start_idx..end_idx {
        let entry_price = candles[t].close;
        if entry_price <= 0.0 {
            continue;
        }

        // Calculate TP and SL levels for LONG and SHORT
        let tp_long = entry_price * (1.0 + target_move_pct / 100.0);
        let sl_long = entry_price * (1.0 - (target_move_pct * sl_fraction) / 100.0);
        
        let tp_short = entry_price * (1.0 - target_move_pct / 100.0);
        let sl_short = entry_price * (1.0 + (target_move_pct * sl_fraction) / 100.0);

        // Track outcomes
        let mut long_win = false;
        let mut short_win = false;
        
        // Track absolute max moves for metadata and fallback
        let mut max_up: f64 = 0.0;
        let mut max_down: f64 = 0.0;

        // Track active state for each direction
        let mut long_active = true;
        let mut short_active = true;

        // Simulate trade path candle-by-candle
        for k in 1..=lookahead {
            let idx = t + k;
            if idx >= n { break; }
            
            let high = candles[idx].high;
            let low = candles[idx].low;

            // Update max absolute moves (for metadata)
            let up_move = (high - entry_price) / entry_price * 100.0;
            let down_move = (entry_price - low) / entry_price * 100.0;
            if up_move > max_up { max_up = up_move; }
            if down_move > max_down { max_down = down_move; }

            // LONG PATH SIMULATION
            // Conservative rule: If low hits SL and high hits TP in same bar, assume SL hit first
            if long_active {
                if low <= sl_long {
                    long_active = false; // Stopped out - SL hit first
                } else if high >= tp_long {
                    long_win = true; // TP hit before SL
                    long_active = false;
                }
            }

            // SHORT PATH SIMULATION
            // Conservative rule: If high hits SL and low hits TP in same bar, assume SL hit first
            if short_active {
                if high >= sl_short {
                    short_active = false; // Stopped out - SL hit first
                } else if low <= tp_short {
                    short_win = true; // TP hit before SL
                    short_active = false;
                }
            }

            // Early exit if both outcomes are resolved
            if !long_active && !short_active { break; }
        }

        // Determine direction and if it's a "super" entry
        let (is_super, direction, magnitude) = if long_win && !short_win {
            // Only LONG won - clear super signal
            (true, 1i8, target_move_pct)
        } else if short_win && !long_win {
            // Only SHORT won - clear super signal
            (true, -1i8, target_move_pct)
        } else if long_win && short_win {
            // Both would have won - take the one with biggest absolute move
            if max_up >= max_down {
                (true, 1i8, max_up)
            } else {
                (true, -1i8, max_down)
            }
        } else {
            // Neither won (choppy or immediate stop loss)
            // Provide fallback direction for the 'direction' model to learn from
            if max_up >= max_down {
                (false, 1i8, max_up)
            } else {
                (false, -1i8, max_down)
            }
        };

        // Future return at exactly lookahead bars
        let future_close_idx = (t + lookahead).min(n - 1);
        let future_return_20 =
            (candles[future_close_idx].close - entry_price) / entry_price * 100.0;

        // Static features (indicators + derived) + dynamic temporal features
        let mut features = candles[t].full_features();
        features.extend(compute_dynamic_features(candles, t));

        examples.push(SuperEntryExample {
            symbol: candles[t].symbol.clone(),
            tf_minutes,
            timestamp: candles[t].time.to_rfc3339(),
            features,
            max_up_move_pct: max_up,
            max_down_move_pct: max_down,
            direction,
            magnitude_pct: magnitude,
            is_super,
            future_return_20,
        });
    }

    examples
}

/// Fetch candles with indicators for a given symbol and timeframe from DB.
/// Returns up to `limit` rows ordered by time ASC.
pub async fn fetch_candles_with_indicators(
    pool: &PgPool,
    symbol: &str,
    tf_minutes: i32,
    limit: usize,
) -> Result<Vec<CandleWithIndicators>> {
    let candle_table = match tf_minutes {
        1 => "market.candles_1m",
        5 => "market.candles_5m",
        15 => "market.candles_15m",
        60 => "market.candles_1h",
        240 => "market.candles_4h",
        1440 => "market.candles_1d",
        _ => anyhow::bail!("Unsupported timeframe: {}", tf_minutes),
    };

    // NOTE: Indicators in market.indicators_wide are FLOAT4 (f32).
    // We cast them to FLOAT8 (f64) in SQL to match Rust types.
    // Candle OHLCV fields are already FLOAT8 in the candle tables.
    let sql = format!(
        r#"
        SELECT
            c.time, c.symbol, p.symbol_id,
            c.open, c.high, c.low, c.close, c.volume,
            COALESCE(i.rsi, 50.0)::FLOAT8 as rsi,
            COALESCE(i.cci, 0.0)::FLOAT8 as cci,
            COALESCE(i.stoch_k, 50.0)::FLOAT8 as stoch_k,
            COALESCE(i.stoch_d, 50.0)::FLOAT8 as stoch_d,
            COALESCE(i.williams, -50.0)::FLOAT8 as williams,
            COALESCE(i.macd, 0.0)::FLOAT8 as macd,
            COALESCE(i.macd_signal, 0.0)::FLOAT8 as macd_signal,
            COALESCE(i.macd_hist, 0.0)::FLOAT8 as macd_hist,
            COALESCE(i.adx, 25.0)::FLOAT8 as adx,
            COALESCE(i.sma, c.close)::FLOAT8 as sma,
            COALESCE(i.ema_20, c.close)::FLOAT8 as ema_20,
            COALESCE(i.ema_50, c.close)::FLOAT8 as ema_50,
            COALESCE(i.ema_200, c.close)::FLOAT8 as ema_200,
            COALESCE(i.bb_upper, c.close)::FLOAT8 as bb_upper,
            COALESCE(i.bb_mid, c.close)::FLOAT8 as bb_mid,
            COALESCE(i.bb_lower, c.close)::FLOAT8 as bb_lower,
            COALESCE(i.atr, 0.001)::FLOAT8 as atr,
            COALESCE(i.obv, 0.0)::FLOAT8 as obv,
            COALESCE(i.vwap, c.close)::FLOAT8 as vwap,
            COALESCE(i.volume_spike, 1.0)::FLOAT8 as volume_spike,
            COALESCE(i.trend, 0.0)::FLOAT8 as trend,
            COALESCE(i.trend_short, 0.0)::FLOAT8 as trend_short,
            COALESCE(i.poc, c.close)::FLOAT8 as poc,
            COALESCE(i.alligator_jaw, c.close)::FLOAT8 as alligator_jaw,
            COALESCE(i.alligator_teeth, c.close)::FLOAT8 as alligator_teeth,
            COALESCE(i.alligator_lips, c.close)::FLOAT8 as alligator_lips,
            COALESCE(i.mfi, 50.0)::FLOAT8 as mfi,
            COALESCE(i.fibo_pivot, c.close)::FLOAT8 as fibo_pivot,
            COALESCE(i.fibo_r1, c.close)::FLOAT8 as fibo_r1,
            COALESCE(i.fibo_s1, c.close)::FLOAT8 as fibo_s1,
            COALESCE(i.supertrend, c.close)::FLOAT8 as supertrend,
            COALESCE(i.supertrend_dir, 0.0)::FLOAT8 as supertrend_dir,
            COALESCE(i.cmf, 0.0)::FLOAT8 as cmf
        FROM {candle_table} c
        JOIN market.pairs p ON p.symbol = c.symbol
        LEFT JOIN market.indicators_wide i
            ON i.symbol_id = p.symbol_id AND i.time = c.time AND i.tf_minutes = $2
        WHERE c.symbol = $1
        ORDER BY c.time ASC
        LIMIT $3
        "#
    );

    let rows = sqlx::query_as::<_, CandleRow>(&sql)
        .bind(symbol)
        .bind(tf_minutes as i16)
        .bind(limit as i64)
        .fetch_all(pool)
        .await?;

    let candles = rows
        .into_iter()
        .map(|r| CandleWithIndicators {
            time: r.time,
            symbol: r.symbol,
            symbol_id: r.symbol_id,
            open: r.open,
            high: r.high,
            low: r.low,
            close: r.close,
            volume: r.volume,
            rsi: r.rsi,
            cci: r.cci,
            stoch_k: r.stoch_k,
            stoch_d: r.stoch_d,
            williams: r.williams,
            macd: r.macd,
            macd_signal: r.macd_signal,
            macd_hist: r.macd_hist,
            adx: r.adx,
            sma: r.sma,
            ema_20: r.ema_20,
            ema_50: r.ema_50,
            ema_200: r.ema_200,
            bb_upper: r.bb_upper,
            bb_mid: r.bb_mid,
            bb_lower: r.bb_lower,
            atr: r.atr,
            obv: r.obv,
            vwap: r.vwap,
            volume_spike: r.volume_spike,
            trend: r.trend,
            trend_short: r.trend_short,
            poc: r.poc,
            alligator_jaw: r.alligator_jaw,
            alligator_teeth: r.alligator_teeth,
            alligator_lips: r.alligator_lips,
            mfi: r.mfi,
            fibo_pivot: r.fibo_pivot,
            fibo_r1: r.fibo_r1,
            fibo_s1: r.fibo_s1,
            supertrend: r.supertrend,
            supertrend_dir: r.supertrend_dir,
            cmf: r.cmf,
        })
        .collect();

    Ok(candles)
}

/// sqlx compatible row struct
#[derive(sqlx::FromRow)]
struct CandleRow {
    time: DateTime<Utc>,
    symbol: String,
    symbol_id: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    rsi: f64,
    cci: f64,
    stoch_k: f64,
    stoch_d: f64,
    williams: f64,
    macd: f64,
    macd_signal: f64,
    macd_hist: f64,
    adx: f64,
    sma: f64,
    ema_20: f64,
    ema_50: f64,
    ema_200: f64,
    bb_upper: f64,
    bb_mid: f64,
    bb_lower: f64,
    atr: f64,
    obv: f64,
    vwap: f64,
    volume_spike: f64,
    trend: f64,
    trend_short: f64,
    poc: f64,
    alligator_jaw: f64,
    alligator_teeth: f64,
    alligator_lips: f64,
    mfi: f64,
    fibo_pivot: f64,
    fibo_r1: f64,
    fibo_s1: f64,
    supertrend: f64,
    supertrend_dir: f64,
    cmf: f64,
}

/// Bulk-fetch ALL candles with indicators for a given TF (all symbols at once).
/// Returns data grouped by symbol. Much faster than per-symbol queries.
pub async fn fetch_all_candles_for_tf(
    pool: &PgPool,
    tf_minutes: i32,
    limit_per_symbol: usize,
) -> Result<std::collections::HashMap<String, Vec<CandleWithIndicators>>> {
    let candle_table = match tf_minutes {
        1 => "market.candles_1m",
        5 => "market.candles_5m",
        15 => "market.candles_15m",
        60 => "market.candles_1h",
        240 => "market.candles_4h",
        1440 => "market.candles_1d",
        _ => anyhow::bail!("Unsupported timeframe: {}", tf_minutes),
    };

    // Single query: get last N candles per symbol with indicators via window function
    let sql = format!(
        r#"
        WITH ranked AS (
            SELECT
                c.time, c.symbol, p.symbol_id,
                c.open, c.high, c.low, c.close, c.volume,
                COALESCE(i.rsi, 50.0)::FLOAT8 as rsi,
                COALESCE(i.cci, 0.0)::FLOAT8 as cci,
                COALESCE(i.stoch_k, 50.0)::FLOAT8 as stoch_k,
                COALESCE(i.stoch_d, 50.0)::FLOAT8 as stoch_d,
                COALESCE(i.williams, -50.0)::FLOAT8 as williams,
                COALESCE(i.macd, 0.0)::FLOAT8 as macd,
                COALESCE(i.macd_signal, 0.0)::FLOAT8 as macd_signal,
                COALESCE(i.macd_hist, 0.0)::FLOAT8 as macd_hist,
                COALESCE(i.adx, 25.0)::FLOAT8 as adx,
                COALESCE(i.sma, c.close)::FLOAT8 as sma,
                COALESCE(i.ema_20, c.close)::FLOAT8 as ema_20,
                COALESCE(i.ema_50, c.close)::FLOAT8 as ema_50,
                COALESCE(i.ema_200, c.close)::FLOAT8 as ema_200,
                COALESCE(i.bb_upper, c.close)::FLOAT8 as bb_upper,
                COALESCE(i.bb_mid, c.close)::FLOAT8 as bb_mid,
                COALESCE(i.bb_lower, c.close)::FLOAT8 as bb_lower,
                COALESCE(i.atr, 0.001)::FLOAT8 as atr,
                COALESCE(i.obv, 0.0)::FLOAT8 as obv,
                COALESCE(i.vwap, c.close)::FLOAT8 as vwap,
                COALESCE(i.volume_spike, 1.0)::FLOAT8 as volume_spike,
                COALESCE(i.trend, 0.0)::FLOAT8 as trend,
                COALESCE(i.trend_short, 0.0)::FLOAT8 as trend_short,
                COALESCE(i.poc, c.close)::FLOAT8 as poc,
                COALESCE(i.alligator_jaw, c.close)::FLOAT8 as alligator_jaw,
                COALESCE(i.alligator_teeth, c.close)::FLOAT8 as alligator_teeth,
                COALESCE(i.alligator_lips, c.close)::FLOAT8 as alligator_lips,
                COALESCE(i.mfi, 50.0)::FLOAT8 as mfi,
                COALESCE(i.fibo_pivot, c.close)::FLOAT8 as fibo_pivot,
                COALESCE(i.fibo_r1, c.close)::FLOAT8 as fibo_r1,
                COALESCE(i.fibo_s1, c.close)::FLOAT8 as fibo_s1,
                COALESCE(i.supertrend, c.close)::FLOAT8 as supertrend,
                COALESCE(i.supertrend_dir, 0.0)::FLOAT8 as supertrend_dir,
                COALESCE(i.cmf, 0.0)::FLOAT8 as cmf,
                ROW_NUMBER() OVER (PARTITION BY c.symbol ORDER BY c.time DESC) as rn
            FROM {candle_table} c
            JOIN market.pairs p ON p.symbol = c.symbol AND p.is_active = true
            LEFT JOIN market.indicators_wide i
                ON i.symbol_id = p.symbol_id AND i.time = c.time AND i.tf_minutes = $1
        )
        SELECT time, symbol, symbol_id, open, high, low, close, volume,
               rsi, cci, stoch_k, stoch_d, williams, macd, macd_signal, macd_hist,
               adx, sma, ema_20, ema_50, ema_200, bb_upper, bb_mid, bb_lower, atr,
               obv, vwap, volume_spike, trend, trend_short, poc,
               alligator_jaw, alligator_teeth, alligator_lips,
               mfi, fibo_pivot, fibo_r1, fibo_s1, supertrend, supertrend_dir, cmf
        FROM ranked
        WHERE rn <= $2
        ORDER BY symbol, time ASC
        "#
    );

    let rows = sqlx::query_as::<_, CandleRow>(&sql)
        .bind(tf_minutes as i16)
        .bind(limit_per_symbol as i64)
        .fetch_all(pool)
        .await?;

    // Group by symbol
    let mut grouped: std::collections::HashMap<String, Vec<CandleWithIndicators>> =
        std::collections::HashMap::new();

    for r in rows {
        let candle = CandleWithIndicators {
            time: r.time, symbol: r.symbol.clone(), symbol_id: r.symbol_id,
            open: r.open, high: r.high, low: r.low, close: r.close, volume: r.volume,
            rsi: r.rsi, cci: r.cci, stoch_k: r.stoch_k, stoch_d: r.stoch_d,
            williams: r.williams, macd: r.macd, macd_signal: r.macd_signal,
            macd_hist: r.macd_hist, adx: r.adx, sma: r.sma,
            ema_20: r.ema_20, ema_50: r.ema_50, ema_200: r.ema_200,
            bb_upper: r.bb_upper, bb_mid: r.bb_mid, bb_lower: r.bb_lower,
            atr: r.atr, obv: r.obv, vwap: r.vwap, volume_spike: r.volume_spike,
            trend: r.trend, trend_short: r.trend_short, poc: r.poc,
            alligator_jaw: r.alligator_jaw, alligator_teeth: r.alligator_teeth,
            alligator_lips: r.alligator_lips, mfi: r.mfi,
            fibo_pivot: r.fibo_pivot, fibo_r1: r.fibo_r1, fibo_s1: r.fibo_s1,
            supertrend: r.supertrend, supertrend_dir: r.supertrend_dir, cmf: r.cmf,
        };
        grouped.entry(r.symbol).or_default().push(candle);
    }

    Ok(grouped)
}

/// Fetch list of active symbols from market.pairs
pub async fn fetch_active_symbols(pool: &PgPool) -> Result<Vec<String>> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT symbol FROM market.pairs WHERE is_active = true ORDER BY symbol"
    )
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(|(s,)| s).collect())
}

/// Export dataset to CSV file
pub fn export_dataset_csv(
    examples: &[SuperEntryExample],
    output_path: &str,
    feature_names: &[&str],
) -> Result<()> {
    let mut file = std::fs::File::create(output_path)?;

    // Header
    let mut header = String::from("symbol,tf_minutes,timestamp");
    for name in feature_names {
        header.push(',');
        header.push_str(name);
    }
    header.push_str(",max_up_move_pct,max_down_move_pct,direction,magnitude_pct,is_super,future_return_20");
    writeln!(file, "{}", header)?;

    // Rows
    for ex in examples {
        let mut line = format!("{},{},{}", ex.symbol, ex.tf_minutes, ex.timestamp);
        for &val in &ex.features {
            line.push_str(&format!(",{:.6}", val));
        }
        line.push_str(&format!(
            ",{:.6},{:.6},{},{:.6},{},{:.6}",
            ex.max_up_move_pct,
            ex.max_down_move_pct,
            ex.direction,
            ex.magnitude_pct,
            if ex.is_super { 1 } else { 0 },
            ex.future_return_20
        ));
        writeln!(file, "{}", line)?;
    }

    Ok(())
}

/// Get all feature names (indicator + derived + dynamic) in order
pub fn all_feature_names() -> Vec<&'static str> {
    let mut names: Vec<&str> = INDICATOR_FEATURES.to_vec();
    names.extend_from_slice(crate::config::DERIVED_FEATURES);
    names.extend_from_slice(crate::config::DYNAMIC_FEATURES);
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_candle(close: f64, high: f64, low: f64) -> CandleWithIndicators {
        CandleWithIndicators {
            time: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            symbol_id: 1,
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
    fn test_build_labels_basic() {
        // Create 25 candles: close=100, but candle 5 has high=110 (10% up)
        let mut candles: Vec<CandleWithIndicators> = (0..25)
            .map(|_| make_test_candle(100.0, 101.0, 99.0))
            .collect();

        // Candle at index 5 has a spike up
        candles[5].high = 110.0;

        // sl_fraction = 0.5 means SL is at 50% of target move
        let examples = build_labels(&candles, 0, 20, 5.0, 0.5, 5);

        // We should get examples for indices 0..5 (since 25 - 20 = 5)
        assert_eq!(examples.len(), 5);

        // Example at t=0 should see the spike at t=5 within lookahead
        let ex0 = &examples[0];
        assert!(ex0.max_up_move_pct >= 9.0); // ~10%
        assert_eq!(ex0.direction, 1); // LONG
        assert!(ex0.is_super); // 10% > 5% threshold
    }

    #[test]
    fn test_build_labels_short_direction() {
        let mut candles: Vec<CandleWithIndicators> = (0..25)
            .map(|_| make_test_candle(100.0, 101.0, 99.0))
            .collect();

        // Candle at index 3 has a drop
        candles[3].low = 90.0;

        let examples = build_labels(&candles, 0, 20, 5.0, 0.5, 5);

        // Example at t=0 should see the drop
        let ex0 = &examples[0];
        assert!(ex0.max_down_move_pct >= 9.0); // ~10%
        assert_eq!(ex0.direction, -1); // SHORT
        assert!(ex0.is_super);
    }

    #[test]
    fn test_build_labels_first_touch_sl_hit_first() {
        // Test First-Touch logic: SL hit before TP = not super
        let mut candles: Vec<CandleWithIndicators> = (0..25)
            .map(|_| make_test_candle(100.0, 101.0, 99.0))
            .collect();

        // Candle at index 3 drops 3% (hits SL at 2.5%), then candle 5 rises 10%
        candles[3].low = 97.0;  // -3% drop (hits SL first)
        candles[5].high = 110.0; // +10% rise (would hit TP, but SL already hit)

        // target=5%, sl_fraction=0.5 -> SL at 2.5% down, TP at 5% up
        let examples = build_labels(&candles, 0, 20, 5.0, 0.5, 5);

        let ex0 = &examples[0];
        // LONG would hit SL first (97 < 97.5), so long_win = false
        // max_up is still 10%, but is_super should be false because SL hit first
        assert!(ex0.max_up_move_pct >= 9.0);
        assert!(!ex0.is_super); // SL hit before TP
    }

    #[test]
    fn test_feature_count() {
        let candle = make_test_candle(100.0, 101.0, 99.0);
        let static_features = candle.full_features();
        // Static features (indicators + derived) = 52
        assert_eq!(static_features.len(), crate::config::static_feature_count());
        // Total with dynamic = 90
        assert_eq!(
            static_features.len() + crate::config::dynamic_feature_count(),
            crate::config::total_feature_count()
        );
    }

    #[test]
    fn test_dynamic_features_basic() {
        // Create 20 candles with default values
        let candles: Vec<CandleWithIndicators> = (0..20)
            .map(|_| make_test_candle(100.0, 101.0, 99.0))
            .collect();

        // At index 15 (enough lookback=15), dynamic features should be computable
        let dyn_feats = compute_dynamic_features(&candles, 15);
        assert_eq!(dyn_feats.len(), crate::config::dynamic_feature_count());

        // With identical candles, most deltas should be 0 or near-0
        for &v in &dyn_feats {
            assert!(v.is_finite(), "Dynamic feature is not finite: {}", v);
        }
    }

    #[test]
    fn test_dynamic_features_not_enough_lookback() {
        let candles: Vec<CandleWithIndicators> = (0..10)
            .map(|_| make_test_candle(100.0, 101.0, 99.0))
            .collect();

        // Index 5 has less than 15 bars lookback — should return zeros
        let dyn_feats = compute_dynamic_features(&candles, 5);
        assert_eq!(dyn_feats.len(), crate::config::dynamic_feature_count());
        assert!(dyn_feats.iter().all(|&v| v == 0.0), "Expected all zeros for insufficient lookback");
    }
}
