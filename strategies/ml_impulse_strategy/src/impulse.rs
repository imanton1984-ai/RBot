// strategies/ml_impulse_strategy/src/impulse.rs
//
// Core logic for Impulse Absorption & Engulfing (IAE) strategy — v3.
//
// KEY CHANGES (v3 — relaxed heuristic for ML pipeline):
//   1. ATR-based dynamic impulse threshold (1.5× ATR) instead of static %
//   2. Multi-bar absorption: 1-bar OR 2-bar engulfing (piercing line extension)
//   3. Volume spike on EITHER impulse OR absorption candle
//   4. Micro-gap leniency (0.1%) for continuous 24/7 crypto markets
//   5. Lowered static floors: 15m=1.0%, 1h=1.5%, 4h=3.5%
//   6. Increased max_hold to 8 candles
//
// PHILOSOPHY:
//   The heuristic ("The Hunter") is now LOOSE on purpose — it generates
//   a high volume of noisy candidate patterns. XGBoost ("The Judge")
//   will learn to separate the high-probability structural reversals
//   from the noise, easily lifting post-filter WR above 50%.
//
// DETECTION LOGIC:
//   1. Find strong impulse candle (body >= 1.5× ATR, with static % floor)
//   2. Check for 1-bar engulfing OR 2-bar piercing-line absorption
//   3. Verify body-to-shadow ratio > 0.5 on impulse (confident move)
//   4. Verify volume spike > 1.3× SMA(20) on EITHER impulse OR absorption
//
// FEATURE EXTRACTION:
//   Multi-TF indicator snapshot BEFORE the signal bar.
//   79 features per candle per TF + 9 engulfing-specific features.
//
// PER-TF THRESHOLDS (v3 — ATR-based primary, static floor):
//   15m: 1.0% floor, TP=1.5%, SL=1.0%, max_hold=8
//   1h:  1.5% floor, TP=2.5%, SL=1.5%, max_hold=8
//   4h:  3.5% floor, TP=4.0%, SL=3.0%, max_hold=8

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ═════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═════════════════════════════════════════════════════════════════════════════

/// Timeframes used for multi-TF feature extraction (ordered high → low).
pub const ANALYSIS_TIMEFRAMES: &[i32] = &[1440, 240, 60, 15, 5];

/// Target timeframes where we detect engulfing patterns and simulate trades.
pub const TARGET_TFS: &[i32] = &[15, 60, 240];

/// Number of candles to look back BEFORE the signal for feature extraction.
pub const PRE_SIGNAL_LOOKBACK: usize = 10;

/// Minimum candles required per symbol per TF.
pub const MIN_CANDLES_PER_TF: usize = 50;

/// Volume SMA lookback for spike detection.
pub const VOLUME_SMA_PERIOD: usize = 20;

/// Maximum candles to hold a position (v3: increased from 5 to 8 for breathing room).
pub const MAX_HOLD_CANDLES: usize = 8;

/// Default ATR multiplier: impulse body must be >= ATR × this.
pub const DEFAULT_ATR_MULTIPLIER: f64 = 1.5;

/// Default micro-gap leniency: 0.1% for open condition in continuous markets.
pub const DEFAULT_OPEN_LENIENCY_PCT: f64 = 0.001;

/// Batch size for XGBoost predictions.
pub const PRED_BATCH_SIZE: usize = 256;

/// Prediction thresholds for analysis buckets.
pub const PRED_THRESHOLDS: &[f32] = &[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90];

// ═════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═════════════════════════════════════════════════════════════════════════════

/// Per-timeframe trade parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfParams {
    pub tf_minutes: i32,
    /// Minimum impulse candle body size (%) — static floor.
    /// The primary threshold is ATR-based; this is a safety floor.
    pub impulse_pct: f64,
    /// Take profit (%) relative to entry.
    pub tp_pct: f64,
    /// Stop loss (%) relative to entry.
    pub sl_pct: f64,
}

/// Full configuration for IAE strategy (v3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpulseConfig {
    /// Per-TF parameters.
    pub tf_params: Vec<TfParams>,

    /// Minimum body-to-candle ratio for the impulse candle.
    /// Filters doji/hammer candles with long wicks.
    /// Env: IAE_MIN_BODY_RATIO (default: 0.5)
    pub min_body_ratio: f64,

    /// Volume spike multiplier: volume > avg_vol × spike_mult
    /// on EITHER impulse OR absorption candle.
    /// Env: IAE_VOLUME_SPIKE_MULT (default: 1.3)
    pub volume_spike_mult: f64,

    /// ATR multiplier: impulse body must be >= ATR × this.
    /// When ATR is unavailable, falls back to static impulse_pct.
    /// Env: IAE_ATR_MULTIPLIER (default: 1.5)
    pub atr_multiplier: f64,

    /// Micro-gap leniency for open condition (fraction, not percent).
    /// e.g., 0.001 = 0.1% leniency on curr.open vs prev.close.
    /// Fixes the "continuous market" engulfing trap where
    /// tick-level micro-gaps kill valid patterns.
    /// Env: IAE_OPEN_LENIENCY_PCT (default: 0.001)
    pub open_leniency_pct: f64,

    /// Number of candles BEFORE signal for feature extraction.
    /// Env: IAE_PRE_SIGNAL_LOOKBACK (default: 10)
    pub pre_signal_lookback: usize,

    /// Max candles to hold position.
    /// Env: IAE_MAX_HOLD (default: 8)
    pub max_hold: usize,

    /// Negative:positive ratio for dataset generation.
    /// Env: IAE_NEGATIVE_RATIO (default: 3)
    pub negative_ratio: usize,
}

impl Default for ImpulseConfig {
    fn default() -> Self {
        Self {
            tf_params: vec![
                // v3: Lowered impulse floors + ATR-based primary threshold
                // These impulse_pct values are static FLOORS — the ATR check is primary.
                // TP>SL for asymmetric R:R ≈ 1.5:1 (breakeven WR ~40%)
                TfParams { tf_minutes: 15,  impulse_pct: 1.0,  tp_pct: 1.5, sl_pct: 1.0 },
                TfParams { tf_minutes: 60,  impulse_pct: 1.5,  tp_pct: 2.5, sl_pct: 1.5 },
                TfParams { tf_minutes: 240, impulse_pct: 3.5,  tp_pct: 4.0, sl_pct: 3.0 },
            ],
            // v3: Lowered from 0.55 — allows impulses with moderate wicks
            min_body_ratio: 0.50,
            // v3: Lowered from 1.5 — volume spike on EITHER bar compensates
            volume_spike_mult: 1.3,
            // v3: NEW — ATR-based dynamic threshold (primary)
            atr_multiplier: DEFAULT_ATR_MULTIPLIER,
            // v3: NEW — micro-gap leniency for continuous crypto markets
            open_leniency_pct: DEFAULT_OPEN_LENIENCY_PCT,
            pre_signal_lookback: PRE_SIGNAL_LOOKBACK,
            max_hold: MAX_HOLD_CANDLES,
            negative_ratio: 3,
        }
    }
}

impl ImpulseConfig {
    /// Load from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("IAE_MIN_BODY_RATIO") {
            if let Ok(n) = v.parse::<f64>() { cfg.min_body_ratio = n.clamp(0.2, 0.95); }
        }
        if let Ok(v) = std::env::var("IAE_VOLUME_SPIKE_MULT") {
            if let Ok(n) = v.parse::<f64>() { cfg.volume_spike_mult = n.clamp(1.0, 10.0); }
        }
        if let Ok(v) = std::env::var("IAE_ATR_MULTIPLIER") {
            if let Ok(n) = v.parse::<f64>() { cfg.atr_multiplier = n.clamp(0.5, 5.0); }
        }
        if let Ok(v) = std::env::var("IAE_OPEN_LENIENCY_PCT") {
            if let Ok(n) = v.parse::<f64>() { cfg.open_leniency_pct = n.clamp(0.0, 0.01); }
        }
        if let Ok(v) = std::env::var("IAE_PRE_SIGNAL_LOOKBACK") {
            if let Ok(n) = v.parse::<usize>() { cfg.pre_signal_lookback = n.clamp(3, 50); }
        }
        if let Ok(v) = std::env::var("IAE_MAX_HOLD") {
            if let Ok(n) = v.parse::<usize>() { cfg.max_hold = n.clamp(1, 20); }
        }
        if let Ok(v) = std::env::var("IAE_NEGATIVE_RATIO") {
            if let Ok(n) = v.parse::<usize>() { cfg.negative_ratio = n.clamp(1, 10); }
        }

        // Override per-TF params from env
        for tf_p in &mut cfg.tf_params {
            let prefix = format!("IAE_TF{}", tf_p.tf_minutes);
            if let Ok(v) = std::env::var(format!("{}_IMPULSE_PCT", prefix)) {
                if let Ok(n) = v.parse::<f64>() { tf_p.impulse_pct = n.clamp(0.3, 30.0); }
            }
            if let Ok(v) = std::env::var(format!("{}_TP_PCT", prefix)) {
                if let Ok(n) = v.parse::<f64>() { tf_p.tp_pct = n.clamp(0.5, 20.0); }
            }
            if let Ok(v) = std::env::var(format!("{}_SL_PCT", prefix)) {
                if let Ok(n) = v.parse::<f64>() { tf_p.sl_pct = n.clamp(0.5, 20.0); }
            }
        }

        cfg
    }

    /// Get TfParams for a given timeframe, or None if not configured.
    pub fn params_for_tf(&self, tf_minutes: i32) -> Option<&TfParams> {
        self.tf_params.iter().find(|p| p.tf_minutes == tf_minutes)
    }

    /// Print config summary.
    pub fn log_summary(&self) {
        tracing::info!("ImpulseConfig (v3 — relaxed heuristic):");
        tracing::info!("  min_body_ratio: {:.2}", self.min_body_ratio);
        tracing::info!("  volume_spike_mult: {:.1}× (EITHER impulse OR absorption)", self.volume_spike_mult);
        tracing::info!("  atr_multiplier: {:.1}× (dynamic impulse threshold)", self.atr_multiplier);
        tracing::info!("  open_leniency_pct: {:.3} ({:.1}% micro-gap tolerance)", self.open_leniency_pct, self.open_leniency_pct * 100.0);
        tracing::info!("  pre_signal_lookback: {} candles", self.pre_signal_lookback);
        tracing::info!("  max_hold: {} candles", self.max_hold);
        tracing::info!("  negative_ratio: {}:1", self.negative_ratio);
        tracing::info!("  multi-bar absorption: ENABLED (1-bar + 2-bar piercing line)");
        for tf_p in &self.tf_params {
            tracing::info!("  TF {}m: floor≥{:.1}%, TP={:.1}%, SL={:.1}%",
                tf_p.tf_minutes, tf_p.impulse_pct, tf_p.tp_pct, tf_p.sl_pct);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// DATA TYPES
// ═════════════════════════════════════════════════════════════════════════════

/// Signal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalDir {
    Long,  // bullish engulfing → expect price to go UP
    Short, // bearish engulfing → expect price to go DOWN
}

impl std::fmt::Display for SignalDir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalDir::Long => write!(f, "LONG"),
            SignalDir::Short => write!(f, "SHORT"),
        }
    }
}

/// A candle + all indicator columns from DB.
/// Same structure as pump_dump — kept independent to avoid coupling.
#[derive(Debug, Clone)]
pub struct CandleInd {
    pub time: DateTime<Utc>,
    pub symbol: String,
    pub symbol_id: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    // Indicators from market.indicators_wide
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
    pub alligator_jaw: f64,
    pub alligator_teeth: f64,
    pub alligator_lips: f64,
    pub mfi: f64,
    pub fibo_pivot: f64,
    pub fibo_r1: f64,
    pub fibo_s1: f64,
    pub supertrend: f64,
    pub supertrend_dir: f64,
    pub cmf: f64,
}

impl CandleInd {
    /// Raw indicator values in canonical order (33 features).
    pub fn raw_indicator_values(&self) -> [f64; 33] {
        [
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

    /// Derived features normalized by price (19 features).
    pub fn derived_features(&self) -> [f64; 19] {
        let close = self.close;
        let safe_div = |a: f64, b: f64| if b.abs() > 1e-12 { a / b } else { 0.0 };

        let bb_range = self.bb_upper - self.bb_lower;
        let bb_position = if bb_range.abs() > 1e-12 {
            (close - self.bb_lower) / bb_range
        } else {
            0.5
        };

        [
            self.rsi / 100.0,
            self.cci / 200.0,
            self.stoch_k / 100.0,
            (self.williams + 100.0) / 100.0,
            bb_position,
            safe_div(bb_range, close) * 100.0,
            safe_div(self.atr, close) * 100.0,
            safe_div(close - self.sma, close) * 100.0,
            safe_div(close - self.ema_20, close) * 100.0,
            safe_div(close - self.ema_50, close) * 100.0,
            safe_div(close - self.ema_200, close) * 100.0,
            safe_div(close - self.vwap, close) * 100.0,
            safe_div(self.macd_hist, close) * 1000.0,
            0.0, // obv_change_pct  (needs history)
            if self.volume_spike > 2.0 { 1.0 } else { 0.0 },
            self.mfi / 100.0,
            safe_div(close - self.fibo_pivot, close) * 100.0,
            safe_div(close - self.supertrend, close) * 100.0,
            safe_div(self.alligator_jaw - self.alligator_lips, close) * 100.0,
        ]
    }

    /// Candle body features (4 features).
    pub fn body_features(&self) -> [f64; 4] {
        let range = self.high - self.low;
        let body = (self.close - self.open).abs();
        let safe_div = |a: f64, b: f64| if b.abs() > 1e-12 { a / b } else { 0.0 };

        [
            safe_div(body, range),
            safe_div(self.high - self.close.max(self.open), range),
            safe_div(self.close.min(self.open) - self.low, range),
            safe_div(self.close - self.open, self.open) * 100.0,
        ]
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// ENGULFING DETECTION (v3 — relaxed, ATR-based, multi-bar)
// ═════════════════════════════════════════════════════════════════════════════

/// Result of heuristic engulfing detection on a single bar.
#[derive(Debug, Clone)]
pub struct EngulfingSignal {
    /// Index in the candle array — the entry bar (last absorption bar).
    pub bar_idx: usize,
    /// Direction of the trade.
    pub direction: SignalDir,
    /// Body size percentage of the impulse candle.
    pub impulse_pct: f64,
    /// Body-to-candle ratio (impulse quality).
    pub body_ratio: f64,
    /// Best volume / SMA(20) ratio across impulse + absorption bars.
    pub volume_ratio: f64,
    /// Number of absorption bars: 1 (standard engulfing) or 2 (piercing line).
    pub absorption_bars: u8,
}

/// Compute SMA of volume over `period` candles ending at `idx` (inclusive).
/// Uses candles `[idx - period + 1 .. idx]`.
#[inline]
fn volume_sma(candles: &[CandleInd], idx: usize, period: usize) -> f64 {
    if idx + 1 < period { return candles[idx].volume; }
    let start = idx + 1 - period;
    let sum: f64 = candles[start..=idx].iter().map(|c| c.volume).sum();
    sum / period as f64
}

/// Check if bar `i` is the entry bar for an impulse absorption pattern.
///
/// v3 DETECTION LOGIC:
///   1-BAR: impulse at `i-1`, absorption at `i`
///     - `prev` (i-1) = strong directional impulse (ATR-based or static floor)
///     - `curr` (i) = reversal candle that fully engulfs the impulse
///     - open leniency: curr.open allowed to deviate by 0.1% from prev.close
///     - volume spike on EITHER prev or curr
///
///   2-BAR (piercing line extension): impulse at `i-2`, absorption spans `i-1` + `i`
///     - `impulse` (i-2) = strong directional impulse
///     - `first` (i-1) = partial reversal (doesn't fully engulf, but closes past impulse.close)
///     - `curr` (i) = completes absorption (close >= impulse.open for LONG)
///     - volume spike on ANY of {impulse, first, curr}
///
/// NO LOOK-AHEAD: Only uses data at or before bar `i`.
pub fn check_engulfing_at_bar(
    candles: &[CandleInd],
    i: usize,
    config: &ImpulseConfig,
    tf_params: &TfParams,
) -> Option<EngulfingSignal> {
    // ══════════════════════════════════════════════════════════════
    //  1-BAR ABSORPTION: impulse at i-1, absorption at i
    // ══════════════════════════════════════════════════════════════
    if i >= 1 && i >= VOLUME_SMA_PERIOD {
        if let Some(sig) = check_1bar_pattern(candles, i, config, tf_params) {
            return Some(sig);
        }
    }

    // ══════════════════════════════════════════════════════════════
    //  2-BAR ABSORPTION: impulse at i-2, first absorb at i-1, completes at i
    // ══════════════════════════════════════════════════════════════
    if i >= 2 && i >= VOLUME_SMA_PERIOD {
        if let Some(sig) = check_2bar_pattern(candles, i, config, tf_params) {
            return Some(sig);
        }
    }

    None
}

/// 1-bar pattern: impulse at `i-1`, full absorption at `i`.
fn check_1bar_pattern(
    candles: &[CandleInd],
    i: usize,
    config: &ImpulseConfig,
    tf_params: &TfParams,
) -> Option<EngulfingSignal> {
    let curr = &candles[i];
    let prev = &candles[i - 1];

    // Skip zero-price candles
    if curr.open.abs() < 1e-12 || prev.open.abs() < 1e-12
       || curr.close.abs() < 1e-12 || prev.close.abs() < 1e-12 {
        return None;
    }

    let threshold = tf_params.impulse_pct / 100.0;
    let atr_mult = config.atr_multiplier;
    let leniency = config.open_leniency_pct;

    // ── PREV candle = the IMPULSE ──
    let prev_range = prev.high - prev.low;
    if prev_range < 1e-12 { return None; }
    let prev_body = (prev.close - prev.open).abs();
    let prev_body_ratio = prev_body / prev_range;

    // Impulse candle must have strong body (not a doji/hammer)
    if prev_body_ratio < config.min_body_ratio { return None; }

    // ── Dynamic ATR threshold (primary) with static floor (fallback) ──
    let impulse_valid = if prev.atr > 1e-12 {
        prev_body >= prev.atr * atr_mult
    } else {
        // ATR unavailable — fall back to static %
        prev_body / prev.close.abs().max(1e-12) >= threshold
    };
    if !impulse_valid { return None; }

    // ── Volume spike on EITHER impulse OR absorption ──
    let avg_vol = volume_sma(candles, i.saturating_sub(1), VOLUME_SMA_PERIOD);
    let prev_vol_ratio = if avg_vol > 1e-12 { prev.volume / avg_vol } else { 1.0 };
    let curr_vol_ratio = if avg_vol > 1e-12 { curr.volume / avg_vol } else { 1.0 };
    let valid_volume = prev_vol_ratio >= config.volume_spike_mult
                    || curr_vol_ratio >= config.volume_spike_mult;
    if !valid_volume { return None; }

    let best_vol_ratio = prev_vol_ratio.max(curr_vol_ratio);

    let prev_bearish = prev.close < prev.open;
    let prev_bullish = prev.close > prev.open;
    let curr_bullish = curr.close > curr.open;
    let curr_bearish = curr.close < curr.open;

    // ── LONG: Bearish impulse absorbed by bullish candle ──
    if prev_bearish && curr_bullish {
        let impulse = (prev.open - prev.close) / prev.close;
        if impulse >= threshold {
            // Micro-gap leniency: allow curr.open to be slightly above prev.close
            let open_leniency = prev.close * (1.0 + leniency);
            if curr.close >= prev.open && curr.open <= open_leniency {
                return Some(EngulfingSignal {
                    bar_idx: i,
                    direction: SignalDir::Long,
                    impulse_pct: impulse * 100.0,
                    body_ratio: prev_body_ratio,
                    volume_ratio: best_vol_ratio,
                    absorption_bars: 1,
                });
            }
        }
    }

    // ── SHORT: Bullish impulse absorbed by bearish candle ──
    if prev_bullish && curr_bearish {
        let impulse = (prev.close - prev.open) / prev.open;
        if impulse >= threshold {
            // Micro-gap leniency: allow curr.open to be slightly below prev.close
            let open_leniency = prev.close * (1.0 - leniency);
            if curr.close <= prev.open && curr.open >= open_leniency {
                return Some(EngulfingSignal {
                    bar_idx: i,
                    direction: SignalDir::Short,
                    impulse_pct: impulse * 100.0,
                    body_ratio: prev_body_ratio,
                    volume_ratio: best_vol_ratio,
                    absorption_bars: 1,
                });
            }
        }
    }

    None
}

/// 2-bar pattern: impulse at `i-2`, partial absorb at `i-1`, completes at `i`.
///
/// This is the "piercing line extension" — the market needs 2 bars to fully
/// reverse the impulse. Common in less liquid pairs or strong impulses.
///
/// NO LOOK-AHEAD: all three bars (i-2, i-1, i) are in the past at bar `i`.
fn check_2bar_pattern(
    candles: &[CandleInd],
    i: usize,
    config: &ImpulseConfig,
    tf_params: &TfParams,
) -> Option<EngulfingSignal> {
    let curr = &candles[i];
    let first = &candles[i - 1];  // first absorption bar
    let impulse = &candles[i - 2]; // the impulse candle

    // Skip zero-price candles
    if impulse.open.abs() < 1e-12 || impulse.close.abs() < 1e-12
       || first.close.abs() < 1e-12 || curr.close.abs() < 1e-12 {
        return None;
    }

    let threshold = tf_params.impulse_pct / 100.0;
    let atr_mult = config.atr_multiplier;

    // ── IMPULSE candle (i-2) checks ──
    let imp_range = impulse.high - impulse.low;
    if imp_range < 1e-12 { return None; }
    let imp_body = (impulse.close - impulse.open).abs();
    let imp_body_ratio = imp_body / imp_range;

    if imp_body_ratio < config.min_body_ratio { return None; }

    // ATR-based dynamic threshold with static floor fallback
    let impulse_valid = if impulse.atr > 1e-12 {
        imp_body >= impulse.atr * atr_mult
    } else {
        imp_body / impulse.close.abs().max(1e-12) >= threshold
    };
    if !impulse_valid { return None; }

    // ── Volume spike on ANY of {impulse, first, curr} ──
    let avg_vol = volume_sma(candles, i.saturating_sub(1), VOLUME_SMA_PERIOD);
    let imp_vol_ratio = if avg_vol > 1e-12 { impulse.volume / avg_vol } else { 1.0 };
    let first_vol_ratio = if avg_vol > 1e-12 { first.volume / avg_vol } else { 1.0 };
    let curr_vol_ratio = if avg_vol > 1e-12 { curr.volume / avg_vol } else { 1.0 };
    let valid_volume = imp_vol_ratio >= config.volume_spike_mult
                    || first_vol_ratio >= config.volume_spike_mult
                    || curr_vol_ratio >= config.volume_spike_mult;
    if !valid_volume { return None; }

    let best_vol_ratio = imp_vol_ratio.max(first_vol_ratio).max(curr_vol_ratio);

    let imp_bearish = impulse.close < impulse.open;
    let imp_bullish = impulse.close > impulse.open;

    // ── LONG: Bearish impulse (i-2) absorbed by bars (i-1, i) ──
    //   first (i-1): started recovering (close > impulse.close)
    //   curr (i): completes absorption (close >= impulse.open)
    if imp_bearish {
        let imp_pct = (impulse.open - impulse.close) / impulse.close;
        if imp_pct >= threshold
            && first.close > impulse.close  // first bar started recovering
            && curr.close >= impulse.open   // second bar completes absorption
        {
            return Some(EngulfingSignal {
                bar_idx: i,
                direction: SignalDir::Long,
                impulse_pct: imp_pct * 100.0,
                body_ratio: imp_body_ratio,
                volume_ratio: best_vol_ratio,
                absorption_bars: 2,
            });
        }
    }

    // ── SHORT: Bullish impulse (i-2) absorbed by bars (i-1, i) ──
    //   first (i-1): started declining (close < impulse.close)
    //   curr (i): completes absorption (close <= impulse.open)
    if imp_bullish {
        let imp_pct = (impulse.close - impulse.open) / impulse.open;
        if imp_pct >= threshold
            && first.close < impulse.close  // first bar started declining
            && curr.close <= impulse.open   // second bar completes absorption
        {
            return Some(EngulfingSignal {
                bar_idx: i,
                direction: SignalDir::Short,
                impulse_pct: imp_pct * 100.0,
                body_ratio: imp_body_ratio,
                volume_ratio: best_vol_ratio,
                absorption_bars: 2,
            });
        }
    }

    None
}

/// Detect all impulse absorption + engulfing patterns on a given TF.
///
/// v3: Uses check_engulfing_at_bar (1-bar + 2-bar) with dedup logic
/// to avoid double-counting the same impulse from different iterations.
///
/// NO LOOK-AHEAD: Only uses data at or before each bar.
pub fn detect_engulfing_patterns(
    candles: &[CandleInd],
    config: &ImpulseConfig,
    tf_params: &TfParams,
    min_lookback: usize,
) -> Vec<EngulfingSignal> {
    let mut signals = Vec::new();

    // Start after we have enough history for volume SMA and feature lookback
    let start = min_lookback.max(VOLUME_SMA_PERIOD).max(2);

    // Track last signal bar to avoid double-counting same impulse
    // If bar i-1 triggered a 1-bar signal (impulse at i-2), we don't want
    // bar i to trigger a 2-bar signal for the same impulse (i-2).
    let mut last_signal_bar: Option<usize> = None;

    for i in start..candles.len() {
        // Try 1-bar first (stronger pattern — full absorption in one bar)
        if i >= VOLUME_SMA_PERIOD {
            if let Some(signal) = check_1bar_pattern(candles, i, config, tf_params) {
                last_signal_bar = Some(i);
                signals.push(signal);
                continue;
            }
        }

        // Try 2-bar only if previous bar wasn't already a signal
        // (avoids double-counting the same impulse candle)
        if i >= 2 && i >= VOLUME_SMA_PERIOD && last_signal_bar != Some(i - 1) {
            if let Some(signal) = check_2bar_pattern(candles, i, config, tf_params) {
                last_signal_bar = Some(i);
                signals.push(signal);
            }
        }
    }

    signals
}

// ═════════════════════════════════════════════════════════════════════════════
// FEATURE EXTRACTION
// ═════════════════════════════════════════════════════════════════════════════

/// Raw indicator feature names (33).
pub const RAW_INDICATOR_NAMES: &[&str] = &[
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

/// Derived feature names (19).
pub const DERIVED_FEATURE_NAMES: &[&str] = &[
    "rsi_norm", "cci_norm", "stoch_norm", "williams_norm",
    "bb_position", "bb_width_pct", "atr_pct",
    "price_vs_sma", "price_vs_ema20", "price_vs_ema50",
    "price_vs_ema200", "price_vs_vwap",
    "macd_norm", "obv_change_pct", "volume_spike_flag",
    "mfi_norm", "price_vs_fibo_pivot", "price_vs_supertrend",
    "alligator_spread",
];

/// Body feature names (4).
pub const BODY_FEATURE_NAMES: &[&str] = &[
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "candle_return_pct",
];

/// Features per candle (raw + derived + body) before temporals.
pub const FEATURES_PER_CANDLE_BASE: usize = 33 + 19 + 4; // 56

/// Temporal feature names (23).
pub const TEMPORAL_FEATURE_NAMES: &[&str] = &[
    "price_ret_lb1", "price_ret_lb3", "price_ret_lb5", "price_ret_lb10",
    "vol_ratio_5", "vol_ratio_10", "vol_roc_1",
    "rsi_slope_3", "rsi_slope_10",
    "atr_ratio_5", "atr_ratio_10",
    "macd_hist_roc_1", "macd_hist_roc_3",
    "trend_persist_5", "supertrend_persist_10",
    "bb_squeeze_pctl",
    "obv_divergence",
    "price_acceleration",
    "vol_up_down_ratio",
    "dist_to_low_50", "dist_to_high_50",
    "wick_pressure_3",
    "ema_convergence_change",
];

/// Number of temporal features.
pub const TEMPORAL_FEATURE_COUNT: usize = 23;

/// Total features per candle = 56 + 23 = 79.
pub const FULL_FEATURES_PER_CANDLE: usize = FEATURES_PER_CANDLE_BASE + TEMPORAL_FEATURE_COUNT;

/// Engulfing-specific feature names (9 features appended once per sample).
/// v3: Added eng_absorption_bars (1-bar vs 2-bar pattern indicator).
pub const ENGULFING_FEATURE_NAMES: &[&str] = &[
    "eng_impulse_pct",        // impulse candle body size (%) — the impulse
    "eng_body_ratio",         // impulse candle body/range ratio — impulse quality
    "eng_volume_ratio",       // best volume / SMA(20) — absorption confirmation
    "eng_absorb_body_pct",    // signal bar candle body (%) — absorption strength
    "eng_ratio_bodies",       // absorption body / impulse body ratio
    "eng_gap_pct",            // gap between impulse close and first absorption open (%)
    "eng_dist_ema200_pct",    // distance from EMA200 (% of price)
    "eng_atr_mult",           // impulse body / ATR — impulse significance
    "eng_absorption_bars",    // v3: 1.0 or 2.0 — single or multi-bar absorption
];

/// Number of engulfing meta-features (v3: 9, was 8).
pub const ENGULFING_FEATURE_COUNT: usize = 9;

/// Compute temporal features for candle at index `t` (23 features).
pub fn compute_temporal_features(candles: &[CandleInd], t: usize) -> Vec<f64> {
    let safe_div = |a: f64, b: f64| -> f64 {
        if b.abs() > 1e-12 { a / b } else { 0.0 }
    };

    let cur = &candles[t];
    let close = cur.close;
    let mut feats = Vec::with_capacity(TEMPORAL_FEATURE_COUNT);

    // Price momentum at different lookbacks (4)
    for &lb in &[1usize, 3, 5, 10] {
        let ret = if t >= lb && candles[t - lb].close.abs() > 1e-12 {
            (close - candles[t - lb].close) / candles[t - lb].close * 100.0
        } else {
            0.0
        };
        feats.push(ret);
    }

    // Volume dynamics (3)
    let vol = cur.volume;
    let vol_avg_5 = if t >= 5 {
        (1..=5).map(|j| candles[t - j].volume).sum::<f64>() / 5.0
    } else { vol };
    feats.push(if vol_avg_5 > 1e-12 { (vol / vol_avg_5).clamp(0.01, 50.0) } else { 1.0 });

    let vol_avg_10 = if t >= 10 {
        (1..=10).map(|j| candles[t - j].volume).sum::<f64>() / 10.0
    } else { vol };
    feats.push(if vol_avg_10 > 1e-12 { (vol / vol_avg_10).clamp(0.01, 50.0) } else { 1.0 });

    let vol_roc_1 = if t >= 1 && candles[t - 1].volume > 1e-12 {
        (vol / candles[t - 1].volume - 1.0).clamp(-10.0, 10.0)
    } else { 0.0 };
    feats.push(vol_roc_1);

    // RSI dynamics (2)
    feats.push(if t >= 3 { (cur.rsi - candles[t - 3].rsi) / 100.0 } else { 0.0 });
    feats.push(if t >= 10 { (cur.rsi - candles[t - 10].rsi) / 100.0 } else { 0.0 });

    // ATR expansion/contraction (2)
    feats.push(if t >= 5 && candles[t - 5].atr.abs() > 1e-12 {
        cur.atr / candles[t - 5].atr - 1.0
    } else { 0.0 });
    feats.push(if t >= 10 && candles[t - 10].atr.abs() > 1e-12 {
        cur.atr / candles[t - 10].atr - 1.0
    } else { 0.0 });

    // MACD histogram momentum (2)
    feats.push(if t >= 1 {
        safe_div(cur.macd_hist - candles[t - 1].macd_hist, close) * 1000.0
    } else { 0.0 });
    feats.push(if t >= 3 {
        safe_div(cur.macd_hist - candles[t - 3].macd_hist, close) * 1000.0
    } else { 0.0 });

    // Trend persistence (2)
    feats.push(if t >= 5 {
        (0..5).map(|j| candles[t - j].trend).sum::<f64>() / 5.0
    } else { 0.0 });
    feats.push(if t >= 10 {
        (0..10).map(|j| candles[t - j].supertrend_dir).sum::<f64>() / 10.0
    } else { 0.0 });

    // BB squeeze percentile (1)
    feats.push(if t >= 50 {
        let cur_bb_width = safe_div(cur.bb_upper - cur.bb_lower, close) * 100.0;
        let mut count_below = 0usize;
        for j in 1..=50 {
            let c = &candles[t - j];
            let w = safe_div(c.bb_upper - c.bb_lower, c.close) * 100.0;
            if w < cur_bb_width { count_below += 1; }
        }
        count_below as f64 / 50.0
    } else { 0.5 });

    // OBV divergence (1)
    feats.push(if t >= 10 {
        let price_slope = close - candles[t - 10].close;
        let obv_slope = cur.obv - candles[t - 10].obv;
        let price_sign = if price_slope > 0.01 * close { 1.0 }
                         else if price_slope < -0.01 * close { -1.0 }
                         else { 0.0 };
        let obv_sign = if obv_slope.abs() > 1e-12 { obv_slope.signum() } else { 0.0 };
        if price_sign == 0.0 && obv_sign > 0.0 { 1.0 }
        else if price_sign == 0.0 && obv_sign < 0.0 { -1.0 }
        else if price_sign > 0.0 && obv_sign < 0.0 { -0.5 }
        else if price_sign < 0.0 && obv_sign > 0.0 { 0.5 }
        else { 0.0 }
    } else { 0.0 });

    // Price acceleration (1)
    let ret_recent_3 = if t >= 3 { safe_div(close - candles[t - 3].close, close) * 100.0 } else { 0.0 };
    let ret_prev_3 = if t >= 6 {
        safe_div(candles[t - 3].close - candles[t - 6].close, candles[t - 3].close) * 100.0
    } else { 0.0 };
    feats.push(ret_recent_3 - ret_prev_3);

    // Volume up vs down ratio (1)
    feats.push(if t >= 10 {
        let (mut vu, mut vd) = (0.0f64, 0.0f64);
        for j in 0..10 {
            let c = &candles[t - j];
            if c.close >= c.open { vu += c.volume; } else { vd += c.volume; }
        }
        if vd > 1e-12 { (vu / vd).clamp(0.1, 10.0) }
        else if vu > 1e-12 { 10.0 }
        else { 1.0 }
    } else { 1.0 });

    // Distance to 50-bar extremes (2)
    let (dist_low, dist_high) = if t >= 50 {
        let min_low = (0..50).map(|j| candles[t - j].low).fold(f64::MAX, f64::min);
        let max_high = (0..50).map(|j| candles[t - j].high).fold(f64::MIN, f64::max);
        (
            safe_div(close - min_low, close) * 100.0,
            safe_div(max_high - close, close) * 100.0,
        )
    } else { (0.0, 0.0) };
    feats.push(dist_low);
    feats.push(dist_high);

    // Wick rejection pressure (1)
    feats.push(if t >= 3 {
        let mut ws = 0.0;
        for j in 0..3 {
            let c = &candles[t - j];
            let uw = c.high - c.close.max(c.open);
            let lw = c.close.min(c.open) - c.low;
            let a = if c.atr > 1e-12 { c.atr } else { 1.0 };
            ws += safe_div(lw - uw, a);
        }
        ws / 3.0
    } else { 0.0 });

    // EMA convergence change (1)
    feats.push(if t >= 5 {
        let gap_now = cur.ema_20 - cur.ema_50;
        let gap_prev = candles[t - 5].ema_20 - candles[t - 5].ema_50;
        safe_div(gap_now - gap_prev, close) * 100.0
    } else { 0.0 });

    debug_assert_eq!(feats.len(), TEMPORAL_FEATURE_COUNT,
        "Temporal features count mismatch: expected {}, got {}", TEMPORAL_FEATURE_COUNT, feats.len());
    feats
}

/// Extract full feature vector for a single candle at index `t`.
/// Returns FULL_FEATURES_PER_CANDLE (79) elements.
pub fn extract_candle_features(candles: &[CandleInd], t: usize) -> Vec<f64> {
    let c = &candles[t];
    let mut feats = Vec::with_capacity(FULL_FEATURES_PER_CANDLE);

    feats.extend_from_slice(&c.raw_indicator_values());   // 33
    feats.extend_from_slice(&c.derived_features());        // 19
    feats.extend_from_slice(&c.body_features());           // 4
    feats.extend(compute_temporal_features(candles, t));    // 23

    debug_assert_eq!(feats.len(), FULL_FEATURES_PER_CANDLE);
    feats
}

/// Extract engulfing-specific meta-features for the signal bar (9 features).
///
/// v3: Handles both 1-bar and 2-bar absorption patterns.
///   - 1-bar: impulse = bar_idx - 1, absorption = bar_idx
///   - 2-bar: impulse = bar_idx - 2, first absorb = bar_idx - 1, completing = bar_idx
///
/// These are appended ONCE per sample (not per-candle × per-TF).
pub fn extract_engulfing_features(
    candles: &[CandleInd],
    signal: &EngulfingSignal,
) -> [f64; ENGULFING_FEATURE_COUNT] {
    let safe_div = |a: f64, b: f64| if b.abs() > 1e-12 { a / b } else { 0.0 };

    let signal_bar = &candles[signal.bar_idx];
    let impulse_offset = signal.absorption_bars as usize; // 1 or 2
    let impulse = &candles[signal.bar_idx - impulse_offset];

    // Impulse body
    let impulse_body = (impulse.close - impulse.open).abs();

    // Absorption candle body % (signal bar — the completing bar)
    let absorb_body_pct = safe_div((signal_bar.close - signal_bar.open).abs(), signal_bar.open) * 100.0;

    // For ratio_bodies: compare total absorption move vs impulse body
    let absorb_body = if signal.absorption_bars == 1 {
        (signal_bar.close - signal_bar.open).abs()
    } else {
        // 2-bar: total absorption from impulse.close to signal_bar.close
        (signal_bar.close - impulse.close).abs()
    };
    let ratio_bodies = if impulse_body > 1e-12 { absorb_body / impulse_body } else { 1.0 };

    // Gap: between impulse close and first absorption bar open
    let first_absorb_bar = &candles[signal.bar_idx - impulse_offset + 1];
    let gap_pct = safe_div(
        (first_absorb_bar.open - impulse.close).abs(),
        impulse.close,
    ) * 100.0;

    let dist_ema200 = safe_div(signal_bar.close - signal_bar.ema_200, signal_bar.close) * 100.0;

    // How many ATRs was the impulse? (impulse significance)
    let atr_mult = if impulse.atr > 1e-12 { impulse_body / impulse.atr } else { 1.0 };

    [
        signal.impulse_pct,                    // impulse candle body %
        signal.body_ratio,                     // impulse candle body/range ratio
        signal.volume_ratio,                   // best volume ratio
        absorb_body_pct,                       // signal bar body %
        ratio_bodies,                          // absorption vs impulse body ratio
        gap_pct,                               // gap between candles
        dist_ema200,                           // distance from EMA200
        atr_mult,                              // impulse body / ATR
        signal.absorption_bars as f64,         // 1.0 or 2.0 (multi-bar indicator)
    ]
}

/// Generate feature names for one candle snapshot (prefixed with TF and candle offset).
pub fn prefixed_feature_names(tf_minutes: i32, candle_offset: usize) -> Vec<String> {
    let prefix = format!("tf{}_c{}", tf_minutes, candle_offset);
    let mut names = Vec::with_capacity(FULL_FEATURES_PER_CANDLE);

    for &name in RAW_INDICATOR_NAMES {
        names.push(format!("{}_{}", prefix, name));
    }
    for &name in DERIVED_FEATURE_NAMES {
        names.push(format!("{}_{}", prefix, name));
    }
    for &name in BODY_FEATURE_NAMES {
        names.push(format!("{}_{}", prefix, name));
    }
    for &name in TEMPORAL_FEATURE_NAMES {
        names.push(format!("{}_{}", prefix, name));
    }

    debug_assert_eq!(names.len(), FULL_FEATURES_PER_CANDLE);
    names
}

/// Generate ALL feature names for the multi-TF pre-signal snapshot + engulfing meta.
///
/// Total = num_tfs × lookback × FULL_FEATURES_PER_CANDLE + ENGULFING_FEATURE_COUNT
pub fn all_feature_names(
    timeframes: &[i32],
    pre_signal_lookback: usize,
) -> Vec<String> {
    let total = timeframes.len() * pre_signal_lookback * FULL_FEATURES_PER_CANDLE
                + ENGULFING_FEATURE_COUNT;
    let mut names = Vec::with_capacity(total);

    for &tf in timeframes {
        for c in 0..pre_signal_lookback {
            names.extend(prefixed_feature_names(tf, c));
        }
    }

    // Engulfing meta features
    for &name in ENGULFING_FEATURE_NAMES {
        names.push(name.to_string());
    }

    names
}

// ═════════════════════════════════════════════════════════════════════════════
// TESTS
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(open: f64, high: f64, low: f64, close: f64, vol: f64) -> CandleInd {
        CandleInd {
            time: Utc::now(), symbol: "TESTUSDT".to_string(), symbol_id: 1,
            open, high, low, close, volume: vol,
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

    fn default_config() -> ImpulseConfig {
        ImpulseConfig::default()
    }

    fn tf15_params() -> TfParams {
        TfParams { tf_minutes: 15, impulse_pct: 1.0, tp_pct: 1.5, sl_pct: 1.0 }
    }

    #[test]
    fn test_bullish_absorption_1bar() {
        // Build 30 normal candles (enough for volume SMA)
        let mut candles: Vec<CandleInd> = (0..30)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.5, 1000.0))
            .collect();

        // Bar 28: BEARISH IMPULSE — big drop
        // open=100, close=96, body=4, range=5.5, ratio≈0.73
        // atr=1.0 (1% of ~100), body=4 >= atr*1.5=1.5 ✓
        // impulse_pct = (100-96)/96 ≈ 4.17% >= 1.0% floor ✓
        candles[28] = make_candle(100.0, 100.5, 95.0, 96.0, 2000.0);
        // Bar 29: BULLISH ABSORPTION — engulfs prev, high volume
        // close(105) >= prev.open(100) ✓, open(96) <= prev.close(96)*1.001 ✓
        candles[29] = make_candle(96.0, 105.5, 95.5, 105.0, 2000.0);

        let config = default_config();
        let tf_p = tf15_params();
        let signals = detect_engulfing_patterns(&candles, &config, &tf_p, 10);
        assert!(!signals.is_empty(), "Should detect bullish 1-bar absorption");
        assert_eq!(signals[0].direction, SignalDir::Long);
        assert_eq!(signals[0].absorption_bars, 1);
        assert!(signals[0].impulse_pct >= 4.0, "impulse_pct={:.1}%", signals[0].impulse_pct);
    }

    #[test]
    fn test_bearish_absorption_1bar() {
        let mut candles: Vec<CandleInd> = (0..30)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.5, 1000.0))
            .collect();

        // Bar 28: BULLISH IMPULSE — big rally
        candles[28] = make_candle(100.0, 106.0, 99.5, 105.0, 2000.0);
        // Bar 29: BEARISH ABSORPTION — engulfs prev
        candles[29] = make_candle(105.0, 106.5, 93.5, 94.0, 2000.0);

        let config = default_config();
        let tf_p = tf15_params();
        let signals = detect_engulfing_patterns(&candles, &config, &tf_p, 10);
        assert!(!signals.is_empty(), "Should detect bearish 1-bar absorption");
        assert_eq!(signals[0].direction, SignalDir::Short);
        assert_eq!(signals[0].absorption_bars, 1);
    }

    #[test]
    fn test_bullish_absorption_2bar() {
        // Build 30 normal candles
        let mut candles: Vec<CandleInd> = (0..31)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.5, 1000.0))
            .collect();

        // Bar 27: BEARISH IMPULSE — big drop from 100 to 96
        // atr=1.0, body=4 >= 1.5*1.0 ✓
        candles[27] = make_candle(100.0, 100.5, 95.0, 96.0, 2000.0);
        // Bar 28: FIRST ABSORPTION — partial recovery (close > impulse.close but < impulse.open)
        // close=98 > impulse.close=96 ✓, but close=98 < impulse.open=100 (not full engulf)
        candles[28] = make_candle(96.0, 98.5, 95.5, 98.0, 1000.0);
        // Bar 29: COMPLETES ABSORPTION — close >= impulse.open
        // close=101 >= impulse.open=100 ✓
        candles[29] = make_candle(98.0, 101.5, 97.5, 101.0, 1500.0);

        let config = default_config();
        let tf_p = tf15_params();
        let signals = detect_engulfing_patterns(&candles, &config, &tf_p, 10);
        assert!(!signals.is_empty(), "Should detect bullish 2-bar absorption");

        let sig = signals.iter().find(|s| s.absorption_bars == 2);
        assert!(sig.is_some(), "Should have a 2-bar signal");
        let sig = sig.unwrap();
        assert_eq!(sig.direction, SignalDir::Long);
        assert_eq!(sig.bar_idx, 29); // signal fires on the completing bar
    }

    #[test]
    fn test_volume_on_impulse_candle() {
        // Volume spike on the IMPULSE candle (not absorption) should work in v3
        let mut candles: Vec<CandleInd> = (0..30)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.5, 1000.0))
            .collect();

        // Bar 28: BEARISH IMPULSE with HIGH VOLUME (3000 vs avg ~1000 = 3x)
        candles[28] = make_candle(100.0, 100.5, 95.0, 96.0, 3000.0);
        // Bar 29: BULLISH ABSORPTION with NORMAL VOLUME (reversal on evaporated sell pressure)
        candles[29] = make_candle(96.0, 105.5, 95.5, 105.0, 1100.0);

        let config = default_config();
        let tf_p = tf15_params();
        let signals = detect_engulfing_patterns(&candles, &config, &tf_p, 10);
        assert!(!signals.is_empty(),
            "v3: Volume spike on impulse candle should be accepted (was rejected in v2)");
    }

    #[test]
    fn test_micro_gap_leniency() {
        // Test that micro-gap doesn't kill valid engulfing patterns
        let mut candles: Vec<CandleInd> = (0..30)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.5, 1000.0))
            .collect();

        // Bar 28: bearish impulse, close=96.0
        candles[28] = make_candle(100.0, 100.2, 95.5, 96.0, 2000.0);
        // Bar 29: absorption opens SLIGHTLY above prev.close (micro-gap of 0.05%)
        // open=96.05 vs prev.close=96.0 → gap=0.05% < 0.1% leniency
        candles[29] = make_candle(96.05, 103.0, 95.8, 101.0, 2000.0);

        let config = default_config();
        let tf_p = tf15_params();
        let signals = detect_engulfing_patterns(&candles, &config, &tf_p, 10);
        assert!(!signals.is_empty(),
            "v3: Micro-gap of 0.05% should be tolerated (0.1% leniency)");
    }

    #[test]
    fn test_no_signal_low_volume() {
        let mut candles: Vec<CandleInd> = (0..30)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.5, 1000.0))
            .collect();

        // Bar 28: bearish impulse — AVERAGE volume
        candles[28] = make_candle(100.0, 100.5, 95.0, 96.0, 1000.0);
        // Bar 29: absorption — AVERAGE volume (no spike anywhere)
        candles[29] = make_candle(96.0, 105.5, 95.5, 105.0, 1100.0);

        let config = default_config();
        let tf_p = tf15_params();
        let signals = detect_engulfing_patterns(&candles, &config, &tf_p, 10);
        assert!(signals.is_empty(), "No volume spike on either bar → no signal");
    }

    #[test]
    fn test_atr_dynamic_threshold() {
        // Test that ATR-based threshold works: small % move but > 1.5x ATR
        let mut candles: Vec<CandleInd> = (0..30)
            .map(|_| {
                let mut c = make_candle(100.0, 100.5, 99.5, 100.2, 1000.0);
                c.atr = 0.5; // ATR = 0.5% of price (low volatility)
                c
            })
            .collect();

        // Bar 28: BEARISH IMPULSE — body=1.0 (1%), which is > 1.5*0.5=0.75 ATR ✓
        // body = 1.0, range = 1.5, ratio = 0.67 > 0.5 ✓
        let mut imp = make_candle(100.0, 100.2, 98.5, 99.0, 2000.0);
        imp.atr = 0.5;
        candles[28] = imp;
        // Bar 29: BULLISH ABSORPTION
        candles[29] = make_candle(99.0, 101.0, 98.8, 100.5, 2000.0);

        let config = default_config();
        let tf_p = tf15_params();
        let signals = detect_engulfing_patterns(&candles, &config, &tf_p, 10);
        assert!(!signals.is_empty(),
            "ATR-based: body=1.0 >= ATR*1.5=0.75 should trigger despite only 1% move");
    }

    #[test]
    fn test_check_engulfing_at_bar_standalone() {
        // Test the standalone function used by backtest
        let mut candles: Vec<CandleInd> = (0..30)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.5, 1000.0))
            .collect();

        candles[28] = make_candle(100.0, 100.5, 95.0, 96.0, 2000.0);
        candles[29] = make_candle(96.0, 105.5, 95.5, 105.0, 2000.0);

        let config = default_config();
        let tf_p = tf15_params();
        let signal = check_engulfing_at_bar(&candles, 29, &config, &tf_p);
        assert!(signal.is_some(), "check_engulfing_at_bar should find signal at bar 29");
        assert_eq!(signal.unwrap().direction, SignalDir::Long);
    }

    #[test]
    fn test_feature_counts() {
        assert_eq!(RAW_INDICATOR_NAMES.len(), 33);
        assert_eq!(DERIVED_FEATURE_NAMES.len(), 19);
        assert_eq!(BODY_FEATURE_NAMES.len(), 4);
        assert_eq!(TEMPORAL_FEATURE_NAMES.len(), TEMPORAL_FEATURE_COUNT);
        assert_eq!(FEATURES_PER_CANDLE_BASE, 56);
        assert_eq!(FULL_FEATURES_PER_CANDLE, 79);
        assert_eq!(ENGULFING_FEATURE_NAMES.len(), ENGULFING_FEATURE_COUNT);
        assert_eq!(ENGULFING_FEATURE_COUNT, 9); // v3: was 8
    }

    #[test]
    fn test_all_feature_names() {
        let names = all_feature_names(ANALYSIS_TIMEFRAMES, 10);
        let expected = ANALYSIS_TIMEFRAMES.len() * 10 * FULL_FEATURES_PER_CANDLE
                       + ENGULFING_FEATURE_COUNT;
        assert_eq!(names.len(), expected);
    }

    #[test]
    fn test_config_default() {
        let cfg = ImpulseConfig::default();
        assert_eq!(cfg.max_hold, 8); // v3: was 5
        assert!((cfg.atr_multiplier - 1.5).abs() < 1e-12);
        assert!((cfg.open_leniency_pct - 0.001).abs() < 1e-12);
        assert!(cfg.params_for_tf(15).is_some());
        assert!(cfg.params_for_tf(60).is_some());
        assert!(cfg.params_for_tf(240).is_some());
        assert!(cfg.params_for_tf(1).is_none());
        // v3: lowered static floors
        assert!((cfg.params_for_tf(15).unwrap().impulse_pct - 1.0).abs() < 1e-12);
        assert!((cfg.params_for_tf(60).unwrap().impulse_pct - 1.5).abs() < 1e-12);
        assert!((cfg.params_for_tf(240).unwrap().impulse_pct - 3.5).abs() < 1e-12);
    }

    #[test]
    fn test_extract_features() {
        let candles: Vec<CandleInd> = (0..60)
            .map(|i| make_candle(100.0 + i as f64 * 0.1, 101.0 + i as f64 * 0.1,
                                  99.0 + i as f64 * 0.1, 100.5 + i as f64 * 0.1, 1000.0))
            .collect();

        let feats = extract_candle_features(&candles, 55);
        assert_eq!(feats.len(), FULL_FEATURES_PER_CANDLE);
        for (i, &v) in feats.iter().enumerate() {
            assert!(v.is_finite(), "Feature {} is not finite: {}", i, v);
        }
    }

    #[test]
    fn test_engulfing_features_1bar() {
        let mut candles: Vec<CandleInd> = (0..30)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.5, 1000.0))
            .collect();
        candles[28] = make_candle(100.0, 100.5, 95.0, 96.0, 2000.0);
        candles[29] = make_candle(96.0, 105.5, 95.5, 105.0, 2000.0);

        let signal = EngulfingSignal {
            bar_idx: 29, direction: SignalDir::Long, impulse_pct: 4.17,
            body_ratio: 0.73, volume_ratio: 2.0, absorption_bars: 1,
        };
        let feats = extract_engulfing_features(&candles, &signal);
        assert_eq!(feats.len(), ENGULFING_FEATURE_COUNT);
        assert!((feats[8] - 1.0).abs() < 1e-12, "absorption_bars should be 1.0 for 1-bar");
    }

    #[test]
    fn test_engulfing_features_2bar() {
        let mut candles: Vec<CandleInd> = (0..31)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.5, 1000.0))
            .collect();
        candles[27] = make_candle(100.0, 100.5, 95.0, 96.0, 2000.0); // impulse
        candles[28] = make_candle(96.0, 98.5, 95.5, 98.0, 1000.0);   // first absorb
        candles[29] = make_candle(98.0, 101.5, 97.5, 101.0, 1500.0); // completes

        let signal = EngulfingSignal {
            bar_idx: 29, direction: SignalDir::Long, impulse_pct: 4.17,
            body_ratio: 0.73, volume_ratio: 2.0, absorption_bars: 2,
        };
        let feats = extract_engulfing_features(&candles, &signal);
        assert_eq!(feats.len(), ENGULFING_FEATURE_COUNT);
        assert!((feats[8] - 2.0).abs() < 1e-12, "absorption_bars should be 2.0 for 2-bar");
    }
}
