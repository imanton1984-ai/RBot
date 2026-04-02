// strategies/ml_pump_dump/src/pump_dump.rs
//
// Core logic for Pump/Dump detection strategy.
//
// ARCHITECTURE:
//   1. PumpDumpConfig — all tunable parameters
//   2. CandleWithIndicators — candle + all indicator columns from DB
//   3. PumpDumpEvent — a detected anomalous move on the daily TF
//   4. MultiTfSnapshot — indicator readings at multiple TFs around the event
//   5. Feature extraction — per-candle + temporal + cross-TF features
//
// DETECTION FLOW:
//   Daily candles → find candle with high/low move ≥ threshold
//   → drill down to 4h/1h/15m/5m/1m to find the EXACT start
//   → extract features from N candles BEFORE the pump/dump onset
//
// INNOVATION:
//   Unlike the direction model which uses pure OHLCV patterns,
//   this model leverages ALL indicators because pumps/dumps have
//   distinct indicator signatures (volume spike, RSI divergence,
//   BB squeeze before explosion, etc.)

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═════════════════════════════════════════════════════════════════════════════

/// Timeframes used for multi-TF analysis, ordered from highest to lowest.
/// NOTE: 1m excluded — too noisy and insufficient historical data.
pub const ANALYSIS_TIMEFRAMES: &[i32] = &[1440, 240, 60, 15, 5];

/// Number of candles to look back BEFORE the pump/dump onset for feature extraction.
pub const PRE_EVENT_LOOKBACK: usize = 3;

/// Minimum candles required per symbol per TF to be useful.
pub const MIN_CANDLES_PER_TF: usize = 50;

/// Full configuration for the Pump/Dump detection model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PumpDumpConfig {
    /// Minimum daily candle move (%) to qualify as a pump/dump event.
    /// A candle where (high - low) / open >= threshold is considered anomalous.
    /// Env: PD_DAILY_THRESHOLD (default: 15.0)
    pub daily_threshold_pct: f64,

    /// Number of candles BEFORE the pump/dump to use for features.
    /// Each of these candles gets a full indicator snapshot.
    /// Env: PD_PRE_EVENT_LOOKBACK (default: 10)
    pub pre_event_lookback: usize,

    /// How many candles after the event to use for labeling negative examples.
    /// We need "normal" candles to contrast with pump/dump pre-patterns.
    /// Env: PD_NEGATIVE_RATIO (default: 3 — 3 negative per 1 positive)
    pub negative_ratio: usize,

    /// Prediction horizon: how many candles ahead the model predicts.
    /// e.g., predict pump will happen within next N 5-minute candles.
    /// Env: PD_PREDICTION_HORIZON (default: 3)
    pub prediction_horizon: usize,

    /// Minimum volume spike to consider a candle as part of pump.
    /// Env: PD_MIN_VOLUME_SPIKE (default: 2.0)
    pub min_volume_spike: f64,

    /// Minimum concentration: what fraction of the daily move must occur
    /// in 1-2 hourly candles to be considered a SHARP pump/dump.
    /// 0.50 = at least 50% of the daily move in one hourly candle.
    /// Env: PD_CONCENTRATION_PCT (default: 0.50)
    pub concentration_pct: f64,

    /// Maximum number of candles to hold on the target TF to reach the target move.
    /// If the ≥15% move is not reached within this many candles, the event is rejected.
    /// Env: PD_MAX_HOLD_CANDLES (default: 6)
    pub max_hold_candles: usize,
}

impl Default for PumpDumpConfig {
    fn default() -> Self {
        Self {
            daily_threshold_pct: 9.0,
            pre_event_lookback: PRE_EVENT_LOOKBACK,
            negative_ratio: 3,
            prediction_horizon: 3,
            min_volume_spike: 2.0,
            concentration_pct: 0.50,
            max_hold_candles: 6,
        }
    }
}

impl PumpDumpConfig {
    /// Load from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("PD_DAILY_THRESHOLD") {
            if let Ok(n) = v.parse::<f64>() { cfg.daily_threshold_pct = n.max(5.0); }
        }
        if let Ok(v) = std::env::var("PD_PRE_EVENT_LOOKBACK") {
            if let Ok(n) = v.parse::<usize>() { cfg.pre_event_lookback = n.clamp(3, 50); }
        }
        if let Ok(v) = std::env::var("PD_NEGATIVE_RATIO") {
            if let Ok(n) = v.parse::<usize>() { cfg.negative_ratio = n.clamp(1, 10); }
        }
        if let Ok(v) = std::env::var("PD_PREDICTION_HORIZON") {
            if let Ok(n) = v.parse::<usize>() { cfg.prediction_horizon = n.clamp(1, 12); }
        }
        if let Ok(v) = std::env::var("PD_MIN_VOLUME_SPIKE") {
            if let Ok(n) = v.parse::<f64>() { cfg.min_volume_spike = n.max(1.0); }
        }
        if let Ok(v) = std::env::var("PD_CONCENTRATION_PCT") {
            if let Ok(n) = v.parse::<f64>() { cfg.concentration_pct = n.clamp(0.2, 0.95); }
        }
        if let Ok(v) = std::env::var("PD_MAX_HOLD_CANDLES") {
            if let Ok(n) = v.parse::<usize>() { cfg.max_hold_candles = n.clamp(2, 20); }
        }

        cfg
    }

    /// Get move threshold — same for all TFs.
    /// A 15% pump is 15% on every timeframe, just seen through different lenses.
    pub fn threshold_for_tf(&self, _tf_minutes: i32) -> f64 {
        self.daily_threshold_pct
    }

    /// Print config summary.
    pub fn log_summary(&self) {
        tracing::info!("PumpDump Config:");
        tracing::info!("  threshold: {:.1}% (same across all TFs)", self.daily_threshold_pct);
        tracing::info!("  pre_event_lookback: {} candles", self.pre_event_lookback);
        tracing::info!("  negative_ratio: {}:1", self.negative_ratio);
        tracing::info!("  prediction_horizon: {} candles", self.prediction_horizon);
        tracing::info!("  concentration_pct: {:.0}% (of daily move in 1 hourly candle)", self.concentration_pct * 100.0);
        tracing::info!("  max_hold_candles: {} (max candles to reach target)", self.max_hold_candles);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// DATA TYPES
// ═════════════════════════════════════════════════════════════════════════════

/// Direction of the anomalous move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    Pump,  // anomalous upward move
    Dump,  // anomalous downward move
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::Pump => write!(f, "PUMP"),
            EventType::Dump => write!(f, "DUMP"),
        }
    }
}

/// A detected pump/dump event on the daily chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PumpDumpEvent {
    pub symbol: String,
    pub event_type: EventType,
    /// Time of the daily candle that triggered detection.
    pub daily_time: DateTime<Utc>,
    /// Magnitude of the move on the daily candle (%).
    pub daily_move_pct: f64,
    /// The exact onset time refined through lower TFs (best estimate).
    /// This is the time of the first candle on the lowest available TF
    /// that shows the beginning of the anomalous move.
    pub onset_time: DateTime<Utc>,
    /// Which TF we were able to drill down to.
    pub finest_tf: i32,
    /// Move magnitudes at each TF where the pump/dump was confirmed.
    pub tf_moves: HashMap<i32, f64>,
}

/// A single candle row with all indicator columns from DB.
/// Mirrors the structure in ml_entry_strategy::dataset but kept independent
/// to avoid coupling with running bot code.
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
    // Raw indicators from market.indicators_wide
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
    /// Extract ALL raw indicator values in canonical order (33 features).
    pub fn raw_indicator_values(&self) -> Vec<f64> {
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

    /// Compute normalized/derived features from raw indicators (19 features).
    pub fn derived_features(&self) -> Vec<f64> {
        let close = self.close;
        let safe_div = |a: f64, b: f64| if b.abs() > 1e-12 { a / b } else { 0.0 };

        let bb_range = self.bb_upper - self.bb_lower;
        let bb_position = if bb_range.abs() > 1e-12 {
            (close - self.bb_lower) / bb_range
        } else {
            0.5
        };

        vec![
            self.rsi / 100.0,                                       // rsi_norm
            self.cci / 200.0,                                        // cci_norm
            self.stoch_k / 100.0,                                    // stoch_norm
            (self.williams + 100.0) / 100.0,                         // williams_norm
            bb_position,                                              // bb_position
            safe_div(bb_range, close) * 100.0,                       // bb_width_pct
            safe_div(self.atr, close) * 100.0,                       // atr_pct
            safe_div(close - self.sma, close) * 100.0,               // price_vs_sma
            safe_div(close - self.ema_20, close) * 100.0,            // price_vs_ema20
            safe_div(close - self.ema_50, close) * 100.0,            // price_vs_ema50
            safe_div(close - self.ema_200, close) * 100.0,           // price_vs_ema200
            safe_div(close - self.vwap, close) * 100.0,              // price_vs_vwap
            safe_div(self.macd_hist, close) * 1000.0,                // macd_norm
            0.0,                                                      // obv_change_pct (needs history)
            if self.volume_spike > 2.0 { 1.0 } else { 0.0 },       // volume_spike_flag
            self.mfi / 100.0,                                        // mfi_norm
            safe_div(close - self.fibo_pivot, close) * 100.0,        // price_vs_fibo_pivot
            safe_div(close - self.supertrend, close) * 100.0,        // price_vs_supertrend
            safe_div(self.alligator_jaw - self.alligator_lips, close) * 100.0, // alligator_spread
        ]
    }

    /// Candle body features (4 features).
    pub fn body_features(&self) -> Vec<f64> {
        let range = self.high - self.low;
        let body = (self.close - self.open).abs();
        let safe_div = |a: f64, b: f64| if b.abs() > 1e-12 { a / b } else { 0.0 };

        vec![
            safe_div(body, range),                                          // body_ratio
            safe_div(self.high - self.close.max(self.open), range),         // upper_wick_ratio
            safe_div(self.close.min(self.open) - self.low, range),          // lower_wick_ratio
            safe_div(self.close - self.open, self.open) * 100.0,           // candle_return_pct
        ]
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// TIME CONVENTION HELPERS
// ═════════════════════════════════════════════════════════════════════════════

/// DB stores candle **close** times (e.g. 1h candle 00:00–01:00 → time=00:59:59.999).
/// This function converts close_time → open_time for a given TF.
pub fn candle_open_time(close_time: DateTime<Utc>, tf_minutes: i32) -> DateTime<Utc> {
    close_time - Duration::minutes(tf_minutes as i64) + Duration::milliseconds(1)
}

/// Compute the search window for lower-TF candles contained within a parent candle.
///
/// Given a parent candle's **close time** and TF duration, returns (search_start, search_end)
/// where search_start/search_end are close-time boundaries for lower TF candles.
///
/// Example: daily candle close = 2025-10-09 23:59:59.999
///   → open = 2025-10-09 00:00:00.000
///   → hourly candles within: close from 2025-10-09 00:59:59.999 to 2025-10-09 23:59:59.999
///   → search_start = open of day, search_end = close of day + 1ms (for inclusive partition_point)
pub fn parent_candle_window(
    parent_close_time: DateTime<Utc>,
    parent_tf_minutes: i32,
) -> (DateTime<Utc>, DateTime<Utc>) {
    let open_time = candle_open_time(parent_close_time, parent_tf_minutes);
    // search_start: looking for child candles with close >= open_time
    // For partition_point(c.time < X): X = open_time → first index with time >= open_time
    // search_end: include the parent_close_time itself
    // For partition_point(c.time < X): X = parent_close_time + 1ms → includes parent_close_time
    (open_time, parent_close_time + Duration::milliseconds(1))
}

// ═════════════════════════════════════════════════════════════════════════════
// PUMP/DUMP DETECTION
// ═════════════════════════════════════════════════════════════════════════════

/// Detect pump/dump events on the daily chart (stage 1).
///
/// A candle qualifies as anomalous if:
///   - (high - open) / open >= threshold (pump) OR (open - low) / open >= threshold (dump)
///   - AND the body confirms direction (close > open for pump, close < open for dump)
///
/// These are candidates — Stage 2 (validate_sharp_move) verifies the move was SHARP
/// (concentrated in 1-2 hourly candles, not a gradual 5-6 candle drift).
pub fn detect_anomalous_candles(
    candles: &[CandleInd],
    threshold_pct: f64,
) -> Vec<(usize, EventType, f64)> {
    let threshold = threshold_pct / 100.0;
    let mut events = Vec::new();

    for (i, c) in candles.iter().enumerate() {
        if c.open.abs() < 1e-12 { continue; }

        // Pump: candle shows a strong upward move
        let up_move = (c.high - c.open) / c.open;
        let body_up = (c.close - c.open) / c.open;
        if up_move >= threshold && body_up > 0.0 {
            events.push((i, EventType::Pump, up_move * 100.0));
            continue;
        }

        // Dump: candle shows a strong downward move
        let down_move = (c.open - c.low) / c.open;
        let body_down = (c.open - c.close) / c.open;
        if down_move >= threshold && body_down > 0.0 {
            events.push((i, EventType::Dump, down_move * 100.0));
            continue;
        }

        // Also check by body directly (close-to-open)
        if body_up.abs() >= threshold {
            events.push((i, EventType::Pump, body_up.abs() * 100.0));
        } else if body_down.abs() >= threshold {
            events.push((i, EventType::Dump, body_down.abs() * 100.0));
        }
    }

    events
}

/// Validate that a pump/dump was SHARP — concentrated in 1-2 hourly candles.
///
/// PHILOSOPHY: We want flash pumps/dumps, not gradual 5-6 hour drifts.
/// A true pump/dump has most of the daily move happening within 1-2 hours.
///
/// NOTE: `daily_candle_close` is the **close time** of the daily candle (DB convention).
/// The search window is computed to cover the CORRECT day's hourly candles.
///
/// # Returns
/// `Some((hourly_idx, hourly_move_pct))` if the move is sharp, `None` otherwise.
pub fn validate_sharp_move(
    hourly_candles: &[CandleInd],
    daily_candle_close: DateTime<Utc>,
    event_type: EventType,
    daily_move_pct: f64,
    concentration_pct: f64,
) -> Option<(usize, f64)> {
    // Compute correct search window using close-time convention
    let (search_start, search_end) = parent_candle_window(daily_candle_close, 1440);

    // Find hourly candles within the daily window
    let start_idx = hourly_candles.partition_point(|c| c.time < search_start);
    let end_idx = hourly_candles.partition_point(|c| c.time < search_end);

    if start_idx >= end_idx || end_idx > hourly_candles.len() {
        return None;
    }

    // Find the single 1H candle with the biggest move in the correct direction
    let mut best_idx: Option<usize> = None;
    let mut best_move: f64 = 0.0;

    // Also track the max 2-consecutive-candle move
    let mut best_2candle_move: f64 = 0.0;
    let mut best_2candle_start: Option<usize> = None;

    for idx in start_idx..end_idx {
        let c = &hourly_candles[idx];
        if c.open.abs() < 1e-12 { continue; }

        // Single candle move (body-based, not just wick)
        let move_pct = match event_type {
            EventType::Pump => {
                let body_move = (c.close - c.open) / c.open * 100.0;
                let wick_move = (c.high - c.open) / c.open * 100.0;
                body_move.max(wick_move * 0.7) // penalize wick-only moves
            }
            EventType::Dump => {
                let body_move = (c.open - c.close) / c.open * 100.0;
                let wick_move = (c.open - c.low) / c.open * 100.0;
                body_move.max(wick_move * 0.7)
            }
        };

        if move_pct > best_move {
            best_move = move_pct;
            best_idx = Some(idx);
        }

        // Two consecutive candles
        if idx + 1 < end_idx {
            let c2 = &hourly_candles[idx + 1];
            if c2.open.abs() < 1e-12 { continue; }
            let two_candle_move = match event_type {
                EventType::Pump => {
                    let combined_open = c.open;
                    let combined_high = c.high.max(c2.high);
                    (combined_high - combined_open) / combined_open * 100.0
                }
                EventType::Dump => {
                    let combined_open = c.open;
                    let combined_low = c.low.min(c2.low);
                    (combined_open - combined_low) / combined_open * 100.0
                }
            };
            if two_candle_move > best_2candle_move {
                best_2candle_move = two_candle_move;
                best_2candle_start = Some(idx);
            }
        }
    }

    let daily_abs = daily_move_pct.abs();
    if daily_abs < 1e-6 { return None; }

    // Check concentration: single candle has >= concentration_pct of the daily move
    if best_move >= daily_abs * concentration_pct {
        return best_idx.map(|idx| (idx, best_move));
    }

    // Fallback: check 2-candle concentration (allow slightly more tolerance)
    if best_2candle_move >= daily_abs * concentration_pct * 0.9 {
        return best_2candle_start.map(|idx| (idx, best_2candle_move));
    }

    // Move was NOT sharp — gradual drift across many hours
    None
}

/// Search for a pump/dump window of ≤max_hold consecutive candles where the
/// cumulative move reaches ≥target_pct% on the given TF.
///
/// This is the core function for finding the pump/dump event on lower TFs.
/// Instead of looking for a single candle with a big move, it searches for
/// the FIRST window of consecutive candles where the cumulative move reaches
/// the target (e.g., 15% within 5-6 candles).
///
/// NOTE: `search_start` and `search_end` are close-time boundaries.
/// Use `parent_candle_window()` to compute them from a parent candle's close time.
///
/// # Arguments
/// * `candles` — sorted by time ASC
/// * `search_start` — earliest close_time to consider
/// * `search_end` — latest close_time to consider (exclusive for partition_point)
/// * `event_type` — Pump or Dump
/// * `target_pct` — minimum cumulative move % (e.g. 15.0)
/// * `max_hold` — maximum candles in the window (e.g. 6)
/// * `min_lookback` — onset must have at least this many candles before it
///
/// # Returns
/// `Some((onset_idx, actual_move_pct, candles_needed))` or `None`.
pub fn find_pump_window(
    candles: &[CandleInd],
    search_start: DateTime<Utc>,
    search_end: DateTime<Utc>,
    event_type: EventType,
    target_pct: f64,
    max_hold: usize,
    min_lookback: usize,
) -> Option<(usize, f64, usize)> {
    let threshold = target_pct / 100.0;

    let start_idx = candles.partition_point(|c| c.time < search_start);
    let end_idx = candles.partition_point(|c| c.time < search_end);

    if start_idx >= end_idx { return None; }

    // Onset must have enough lookback for feature extraction
    let onset_min = start_idx.max(min_lookback);

    for onset in onset_min..end_idx {
        let entry = candles[onset].open;
        if entry.abs() < 1e-12 { continue; }

        let window_end = (onset + max_hold).min(candles.len());
        let mut max_high = f64::MIN;
        let mut min_low = f64::MAX;

        for i in onset..window_end {
            max_high = max_high.max(candles[i].high);
            min_low = min_low.min(candles[i].low);

            let move_frac = match event_type {
                EventType::Pump => (max_high - entry) / entry,
                EventType::Dump => (entry - min_low) / entry,
            };

            if move_frac >= threshold {
                let candles_needed = i - onset + 1;
                return Some((onset, move_frac * 100.0, candles_needed));
            }
        }
    }

    None
}

/// Given a daily pump/dump event, find the corresponding candle(s) on a lower TF.
///
/// Searches within `search_start..search_end` (close-time boundaries) for the
/// candle with the biggest move in the event direction.
///
/// Returns `(index_of_onset_candle, move_pct)` or `None` if threshold not met.
pub fn find_event_on_lower_tf(
    lower_tf_candles: &[CandleInd],
    search_start: DateTime<Utc>,
    search_end: DateTime<Utc>,
    event_type: EventType,
    threshold_pct: f64,
) -> Option<(usize, f64)> {
    let threshold = threshold_pct / 100.0;

    let start_idx = lower_tf_candles.partition_point(|c| c.time < search_start);
    let end_idx = lower_tf_candles.partition_point(|c| c.time < search_end);

    if start_idx >= end_idx || end_idx > lower_tf_candles.len() {
        return None;
    }

    let mut best_idx = None;
    let mut best_move: f64 = 0.0;

    for idx in start_idx..end_idx {
        let c = &lower_tf_candles[idx];
        if c.open.abs() < 1e-12 { continue; }

        let move_pct = match event_type {
            EventType::Pump => (c.high - c.open) / c.open,
            EventType::Dump => (c.open - c.low) / c.open,
        };

        if move_pct > best_move {
            best_move = move_pct;
            best_idx = Some(idx);
        }
    }

    // Only return if the threshold is actually met
    if best_move >= threshold {
        best_idx.map(|idx| (idx, best_move * 100.0))
    } else {
        None
    }
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

/// Number of features per candle (raw + derived + body).
pub const FEATURES_PER_CANDLE: usize = 33 + 19 + 4; // 56

/// Compute temporal/dynamic features for candle at index `t`.
/// These capture how indicators are changing over recent bars.
///
/// Returns a vector of temporal features (26 features).
pub fn compute_temporal_features(candles: &[CandleInd], t: usize) -> Vec<f64> {
    let safe_div = |a: f64, b: f64| -> f64 {
        if b.abs() > 1e-12 { a / b } else { 0.0 }
    };

    let cur = &candles[t];
    let close = cur.close;
    let mut feats = Vec::with_capacity(TEMPORAL_FEATURE_COUNT);

    // Price momentum at different lookbacks (4 features)
    for &lb in &[1usize, 3, 5, 10] {
        let ret = if t >= lb && candles[t - lb].close.abs() > 1e-12 {
            (close - candles[t - lb].close) / candles[t - lb].close * 100.0
        } else {
            0.0
        };
        feats.push(ret);
    }

    // Volume dynamics (3 features)
    let vol = cur.volume;
    let vol_avg_5 = if t >= 5 {
        let s: f64 = (1..=5).map(|j| candles[t - j].volume).sum();
        s / 5.0
    } else { vol };
    feats.push(if vol_avg_5 > 1e-12 { (vol / vol_avg_5).clamp(0.01, 50.0) } else { 1.0 });

    let vol_avg_10 = if t >= 10 {
        let s: f64 = (1..=10).map(|j| candles[t - j].volume).sum();
        s / 10.0
    } else { vol };
    feats.push(if vol_avg_10 > 1e-12 { (vol / vol_avg_10).clamp(0.01, 50.0) } else { 1.0 });

    let vol_roc_1 = if t >= 1 && candles[t - 1].volume > 1e-12 {
        (vol / candles[t - 1].volume - 1.0).clamp(-10.0, 10.0)
    } else { 0.0 };
    feats.push(vol_roc_1);

    // RSI dynamics (2 features)
    let rsi_slope_3 = if t >= 3 { (cur.rsi - candles[t - 3].rsi) / 100.0 } else { 0.0 };
    let rsi_slope_10 = if t >= 10 { (cur.rsi - candles[t - 10].rsi) / 100.0 } else { 0.0 };
    feats.push(rsi_slope_3);
    feats.push(rsi_slope_10);

    // ATR expansion/contraction (2 features)
    let atr_ratio_5 = if t >= 5 && candles[t - 5].atr.abs() > 1e-12 {
        cur.atr / candles[t - 5].atr - 1.0
    } else { 0.0 };
    let atr_ratio_10 = if t >= 10 && candles[t - 10].atr.abs() > 1e-12 {
        cur.atr / candles[t - 10].atr - 1.0
    } else { 0.0 };
    feats.push(atr_ratio_5);
    feats.push(atr_ratio_10);

    // MACD histogram momentum (2 features)
    let macd_hist_roc_1 = if t >= 1 {
        safe_div(cur.macd_hist - candles[t - 1].macd_hist, close) * 1000.0
    } else { 0.0 };
    let macd_hist_roc_3 = if t >= 3 {
        safe_div(cur.macd_hist - candles[t - 3].macd_hist, close) * 1000.0
    } else { 0.0 };
    feats.push(macd_hist_roc_1);
    feats.push(macd_hist_roc_3);

    // Trend persistence (2 features)
    let trend_persist_5 = if t >= 5 {
        (0..5).map(|j| candles[t - j].trend).sum::<f64>() / 5.0
    } else { 0.0 };
    let supertrend_persist_10 = if t >= 10 {
        (0..10).map(|j| candles[t - j].supertrend_dir).sum::<f64>() / 10.0
    } else { 0.0 };
    feats.push(trend_persist_5);
    feats.push(supertrend_persist_10);

    // BB squeeze percentile (1 feature)
    let bb_squeeze_pctl = if t >= 50 {
        let cur_bb_width = safe_div(cur.bb_upper - cur.bb_lower, close) * 100.0;
        let mut count_below = 0usize;
        for j in 1..=50 {
            let c = &candles[t - j];
            let w = safe_div(c.bb_upper - c.bb_lower, c.close) * 100.0;
            if w < cur_bb_width { count_below += 1; }
        }
        count_below as f64 / 50.0
    } else { 0.5 };
    feats.push(bb_squeeze_pctl);

    // OBV divergence (1 feature)
    let obv_div = if t >= 10 {
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
    } else { 0.0 };
    feats.push(obv_div);

    // Price acceleration (1 feature)
    let ret_recent_3 = if t >= 3 { safe_div(close - candles[t - 3].close, close) * 100.0 } else { 0.0 };
    let ret_prev_3 = if t >= 6 {
        safe_div(candles[t - 3].close - candles[t - 6].close, candles[t - 3].close) * 100.0
    } else { 0.0 };
    feats.push(ret_recent_3 - ret_prev_3);

    // Volume up vs down ratio (1 feature)
    let vol_up_down = if t >= 10 {
        let (mut vu, mut vd) = (0.0f64, 0.0f64);
        for j in 0..10 {
            let c = &candles[t - j];
            if c.close >= c.open { vu += c.volume; } else { vd += c.volume; }
        }
        if vd > 1e-12 { (vu / vd).clamp(0.1, 10.0) }
        else if vu > 1e-12 { 10.0 }
        else { 1.0 }
    } else { 1.0 };
    feats.push(vol_up_down);

    // Distance to 50-bar extremes (2 features)
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

    // Wick rejection pressure (1 feature)
    let wick_pressure = if t >= 3 {
        let mut ws = 0.0;
        for j in 0..3 {
            let c = &candles[t - j];
            let uw = c.high - c.close.max(c.open);
            let lw = c.close.min(c.open) - c.low;
            let a = if c.atr > 1e-12 { c.atr } else { 1.0 };
            ws += safe_div(lw - uw, a);
        }
        ws / 3.0
    } else { 0.0 };
    feats.push(wick_pressure);

    // EMA convergence change (1 feature)
    let ema_conv_change = if t >= 5 {
        let gap_now = cur.ema_20 - cur.ema_50;
        let gap_prev = candles[t - 5].ema_20 - candles[t - 5].ema_50;
        safe_div(gap_now - gap_prev, close) * 100.0
    } else { 0.0 };
    feats.push(ema_conv_change);

    debug_assert_eq!(feats.len(), TEMPORAL_FEATURE_COUNT,
        "Temporal features count mismatch: expected {}, got {}", TEMPORAL_FEATURE_COUNT, feats.len());
    feats
}

/// Temporal feature names (26).
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

/// Total features per candle snapshot = raw(33) + derived(19) + body(4) + temporal(23) = 79.
pub const FULL_FEATURES_PER_CANDLE: usize = FEATURES_PER_CANDLE + TEMPORAL_FEATURE_COUNT;

/// Extract full feature vector for a single candle at index `t`.
///
/// Returns a vector of FULL_FEATURES_PER_CANDLE (82) elements.
pub fn extract_candle_features(candles: &[CandleInd], t: usize) -> Vec<f64> {
    let c = &candles[t];
    let mut feats = Vec::with_capacity(FULL_FEATURES_PER_CANDLE);

    feats.extend(c.raw_indicator_values());   // 33
    feats.extend(c.derived_features());        // 19
    feats.extend(c.body_features());           // 4
    feats.extend(compute_temporal_features(candles, t)); // 26

    debug_assert_eq!(feats.len(), FULL_FEATURES_PER_CANDLE);
    feats
}

/// Generate feature names for one candle snapshot, prefixed with TF and candle offset.
///
/// Example: "tf15_c0_rsi", "tf15_c0_cci", ..., "tf15_c9_ema_convergence_change"
///
/// # Arguments
/// * `tf_minutes` — timeframe label
/// * `candle_offset` — offset from the event (0 = furthest back, N-1 = closest to event)
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

/// Generate ALL feature names for the multi-TF pre-event snapshot.
///
/// Structure: for each TF (from highest to lowest), for each of the
/// pre_event_lookback candles, emit FULL_FEATURES_PER_CANDLE feature names.
///
/// Total = num_tfs × pre_event_lookback × FULL_FEATURES_PER_CANDLE
pub fn all_feature_names(
    timeframes: &[i32],
    pre_event_lookback: usize,
) -> Vec<String> {
    let total = timeframes.len() * pre_event_lookback * FULL_FEATURES_PER_CANDLE;
    let mut names = Vec::with_capacity(total);

    for &tf in timeframes {
        for c in 0..pre_event_lookback {
            names.extend(prefixed_feature_names(tf, c));
        }
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

    #[test]
    fn test_detect_pump() {
        let mut candles: Vec<CandleInd> = (0..10)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.5, 1000.0))
            .collect();

        // Candle 5 is a pump: opens at 100, goes to 120 = 20%
        candles[5] = make_candle(100.0, 120.0, 99.0, 118.0, 5000.0);

        let events = detect_anomalous_candles(&candles, 15.0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, 5);
        assert_eq!(events[0].1, EventType::Pump);
        assert!(events[0].2 >= 19.0); // ~20%
    }

    #[test]
    fn test_detect_dump() {
        let mut candles: Vec<CandleInd> = (0..10)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.5, 1000.0))
            .collect();

        // Candle 3 is a dump: opens at 100, drops to 82 = 18%
        candles[3] = make_candle(100.0, 101.0, 82.0, 84.0, 5000.0);

        let events = detect_anomalous_candles(&candles, 15.0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, 3);
        assert_eq!(events[0].1, EventType::Dump);
        assert!(events[0].2 >= 17.0);
    }

    #[test]
    fn test_no_events() {
        let candles: Vec<CandleInd> = (0..10)
            .map(|_| make_candle(100.0, 101.0, 99.0, 100.5, 1000.0))
            .collect();

        let events = detect_anomalous_candles(&candles, 15.0);
        assert!(events.is_empty());
    }

    #[test]
    fn test_feature_count() {
        assert_eq!(RAW_INDICATOR_NAMES.len(), 33);
        assert_eq!(DERIVED_FEATURE_NAMES.len(), 19);
        assert_eq!(BODY_FEATURE_NAMES.len(), 4);
        assert_eq!(TEMPORAL_FEATURE_NAMES.len(), TEMPORAL_FEATURE_COUNT);
        assert_eq!(FEATURES_PER_CANDLE, 56);
        // 56 (raw+derived+body) + 23 (temporal) = 79
        assert_eq!(FULL_FEATURES_PER_CANDLE, 79);
    }

    #[test]
    fn test_prefixed_names() {
        let names = prefixed_feature_names(15, 5);
        assert_eq!(names.len(), FULL_FEATURES_PER_CANDLE);
        assert_eq!(names[0], "tf15_c5_rsi");
        assert_eq!(names[32], "tf15_c5_cmf");
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
    fn test_config_default() {
        let cfg = PumpDumpConfig::default();
        assert!((cfg.daily_threshold_pct - 9.0).abs() < 1e-6);
        assert_eq!(cfg.pre_event_lookback, 3);
        assert_eq!(cfg.negative_ratio, 3);
        assert!((cfg.concentration_pct - 0.50).abs() < 1e-6);
        assert_eq!(cfg.max_hold_candles, 6);
        // Threshold is the same for all TFs
        assert_eq!(cfg.threshold_for_tf(60), cfg.threshold_for_tf(1440));
    }

    #[test]
    fn test_all_feature_names() {
        let tfs = &[1440, 240, 60];
        let lookback = 5;
        let names = all_feature_names(tfs, lookback);
        assert_eq!(names.len(), 3 * 5 * FULL_FEATURES_PER_CANDLE);
    }

    #[test]
    fn test_candle_open_time() {
        use chrono::TimeZone;
        // Daily candle close = 2025-10-09 23:59:59.999
        let close = Utc.with_ymd_and_hms(2025, 10, 9, 23, 59, 59).unwrap()
            + Duration::milliseconds(999);
        let open = candle_open_time(close, 1440);
        assert_eq!(open.format("%Y-%m-%d %H:%M:%S").to_string(), "2025-10-09 00:00:00");

        // Hourly candle close = 2025-10-09 05:59:59.999
        let close_h = Utc.with_ymd_and_hms(2025, 10, 9, 5, 59, 59).unwrap()
            + Duration::milliseconds(999);
        let open_h = candle_open_time(close_h, 60);
        assert_eq!(open_h.format("%Y-%m-%d %H:%M:%S").to_string(), "2025-10-09 05:00:00");
    }

    #[test]
    fn test_parent_candle_window() {
        use chrono::TimeZone;
        let daily_close = Utc.with_ymd_and_hms(2025, 10, 9, 23, 59, 59).unwrap()
            + Duration::milliseconds(999);
        let (start, end) = parent_candle_window(daily_close, 1440);
        // start should be at or before first hourly close of the day
        assert!(start <= Utc.with_ymd_and_hms(2025, 10, 9, 0, 59, 59).unwrap()
            + Duration::milliseconds(999));
        // end should be just past the daily close
        assert!(end > daily_close);
    }

    #[test]
    fn test_find_event_on_lower_tf_threshold() {
        use chrono::TimeZone;
        // Create candles with small moves (< 15%)
        let mut candles: Vec<CandleInd> = (0..10)
            .map(|i| {
                let mut c = make_candle(100.0, 103.0, 98.0, 101.0, 1000.0);
                c.time = Utc.with_ymd_and_hms(2025, 10, 9, i, 59, 59).unwrap()
                    + Duration::milliseconds(999);
                c
            })
            .collect();

        let search_start = Utc.with_ymd_and_hms(2025, 10, 9, 0, 0, 0).unwrap();
        let search_end = Utc.with_ymd_and_hms(2025, 10, 10, 0, 0, 0).unwrap();

        // Should return None — no candle has 15%+ move
        let result = find_event_on_lower_tf(&candles, search_start, search_end, EventType::Pump, 15.0);
        assert!(result.is_none(), "Should return None when no candle meets threshold");

        // Add a candle with 20% pump
        candles[5] = make_candle(100.0, 120.0, 99.0, 118.0, 5000.0);
        candles[5].time = Utc.with_ymd_and_hms(2025, 10, 9, 5, 59, 59).unwrap()
            + Duration::milliseconds(999);

        let result = find_event_on_lower_tf(&candles, search_start, search_end, EventType::Pump, 15.0);
        assert!(result.is_some(), "Should find the 20% pump candle");
    }
}
