// strategies/ml_entry_strategy/src/heuristic.rs
//
// Cross-TF Heuristic Direction Filter
//
// Shared between backtest and production pipeline.
// Uses weighted voting from current, higher, and lower TF indicators
// to confirm or reject ML-predicted direction.
//
// MODES:
//   Filter (default) — reject signals where heuristic disagrees with ML direction
//   Override — replace ML direction with heuristic (TESTED: hurts performance)
//   Off — pure ML direction, no cross-TF filter
//
// BACKTEST RESULTS (1h TF):
//   filter mode:  WR ~67.4% (best)
//   override:     WR ~45% (terrible — DO NOT USE)
//   off:          WR ~63.9%
//
// ENV VARS:
//   HEURISTIC_DIR_MODE=filter         — "filter" (default), "override", "off"
//   HEURISTIC_DIR_CONFIDENCE=0.3      — min confidence to apply filter/override
//
// ARCHITECTURE:
//   In production (super_entry_stage.rs) we maintain a CrossTfStore that stores
//   the LATEST candle per (symbol, tf). Every FeatureSnapshot updates the store
//   (even for non-active TFs), so when a signal is generated for e.g. 15m,
//   we can look up the latest 60m (higher) and 5m (lower) candles.

use std::collections::HashMap;
use crate::dataset::CandleWithIndicators;
use crate::signal_generator::SuperEntrySignal;
use crate::config::SuperEntryConfig;

// ─────────────────────────────────────────────────────────────────────
// TF Hierarchy
// ─────────────────────────────────────────────────────────────────────

/// Get the next higher timeframe for cross-TF lookups
pub fn get_higher_tf(tf: i32) -> Option<i32> {
    match tf {
        1 => Some(5),
        5 => Some(15),
        15 => Some(60),
        60 => Some(240),
        240 => Some(1440),
        _ => None,
    }
}

/// Get the next lower timeframe for cross-TF lookups
pub fn get_lower_tf(tf: i32) -> Option<i32> {
    match tf {
        5 => Some(1),
        15 => Some(5),
        60 => Some(15),
        240 => Some(60),
        1440 => Some(240),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Heuristic Mode
// ─────────────────────────────────────────────────────────────────────

/// Heuristic direction mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeuristicMode {
    /// Replace ML direction with heuristic when confident (TESTED: BAD)
    Override,
    /// Reject signal if heuristic disagrees with ML direction (RECOMMENDED)
    Filter,
    /// Disable heuristic — use pure ML direction
    Off,
}

impl HeuristicMode {
    /// Parse from string (env var value)
    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "override" => Self::Override,
            "off" | "disabled" | "none" => Self::Off,
            _ => Self::Filter, // default
        }
    }

    /// Load from environment variable HEURISTIC_DIR_MODE
    pub fn from_env() -> Self {
        let raw = std::env::var("HEURISTIC_DIR_MODE")
            .unwrap_or_else(|_| "filter".to_string());
        Self::from_str_lossy(&raw)
    }
}

/// Default heuristic min confidence from env or 0.3
pub fn heuristic_min_confidence_from_env() -> f32 {
    std::env::var("HEURISTIC_DIR_CONFIDENCE")
        .unwrap_or_else(|_| "0.3".to_string())
        .parse()
        .unwrap_or(0.3)
}

// ─────────────────────────────────────────────────────────────────────
// Heuristic Result
// ─────────────────────────────────────────────────────────────────────

/// Result of heuristic direction computation
#[derive(Debug, Clone, Copy)]
pub struct HeuristicResult {
    /// Heuristic direction: 1 = LONG, -1 = SHORT
    pub direction: i8,
    /// Confidence: 0.0..1.0 (|weighted_score| / max_points)
    pub confidence: f32,
    /// Number of TF sources used (1=current only, 2=+higher, 3=+lower)
    pub sources_used: u8,
}

// ─────────────────────────────────────────────────────────────────────
// Cross-TF Store (Production)
// ─────────────────────────────────────────────────────────────────────

/// Cross-TF data store for production pipeline.
/// Stores the LATEST candle per (symbol, tf) for heuristic lookups.
///
/// Unlike the backtest MultiTfStore (which stores full history for binary search),
/// production only needs the most recent candle per (symbol, tf).
///
/// Updated on every incoming FeatureSnapshot (before the production_tfs filter),
/// so data is available for ALL TFs, not just active ones.
pub struct CrossTfStore {
    /// (symbol, tf_minutes) -> latest CandleWithIndicators
    latest: HashMap<(String, i32), CandleWithIndicators>,
}

impl CrossTfStore {
    pub fn new() -> Self {
        Self { latest: HashMap::new() }
    }

    /// Update the store with a new candle. Only replaces if newer.
    pub fn update(&mut self, candle: CandleWithIndicators, tf_minutes: i32) {
        let key = (candle.symbol.clone(), tf_minutes);
        // Only update if this candle is newer than what we have
        if let Some(existing) = self.latest.get(&key) {
            if candle.time <= existing.time {
                return;
            }
        }
        self.latest.insert(key, candle);
    }

    /// Get the latest candle for (symbol, tf)
    pub fn get_latest(&self, symbol: &str, tf: i32) -> Option<&CandleWithIndicators> {
        self.latest.get(&(symbol.to_string(), tf))
    }

    /// Number of entries in the store
    pub fn len(&self) -> usize {
        self.latest.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.latest.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Multi-TF Store (Backtest) — for binary search over full history
// ─────────────────────────────────────────────────────────────────────

/// Store for cross-TF candle lookups in backtest mode.
/// Pre-loads all TFs so we can quickly find the higher/lower TF candle
/// at any given timestamp for any symbol.
pub struct MultiTfStore {
    /// tf_minutes -> symbol -> sorted Vec<CandleWithIndicators>
    data: HashMap<i32, HashMap<String, Vec<CandleWithIndicators>>>,
}

impl MultiTfStore {
    pub fn new() -> Self {
        Self { data: HashMap::new() }
    }

    pub fn insert_tf(&mut self, tf: i32, grouped: HashMap<String, Vec<CandleWithIndicators>>) {
        self.data.insert(tf, grouped);
    }

    /// Find the latest candle for (symbol, tf) at time <= target_time.
    /// Uses binary search for O(log n) lookup.
    pub fn find_candle_at(
        &self,
        symbol: &str,
        tf: i32,
        target_time: chrono::DateTime<chrono::Utc>,
    ) -> Option<&CandleWithIndicators> {
        let tf_data = self.data.get(&tf)?;
        let candles = tf_data.get(symbol)?;
        if candles.is_empty() {
            return None;
        }
        // Binary search: find rightmost candle with time <= target_time
        let idx = candles.partition_point(|c| c.time <= target_time);
        if idx > 0 {
            Some(&candles[idx - 1])
        } else {
            None
        }
    }

    /// Get candles for a specific TF (all symbols)
    pub fn get_tf(&self, tf: i32) -> Option<&HashMap<String, Vec<CandleWithIndicators>>> {
        self.data.get(&tf)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Heuristic Direction Computation
// ─────────────────────────────────────────────────────────────────────

/// Sign of a float: +1, -1, or 0 (with 0.01 deadzone)
fn sign_f64(v: f64) -> i32 {
    if v > 0.01 { 1 } else if v < -0.01 { -1 } else { 0 }
}

/// Compute cross-TF heuristic direction using weighted voting.
///
/// Votes from 3 sources:
///   Current TF:  supertrend_dir, trend, trend_short  → weight=1 each → max ±3
///   Higher TF:   supertrend_dir, trend, trend_short  → weight=2 each → max ±6  (dominant)
///   Lower TF:    trend_short only                    → weight=1       → max ±1  (timing)
///
/// Total max = 10 points
/// Direction = sign(total)
/// Confidence = |total| / 10.0
///
/// This function is generic over the store type to support both
/// production (CrossTfStore) and backtest (MultiTfStore).
fn compute_heuristic_inner(
    current_candle: &CandleWithIndicators,
    current_tf: i32,
    get_higher: impl Fn(&str, i32) -> Option<(f64, f64, f64)>, // (supertrend_dir, trend, trend_short)
    get_lower: impl Fn(&str, i32) -> Option<f64>,              // trend_short only
) -> HeuristicResult {
    let mut score: i32 = 0;
    let max_points: i32 = 10;
    let mut n_sources: u8 = 1;

    // ── Current TF (weight=1 each, max ±3) ──
    score += sign_f64(current_candle.supertrend_dir);
    score += sign_f64(current_candle.trend);
    score += sign_f64(current_candle.trend_short);

    // ── Higher TF (weight=2 each, max ±6) — THE KEY FOR DIRECTION ──
    if let Some(htf) = get_higher_tf(current_tf) {
        if let Some((st_dir, trend, trend_s)) = get_higher(&current_candle.symbol, htf) {
            score += 2 * sign_f64(st_dir);
            score += 2 * sign_f64(trend);
            score += 2 * sign_f64(trend_s);
            n_sources += 1;
        }
    }

    // ── Lower TF (weight=1, trend_short only — for timing) ──
    if let Some(ltf) = get_lower_tf(current_tf) {
        if let Some(ts) = get_lower(&current_candle.symbol, ltf) {
            score += sign_f64(ts);
            n_sources += 1;
        }
    }

    let direction: i8 = if score > 0 { 1 } else { -1 };
    let confidence = (score.abs() as f32) / (max_points as f32);

    HeuristicResult {
        direction,
        confidence,
        sources_used: n_sources,
    }
}

/// Compute heuristic direction using CrossTfStore (production mode).
pub fn compute_heuristic_direction(
    current_candle: &CandleWithIndicators,
    current_tf: i32,
    store: &CrossTfStore,
) -> HeuristicResult {
    compute_heuristic_inner(
        current_candle,
        current_tf,
        |symbol, htf| {
            store.get_latest(symbol, htf).map(|c| {
                (c.supertrend_dir, c.trend, c.trend_short)
            })
        },
        |symbol, ltf| {
            store.get_latest(symbol, ltf).map(|c| c.trend_short)
        },
    )
}

/// Compute heuristic direction using MultiTfStore (backtest mode).
pub fn compute_heuristic_direction_backtest(
    current_candle: &CandleWithIndicators,
    current_tf: i32,
    store: &MultiTfStore,
) -> HeuristicResult {
    compute_heuristic_inner(
        current_candle,
        current_tf,
        |symbol, htf| {
            store.find_candle_at(symbol, htf, current_candle.time).map(|c| {
                (c.supertrend_dir, c.trend, c.trend_short)
            })
        },
        |symbol, ltf| {
            store.find_candle_at(symbol, ltf, current_candle.time).map(|c| c.trend_short)
        },
    )
}

// ─────────────────────────────────────────────────────────────────────
// Signal Filtering
// ─────────────────────────────────────────────────────────────────────

/// Apply heuristic filter result to a signal.
///
/// Returns:
///   - `Some(filtered_signal)` if the signal should be kept (possibly with modified direction)
///   - `None` if the signal should be rejected
///
/// In Filter mode: rejects if heuristic disagrees with ML direction AND confidence >= threshold
/// In Override mode: replaces ML direction with heuristic when confident enough
/// In Off mode: always keeps the signal with ML direction
pub fn apply_heuristic_filter(
    mut signal: SuperEntrySignal,
    candle: &CandleWithIndicators,
    tf_minutes: i32,
    store: &CrossTfStore,
    mode: HeuristicMode,
    min_confidence: f32,
    config: &SuperEntryConfig,
) -> Option<SuperEntrySignal> {
    if mode == HeuristicMode::Off {
        // Enrich reason JSON with heuristic=off marker
        if let Some(reason) = signal.reason.as_object_mut() {
            reason.insert("heuristic_mode".to_string(), serde_json::json!("off"));
        }
        return Some(signal);
    }

    let result = compute_heuristic_direction(candle, tf_minutes, store);
    let ml_direction = signal.side as i8;

    // Enrich signal reason with heuristic metadata (for observability)
    if let Some(reason) = signal.reason.as_object_mut() {
        reason.insert("heuristic_mode".to_string(), serde_json::json!(format!("{:?}", mode)));
        reason.insert("heuristic_direction".to_string(), serde_json::json!(result.direction));
        reason.insert("heuristic_confidence".to_string(), serde_json::json!(result.confidence));
        reason.insert("heuristic_sources".to_string(), serde_json::json!(result.sources_used));
        reason.insert("heuristic_agrees".to_string(), serde_json::json!(result.direction == ml_direction));
    }

    match mode {
        HeuristicMode::Filter => {
            if result.confidence >= min_confidence && result.direction != ml_direction {
                None // Reject: heuristic disagrees with ML direction
            } else {
                Some(signal) // Keep with ML direction
            }
        }
        HeuristicMode::Override => {
            if result.confidence >= min_confidence && result.direction != ml_direction {
                // Flip direction — recalculate TP/SL prices
                let new_side = result.direction as i16;
                let tp_pct = config.target_pct_for_tf(tf_minutes);
                let sl_pct = config.sl_pct_for_tf(tf_minutes);
                let close = signal.entry_price;

                let (tp_price, sl_price) = if new_side == 1 {
                    (close * (1.0 + tp_pct / 100.0), close * (1.0 - sl_pct / 100.0))
                } else {
                    (close * (1.0 - tp_pct / 100.0), close * (1.0 + sl_pct / 100.0))
                };

                signal.side = new_side;
                signal.tp_price = tp_price;
                signal.sl_price = sl_price;

                if let Some(reason) = signal.reason.as_object_mut() {
                    reason.insert("heuristic_override".to_string(), serde_json::json!(true));
                    reason.insert("original_ml_direction".to_string(),
                        serde_json::json!(if ml_direction == 1 { "LONG" } else { "SHORT" }));
                    reason.insert("direction".to_string(),
                        serde_json::json!(if new_side == 1 { "LONG" } else { "SHORT" }));
                }

                Some(signal)
            } else {
                Some(signal) // Not confident enough, keep ML direction
            }
        }
        HeuristicMode::Off => unreachable!(), // Handled above
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_candle(symbol: &str, trend: f64, trend_short: f64, supertrend_dir: f64) -> CandleWithIndicators {
        CandleWithIndicators {
            time: Utc::now(),
            symbol: symbol.to_string(),
            symbol_id: 1,
            open: 100.0, high: 101.0, low: 99.0, close: 100.0, volume: 1000.0,
            rsi: 50.0, cci: 0.0, stoch_k: 50.0, stoch_d: 50.0, williams: -50.0,
            macd: 0.0, macd_signal: 0.0, macd_hist: 0.0,
            adx: 25.0, sma: 100.0, ema_20: 100.0, ema_50: 100.0, ema_200: 100.0,
            bb_upper: 102.0, bb_mid: 100.0, bb_lower: 98.0,
            atr: 1.0, obv: 0.0, vwap: 100.0, volume_spike: 1.0,
            trend, trend_short, poc: 100.0,
            alligator_jaw: 100.0, alligator_teeth: 100.0, alligator_lips: 100.0,
            mfi: 50.0, fibo_pivot: 100.0, fibo_r1: 101.0, fibo_s1: 99.0,
            supertrend: 100.0, supertrend_dir, cmf: 0.0,
        }
    }

    #[test]
    fn test_heuristic_mode_from_str() {
        assert_eq!(HeuristicMode::from_str_lossy("filter"), HeuristicMode::Filter);
        assert_eq!(HeuristicMode::from_str_lossy("Filter"), HeuristicMode::Filter);
        assert_eq!(HeuristicMode::from_str_lossy("override"), HeuristicMode::Override);
        assert_eq!(HeuristicMode::from_str_lossy("off"), HeuristicMode::Off);
        assert_eq!(HeuristicMode::from_str_lossy("disabled"), HeuristicMode::Off);
        assert_eq!(HeuristicMode::from_str_lossy("anything"), HeuristicMode::Filter); // default
    }

    #[test]
    fn test_tf_hierarchy() {
        assert_eq!(get_higher_tf(15), Some(60));
        assert_eq!(get_higher_tf(60), Some(240));
        assert_eq!(get_higher_tf(1440), None);
        assert_eq!(get_lower_tf(60), Some(15));
        assert_eq!(get_lower_tf(15), Some(5));
        assert_eq!(get_lower_tf(1), None);
    }

    #[test]
    fn test_compute_heuristic_bullish() {
        let mut store = CrossTfStore::new();

        // Current TF (15m): all bullish (+3)
        let candle = make_candle("BTCUSDT", 1.0, 1.0, 1.0);

        // Higher TF (60m): all bullish (+6)
        let htf_candle = make_candle("BTCUSDT", 1.0, 1.0, 1.0);
        store.update(htf_candle, 60);

        // Lower TF (5m): bullish (+1)
        let ltf_candle = make_candle("BTCUSDT", 1.0, 1.0, 1.0);
        store.update(ltf_candle, 5);

        let result = compute_heuristic_direction(&candle, 15, &store);
        assert_eq!(result.direction, 1); // LONG
        assert!((result.confidence - 1.0).abs() < 0.01); // max confidence (10/10)
        assert_eq!(result.sources_used, 3);
    }

    #[test]
    fn test_compute_heuristic_bearish() {
        let mut store = CrossTfStore::new();

        // Current TF (15m): all bearish (-3)
        let candle = make_candle("BTCUSDT", -1.0, -1.0, -1.0);

        // Higher TF (60m): all bearish (-6)
        let htf_candle = make_candle("BTCUSDT", -1.0, -1.0, -1.0);
        store.update(htf_candle, 60);

        let result = compute_heuristic_direction(&candle, 15, &store);
        assert_eq!(result.direction, -1); // SHORT
        assert!(result.confidence > 0.8); // high confidence
    }

    #[test]
    fn test_filter_rejects_disagree() {
        let mut store = CrossTfStore::new();

        // Higher TF is strongly bearish
        let htf_candle = make_candle("BTCUSDT", -1.0, -1.0, -1.0);
        store.update(htf_candle, 60);

        // Current candle is bearish too
        let candle = make_candle("BTCUSDT", -1.0, -1.0, -1.0);

        // ML says LONG (side=1), but heuristic says SHORT
        let result = compute_heuristic_direction(&candle, 15, &store);
        assert_eq!(result.direction, -1); // heuristic says SHORT
        assert!(result.confidence >= 0.3); // confident enough to filter

        // If ml_direction is 1 (LONG) and heuristic is -1 (SHORT) with high confidence,
        // filter mode should reject
        let ml_direction: i8 = 1;
        if result.confidence >= 0.3 && result.direction != ml_direction {
            // Signal rejected — this is what we'd expect
        } else {
            panic!("Expected filter to reject this signal");
        }
    }

    #[test]
    fn test_cross_tf_store_update() {
        let mut store = CrossTfStore::new();
        assert!(store.is_empty());

        let candle = make_candle("BTCUSDT", 1.0, 1.0, 1.0);
        store.update(candle, 60);

        assert_eq!(store.len(), 1);
        assert!(store.get_latest("BTCUSDT", 60).is_some());
        assert!(store.get_latest("ETHUSDT", 60).is_none());
        assert!(store.get_latest("BTCUSDT", 15).is_none());
    }
}
