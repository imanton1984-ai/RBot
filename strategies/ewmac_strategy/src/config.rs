// strategies/ewmac_strategy/src/config.rs
//
// Configuration for EWMAC (Exponentially Weighted Moving Average Crossover) Strategy
//
// EWMAC is a trend-following strategy from Robert Carver's "Systematic Trading".
// Key parameters:
//   - EMA pairs: (fast, slow) spans for crossover calculation
//   - Forecast scalar: maps raw signal to [-20, +20] scale
//   - Signal threshold: minimum |forecast| to trigger entry
//   - ATR-based SL/TP sizing
//   - Warmup bars for EMA convergence

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single EWMAC crossover pair (fast_span, slow_span).
/// Signal = (EMA(fast) - EMA(slow)) / ATR
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EwmacPair {
    pub fast: usize,
    pub slow: usize,
    /// Weight when combining into aggregate forecast (default: equal)
    pub weight: f64,
}

impl EwmacPair {
    pub fn new(fast: usize, slow: usize) -> Self {
        Self { fast, slow, weight: 1.0 }
    }

    /// Label for the pair: "ewmac_{fast}_{slow}"
    pub fn label(&self) -> String {
        format!("ewmac_{}_{}", self.fast, self.slow)
    }
}

/// Default EWMAC pairs following Carver's methodology.
/// Pairs span from very fast (2,8) to very slow (64,256).
pub fn default_ewmac_pairs() -> Vec<EwmacPair> {
    vec![
        EwmacPair::new(2, 8),
        EwmacPair::new(4, 16),
        EwmacPair::new(8, 32),
        EwmacPair::new(16, 64),
        EwmacPair::new(32, 128),
        EwmacPair::new(64, 256),
    ]
}

/// Forecast scalar per EWMAC pair.
/// These convert raw crossover values into the [-20, +20] forecast range.
/// Values from Carver's book (Table 15.2), adjusted for crypto volatility.
///
/// The scalar = 10.6 / avg_abs_value_of_raw_signal (empirically measured).
/// For crypto the raw signals tend to be larger, so scalars are somewhat lower.
pub fn default_forecast_scalars() -> HashMap<(usize, usize), f64> {
    let mut m = HashMap::new();
    m.insert((2, 8), 12.1);
    m.insert((4, 16), 8.5);
    m.insert((8, 32), 5.8);
    m.insert((16, 64), 3.7);
    m.insert((32, 128), 2.4);
    m.insert((64, 256), 1.6);
    m
}

/// ATR-based SL/TP multipliers per timeframe.
/// SL = ATR * sl_atr_mult, TP = ATR * tp_atr_mult
///
/// For crypto: SL wider than traditional equities, TP more realistic.
/// Ratio SL:TP ~ 1:1.5 to 1:2 (need ~40-50% WR to break even).
pub fn default_atr_multipliers() -> HashMap<i32, (f64, f64)> {
    let mut m = HashMap::new();
    // (sl_atr_mult, tp_atr_mult)
    m.insert(1, (2.0, 3.0));      // 1m
    m.insert(5, (2.0, 3.0));      // 5m
    m.insert(15, (2.5, 4.0));     // 15m
    m.insert(60, (2.5, 4.0));     // 1h
    m.insert(240, (3.0, 5.0));    // 4h
    m.insert(1440, (3.5, 6.0));   // 1d
    m
}

/// Full configuration for the EWMAC Strategy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EwmacConfig {
    /// Number of warmup candles needed for the slowest EMA to converge.
    /// Must be >= max(slow_span) * 3 for proper convergence.
    /// Default: 300 (covers 64,256 pair with 3x margin).
    pub warmup_bars: usize,

    /// EWMAC pairs to compute
    pub pairs: Vec<EwmacPair>,

    /// Forecast scalars per pair (key = (fast, slow))
    pub forecast_scalars: HashMap<(usize, usize), f64>,

    /// Minimum absolute forecast to trigger an entry signal.
    /// Carver uses 10 as "strong". Default: 10 for crypto (higher noise).
    pub min_forecast: f64,

    /// Maximum forecast cap (absolute). Carver standard = 20.
    pub max_forecast: f64,

    /// Maximum bars to hold a trade before force-closing.
    pub max_hold_bars: usize,

    /// Cooldown bars: after generating a signal, suppress same-direction signals
    /// for this many bars. Prevents consecutive duplicate entries in strong trends.
    /// Default: 20 (= ~2/3 of max_hold). Set 0 to disable.
    pub cooldown_bars: usize,

    /// ATR multipliers per timeframe: (sl_mult, tp_mult)
    pub atr_multipliers: HashMap<i32, (f64, f64)>,

    /// Minimum ATR% to consider a pair tradeable (filter dead / illiquid pairs).
    pub min_atr_pct: f64,

    /// Forecast Diversification Multiplier (FDM).
    /// When combining multiple EWMAC pairs, accounts for correlation.
    /// Carver suggests ~1.0-1.5 for 6 variations.
    pub fdm: f64,

    /// Require minimum number of pairs in agreement for signal generation.
    /// E.g., 4 means at least 4 out of 6 pairs must agree on direction.
    /// Filters noise from single fast-pair crossovers.
    /// Default: 4.
    pub min_pairs_agree: usize,
}

impl Default for EwmacConfig {
    fn default() -> Self {
        Self {
            warmup_bars: 300,
            pairs: default_ewmac_pairs(),
            forecast_scalars: default_forecast_scalars(),
            min_forecast: 10.0,
            max_forecast: 20.0,
            max_hold_bars: 30,
            cooldown_bars: 20,
            atr_multipliers: default_atr_multipliers(),
            min_atr_pct: 0.05,
            fdm: 1.0,
            min_pairs_agree: 4,
        }
    }
}

impl EwmacConfig {
    /// Load config from environment variables with defaults
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("EWMAC_WARMUP_BARS") {
            if let Ok(n) = v.parse() { cfg.warmup_bars = n; }
        }
        if let Ok(v) = std::env::var("EWMAC_MIN_FORECAST") {
            if let Ok(n) = v.parse() { cfg.min_forecast = n; }
        }
        if let Ok(v) = std::env::var("EWMAC_MAX_FORECAST") {
            if let Ok(n) = v.parse() { cfg.max_forecast = n; }
        }
        if let Ok(v) = std::env::var("EWMAC_MAX_HOLD_BARS") {
            if let Ok(n) = v.parse() { cfg.max_hold_bars = n; }
        }
        if let Ok(v) = std::env::var("EWMAC_MIN_ATR_PCT") {
            if let Ok(n) = v.parse() { cfg.min_atr_pct = n; }
        }
        if let Ok(v) = std::env::var("EWMAC_FDM") {
            if let Ok(n) = v.parse() { cfg.fdm = n; }
        }
        if let Ok(v) = std::env::var("EWMAC_COOLDOWN_BARS") {
            if let Ok(n) = v.parse() { cfg.cooldown_bars = n; }
        }
        if let Ok(v) = std::env::var("EWMAC_MIN_PAIRS_AGREE") {
            if let Ok(n) = v.parse() { cfg.min_pairs_agree = n; }
        }

        cfg
    }

    /// Get ATR multipliers for a given timeframe.
    /// Returns (sl_mult, tp_mult).
    pub fn atr_mults_for_tf(&self, tf_minutes: i32) -> (f64, f64) {
        self.atr_multipliers
            .get(&tf_minutes)
            .copied()
            .unwrap_or((2.0, 4.0))
    }

    /// Maximum slow span across all pairs (determines minimum warmup).
    pub fn max_slow_span(&self) -> usize {
        self.pairs.iter().map(|p| p.slow).max().unwrap_or(256)
    }

    /// All supported timeframes.
    pub fn timeframes() -> &'static [i32] {
        &[1, 5, 15, 60, 240, 1440]
    }

    /// Number of candles needed to compute all EMAs reliably.
    /// We need at least max_slow_span * 2 candles for EMA convergence.
    pub fn min_candles_needed(&self) -> usize {
        self.max_slow_span() * 2
    }

    /// Get the forecast scalar for a given pair.
    pub fn scalar_for_pair(&self, fast: usize, slow: usize) -> f64 {
        self.forecast_scalars
            .get(&(fast, slow))
            .copied()
            .unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = EwmacConfig::default();
        assert_eq!(cfg.warmup_bars, 300);
        assert_eq!(cfg.pairs.len(), 6);
        assert!((cfg.min_forecast - 10.0).abs() < 1e-6);
        assert!((cfg.max_forecast - 20.0).abs() < 1e-6);
        assert_eq!(cfg.cooldown_bars, 20);
        assert_eq!(cfg.min_pairs_agree, 4);
    }

    #[test]
    fn test_max_slow_span() {
        let cfg = EwmacConfig::default();
        assert_eq!(cfg.max_slow_span(), 256);
    }

    #[test]
    fn test_atr_mults() {
        let cfg = EwmacConfig::default();
        let (sl, tp) = cfg.atr_mults_for_tf(60);
        assert!((sl - 2.5).abs() < 1e-6);
        assert!((tp - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_pair_labels() {
        let pair = EwmacPair::new(4, 16);
        assert_eq!(pair.label(), "ewmac_4_16");
    }

    #[test]
    fn test_scalar_for_pair() {
        let cfg = EwmacConfig::default();
        assert!((cfg.scalar_for_pair(2, 8) - 12.1).abs() < 1e-6);
        assert!((cfg.scalar_for_pair(64, 256) - 1.6).abs() < 1e-6);
        // Unknown pair falls back to 1.0
        assert!((cfg.scalar_for_pair(3, 9) - 1.0).abs() < 1e-6);
    }
}
