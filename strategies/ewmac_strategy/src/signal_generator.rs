// strategies/ewmac_strategy/src/signal_generator.rs
//
// Signal Generator for EWMAC Strategy
//
// Converts EwmacResult + market context into EwmacSignal.
// Signal is generated only when |forecast| >= min_forecast.
//
// SL/TP are ATR-based (not percentage-based like super_entry).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::EwmacConfig;
use crate::ewmac::EwmacResult;

/// A generated EWMAC signal in the system format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EwmacSignal {
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Symbol ID from market.pairs
    pub symbol_id: i64,
    /// Timeframe in minutes
    pub tf_minutes: i32,
    /// Signal timestamp
    pub time: DateTime<Utc>,
    /// Signal time in milliseconds (epoch)
    pub time_ms: i64,
    /// Side: 1 = LONG, -1 = SHORT
    pub side: i16,
    /// Entry price (close of the candle when signal was generated)
    pub entry_price: f64,
    /// Stop-loss price (ATR-based)
    pub sl_price: f64,
    /// Take-profit price (ATR-based)
    pub tp_price: f64,

    // EWMAC-specific fields
    /// Raw aggregate EWMAC signal (unnormalized)
    pub raw_signal: f64,
    /// Normalized aggregate signal
    pub norm_signal: f64,
    /// Scaled forecast [-20, +20]
    pub forecast: f64,
    /// Signal strength: |forecast| / 20 (0..1)
    pub signal_strength: f32,

    /// Per-pair normalized values
    pub ewmac_2_8: Option<f64>,
    pub ewmac_4_16: Option<f64>,
    pub ewmac_8_32: Option<f64>,
    pub ewmac_16_64: Option<f64>,
    pub ewmac_32_128: Option<f64>,
    pub ewmac_64_256: Option<f64>,

    /// ATR as percentage of close
    pub atr_pct: f32,
    /// Strategy metadata as JSON
    pub reason: serde_json::Value,
}

impl EwmacSignal {
    /// Check if this is a LONG signal
    pub fn is_long(&self) -> bool {
        self.side == 1
    }

    /// Check if this is a SHORT signal
    pub fn is_short(&self) -> bool {
        self.side == -1
    }

    /// Get the SL distance in %
    pub fn sl_pct(&self) -> f64 {
        if self.entry_price > 0.0 {
            (self.sl_price - self.entry_price).abs() / self.entry_price * 100.0
        } else {
            0.0
        }
    }

    /// Get the TP distance in %
    pub fn tp_pct(&self) -> f64 {
        if self.entry_price > 0.0 {
            (self.tp_price - self.entry_price).abs() / self.entry_price * 100.0
        } else {
            0.0
        }
    }
}

/// Signal Generator — creates signals from EWMAC results
pub struct SignalGenerator {
    config: EwmacConfig,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(config: EwmacConfig) -> Self {
        Self { config }
    }

    /// Generate an EWMAC signal from a computation result and market context.
    ///
    /// Returns Some(EwmacSignal) if |forecast| >= min_forecast, None otherwise.
    pub fn generate(
        &self,
        result: &EwmacResult,
        symbol: &str,
        symbol_id: i64,
        tf_minutes: i32,
        time: DateTime<Utc>,
        close_price: f64,
    ) -> Option<EwmacSignal> {
        // Filter: minimum forecast threshold
        if result.forecast.abs() < self.config.min_forecast {
            return None;
        }

        // Filter: minimum ATR% (skip illiquid / dead pairs)
        if result.atr_pct < self.config.min_atr_pct {
            return None;
        }

        // Filter: require minimum number of pairs in agreement
        if result.pairs_agree < self.config.min_pairs_agree {
            return None;
        }

        let side: i16 = if result.forecast > 0.0 { 1 } else { -1 };
        let (sl_mult, tp_mult) = self.config.atr_mults_for_tf(tf_minutes);

        // ATR-based SL and TP
        let (tp_price, sl_price) = if side == 1 {
            // LONG
            let tp = close_price + result.atr * tp_mult;
            let sl = close_price - result.atr * sl_mult;
            (tp, sl)
        } else {
            // SHORT
            let tp = close_price - result.atr * tp_mult;
            let sl = close_price + result.atr * sl_mult;
            (tp, sl)
        };

        // Extract per-pair values
        let per_pair = &result.per_pair;

        let reason = json!({
            "strategy": "ewmac_v1",
            "forecast": result.forecast,
            "raw_signal": result.raw_signal,
            "norm_signal": result.norm_signal,
            "signal_strength": result.signal_strength,
            "direction": if side == 1 { "LONG" } else { "SHORT" },
            "atr": result.atr,
            "atr_pct": result.atr_pct,
            "sl_mult": sl_mult,
            "tp_mult": tp_mult,
            "fdm": self.config.fdm,
            "min_forecast": self.config.min_forecast,
            "pairs": {
                "ewmac_2_8": per_pair[0],
                "ewmac_4_16": per_pair[1],
                "ewmac_8_32": per_pair[2],
                "ewmac_16_64": per_pair[3],
                "ewmac_32_128": per_pair[4],
                "ewmac_64_256": per_pair[5],
            }
        });

        Some(EwmacSignal {
            symbol: symbol.to_string(),
            symbol_id,
            tf_minutes,
            time,
            time_ms: time.timestamp_millis(),
            side,
            entry_price: close_price,
            sl_price,
            tp_price,
            raw_signal: result.raw_signal,
            norm_signal: result.norm_signal,
            forecast: result.forecast,
            signal_strength: result.signal_strength as f32,
            ewmac_2_8: Some(per_pair[0]),
            ewmac_4_16: Some(per_pair[1]),
            ewmac_8_32: Some(per_pair[2]),
            ewmac_16_64: Some(per_pair[3]),
            ewmac_32_128: Some(per_pair[4]),
            ewmac_64_256: Some(per_pair[5]),
            atr_pct: result.atr_pct as f32,
            reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_long_signal() {
        let config = EwmacConfig::default();
        let generator = SignalGenerator::new(config.clone());

        let result = EwmacResult {
            forecast: 12.0,
            raw_signal: 50.0,
            norm_signal: 0.5,
            signal_strength: 0.6,
            per_pair: [0.8, 0.6, 0.5, 0.4, 0.3, 0.2],
            atr: 500.0,
            atr_pct: 1.0,
            pairs_agree: 6,
        };

        let signal = generator.generate(
            &result,
            "BTCUSDT", 1, 60,
            Utc::now(),
            50000.0,
        );

        assert!(signal.is_some());
        let sig = signal.unwrap();
        assert!(sig.is_long());
        assert_eq!(sig.side, 1);
        assert!(sig.tp_price > sig.entry_price);
        assert!(sig.sl_price < sig.entry_price);
    }

    #[test]
    fn test_generate_short_signal() {
        let config = EwmacConfig::default();
        let generator = SignalGenerator::new(config);

        let result = EwmacResult {
            forecast: -15.0,
            raw_signal: -80.0,
            norm_signal: -0.8,
            signal_strength: 0.75,
            per_pair: [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4],
            atr: 30.0,
            atr_pct: 1.0,
            pairs_agree: 6,
        };

        let signal = generator.generate(
            &result,
            "ETHUSDT", 2, 15,
            Utc::now(),
            3000.0,
        );

        assert!(signal.is_some());
        let sig = signal.unwrap();
        assert!(sig.is_short());
        assert!(sig.tp_price < sig.entry_price);
        assert!(sig.sl_price > sig.entry_price);
    }

    #[test]
    fn test_no_signal_below_threshold() {
        let config = EwmacConfig::default();
        let generator = SignalGenerator::new(config);

        let result = EwmacResult {
            forecast: 2.0, // below min_forecast of 10.0
            raw_signal: 10.0,
            norm_signal: 0.1,
            signal_strength: 0.1,
            per_pair: [0.1, 0.05, 0.02, 0.01, 0.0, -0.01],
            atr: 500.0,
            atr_pct: 1.0,
            pairs_agree: 5,
        };

        let signal = generator.generate(
            &result,
            "BTCUSDT", 1, 60,
            Utc::now(),
            50000.0,
        );

        assert!(signal.is_none());
    }

    #[test]
    fn test_no_signal_low_atr() {
        let config = EwmacConfig::default();
        let generator = SignalGenerator::new(config);

        let result = EwmacResult {
            forecast: 15.0,
            raw_signal: 100.0,
            norm_signal: 1.0,
            signal_strength: 0.75,
            per_pair: [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
            atr: 0.001,
            atr_pct: 0.001, // way below min_atr_pct
            pairs_agree: 6,
        };

        let signal = generator.generate(
            &result,
            "BTCUSDT", 1, 60,
            Utc::now(),
            50000.0,
        );

        assert!(signal.is_none());
    }
}
