// strategies/ml_entry_strategy/src/signal_generator.rs
//
// Signal Generator for Super Entry Strategy
//
// Converts SuperEntryDecision into a trade signal compatible with
// the existing system format (trade.final_signals table schema).
//
// The generated signal includes:
//   - entry_price (close of the current candle)
//   - sl_price (derived from target move * sl_fraction)
//   - tp_price (= target move %)
//   - direction (LONG/SHORT from model)
//   - reason JSON with strategy metadata

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::SuperEntryConfig;
use crate::scorer::SuperEntryDecision;

/// A generated super entry signal in the system format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperEntrySignal {
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
    /// Stop-loss price
    pub sl_price: f64,
    /// Take-profit price (target move %)
    pub tp_price: f64,
    /// Combined score from the scorer (0..1+)
    pub final_score: f32,
    /// P(super) from the model
    pub p_super: f32,
    /// P(LONG) / P(UP) from the direction model
    pub p_long: f32,
    /// Direction confidence from model:
    ///   v4: P(predicted_class) ∈ [0.5, 1.0]
    ///   v3: abs(regression)
    ///   legacy: |p_long - 0.5|
    pub dir_confidence: f32,
    /// Strategy metadata as JSON
    pub reason: serde_json::Value,
}

impl SuperEntrySignal {
    /// Check if this is a LONG signal
    pub fn is_long(&self) -> bool {
        self.side == 1
    }

    /// Check if this is a SHORT signal
    pub fn is_short(&self) -> bool {
        self.side == -1
    }

    /// Get the target move in % for this signal's TF
    pub fn target_move_pct(&self, config: &SuperEntryConfig) -> f64 {
        config.target_pct_for_tf(self.tf_minutes)
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

/// Signal Generator — creates signals from scorer decisions
pub struct SignalGenerator {
    config: SuperEntryConfig,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(config: SuperEntryConfig) -> Self {
        Self { config }
    }

    /// Generate a super entry signal from a decision and market context.
    ///
    /// # Arguments
    /// * `decision` - Scorer decision (must be SuperEntry)
    /// * `symbol` - Trading pair symbol
    /// * `symbol_id` - Database symbol ID
    /// * `tf_minutes` - Timeframe in minutes
    /// * `time` - Current candle timestamp
    /// * `close_price` - Current close price (entry price)
    /// * `atr` - Current ATR value (for SL calculation, optional fallback)
    ///
    /// # Returns
    /// Some(SuperEntrySignal) if decision is SuperEntry, None otherwise
    pub fn generate(
        &self,
        decision: &SuperEntryDecision,
        symbol: &str,
        symbol_id: i64,
        tf_minutes: i32,
        time: DateTime<Utc>,
        close_price: f64,
        atr: f64,
    ) -> Option<SuperEntrySignal> {
        let (direction, p_super, dir_confidence, combined_score) = match decision {
            SuperEntryDecision::SuperEntry {
                direction,
                p_super,
                dir_confidence,
                combined_score,
            } => (*direction, *p_super, *dir_confidence, *combined_score),
            SuperEntryDecision::NoSignal { .. } => return None,
        };

        let side: i16 = direction as i16;
        let target_pct = self.config.target_pct_for_tf(tf_minutes);
        let sl_pct = self.config.sl_pct_for_tf(tf_minutes);

        // Calculate TP and SL prices
        let (tp_price, sl_price) = if direction == 1 {
            // LONG
            let tp = close_price * (1.0 + target_pct / 100.0);
            let sl = close_price * (1.0 - sl_pct / 100.0);
            (tp, sl)
        } else {
            // SHORT
            let tp = close_price * (1.0 - target_pct / 100.0);
            let sl = close_price * (1.0 + sl_pct / 100.0);
            (tp, sl)
        };

        // p_long: for v4 this is P(UP) directly, for legacy it's 0.5 ± confidence
        let p_long = if direction == 1 {
            0.5 + dir_confidence.min(0.5)
        } else {
            0.5 - dir_confidence.min(0.5)
        };

        let reason = json!({
            "strategy": "ml_entry_strategy",
            "model_version": "v1",
            "p_super": p_super,
            "p_long": p_long,
            "direction": if direction == 1 { "LONG" } else { "SHORT" },
            "dir_confidence": dir_confidence,
            "combined_score": combined_score,
            "target_move_pct": target_pct,
            "sl_pct": sl_pct,
            "atr": atr,
            "lookahead_bars": self.config.lookahead_bars,
            "p_threshold": self.config.p_threshold,
        });

        let time_ms = time.timestamp_millis();

        Some(SuperEntrySignal {
            symbol: symbol.to_string(),
            symbol_id,
            tf_minutes,
            time,
            time_ms,
            side,
            entry_price: close_price,
            sl_price,
            tp_price,
            final_score: combined_score,
            p_super,
            p_long: p_long as f32,
            dir_confidence,
            reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_long_signal() {
        let config = SuperEntryConfig::default();
        let generator = SignalGenerator::new(config.clone());

        let decision = SuperEntryDecision::SuperEntry {
            direction: 1,
            p_super: 0.80,
            dir_confidence: 0.30,
            combined_score: 1.04,
        };

        let signal = generator.generate(
            &decision,
            "BTCUSDT",
            1,
            60,
            Utc::now(),
            50000.0,
            500.0,
        );

        assert!(signal.is_some());
        let sig = signal.unwrap();
        assert!(sig.is_long());
        assert_eq!(sig.side, 1);
        assert!(sig.tp_price > sig.entry_price); // TP above entry for LONG
        assert!(sig.sl_price < sig.entry_price); // SL below entry for LONG
        assert!((sig.tp_pct() - config.target_pct_for_tf(60)).abs() < 0.01);
    }

    #[test]
    fn test_generate_short_signal() {
        let config = SuperEntryConfig::default();
        let generator = SignalGenerator::new(config);

        let decision = SuperEntryDecision::SuperEntry {
            direction: -1,
            p_super: 0.70,
            dir_confidence: 0.25,
            combined_score: 0.88,
        };

        let signal = generator.generate(
            &decision,
            "ETHUSDT",
            2,
            15,
            Utc::now(),
            3000.0,
            30.0,
        );

        assert!(signal.is_some());
        let sig = signal.unwrap();
        assert!(sig.is_short());
        assert_eq!(sig.side, -1);
        assert!(sig.tp_price < sig.entry_price); // TP below entry for SHORT
        assert!(sig.sl_price > sig.entry_price); // SL above entry for SHORT
    }

    #[test]
    fn test_no_signal_generation() {
        let config = SuperEntryConfig::default();
        let generator = SignalGenerator::new(config);

        let decision = SuperEntryDecision::NoSignal {
            reason: crate::scorer::RejectReason::BelowThreshold,
            p_super: 0.30,
        };

        let signal = generator.generate(
            &decision,
            "BTCUSDT",
            1,
            60,
            Utc::now(),
            50000.0,
            500.0,
        );

        assert!(signal.is_none());
    }
}
