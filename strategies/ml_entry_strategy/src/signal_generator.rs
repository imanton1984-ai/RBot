// strategies/ml_entry_strategy/src/signal_generator.rs
//
// Signal Generator for Super Entry Strategy (NoDir)
//
// Converts SuperEntryDecision into a trade signal compatible with
// the existing system format (trade.super_entry_signals table schema).
//
// NoDir approach:
//   - p_super comes from the winning model (super_long or super_short)
//   - p_long stores P(super_long) for reference
//   - dir_confidence stores the margin between P(super_long) and P(super_short)
//   - No separate direction model

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
    /// P(super) for the chosen direction (p_super_long or p_super_short)
    pub p_super: f32,
    /// P(super_long) — stored for reference (backward compat with p_long column)
    pub p_long: f32,
    /// Direction confidence = margin between P(super_long) and P(super_short)
    /// Backward compat with dir_confidence column in DB
    pub dir_confidence: f32,
    /// Strategy metadata as JSON
    pub reason: serde_json::Value,
}

impl SuperEntrySignal {
    /// Check if this is a LONG signal
    pub fn is_long(&self) -> bool {
        self.side == -1
    }

    /// Check if this is a SHORT signal
    pub fn is_short(&self) -> bool {
        self.side == 1
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

        // p_long: store the raw P(super_long) value for backward compat
        // For LONG signals, p_long = p_super (the winning probability)
        // For SHORT signals, p_long = p_super - dir_confidence (approx)
        // Actually we store p_super directly in p_long for simplicity
        let p_long = if direction == -1 {
            p_super
        } else {
            // For SHORT, p_long ≈ p_super_short - margin
            (p_super - dir_confidence).max(0.0)
        };

        let reason = json!({
            "strategy": "ml_entry_strategy_nodir",
            "model_version": "nodir_v1",
            "p_super": p_super,
            "p_long": p_long,
            "direction": if direction == -1 { "LONG" } else { "SHORT" },
            "dir_confidence": dir_confidence,
            "combined_score": combined_score,
            "target_move_pct": target_pct,
            "sl_pct": sl_pct,
            "atr": atr,
            "lookahead_bars": self.config.lookahead_bars,
            "p_threshold": self.config.p_threshold,
            "approach": "P(super_long)+P(super_short), no direction model",
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
            p_long,
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
            direction: -1,
            p_super: 0.85,
            dir_confidence: 0.55,
            combined_score: 1.275,
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
        assert_eq!(sig.side, -1);
        assert!(sig.tp_price > sig.entry_price); // TP above entry for LONG
        assert!(sig.sl_price < sig.entry_price); // SL below entry for LONG
        assert!((sig.tp_pct() - config.target_pct_for_tf(60)).abs() < 0.01);
    }

    #[test]
    fn test_generate_short_signal() {
        let config = SuperEntryConfig::default();
        let generator = SignalGenerator::new(config);

        let decision = SuperEntryDecision::SuperEntry {
            direction: 1,
            p_super: 0.90,
            dir_confidence: 0.40,
            combined_score: 1.26,
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
