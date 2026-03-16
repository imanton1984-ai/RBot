// strategies/super_level_strategy/src/signal_generator.rs
//
// Signal Generator for Super Level Strategy
//
// Converts SuperLevelDecision into a trade signal with TP/SL levels.
// TP/SL — из tf_target_move_pct (как в super_entry).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::SuperLevelConfig;
use crate::scorer::SuperLevelDecision;

/// A generated super level signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperLevelSignal {
    pub symbol: String,
    pub symbol_id: i64,
    pub tf_minutes: i32,
    pub time: DateTime<Utc>,
    pub time_ms: i64,
    pub side: i16,            // 1=LONG, -1=SHORT
    pub entry_price: f64,
    pub sl_price: f64,
    pub tp_price: f64,
    pub final_score: f32,     // combined_score from scorer
    pub p_eval: f32,          // evaluator P(win)
    pub p_level: f32,         // P(strong_level)
    pub p_entry: f32,         // P(good_entry)
    pub p_direction: f32,     // P(correct direction)
    pub p_bounce: f32,        // P(bounce)
    pub is_bounce: bool,      // bounce vs breakout
    pub reason: serde_json::Value,
}

impl SuperLevelSignal {
    pub fn is_long(&self) -> bool { self.side == 1 }
    pub fn is_short(&self) -> bool { self.side == -1 }

    pub fn sl_pct(&self) -> f64 {
        if self.entry_price > 0.0 {
            (self.sl_price - self.entry_price).abs() / self.entry_price * 100.0
        } else { 0.0 }
    }

    pub fn tp_pct(&self) -> f64 {
        if self.entry_price > 0.0 {
            (self.tp_price - self.entry_price).abs() / self.entry_price * 100.0
        } else { 0.0 }
    }
}

/// Signal Generator
pub struct SignalGenerator {
    config: SuperLevelConfig,
}

impl SignalGenerator {
    pub fn new(config: SuperLevelConfig) -> Self {
        Self { config }
    }

    pub fn generate(
        &self,
        decision: &SuperLevelDecision,
        symbol: &str,
        symbol_id: i64,
        tf_minutes: i32,
        time: DateTime<Utc>,
        close_price: f64,
        atr: f64,
    ) -> Option<SuperLevelSignal> {
        let (direction, is_bounce, p_level, p_entry, p_direction, p_bounce, p_eval, combined_score) =
            match decision {
                SuperLevelDecision::Signal {
                    direction, is_bounce, p_level, p_entry, p_direction,
                    p_bounce, p_eval, combined_score,
                } => (*direction, *is_bounce, *p_level, *p_entry, *p_direction,
                       *p_bounce, *p_eval, *combined_score),
                SuperLevelDecision::NoSignal { .. } => return None,
            };

        let side: i16 = direction as i16;
        let target_pct = self.config.target_pct_for_tf(tf_minutes);
        let sl_pct = self.config.sl_pct_for_tf(tf_minutes);

        let (tp_price, sl_price) = if direction == 1 {
            (
                close_price * (1.0 + target_pct / 100.0),
                close_price * (1.0 - sl_pct / 100.0),
            )
        } else {
            (
                close_price * (1.0 - target_pct / 100.0),
                close_price * (1.0 + sl_pct / 100.0),
            )
        };

        let scenario_str = if is_bounce { "bounce" } else { "breakout" };

        let reason = json!({
            "strategy": "super_level_strategy",
            "model_version": "v1",
            "p_level": p_level,
            "p_entry": p_entry,
            "p_direction": p_direction,
            "p_bounce": p_bounce,
            "p_eval": p_eval,
            "combined_score": combined_score,
            "scenario": scenario_str,
            "direction": if direction == 1 { "LONG" } else { "SHORT" },
            "target_move_pct": target_pct,
            "sl_pct": sl_pct,
            "atr": atr,
            "lookahead_bars": self.config.lookahead_bars,
        });

        Some(SuperLevelSignal {
            symbol: symbol.to_string(),
            symbol_id,
            tf_minutes,
            time,
            time_ms: time.timestamp_millis(),
            side,
            entry_price: close_price,
            sl_price,
            tp_price,
            final_score: combined_score,
            p_eval,
            p_level,
            p_entry,
            p_direction,
            p_bounce,
            is_bounce,
            reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scorer::RejectReason;

    #[test]
    fn test_generate_long_signal() {
        let config = SuperLevelConfig::default();
        let gen = SignalGenerator::new(config.clone());
        let dec = SuperLevelDecision::Signal {
            direction: 1, is_bounce: true,
            p_level: 0.7, p_entry: 0.6, p_direction: 0.8,
            p_bounce: 0.7, p_eval: 0.75, combined_score: 0.72,
        };
        let sig = gen.generate(&dec, "BTCUSDT", 1, 60, Utc::now(), 50000.0, 500.0);
        assert!(sig.is_some());
        let s = sig.unwrap();
        assert!(s.is_long());
        assert!(s.tp_price > s.entry_price);
        assert!(s.sl_price < s.entry_price);
    }

    #[test]
    fn test_no_signal() {
        let config = SuperLevelConfig::default();
        let gen = SignalGenerator::new(config);
        let dec = SuperLevelDecision::NoSignal {
            reason: RejectReason::WeakLevel,
            p_eval: 0.3,
        };
        assert!(gen.generate(&dec, "BTCUSDT", 1, 60, Utc::now(), 50000.0, 500.0).is_none());
    }
}
