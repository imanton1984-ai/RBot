// strategies/ml_entry_strategy/src/scorer.rs
//
// Scorer for Super Entry Strategy
//
// Decision logic:
//   1. If p_super >= p_threshold → potential super signal
//   2. Direction from P(LONG) model
//   3. Confidence = p_super (higher = more confident)
//
// The scorer also applies:
//   - "Overheated" detection: filters entries where indicators are extreme
//     but p_super is marginally above threshold (low-quality super signals)
//   - Directional confidence: requires p_long > dir_threshold OR p_long < (1 - dir_threshold)

use crate::config::SuperEntryConfig;
use crate::model::SuperEntryPrediction;
use serde::{Deserialize, Serialize};

/// Decision from the scorer
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SuperEntryDecision {
    /// Strong super entry signal detected
    SuperEntry {
        /// 1 = LONG, -1 = SHORT
        direction: i8,
        /// P(super) confidence
        p_super: f32,
        /// Directional confidence: distance from 0.5
        dir_confidence: f32,
        /// Combined score (p_super * dir_confidence_factor)
        combined_score: f32,
    },
    /// Below threshold — no signal
    NoSignal {
        /// Reason for rejection
        reason: RejectReason,
        p_super: f32,
    },
}

/// Reason why a signal was not generated
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RejectReason {
    /// P(super) below threshold
    BelowThreshold,
    /// Directional confidence too low (close to 0.5)
    WeakDirection,
    /// Indicators suggest overheated entry (marginal p_super + extreme indicators)
    Overheated,
}

impl SuperEntryDecision {
    /// Check if this is a super entry signal
    pub fn is_super_entry(&self) -> bool {
        matches!(self, SuperEntryDecision::SuperEntry { .. })
    }

    /// Get direction if it's a super entry
    pub fn direction(&self) -> Option<i8> {
        match self {
            SuperEntryDecision::SuperEntry { direction, .. } => Some(*direction),
            _ => None,
        }
    }

    /// Get p_super value regardless of decision
    pub fn p_super(&self) -> f32 {
        match self {
            SuperEntryDecision::SuperEntry { p_super, .. } => *p_super,
            SuperEntryDecision::NoSignal { p_super, .. } => *p_super,
        }
    }
}

/// Configuration for the scorer
#[derive(Debug, Clone)]
pub struct ScorerConfig {
    /// Minimum P(super) to consider as a potential signal
    pub p_threshold: f64,
    /// Minimum directional confidence (distance from 0.5 for p_long)
    pub min_dir_confidence: f32,
    /// Maximum marginal P(super) above threshold to be considered "overheated"
    /// if indicator extremes are detected
    pub overheated_margin: f32,
    /// Enable overheated detection
    pub enable_overheated_filter: bool,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            p_threshold: 0.55,
            min_dir_confidence: 0.10, // p_long must be > 0.6 or < 0.4
            overheated_margin: 0.05,  // p_super in [threshold, threshold+0.05] is marginal
            enable_overheated_filter: true,
        }
    }
}

impl From<&SuperEntryConfig> for ScorerConfig {
    fn from(cfg: &SuperEntryConfig) -> Self {
        Self {
            p_threshold: cfg.p_threshold,
            ..Default::default()
        }
    }
}

/// Super Entry Scorer
///
/// Evaluates predictions from the model and makes entry decisions.
pub struct SuperEntryScorer {
    config: ScorerConfig,
}

impl SuperEntryScorer {
    /// Create a new scorer with the given configuration
    pub fn new(config: ScorerConfig) -> Self {
        Self { config }
    }

    /// Create scorer from strategy config
    pub fn from_strategy_config(cfg: &SuperEntryConfig) -> Self {
        Self::new(ScorerConfig::from(cfg))
    }

    /// Score a prediction and return a decision.
    ///
    /// # Arguments
    /// * `prediction` - Model prediction output
    /// * `features` - Raw feature values (for overheated detection)
    ///
    /// # Returns
    /// SuperEntryDecision
    pub fn score(
        &self,
        prediction: &SuperEntryPrediction,
        features: Option<&OverheatedFeatures>,
    ) -> SuperEntryDecision {
        let p_super = prediction.p_super;
        let p_long = prediction.p_long;
        let direction = prediction.direction;

        // 1. Check P(super) threshold
        if (p_super as f64) < self.config.p_threshold {
            return SuperEntryDecision::NoSignal {
                reason: RejectReason::BelowThreshold,
                p_super,
            };
        }

        // 2. Check directional confidence
        let dir_confidence = (p_long - 0.5).abs();
        if dir_confidence < self.config.min_dir_confidence {
            return SuperEntryDecision::NoSignal {
                reason: RejectReason::WeakDirection,
                p_super,
            };
        }

        // 3. Overheated filter (optional)
        if self.config.enable_overheated_filter {
            let is_marginal = (p_super as f64) < self.config.p_threshold + self.config.overheated_margin as f64;
            if is_marginal {
                if let Some(oh) = features {
                    if oh.is_overheated(direction) {
                        return SuperEntryDecision::NoSignal {
                            reason: RejectReason::Overheated,
                            p_super,
                        };
                    }
                }
            }
        }

        // 4. Calculate combined score
        // Higher is better: p_super weighted by directional confidence
        let dir_factor = 1.0 + dir_confidence; // Range [1.0, 1.5]
        let combined_score = p_super * dir_factor;

        SuperEntryDecision::SuperEntry {
            direction,
            p_super,
            dir_confidence,
            combined_score,
        }
    }

    /// Get the config
    pub fn config(&self) -> &ScorerConfig {
        &self.config
    }
}

/// Features used for overheated detection.
/// Extracted from the raw indicator values.
pub struct OverheatedFeatures {
    pub rsi: f64,
    pub stoch_k: f64,
    pub cci: f64,
    pub williams: f64,
    pub bb_position: f64, // 0 = lower band, 1 = upper band
    pub atr_pct: f64,
}

impl OverheatedFeatures {
    /// Check if indicators suggest an "overheated" entry.
    ///
    /// An entry is overheated if:
    ///   - For LONG: RSI > 75, Stoch > 85, CCI > 150, or BB position > 0.95
    ///   - For SHORT: RSI < 25, Stoch < 15, CCI < -150, or BB position < 0.05
    ///
    /// These are "extreme" readings that often precede reversals rather than
    /// sustained moves, meaning a "super" signal at these levels is likely
    /// a false positive.
    pub fn is_overheated(&self, direction: i8) -> bool {
        let mut extreme_count = 0;

        if direction == 1 {
            // LONG: check for overbought extremes
            if self.rsi > 75.0 { extreme_count += 1; }
            if self.stoch_k > 85.0 { extreme_count += 1; }
            if self.cci > 150.0 { extreme_count += 1; }
            if self.williams > -10.0 { extreme_count += 1; } // Williams near 0 = overbought
            if self.bb_position > 0.95 { extreme_count += 1; }
        } else {
            // SHORT: check for oversold extremes
            if self.rsi < 25.0 { extreme_count += 1; }
            if self.stoch_k < 15.0 { extreme_count += 1; }
            if self.cci < -150.0 { extreme_count += 1; }
            if self.williams < -90.0 { extreme_count += 1; } // Williams near -100 = oversold
            if self.bb_position < 0.05 { extreme_count += 1; }
        }

        // Need at least 2 extreme readings to call it overheated
        extreme_count >= 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_super_entry_above_threshold() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        let pred = SuperEntryPrediction {
            p_super: 0.75,
            p_long: 0.80,
            direction: 1,
            estimated_magnitude_pct: 2.0,
        };

        let decision = scorer.score(&pred, None);
        assert!(decision.is_super_entry());
        assert_eq!(decision.direction(), Some(1));
    }

    #[test]
    fn test_below_threshold() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        let pred = SuperEntryPrediction {
            p_super: 0.40,
            p_long: 0.80,
            direction: 1,
            estimated_magnitude_pct: 1.0,
        };

        let decision = scorer.score(&pred, None);
        assert!(!decision.is_super_entry());
        match decision {
            SuperEntryDecision::NoSignal { reason, .. } => {
                assert_eq!(reason, RejectReason::BelowThreshold);
            }
            _ => panic!("Expected NoSignal"),
        }
    }

    #[test]
    fn test_weak_direction() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        let pred = SuperEntryPrediction {
            p_super: 0.75,
            p_long: 0.52, // very close to 0.5
            direction: 1,
            estimated_magnitude_pct: 2.0,
        };

        let decision = scorer.score(&pred, None);
        assert!(!decision.is_super_entry());
        match decision {
            SuperEntryDecision::NoSignal { reason, .. } => {
                assert_eq!(reason, RejectReason::WeakDirection);
            }
            _ => panic!("Expected NoSignal for weak direction"),
        }
    }

    #[test]
    fn test_overheated_filter() {
        let scorer = SuperEntryScorer::new(ScorerConfig {
            p_threshold: 0.55,
            overheated_margin: 0.05,
            enable_overheated_filter: true,
            ..Default::default()
        });

        // Marginal p_super (0.57 < 0.55 + 0.05) with extreme indicators
        let pred = SuperEntryPrediction {
            p_super: 0.57,
            p_long: 0.80,
            direction: 1,
            estimated_magnitude_pct: 1.0,
        };

        let oh = OverheatedFeatures {
            rsi: 82.0,      // overbought
            stoch_k: 90.0,  // overbought
            cci: 160.0,     // extreme
            williams: -5.0, // overbought
            bb_position: 0.98,
            atr_pct: 2.0,
        };

        let decision = scorer.score(&pred, Some(&oh));
        assert!(!decision.is_super_entry());
        match decision {
            SuperEntryDecision::NoSignal { reason, .. } => {
                assert_eq!(reason, RejectReason::Overheated);
            }
            _ => panic!("Expected Overheated rejection"),
        }
    }

    #[test]
    fn test_not_overheated_when_strong_signal() {
        let scorer = SuperEntryScorer::new(ScorerConfig {
            p_threshold: 0.55,
            overheated_margin: 0.05,
            enable_overheated_filter: true,
            ..Default::default()
        });

        // Strong p_super (0.80 > 0.55 + 0.05) — not marginal, so overheated filter skipped
        let pred = SuperEntryPrediction {
            p_super: 0.80,
            p_long: 0.85,
            direction: 1,
            estimated_magnitude_pct: 3.0,
        };

        let oh = OverheatedFeatures {
            rsi: 82.0,
            stoch_k: 90.0,
            cci: 160.0,
            williams: -5.0,
            bb_position: 0.98,
            atr_pct: 2.0,
        };

        let decision = scorer.score(&pred, Some(&oh));
        // Should still be super entry because p_super is not marginal
        assert!(decision.is_super_entry());
    }
}
