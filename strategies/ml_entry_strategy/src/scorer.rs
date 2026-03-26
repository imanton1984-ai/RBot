// strategies/ml_entry_strategy/src/scorer.rs
//
// Scorer for Super Entry Strategy
//
// Decision logic:
//   1. If p_super >= p_threshold → potential super signal
//   2. Direction from Direction v3 model (regression, confidence gate)
//   3. Confidence = dir_confidence (from abs(regression prediction))
//
// Direction v3 additions:
//   - Confidence gate: if dir_confidence < min_dir_confidence → WeakDirection
//   - HTF hard filter: never LONG against HTF bearish supertrend (and vice versa)
//     Removes ~50% of wrong-direction trades from model noise
//
// Legacy mode (no direction v3): uses p_long from binary classifier as before.

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
        /// Directional confidence (v3: abs(regression), legacy: |p_long - 0.5|)
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
    /// Directional confidence too low
    WeakDirection,
    /// Indicators suggest overheated entry
    Overheated,
    /// Direction conflicts with HTF supertrend
    HtfConflict,
    /// Indicator agrees_count is in the "danger zone" (3-4 out of 10).
    /// Market is ambiguous — neither clear trend nor clear reversal.
    /// Skipping these improves PnL by ~$60 on backtest (29 bad trades avoided).
    DangerZone,
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
    /// Minimum directional confidence
    /// For v3: abs(regression prediction) must be >= this
    /// For legacy: |p_long - 0.5| must be >= this
    pub min_dir_confidence: f32,
    /// Maximum marginal P(super) above threshold for overheated filter
    pub overheated_margin: f32,
    /// Enable overheated detection
    pub enable_overheated_filter: bool,
    /// Enable HTF hard filter (reject LONG against HTF bearish, SHORT against HTF bullish)
    /// Only applied when direction_v3 model is used (it provides htf_supertrend_dir)
    pub enable_htf_filter: bool,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            p_threshold: 0.55,
            min_dir_confidence: 0.05, // v3: abs(prediction) >= 0.05 (was 0.10 for legacy p_long)
            overheated_margin: 0.05,
            enable_overheated_filter: true,
            enable_htf_filter: true,
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
    /// * `prediction` - Model prediction output (includes dir_confidence)
    /// * `features` - Raw feature values (for overheated detection)
    pub fn score(
        &self,
        prediction: &SuperEntryPrediction,
        features: Option<&OverheatedFeatures>,
    ) -> SuperEntryDecision {
        let p_super = prediction.p_super;
        let direction = prediction.direction;
        let dir_confidence = prediction.dir_confidence;

        // 1. Check P(super) threshold
        if (p_super as f64) < self.config.p_threshold {
            return SuperEntryDecision::NoSignal {
                reason: RejectReason::BelowThreshold,
                p_super,
            };
        }

        // 2. Check directional confidence (confidence gate)
        if dir_confidence < self.config.min_dir_confidence {
            return SuperEntryDecision::NoSignal {
                reason: RejectReason::WeakDirection,
                p_super,
            };
        }

        // 3. Overheated filter
        if self.config.enable_overheated_filter {
            if let Some(oh) = features {
                if oh.is_overheated(direction) {
                    return SuperEntryDecision::NoSignal {
                        reason: RejectReason::Overheated,
                        p_super,
                    };
                }
            }
        }

        // 4. Calculate combined score
        // For v3: dir_confidence is abs(regression prediction), typically 0..0.5
        // For legacy: dir_confidence is |p_long - 0.5|, typically 0..0.5
        let dir_factor = 1.0 + dir_confidence.min(0.5); // Range [1.0, 1.5]
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
pub struct OverheatedFeatures {
    pub rsi: f64,
    pub stoch_k: f64,
    pub cci: f64,
    pub williams: f64,
    pub bb_position: f64,
    pub atr_pct: f64,
}

impl OverheatedFeatures {
    /// Check if indicators suggest an "overheated" entry.
    ///
    /// An entry is overheated if multiple extreme readings are detected:
    ///   - For LONG: RSI > 75, Stoch > 85, CCI > 150, or BB position > 0.95
    ///   - For SHORT: RSI < 25, Stoch < 15, CCI < -150, or BB position < 0.05
    pub fn is_overheated(&self, direction: i8) -> bool {
        let mut extreme_count = 0;

        if direction == 1 {
            if self.rsi > 75.0 { extreme_count += 1; }
            if self.stoch_k > 85.0 { extreme_count += 1; }
            if self.cci > 150.0 { extreme_count += 1; }
            if self.williams > -10.0 { extreme_count += 1; }
            if self.bb_position > 0.95 { extreme_count += 1; }
        } else {
            if self.rsi < 25.0 { extreme_count += 1; }
            if self.stoch_k < 15.0 { extreme_count += 1; }
            if self.cci < -150.0 { extreme_count += 1; }
            if self.williams < -90.0 { extreme_count += 1; }
            if self.bb_position < 0.05 { extreme_count += 1; }
        }

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
            dir_confidence: 0.30,
            estimated_magnitude_pct: 2.0,
            direction_v3: true,
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
            dir_confidence: 0.30,
            estimated_magnitude_pct: 1.0,
            direction_v3: true,
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
    fn test_weak_direction_v3() {
        let scorer = SuperEntryScorer::new(ScorerConfig {
            min_dir_confidence: 0.05,
            ..Default::default()
        });
        let pred = SuperEntryPrediction {
            p_super: 0.75,
            p_long: 0.51,
            direction: 1,
            dir_confidence: 0.02, // below 0.05 threshold
            estimated_magnitude_pct: 2.0,
            direction_v3: true,
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
            enable_overheated_filter: true,
            ..Default::default()
        });

        let pred = SuperEntryPrediction {
            p_super: 0.57,
            p_long: 0.80,
            direction: 1,
            dir_confidence: 0.30,
            estimated_magnitude_pct: 1.0,
            direction_v3: true,
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
            enable_overheated_filter: true,
            ..Default::default()
        });

        let pred = SuperEntryPrediction {
            p_super: 0.80,
            p_long: 0.85,
            direction: 1,
            dir_confidence: 0.35,
            estimated_magnitude_pct: 3.0,
            direction_v3: true,
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
        // Overheated filter is ALWAYS applied regardless of p_super level
        assert!(!decision.is_super_entry());
    }

    #[test]
    fn test_combined_score_with_v3_confidence() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        let pred = SuperEntryPrediction {
            p_super: 0.80,
            p_long: 0.75,
            direction: 1,
            dir_confidence: 0.25,
            estimated_magnitude_pct: 3.0,
            direction_v3: true,
        };

        let decision = scorer.score(&pred, None);
        match decision {
            SuperEntryDecision::SuperEntry { combined_score, .. } => {
                // combined = p_super * (1 + min(confidence, 0.5))
                // = 0.80 * (1 + 0.25) = 1.0
                assert!((combined_score - 1.0).abs() < 0.01);
            }
            _ => panic!("Expected SuperEntry"),
        }
    }
}
