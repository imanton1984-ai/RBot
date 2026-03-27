// strategies/ml_entry_strategy/src/scorer.rs
//
// Scorer for Super Entry Strategy (v2 — ML-only)
//
// Decision logic (SIMPLIFIED — only ML models matter):
//   1. If p_super >= p_threshold → potential super signal
//   2. Direction from Direction model (v4 preferred, v3/legacy fallback)
//   3. Confidence gate: dir_confidence >= per-TF threshold
//      - 15m: ≥ 0.75
//      - 1h:  ≥ 0.70
//      - 4h:  ≥ 0.65
//   4. Generate signal with combined score
//
// REMOVED (v2):
//   - Overheated (indicator) filter — was indicator-based, not ML
//   - HTF hard filter — now subsumed by direction v4 model's pattern learning
//   - Danger zone filter — indicator agrees_count removed from pipeline
//   - Heuristic cross-TF filter — removed from super_entry_stage.rs
//
// Only P(super) and Direction model confidence affect signal generation.

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
        /// Directional confidence
        /// v4: P(predicted_class) ∈ [0.5, 1.0]
        /// v3: abs(regression)
        /// legacy: |p_long - 0.5|
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
    /// Directional confidence too low (per-TF threshold)
    WeakDirection,
    /// DEPRECATED: Indicators suggest overheated entry (no longer used)
    Overheated,
    /// DEPRECATED: Direction conflicts with HTF supertrend (no longer used)
    HtfConflict,
    /// DEPRECATED: Indicator agrees_count in danger zone (no longer used)
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
    /// Per-TF direction confidence thresholds (from SuperEntryConfig)
    /// Key = tf_minutes, Value = min P(predicted_class)
    pub direction_confidence_thresholds: std::collections::HashMap<i32, f64>,
    /// Fallback minimum directional confidence if TF not in thresholds map
    pub default_min_dir_confidence: f64,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        let cfg = SuperEntryConfig::default();
        Self {
            p_threshold: 0.55,
            direction_confidence_thresholds: cfg.direction_confidence_thresholds,
            default_min_dir_confidence: 0.65,
        }
    }
}

impl From<&SuperEntryConfig> for ScorerConfig {
    fn from(cfg: &SuperEntryConfig) -> Self {
        Self {
            p_threshold: cfg.p_threshold,
            direction_confidence_thresholds: cfg.direction_confidence_thresholds.clone(),
            default_min_dir_confidence: 0.65,
        }
    }
}

/// Super Entry Scorer
///
/// Evaluates predictions from the model and makes entry decisions.
/// Only ML model outputs (P(super) + direction confidence) affect decisions.
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
    /// Only ML models (P(super) + direction) influence the decision.
    /// No indicator-based filters (overheated, danger zone, heuristic).
    ///
    /// # Arguments
    /// * `prediction` - Model prediction output (includes dir_confidence)
    /// * `tf_minutes` - Timeframe for per-TF confidence threshold
    pub fn score(
        &self,
        prediction: &SuperEntryPrediction,
        tf_minutes: i32,
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

        // 2. Check directional confidence (per-TF threshold)
        let min_confidence = self.config.direction_confidence_thresholds
            .get(&tf_minutes)
            .copied()
            .unwrap_or(self.config.default_min_dir_confidence);

        if (dir_confidence as f64) < min_confidence {
            return SuperEntryDecision::NoSignal {
                reason: RejectReason::WeakDirection,
                p_super,
            };
        }

        // 3. Calculate combined score
        // For v4: dir_confidence is P(predicted_class) ∈ [0.5, 1.0]
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

    /// Legacy score method that accepts OverheatedFeatures for backward compatibility.
    /// Ignores the features — only ML models affect decision.
    pub fn score_legacy(
        &self,
        prediction: &SuperEntryPrediction,
        _features: Option<&OverheatedFeatures>,
        tf_minutes: i32,
    ) -> SuperEntryDecision {
        self.score(prediction, tf_minutes)
    }

    /// Get the config
    pub fn config(&self) -> &ScorerConfig {
        &self.config
    }
}

/// Features used for overheated detection (DEPRECATED — kept for backward compat).
pub struct OverheatedFeatures {
    pub rsi: f64,
    pub stoch_k: f64,
    pub cci: f64,
    pub williams: f64,
    pub bb_position: f64,
    pub atr_pct: f64,
}

impl OverheatedFeatures {
    /// DEPRECATED: Always returns false. Overheated filter is disabled.
    /// Only ML model confidence gates are used for signal filtering.
    pub fn is_overheated(&self, _direction: i8) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DirectionModelVersion;

    #[test]
    fn test_super_entry_above_threshold() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        let pred = SuperEntryPrediction {
            p_super: 0.75,
            p_long: 0.80,
            direction: 1,
            dir_confidence: 0.80, // v4: P(UP) = 0.80, above any TF threshold
            estimated_magnitude_pct: 2.0,
            direction_model_version: DirectionModelVersion::V4,
        };

        let decision = scorer.score(&pred, 60); // 1h TF
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
            dir_confidence: 0.80,
            estimated_magnitude_pct: 1.0,
            direction_model_version: DirectionModelVersion::V4,
        };

        let decision = scorer.score(&pred, 60);
        assert!(!decision.is_super_entry());
        match decision {
            SuperEntryDecision::NoSignal { reason, .. } => {
                assert_eq!(reason, RejectReason::BelowThreshold);
            }
            _ => panic!("Expected NoSignal"),
        }
    }

    #[test]
    fn test_weak_direction_per_tf() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());

        // v4 model: P(UP) = 0.68 → confidence = 0.68
        let pred = SuperEntryPrediction {
            p_super: 0.75,
            p_long: 0.68,
            direction: 1,
            dir_confidence: 0.68,
            estimated_magnitude_pct: 2.0,
            direction_model_version: DirectionModelVersion::V4,
        };

        // For 15m (threshold 0.75): should be rejected (0.68 < 0.75)
        let decision_15m = scorer.score(&pred, 15);
        assert!(!decision_15m.is_super_entry());
        match decision_15m {
            SuperEntryDecision::NoSignal { reason, .. } => {
                assert_eq!(reason, RejectReason::WeakDirection);
            }
            _ => panic!("Expected WeakDirection for 15m"),
        }

        // For 1h (threshold 0.70): should be rejected (0.68 < 0.70)
        let decision_1h = scorer.score(&pred, 60);
        assert!(!decision_1h.is_super_entry());

        // For 4h (threshold 0.65): should PASS (0.68 >= 0.65)
        let decision_4h = scorer.score(&pred, 240);
        assert!(decision_4h.is_super_entry());
    }

    #[test]
    fn test_no_overheated_filter() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());

        let pred = SuperEntryPrediction {
            p_super: 0.80,
            p_long: 0.85,
            direction: 1,
            dir_confidence: 0.85, // v4 confidence
            estimated_magnitude_pct: 3.0,
            direction_model_version: DirectionModelVersion::V4,
        };

        // Should pass even with "overheated" indicators — no indicator filter
        let decision = scorer.score(&pred, 240);
        assert!(decision.is_super_entry());
    }

    #[test]
    fn test_combined_score_with_v4_confidence() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        let pred = SuperEntryPrediction {
            p_super: 0.80,
            p_long: 0.75,
            direction: 1,
            dir_confidence: 0.75,
            estimated_magnitude_pct: 3.0,
            direction_model_version: DirectionModelVersion::V4,
        };

        let decision = scorer.score(&pred, 240);
        match decision {
            SuperEntryDecision::SuperEntry { combined_score, .. } => {
                // combined = p_super * (1 + min(0.75, 0.5))
                // = 0.80 * 1.5 = 1.2
                assert!((combined_score - 1.2).abs() < 0.01);
            }
            _ => panic!("Expected SuperEntry"),
        }
    }
}
