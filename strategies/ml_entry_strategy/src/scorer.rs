// strategies/ml_entry_strategy/src/scorer.rs
//
// Scorer for Super Entry Strategy (NoDir — P(super_long) + P(super_short))
//
// Decision logic (SYMMETRIC — identical to backtest):
//   1. If P(super_long) >= threshold AND P(super_short) < threshold → LONG
//   2. If P(super_short) >= threshold AND P(super_long) < threshold → SHORT
//   3. CONFLICT: Both >= threshold → pick higher probability (like backtest)
//   4. Neither >= threshold → BelowThreshold
//
// UNIFIED thresholds per TF (same for LONG and SHORT):
//   Eliminates directional bias. If models have different distributions,
//   the fix is model calibration (Platt scaling), NOT asymmetric thresholds.
//
// Per-TF thresholds (from backtest validation):
//   5m:  0.85 (noisy TF, high bar)
//   15m: 0.85
//   1h:  0.75
//   4h:  0.75
//   1d:  0.75

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
        /// P(super) for the chosen direction (p_super_long or p_super_short)
        p_super: f32,
        /// Directional confidence = margin between long and short probabilities
        /// Stored for backward compat with DB schema (dir_confidence column)
        dir_confidence: f32,
        /// Combined score (p_super * margin_factor)
        combined_score: f32,
    },
    /// Below threshold — no signal
    NoSignal {
        /// Reason for rejection
        reason: RejectReason,
        /// Max of (p_super_long, p_super_short) for diagnostics
        p_super: f32,
    },
}

/// Reason why a signal was not generated
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RejectReason {
    /// Both P(super_long) and P(super_short) below threshold
    BelowThreshold,
    /// DEPRECATED: kept for backward compat (conflict now resolved by picking higher p)
    ConflictAmbiguous,
    /// DEPRECATED: kept for backward compat
    WeakDirection,
    /// DEPRECATED
    Overheated,
    /// DEPRECATED
    HtfConflict,
    /// DEPRECATED
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

/// Per-TF thresholds — UNIFIED for both LONG and SHORT.
/// Kept as struct for backward compat but long_threshold == short_threshold.
#[derive(Debug, Clone)]
pub struct TfThresholds {
    pub long_threshold: f32,
    pub short_threshold: f32,
}

impl TfThresholds {
    /// Create unified threshold (same for both directions)
    pub fn unified(threshold: f32) -> Self {
        Self {
            long_threshold: threshold,
            short_threshold: threshold,
        }
    }
}

/// Configuration for the scorer
#[derive(Debug, Clone)]
pub struct ScorerConfig {
    /// Per-TF thresholds — UNIFIED (same for LONG and SHORT).
    /// Key = tf_minutes
    pub tf_thresholds: std::collections::HashMap<i32, TfThresholds>,
    /// Fallback threshold if TF not in map (same for both directions)
    pub default_long_threshold: f32,
    pub default_short_threshold: f32,
    /// DEPRECATED: conflict is now resolved by picking higher p (like backtest)
    pub conflict_min_margin: f32,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        let mut tf_thresholds = std::collections::HashMap::new();

        // UNIFIED thresholds — same for LONG and SHORT.
        // Eliminates directional bias. Values from backtest validation:
        //   15m=0.85 → positive WR in backtest
        //   1h=0.75  → positive WR in backtest
        //   4h=0.75  → positive WR in backtest
        tf_thresholds.insert(5, TfThresholds::unified(0.85));
        tf_thresholds.insert(15, TfThresholds::unified(0.85));
        tf_thresholds.insert(60, TfThresholds::unified(0.75));
        tf_thresholds.insert(240, TfThresholds::unified(0.75));
        tf_thresholds.insert(1440, TfThresholds::unified(0.75));

        Self {
            tf_thresholds,
            default_long_threshold: 0.75,
            default_short_threshold: 0.75,
            conflict_min_margin: 0.0, // Not used — conflict resolved by picking higher p
        }
    }
}

impl From<&SuperEntryConfig> for ScorerConfig {
    fn from(_cfg: &SuperEntryConfig) -> Self {
        Self::default()
    }
}

/// Super Entry Scorer (NoDir)
///
/// Evaluates predictions from P(super_long) + P(super_short) models.
/// Uses UNIFIED thresholds per TF (same for both directions) to eliminate
/// directional bias. Conflict resolved by picking higher probability
/// (identical to backtest logic).
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

    /// Score a prediction (NoDir).
    ///
    /// SYMMETRIC logic identical to backtest:
    ///   - If P(super_long) >= threshold → LONG candidate
    ///   - If P(super_short) >= threshold → SHORT candidate
    ///   - If both → pick higher probability
    ///   - If neither → no signal
    pub fn score(
        &self,
        prediction: &SuperEntryPrediction,
        tf_minutes: i32,
    ) -> SuperEntryDecision {
        let p_long = prediction.p_super_long;
        let p_short = prediction.p_super_short;

        // UNIFIED per-TF threshold (same for both directions)
        let threshold = match self.config.tf_thresholds.get(&tf_minutes) {
            Some(t) => t.long_threshold, // == short_threshold (unified)
            None => self.config.default_long_threshold,
        };

        let long_fires = p_long >= threshold;
        let short_fires = p_short >= threshold;

        match (long_fires, short_fires) {
            // Case 1: Only LONG fires
            (true, false) => {
                let margin = (p_long - p_short).abs();
                let combined = p_long * (1.0 + margin.min(0.5));
                SuperEntryDecision::SuperEntry {
                    direction: 1,
                    p_super: p_long,
                    dir_confidence: margin,
                    combined_score: combined,
                }
            }
            // Case 2: Only SHORT fires
            (false, true) => {
                let margin = (p_short - p_long).abs();
                let combined = p_short * (1.0 + margin.min(0.5));
                SuperEntryDecision::SuperEntry {
                    direction: -1,
                    p_super: p_short,
                    dir_confidence: margin,
                    combined_score: combined,
                }
            }
            // Case 3: CONFLICT — both fire → pick higher probability (like backtest)
            (true, true) => {
                let margin = (p_long - p_short).abs();
                if p_long >= p_short {
                    let combined = p_long * (1.0 + margin.min(0.5));
                    SuperEntryDecision::SuperEntry {
                        direction: 1,
                        p_super: p_long,
                        dir_confidence: margin,
                        combined_score: combined,
                    }
                } else {
                    let combined = p_short * (1.0 + margin.min(0.5));
                    SuperEntryDecision::SuperEntry {
                        direction: -1,
                        p_super: p_short,
                        dir_confidence: margin,
                        combined_score: combined,
                    }
                }
            }
            // Case 4: Neither fires
            (false, false) => {
                SuperEntryDecision::NoSignal {
                    reason: RejectReason::BelowThreshold,
                    p_super: p_long.max(p_short),
                }
            }
        }
    }

    /// Legacy score method for backward compatibility.
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
    /// DEPRECATED: Always returns false.
    pub fn is_overheated(&self, _direction: i8) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pred(p_long: f32, p_short: f32) -> SuperEntryPrediction {
        SuperEntryPrediction {
            p_super_long: p_long,
            p_super_short: p_short,
            direction: if p_long >= p_short { 1 } else { -1 },
            p_super: p_long.max(p_short),
            conflict_margin: (p_long - p_short).abs(),
            is_conflict: p_long > 0.5 && p_short > 0.5,
            estimated_magnitude_pct: 5.0,
        }
    }

    #[test]
    fn test_long_fires_above_unified_threshold() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        // 1h: unified threshold = 0.75
        // P(super_long)=0.80 >= 0.75 → LONG fires
        // P(super_short)=0.40 < 0.75 → SHORT doesn't
        let pred = make_pred(0.80, 0.40);

        let decision = scorer.score(&pred, 60);
        assert!(decision.is_super_entry());
        assert_eq!(decision.direction(), Some(1));
    }

    #[test]
    fn test_short_fires_above_unified_threshold() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        // 1h: unified threshold = 0.75
        // P(super_short)=0.80 >= 0.75, P(super_long)=0.40 < 0.75
        let pred = make_pred(0.40, 0.80);

        let decision = scorer.score(&pred, 60);
        assert!(decision.is_super_entry());
        assert_eq!(decision.direction(), Some(-1));
    }

    #[test]
    fn test_neither_fires() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        // 1h: unified threshold = 0.75
        // P(long)=0.50 < 0.75, P(short)=0.60 < 0.75
        let pred = make_pred(0.50, 0.60);

        let decision = scorer.score(&pred, 60);
        assert!(!decision.is_super_entry());
        match decision {
            SuperEntryDecision::NoSignal { reason, .. } => {
                assert_eq!(reason, RejectReason::BelowThreshold);
            }
            _ => panic!("Expected BelowThreshold"),
        }
    }

    #[test]
    fn test_conflict_resolved_by_higher_p() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        // 1h: unified threshold = 0.75
        // Both fire: P(long)=0.80, P(short)=0.85
        // SHORT wins because higher probability (like backtest)
        let pred = make_pred(0.80, 0.85);

        let decision = scorer.score(&pred, 60);
        assert!(decision.is_super_entry());
        assert_eq!(decision.direction(), Some(-1)); // SHORT wins — higher p
    }

    #[test]
    fn test_conflict_long_wins_when_higher() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        // 1h: unified threshold = 0.75
        // Both fire: P(long)=0.90, P(short)=0.80
        // LONG wins because higher probability
        let pred = make_pred(0.90, 0.80);

        let decision = scorer.score(&pred, 60);
        assert!(decision.is_super_entry());
        assert_eq!(decision.direction(), Some(1)); // LONG wins — higher p
    }

    #[test]
    fn test_symmetric_thresholds_no_bias() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        // 1h: unified threshold = 0.75
        // Same probability for both → LONG (default tie-break)
        let pred = make_pred(0.76, 0.76);
        let decision = scorer.score(&pred, 60);
        assert!(decision.is_super_entry());
        assert_eq!(decision.direction(), Some(1)); // Tie → LONG (p_long >= p_short)

        // Slightly higher short → SHORT wins
        let pred2 = make_pred(0.76, 0.77);
        let decision2 = scorer.score(&pred2, 60);
        assert!(decision2.is_super_entry());
        assert_eq!(decision2.direction(), Some(-1)); // SHORT wins
    }

    #[test]
    fn test_15m_needs_085() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        // 15m: unified threshold = 0.85
        // P(long)=0.80 < 0.85 → doesn't fire
        let pred = make_pred(0.80, 0.40);
        let decision = scorer.score(&pred, 15);
        assert!(!decision.is_super_entry());

        // P(long)=0.88 >= 0.85 → fires
        let pred2 = make_pred(0.88, 0.40);
        let decision2 = scorer.score(&pred2, 15);
        assert!(decision2.is_super_entry());
        assert_eq!(decision2.direction(), Some(1));

        // P(short)=0.80 < 0.85 → doesn't fire
        let pred3 = make_pred(0.30, 0.80);
        let decision3 = scorer.score(&pred3, 15);
        assert!(!decision3.is_super_entry());

        // P(short)=0.88 >= 0.85 → fires
        let pred4 = make_pred(0.30, 0.88);
        let decision4 = scorer.score(&pred4, 15);
        assert!(decision4.is_super_entry());
        assert_eq!(decision4.direction(), Some(-1));
    }

    #[test]
    fn test_combined_score_for_long() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        // 1h: P(long)=0.80, P(short)=0.40
        // margin = 0.40, combined = 0.80 * (1 + 0.40) = 1.12
        let pred = make_pred(0.80, 0.40);
        let decision = scorer.score(&pred, 60);

        match decision {
            SuperEntryDecision::SuperEntry { p_super, combined_score, dir_confidence, .. } => {
                assert!((p_super - 0.80).abs() < 1e-5);
                assert!((dir_confidence - 0.40).abs() < 1e-5);
                assert!((combined_score - 1.12).abs() < 1e-3);
            }
            _ => panic!("Expected SuperEntry"),
        }
    }

    #[test]
    fn test_below_threshold_not_biased() {
        let scorer = SuperEntryScorer::new(ScorerConfig::default());
        // 1h: unified threshold = 0.75
        // P(long)=0.60 — would have fired with old long_thresh=0.55, now doesn't
        let pred = make_pred(0.60, 0.40);
        let decision = scorer.score(&pred, 60);
        assert!(!decision.is_super_entry()); // No bias — same threshold for both
    }
}
