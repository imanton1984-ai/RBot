// strategies/super_level_strategy/src/scorer.rs
//
// Multi-Model Scorer for Super Level Strategy
//
// Оценивает ансамбль из 5 моделей и выдаёт финальное решение:
//   1. Level Model  — минимальный порог P(strong_level) ≥ 0.45
//   2. Entry Model  — минимальный порог P(good_entry) ≥ 0.50
//   3. Direction     — определяет LONG/SHORT с уверенностью
//   4. BounceBreak   — определяет сценарий (bounce/breakout) с уверенностью
//   5. Evaluator     — финальный вердикт P(win) ≥ p_threshold
//
// Cascading filter: если любая модель не прошла порог → NoSignal.
// Модель Evaluator имеет решающее слово.

use crate::config::SuperLevelConfig;
use crate::model::SuperLevelPrediction;
use serde::{Deserialize, Serialize};

/// Финальное решение скорера
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SuperLevelDecision {
    /// Все модели согласны — входим
    Signal {
        direction: i8,       // 1=LONG, -1=SHORT
        is_bounce: bool,     // true=bounce, false=breakout
        p_level: f32,        // P(strong_level)
        p_entry: f32,        // P(good_entry)
        p_direction: f32,    // P(LONG) or 1-P(LONG) depending on dir
        p_bounce: f32,       // P(bounce)
        p_eval: f32,         // P(win) — evaluator
        combined_score: f32,  // weighted combination
    },
    /// Не прошёл хотя бы одну модель
    NoSignal {
        reason: RejectReason,
        p_eval: f32,
    },
}

/// Причины отклонения
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RejectReason {
    /// Level model: уровень слишком слабый
    WeakLevel,
    /// Entry model: плохая точка входа
    BadEntry,
    /// Direction: низкая уверенность в направлении
    WeakDirection,
    /// BounceBreak: неясный сценарий
    UnclearScenario,
    /// Evaluator: финальная оценка ниже порога
    EvaluatorRejected,
}

impl SuperLevelDecision {
    pub fn is_signal(&self) -> bool {
        matches!(self, SuperLevelDecision::Signal { .. })
    }

    pub fn direction(&self) -> Option<i8> {
        match self {
            SuperLevelDecision::Signal { direction, .. } => Some(*direction),
            _ => None,
        }
    }

    pub fn p_eval(&self) -> f32 {
        match self {
            SuperLevelDecision::Signal { p_eval, .. } => *p_eval,
            SuperLevelDecision::NoSignal { p_eval, .. } => *p_eval,
        }
    }
}

/// Конфигурация порогов скорера
#[derive(Debug, Clone)]
pub struct ScorerConfig {
    /// Минимальный P(strong_level) для Level Model
    pub min_p_level: f32,
    /// Минимальный P(good_entry) для Entry Model
    pub min_p_entry: f32,
    /// Минимальная уверенность в направлении (|p_long - 0.5|)
    pub min_dir_confidence: f32,
    /// Минимальная уверенность в bounce/break (|p_bounce - 0.5|)
    pub min_scenario_confidence: f32,
    /// Минимальный P(win) от Evaluator
    pub min_p_eval: f64,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            min_p_level: 0.45,
            min_p_entry: 0.50,
            min_dir_confidence: 0.08,
            min_scenario_confidence: 0.05,
            min_p_eval: 0.55,
        }
    }
}

impl From<&SuperLevelConfig> for ScorerConfig {
    fn from(cfg: &SuperLevelConfig) -> Self {
        Self {
            min_p_eval: cfg.p_threshold,
            ..Default::default()
        }
    }
}

/// Multi-model scorer
pub struct SuperLevelScorer {
    config: ScorerConfig,
}

impl SuperLevelScorer {
    pub fn new(config: ScorerConfig) -> Self {
        Self { config }
    }

    pub fn from_strategy_config(cfg: &SuperLevelConfig) -> Self {
        Self::new(ScorerConfig::from(cfg))
    }

    /// Score a prediction from all 5 models
    pub fn score(&self, pred: &SuperLevelPrediction) -> SuperLevelDecision {
        // ── Gate 1: Level quality ──
        if pred.p_level < self.config.min_p_level {
            return SuperLevelDecision::NoSignal {
                reason: RejectReason::WeakLevel,
                p_eval: pred.p_eval,
            };
        }

        // ── Gate 2: Entry quality ──
        if pred.p_entry < self.config.min_p_entry {
            return SuperLevelDecision::NoSignal {
                reason: RejectReason::BadEntry,
                p_eval: pred.p_eval,
            };
        }

        // ── Gate 3: Direction confidence ──
        let dir_confidence = (pred.p_long - 0.5).abs();
        if dir_confidence < self.config.min_dir_confidence {
            return SuperLevelDecision::NoSignal {
                reason: RejectReason::WeakDirection,
                p_eval: pred.p_eval,
            };
        }

        // ── Gate 4: Scenario confidence ──
        let scenario_confidence = (pred.p_bounce - 0.5).abs();
        if scenario_confidence < self.config.min_scenario_confidence {
            return SuperLevelDecision::NoSignal {
                reason: RejectReason::UnclearScenario,
                p_eval: pred.p_eval,
            };
        }

        // ── Gate 5: Evaluator (final verdict) ──
        if (pred.p_eval as f64) < self.config.min_p_eval {
            return SuperLevelDecision::NoSignal {
                reason: RejectReason::EvaluatorRejected,
                p_eval: pred.p_eval,
            };
        }

        // All gates passed — generate signal
        let direction = pred.direction;
        let is_bounce = pred.is_bounce;
        let p_direction = if direction == 1 { pred.p_long } else { 1.0 - pred.p_long };

        // Combined score: weighted average of all model confidences
        let combined_score = 0.15 * pred.p_level
            + 0.20 * pred.p_entry
            + 0.20 * p_direction
            + 0.10 * (0.5 + scenario_confidence)
            + 0.35 * pred.p_eval;

        SuperLevelDecision::Signal {
            direction,
            is_bounce,
            p_level: pred.p_level,
            p_entry: pred.p_entry,
            p_direction,
            p_bounce: pred.p_bounce,
            p_eval: pred.p_eval,
            combined_score,
        }
    }

    pub fn config(&self) -> &ScorerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_gates_pass() {
        let scorer = SuperLevelScorer::new(ScorerConfig::default());
        let pred = SuperLevelPrediction {
            p_level: 0.7,
            p_entry: 0.65,
            p_long: 0.8,
            p_bounce: 0.7,
            p_eval: 0.75,
            direction: 1,
            is_bounce: true,
        };
        let dec = scorer.score(&pred);
        assert!(dec.is_signal());
        assert_eq!(dec.direction(), Some(1));
    }

    #[test]
    fn test_weak_level_reject() {
        let scorer = SuperLevelScorer::new(ScorerConfig::default());
        let pred = SuperLevelPrediction {
            p_level: 0.3, // below 0.45
            p_entry: 0.65,
            p_long: 0.8,
            p_bounce: 0.7,
            p_eval: 0.75,
            direction: 1,
            is_bounce: true,
        };
        let dec = scorer.score(&pred);
        assert!(!dec.is_signal());
        match dec {
            SuperLevelDecision::NoSignal { reason, .. } => assert_eq!(reason, RejectReason::WeakLevel),
            _ => panic!("Expected NoSignal"),
        }
    }

    #[test]
    fn test_evaluator_reject() {
        let scorer = SuperLevelScorer::new(ScorerConfig::default());
        let pred = SuperLevelPrediction {
            p_level: 0.7,
            p_entry: 0.65,
            p_long: 0.8,
            p_bounce: 0.7,
            p_eval: 0.40, // below 0.55
            direction: 1,
            is_bounce: true,
        };
        let dec = scorer.score(&pred);
        assert!(!dec.is_signal());
        match dec {
            SuperLevelDecision::NoSignal { reason, .. } => assert_eq!(reason, RejectReason::EvaluatorRejected),
            _ => panic!("Expected NoSignal"),
        }
    }
}
