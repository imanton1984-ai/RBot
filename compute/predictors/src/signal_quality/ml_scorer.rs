// signal_quality/ml_scorer.rs
//
// XGBoost-based signal quality predictor.
// Trained on backtested signal outcomes to learn which signal patterns are profitable.
//
// Model predicts a "win probability" in [0, 1] which is converted to a quality_multiplier.
//
// TRAINING WORKFLOW:
//   1. Backtester evaluates historical signals → TradeOutcome per signal
//   2. trainer.rs builds TrainingExample dataset → exports CSV/DMatrix
//   3. Python or Rust XGBoost trains model → saves .ubj artifact
//   4. MlQualityScorer loads .ubj and predicts on new signals
//
// NOT YET WIRED INTO PIPELINE — standalone, ready for integration after backtester.

use anyhow::Result;
use super::types::*;
use super::heuristic_scorer::HeuristicQualityScorer;

/// XGBoost-based signal quality scorer
pub struct MlQualityScorer {
    /// Path to the XGBoost model file (.ubj format)
    _model_path: String,
    /// Whether the model is loaded and ready for inference
    model_loaded: bool,
    /// Fallback heuristic scorer (used when model is not loaded)
    heuristic_fallback: HeuristicQualityScorer,
    /// Weight for ML vs heuristic blend (0.0 = pure heuristic, 1.0 = pure ML)
    ml_weight: f64,
    // TODO: Add actual XGBoost booster handle when integrating with ml/ module
    // booster: Option<crate::ml::xgb_runtime::XgbModel>,
}

impl MlQualityScorer {
    /// Create a new ML quality scorer.
    /// Attempts to load the XGBoost model from the given path.
    pub fn new(model_path: &str) -> Self {
        let model_loaded = std::path::Path::new(model_path).exists();

        if model_loaded {
            tracing::info!(
                target: "signal_quality",
                "ML quality model found at: {}. Will load on first prediction.",
                model_path
            );
        } else {
            tracing::warn!(
                target: "signal_quality",
                "ML quality model not found at: {}. Using heuristic-only fallback.",
                model_path
            );
        }

        Self {
            _model_path: model_path.to_string(),
            model_loaded,
            heuristic_fallback: HeuristicQualityScorer::new(),
            ml_weight: 0.6, // 60% ML, 40% heuristic when both available
        }
    }

    /// Score a signal's quality using ML model + heuristic blend.
    pub fn score(&self, features: &SignalFeatures) -> QualityResult {
        // Always compute heuristic score
        let heur_result = self.heuristic_fallback.score(features);
        let heur_quality = heur_result.breakdown.heuristic_quality;

        // Try ML prediction
        let ml_quality = if self.model_loaded {
            match self.predict_ml(features) {
                Ok(Some(q)) => Some(q),
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!(
                        target: "signal_quality",
                        "ML quality prediction failed: {}. Using heuristic only.",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        // Blend ML and heuristic
        let combined_quality = match ml_quality {
            Some(ml_q) => {
                // Weighted blend
                let blended = self.ml_weight * ml_q + (1.0 - self.ml_weight) * heur_quality;
                blended.clamp(0.0, 1.0)
            }
            None => heur_quality, // Pure heuristic
        };

        // Convert to multiplier: [0, 1] → [0.5, 1.5]
        let quality_multiplier = (0.5 + combined_quality).clamp(0.5, 1.5);

        let grade = match combined_quality {
            q if q >= 0.80 => SignalGrade::A,
            q if q >= 0.60 => SignalGrade::B,
            q if q >= 0.40 => SignalGrade::C,
            _ => SignalGrade::D,
        };

        QualityResult {
            quality_multiplier,
            grade,
            breakdown: QualityBreakdown {
                heuristic_quality: heur_quality,
                ml_quality,
                combined_quality,
                factors: heur_result.breakdown.factors,
            },
        }
    }

    /// Run XGBoost inference on the feature vector.
    /// Returns win probability in [0, 1].
    fn predict_ml(&self, features: &SignalFeatures) -> Result<Option<f64>> {
        if !self.model_loaded {
            return Ok(None);
        }

        let feature_vec = features.to_feature_vector();

        // TODO: Integrate with crate::ml::xgb_runtime::XgbModel
        // Pseudocode for when the model is integrated:
        //
        // let booster = self.booster.as_ref().ok_or(anyhow!("Model not loaded"))?;
        // let prediction = booster.predict_one(&feature_vec)?;
        // Ok(Some(prediction[0] as f64))
        //
        // For now, return None (heuristic-only mode)
        let _ = feature_vec; // suppress unused warning
        Ok(None)
    }

    /// Apply quality multiplier to a final_score.
    /// Caps the boost to prevent extreme values.
    pub fn apply_multiplier(final_score: f64, quality_result: &QualityResult) -> f64 {
        let boosted = final_score * quality_result.quality_multiplier;
        // Cap at 0.99 to leave room for improvement with better models
        boosted.clamp(0.0, 0.99)
    }

    /// Check if the ML model is available
    pub fn is_model_loaded(&self) -> bool {
        self.model_loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_to_heuristic() {
        let scorer = MlQualityScorer::new("/nonexistent/model.ubj");
        assert!(!scorer.is_model_loaded());

        let features = SignalFeatures {
            symbol: "BTCUSDT".into(),
            tf_minutes: 5,
            side: 1,
            final_score: 0.70,
            base_score: 0.77,
            predictors_score: 0.7,
            raw_signals_score: 0.6,
            indicators_score: 0.65,
            market_score: 0.6,
            coverage_score: 1.0,
            consensus_score: 0.92,
            ml_score: Some(0.7),
            heur_score: Some(0.65),
            price10_target: Some(70000.0),
            price10_score: Some(0.7),
            bounce_prob: Some(0.6),
            bounce_score: Some(0.5),
            breakout_prob: Some(0.4),
            breakout_score: Some(0.4),
            atr: Some(500.0),
            atr_pct: Some(0.007),
            trend_strength: Some(0.7),
            momentum_strength: Some(0.6),
            volatility_regime: Some(0.3),
            volume_spike_score: Some(0.5),
            level_aware: true,
            n_support_levels: 2,
            n_resistance_levels: 1,
            market_situation: Some("uptrend".into()),
            market_quality_score: Some(0.7),
            entry_price: 69000.0,
            stop_loss: 68500.0,
            tp1: 70000.0,
            tp2: 70800.0,
            tp3: 71900.0,
            risk_reward_ratio: 2.0,
            sl_pct: 500.0 / 69000.0,
            tp1_pct: 1000.0 / 69000.0,
        };

        let result = scorer.score(&features);
        assert!(result.breakdown.ml_quality.is_none());
        assert!(result.quality_multiplier >= 0.5 && result.quality_multiplier <= 1.5);
    }

    #[test]
    fn test_apply_multiplier_caps() {
        let result = QualityResult {
            quality_multiplier: 1.5,
            grade: SignalGrade::A,
            breakdown: QualityBreakdown {
                heuristic_quality: 1.0,
                ml_quality: Some(1.0),
                combined_quality: 1.0,
                factors: QualityFactors {
                    score_strength: 1.0,
                    component_agreement: 1.0,
                    risk_reward: 1.0,
                    level_confluence: 1.0,
                    market_alignment: 1.0,
                    predictor_confidence: 1.0,
                    source_agreement: 1.0,
                },
            },
        };

        let boosted = MlQualityScorer::apply_multiplier(0.80, &result);
        assert!(boosted <= 0.99, "Should cap at 0.99, got {}", boosted);
    }
}
