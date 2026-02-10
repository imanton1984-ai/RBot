// compute/scoring/final_scorer.rs

use anyhow::Result;
use serde_json::Value;
use crate::predictions::types::PredictionRow;
use crate::predictions::persistence;

/// Final scorer that combines predictions with other signals to produce final scores
pub struct FinalScorer {
    min_final_score: f64,
    strategy_weights: StrategyWeights,
}

/// Weights for different strategies/components in the final scoring
#[derive(Debug, Clone)]
pub struct StrategyWeights {
    pub predictions_weight: f64,
    pub levels_context_weight: f64,
    pub trend_weight: f64,
    pub momentum_weight: f64,
    pub volatility_weight: f64,
    pub volume_weight: f64,
}

impl Default for StrategyWeights {
    fn default() -> Self {
        Self {
            predictions_weight: 0.3,
            levels_context_weight: 0.25,
            trend_weight: 0.15,
            momentum_weight: 0.15,
            volatility_weight: 0.075,
            volume_weight: 0.075,
        }
    }
}

impl FinalScorer {
    /// Creates a new final scorer with default weights
    pub fn new(min_final_score: f64) -> Self {
        Self {
            min_final_score,
            strategy_weights: StrategyWeights::default(),
        }
    }

    /// Creates a new final scorer with custom weights
    pub fn new_with_weights(min_final_score: f64, weights: StrategyWeights) -> Self {
        Self {
            min_final_score,
            strategy_weights: weights,
        }
    }

    /// Scores a signal based on predictions and other factors
    pub async fn score_signal(
        &self,
        symbol: &str,
        timeframe: &str,
        timestamp: chrono::DateTime<chrono::Utc>,
        raw_signals_summary: &Value,
        predictions: &[PredictionRow],
    ) -> Result<Option<f64>> {
        // Calculate base score from predictions
        let predictions_score = self.calculate_predictions_score(predictions, symbol, timeframe).await?;
        
        // Extract other signal components
        let levels_context_score = self.extract_levels_context(raw_signals_summary);
        let trend_score = self.extract_trend_score(raw_signals_summary);
        let momentum_score = self.extract_momentum_score(raw_signals_summary);
        let volatility_score = self.extract_volatility_score(raw_signals_summary);
        let volume_score = self.extract_volume_score(raw_signals_summary);

        // Combine all scores using weights
        let final_score = 
            predictions_score * self.strategy_weights.predictions_weight +
            levels_context_score * self.strategy_weights.levels_context_weight +
            trend_score * self.strategy_weights.trend_weight +
            momentum_score * self.strategy_weights.momentum_weight +
            volatility_score * self.strategy_weights.volatility_weight +
            volume_score * self.strategy_weights.volume_weight;

        // Only return score if it meets the minimum threshold
        if final_score >= self.min_final_score {
            Ok(Some(final_score))
        } else {
            Ok(None)
        }
    }

    /// Calculates score based on predictions
    async fn calculate_predictions_score(
        &self,
        predictions: &[PredictionRow],
        symbol: &str,
        timeframe: &str,
    ) -> Result<f64> {
        if predictions.is_empty() {
            return Ok(0.0);
        }

        // Group predictions by aspect
        use std::collections::HashMap;
        let mut aspect_scores: HashMap<i16, Vec<f64>> = HashMap::new();

        for pred in predictions {
            let aspect_key = pred.aspect.as_int();
            let score = pred.score_norm as f64;
            aspect_scores.entry(aspect_key).or_insert_with(Vec::new).push(score);
        }

        // Calculate weighted average of predictions
        let mut total_weighted_score = 0.0;
        let mut total_weight = 0.0;

        for (aspect, scores) in aspect_scores {
            // Calculate average score for this aspect
            let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
            
            // Assign weight based on aspect importance
            let aspect_weight = match aspect {
                1 => 1.0, // Price target
                2 => 0.8, // Level bounce
                3 => 0.8, // Level break
                _ => 0.5, // Other aspects
            };

            total_weighted_score += avg_score * aspect_weight;
            total_weight += aspect_weight;
        }

        if total_weight > 0.0 {
            Ok(total_weighted_score / total_weight)
        } else {
            Ok(0.0)
        }
    }

    /// Extracts levels context score from raw signals
    fn extract_levels_context(&self, raw_signals_summary: &Value) -> f64 {
        // Look for level-related signals in the raw signals
        let levels_signal = raw_signals_summary.get("levels_signal")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Also consider proximity to levels
        let level_proximity = raw_signals_summary.get("level_proximity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Combine both signals
        (levels_signal.abs() + level_proximity) / 2.0
    }

    /// Extracts trend score from raw signals
    fn extract_trend_score(&self, raw_signals_summary: &Value) -> f64 {
        // Look for trend signals
        let trend_strength = raw_signals_summary.get("trend_strength")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Look for trend direction agreement
        let trend_agreement = raw_signals_summary.get("trend_agreement")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Combine trend metrics
        (trend_strength.abs() + trend_agreement.abs()) / 2.0
    }

    /// Extracts momentum score from raw signals
    fn extract_momentum_score(&self, raw_signals_summary: &Value) -> f64 {
        // Look for momentum signals
        let momentum_strength = raw_signals_summary.get("momentum_strength")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Look for oscillator alignment
        let oscillator_alignment = raw_signals_summary.get("oscillator_alignment")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Combine momentum metrics
        (momentum_strength.abs() + oscillator_alignment.abs()) / 2.0
    }

    /// Extracts volatility score from raw signals
    fn extract_volatility_score(&self, raw_signals_summary: &Value) -> f64 {
        // Look for volatility regime signals
        let volatility_regime = raw_signals_summary.get("volatility_regime")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Look for ATR-based signals
        let atr_signal = raw_signals_summary.get("atr_signal")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Combine volatility metrics
        (volatility_regime.abs() + atr_signal.abs()) / 2.0
    }

    /// Extracts volume score from raw signals
    fn extract_volume_score(&self, raw_signals_summary: &Value) -> f64 {
        // Look for volume signals
        let volume_spike = raw_signals_summary.get("volume_spike")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Look for OBV trend
        let obv_trend = raw_signals_summary.get("obv_trend")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Combine volume metrics
        (volume_spike.abs() + obv_trend.abs()) / 2.0
    }

    /// Updates the strategy weights dynamically based on market conditions
    pub fn update_weights_based_on_conditions(&mut self, market_conditions: &MarketConditions) {
        // Adjust weights based on market conditions
        if market_conditions.is_trending {
            self.strategy_weights.trend_weight = 0.25;
            self.strategy_weights.momentum_weight = 0.20;
            self.strategy_weights.predictions_weight = 0.25;
            self.strategy_weights.levels_context_weight = 0.15;
            self.strategy_weights.volatility_weight = 0.075;
            self.strategy_weights.volume_weight = 0.075;
        } else if market_conditions.is_choppy {
            self.strategy_weights.levels_context_weight = 0.3;
            self.strategy_weights.predictions_weight = 0.25;
            self.strategy_weights.volatility_weight = 0.2;
            self.strategy_weights.trend_weight = 0.1;
            self.strategy_weights.momentum_weight = 0.075;
            self.strategy_weights.volume_weight = 0.075;
        } else if market_conditions.high_volatility {
            self.strategy_weights.volatility_weight = 0.2;
            self.strategy_weights.predictions_weight = 0.25;
            self.strategy_weights.volume_weight = 0.2;
            self.strategy_weights.levels_context_weight = 0.15;
            self.strategy_weights.trend_weight = 0.1;
            self.strategy_weights.momentum_weight = 0.1;
        }
    }

    /// Gets recent predictions for a symbol and timeframe
    pub async fn get_recent_predictions(
        &self,
        db_pool: &sqlx::PgPool,
        symbol_id: i64,
        tf_minutes: i32,
        min_score: f32,
        limit: i32,
    ) -> Result<Vec<PredictionRow>> {
        persistence::get_recent_predictions(db_pool, symbol_id, tf_minutes, 
                                         crate::predictions::types::PredictionAspect::PriceTarget, 
                                         min_score, limit).await
    }
}

/// Market conditions that affect strategy weights
#[derive(Debug, Clone)]
pub struct MarketConditions {
    pub is_trending: bool,
    pub is_choppy: bool,
    pub high_volatility: bool,
    pub high_volume: bool,
    pub trend_direction: Option<i8>, // None = neutral, 1 = up, -1 = down
}

impl MarketConditions {
    pub fn new() -> Self {
        Self {
            is_trending: false,
            is_choppy: false,
            high_volatility: false,
            high_volume: false,
            trend_direction: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_weights_default() {
        let weights = StrategyWeights::default();
        assert_eq!(weights.predictions_weight, 0.3);
        assert_eq!(weights.levels_context_weight, 0.25);
        assert_eq!(weights.trend_weight, 0.15);
        assert_eq!(weights.momentum_weight, 0.15);
        assert_eq!(weights.volatility_weight, 0.075);
        assert_eq!(weights.volume_weight, 0.075);
    }

    #[test]
    fn test_final_scorer_creation() {
        let scorer = FinalScorer::new(0.90);
        assert_eq!(scorer.min_final_score, 0.90);
    }

    #[test]
    fn test_market_conditions_new() {
        let conditions = MarketConditions::new();
        assert!(!conditions.is_trending);
        assert!(!conditions.is_choppy);
        assert!(!conditions.high_volatility);
        assert!(!conditions.high_volume);
        assert_eq!(conditions.trend_direction, None);
    }
}