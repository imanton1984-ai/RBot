// compute/predictors/consensus.rs

use anyhow::Result;
use crate::types::{PredictionRow};
use crate::feature_view::FeatureView;

/// Consensus logic for combining hardcode and ML predictors
pub struct ConsensusEngine {}

impl ConsensusEngine {
    /// Creates a new consensus engine
    pub fn new() -> Self {
        Self {}
    }

    /// Applies gate and fuse logic to combine hardcode and ML predictors
    pub async fn apply_gate_and_fuse(
        &self,
        hard_predictors: Vec<PredictionRow>,
        ml_predictors: Vec<PredictionRow>,
        feature_view: &FeatureView,
    ) -> Result<Vec<PredictionRow>> {
        let mut consensus_predictors = Vec::new();

        // Group predictors by aspect and level (if applicable)
        use std::collections::HashMap;
        let mut grouped_hard: HashMap<String, Vec<PredictionRow>> = HashMap::new();
        let mut grouped_ml: HashMap<String, Vec<PredictionRow>> = HashMap::new();

        // Group hardcode predictors
        for pred in hard_predictors {
            let key = self.create_prediction_key(&pred);
            grouped_hard.entry(key).or_insert_with(Vec::new).push(pred);
        }

        // Group ML predictors
        for pred in ml_predictors {
            let key = self.create_prediction_key(&pred);
            grouped_ml.entry(key).or_insert_with(Vec::new).push(pred);
        }

        // Process each group
        for (key, hard_group) in grouped_hard {
            let ml_group = grouped_ml.remove(&key).unwrap_or_default();

            if hard_group.is_empty() && ml_group.is_empty() {
                continue;
            }

            if hard_group.is_empty() {
                // Only ML predictors exist
                consensus_predictors.extend(ml_group);
            } else if ml_group.is_empty() {
                // Only hardcode predictors exist
                consensus_predictors.extend(hard_group);
            } else {
                // Both exist, apply consensus logic
                let consensus_group = self.combine_predictors(&hard_group, &ml_group, feature_view).await?;
                consensus_predictors.extend(consensus_group);
            }
        }

        // Add remaining ML groups that didn't have hardcode counterparts
        for (_, ml_group) in grouped_ml {
            consensus_predictors.extend(ml_group);
        }

        Ok(consensus_predictors)
    }

    /// Combines hardcode and ML predictors for the same aspect/level
    async fn combine_predictors(
        &self,
        hard_predictors: &[PredictionRow],
        ml_predictors: &[PredictionRow],
        feature_view: &FeatureView,
    ) -> Result<Vec<PredictionRow>> {
        let mut combined_predictors = Vec::new();

        for hard_pred in hard_predictors {
            // Find corresponding ML prediction for the same aspect
            let ml_pred = ml_predictors.iter()
                .find(|ml| ml.aspect == hard_pred.aspect && 
                         self.levels_match(&ml.level_hash, &hard_pred.level_hash));

            if let Some(ml_pred) = ml_pred {
                // Apply gate and fuse logic
                let consensus_pred = self.apply_gate_and_fuse_logic(hard_pred, ml_pred, feature_view).await?;
                if let Some(consensus_pred) = consensus_pred {
                    combined_predictors.push(consensus_pred);
                }
            } else {
                // No corresponding ML prediction, use hardcode
                combined_predictors.push(hard_pred.clone());
            }
        }

        // Add ML predictors that don't have hardcode counterparts
        for ml_pred in ml_predictors {
            let has_hard_counterpart = hard_predictors.iter()
                .any(|hard| hard.aspect == ml_pred.aspect && 
                           self.levels_match(&hard.level_hash, &ml_pred.level_hash));

            if !has_hard_counterpart {
                combined_predictors.push(ml_pred.clone());
            }
        }

        Ok(combined_predictors)
    }

    /// Applies the gate and fuse logic to combine two predictors
    async fn apply_gate_and_fuse_logic(
        &self,
        hard_pred: &PredictionRow,
        ml_pred: &PredictionRow,
        feature_view: &FeatureView,
    ) -> Result<Option<PredictionRow>> {
        
        let hc_pred_score = hard_pred.score_norm as f64;
        let ml_pred_score = ml_pred.score_norm as f64;

        // Dynamic weights based on market conditions (replaces hardcoded ml_trust = 0.5)
        let mut w_hc = self.calculate_hardcode_weight(feature_view, hard_pred);
        let mut w_ml = self.calculate_ml_weight(feature_view, ml_pred);

        // Normalize so weights sum to 1.0
        let total = w_hc + w_ml;
        w_hc /= total;
        w_ml /= total;
        
        // Smooth gate: if the hardcore predictor is confident, it "opens the way" for ML
        let gate = 1.0 / (1.0 + f64::exp(-10.0 * (hc_pred_score.abs() - 0.5)));
        
        let fused_score = (w_hc * hc_pred_score + w_ml * ml_pred_score) * gate;

        // Create fused prediction
        let mut fused_pred = if ml_pred_score >= hc_pred_score { ml_pred.clone() } else { hard_pred.clone() };
        fused_pred.score_norm = fused_score as f32;

        // Update details
        if let Some(ref mut details) = fused_pred.details_json {
            if let serde_json::Value::Object(ref mut obj) = details {
                obj.insert("consensus_applied".to_string(), serde_json::Value::Bool(true));
                obj.insert("gate_value".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(gate).unwrap()));
                obj.insert("w_hc".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(w_hc).unwrap()));
                obj.insert("w_ml".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(w_ml).unwrap()));
                obj.insert("original_ml_score".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(ml_pred_score).unwrap()));
                obj.insert("original_hard_score".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(hc_pred_score).unwrap()));
                obj.insert("fused_score".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(fused_score).unwrap()));
            }
        } else {
            fused_pred.details_json = Some(serde_json::json!({
                "consensus_applied": true,
                "gate_value": gate,
                "w_hc": w_hc,
                "w_ml": w_ml,
                "original_ml_score": ml_pred_score,
                "original_hard_score": hc_pred_score,
                "fused_score": fused_score,
            }));
        }

        Ok(Some(fused_pred))
    }

    /// Calculates weight for hardcode prediction based on market conditions
    fn calculate_hardcode_weight(&self, feature_view: &FeatureView, _pred: &PredictionRow) -> f64 {
        // Calculate weight based on how well the market conditions align with hardcode assumptions
        let mut weight: f64 = 1.0;

        // Adjust weight based on volatility regime
        let atr = feature_view.indicators.atr;
        let atr_ratio = atr / feature_view.indicators.close;
        if atr_ratio > 0.05 { // Very high volatility
            weight *= 0.7; // Reduce hardcode weight
        } else if atr_ratio < 0.005 { // Very low volatility
            weight *= 0.8; // Reduce hardcode weight (may not be picking up moves)
        }

        // Adjust weight based on trend strength
        let trend_strength = feature_view.indicators.trend_short;
        if trend_strength.abs() > 0.7 { // Strong trend
            weight *= 1.1; // Increase hardcode weight
        } else if trend_strength.abs() < 0.2 { // Weak trend
            weight *= 0.8; // Decrease hardcode weight
        }

        // Adjust weight based on oscillator alignment
        let osc_alignment = self.calculate_oscillator_alignment(feature_view);
        if osc_alignment < 0.3 { // Poor alignment
            weight *= 0.7;
        } else if osc_alignment > 0.8 { // Good alignment
            weight *= 1.1;
        }

        weight.clamp(0.5_f64, 1.5_f64) // Clamp between 0.5x and 1.5x base weight
    }

    /// Calculates weight for ML prediction based on market conditions
    fn calculate_ml_weight(&self, feature_view: &FeatureView, _pred: &PredictionRow) -> f64 {
        // Calculate weight based on how well the market conditions align with ML training data
        let mut weight: f64 = 1.0;

        // Adjust weight based on volatility regime
        let atr = feature_view.indicators.atr;
        let atr_ratio = atr / feature_view.indicators.close;
        if atr_ratio > 0.05 { // Very high volatility
            weight *= 1.2; // ML might be better in high vol situations
        } else if atr_ratio < 0.005 { // Very low volatility
            weight *= 0.9; // ML might struggle with no movement
        }

        // Adjust weight based on market regime stability
        let regime_stability = self.calculate_regime_stability(feature_view);
        if regime_stability < 0.3 { // Unstable regime
            weight *= 0.8; // ML might be less reliable
        } else if regime_stability > 0.8 { // Stable regime
            weight *= 1.1; // ML should perform well
        }

        weight.clamp(0.5_f64, 1.5_f64) // Clamp between 0.5x and 1.5x base weight
    }

    /// Calculates how aligned oscillators are with each other
    fn calculate_oscillator_alignment(&self, feature_view: &FeatureView) -> f64 {
        let mut alignment_score = 0.0;
        let mut count = 0;

        let rsi = feature_view.indicators.rsi;
        // RSI in middle range is more reliable
        if rsi > 30.0 && rsi < 70.0 {
            alignment_score += 0.5;
        }
        count += 1;

        let stoch_k = feature_view.indicators.stoch_k;
        let stoch_d = feature_view.indicators.stoch_d;
        // Stochastic in middle range and K close to D
        if stoch_k > 20.0 && stoch_k < 80.0 && stoch_d > 20.0 && stoch_d < 80.0 {
            alignment_score += 0.3;
            // Bonus if K and D are close (confluence)
            if (stoch_k - stoch_d).abs() < 5.0 {
                alignment_score += 0.2;
            }
        }
        count += 1;

        let williams_r = feature_view.indicators.williams_r;
        if williams_r > -80.0 && williams_r < -20.0 {
            alignment_score += 0.5;
        }
        count += 1;

        if count > 0 {
            alignment_score / count as f64
        } else {
            0.0
        }
    }

    /// Calculates market regime stability
    fn calculate_regime_stability(&self, feature_view: &FeatureView) -> f64 {
        let mut stability_score = 0.0;
        let mut count = 0;

        // Check trend stability
        let trend_short = feature_view.indicators.trend_short;
        let trend_medium = feature_view.indicators.trend_medium;
        // If short and medium trends align, it's more stable
        if trend_short.signum() == trend_medium.signum() {
            stability_score += 0.4;
        } else {
            stability_score -= 0.2; // Conflicting trends
        }
        count += 1;

        // Check volume stability
        let volume_sma = feature_view.indicators.volume_sma;
        let volume = feature_view.indicators.volume;
        let volume_ratio = volume / volume_sma;
        // Stable volumes (close to average) indicate stable regime
        if volume_ratio > 0.7 && volume_ratio < 1.3 {
            stability_score += 0.3;
        }
        count += 1;

        // Check volatility stability
        let atr = feature_view.indicators.atr;
        // Very high or very low volatility might indicate unstable regime
        let atr_ratio = atr / feature_view.indicators.close;
        if atr_ratio > 0.005 && atr_ratio < 0.05 {
            stability_score += 0.3;
        }
        count += 1;

        if count > 0 {
            stability_score / count as f64
        } else {
            0.5 // Neutral if no data
        }
    }

    /// Creates a unique key for grouping predictors
    fn create_prediction_key(&self, pred: &PredictionRow) -> String {
        match &pred.level_hash {
            Some(hash) => format!("{}_{}_{}", pred.aspect.as_int(), hash, pred.symbol),
            None => format!("{}_{}", pred.aspect.as_int(), pred.symbol),
        }
    }

    /// Checks if two level hashes match (both None or both Some and equal)
    fn levels_match(&self, level1: &Option<String>, level2: &Option<String>) -> bool {
        match (level1, level2) {
            (Some(l1), Some(l2)) => l1 == l2,
            (None, None) => true,
            _ => false,
        }
    }

    /// Applies consensus logic to determine if predictors should be combined
    pub async fn should_combine_predictors(
        &self,
        hard_pred: &PredictionRow,
        ml_pred: &PredictionRow,
        _feature_view: &FeatureView,
    ) -> bool {
        // Don't combine if they're for different aspects
        if hard_pred.aspect != ml_pred.aspect {
            return false;
        }

        // Don't combine if they're for different levels (if applicable)
        if !self.levels_match(&hard_pred.level_hash, &ml_pred.level_hash) {
            return false;
        }

        // Don't combine if they're too far apart in time
        let time_diff = (hard_pred.time.timestamp_millis() - ml_pred.time.timestamp_millis()).abs();
        if time_diff > 10000 { // More than 10 seconds apart
            return false;
        }

        // Check if both predictors are for the same symbol and timeframe
        hard_pred.symbol == ml_pred.symbol && hard_pred.tf_minutes == ml_pred.tf_minutes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PredictionAspect, CalcSource};

    #[test]
    fn test_create_prediction_key() {
        let consensus = ConsensusEngine::new();
        
        let pred1 = PredictionRow {
            aspect: PredictionAspect::PriceTarget,
            symbol: "BTCUSDT".to_string(),
            level_hash: None,
            ..create_mock_prediction()
        };
        
        let pred2 = PredictionRow {
            aspect: PredictionAspect::LevelBounce,
            symbol: "BTCUSDT".to_string(),
            level_hash: Some("level_12345".to_string()),
            ..create_mock_prediction()
        };
        
        assert_eq!(consensus.create_prediction_key(&pred1), "1_BTCUSDT");
        assert_eq!(consensus.create_prediction_key(&pred2), "2_level_12345_BTCUSDT");
    }

    #[test]
    fn test_levels_match() {
        let consensus = ConsensusEngine::new();
        
        assert!(consensus.levels_match(&None, &None));
        assert!(!consensus.levels_match(&None, &Some("level_1".to_string())));
        assert!(!consensus.levels_match(&Some("level_1".to_string()), &None));
        assert!(consensus.levels_match(&Some("level_1".to_string()), &Some("level_1".to_string())));
        assert!(!consensus.levels_match(&Some("level_1".to_string()), &Some("level_2".to_string())));
    }

    // Helper function to create mock prediction
    fn create_mock_prediction() -> PredictionRow {
        use chrono::Utc;
        
        PredictionRow {
            time: Utc::now(),
            time_ms: Utc::now().timestamp_millis(),
            symbol_id: 1,
            symbol: "BTCUSDT".to_string(),
            tf_minutes: 5,
            horizon_bars: 10,
            aspect: PredictionAspect::PriceTarget,
            calc_source: CalcSource::Hard,
            predictor_id: 1,
            score_norm: 0.85,
            value: 100.0,
            value_low: None,
            value_high: None,
            side: Some(1),
            level_hash: None,
            level_kind: None,
            level_price: None,
            level_strength: None,
            level_distance_atr: None,
            candle_is_final: true,
            event_time_ms: None,
            details_json: None,
            prediction_key: "mock_key".to_string(),
        }
    }
}