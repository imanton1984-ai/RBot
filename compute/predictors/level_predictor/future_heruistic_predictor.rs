// compute/predictors/level_predictor/heuristic_predictor.rs

use anyhow::Result;
use crate::feature_view::FeatureView;
use crate::level_view::LevelView;

pub struct LevelPredictorHeuristic;

impl LevelPredictorHeuristic {
    pub fn new() -> Self { Self }

    pub fn predict(&self, symbol: &str, timeframe: &str, features: &FeatureView) -> Result<Option<(f64, f64, f64, f64)>> {
        // 1. Получаем уровни
        let sr_levels = features.get_sr_levels()?;
        if sr_levels.is_empty() { return Ok(None); }

        let atr = features.atr.unwrap_or(1.0);
        // Создаем LevelView (уже есть в коде)
        let level_view = LevelView::new(sr_levels, features.close, atr, features.timestamp, symbol.to_string(), timeframe.to_string(), 2)?;

        // 2. Ищем ближайший уровень
        let levels = level_view.get_near_levels();
        if let Some(level) = levels.first() {
            // Эвристика:
            // Если цена подходит к уровню на малом объеме -> отскок (Bounce)
            // Если на большом объеме + сильный моментум -> пробой (Breakout)
            
            let volume_spike = features.volume_spike.unwrap_or(0.0);
            let momentum = features.macd_histogram.unwrap_or(0.0).abs();
            
            let mut prob_break = 0.3;
            if volume_spike > 2.0 && momentum > 0.5 {
                prob_break = 0.85;
            }
            
            let prob_bounce = 1.0 - prob_break;
            let score = prob_break.max(prob_bounce); // Уверенность

            if score >= 0.80 {
                return Ok(Some((level.level_price, prob_bounce, prob_break, score)));
            }
        }

        Ok(None)
    }
}