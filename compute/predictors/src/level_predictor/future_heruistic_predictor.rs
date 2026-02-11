// compute/predictors/level_predictor/heuristic_predictor.rs

use anyhow::Result;
use crate::feature_view::FeatureView;
use crate::level_view::{LevelView, ProcessedLevel};

pub struct LevelPredictorHeuristic;

impl LevelPredictorHeuristic {
    pub fn new() -> Self { Self }

    pub fn predict(
        &self,
        symbol: &str,
        timeframe: &str,
        features: &FeatureView,
    ) -> Result<Option<(ProcessedLevel, f64, f64, f64)>> {

        let sr_levels = features.get_sr_levels()?;
        if sr_levels.is_empty() {
            return Ok(None);
        }

        let atr = features.indicators.atr;

        let level_view = LevelView::new(
            sr_levels,
            features.indicators.close as f64,
            atr as f64,
            features.timestamp,
            symbol.to_string(),
            timeframe.to_string(),
            2,
        )?;

        // ⬇️ ВАЖНО: если near нет — берём ближайший уровень
        let level = level_view
            .get_near_levels()
            .first()
            .cloned()
            .or_else(|| level_view.levels.first());

        let Some(level) = level else {
            return Ok(None);
        };

        let volume_spike = features.indicators.volume_spike as f64;
        let momentum = features.indicators.macd_histogram.abs() as f64;

        // Базовая логика
        let mut prob_break: f64 = 0.3;

        if volume_spike > 1.5 {
            prob_break += 0.2;
        }

        if momentum > 0.3 {
            prob_break += 0.2;
        }

        prob_break = prob_break.min(0.95);
        let prob_bounce = 1.0 - prob_break;

        let score = prob_bounce.max(prob_break);

        // ❗ Убираем жёсткий фильтр 0.80
        Ok(Some((level.clone(), prob_bounce, prob_break, score)))
    }
}