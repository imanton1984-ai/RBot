// strategies/level_ml_strategy/src/scorer.rs
// Level ML Strategy - использует ТОЛЬКО ML предсказания с уровнями

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use predictors::types::{PredictionAspect, PredictionRow, CalcSource};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SetupKind {
    Bounce,
    Breakout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelMlScoreBreakdown {
    pub final_score: f64,
    pub base_score: f64,
    pub ml_predictors_score: f64,
    pub raw_signals_score: f64,
    pub indicators_score: f64,
    pub setup_kind: SetupKind,
    pub setup_confidence: f64,
    pub bounce_prob: Option<f64>,
    pub breakout_prob: Option<f64>,
    pub trend_align: f64,
    pub debug: Value,
}

pub struct LevelMlScorer {
    min_final_score: f64,
    adx_filter_threshold: f64,
}

impl LevelMlScorer {
    pub fn new(min_final_score: f64) -> Self {
        Self {
            min_final_score,
            adx_filter_threshold: 25.0,
        }
    }

    pub async fn score_signal(
        &self,
        symbol: &str,
        tf_minutes: i16,
        timestamp: DateTime<Utc>,
        side: i16,
        raw_signals_summary: &Value,
        predictors: &[PredictionRow],
    ) -> Result<Option<f64>> {
        Ok(self
            .score_signal_verbose(symbol, tf_minutes, timestamp, side, raw_signals_summary, predictors)
            .await?
            .map(|b| b.final_score))
    }

    pub async fn score_signal_verbose(
        &self,
        symbol: &str,
        tf_minutes: i16,
        timestamp: DateTime<Utc>,
        side: i16,
        raw_signals_summary: &Value,
        predictors: &[PredictionRow],
    ) -> Result<Option<LevelMlScoreBreakdown>> {
        let side_i8 = side.signum() as i8;

        // ML Predictors score - ТОЛЬКО ML
        let ml_predictors_score = self.calculate_ml_predictors_score(predictors);

        // Raw signals score
        let raw_signals_score = self.calculate_raw_signals_score(raw_signals_summary);

        // Indicators score
        let indicators_score = self.calculate_indicator_score(raw_signals_summary, side_i8);

        // Base score: 50% ML predictors, 30% raw signals, 20% indicators
        let base_score = ml_predictors_score * 0.50 + raw_signals_score * 0.30 + indicators_score * 0.20;

        // Extract level probs from ML predictors
        let (bounce_prob, breakout_prob) = self.extract_ml_level_probs(predictors);

        // Infer setup
        let (setup_kind, setup_confidence) = self.infer_setup(bounce_prob, breakout_prob);

        // Regime alignment
        let trend_align = raw_signals_summary.get("trend_strength").and_then(|v| v.as_f64()).unwrap_or(0.5);

        // ADX filter
        let adx = raw_signals_summary.get("adx").and_then(|v| v.as_f64()).unwrap_or(0.0);
        if adx > 0.0 && adx < self.adx_filter_threshold {
            return Ok(None);
        }

        // Level boost
        let sr_present = raw_signals_summary.get("best_levels_score")
            .and_then(|v| v.as_f64())
            .map(|s| s > 0.3)
            .unwrap_or(false);

        let mut level_adjusted_score = base_score;
        if sr_present {
            let trend_short = raw_signals_summary.get("trend_short").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let trend_aligned = (trend_short > 0.0 && side_i8 > 0) || (trend_short < 0.0 && side_i8 < 0);
            if trend_aligned {
                level_adjusted_score *= 1.08;
            }
        }

        level_adjusted_score = level_adjusted_score.min(base_score * 1.3);

        let setup_damp = 0.8 + 0.2 * setup_confidence;
        let final_score = (level_adjusted_score * setup_damp).clamp(0.0, 1.0);

        let debug = json!({
            "symbol": symbol,
            "tf_minutes": tf_minutes,
            "timestamp": timestamp.to_rfc3339(),
            "side": side_i8,
            "base_score": base_score,
            "ml_predictors_score": ml_predictors_score,
            "raw_signals_score": raw_signals_score,
            "indicators_score": indicators_score,
            "final_score": final_score,
            "setup_kind": format!("{:?}", setup_kind),
            "setup_confidence": setup_confidence,
            "bounce_prob": bounce_prob,
            "breakout_prob": breakout_prob,
        });

        if final_score >= self.min_final_score {
            Ok(Some(LevelMlScoreBreakdown {
                final_score,
                base_score: level_adjusted_score,
                ml_predictors_score: ml_predictors_score,
                raw_signals_score,
                indicators_score,
                setup_kind,
                setup_confidence,
                bounce_prob,
                breakout_prob,
                trend_align,
                debug,
            }))
        } else {
            Ok(None)
        }
    }

    fn calculate_ml_predictors_score(&self, predictors: &[PredictionRow]) -> f64 {
        let ml_predictors: Vec<&PredictionRow> = predictors.iter()
            .filter(|p| p.calc_source == CalcSource::Ml)
            .collect();

        if ml_predictors.is_empty() {
            return 0.0;
        }

        let aspect_weight = |a: i16| -> f64 {
            match a {
                x if x == PredictionAspect::PriceTarget.as_int() => 1.0,
                x if x == PredictionAspect::LevelBounce.as_int() => 0.85,
                x if x == PredictionAspect::LevelBreakout.as_int() => 0.85,
                _ => 0.6,
            }
        };

        let mut total_ws = 0.0;
        let mut total_w = 0.0;

        for p in &ml_predictors {
            let aspect = p.aspect.as_int();
            let score = (p.score_norm as f64).clamp(0.0, 1.0);
            let w = aspect_weight(aspect);
            total_ws += score * w;
            total_w += w;
        }

        if total_w > 0.0 {
            (total_ws / total_w).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn calculate_raw_signals_score(&self, raw_signals_summary: &Value) -> f64 {
        let best_raw = raw_signals_summary.get("best_raw_signal_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let best_levels = raw_signals_summary.get("best_levels_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let best_momentum = raw_signals_summary.get("best_momentum_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let best_volume = raw_signals_summary.get("best_volume_score").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let m = best_raw.max(best_levels).max(best_momentum).max(best_volume);
        let avg = (best_raw + best_levels + best_momentum + best_volume) / 4.0;
        (0.75 * m + 0.25 * avg).clamp(0.0, 1.0)
    }

    fn calculate_indicator_score(&self, raw_signals_summary: &Value, side: i8) -> f64 {
        let rsi = raw_signals_summary.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0);
        let macd_hist = raw_signals_summary.get("macd_hist").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let rsi_score = if side > 0 {
            if rsi > 50.0 && rsi < 70.0 { 1.0 } else { 0.5 }
        } else {
            if rsi < 50.0 && rsi > 30.0 { 1.0 } else { 0.5 }
        };

        let macd_score = if side > 0 {
            if macd_hist > 0.0 { 1.0 } else { 0.3 }
        } else {
            if macd_hist < 0.0 { 1.0 } else { 0.3 }
        };

        (rsi_score * 0.6_f64 + macd_score * 0.4_f64).clamp(0.0_f64, 1.0_f64)
    }

    fn extract_ml_level_probs(&self, predictors: &[PredictionRow]) -> (Option<f64>, Option<f64>) {
        let bounce = predictors.iter()
            .find(|p| p.aspect == PredictionAspect::LevelBounce && p.calc_source == CalcSource::Ml)
            .map(|p| p.value);
        let breakout = predictors.iter()
            .find(|p| p.aspect == PredictionAspect::LevelBreakout && p.calc_source == CalcSource::Ml)
            .map(|p| p.value);
        (bounce, breakout)
    }

    fn infer_setup(&self, bounce_prob: Option<f64>, breakout_prob: Option<f64>) -> (SetupKind, f64) {
        match (bounce_prob, breakout_prob) {
            (Some(b), Some(br)) => {
                if b > br { (SetupKind::Bounce, b) } else { (SetupKind::Breakout, br) }
            }
            (Some(b), None) => (SetupKind::Bounce, b),
            (None, Some(br)) => (SetupKind::Breakout, br),
            (None, None) => (SetupKind::Bounce, 0.5),
        }
    }
}
