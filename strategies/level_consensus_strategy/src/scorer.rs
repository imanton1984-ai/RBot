// strategies/level_consensus_strategy/src/scorer.rs
//
// Level Consensus Strategy Scorer - копия FinalScorer из compute/scorer/final_score.rs
// Использует уровни поддержки/сопротивления + консенсус предсказаний

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
pub struct StrategyWeights {
    pub predictors_weight: f64,
    pub raw_signals_weight: f64,
    pub indicators_weight: f64,
    pub market_weight: f64,
}

impl Default for StrategyWeights {
    fn default() -> Self {
        Self {
            predictors_weight: 0.35,
            raw_signals_weight: 0.30,
            indicators_weight: 0.20,
            market_weight: 0.15,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelConsensusScoreBreakdown {
    pub final_score: f64,
    pub base_score: f64,
    pub predictors_score: f64,
    pub predictors_ml_score: f64,
    pub predictors_heur_score: f64,
    pub raw_signals_score: f64,
    pub indicators_score: f64,
    pub market_score: f64,
    pub coverage_score: f64,
    pub consensus_score: f64,
    pub setup_kind: SetupKind,
    pub setup_confidence: f64,
    pub bounce_prob: Option<f64>,
    pub breakout_prob: Option<f64>,
    pub level_distance_atr: Option<f64>,
    pub level_strength: Option<f64>,
    pub trend_align: f64,
    pub volatility_ok: f64,
    pub momentum_ok: f64,
    pub debug: Value,
}

pub struct LevelConsensusScorer {
    min_final_score: f64,
    weights: StrategyWeights,
    coverage_gamma: f64,
    consensus_gamma: f64,
    sr_align_boost: f64,
    synergy_boost: f64,
    trend_conflict_penalty: f64,
    correction_penalty_weight: f64,
    stoch_correction_long_min: f64,
    stoch_correction_short_max: f64,
    adx_filter_threshold: f64,
}

impl LevelConsensusScorer {
    pub fn new(min_final_score: f64) -> Self {
        Self {
            min_final_score,
            weights: StrategyWeights::default(),
            coverage_gamma: 0.5,
            consensus_gamma: 0.3,
            sr_align_boost: 1.08,
            synergy_boost: 1.12,
            trend_conflict_penalty: 0.75,
            correction_penalty_weight: 0.70,
            stoch_correction_long_min: 78.0,
            stoch_correction_short_max: 22.0,
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
    ) -> Result<Option<LevelConsensusScoreBreakdown>> {
        let side_i8 = side.signum() as i8;

        // Predictors score
        let pred_comp = self.calculate_predictors_component(predictors)?;
        let predictors_score = pred_comp.total;
        let predictors_ml_score = pred_comp.ml;
        let predictors_heur_score = pred_comp.heur;

        let (bounce_prob, breakout_prob) = self.extract_level_probs(predictors);

        // Raw signals score
        let raw_signals_score = self.calculate_raw_signals_score(raw_signals_summary);

        // Indicators score
        let indicators_score = self.calculate_indicator_score(raw_signals_summary, side_i8);

        // Market score (neutral для simplicity)
        let market_score = 0.5;

        // Base score
        let base_score =
            predictors_score * self.weights.predictors_weight +
            raw_signals_score * self.weights.raw_signals_weight +
            indicators_score * self.weights.indicators_weight +
            market_score * self.weights.market_weight;

        // Coverage & consensus
        let coverage_score = self.calculate_feature_coverage_score(predictors, raw_signals_summary);
        let consensus_score = self.calculate_ml_heuristic_consensus(predictors_ml_score, predictors_heur_score);

        // Infer setup
        let (setup_kind, setup_confidence) = self.infer_setup(bounce_prob, breakout_prob);

        // Level context
        let level_distance_atr = self.extract_level_distance_atr(raw_signals_summary, side_i8);
        let level_strength = self.extract_level_strength(raw_signals_summary, side_i8);

        // Regime alignment
        let trend_align = raw_signals_summary.get("trend_strength").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let volatility_regime = raw_signals_summary.get("volatility_regime").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let momentum_strength = raw_signals_summary.get("momentum_strength").and_then(|v| v.as_f64()).unwrap_or(0.5);

        let volatility_ok = match setup_kind {
            SetupKind::Bounce => (1.0 - volatility_regime).clamp(0.0, 1.0),
            SetupKind::Breakout => (0.6 + 0.4 * volatility_regime).clamp(0.0, 1.0),
        };

        let momentum_ok = match setup_kind {
            SetupKind::Bounce => (1.0 - (momentum_strength - 0.5).abs() * 2.0).clamp(0.0, 1.0),
            SetupKind::Breakout => momentum_strength,
        };

        // ADX filter
        let adx = raw_signals_summary.get("adx").and_then(|v| v.as_f64()).unwrap_or(0.0);
        if adx > 0.0 && adx < self.adx_filter_threshold {
            return Ok(None);
        }

        // Level strategy adjustments
        let mut level_adjusted_score = base_score;
        let mut cumulative_boost = 1.0_f64;

        let sr_present = raw_signals_summary.get("best_levels_score")
            .and_then(|v| v.as_f64())
            .map(|s| s > 0.3)
            .unwrap_or(false);

        if sr_present {
            let trend_short = raw_signals_summary.get("trend_short").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let trend_aligned = (trend_short > 0.0 && side_i8 > 0) || (trend_short < 0.0 && side_i8 < 0);

            if trend_aligned {
                cumulative_boost *= self.sr_align_boost;
            }

            let touches = raw_signals_summary.get("level_touches")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            if touches >= 3 {
                cumulative_boost *= 1.25;
            }
        }

        cumulative_boost = cumulative_boost.min(1.50);
        level_adjusted_score *= cumulative_boost;

        // Trend conflict
        let trend_short = raw_signals_summary.get("trend_short").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let trend_medium = raw_signals_summary.get("trend_medium").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let trend_conflict = (trend_short > 0.0 && trend_medium < 0.0) ||
                            (trend_short < 0.0 && trend_medium > 0.0);

        if trend_conflict {
            level_adjusted_score *= self.trend_conflict_penalty;
        }

        // Correction penalty
        let stoch_k = raw_signals_summary.get("stoch_k").and_then(|v| v.as_f64()).unwrap_or(50.0);
        let rsi = raw_signals_summary.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0);
        
        let is_risky_long = stoch_k >= self.stoch_correction_long_min || rsi > 70.0;
        let is_risky_short = stoch_k <= self.stoch_correction_short_max || rsi < 30.0;

        if (side_i8 > 0 && is_risky_long) || (side_i8 < 0 && is_risky_short) {
            level_adjusted_score *= self.correction_penalty_weight;
        }

        let adjusted_base_score = if (level_adjusted_score - base_score).abs() > 0.01 {
            level_adjusted_score
        } else {
            base_score
        };

        let setup_damp = 0.75 + 0.25 * setup_confidence;
        let final_score = (adjusted_base_score
            * coverage_score.powf(self.coverage_gamma)
            * consensus_score.powf(self.consensus_gamma)
            * setup_damp)
            .clamp(0.0, 1.0);

        let debug = json!({
            "symbol": symbol,
            "tf_minutes": tf_minutes,
            "timestamp": timestamp.to_rfc3339(),
            "side": side_i8,
            "weights": self.weights,
            "base_score": base_score,
            "adjusted_base_score": adjusted_base_score,
            "predictors_score": predictors_score,
            "raw_signals_score": raw_signals_score,
            "indicators_score": indicators_score,
            "market_score": market_score,
            "coverage_score": coverage_score,
            "consensus_score": consensus_score,
            "final_score": final_score,
            "setup_kind": format!("{:?}", setup_kind),
            "setup_confidence": setup_confidence,
            "bounce_prob": bounce_prob,
            "breakout_prob": breakout_prob,
        });

        if final_score >= self.min_final_score {
            Ok(Some(LevelConsensusScoreBreakdown {
                final_score,
                base_score: adjusted_base_score,
                predictors_score,
                predictors_ml_score,
                predictors_heur_score,
                raw_signals_score,
                indicators_score,
                market_score,
                coverage_score,
                consensus_score,
                setup_kind,
                setup_confidence,
                bounce_prob,
                breakout_prob,
                level_distance_atr,
                level_strength,
                trend_align,
                volatility_ok,
                momentum_ok,
                debug,
            }))
        } else {
            Ok(None)
        }
    }

    fn calculate_predictors_component(&self, predictors: &[PredictionRow]) -> Result<PredComponent> {
        if predictors.is_empty() {
            return Ok(PredComponent::zero());
        }

        let mut best_ml: std::collections::HashMap<i16, f64> = std::collections::HashMap::new();
        let mut best_heur: std::collections::HashMap<i16, f64> = std::collections::HashMap::new();

        for p in predictors {
            let aspect = p.aspect.as_int();
            let score = (p.score_norm as f64).clamp(0.0, 1.0);

            match p.calc_source {
                CalcSource::Ml => {
                    best_ml.entry(aspect).and_modify(|x| *x = x.max(score)).or_insert(score);
                }
                CalcSource::Hard => {
                    best_heur.entry(aspect).and_modify(|x| *x = x.max(score)).or_insert(score);
                }
            }
        }

        let aspect_weight = |a: i16| -> f64 {
            match a {
                x if x == PredictionAspect::PriceTarget.as_int() => 1.00,
                x if x == PredictionAspect::LevelBounce.as_int() => 0.85,
                x if x == PredictionAspect::LevelBreakout.as_int() => 0.85,
                _ => 0.60,
            }
        };

        let mut total_ws = 0.0;
        let mut total_w = 0.0;
        let mut ml_ws = 0.0;
        let mut ml_w = 0.0;
        let mut heur_ws = 0.0;
        let mut heur_w = 0.0;

        let mut aspects: std::collections::BTreeSet<i16> = std::collections::BTreeSet::new();
        for k in best_ml.keys() { aspects.insert(*k); }
        for k in best_heur.keys() { aspects.insert(*k); }

        for a in aspects {
            let w = aspect_weight(a);
            let s_ml = best_ml.get(&a).copied().unwrap_or(0.0);
            let s_heur = best_heur.get(&a).copied().unwrap_or(0.0);
            let s_total = s_ml.max(s_heur);

            total_ws += s_total * w;
            total_w += w;

            if s_ml > 0.0 { ml_ws += s_ml * w; ml_w += w; }
            if s_heur > 0.0 { heur_ws += s_heur * w; heur_w += w; }
        }

        let total = if total_w > 0.0 { total_ws / total_w } else { 0.0 };
        let ml = if ml_w > 0.0 { ml_ws / ml_w } else { 0.0 };
        let heur = if heur_w > 0.0 { heur_ws / heur_w } else { 0.0 };

        Ok(PredComponent { total, ml, heur })
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
        let stoch_k = raw_signals_summary.get("stoch_k").and_then(|v| v.as_f64()).unwrap_or(50.0);

        let rsi_score = if side > 0 {
            if rsi > 50.0 && rsi < 70.0 { 1.0 } else if rsi >= 70.0 { 0.5 } else { 0.3 }
        } else {
            if rsi < 50.0 && rsi > 30.0 { 1.0 } else if rsi <= 30.0 { 0.5 } else { 0.3 }
        };

        let macd_score = if side > 0 {
            if macd_hist > 0.0 { 1.0 } else { 0.3 }
        } else {
            if macd_hist < 0.0 { 1.0 } else { 0.3 }
        };

        let stoch_score = if side > 0 {
            if stoch_k > 50.0 && stoch_k < 80.0 { 1.0 } else if stoch_k >= 80.0 { 0.3 } else { 0.5 }
        } else {
            if stoch_k < 50.0 && stoch_k > 20.0 { 1.0 } else if stoch_k <= 20.0 { 0.3 } else { 0.5 }
        };

        (rsi_score * 0.4_f64 + macd_score * 0.35_f64 + stoch_score * 0.25_f64).clamp(0.0_f64, 1.0_f64)
    }

    fn calculate_feature_coverage_score(&self, predictors: &[PredictionRow], raw_signals_summary: &Value) -> f64 {
        let has_predictors = !predictors.is_empty();
        let has_raw = raw_signals_summary.get("best_raw_signal_score").and_then(|v| v.as_f64()).unwrap_or(0.0) > 0.0;
        
        if has_predictors && has_raw { 1.0 } else if has_predictors || has_raw { 0.7 } else { 0.5 }
    }

    fn calculate_ml_heuristic_consensus(&self, ml_score: f64, heur_score: f64) -> f64 {
        let avg = (ml_score + heur_score) / 2.0;
        let disagreement = (ml_score - heur_score).abs();
        (avg * (1.0 - disagreement * 0.2)).clamp(0.0, 1.0)
    }

    fn infer_setup(&self, bounce_prob: Option<f64>, breakout_prob: Option<f64>) -> (SetupKind, f64) {
        match (bounce_prob, breakout_prob) {
            (Some(b), Some(br)) => {
                if b > br {
                    (SetupKind::Bounce, b)
                } else {
                    (SetupKind::Breakout, br)
                }
            }
            (Some(b), None) => (SetupKind::Bounce, b),
            (None, Some(br)) => (SetupKind::Breakout, br),
            (None, None) => (SetupKind::Bounce, 0.5),
        }
    }

    fn extract_level_probs(&self, predictors: &[PredictionRow]) -> (Option<f64>, Option<f64>) {
        let bounce = predictors.iter()
            .find(|p| p.aspect == PredictionAspect::LevelBounce)
            .map(|p| p.value);
        let breakout = predictors.iter()
            .find(|p| p.aspect == PredictionAspect::LevelBreakout)
            .map(|p| p.value);
        (bounce, breakout)
    }

    fn extract_level_distance_atr(&self, raw_signals_summary: &Value, _side: i8) -> Option<f64> {
        raw_signals_summary.get("level_distance_atr").and_then(|v| v.as_f64())
    }

    fn extract_level_strength(&self, raw_signals_summary: &Value, _side: i8) -> Option<f64> {
        raw_signals_summary.get("level_strength").and_then(|v| v.as_f64())
    }
}

struct PredComponent {
    total: f64,
    ml: f64,
    heur: f64,
}

impl PredComponent {
    fn zero() -> Self {
        Self { total: 0.0, ml: 0.0, heur: 0.0 }
    }
}
