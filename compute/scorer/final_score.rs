// compute/scoring/final_scorer.rs

// compute/scoring/final_score.rs

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::predictors::types::{PredictionAspect, PredictionRow, CalcSource};

use super::market_params_calculator::MarketParams;

/// Final scorer configuration (weights + strictness)
pub struct FinalScorer {
    min_final_score: f64,
    weights: StrategyWeights,
    /// How strict to penalize missing feature coverage (>= 1.0)
    coverage_gamma: f64,
    /// How strict to penalize ML vs heuristic disagreement (>= 1.0)
    consensus_gamma: f64,
}

/// Weights for base score components (must sum to 1.0 after normalization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyWeights {
    pub predictors_weight: f64,
    pub raw_signals_weight: f64,
    pub indicators_weight: f64,
    pub market_weight: f64,
}

impl Default for StrategyWeights {
    fn default() -> Self {
        // Keep market meaningful; without it you will trade against BTC regime.
        Self {
            predictors_weight: 0.35,
            raw_signals_weight: 0.30,
            indicators_weight: 0.20,
            market_weight: 0.15,
        }
    }
}

impl StrategyWeights {
    pub fn normalized(mut self) -> Self {
        let sum = self.predictors_weight + self.raw_signals_weight + self.indicators_weight + self.market_weight;
        if sum > 0.0 {
            self.predictors_weight /= sum;
            self.raw_signals_weight /= sum;
            self.indicators_weight /= sum;
            self.market_weight /= sum;
        }
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalScoreBreakdown {
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

    pub debug: Value,
}

impl FinalScorer {
    pub fn new(min_final_score: f64) -> Self {
        Self {
            min_final_score,
            weights: StrategyWeights::default().normalized(),
            coverage_gamma: 1.8,
            consensus_gamma: 1.5,
        }
    }

    pub fn new_with_weights(min_final_score: f64, weights: StrategyWeights) -> Self {
        Self {
            min_final_score,
            weights: weights.normalized(),
            coverage_gamma: 1.8,
            consensus_gamma: 1.5,
        }
    }

    pub fn with_strictness(mut self, coverage_gamma: f64, consensus_gamma: f64) -> Self {
        self.coverage_gamma = coverage_gamma.max(1.0);
        self.consensus_gamma = consensus_gamma.max(1.0);
        self
    }

    /// Returns only final number (compat)
    pub async fn score_signal(
        &self,
        symbol: &str,
        tf_minutes: i16,
        timestamp: DateTime<Utc>,
        side: i16,
        raw_signals_summary: &Value,
        predictors: &[PredictionRow],
        market_params: Option<&MarketParams>,
    ) -> Result<Option<f64>> {
        Ok(self
            .score_signal_verbose(symbol, tf_minutes, timestamp, side, raw_signals_summary, predictors, market_params)
            .await?
            .map(|b| b.final_score))
    }

    /// Verbose score with breakdown for persistence/debug
    pub async fn score_signal_verbose(
        &self,
        symbol: &str,
        tf_minutes: i16,
        timestamp: DateTime<Utc>,
        side: i16,
        raw_signals_summary: &Value,
        predictors: &[PredictionRow],
        market_params: Option<&MarketParams>,
    ) -> Result<Option<FinalScoreBreakdown>> {
        let side_i8 = side.signum() as i8;

        // predictors score
        let pred_comp = self.calculate_predictors_component(predictors)?;
        let predictors_score = pred_comp.total;
        let predictors_ml_score = pred_comp.ml;
        let predictors_heur_score = pred_comp.heur;

        // Raw signals score (use BEST signals, but penalize missing coverage)
        let raw_signals_score = self.calculate_raw_signals_score(raw_signals_summary);

        // Indicators score from summary fields (already normalized 0..1 expected)
        let indicators_score = self.calculate_indicator_score(raw_signals_summary);

        // Market score (BTC regime alignment)
        let market_score = market_params
            .map(|m| m.score_for_side(side_i8))
            .unwrap_or(0.0);

        // Base score (weights sum to 1)
        let base_score =
            predictors_score * self.weights.predictors_weight +
            raw_signals_score * self.weights.raw_signals_weight +
            indicators_score * self.weights.indicators_weight +
            market_score * self.weights.market_weight;

        // Coverage & consensus
        let has_market = market_params.is_some();
        let coverage_score = self.calculate_feature_coverage_score(predictors, raw_signals_summary, has_market);
        let consensus_score = self.calculate_ml_heuristic_consensus(predictors_ml_score, predictors_heur_score);

        // “AND-like” penalties:
        let final_score = (base_score
            * coverage_score.powf(self.coverage_gamma)
            * consensus_score.powf(self.consensus_gamma))
            .clamp(0.0, 1.0);

        let debug = json!({
            "symbol": symbol,
            "tf_minutes": tf_minutes,
            "timestamp": timestamp.to_rfc3339(),
            "side": side_i8,
            "weights": self.weights,
            "coverage_gamma": self.coverage_gamma,
            "consensus_gamma": self.consensus_gamma,
            "has_market_params": has_market,
            "market_params": market_params.map(|m| m.details_json.clone()),
        });

        if final_score >= self.min_final_score {
            Ok(Some(FinalScoreBreakdown {
                final_score,
                base_score,
                predictors_score,
                predictors_ml_score,
                predictors_heur_score,
                raw_signals_score,
                indicators_score,
                market_score,
                coverage_score,
                consensus_score,
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

        // Per aspect best-by-score for ML and HEUR separately
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

        // Aspect weights: target price is king, bounce/breakout slightly less.
        let aspect_weight = |a: i16| -> f64 {
            match a {
                x if x == PredictionAspect::PriceTarget.as_int() => 1.00,
                x if x == PredictionAspect::LevelBounce.as_int() => 0.85,
                x if x == PredictionAspect::LevelBreakout.as_int() => 0.85,
                _ => 0.60,
            }
        };

        // Merge ML+HEUR into total by taking BEST-of-two per aspect, but keep their averages for consensus later
        let mut total_ws = 0.0;
        let mut total_w = 0.0;

        let mut ml_ws = 0.0;
        let mut ml_w = 0.0;

        let mut heur_ws = 0.0;
        let mut heur_w = 0.0;

        // union of aspects
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

            if s_ml > 0.0 {
                ml_ws += s_ml * w;
                ml_w += w;
            }
            if s_heur > 0.0 {
                heur_ws += s_heur * w;
                heur_w += w;
            }
        }

        let total = if total_w > 0.0 { total_ws / total_w } else { 0.0 };
        let ml = if ml_w > 0.0 { ml_ws / ml_w } else { 0.0 };
        let heur = if heur_w > 0.0 { heur_ws / heur_w } else { 0.0 };

        Ok(PredComponent { total, ml, heur })
    }

    fn calculate_raw_signals_score(&self, raw_signals_summary: &Value) -> f64 {
        // Use the strongest signals, not averages (you explicitly want "maximum of possible high-score signals")
        let best_raw = raw_signals_summary.get("best_raw_signal_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let best_levels = raw_signals_summary.get("best_levels_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let best_momentum = raw_signals_summary.get("best_momentum_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let best_volume = raw_signals_summary.get("best_volume_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Keep it simple: max pool with slight smoothing to avoid single-noise spikes
        let m = best_raw.max(best_levels).max(best_momentum).max(best_volume);

        // Smooth: if only one is high and others are 0, we don't want 1.0
        let avg = (best_raw + best_levels + best_momentum + best_volume) / 4.0;
        (0.75 * m + 0.25 * avg).clamp(0.0, 1.0)
    }

    fn calculate_indicator_score(&self, raw_signals_summary: &Value) -> f64 {
        // Expected already normalized 0..1 in summary.
        // If you don't have those fields yet, start writing them into raw_signals_summary in your aggregator.
        let trend_strength = raw_signals_summary.get("trend_strength").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let momentum_strength = raw_signals_summary.get("momentum_strength").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let volatility_regime = raw_signals_summary.get("volatility_regime").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let volume_spike = raw_signals_summary.get("volume_spike_score").and_then(|v| v.as_f64()).unwrap_or(0.0);

        // You can tune weights; keep balanced.
        (0.30 * trend_strength + 0.30 * momentum_strength + 0.20 * (1.0 - volatility_regime) + 0.20 * volume_spike)
            .clamp(0.0, 1.0)
    }

    fn calculate_feature_coverage_score(&self, predictors: &[PredictionRow], raw_signals_summary: &Value, has_market: bool) -> f64 {
        // Prediction coverage: do we have main aspects?
        let mut has_price_target = false;
        let mut has_bounce = false;
        let mut has_break = false;

        for p in predictors {
            match p.aspect {
                PredictionAspect::PriceTarget => has_price_target = true,
                PredictionAspect::LevelBounce => has_bounce = true,
                PredictionAspect::LevelBreakout => has_break = true,
            }
        }

        let pred_cov = (has_price_target as i32 + has_bounce as i32 + has_break as i32) as f64 / 3.0;

        // Raw signals coverage: how many expected buckets are present?
        // You should store these counts in summary (your aggregator must do it).
        let raw_cov = raw_signals_summary.get("feature_coverage")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.6); // fallback: assume partial

        let market_cov = if has_market { 1.0 } else { 0.0 };

        // Weighted coverage: predictors + raw + market
        (0.45 * pred_cov + 0.45 * raw_cov + 0.10 * market_cov).clamp(0.0, 1.0)
    }

    fn calculate_ml_heuristic_consensus(&self, ml_score: f64, heur_score: f64) -> f64 {
        // If one source missing -> treat as weaker consensus
        if ml_score <= 0.0 || heur_score <= 0.0 {
            return 0.65;
        }
        let diff = (ml_score - heur_score).abs();
        // 0 diff => 1.0, 0.4 diff => ~0.6
        (1.0 - (diff / 0.4)).clamp(0.0, 1.0) * 0.4 + 0.6
    }
}

#[derive(Debug, Clone)]
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
