// compute/scoring/final_scorer.rs

// compute/scoring/final_score.rs

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::predictors::types::{PredictionAspect, PredictionRow, CalcSource};

use super::market_params_calculator::MarketParams;

/// Setup kind: distinguishes bounce vs breakout trades
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SetupKind {
    Bounce,
    Breakout,
}

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

    // Setup inference fields
    pub setup_kind: SetupKind,
    pub setup_confidence: f64,
    pub bounce_prob: Option<f64>,
    pub breakout_prob: Option<f64>,

    // Level context
    pub level_distance_atr: Option<f64>,
    pub level_strength: Option<f64>,

    // Regime alignment
    pub trend_align: f64,
    pub volatility_ok: f64,
    pub momentum_ok: f64,

    pub debug: Value,
}

impl FinalScorer {
    pub fn new(min_final_score: f64) -> Self {
        Self {
            min_final_score,
            weights: StrategyWeights::default().normalized(),
            // Reduced from 1.8/1.5: the previous values made it mathematically impossible
            // to reach even 0.50 final score with any combination of inputs.
            // 0.5/0.3 applies meaningful but achievable penalties.
            coverage_gamma: 0.5,
            consensus_gamma: 0.3,
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

        // Extract bounce/breakout probs from predictors
        let (bounce_prob, breakout_prob) = self.extract_level_probs(predictors);

        // Raw signals score (use BEST signals, but penalize missing coverage)
        let raw_signals_score = self.calculate_raw_signals_score(raw_signals_summary);

        // Indicators score WITH direction alignment check
        let indicators_score = self.calculate_indicator_score_directional(raw_signals_summary, side_i8);

        // Market score (BTC regime alignment)
        // If market params not available, use neutral 0.5 instead of 0.0
        let market_score = market_params
            .map(|m| m.score_for_side(side_i8))
            .unwrap_or(0.5);

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

        // Infer setup kind (bounce vs breakout)
        let (setup_kind, setup_confidence) = self.infer_setup(bounce_prob, breakout_prob);

        // Extract level context
        let level_distance_atr = self.extract_level_distance_atr(raw_signals_summary, side_i8);
        let level_strength = self.extract_level_strength(raw_signals_summary, side_i8);

        // Regime alignment factors
        let trend_align = raw_signals_summary.get("trend_strength").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let volatility_regime = raw_signals_summary.get("volatility_regime").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let momentum_strength = raw_signals_summary.get("momentum_strength").and_then(|v| v.as_f64()).unwrap_or(0.5);

        // volatility_ok: bounce prefers calm, breakout can handle volatility
        let volatility_ok = match setup_kind {
            SetupKind::Bounce => (1.0 - volatility_regime).clamp(0.0, 1.0),
            SetupKind::Breakout => (0.6 + 0.4 * volatility_regime).clamp(0.0, 1.0),
        };

        // momentum_ok: bounce prefers moderate momentum, breakout prefers high
        let momentum_ok = match setup_kind {
            SetupKind::Bounce => (1.0 - (momentum_strength - 0.5).abs() * 2.0).clamp(0.0, 1.0),
            SetupKind::Breakout => momentum_strength,
        };

        // Soft penalties: coverage and consensus reduce score, but not catastrophically.
        // Previous formula (base * cov^1.8 * cons^1.5) made ≥0.50 impossible.
        // Apply setup confidence dampening
        let setup_damp = 0.75 + 0.25 * setup_confidence;
        let final_score = (base_score
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
            "coverage_gamma": self.coverage_gamma,
            "consensus_gamma": self.consensus_gamma,
            "has_market_params": has_market,
            "base_score": base_score,
            "predictors_score": predictors_score,
            "raw_signals_score": raw_signals_score,
            "indicators_score": indicators_score,
            "market_score": market_score,
            "coverage_score": coverage_score,
            "consensus_score": consensus_score,
            "final_score": final_score,
            "market_params": market_params.map(|m| m.details_json.clone()),
            "setup_kind": format!("{:?}", setup_kind),
            "setup_confidence": setup_confidence,
            "bounce_prob": bounce_prob,
            "breakout_prob": breakout_prob,
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
            // Diagnostic: sample-log rejected signals so we can tune thresholds
            static DIAG_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let cnt = DIAG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if cnt % 5000 == 0 {
                tracing::info!(
                    target: "final_scorer",
                    "REJECTED signal #{} {} tf={} side={}: final={:.4} (base={:.4} cov={:.4} cons={:.4}) \
                     pred={:.3} raw={:.3} ind={:.3} mkt={:.3} | threshold={:.2}",
                    cnt, symbol, tf_minutes, side_i8,
                    final_score, base_score, coverage_score, consensus_score,
                    predictors_score, raw_signals_score, indicators_score, market_score,
                    self.min_final_score
                );
            }
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

    /// Conservative multi-indicator directional score.
    ///
    /// Philosophy: **Generous direction alignment base** (like the original 57% WR version)
    /// **+ confluence bonuses** that push high-quality signals higher
    /// **+ penalties ONLY at TRUE extremes** (RSI>85, Stoch>90, etc.)
    ///
    /// Crypto momentum signals work. Don't fight the trend. Just reward confluence.
    fn calculate_indicator_score_directional(&self, raw: &Value, side: i8) -> f64 {
        let trend_strength = raw.get("trend_strength").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let momentum_strength = raw.get("momentum_strength").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let volatility_regime = raw.get("volatility_regime").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let volume_spike = raw.get("volume_spike_score").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let trend_short = raw.get("trend_short").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let trend_medium = raw.get("trend_medium").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let trend_long = raw.get("trend_long").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let rsi = raw.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0);
        let macd_hist = raw.get("macd_hist").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let ema_20 = raw.get("ema_20").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let ema_50 = raw.get("ema_50").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let ema_200 = raw.get("ema_200").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let close = raw.get("close").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let stoch_k = raw.get("stoch_k").and_then(|v| v.as_f64()).unwrap_or(50.0);
        let _stoch_d = raw.get("stoch_d").and_then(|v| v.as_f64()).unwrap_or(50.0);
        let williams = raw.get("williams_r").and_then(|v| v.as_f64()).unwrap_or(-50.0);
        let cci = raw.get("cci").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let bb_upper = raw.get("bb_upper").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let bb_lower = raw.get("bb_lower").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let vwap = raw.get("vwap").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let adx = raw.get("adx").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let atr = raw.get("atr").and_then(|v| v.as_f64()).unwrap_or(0.0);

        // ====================================================================
        //  PART A: Direction Alignment (generous, like original — pass/fail)
        //  This produced ~57% WR before. Keep it as the solid base.
        // ====================================================================
        let mut align_score: f64 = 0.0;
        let mut align_count: f64 = 0.0;

        // 1. Short-term trend
        if trend_short != 0.0 {
            let aligned = (trend_short > 0.0 && side > 0) || (trend_short < 0.0 && side < 0);
            align_score += if aligned { 1.0 } else { 0.0 };
            align_count += 1.0;
        }

        // 2. Medium-term trend
        if trend_medium != 0.0 {
            let aligned = (trend_medium > 0.0 && side > 0) || (trend_medium < 0.0 && side < 0);
            align_score += if aligned { 1.0 } else { 0.0 };
            align_count += 1.0;
        }

        // 3. Trend short vs medium confluence
        if trend_short != 0.0 && trend_medium != 0.0 {
            let same = trend_short.signum() == trend_medium.signum();
            align_score += if same { 0.5 } else { 0.0 };
            align_count += 0.5;
        }

        // 4. MACD direction
        if macd_hist.abs() > 0.0001 {
            let aligned = (macd_hist > 0.0 && side > 0) || (macd_hist < 0.0 && side < 0);
            align_score += if aligned { 1.0 } else { 0.0 };
            align_count += 1.0;
        }

        // 5. RSI: generous — only penalize TRUE extreme (>80/<20)
        if rsi > 0.0 {
            let rsi_ok = if side > 0 { rsi < 80.0 } else { rsi > 20.0 };
            align_score += if rsi_ok { 0.7 } else { 0.0 };
            align_count += 0.7;
        }

        // 6. EMA20: close on right side (generous, binary)
        if close > 0.0 && ema_20 > 0.0 {
            let aligned = (close > ema_20 && side > 0) || (close < ema_20 && side < 0);
            align_score += if aligned { 0.8 } else { 0.0 };
            align_count += 0.8;
        }

        // 7. EMA50 alignment
        if close > 0.0 && ema_50 > 0.0 {
            let aligned = (close > ema_50 && side > 0) || (close < ema_50 && side < 0);
            align_score += if aligned { 0.6 } else { 0.0 };
            align_count += 0.6;
        }

        // 8. EMA200 alignment
        if close > 0.0 && ema_200 > 0.0 {
            let aligned = (close > ema_200 && side > 0) || (close < ema_200 && side < 0);
            align_score += if aligned { 0.4 } else { 0.0 };
            align_count += 0.4;
        }

        let direction_alignment = if align_count > 0.0 {
            (align_score / align_count).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // Base quality (same formula as original)
        let base_ind = (0.30 * trend_strength + 0.30 * momentum_strength
            + 0.20 * (1.0 - volatility_regime) + 0.20 * volume_spike)
            .clamp(0.0, 1.0);

        // Main score: 40% base quality + 60% direction alignment (original formula)
        let mut score = 0.40 * base_ind + 0.60 * direction_alignment;

        // ====================================================================
        //  PART B: LEVEL-FIRST BONUSES
        //  Biggest bonus = near favorable SR level with oscillators in recovery
        //  zone. This identifies BEGINNING of moves, not peaks.
        // ====================================================================
        let mut bonus: f64 = 0.0;

        // B1. SR LEVEL PROXIMITY — THE PRIMARY BONUS (up to +0.10)
        // This is the CORE of the level strategy: enter near support (longs) or resistance (shorts)
        if close > 0.0 && atr > 0.0 {
            let sr = raw.get("sr_levels").unwrap_or(raw);
            bonus += self.score_sr_level_bonus(sr, side, close, atr);
        }

        // B2. OSCILLATORS IN RECOVERY ZONE (not exhausted!) — up to +0.08
        // For LONG: best when oscillators are leaving oversold / in early momentum zone
        // For SHORT: best when oscillators are leaving overbought
        {
            // RSI recovery zone bonus
            let rsi_recovery = if side > 0 {
                rsi >= 30.0 && rsi <= 50.0  // Leaving oversold, early bullish
            } else {
                rsi >= 50.0 && rsi <= 70.0  // Leaving overbought, early bearish
            };
            if rsi_recovery { bonus += 0.03; }

            // Stochastic recovery zone bonus
            let stoch_recovery = if side > 0 {
                stoch_k >= 15.0 && stoch_k <= 45.0  // Leaving oversold zone
            } else {
                stoch_k >= 55.0 && stoch_k <= 85.0  // Leaving overbought zone
            };
            if stoch_recovery { bonus += 0.03; }

            // Williams recovery zone bonus
            let will_recovery = if side > 0 {
                williams <= -50.0 && williams >= -85.0  // Leaving oversold
            } else {
                williams >= -50.0 && williams <= -15.0  // Leaving overbought
            };
            if will_recovery { bonus += 0.02; }
        }

        // B3. Higher-TF trend confirmation via trend_long (+0.03)
        if trend_long != 0.0 {
            let aligned = (trend_long > 0.0 && side > 0) || (trend_long < 0.0 && side < 0);
            if aligned { bonus += 0.03; }
        }

        // B4. ADX shows trending market (momentum exists to ride) (+0.02)
        if adx > 25.0 { bonus += 0.02; }

        // B5. Bollinger Band favorable zone (+0.02)
        if bb_upper > bb_lower && bb_lower > 0.0 {
            let bb_pos = (close - bb_lower) / (bb_upper - bb_lower);
            // For LONG: best in lower half of bands; for SHORT: upper half
            let bb_favorable = if side > 0 { bb_pos < 0.5 } else { bb_pos > 0.5 };
            if bb_favorable { bonus += 0.02; }
        }

        // B6. VWAP confirms direction (+0.02)
        if vwap > 0.0 && close > 0.0 {
            let vwap_ok = if side > 0 { close >= vwap * 0.997 } else { close <= vwap * 1.003 };
            if vwap_ok { bonus += 0.02; }
        }

        score += bonus;

        // ====================================================================
        //  PART C: EXHAUSTION PENALTIES — HEAVY on overheated indicators
        //  This is the KEY fix: when ALL indicators are screaming one direction,
        //  the signal score should DECREASE not increase. Graduated penalties.
        // ====================================================================
        let mut penalty: f64 = 0.0;

        // C1. RSI EXHAUSTION — progressive penalty (the main trap indicator)
        if side > 0 {
            if rsi > 75.0 { penalty += 0.06; }      // Getting hot
            if rsi > 80.0 { penalty += 0.06; }      // Overbought (cumulative: 0.12)
            if rsi > 85.0 { penalty += 0.06; }      // Extreme (cumulative: 0.18)
        } else {
            if rsi < 25.0 { penalty += 0.06; }
            if rsi < 20.0 { penalty += 0.06; }
            if rsi < 15.0 { penalty += 0.06; }
        }

        // C2. STOCHASTIC EXHAUSTION — progressive
        if side > 0 {
            if stoch_k > 75.0 { penalty += 0.04; }
            if stoch_k > 85.0 { penalty += 0.04; }  // Cumulative: 0.08
            if stoch_k > 92.0 { penalty += 0.04; }  // Cumulative: 0.12
        } else {
            if stoch_k < 25.0 { penalty += 0.04; }
            if stoch_k < 15.0 { penalty += 0.04; }
            if stoch_k < 8.0 { penalty += 0.04; }
        }

        // C3. WILLIAMS EXHAUSTION — progressive
        if side > 0 {
            if williams > -20.0 { penalty += 0.03; }
            if williams > -10.0 { penalty += 0.03; }  // Cumulative: 0.06
        } else {
            if williams < -80.0 { penalty += 0.03; }
            if williams < -90.0 { penalty += 0.03; }
        }

        // C4. CCI EXHAUSTION
        if side > 0 && cci > 150.0 { penalty += 0.04; }
        if side > 0 && cci > 250.0 { penalty += 0.04; }  // Cumulative: 0.08
        if side < 0 && cci < -150.0 { penalty += 0.04; }
        if side < 0 && cci < -250.0 { penalty += 0.04; }

        // C5. PRICE OUTSIDE BOLLINGER BANDS
        if bb_upper > 0.0 && bb_lower > 0.0 {
            if side > 0 && close > bb_upper { penalty += 0.05; }  // Above band = overbought
            if side < 0 && close < bb_lower { penalty += 0.05; }  // Below band = oversold
        }

        // C6. OVEREXTENSION from EMA20 — price too far in trade direction
        if close > 0.0 && ema_20 > 0.0 {
            let ema_dist_pct = ((close - ema_20) / ema_20).abs();
            if side > 0 && close > ema_20 && ema_dist_pct > 0.025 { penalty += 0.04; }
            if side > 0 && close > ema_20 && ema_dist_pct > 0.04 { penalty += 0.04; }  // Cumulative: 0.08
            if side < 0 && close < ema_20 && ema_dist_pct > 0.025 { penalty += 0.04; }
            if side < 0 && close < ema_20 && ema_dist_pct > 0.04 { penalty += 0.04; }
        }

        // C7. MULTI-INDICATOR EXHAUSTION CONFLUENCE — the deadliest trap
        // When RSI AND Stoch AND Williams are ALL exhausted → maximum penalty
        {
            let rsi_hot = if side > 0 { rsi > 70.0 } else { rsi < 30.0 };
            let stoch_hot = if side > 0 { stoch_k > 70.0 } else { stoch_k < 30.0 };
            let will_hot = if side > 0 { williams > -25.0 } else { williams < -75.0 };
            let cci_hot = if side > 0 { cci > 100.0 } else { cci < -100.0 };

            let hot_count = rsi_hot as u8 + stoch_hot as u8 + will_hot as u8 + cci_hot as u8;
            if hot_count >= 3 { penalty += 0.08; }  // 3+ oscillators exhausted = massive penalty
            if hot_count >= 4 { penalty += 0.06; }  // All 4 = cumulative 0.14 extra
        }

        score -= penalty;

        score.clamp(0.0, 1.0)
    }

    /// Score SR level proximity bonus (up to +0.10).
    /// Biggest bonus when price is near a STRONG favorable level.
    fn score_sr_level_bonus(&self, sr: &Value, side: i8, close: f64, atr: f64) -> f64 {
        // Favorable levels: support for longs, resistance for shorts
        let fav_keys_strengths: [(& str, f64); 3] = if side > 0 {
            [("strong_support", 1.0), ("mid_support", 0.6), ("light_support", 0.3)]
        } else {
            [("strong_resistance", 1.0), ("mid_resistance", 0.6), ("light_resistance", 0.3)]
        };

        let mut best_bonus: f64 = 0.0;

        for (key, strength) in fav_keys_strengths {
            if let Some(level_price) = sr.get(key).and_then(|v| v.as_f64()) {
                if level_price > 0.0 {
                    let dist_atr = (close - level_price).abs() / atr;
                    // Graduated proximity bonus
                    let proximity_bonus = if dist_atr < 0.5 {
                        0.10 * strength  // Right at level
                    } else if dist_atr < 1.0 {
                        0.07 * strength  // Very close
                    } else if dist_atr < 2.0 {
                        0.04 * strength  // Within range
                    } else if dist_atr < 3.0 {
                        0.02 * strength  // Moderate distance
                    } else {
                        0.0
                    };
                    if proximity_bonus > best_bonus { best_bonus = proximity_bonus; }
                }
            }
        }
        best_bonus
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
        // Both sources present: evaluate agreement
        if ml_score > 0.0 && heur_score > 0.0 {
            let diff = (ml_score - heur_score).abs();
            // 0 diff => 1.0, 0.4 diff => ~0.6
            return (1.0 - (diff / 0.4)).clamp(0.0, 1.0) * 0.4 + 0.6;
        }

        // Single source present: acceptable, slight penalty
        // Previous value 0.65 was too harsh — with consensus_gamma=1.5 it caused a 48% multiplicative penalty,
        // making it impossible for signals to pass any reasonable threshold.
        if ml_score > 0.0 || heur_score > 0.0 {
            return 0.92;
        }

        // Neither source: big problem, penalize heavily
        0.50
    }

    /// Extract bounce and breakout probabilities from predictors
    fn extract_level_probs(&self, predictors: &[PredictionRow]) -> (Option<f64>, Option<f64>) {
        let mut bounce_prob: Option<f64> = None;
        let mut breakout_prob: Option<f64> = None;

        for p in predictors {
            match p.aspect {
                PredictionAspect::LevelBounce => {
                    bounce_prob = Some(p.value as f64);
                }
                PredictionAspect::LevelBreakout => {
                    breakout_prob = Some(p.value as f64);
                }
                _ => {}
            }
        }

        (bounce_prob, breakout_prob)
    }

    /// Infer setup kind from bounce/breakout probabilities
    fn infer_setup(&self, bounce_prob: Option<f64>, breakout_prob: Option<f64>) -> (SetupKind, f64) {
        let b = bounce_prob.unwrap_or(0.5);
        let k = breakout_prob.unwrap_or(0.5);

        let diff = (b - k).abs();
        let margin = 0.10; // Minimum difference to be confident
        let conf = ((diff - margin) / (1.0 - margin)).clamp(0.0, 1.0);

        if b >= k + margin {
            (SetupKind::Bounce, conf)
        } else if k >= b + margin {
            (SetupKind::Breakout, conf)
        } else {
            // Ambiguous: default to Bounce (safer for level trading), but low confidence
            (SetupKind::Bounce, conf * 0.5)
        }
    }

    /// Extract level distance in ATR units from raw_signals_summary
    fn extract_level_distance_atr(&self, raw: &Value, side: i8) -> Option<f64> {
        let close = raw.get("close").and_then(|v| v.as_f64())?;
        let atr = raw.get("atr").and_then(|v| v.as_f64())?;
        if atr <= 0.0 { return None; }

        let sr = raw.get("sr_levels").unwrap_or(raw);

        // For LONG: distance to nearest support below
        // For SHORT: distance to nearest resistance above
        let level_price = if side > 0 {
            ["strong_support", "mid_support", "light_support"]
                .iter()
                .filter_map(|&k| sr.get(k).and_then(|v| v.as_f64()))
                .filter(|&p| p < close)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
        } else {
            ["strong_resistance", "mid_resistance", "light_resistance"]
                .iter()
                .filter_map(|&k| sr.get(k).and_then(|v| v.as_f64()))
                .filter(|&p| p > close)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
        }?;

        let dist_atr = (close - level_price).abs() / atr;
        // Normalize to 0..1 (0 = at level, 1 = far: 3+ ATR)
        Some((dist_atr / 3.0).clamp(0.0, 1.0))
    }

    /// Extract level strength from raw_signals_summary
    fn extract_level_strength(&self, raw: &Value, side: i8) -> Option<f64> {
        let sr = raw.get("sr_levels").unwrap_or(raw);

        // Check if we have a strong level in trade direction
        let has_strong = if side > 0 {
            sr.get("strong_support").and_then(|v| v.as_f64()).is_some()
        } else {
            sr.get("strong_resistance").and_then(|v| v.as_f64()).is_some()
        };

        let has_mid = if side > 0 {
            sr.get("mid_support").and_then(|v| v.as_f64()).is_some()
        } else {
            sr.get("mid_resistance").and_then(|v| v.as_f64()).is_some()
        };

        if has_strong {
            Some(1.0)
        } else if has_mid {
            Some(0.6)
        } else {
            Some(0.3)
        }
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
