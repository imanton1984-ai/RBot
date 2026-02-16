// signal_quality/types.rs
//
// Shared types for the Signal Quality scoring system.

use serde::{Deserialize, Serialize};

/// Input features extracted from a trade.final_signals row for quality scoring.
/// These are the features that both ML and heuristic scorers consume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalFeatures {
    // --- Core signal attributes ---
    pub symbol: String,
    pub tf_minutes: i16,
    pub side: i16,          // +1 long, -1 short
    pub final_score: f64,

    // --- Component scores from FinalScorer ---
    pub base_score: f64,
    pub predictors_score: f64,
    pub raw_signals_score: f64,
    pub indicators_score: f64,
    pub market_score: f64,
    pub coverage_score: f64,
    pub consensus_score: f64,

    // --- ML vs Heuristic agreement ---
    pub ml_score: Option<f64>,
    pub heur_score: Option<f64>,

    // --- Prediction quality ---
    pub price10_target: Option<f64>,
    pub price10_score: Option<f64>,
    pub bounce_prob: Option<f64>,
    pub bounce_score: Option<f64>,
    pub breakout_prob: Option<f64>,
    pub breakout_score: Option<f64>,

    // --- Market context ---
    pub atr: Option<f64>,
    pub atr_pct: Option<f64>,
    pub trend_strength: Option<f64>,
    pub momentum_strength: Option<f64>,
    pub volatility_regime: Option<f64>,
    pub volume_spike_score: Option<f64>,

    // --- Level awareness ---
    pub level_aware: bool,
    pub n_support_levels: usize,
    pub n_resistance_levels: usize,

    // --- Market params ---
    pub market_situation: Option<String>,  // "uptrend", "downtrend", "flat", etc.
    pub market_quality_score: Option<f64>,

    // --- Entry/risk structure ---
    pub entry_price: f64,
    pub stop_loss: f64,
    pub tp1: f64,
    pub tp2: f64,
    pub tp3: f64,

    // Derived from TP/SL
    pub risk_reward_ratio: f64,    // tp1 distance / sl distance
    pub sl_pct: f64,               // |SL - entry| / entry
    pub tp1_pct: f64,              // |TP1 - entry| / entry
}

/// Output of the quality scorer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityResult {
    /// Multiplicative boost for final_score: range [0.5, 1.5]
    /// > 1.0 = signal is higher quality than average
    /// < 1.0 = signal is lower quality than average
    /// = 1.0 = neutral (no adjustment)
    pub quality_multiplier: f64,

    /// Quality grade for filtering: A (best), B, C, D (worst)
    pub grade: SignalGrade,

    /// Breakdown for debugging/analysis
    pub breakdown: QualityBreakdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalGrade {
    A,  // top ~10% quality, boost 1.2-1.5
    B,  // good, boost 1.0-1.2
    C,  // average, neutral 0.85-1.0
    D,  // poor quality, penalty 0.5-0.85
}

impl SignalGrade {
    pub fn as_str(&self) -> &'static str {
        match self {
            SignalGrade::A => "A",
            SignalGrade::B => "B",
            SignalGrade::C => "C",
            SignalGrade::D => "D",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityBreakdown {
    /// Score from heuristic rules [0, 1]
    pub heuristic_quality: f64,
    /// Score from ML model [0, 1] (None if model not loaded)
    pub ml_quality: Option<f64>,
    /// Combined quality [0, 1]
    pub combined_quality: f64,
    /// Individual factor scores
    pub factors: QualityFactors,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityFactors {
    pub score_strength: f64,        // How strong is the final_score itself
    pub component_agreement: f64,   // Do all components agree (all high vs mixed)
    pub risk_reward: f64,           // Is TP/SL ratio attractive
    pub level_confluence: f64,      // Are S/R levels supporting the trade
    pub market_alignment: f64,      // Is market regime supporting the direction
    pub predictor_confidence: f64,  // How confident are the predictors
    pub source_agreement: f64,      // Do ML and heuristic agree
}

/// Training example for the ML quality model.
/// Created by backtester analyzing historical signals against actual outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub features: SignalFeatures,
    /// Actual outcome from backtest: did the trade reach TP1/TP2/TP3 or hit SL?
    pub outcome: TradeOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeOutcome {
    /// 1.0 = hit TP1, 2.0 = hit TP2, 3.0 = hit TP3, -1.0 = hit SL, 0.0 = expired
    pub result: f64,
    /// Actual PnL percentage (positive = profit, negative = loss)
    pub pnl_pct: f64,
    /// Max favorable excursion (%) before close
    pub max_favorable: f64,
    /// Max adverse excursion (%) before close
    pub max_adverse: f64,
    /// Time to outcome in minutes
    pub duration_minutes: i64,
}

impl SignalFeatures {
    /// Extract features from a reason JSON (breakdown from FinalScorer)
    pub fn from_reason_json(
        reason: &serde_json::Value,
        symbol: &str,
        tf_minutes: i16,
        side: i16,
        entry_price: f64,
        sl: f64,
        tp1: f64,
        tp2: f64,
        tp3: f64,
        ml_score_ext: Option<f64>,
        heur_score_ext: Option<f64>,
        price10_target: Option<f64>,
        price10_score: Option<f64>,
        bounce_prob: Option<f64>,
        bounce_score: Option<f64>,
        breakout_prob: Option<f64>,
        breakout_score: Option<f64>,
    ) -> Self {
        let get_f64 = |key: &str| reason.get(key).and_then(|v| v.as_f64());
        let _get_str = |key: &str| reason.get(key).and_then(|v| v.as_str()).map(String::from);

        let final_score = get_f64("final").unwrap_or(0.0);
        let base_score = get_f64("base_score").unwrap_or(0.0);

        // TP/SL distances
        let sl_dist = (sl - entry_price).abs();
        let tp1_dist = (tp1 - entry_price).abs();
        let sl_pct = if entry_price > 0.0 { sl_dist / entry_price } else { 0.0 };
        let tp1_pct = if entry_price > 0.0 { tp1_dist / entry_price } else { 0.0 };
        let risk_reward_ratio = if sl_dist > 0.0 { tp1_dist / sl_dist } else { 0.0 };

        // Level info
        let n_support = reason.get("support_levels_used")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let n_resistance = reason.get("resistance_levels_used")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let level_aware = reason.get("level_aware")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Market params from debug
        let debug = reason.get("debug").unwrap_or(reason);
        let market_params = debug.get("market_params")
            .or_else(|| reason.get("market_params"));
        let market_situation = market_params
            .and_then(|mp| mp.get("situation"))
            .and_then(|v| v.as_str())
            .map(String::from);
        let market_quality_score = market_params
            .and_then(|mp| mp.get("market_quality_score"))
            .and_then(|v| v.as_f64());

        Self {
            symbol: symbol.to_string(),
            tf_minutes,
            side,
            final_score,
            base_score,
            predictors_score: get_f64("predictors_score")
                .or_else(|| debug.get("predictors_score").and_then(|v| v.as_f64()))
                .unwrap_or(0.0),
            raw_signals_score: get_f64("raw_signals_score")
                .or_else(|| debug.get("raw_signals_score").and_then(|v| v.as_f64()))
                .unwrap_or(0.0),
            indicators_score: get_f64("indicators_score")
                .or_else(|| debug.get("indicators_score").and_then(|v| v.as_f64()))
                .unwrap_or(0.0),
            market_score: get_f64("market_score")
                .or_else(|| debug.get("market_score").and_then(|v| v.as_f64()))
                .unwrap_or(0.0),
            coverage_score: get_f64("coverage")
                .or_else(|| debug.get("coverage_score").and_then(|v| v.as_f64()))
                .unwrap_or(0.0),
            consensus_score: get_f64("consensus")
                .or_else(|| debug.get("consensus_score").and_then(|v| v.as_f64()))
                .unwrap_or(0.0),

            ml_score: ml_score_ext,
            heur_score: heur_score_ext,

            price10_target,
            price10_score,
            bounce_prob,
            bounce_score,
            breakout_prob,
            breakout_score,

            atr: get_f64("atr"),
            atr_pct: debug.get("atr_pct").and_then(|v| v.as_f64()),
            trend_strength: debug.get("trend_strength").and_then(|v| v.as_f64()),
            momentum_strength: debug.get("momentum_strength").and_then(|v| v.as_f64()),
            volatility_regime: debug.get("volatility_regime").and_then(|v| v.as_f64()),
            volume_spike_score: debug.get("volume_spike_score").and_then(|v| v.as_f64()),

            level_aware,
            n_support_levels: n_support,
            n_resistance_levels: n_resistance,

            market_situation,
            market_quality_score,

            entry_price,
            stop_loss: sl,
            tp1,
            tp2,
            tp3,

            risk_reward_ratio,
            sl_pct,
            tp1_pct,
        }
    }

    /// Convert to flat f32 vector for XGBoost inference.
    /// Order must match training schema exactly (see trainer.rs).
    pub fn to_feature_vector(&self) -> Vec<f32> {
        vec![
            self.tf_minutes as f32,
            self.side as f32,
            self.final_score as f32,
            self.base_score as f32,
            self.predictors_score as f32,
            self.raw_signals_score as f32,
            self.indicators_score as f32,
            self.market_score as f32,
            self.coverage_score as f32,
            self.consensus_score as f32,
            self.ml_score.unwrap_or(0.0) as f32,
            self.heur_score.unwrap_or(0.0) as f32,
            self.price10_score.unwrap_or(0.0) as f32,
            self.bounce_prob.unwrap_or(0.0) as f32,
            self.bounce_score.unwrap_or(0.0) as f32,
            self.breakout_prob.unwrap_or(0.0) as f32,
            self.breakout_score.unwrap_or(0.0) as f32,
            self.atr_pct.unwrap_or(0.0) as f32,
            self.trend_strength.unwrap_or(0.0) as f32,
            self.momentum_strength.unwrap_or(0.0) as f32,
            self.volatility_regime.unwrap_or(0.0) as f32,
            self.volume_spike_score.unwrap_or(0.0) as f32,
            if self.level_aware { 1.0 } else { 0.0 },
            self.n_support_levels as f32,
            self.n_resistance_levels as f32,
            self.market_quality_score.unwrap_or(0.0) as f32,
            self.risk_reward_ratio as f32,
            self.sl_pct as f32,
            self.tp1_pct as f32,
        ]
    }

    /// Feature names matching to_feature_vector() order (for training schema)
    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "tf_minutes", "side", "final_score", "base_score",
            "predictors_score", "raw_signals_score", "indicators_score",
            "market_score", "coverage_score", "consensus_score",
            "ml_score", "heur_score", "price10_score",
            "bounce_prob", "bounce_score", "breakout_prob", "breakout_score",
            "atr_pct", "trend_strength", "momentum_strength",
            "volatility_regime", "volume_spike_score",
            "level_aware", "n_support_levels", "n_resistance_levels",
            "market_quality_score", "risk_reward_ratio", "sl_pct", "tp1_pct",
        ]
    }
}
