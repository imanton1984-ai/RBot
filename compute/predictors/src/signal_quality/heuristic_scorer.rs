// signal_quality/heuristic_scorer.rs
//
// Rule-based quality assessment of final trade signals.
// Produces a quality_multiplier in [0.5, 1.5] based on signal feature analysis.
//
// NOT YET WIRED INTO PIPELINE — standalone, ready for Strategy/Backtester integration.

use super::types::*;

pub struct HeuristicQualityScorer {
    /// Minimum risk/reward ratio to consider a signal attractive
    min_risk_reward: f64,
    /// Strong trend threshold (trend_strength)
    strong_trend_threshold: f64,
}

impl Default for HeuristicQualityScorer {
    fn default() -> Self {
        Self {
            min_risk_reward: 1.2,
            strong_trend_threshold: 0.6,
        }
    }
}

impl HeuristicQualityScorer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Score a signal's quality based on heuristic rules.
    /// Returns QualityResult with multiplier in [0.5, 1.5].
    pub fn score(&self, features: &SignalFeatures) -> QualityResult {
        let factors = self.compute_factors(features);

        // Weighted combination of factors
        let heuristic_quality =
            factors.score_strength * 0.15
            + factors.component_agreement * 0.20
            + factors.risk_reward * 0.20
            + factors.level_confluence * 0.10
            + factors.market_alignment * 0.15
            + factors.predictor_confidence * 0.10
            + factors.source_agreement * 0.10;

        let heuristic_quality = heuristic_quality.clamp(0.0, 1.0);

        // Convert to multiplier: 0.0 quality → 0.5x, 0.5 quality → 1.0x, 1.0 quality → 1.5x
        let quality_multiplier = (0.5 + heuristic_quality).clamp(0.5, 1.5);

        let grade = match heuristic_quality {
            q if q >= 0.80 => SignalGrade::A,
            q if q >= 0.60 => SignalGrade::B,
            q if q >= 0.40 => SignalGrade::C,
            _ => SignalGrade::D,
        };

        QualityResult {
            quality_multiplier,
            grade,
            breakdown: QualityBreakdown {
                heuristic_quality,
                ml_quality: None,
                combined_quality: heuristic_quality,
                factors,
            },
        }
    }

    fn compute_factors(&self, f: &SignalFeatures) -> QualityFactors {
        QualityFactors {
            score_strength: self.score_strength(f),
            component_agreement: self.component_agreement(f),
            risk_reward: self.risk_reward_quality(f),
            level_confluence: self.level_confluence(f),
            market_alignment: self.market_alignment(f),
            predictor_confidence: self.predictor_confidence(f),
            source_agreement: self.source_agreement(f),
        }
    }

    /// How strong is the raw final_score? Higher = better.
    fn score_strength(&self, f: &SignalFeatures) -> f64 {
        // Map [0.55, 0.85] → [0, 1] (signals below 0.55 shouldn't exist)
        ((f.final_score - 0.55) / 0.30).clamp(0.0, 1.0)
    }

    /// Do all component scores agree (all high) or are they mixed?
    /// Penalizes signals where one component is very high but others are low.
    fn component_agreement(&self, f: &SignalFeatures) -> f64 {
        let components = [
            f.predictors_score,
            f.raw_signals_score,
            f.indicators_score,
            f.market_score,
        ];

        let mean = components.iter().sum::<f64>() / components.len() as f64;
        let variance = components.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / components.len() as f64;
        let std_dev = variance.sqrt();

        // Low std_dev = good agreement. Map [0, 0.3] → [1, 0]
        (1.0 - (std_dev / 0.3)).clamp(0.0, 1.0)
    }

    /// Is the risk/reward ratio attractive?
    fn risk_reward_quality(&self, f: &SignalFeatures) -> f64 {
        if f.risk_reward_ratio <= 0.0 {
            return 0.0;
        }

        // Map R:R ratio to quality score
        // 0.5 → 0.0 (terrible), 1.0 → 0.3, 1.5 → 0.6, 2.0 → 0.8, 3.0+ → 1.0
        let rr = f.risk_reward_ratio;
        if rr >= 3.0 { 1.0 }
        else if rr >= 2.0 { 0.8 + 0.2 * ((rr - 2.0) / 1.0) }
        else if rr >= 1.5 { 0.6 + 0.2 * ((rr - 1.5) / 0.5) }
        else if rr >= 1.0 { 0.3 + 0.3 * ((rr - 1.0) / 0.5) }
        else { (rr / 1.0) * 0.3 }
    }

    /// Do support/resistance levels support the trade direction?
    fn level_confluence(&self, f: &SignalFeatures) -> f64 {
        if !f.level_aware {
            return 0.5; // Neutral if no levels used
        }

        let mut score = 0.5;

        // For LONG: want support below and resistance far above
        // For SHORT: want resistance above and support far below
        if f.side > 0 {
            // LONG benefits from support below
            if f.n_support_levels > 0 { score += 0.2; }
            if f.n_support_levels >= 2 { score += 0.1; }
            // Resistance above is fine (targets)
            if f.n_resistance_levels > 0 { score += 0.1; }
        } else {
            // SHORT benefits from resistance above
            if f.n_resistance_levels > 0 { score += 0.2; }
            if f.n_resistance_levels >= 2 { score += 0.1; }
            if f.n_support_levels > 0 { score += 0.1; }
        }

        score.clamp(0.0, 1.0)
    }

    /// Is the market regime aligned with the trade direction?
    fn market_alignment(&self, f: &SignalFeatures) -> f64 {
        let situation = f.market_situation.as_deref().unwrap_or("flat");

        let alignment = match (situation, f.side > 0) {
            ("uptrend", true) => 1.0,         // Long in uptrend = great
            ("downtrend", false) => 1.0,       // Short in downtrend = great
            ("uptrend", false) => 0.2,         // Short in uptrend = risky
            ("downtrend", true) => 0.2,        // Long in downtrend = risky
            ("flat", _) => 0.5,                // Neutral
            ("pullback_in_uptrend", true) => 0.7,  // Buying the dip
            ("pullback_in_uptrend", false) => 0.4,
            ("relief_in_downtrend", false) => 0.7,  // Shorting the relief
            ("relief_in_downtrend", true) => 0.4,
            _ => 0.5,
        };

        // Boost if trend is strong
        let trend_boost = f.trend_strength.unwrap_or(0.0);
        let quality = f.market_quality_score.unwrap_or(0.0);

        (alignment * 0.6 + trend_boost * 0.2 + quality * 0.2).clamp(0.0, 1.0)
    }

    /// How confident are the predictors?
    fn predictor_confidence(&self, f: &SignalFeatures) -> f64 {
        let pred = f.predictors_score;
        let price_conf = f.price10_score.unwrap_or(0.0);

        // Bonus for high bounce/breakout scores
        let level_conf = f.bounce_score.unwrap_or(0.0)
            .max(f.breakout_score.unwrap_or(0.0));

        ((pred * 0.5 + price_conf * 0.3 + level_conf * 0.2) as f64).clamp(0.0, 1.0)
    }

    /// Do ML and heuristic predictors agree?
    fn source_agreement(&self, f: &SignalFeatures) -> f64 {
        match (f.ml_score, f.heur_score) {
            (Some(ml), Some(heur)) => {
                let diff = (ml - heur).abs();
                // Small diff = good agreement
                (1.0 - diff / 0.4).clamp(0.0, 1.0)
            }
            (Some(_), None) | (None, Some(_)) => {
                // Single source: acceptable but not great
                0.6
            }
            (None, None) => 0.3, // No predictor scores at all
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_features(final_score: f64, rr: f64) -> SignalFeatures {
        SignalFeatures {
            symbol: "BTCUSDT".into(),
            tf_minutes: 5,
            side: 1,
            final_score,
            base_score: final_score * 1.1,
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
            tp1: 69000.0 + 500.0 * rr,
            tp2: 69000.0 + 900.0 * rr,
            tp3: 69000.0 + 1400.0 * rr,
            risk_reward_ratio: rr,
            sl_pct: 500.0 / 69000.0,
            tp1_pct: 500.0 * rr / 69000.0,
        }
    }

    #[test]
    fn test_high_quality_signal() {
        let scorer = HeuristicQualityScorer::new();
        let features = make_test_features(0.75, 2.0);
        let result = scorer.score(&features);

        assert!(result.quality_multiplier > 1.0, "High quality signal should boost: got {}", result.quality_multiplier);
        assert!(matches!(result.grade, SignalGrade::A | SignalGrade::B));
    }

    #[test]
    fn test_low_quality_signal() {
        let scorer = HeuristicQualityScorer::new();
        let mut features = make_test_features(0.56, 0.5);
        features.ml_score = None;
        features.heur_score = None;
        features.level_aware = false;
        features.market_situation = Some("downtrend".into()); // long in downtrend = bad
        let result = scorer.score(&features);

        assert!(result.quality_multiplier < 1.0, "Low quality signal should penalize: got {}", result.quality_multiplier);
    }
}
