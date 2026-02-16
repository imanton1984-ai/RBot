// compute/predictors/future_price/heuristic_predictor.rs
//
// Multi-indicator heuristic price predictor.
// Uses ALL available indicators for direction + strength + confidence scoring.

use anyhow::Result;
use crate::feature_view::FeatureView;

pub struct FuturePriceHeuristic;

impl FuturePriceHeuristic {
    pub fn new() -> Self { Self }

    pub fn predict(&self, _symbol: &str, _timeframe: &str, view: &FeatureView) -> Result<Option<(Vec<f64>, f64)>> {
        let ind = &view.indicators;
        let close = ind.close as f64;
        if close <= 0.0 { return Ok(None); }

        let atr = ind.atr as f64;
        let atr_pct = if close > 0.0 { atr / close } else { 0.01 };

        // === DIRECTION SIGNALS: each contributes +1 (bull) or -1 (bear) ===
        let mut bull_signals: f64 = 0.0;
        let mut bear_signals: f64 = 0.0;
        let mut total_weight: f64 = 0.0;

        // 1. Trend short (weight: 2.0 — most important)
        let trend_short = ind.trend_short as f64;
        if trend_short > 0.0 { bull_signals += 2.0; }
        else if trend_short < 0.0 { bear_signals += 2.0; }
        total_weight += 2.0;

        // 2. Trend medium (weight: 1.5)
        let trend_medium = ind.trend_medium as f64;
        if trend_medium > 0.0 { bull_signals += 1.5; }
        else if trend_medium < 0.0 { bear_signals += 1.5; }
        total_weight += 1.5;

        // 3. EMA alignment: close above/below EMAs (weight: 1.5 total)
        let ema_20 = ind.ema_20 as f64;
        let ema_50 = ind.ema_50 as f64;
        let ema_200 = ind.ema_200 as f64;

        if ema_20 > 0.0 {
            if close > ema_20 { bull_signals += 0.5; } else { bear_signals += 0.5; }
            total_weight += 0.5;
        }
        if ema_50 > 0.0 {
            if close > ema_50 { bull_signals += 0.5; } else { bear_signals += 0.5; }
            total_weight += 0.5;
        }
        if ema_200 > 0.0 {
            if close > ema_200 { bull_signals += 0.5; } else { bear_signals += 0.5; }
            total_weight += 0.5;
        }

        // 4. EMA order: EMA20 > EMA50 > EMA200 = bullish stack (weight: 1.0)
        if ema_20 > 0.0 && ema_50 > 0.0 && ema_200 > 0.0 {
            if ema_20 > ema_50 && ema_50 > ema_200 { bull_signals += 1.0; }
            else if ema_20 < ema_50 && ema_50 < ema_200 { bear_signals += 1.0; }
            total_weight += 1.0;
        }

        // 5. MACD histogram (weight: 1.5)
        let macd_hist = ind.macd_histogram as f64;
        if macd_hist > 0.001 { bull_signals += 1.5; }
        else if macd_hist < -0.001 { bear_signals += 1.5; }
        total_weight += 1.5;

        // 6. RSI with mean reversion logic (weight: 1.0)
        let rsi = ind.rsi as f64;
        if rsi > 70.0 {
            // Overbought: mean reversion → bearish
            bear_signals += 1.0;
        } else if rsi < 30.0 {
            // Oversold: mean reversion → bullish
            bull_signals += 1.0;
        } else if rsi > 55.0 {
            bull_signals += 0.3;
        } else if rsi < 45.0 {
            bear_signals += 0.3;
        }
        total_weight += 1.0;

        // 7. Stochastic (weight: 0.8)
        let stoch_k = ind.stoch_k as f64;
        let stoch_d = ind.stoch_d as f64;
        if stoch_k > stoch_d && stoch_k > 20.0 && stoch_k < 80.0 {
            bull_signals += 0.8; // K crossing above D = bullish
        } else if stoch_k < stoch_d && stoch_k > 20.0 && stoch_k < 80.0 {
            bear_signals += 0.8; // K crossing below D = bearish
        }
        total_weight += 0.8;

        // 8. Williams %R (weight: 0.5)
        let williams = ind.williams_r as f64;
        if williams > -20.0 {
            bear_signals += 0.5; // Overbought
        } else if williams < -80.0 {
            bull_signals += 0.5; // Oversold
        }
        total_weight += 0.5;

        // 9. CCI (weight: 0.5)
        let cci = ind.cci as f64;
        if cci > 100.0 { bull_signals += 0.3; bear_signals += 0.2; } // Strong momentum but overextended
        else if cci < -100.0 { bear_signals += 0.3; bull_signals += 0.2; }
        else if cci > 0.0 { bull_signals += 0.3; }
        else { bear_signals += 0.3; }
        total_weight += 0.5;

        // 10. Bollinger Bands position (weight: 0.8)
        let bb_upper = ind.bb_upper as f64;
        let bb_lower = ind.bb_lower as f64;
        if bb_upper > 0.0 && bb_lower > 0.0 {
            if close > bb_upper {
                bear_signals += 0.8; // Above upper band → mean reversion down
            } else if close < bb_lower {
                bull_signals += 0.8; // Below lower band → mean reversion up
            }
            total_weight += 0.8;
        }

        // 11. Volume confirmation (weight: 0.5)
        let vol_spike = ind.volume_spike as f64;
        if vol_spike > 1.5 {
            // Strong volume — amplifies the dominant direction
            if bull_signals > bear_signals { bull_signals += 0.5; }
            else { bear_signals += 0.5; }
            total_weight += 0.5;
        }

        // 12. ADX strength (weight: 0.5 — modifies confidence, not direction)
        let adx = ind.adx as f64;
        let adx_strength = ((adx - 15.0) / 25.0).clamp(0.0, 1.0); // 0..1

        // === COMPUTE DIRECTIONAL SCORE ===
        if total_weight < 1.0 { return Ok(None); }

        // Net direction: positive = bullish, negative = bearish
        let net = (bull_signals - bear_signals) / total_weight; // range [-1, 1]

        // Score: combines direction strength with ADX trend strength
        // ADX boosts score in trending, reduces in ranging
        let direction_conf = net.abs(); // how strong is the consensus
        let score = net * (0.5 + 0.5 * adx_strength) * (0.5 + 0.5 * direction_conf);

        // Require minimum consensus to produce a prediction
        if score.abs() < 0.15 {
            return Ok(None); // No clear direction — don't predict
        }

        // Clamp to [-1, 1] — pipeline uses score.abs() for score_norm which must be [0, 1]
        let score_clamped = score.clamp(-1.0, 1.0);
        // Extra safety: ensure abs(score) <= 1.0 for DB CHECK constraint
        debug_assert!(score_clamped.abs() <= 1.0, "score_clamped out of range: {}", score_clamped);

        // === PRICE PREDICTION ===
        // Use ATR-scaled predictions with volatility adjustment
        let vol_adj = atr_pct.clamp(0.002, 0.05);
        let mut predicted_prices = Vec::with_capacity(10);
        for i in 1..=10 {
            let step = (i as f64) * 0.1;
            // Movement scaled by ATR and score confidence
            let movement = score_clamped * atr * step * (1.0 + vol_adj * 2.0);
            let predicted_price = (close + movement).max(0.00000001);
            predicted_prices.push(predicted_price);
        }

        Ok(Some((predicted_prices, score_clamped)))
    }
}
