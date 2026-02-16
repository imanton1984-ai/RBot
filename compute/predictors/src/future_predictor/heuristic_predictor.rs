// compute/predictors/future_price/heuristic_predictor.rs
//
// Consensus-based heuristic predictor with 3 independent "agencies":
// 1. Trend Agency (direction & strength)
// 2. Momentum Agency (RSI, MACD)
// 3. Overheat Filter (mean reversion protection)
//
// Key insight: When agencies disagree, score is reduced (conflict = noise/divergence)

use anyhow::Result;
use crate::feature_view::FeatureView;

pub struct FuturePriceHeuristic;

impl FuturePriceHeuristic {
    pub fn new() -> Self { Self }

    pub fn predict(&self, _symbol: &str, _timeframe: &str, view: &FeatureView) -> Result<Option<(Vec<f64>, f64)>> {
        let ind = &view.indicators;
        let close = ind.close as f64;
        let atr = ind.atr as f64;
        
        // === 1. TREND AGENCY (Strength & Direction) ===
        let mut trend_score: f64 = 0.0;
        if ind.trend_short > 0.0 { trend_score += 1.0; } else { trend_score -= 1.0; }
        if ind.trend_medium > 0.0 { trend_score += 1.0; } else { trend_score -= 1.0; }
        if close > ind.ema_200 as f64 { trend_score += 1.0; } else { trend_score -= 1.0; }
        let trend_norm: f64 = trend_score / 3.0; // [-1.0 ... 1.0]

        // === 2. MOMENTUM AGENCY (RSI, MACD) ===
        let mut mom_score: f64 = 0.0;
        let rsi = ind.rsi as f64;
        // Logic: Buy when RSI exits oversold (30->40), not when it's already 80!
        if rsi < 35.0 { mom_score += 1.0; }
        else if rsi > 65.0 { mom_score -= 1.0; }
        else if rsi > 50.0 && rsi < 60.0 { mom_score += 0.5; } // Moderate bullish
        
        if ind.macd_histogram > 0.0 { mom_score += 0.5; } else { mom_score -= 0.5; }
        let mom_norm: f64 = mom_score.clamp(-1.0, 1.0);

        // === 3. OVERHEAT FILTER (Mean Reversion Protection) ===
        // If price is too far from EMA20 — dangerous to enter (pullback likely)
        let ema_20 = ind.ema_20 as f64;
        let dist_from_ema = (close - ema_20) / ema_20;
        let atr_threshold = atr * 2.0 / close;
        let mean_reversion_penalty: f64 = if dist_from_ema.abs() > atr_threshold { 0.5 } else { 1.0 };

        // === 4. RVOL FILTER (Volume Confirmation) ===
        let vol = ind.volume as f64;
        let vol_sma = ind.volume_sma as f64;
        let rvol = if vol_sma > 0.0 { vol / vol_sma } else { 1.0 };
        
        // Ignore signals on low volume (false breakouts)
        if rvol < 1.2 {
            return Ok(None);
        }

        // === 5. FINAL CONSENSUS ===
        // If Trend and Momentum disagree — this is "noise" or "divergence"
        let mut final_score: f64 = if trend_norm.signum() == mom_norm.signum() {
            // Signals confirm each other — strong consensus
            (trend_norm.abs() + mom_norm.abs()) / 2.0 * trend_norm.signum()
        } else {
            // Conflict! Reduce confidence to minimum
            (trend_norm + mom_norm) * 0.2
        };

        final_score *= mean_reversion_penalty;

        // Ignore weak signals
        if final_score.abs() < 0.4 { return Ok(None); }

        // Price prediction based on ATR
        let mut predicted_prices = Vec::with_capacity(10);
        for i in 1..=10 {
            let move_size = (i as f64 * 0.2) * atr * final_score.signum();
            predicted_prices.push((close + move_size).max(0.00000001));
        }

        Ok(Some((predicted_prices, final_score.clamp(-1.0, 1.0))))
    }
}
