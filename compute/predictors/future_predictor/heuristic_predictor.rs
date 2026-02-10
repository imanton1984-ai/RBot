// compute/predictors/future_price/heuristic_predictor.rs

use anyhow::Result;
use crate::predictors::feature_view::FeatureView;

pub struct FuturePriceHeuristic;

impl FuturePriceHeuristic {
    pub fn new() -> Self { Self }

    pub fn predict(&self, _symbol: &str, _timeframe: &str, view: &FeatureView) -> Result<Option<(Vec<f64>, f64)>> {
        
        let adx = view.indicators.adx as f64;
        let rsi = view.indicators.rsi as f64;
        let trend_short = view.indicators.trend_short as f64;
        
        let trend_direction = trend_short.signum();
        
        // This is from the user snippet for calculate_score
        let trend = adx * trend_direction;
        let atr = view.indicators.atr as f64;
        let close = view.indicators.close as f64;
        let atr_pct = if close > 0.0 { atr / close } else { 0.0 };
        let vol_adj = atr_pct.clamp(0.001, 0.05); // not used in user snippet, but I'll keep it.
        
        let mut score = trend * 0.5;
        if rsi < 40.0 && trend > 0.0 { score += 0.2; }
        
        let score = score.clamp(-1.0, 1.0);

        // Price prediction logic
        let current_price = view.indicators.close as f64;
        let mut predicted_prices = Vec::with_capacity(10);
        for i in 1..=10 {
            let step = (i as f64) * 0.1;
            let movement = score * atr * step; 
            predicted_prices.push(current_price + movement);
        }
        
        if score.abs() > 0.1 {
            Ok(Some((predicted_prices, score)))
        } else {
            Ok(None)
        }
    }
}