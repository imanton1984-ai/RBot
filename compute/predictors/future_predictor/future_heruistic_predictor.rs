// compute/predictors/future_price/heuristic_predictor.rs

use anyhow::Result;
use crate::predictors::feature_view::FeatureView;

pub struct FuturePriceHeuristic;

impl FuturePriceHeuristic {
    pub fn new() -> Self { Self }

    pub fn predict(&self, _symbol: &str, _timeframe: &str, features: &FeatureView) -> Result<Option<(Vec<f64>, f64)>> {
        // Логика: Линейная экстраполяция на основе EMA и тренда
        let current_price = features.close;
        let trend = features.trend_short.unwrap_or(0.0); // -1.0 to 1.0
        let atr = features.atr.unwrap_or(current_price * 0.01);
        
        let mut predicted_prices = Vec::with_capacity(10);
        
        // Простая модель: цена движется по тренду с затуханием
        for i in 1..=10 {
            let step = (i as f64) * 0.1; 
            let movement = trend * atr * step; 
            predicted_prices.push(current_price + movement);
        }

        // Score: уверенность на основе силы тренда и RSI
        let rsi = features.rsi.unwrap_or(50.0);
        let mut score = 0.5;
        
        // Если тренд сильный и RSI не в экстремуме -> высокий скор
        if trend.abs() > 0.5 && rsi > 30.0 && rsi < 70.0 {
            score = 0.85 + (trend.abs() * 0.1); 
        }

        if score >= 0.80 {
            Ok(Some((predicted_prices, score)))
        } else {
            Ok(None)
        }
    }
}