use crate::thresholds::{RawSignal, RawSignalType, SignalConfig, SignalKind};
use crate::scoring::sigmoid_normalize;
use common::{Symbol, Timeframe};
use std::vec::Vec;

/// Calculate EMA raw signals based on price vs EMA relationship
pub fn calculate_ema_raw_signals(
    prices: &[f64],
    ema_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..prices.len() {
        if i >= ema_values.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let price = prices[i];
        let ema = ema_values[i];
        
        if price.is_nan() || ema.is_nan() || ema == 0.0 {
            continue;
        }
        
        // Calculate distance from EMA as percentage
        let distance_pct = ((price - ema) / ema) * 100.0;
        let abs_distance = distance_pct.abs();
        
        // Normalize distance to score
        let normalized = (abs_distance / 10.0).min(1.0); // Assuming max 10% deviation
        let score = sigmoid_normalize(normalized, 0.3, 6.0);
        let side = if distance_pct > 0.0 { 1 } else { -1 };
        
        // Create raw signal
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::Ema.to_indicator_id(),
            SignalKind::PriceRelation.to_i16(),
            side,
            score as f32,
            distance_pct as f32,
            None,
        );
        
        // Only include signals that meet our threshold criteria
        if signal.is_above_threshold(config) {
            signals.push(signal);
        }
    }
    
    signals
}

/// Calculate EMA crossover signals (price crossing EMA)
pub fn calculate_ema_crossover_signals(
    prices: &[f64],
    ema_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to detect crossovers
    for i in 1..prices.len() {
        if i >= ema_values.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_price = prices[i - 1];
        let prev_ema = ema_values[i - 1];
        let curr_price = prices[i];
        let curr_ema = ema_values[i];
        
        if prev_price.is_nan() || prev_ema.is_nan() || curr_price.is_nan() || curr_ema.is_nan() {
            continue;
        }
        
        // Detect bullish crossover (price crosses above EMA)
        let is_bullish_cross = prev_price <= prev_ema && curr_price > curr_ema;
        // Detect bearish crossover (price crosses below EMA)
        let is_bearish_cross = prev_price >= prev_ema && curr_price < curr_ema;
        
        if is_bullish_cross || is_bearish_cross {
            // Use the absolute distance from EMA as signal strength
            let cross_strength = ((curr_price - curr_ema) / curr_ema).abs() * 100.0;
            let score = normalize_crossover_score(cross_strength);
            let side = if is_bullish_cross { 1 } else { -1 };
            
            // Create raw signal for crossover
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::Ema.to_indicator_id(),
                SignalKind::Crossover.to_i16(),
                side,
                score as f32,
                cross_strength as f32,
                None,
            );
            
            // Only include signals that meet our threshold criteria
            if signal.is_above_threshold(config) {
                signals.push(signal);
            }
        }
    }
    
    signals
}

/// Calculate EMA trend signals based on EMA slope
pub fn calculate_ema_trend_signals(
    ema_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate slope
    for i in 1..ema_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_ema = ema_values[i - 1];
        let curr_ema = ema_values[i];
        
        if prev_ema.is_nan() || curr_ema.is_nan() || prev_ema == 0.0 {
            continue;
        }
        
        // Calculate EMA slope as percentage change
        let slope_pct = ((curr_ema - prev_ema) / prev_ema) * 100.0;
        let abs_slope = slope_pct.abs();
        
        // Normalize slope to score
        let normalized = (abs_slope / 5.0).min(1.0); // Assuming max 5% change
        let score = sigmoid_normalize(normalized, 0.2, 5.0);
        let side = if slope_pct > 0.0 { 1 } else { -1 };

        // Create raw signal for trend
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::Ema.to_indicator_id(),
            SignalKind::Volatility.to_i16(),
            side,
            score as f32,
            slope_pct as f32,
            None,
        );
        
        // Only include signals that meet our threshold criteria
        if signal.is_above_threshold(config) {
            signals.push(signal);
        }
    }
    
    signals
}

/// Helper function to normalize crossover score
fn normalize_crossover_score(cross_strength: f64) -> f64 {
    // Normalize assuming max meaningful crossover strength is 5%
    let normalized = (cross_strength / 5.0).min(1.0);
    
    // Apply sigmoid to emphasize strong crossovers
    sigmoid_normalize(normalized, 0.2, 5.0)
}