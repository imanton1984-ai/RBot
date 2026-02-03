use crate::thresholds::{RawSignal, RawSignalType, SignalConfig, SignalKind};
use crate::scoring::sigmoid_normalize;
use common::{Symbol, Timeframe};
use std::vec::Vec;

/// Calculate POC raw signals based on price relation to Point of Control
pub fn calculate_poc_raw_signals(
    prices: &[f64],
    poc_levels: &[f64],  // POC levels for each time period
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..prices.len() {
        if i >= poc_levels.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let current_price = prices[i];
        let poc_level = poc_levels[i];
        
        if current_price.is_nan() || poc_level.is_nan() {
            continue;
        }
        
        // Calculate distance from POC as percentage
        let distance_pct = ((current_price - poc_level) / poc_level).abs() * 100.0;
        
        // Inverse relationship: closer to POC = higher significance
        // But we want to detect when price moves away from POC significantly
        let normalized_distance = (distance_pct / 5.0).min(1.0); // Assuming max 5% deviation matters
        let score = sigmoid_normalize(normalized_distance, 0.3, 6.0);
        
        // Create raw signal
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::Poc.to_indicator_id(),
            SignalKind::PriceRelation.to_i16(),
            0, // side
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

/// Calculate POC breakout signals (when price moves significantly away from POC)
pub fn calculate_poc_breakout_signals(
    prices: &[f64],
    poc_levels: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to detect breakouts
    for i in 1..prices.len() {
        if i >= poc_levels.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_price = prices[i - 1];
        let curr_price = prices[i];
        let poc_level = poc_levels[i];
        
        if prev_price.is_nan() || curr_price.is_nan() || poc_level.is_nan() {
            continue;
        }
        
        // Calculate distances from POC for previous and current prices
        let prev_distance = ((prev_price - poc_level) / poc_level).abs();
        let curr_distance = ((curr_price - poc_level) / poc_level).abs();
        
        // Detect breakout: price moving significantly away from POC
        let breakout_threshold = 0.02; // 2% threshold
        let is_breakout = curr_distance > breakout_threshold && prev_distance <= breakout_threshold;
        
        if is_breakout {
            // Calculate breakout strength
            let breakout_strength = curr_distance * 100.0; // Convert to percentage
            let normalized = (breakout_strength / 10.0).min(1.0); // Max 10% breakout
            let score = sigmoid_normalize(normalized, 0.3, 6.0);
            
            let side = if curr_price > poc_level { 1 } else { -1 };

            // Create raw signal for breakout
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::Poc.to_indicator_id(),
                SignalKind::Breakout.to_i16(),
                side,
                score as f32,
                breakout_strength as f32,
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

/// Calculate POC convergence signals (when price moves toward POC)
pub fn calculate_poc_convergence_signals(
    prices: &[f64],
    poc_levels: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to detect convergence
    for i in 1..prices.len() {
        if i >= poc_levels.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_price = prices[i - 1];
        let curr_price = prices[i];
        let poc_level = poc_levels[i];
        
        if prev_price.is_nan() || curr_price.is_nan() || poc_level.is_nan() {
            continue;
        }
        
        // Calculate distances from POC for previous and current prices
        let prev_distance = ((prev_price - poc_level) / poc_level).abs();
        let curr_distance = ((curr_price - poc_level) / poc_level).abs();
        
        // Detect convergence: price moving closer to POC
        let is_converging = curr_distance < prev_distance && prev_distance > 0.01; // More than 1% away initially
        
        if is_converging {
            // Calculate convergence strength
            let convergence_strength = (prev_distance - curr_distance) * 100.0; // Percentage improvement
            let normalized = (convergence_strength / 5.0).min(1.0); // Max 5% convergence
            let score = sigmoid_normalize(normalized, 0.3, 6.0);
            
            let side = if curr_price > poc_level { 1 } else { -1 };

            // Create raw signal for convergence
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::Poc.to_indicator_id(),
                SignalKind::Convergence.to_i16(),
                side,
                score as f32,
                convergence_strength as f32,
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

/// Calculate POC volatility signals (based on changes in POC levels)
pub fn calculate_poc_volatility_signals(
    poc_levels: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate volatility
    for i in 1..poc_levels.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_poc = poc_levels[i - 1];
        let curr_poc = poc_levels[i];
        
        if prev_poc.is_nan() || curr_poc.is_nan() || prev_poc == 0.0 {
            continue;
        }
        
        // Calculate POC volatility (rate of change)
        let poc_change_pct = ((curr_poc - prev_poc) / prev_poc) * 100.0;
        let abs_change = poc_change_pct.abs();
        
        // Normalize volatility to score
        let normalized = (abs_change / 5.0).min(1.0); // Max 5% change
        let score = sigmoid_normalize(normalized, 0.2, 5.0);
        
        let side = if poc_change_pct > 0.0 { 1 } else { -1 };

        // Create raw signal for volatility
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::Poc.to_indicator_id(),
            SignalKind::Volatility.to_i16(),
            side,
            score as f32,
            poc_change_pct as f32,
            None,
        );
        
        // Only include signals that meet our threshold criteria
        if signal.is_above_threshold(config) {
            signals.push(signal);
        }
    }
    
    signals
}