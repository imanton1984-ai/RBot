use crate::thresholds::{RawSignal, RawSignalType, SignalConfig, SignalKind};
use crate::scoring::sigmoid_normalize;
use common::{Symbol, Timeframe};
use std::vec::Vec;

/// Calculate Stochastic raw signals for extreme values
pub fn calculate_stoch_raw_signals(
    k_values: &[f64],
    d_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..k_values.len() {
        if i >= d_values.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let k_val = k_values[i];
        let d_val = d_values[i];
        
        if k_val.is_nan() || d_val.is_nan() {
            continue;
        }
        
        // Calculate signal based on how extreme the values are
        let mut signal_strength = 0.0;
        let mut is_valid_signal = false;
        let mut side = 0;
        
        // Check for overbought conditions (both %K and %D above 80)
        if k_val >= 80.0 && d_val >= 80.0 {
            signal_strength = ((k_val - 50.0) / 50.0).min(1.0);
            is_valid_signal = true;
            side = -1;
        }
        // Check for oversold conditions (both %K and %D below 20)
        else if k_val <= 20.0 && d_val <= 20.0 {
            signal_strength = ((50.0 - k_val) / 50.0).min(1.0);
            is_valid_signal = true;
            side = 1;
        }
        
        if is_valid_signal {
            // Normalize the signal strength to score
            let score = sigmoid_normalize(signal_strength, 0.3, 6.0);
            
            // Create raw signal
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::Stochastic.to_indicator_id(),
                SignalKind::PriceRelation.to_i16(),
                side,
                score as f32,
                k_val as f32,
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

/// Calculate Stochastic crossover signals (%K crossing %D)
pub fn calculate_stoch_crossover_signals(
    k_values: &[f64],
    d_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to detect crossovers
    for i in 1..k_values.len() {
        if i >= d_values.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_k = k_values[i - 1];
        let prev_d = d_values[i - 1];
        let curr_k = k_values[i];
        let curr_d = d_values[i];
        
        if prev_k.is_nan() || prev_d.is_nan() || curr_k.is_nan() || curr_d.is_nan() {
            continue;
        }
        
        // Detect bullish crossover (%K crosses above %D)
        let is_bullish_cross = prev_k <= prev_d && curr_k > curr_d;
        // Detect bearish crossover (%K crosses below %D)
        let is_bearish_cross = prev_k >= prev_d && curr_k < curr_d;
        
        if is_bullish_cross || is_bearish_cross {
            // Use the absolute difference as signal strength
            let cross_strength = (curr_k - curr_d).abs();
            let score = normalize_crossover_score(cross_strength);
            let side = if is_bullish_cross { 1 } else { -1 };
            
            // Create raw signal for crossover
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::Stochastic.to_indicator_id(),
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

/// Calculate Stochastic divergence signals
pub fn calculate_stoch_divergence_signals(
    prices: &[f64],
    k_values: &[f64],
    d_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate divergence
    for i in 1..prices.len() {
        if i >= k_values.len() || i >= d_values.len() ||
           i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_price = prices[i - 1];
        let curr_price = prices[i];
        let prev_k = k_values[i - 1];
        let curr_k = k_values[i];
        let prev_d = d_values[i - 1];
        let curr_d = d_values[i];
        
        if prev_price.is_nan() || curr_price.is_nan() || 
           prev_k.is_nan() || curr_k.is_nan() || 
           prev_d.is_nan() || curr_d.is_nan() {
            continue;
        }
        
        // Calculate price and stochastic changes
        let price_change = curr_price - prev_price;
        let k_change = curr_k - prev_k;
        let d_change = curr_d - prev_d;
        
        // Bullish divergence: price makes lower low, stochastic makes higher low
        let is_bullish_div = price_change < 0.0 && (k_change > 0.0 || d_change > 0.0) && curr_k < 20.0;
        
        // Bearish divergence: price makes higher high, stochastic makes lower high
        let is_bearish_div = price_change > 0.0 && (k_change < 0.0 || d_change < 0.0) && curr_k > 80.0;
        
        if is_bullish_div || is_bearish_div {
            // Calculate divergence strength
            let divergence_strength = (price_change.abs() + k_change.abs() + d_change.abs()) / 3.0;
            let score = normalize_divergence_score(divergence_strength);
            let side = if is_bullish_div { 1 } else { -1 };
            
            // Create raw signal for divergence
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::Stochastic.to_indicator_id(),
                SignalKind::PriceRelation.to_i16(),
                side,
                score as f32,
                divergence_strength as f32,
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

/// Calculate Stochastic momentum signals based on rate of change
pub fn calculate_stoch_momentum_signals(
    k_values: &[f64],
    d_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate momentum
    for i in 1..k_values.len() {
        if i >= d_values.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_k = k_values[i - 1];
        let curr_k = k_values[i];
        let prev_d = d_values[i - 1];
        let curr_d = d_values[i];
        
        if prev_k.is_nan() || curr_k.is_nan() || prev_d.is_nan() || curr_d.is_nan() {
            continue;
        }
        
        // Calculate Stochastic momentum (rate of change)
        let k_momentum = curr_k - prev_k;
        let d_momentum = curr_d - prev_d;
        let avg_momentum = (k_momentum.abs() + d_momentum.abs()) / 2.0;
        
        // Normalize momentum to score
        let score = normalize_momentum_score(avg_momentum);
        let side = if k_momentum > 0.0 { 1 } else { -1 };
        
        // Create raw signal for momentum
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::Stochastic.to_indicator_id(),
            SignalKind::Volatility.to_i16(),
            side,
            score as f32,
            avg_momentum as f32,
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
    // Normalize assuming max meaningful crossover strength is 50 units
    let normalized = (cross_strength / 50.0).min(1.0);
    
    // Apply sigmoid to emphasize strong crossovers
    sigmoid_normalize(normalized, 0.2, 5.0)
}

/// Helper function to normalize divergence score
fn normalize_divergence_score(divergence_strength: f64) -> f64 {
    // Normalize assuming max meaningful divergence is 50 units
    let normalized = (divergence_strength / 50.0).min(1.0);
    
    // Apply sigmoid to emphasize strong divergences
    sigmoid_normalize(normalized, 0.2, 5.0)
}

/// Helper function to normalize momentum score
fn normalize_momentum_score(momentum: f64) -> f64 {
    // Normalize assuming max meaningful momentum is 100 units
    let normalized = (momentum / 100.0).min(1.0);
    
    // Apply sigmoid to emphasize strong momentum
    sigmoid_normalize(normalized, 0.2, 5.0)
}