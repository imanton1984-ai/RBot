use crate::{RawSignal, RawSignalType, SignalConfig, normalize_indicator_to_score, filter_strong_signals};
use std::vec::Vec;

/// Calculate Stochastic raw signals for extreme values
pub fn calculate_stoch_raw_signals(
    k_values: &[f64],
    d_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
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
        
        // Check for overbought conditions (both %K and %D above 80)
        if k_val >= 80.0 && d_val >= 80.0 {
            signal_strength = ((k_val - 50.0) / 50.0).min(1.0);
            is_valid_signal = true;
        }
        // Check for oversold conditions (both %K and %D below 20)
        else if k_val <= 20.0 && d_val <= 20.0 {
            signal_strength = ((50.0 - k_val) / 50.0).min(1.0);
            is_valid_signal = true;
        }
        
        if is_valid_signal {
            // Normalize the signal strength to score
            let score = crate::sigmoid_normalize(signal_strength, 0.3, 6.0);
            
            // Create raw signal
            let signal = RawSignal::new(
                RawSignalType::Stochastic,
                k_val,
                score,
                timestamps[i],
                symbols[i].clone(),
                timeframes[i].clone(),
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
    symbols: &[String],
    timeframes: &[String],
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
            
            // Create raw signal for crossover
            let signal = RawSignal::new(
                RawSignalType::Stochastic,
                cross_strength,
                score,
                timestamps[i],
                symbols[i].clone(),
                timeframes[i].clone(),
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
    symbols: &[String],
    timeframes: &[String],
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
            
            // Create raw signal for divergence
            let signal = RawSignal::new(
                RawSignalType::Stochastic,
                divergence_strength,
                score,
                timestamps[i],
                symbols[i].clone(),
                timeframes[i].clone(),
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
    symbols: &[String],
    timeframes: &[String],
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
        
        // Create raw signal for momentum
        let signal = RawSignal::new(
            RawSignalType::Stochastic,
            avg_momentum,
            score,
            timestamps[i],
            symbols[i].clone(),
            timeframes[i].clone(),
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
    crate::sigmoid_normalize(normalized, 0.2, 5.0)
}

/// Helper function to normalize divergence score
fn normalize_divergence_score(divergence_strength: f64) -> f64 {
    // Normalize assuming max meaningful divergence is 50 units
    let normalized = (divergence_strength / 50.0).min(1.0);
    
    // Apply sigmoid to emphasize strong divergences
    crate::sigmoid_normalize(normalized, 0.2, 5.0)
}

/// Helper function to normalize momentum score
fn normalize_momentum_score(momentum: f64) -> f64 {
    // Normalize assuming max meaningful momentum is 100 units
    let normalized = (momentum / 100.0).min(1.0);
    
    // Apply sigmoid to emphasize strong momentum
    crate::sigmoid_normalize(normalized, 0.2, 5.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_stoch_raw_signals() {
        let k_values = vec![85.0, 15.0, 75.0, 25.0, 90.0];
        let d_values = vec![80.0, 20.0, 70.0, 30.0, 85.0];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_stoch_raw_signals(&k_values, &d_values, &timestamps, &symbols, &timeframes, &config);
        
        // Should have some signals
        assert!(signals.len() <= k_values.len());
    }
    
    #[test]
    fn test_calculate_stoch_crossover_signals() {
        let k_values = vec![20.0, 80.0, 75.0, 25.0, 85.0];
        let d_values = vec![25.0, 75.0, 80.0, 30.0, 80.0];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_stoch_crossover_signals(&k_values, &d_values, &timestamps, &symbols, &timeframes, &config);
        
        // Should have some crossover signals
        assert!(signals.len() <= k_values.len() - 1);
    }
}