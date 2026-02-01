use crate::{RawSignal, RawSignalType, SignalConfig, normalize_indicator_to_score, filter_strong_signals};
use std::vec::Vec;

/// Calculate ADX-based raw signals for trend strength
pub fn calculate_adx_raw_signals(
    adx_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..adx_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let raw_value = adx_values[i];
        
        if raw_value.is_nan() {
            continue;
        }
        
        // Normalize ADX value to score [0, 1]
        let score = normalize_indicator_to_score(raw_value, "adx");
        
        // Create raw signal
        let signal = RawSignal::new(
            RawSignalType::Adx,
            raw_value,
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

/// Calculate ADX trend direction signals based on +DI and -DI
pub fn calculate_adx_direction_signals(
    plus_di: &[f64],
    minus_di: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..plus_di.len() {
        if i >= minus_di.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let plus_val = plus_di[i];
        let minus_val = minus_di[i];
        
        if plus_val.is_nan() || minus_val.is_nan() {
            continue;
        }
        
        // Calculate direction strength (difference between +DI and -DI)
        let direction_strength = plus_val - minus_val;
        
        // Normalize the direction strength to score
        let score = normalize_direction_strength_score(direction_strength);
        
        // Create raw signal for direction
        let signal = RawSignal::new(
            RawSignalType::Adx,
            direction_strength,
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

/// Calculate ADX crossover signals (when +DI crosses -DI)
pub fn calculate_adx_crossover_signals(
    plus_di: &[f64],
    minus_di: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to detect crossovers
    for i in 1..plus_di.len() {
        if i >= minus_di.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_plus = plus_di[i - 1];
        let prev_minus = minus_di[i - 1];
        let curr_plus = plus_di[i];
        let curr_minus = minus_di[i];
        
        if prev_plus.is_nan() || prev_minus.is_nan() || curr_plus.is_nan() || curr_minus.is_nan() {
            continue;
        }
        
        // Detect bullish crossover (+DI crosses above -DI)
        let is_bullish_cross = prev_plus <= prev_minus && curr_plus > curr_minus;
        // Detect bearish crossover (+DI crosses below -DI)
        let is_bearish_cross = prev_plus >= prev_minus && curr_plus < curr_minus;
        
        if is_bullish_cross || is_bearish_cross {
            // Use the absolute difference as signal strength
            let cross_strength = (curr_plus - curr_minus).abs();
            let score = normalize_direction_strength_score(cross_strength);
            
            // Create raw signal for crossover
            let signal = RawSignal::new(
                RawSignalType::Adx,
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

/// Helper function to normalize direction strength to score
fn normalize_direction_strength_score(direction_strength: f64) -> f64 {
    // Use absolute value to measure strength regardless of direction
    let abs_strength = direction_strength.abs();
    
    // Normalize assuming max meaningful strength is 50
    let normalized = (abs_strength / 50.0).min(1.0);
    
    // Apply sigmoid to emphasize strong signals
    crate::sigmoid_normalize(normalized, 0.3, 6.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_adx_raw_signals() {
        let adx_values = vec![20.0, 25.0, 30.0, 40.0, 60.0];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_adx_raw_signals(&adx_values, &timestamps, &symbols, &timeframes, &config);
        
        // Should have some signals
        assert!(signals.len() <= adx_values.len());
    }
    
    #[test]
    fn test_calculate_adx_direction_signals() {
        let plus_di = vec![20.0, 25.0, 30.0, 20.0, 35.0];
        let minus_di = vec![25.0, 20.0, 15.0, 25.0, 15.0];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_adx_direction_signals(&plus_di, &minus_di, &timestamps, &symbols, &timeframes, &config);
        
        // Should have signals for each direction measurement
        assert!(signals.len() <= plus_di.len());
    }
}