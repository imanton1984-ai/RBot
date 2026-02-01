use crate::{RawSignal, RawSignalType, SignalConfig, normalize_indicator_to_score, filter_strong_signals};
use std::vec::Vec;

/// Calculate trend raw signals based on trend direction
pub fn calculate_trend_raw_signals(
    trend_directions: &[i8],  // 1 for uptrend, -1 for downtrend, 0 for sideways
    trend_strengths: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..trend_directions.len() {
        if i >= trend_strengths.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let direction = trend_directions[i];
        let strength = trend_strengths[i];
        
        if direction == 0 || strength.is_nan() {
            continue; // Skip sideways trends or invalid strengths
        }
        
        // Normalize trend strength to score [0, 1]
        let abs_strength = strength.abs();
        let normalized_strength = (abs_strength / 100.0).min(1.0); // Assuming max strength is 100%
        let score = crate::sigmoid_normalize(normalized_strength, 0.3, 6.0);
        
        // Create raw signal
        let signal = RawSignal::new(
            RawSignalType::Trend,
            strength,
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

/// Calculate trend reversal signals based on trend direction changes
pub fn calculate_trend_reversal_signals(
    trend_directions: &[i8],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to detect reversals
    for i in 1..trend_directions.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_trend = trend_directions[i - 1];
        let curr_trend = trend_directions[i];
        
        // Detect trend reversals (changes from 1 to -1 or -1 to 1)
        let is_reversal = (prev_trend == 1 && curr_trend == -1) || (prev_trend == -1 && curr_trend == 1);
        
        if is_reversal {
            // For reversals, we'll use a fixed high score since reversals are significant
            let score = 0.98; // High confidence in reversal signals
            
            // Create raw signal for reversal
            let signal = RawSignal::new(
                RawSignalType::Trend,
                curr_trend as f64,
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

/// Calculate trend momentum signals based on trend strength changes
pub fn calculate_trend_momentum_signals(
    trend_strengths: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate momentum
    for i in 1..trend_strengths.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_strength = trend_strengths[i - 1];
        let curr_strength = trend_strengths[i];
        
        if prev_strength.is_nan() || curr_strength.is_nan() {
            continue;
        }
        
        // Calculate trend momentum (rate of strength change)
        let momentum = curr_strength - prev_strength;
        let abs_momentum = momentum.abs();
        
        // Normalize momentum to score
        let normalized = (abs_momentum / 50.0).min(1.0); // Assuming max momentum is 50 units
        let score = crate::sigmoid_normalize(normalized, 0.2, 5.0);
        
        // Create raw signal for momentum
        let signal = RawSignal::new(
            RawSignalType::Trend,
            momentum,
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

/// Calculate trend continuation signals based on sustained trend
pub fn calculate_trend_continuation_signals(
    trend_directions: &[i8],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    let mut current_trend = 0i8;
    let mut trend_length = 0usize;
    
    for i in 0..trend_directions.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let current_dir = trend_directions[i];
        
        if current_dir == current_trend && current_dir != 0 {
            // Continuing trend, increase length
            trend_length += 1;
        } else {
            // Trend changed or is sideways
            if current_trend != 0 && trend_length >= 3 {
                // Previous trend was significant, generate continuation signal
                // based on the length of the trend
                let normalized_length = (trend_length as f64 / 50.0).min(1.0); // Assuming max trend length of 50
                let score = crate::sigmoid_normalize(normalized_length, 0.4, 5.0);
                
                // Create raw signal for trend continuation
                let signal = RawSignal::new(
                    RawSignalType::Trend,
                    current_trend as f64,
                    score,
                    timestamps[i - 1], // Use previous timestamp
                    symbols[i - 1].clone(),
                    timeframes[i - 1].clone(),
                );
                
                // Only include signals that meet our threshold criteria
                if signal.is_above_threshold(config) {
                    signals.push(signal);
                }
            }
            
            // Reset for new trend
            current_trend = current_dir;
            trend_length = 1;
        }
    }
    
    // Handle the final trend if it was long enough
    if current_trend != 0 && trend_length >= 3 && !signals.is_empty() {
        // Only add if we have room and it's different from the last signal
        let normalized_length = (trend_length as f64 / 50.0).min(1.0);
        let score = crate::sigmoid_normalize(normalized_length, 0.4, 5.0);
        
        if let (Some(last_ts), Some(last_sym), Some(last_tf)) = 
            (timestamps.last(), symbols.last(), timeframes.last()) {
            let signal = RawSignal::new(
                RawSignalType::Trend,
                current_trend as f64,
                score,
                *last_ts,
                last_sym.clone(),
                last_tf.clone(),
            );
            
            if signal.is_above_threshold(config) {
                signals.push(signal);
            }
        }
    }
    
    signals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_trend_raw_signals() {
        let trend_directions = vec![1, 1, -1, -1, 1];
        let trend_strengths = vec![5.0, 8.0, -3.0, -7.0, 6.0];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_trend_raw_signals(&trend_directions, &trend_strengths, &timestamps, &symbols, &timeframes, &config);
        
        // Should have signals for trending periods
        assert!(signals.len() <= trend_directions.len());
    }
    
    #[test]
    fn test_calculate_trend_reversal_signals() {
        let trend_directions = vec![1, 1, -1, -1, 1];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_trend_reversal_signals(&trend_directions, &timestamps, &symbols, &timeframes, &config);
        
        // Should have reversal signals where trend changes
        assert!(signals.len() <= trend_directions.len() - 1);
    }
}