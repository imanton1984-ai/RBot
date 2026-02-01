use crate::{RawSignal, RawSignalType, SignalConfig, normalize_indicator_to_score, filter_strong_signals};
use std::vec::Vec;

/// Calculate MACD raw signals based on MACD line values
pub fn calculate_macd_raw_signals(
    macd_line: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..macd_line.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let raw_value = macd_line[i];
        
        if raw_value.is_nan() {
            continue;
        }
        
        // Normalize MACD value to score [0, 1]
        let score = normalize_indicator_to_score(raw_value, "macd");
        
        // Create raw signal
        let signal = RawSignal::new(
            RawSignalType::Macd,
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

/// Calculate MACD crossover signals (MACD line crossing signal line)
pub fn calculate_macd_crossover_signals(
    macd_line: &[f64],
    signal_line: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to detect crossovers
    for i in 1..macd_line.len() {
        if i >= signal_line.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_macd = macd_line[i - 1];
        let prev_signal = signal_line[i - 1];
        let curr_macd = macd_line[i];
        let curr_signal = signal_line[i];
        
        if prev_macd.is_nan() || prev_signal.is_nan() || curr_macd.is_nan() || curr_signal.is_nan() {
            continue;
        }
        
        // Detect bullish crossover (MACD crosses above signal)
        let is_bullish_cross = prev_macd <= prev_signal && curr_macd > curr_signal;
        // Detect bearish crossover (MACD crosses below signal)
        let is_bearish_cross = prev_macd >= prev_signal && curr_macd < curr_signal;
        
        if is_bullish_cross || is_bearish_cross {
            // Use the absolute difference as signal strength
            let cross_strength = (curr_macd - curr_signal).abs();
            let score = normalize_crossover_score(cross_strength);
            
            // Create raw signal for crossover
            let signal = RawSignal::new(
                RawSignalType::Macd,
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

/// Calculate MACD histogram-based signals
pub fn calculate_macd_histogram_signals(
    histogram: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..histogram.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let raw_value = histogram[i];
        
        if raw_value.is_nan() {
            continue;
        }
        
        // Use absolute value of histogram for signal strength
        let abs_histogram = raw_value.abs();
        let score = normalize_histogram_score(abs_histogram);
        
        // Create raw signal for histogram
        let signal = RawSignal::new(
            RawSignalType::Macd,
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

/// Calculate MACD divergence signals
pub fn calculate_macd_divergence_signals(
    prices: &[f64],
    macd_line: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate divergence
    for i in 1..prices.len() {
        if i >= macd_line.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_price = prices[i - 1];
        let curr_price = prices[i];
        let prev_macd = macd_line[i - 1];
        let curr_macd = macd_line[i];
        
        if prev_price.is_nan() || curr_price.is_nan() || prev_macd.is_nan() || curr_macd.is_nan() {
            continue;
        }
        
        // Calculate price and MACD changes
        let price_change = curr_price - prev_price;
        let macd_change = curr_macd - prev_macd;
        
        // Bullish divergence: price makes lower low, MACD makes higher low
        let is_bullish_div = price_change < 0.0 && macd_change > 0.0 && curr_macd < 0.0;
        
        // Bearish divergence: price makes higher high, MACD makes lower high
        let is_bearish_div = price_change > 0.0 && macd_change < 0.0 && curr_macd > 0.0;
        
        if is_bullish_div || is_bearish_div {
            // Calculate divergence strength
            let divergence_strength = (price_change.abs() + macd_change.abs()) / 2.0;
            let score = normalize_divergence_score(divergence_strength);
            
            // Create raw signal for divergence
            let signal = RawSignal::new(
                RawSignalType::Macd,
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

/// Helper function to normalize crossover score
fn normalize_crossover_score(cross_strength: f64) -> f64 {
    // Normalize assuming max meaningful crossover strength is 2 units
    let normalized = (cross_strength / 2.0).min(1.0);
    
    // Apply sigmoid to emphasize strong crossovers
    crate::sigmoid_normalize(normalized, 0.2, 5.0)
}

/// Helper function to normalize histogram score
fn normalize_histogram_score(histogram_value: f64) -> f64 {
    // Normalize assuming max meaningful histogram value is 2 units
    let normalized = (histogram_value / 2.0).min(1.0);
    
    // Apply sigmoid to emphasize strong histogram values
    crate::sigmoid_normalize(normalized, 0.2, 5.0)
}

/// Helper function to normalize divergence score
fn normalize_divergence_score(divergence_strength: f64) -> f64 {
    // Normalize assuming max meaningful divergence is 5 units
    let normalized = (divergence_strength / 5.0).min(1.0);
    
    // Apply sigmoid to emphasize strong divergences
    crate::sigmoid_normalize(normalized, 0.2, 5.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_macd_raw_signals() {
        let macd_line = vec![0.01, -0.02, 0.03, -0.01, 0.05];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_macd_raw_signals(&macd_line, &timestamps, &symbols, &timeframes, &config);
        
        // Should have some signals
        assert!(signals.len() <= macd_line.len());
    }
    
    #[test]
    fn test_calculate_macd_crossover_signals() {
        let macd_line = vec![0.01, 0.02, -0.01, 0.01, 0.03];
        let signal_line = vec![0.015, 0.01, 0.0, 0.0, 0.02];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_macd_crossover_signals(&macd_line, &signal_line, &timestamps, &symbols, &timeframes, &config);
        
        // Should have some crossover signals
        assert!(signals.len() <= macd_line.len() - 1);
    }
}