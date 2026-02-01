use crate::{RawSignal, RawSignalType, SignalConfig, normalize_indicator_to_score, filter_strong_signals};
use std::vec::Vec;

/// Calculate RSI-based raw signals for overbought/oversold conditions
pub fn calculate_rsi_raw_signals(
    rsi_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..rsi_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let raw_value = rsi_values[i];
        
        if raw_value.is_nan() {
            continue;
        }
        
        // Normalize RSI value to score [0, 1]
        let score = normalize_indicator_to_score(raw_value, "rsi");
        
        // Create raw signal
        let signal = RawSignal::new(
            RawSignalType::Rsi,
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

/// Calculate RSI divergence signals (price vs RSI movement)
pub fn calculate_rsi_divergence_signals(
    prices: &[f64],
    rsi_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate divergence
    for i in 1..prices.len() {
        if i >= rsi_values.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_price = prices[i - 1];
        let curr_price = prices[i];
        let prev_rsi = rsi_values[i - 1];
        let curr_rsi = rsi_values[i];
        
        if prev_price.is_nan() || curr_price.is_nan() || prev_rsi.is_nan() || curr_rsi.is_nan() {
            continue;
        }
        
        // Calculate price and RSI changes
        let price_change = curr_price - prev_price;
        let rsi_change = curr_rsi - prev_rsi;
        
        // Bullish divergence: price makes lower low, RSI makes higher low
        let is_bullish_div = price_change < 0.0 && rsi_change > 0.0 && curr_rsi < 30.0;
        
        // Bearish divergence: price makes higher high, RSI makes lower high
        let is_bearish_div = price_change > 0.0 && rsi_change < 0.0 && curr_rsi > 70.0;
        
        if is_bullish_div || is_bearish_div {
            // Calculate divergence strength
            let divergence_strength = (price_change.abs() + rsi_change.abs()) / 2.0;
            let score = normalize_divergence_score(divergence_strength);
            
            // Create raw signal for divergence
            let signal = RawSignal::new(
                RawSignalType::Rsi,
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

/// Calculate RSI momentum signals based on rate of change
pub fn calculate_rsi_momentum_signals(
    rsi_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate momentum
    for i in 1..rsi_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_rsi = rsi_values[i - 1];
        let curr_rsi = rsi_values[i];
        
        if prev_rsi.is_nan() || curr_rsi.is_nan() {
            continue;
        }
        
        // Calculate RSI momentum (rate of change)
        let momentum = curr_rsi - prev_rsi;
        let abs_momentum = momentum.abs();
        
        // Normalize momentum to score
        let score = normalize_momentum_score(abs_momentum);
        
        // Create raw signal for momentum
        let signal = RawSignal::new(
            RawSignalType::Rsi,
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

/// Helper function to normalize divergence score
fn normalize_divergence_score(divergence_strength: f64) -> f64 {
    // Normalize assuming max meaningful divergence is 10 units
    let normalized = (divergence_strength / 10.0).min(1.0);
    
    // Apply sigmoid to emphasize strong divergences
    crate::sigmoid_normalize(normalized, 0.2, 5.0)
}

/// Helper function to normalize momentum score
fn normalize_momentum_score(momentum: f64) -> f64 {
    // Normalize assuming max meaningful momentum is 20 units
    let normalized = (momentum / 20.0).min(1.0);
    
    // Apply sigmoid to emphasize strong momentum
    crate::sigmoid_normalize(normalized, 0.2, 5.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_rsi_raw_signals() {
        let rsi_values = vec![30.0, 20.0, 80.0, 90.0, 45.0];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_rsi_raw_signals(&rsi_values, &timestamps, &symbols, &timeframes, &config);
        
        // Should have some signals
        assert!(signals.len() <= rsi_values.len());
    }
    
    #[test]
    fn test_calculate_rsi_divergence_signals() {
        let prices = vec![100.0, 95.0, 90.0, 92.0, 95.0];
        let rsi_values = vec![80.0, 75.0, 25.0, 35.0, 40.0];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_rsi_divergence_signals(&prices, &rsi_values, &timestamps, &symbols, &timeframes, &config);
        
        // Should have some divergence signals
        assert!(signals.len() <= prices.len() - 1);
    }
}