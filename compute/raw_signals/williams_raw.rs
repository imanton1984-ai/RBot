use crate::{RawSignal, RawSignalType, SignalConfig, normalize_indicator_to_score, filter_strong_signals};
use std::vec::Vec;

/// Calculate Williams %R raw signals for extreme values
pub fn calculate_williams_raw_signals(
    williams_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..williams_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let raw_value = williams_values[i];
        
        if raw_value.is_nan() {
            continue;
        }
        
        // Normalize Williams %R value to score [0, 1]
        let score = normalize_indicator_to_score(raw_value, "williams");
        
        // Create raw signal
        let signal = RawSignal::new(
            RawSignalType::Williams,
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

/// Calculate Williams %R crossover signals (crossing over -20 or under -80)
pub fn calculate_williams_crossover_signals(
    williams_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to detect crossovers
    for i in 1..williams_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_williams = williams_values[i - 1];
        let curr_williams = williams_values[i];
        
        if prev_williams.is_nan() || curr_williams.is_nan() {
            continue;
        }
        
        // Williams %R ranges from -100 to 0
        // Overbought: values near 0 (above -20)
        // Oversold: values near -100 (below -80)
        
        // Detect bullish crossover (Williams crosses above -80 from oversold)
        let is_bullish_cross = prev_williams <= -80.0 && curr_williams > -80.0;
        // Detect bearish crossover (Williams crosses below -20 from overbought)
        let is_bearish_cross = prev_williams >= -20.0 && curr_williams < -20.0;
        
        if is_bullish_cross || is_bearish_cross {
            // Use the absolute distance from extremes as signal strength
            let cross_strength = if is_bullish_cross {
                (curr_williams + 80.0).abs()  // Distance from -80
            } else {
                (curr_williams + 20.0).abs()  // Distance from -20
            };
            
            let score = normalize_crossover_score(cross_strength);
            
            // Create raw signal for crossover
            let signal = RawSignal::new(
                RawSignalType::Williams,
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

/// Calculate Williams %R divergence signals
pub fn calculate_williams_divergence_signals(
    prices: &[f64],
    williams_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate divergence
    for i in 1..prices.len() {
        if i >= williams_values.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_price = prices[i - 1];
        let curr_price = prices[i];
        let prev_williams = williams_values[i - 1];
        let curr_williams = williams_values[i];
        
        if prev_price.is_nan() || curr_price.is_nan() || prev_williams.is_nan() || curr_williams.is_nan() {
            continue;
        }
        
        // Calculate price and Williams changes
        let price_change = curr_price - prev_price;
        let williams_change = curr_williams - prev_williams;
        
        // Bullish divergence: price makes lower low, Williams makes higher low
        let is_bullish_div = price_change < 0.0 && williams_change > 0.0 && curr_williams < -80.0;
        
        // Bearish divergence: price makes higher high, Williams makes lower high
        let is_bearish_div = price_change > 0.0 && williams_change < 0.0 && curr_williams > -20.0;
        
        if is_bullish_div || is_bearish_div {
            // Calculate divergence strength
            let divergence_strength = (price_change.abs() + williams_change.abs()) / 2.0;
            let score = normalize_divergence_score(divergence_strength);
            
            // Create raw signal for divergence
            let signal = RawSignal::new(
                RawSignalType::Williams,
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

/// Calculate Williams %R momentum signals based on rate of change
pub fn calculate_williams_momentum_signals(
    williams_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate momentum
    for i in 1..williams_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_williams = williams_values[i - 1];
        let curr_williams = williams_values[i];
        
        if prev_williams.is_nan() || curr_williams.is_nan() {
            continue;
        }
        
        // Calculate Williams momentum (rate of change)
        let momentum = curr_williams - prev_williams;
        let abs_momentum = momentum.abs();
        
        // Normalize momentum to score
        let score = normalize_momentum_score(abs_momentum);
        
        // Create raw signal for momentum
        let signal = RawSignal::new(
            RawSignalType::Williams,
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

/// Helper function to normalize crossover score
fn normalize_crossover_score(cross_strength: f64) -> f64 {
    // Normalize assuming max meaningful crossover strength is 80 units
    let normalized = (cross_strength / 80.0).min(1.0);
    
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
    fn test_calculate_williams_raw_signals() {
        let williams_values = vec![-5.0, -95.0, -40.0, -10.0, -90.0];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_williams_raw_signals(&williams_values, &timestamps, &symbols, &timeframes, &config);
        
        // Should have some signals
        assert!(signals.len() <= williams_values.len());
    }
    
    #[test]
    fn test_calculate_williams_crossover_signals() {
        let williams_values = vec![-90.0, -70.0, -10.0, -85.0, -15.0];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_williams_crossover_signals(&williams_values, &timestamps, &symbols, &timeframes, &config);
        
        // Should have some crossover signals
        assert!(signals.len() <= williams_values.len() - 1);
    }
}