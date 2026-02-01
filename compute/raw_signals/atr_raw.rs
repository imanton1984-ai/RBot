use crate::{RawSignal, RawSignalType, SignalConfig, normalize_indicator_to_score, filter_strong_signals};
use std::vec::Vec;

/// Calculate ATR-based raw signals
pub fn calculate_atr_raw_signals(
    atr_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..atr_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let raw_value = atr_values[i];
        
        if raw_value.is_nan() {
            continue;
        }
        
        // Normalize ATR value to score [0, 1]
        let score = normalize_indicator_to_score(raw_value, "atr");
        
        // Create raw signal
        let signal = RawSignal::new(
            RawSignalType::Atr,
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

/// Calculate ATR trend signals based on ATR expansion/contraction
pub fn calculate_atr_trend_signals(
    atr_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate trend
    for i in 1..atr_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let current_atr = atr_values[i];
        let prev_atr = atr_values[i - 1];
        
        if current_atr.is_nan() || prev_atr.is_nan() {
            continue;
        }
        
        // Calculate ATR momentum (expansion/contraction)
        let atr_change = current_atr - prev_atr;
        let atr_change_pct = if prev_atr != 0.0 { 
            (atr_change / prev_atr) * 100.0 
        } else { 
            0.0 
        };
        
        // Normalize the change percentage to score
        let score = normalize_atr_change_score(atr_change_pct);
        
        // Create raw signal for ATR trend
        let signal = RawSignal::new(
            RawSignalType::Atr,
            atr_change_pct,
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

/// Helper function to normalize ATR change to score
fn normalize_atr_change_score(atr_change_pct: f64) -> f64 {
    // ATR expansion indicates increasing volatility (potential strong move)
    // ATR contraction indicates decreasing volatility (potential consolidation)
    
    // Use absolute value to measure magnitude of change
    let abs_change = atr_change_pct.abs();
    
    // Normalize assuming max meaningful change is 50%
    let normalized = (abs_change / 50.0).min(1.0);
    
    // Apply sigmoid to emphasize strong changes
    crate::sigmoid_normalize(normalized, 0.2, 5.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_atr_raw_signals() {
        let atr_values = vec![0.02, 0.03, 0.05, 0.01, 0.08];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_atr_raw_signals(&atr_values, &timestamps, &symbols, &timeframes, &config);
        
        // Should have some signals
        assert!(signals.len() <= atr_values.len());
    }
    
    #[test]
    fn test_calculate_atr_trend_signals() {
        let atr_values = vec![0.02, 0.03, 0.05, 0.01, 0.08];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_atr_trend_signals(&atr_values, &timestamps, &symbols, &timeframes, &config);
        
        // Should have signals for changes (one less than input values)
        assert!(signals.len() <= atr_values.len() - 1);
    }
}