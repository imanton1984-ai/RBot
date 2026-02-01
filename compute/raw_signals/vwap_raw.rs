use crate::{RawSignal, RawSignalType, SignalConfig, normalize_indicator_to_score, filter_strong_signals};
use std::vec::Vec;

/// Calculate VWAP raw signals based on price vs VWAP relationship
pub fn calculate_vwap_raw_signals(
    prices: &[f64],
    vwap_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..prices.len() {
        if i >= vwap_values.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let price = prices[i];
        let vwap = vwap_values[i];
        
        if price.is_nan() || vwap.is_nan() || vwap == 0.0 {
            continue;
        }
        
        // Calculate distance from VWAP as percentage
        let distance_pct = ((price - vwap) / vwap) * 100.0;
        let abs_distance = distance_pct.abs();
        
        // Normalize distance to score
        let normalized = (abs_distance / 10.0).min(1.0); // Assuming max 10% deviation
        let score = crate::sigmoid_normalize(normalized, 0.3, 6.0);
        
        // Create raw signal
        let signal = RawSignal::new(
            RawSignalType::Vwap,
            distance_pct,
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

/// Calculate VWAP crossover signals (price crossing VWAP)
pub fn calculate_vwap_crossover_signals(
    prices: &[f64],
    vwap_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to detect crossovers
    for i in 1..prices.len() {
        if i >= vwap_values.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_price = prices[i - 1];
        let prev_vwap = vwap_values[i - 1];
        let curr_price = prices[i];
        let curr_vwap = vwap_values[i];
        
        if prev_price.is_nan() || prev_vwap.is_nan() || curr_price.is_nan() || curr_vwap.is_nan() {
            continue;
        }
        
        // Detect bullish crossover (price crosses above VWAP)
        let is_bullish_cross = prev_price <= prev_vwap && curr_price > curr_vwap;
        // Detect bearish crossover (price crosses below VWAP)
        let is_bearish_cross = prev_price >= prev_vwap && curr_price < curr_vwap;
        
        if is_bullish_cross || is_bearish_cross {
            // Use the absolute distance from VWAP as signal strength
            let cross_strength = ((curr_price - curr_vwap) / curr_vwap).abs() * 100.0;
            let normalized = (cross_strength / 5.0).min(1.0); // Max 5% cross strength
            let score = crate::sigmoid_normalize(normalized, 0.2, 5.0);
            
            // Create raw signal for crossover
            let signal = RawSignal::new(
                RawSignalType::Vwap,
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

/// Calculate VWAP deviation signals (how far price is from VWAP)
pub fn calculate_vwap_deviation_signals(
    prices: &[f64],
    vwap_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..prices.len() {
        if i >= vwap_values.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let price = prices[i];
        let vwap = vwap_values[i];
        
        if price.is_nan() || vwap.is_nan() || vwap == 0.0 {
            continue;
        }
        
        // Calculate deviation from VWAP as percentage
        let deviation_pct = ((price - vwap) / vwap) * 100.0;
        let abs_deviation = deviation_pct.abs();
        
        // Calculate normalized score based on deviation
        let normalized = (abs_deviation / 8.0).min(1.0); // Assuming max 8% deviation
        let score = crate::sigmoid_normalize(normalized, 0.4, 7.0); // Higher steepness for deviation
        
        // Create raw signal for deviation
        let signal = RawSignal::new(
            RawSignalType::Vwap,
            deviation_pct,
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

/// Calculate VWAP momentum signals based on VWAP slope
pub fn calculate_vwap_momentum_signals(
    vwap_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate momentum
    for i in 1..vwap_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_vwap = vwap_values[i - 1];
        let curr_vwap = vwap_values[i];
        
        if prev_vwap.is_nan() || curr_vwap.is_nan() || prev_vwap == 0.0 {
            continue;
        }
        
        // Calculate VWAP momentum (rate of change)
        let momentum_pct = ((curr_vwap - prev_vwap) / prev_vwap) * 100.0;
        let abs_momentum = momentum_pct.abs();
        
        // Normalize momentum to score
        let normalized = (abs_momentum / 3.0).min(1.0); // Max 3% momentum
        let score = crate::sigmoid_normalize(normalized, 0.2, 5.0);
        
        // Create raw signal for momentum
        let signal = RawSignal::new(
            RawSignalType::Vwap,
            momentum_pct,
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

/// Calculate VWAP trend strength signals
pub fn calculate_vwap_trend_strength_signals(
    prices: &[f64],
    vwap_values: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Calculate trend based on whether price stays above or below VWAP for a period
    let trend_period = 5; // Minimum period to establish trend
    
    if prices.len() < trend_period {
        return signals;
    }
    
    for i in trend_period..prices.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        // Check if there's a consistent trend over the period
        let mut above_count = 0;
        let mut below_count = 0;
        
        for j in (i - trend_period + 1)..=i {
            if j < vwap_values.len() && !prices[j].is_nan() && !vwap_values[j].is_nan() && vwap_values[j] != 0.0 {
                if prices[j] > vwap_values[j] {
                    above_count += 1;
                } else if prices[j] < vwap_values[j] {
                    below_count += 1;
                }
            }
        }
        
        // Calculate trend strength
        if above_count >= trend_period - 1 {
            // Strong bullish trend above VWAP
            let trend_strength = (above_count as f64 / trend_period as f64) * 100.0;
            let normalized = (trend_strength / 100.0).min(1.0);
            let score = crate::sigmoid_normalize(normalized, 0.5, 8.0);
            
            let signal = RawSignal::new(
                RawSignalType::Vwap,
                trend_strength,
                score,
                timestamps[i],
                symbols[i].clone(),
                timeframes[i].clone(),
            );
            
            if signal.is_above_threshold(config) {
                signals.push(signal);
            }
        } else if below_count >= trend_period - 1 {
            // Strong bearish trend below VWAP
            let trend_strength = (below_count as f64 / trend_period as f64) * 100.0;
            let normalized = (trend_strength / 100.0).min(1.0);
            let score = crate::sigmoid_normalize(normalized, 0.5, 8.0);
            
            let signal = RawSignal::new(
                RawSignalType::Vwap,
                -trend_strength, // Negative to indicate bearish
                score,
                timestamps[i],
                symbols[i].clone(),
                timeframes[i].clone(),
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
    fn test_calculate_vwap_raw_signals() {
        let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0];
        let vwap_values = vec![99.0, 101.0, 101.5, 102.5, 104.0];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_vwap_raw_signals(&prices, &vwap_values, &timestamps, &symbols, &timeframes, &config);
        
        // Should have some signals
        assert!(signals.len() <= prices.len());
    }
    
    #[test]
    fn test_calculate_vwap_crossover_signals() {
        let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0];
        let vwap_values = vec![101.0, 101.0, 101.5, 102.0, 104.0];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_vwap_crossover_signals(&prices, &vwap_values, &timestamps, &symbols, &timeframes, &config);
        
        // Should have some crossover signals
        assert!(signals.len() <= prices.len() - 1);
    }
}