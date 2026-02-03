use crate::thresholds::{RawSignal, RawSignalType, SignalConfig, SignalKind};
use crate::scoring::{normalize_indicator_to_score, sigmoid_normalize};
use common::{Symbol, Timeframe};
use std::vec::Vec;

/// Calculate CCI raw signals for extreme values
pub fn calculate_cci_raw_signals(
    cci_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..cci_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let raw_value = cci_values[i];
        
        if raw_value.is_nan() {
            continue;
        }
        
        // Normalize CCI value to score [0, 1]
        let score = normalize_indicator_to_score(raw_value, "cci");
        let side = if raw_value > 100.0 { -1 } else if raw_value < -100.0 { 1 } else { 0 };

        // Create raw signal
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::Cci.to_indicator_id(),
            SignalKind::PriceRelation.to_i16(),
            side,
            score as f32,
            raw_value as f32,
            None,
        );
        
        // Only include signals that meet our threshold criteria
        if signal.is_above_threshold(config) {
            signals.push(signal);
        }
    }
    
    signals
}

/// Calculate CCI crossover signals (crossing over +100 or under -100)
pub fn calculate_cci_crossover_signals(
    cci_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to detect crossovers
    for i in 1..cci_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_cci = cci_values[i - 1];
        let curr_cci = cci_values[i];
        
        if prev_cci.is_nan() || curr_cci.is_nan() {
            continue;
        }
        
        // Detect bullish crossover (CCI crosses above +100)
        let is_bullish_cross = prev_cci <= 100.0 && curr_cci > 100.0;
        // Detect bearish crossover (CCI crosses below -100)
        let is_bearish_cross = prev_cci >= -100.0 && curr_cci < -100.0;
        
        if is_bullish_cross || is_bearish_cross {
            // Use the absolute distance from extremes as signal strength
            let cross_strength = if is_bullish_cross {
                (curr_cci - 100.0).abs()
            } else {
                (curr_cci + 100.0).abs()
            };
            
            let score = normalize_crossover_score(cross_strength);
            let side = if is_bullish_cross { 1 } else { -1 };
            
            // Create raw signal for crossover
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::Cci.to_indicator_id(),
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

/// Calculate CCI divergence signals
pub fn calculate_cci_divergence_signals(
    prices: &[f64],
    cci_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate divergence
    for i in 1..prices.len() {
        if i >= cci_values.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_price = prices[i - 1];
        let curr_price = prices[i];
        let prev_cci = cci_values[i - 1];
        let curr_cci = cci_values[i];
        
        if prev_price.is_nan() || curr_price.is_nan() || prev_cci.is_nan() || curr_cci.is_nan() {
            continue;
        }
        
        // Calculate price and CCI changes
        let price_change = curr_price - prev_price;
        let cci_change = curr_cci - prev_cci;
        
        // Bullish divergence: price makes lower low, CCI makes higher low
        let is_bullish_div = price_change < 0.0 && cci_change > 0.0 && curr_cci < -100.0;
        
        // Bearish divergence: price makes higher high, CCI makes lower high
        let is_bearish_div = price_change > 0.0 && cci_change < 0.0 && curr_cci > 100.0;
        
        if is_bullish_div || is_bearish_div {
            // Calculate divergence strength
            let divergence_strength = (price_change.abs() + cci_change.abs()) / 2.0;
            let score = normalize_divergence_score(divergence_strength);
            let side = if is_bullish_div { 1 } else { -1 };

            // Create raw signal for divergence
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::Cci.to_indicator_id(),
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

/// Calculate CCI momentum signals based on rate of change
pub fn calculate_cci_momentum_signals(
    cci_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate momentum
    for i in 1..cci_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_cci = cci_values[i - 1];
        let curr_cci = cci_values[i];
        
        if prev_cci.is_nan() || curr_cci.is_nan() {
            continue;
        }
        
        // Calculate CCI momentum (rate of change)
        let momentum = curr_cci - prev_cci;
        let abs_momentum = momentum.abs();
        
        // Normalize momentum to score
        let score = normalize_momentum_score(abs_momentum);
        let side = if momentum > 0.0 { 1 } else { -1 };
        
        // Create raw signal for momentum
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::Cci.to_indicator_id(),
            SignalKind::Volatility.to_i16(),
            side,
            score as f32,
            momentum as f32,
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
    // Normalize assuming max meaningful crossover strength is 100 units
    let normalized = (cross_strength / 100.0).min(1.0);
    
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