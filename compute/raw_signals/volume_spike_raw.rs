use crate::thresholds::{RawSignal, RawSignalType, SignalConfig, SignalKind};
use crate::scoring::sigmoid_normalize;
use common::{Symbol, Timeframe};
use std::vec::Vec;

/// Calculate volume spike raw signals based on volume ratios
pub fn calculate_volume_spike_raw_signals(
    volume_ratios: &[f64], // Changed from &[bool] to &[f64]
    volumes: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    // Threshold for considering it a "spike" signal (e.g., 2.0x average)
    let spike_threshold = 2.0;

    for i in 0..volume_ratios.len() {
        if i >= volumes.len() || i >= timestamps.len() || 
           i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let ratio = volume_ratios[i];
        
        if ratio > spike_threshold {
            let _raw_volume = volumes[i];
            
            // Normalize score: 
            // 2.0x -> ~0.5 score
            // 5.0x -> ~0.9 score
            // We shift input by threshold so 2.0 maps to 0 in sigmoid center logic
            let score = sigmoid_normalize(ratio - spike_threshold, 1.0, 1.0);

            // Create raw signal
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::VolumeSpike.to_indicator_id(),
                SignalKind::PriceRelation.to_i16(),
                0, // No specific direction
                score as f32,
                ratio as f32, // Store the ratio as the value, not raw volume
                None,
            );

            if signal.is_above_threshold(config) {
                signals.push(signal);
            }
        }
    }
    
    signals
}

/// Calculate volume momentum signals based on rate of change
pub fn calculate_volume_momentum_signals(
    volumes: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate momentum
    for i in 1..volumes.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_volume = volumes[i - 1];
        let curr_volume = volumes[i];
        
        if prev_volume.is_nan() || curr_volume.is_nan() {
            continue;
        }
        
        // Calculate volume momentum (rate of change)
        let volume_change = curr_volume - prev_volume;
        let volume_change_pct = if prev_volume > 0.0 {
            (volume_change / prev_volume) * 100.0
        } else {
            0.0
        };
        
        let abs_change_pct = volume_change_pct.abs();
        
        // Normalize momentum to score
        let normalized = (abs_change_pct / 200.0).min(1.0); // Assuming max 200% change
        let score = sigmoid_normalize(normalized, 0.2, 5.0);
        let side = if volume_change > 0.0 { 1 } else { -1 };
        
        // Create raw signal for momentum
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::VolumeSpike.to_indicator_id(),
            SignalKind::Volatility.to_i16(),
            side,
            score as f32,
            volume_change_pct as f32,
            None,
        );
        
        // Only include signals that meet our threshold criteria
        if signal.is_above_threshold(config) {
            signals.push(signal);
        }
    }
    
    signals
}

/// Calculate volume trend signals based on volume moving averages
pub fn calculate_volume_trend_signals(
    volumes: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Calculate short and long-term volume moving averages
    let short_period = 5;
    let long_period = 20;
    
    if volumes.len() < long_period {
        return signals;
    }
    
    // Calculate volume moving averages
    let mut short_ma = vec![0.0; volumes.len()];
    let mut long_ma = vec![0.0; volumes.len()];
    
    // Calculate initial sums
    let mut short_sum = 0.0;
    let mut long_sum = 0.0;
    
    for i in 0..long_period {
        if i < short_period {
            short_sum += volumes[i];
        }
        long_sum += volumes[i];
        
        if i < short_period {
            short_ma[i] = f64::NAN;
        } else {
            short_ma[i] = short_sum / short_period as f64;
        }
        
        if i < long_period - 1 {
            long_ma[i] = f64::NAN;
        } else {
            long_ma[i] = long_sum / long_period as f64;
        }
    }
    
    // Calculate remaining values
    for i in long_period..volumes.len() {
        // Update short MA
        short_sum = short_sum - volumes[i - short_period] + volumes[i];
        short_ma[i] = short_sum / short_period as f64;
        
        // Update long MA
        long_sum = long_sum - volumes[i - long_period] + volumes[i];
        long_ma[i] = long_sum / long_period as f64;
    }
    
    // Generate signals based on volume MA crossovers
    for i in long_period..volumes.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let short_avg = short_ma[i];
        let long_avg = long_ma[i];
        
        if short_avg.is_nan() || long_avg.is_nan() {
            continue;
        }
        
        // Bullish signal: short MA crosses above long MA (increasing volume trend)
        let is_bullish = short_avg > long_avg && short_avg > volumes[i] * 0.5; // Ensure volume is significant
        // Bearish signal: short MA crosses below long MA (decreasing volume trend)
        let is_bearish = short_avg < long_avg && short_avg > volumes[i] * 0.3; // Ensure some volume exists
        
        if is_bullish || is_bearish {
            // Calculate trend strength based on the difference between MAs
            let trend_strength = ((short_avg - long_avg) / long_avg).abs();
            let normalized = (trend_strength * 100.0).min(1.0); // Scale appropriately
            let score = sigmoid_normalize(normalized, 0.2, 5.0);
            let side = if is_bullish { 1 } else { -1 };
            
            // Create raw signal for volume trend
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::VolumeSpike.to_indicator_id(),
                SignalKind::Crossover.to_i16(),
                side,
                score as f32,
                trend_strength as f32,
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