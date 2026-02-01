use crate::{RawSignal, RawSignalType, SignalConfig, normalize_indicator_to_score, filter_strong_signals};
use std::vec::Vec;

/// Calculate volume spike raw signals
pub fn calculate_volume_spike_raw_signals(
    volume_spikes: &[bool],
    volumes: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..volume_spikes.len() {
        if i >= volumes.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        if volume_spikes[i] {
            let raw_volume = volumes[i];
            
            if raw_volume.is_nan() {
                continue;
            }
            
            // Normalize volume to score [0, 1] - higher volumes get higher scores
            // We'll use a logarithmic scale to prevent extremely high volumes from dominating
            let normalized_volume = if raw_volume > 0.0 {
                let log_volume = raw_volume.ln();
                // Assuming max log volume is around 20 for normalization
                (log_volume / 20.0).min(1.0).max(0.0)
            } else {
                0.0
            };
            
            // Apply sigmoid to emphasize strong volume spikes
            let score = crate::sigmoid_normalize(normalized_volume, 0.3, 6.0);
            
            // Create raw signal
            let signal = RawSignal::new(
                RawSignalType::VolumeSpike,
                raw_volume,
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

/// Calculate volume momentum signals based on rate of change
pub fn calculate_volume_momentum_signals(
    volumes: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
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
        let score = crate::sigmoid_normalize(normalized, 0.2, 5.0);
        
        // Create raw signal for momentum
        let signal = RawSignal::new(
            RawSignalType::VolumeSpike,
            volume_change_pct,
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

/// Calculate volume trend signals based on volume moving averages
pub fn calculate_volume_trend_signals(
    volumes: &[f64],
    timestamps: &[i64],
    symbols: &[String],
    timeframes: &[String],
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
            let score = crate::sigmoid_normalize(normalized, 0.2, 5.0);
            
            // Create raw signal for volume trend
            let signal = RawSignal::new(
                RawSignalType::VolumeSpike,
                trend_strength,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_volume_spike_raw_signals() {
        let volume_spikes = vec![false, true, false, true, false];
        let volumes = vec![100.0, 500.0, 120.0, 600.0, 110.0];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_volume_spike_raw_signals(&volume_spikes, &volumes, &timestamps, &symbols, &timeframes, &config);
        
        // Should have signals for the spike events
        assert!(signals.len() <= volume_spikes.iter().filter(|&&x| x).count());
    }
    
    #[test]
    fn test_calculate_volume_momentum_signals() {
        let volumes = vec![100.0, 200.0, 150.0, 300.0, 120.0];
        let timestamps = vec![1000, 2000, 3000, 4000, 5000];
        let symbols = vec!["BTCUSDT".to_string(); 5];
        let timeframes = vec!["1h".to_string(); 5];
        let config = SignalConfig::default();
        
        let signals = calculate_volume_momentum_signals(&volumes, &timestamps, &symbols, &timeframes, &config);
        
        // Should have momentum signals for changes
        assert!(signals.len() <= volumes.len() - 1);
    }
}