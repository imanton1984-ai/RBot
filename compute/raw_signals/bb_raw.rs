use crate::thresholds::{RawSignal, RawSignalType, SignalConfig, SignalKind};
use crate::scoring::sigmoid_normalize;
use common::{Symbol, Timeframe};
use std::vec::Vec;

/// Calculate Bollinger Bands raw signals for price touching bands
pub fn calculate_bb_raw_signals(
    prices: &[f64],
    upper_band: &[f64],
    middle_band: &[f64],
    lower_band: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..prices.len() {
        if i >= upper_band.len() || i >= middle_band.len() || i >= lower_band.len() ||
           i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let price = prices[i];
        let upper = upper_band[i];
        let middle = middle_band[i];
        let lower = lower_band[i];
        
        if price.is_nan() || upper.is_nan() || middle.is_nan() || lower.is_nan() {
            continue;
        }
        
        // Check if price touches upper or lower band
        let mut signal_strength = 0.0;
        let mut is_valid_signal = false;
        let mut side = 0;
        
        // Calculate distance from bands as percentage of band width
        let band_width = upper - lower;
        if band_width > 0.0 {
            if price >= upper * 0.99 { // Touching upper band (overbought)
                signal_strength = ((price - middle) / band_width).min(1.0);
                is_valid_signal = true;
                side = -1; // Short signal
            } else if price <= lower * 1.01 { // Touching lower band (oversold)
                signal_strength = ((middle - price) / band_width).min(1.0);
                is_valid_signal = true;
                side = 1; // Long signal
            }
        }
        
        if is_valid_signal {
            // Normalize the signal strength to score
            let score = sigmoid_normalize(signal_strength, 0.3, 6.0);
            
            // Create raw signal
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::BollingerBands.to_indicator_id(),
                SignalKind::PriceRelation.to_i16(),
                side,
                score as f32,
                signal_strength as f32,
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

/// Calculate Bollinger Bands squeeze signals (when bands contract)
pub fn calculate_bb_squeeze_signals(
    upper_band: &[f64],
    lower_band: &[f64],
    atr_values: &[f64], // Using ATR as reference for normal volatility
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..upper_band.len() {
        if i >= lower_band.len() || i >= atr_values.len() ||
           i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let upper = upper_band[i];
        let lower = lower_band[i];
        let atr = atr_values[i];
        
        if upper.is_nan() || lower.is_nan() || atr.is_nan() || atr <= 0.0 {
            continue;
        }
        
        // Calculate band width
        let band_width = upper - lower;
        
        // Calculate squeeze ratio (band width relative to ATR)
        let squeeze_ratio = if atr > 0.0 { band_width / atr } else { 0.0 };
        
        // A squeeze occurs when bands are narrow relative to normal volatility
        // We invert the ratio so that smaller ratios (more squeeze) get higher scores
        let squeeze_strength = if squeeze_ratio < 1.0 { 
            (1.0 - squeeze_ratio).min(1.0) 
        } else { 
            0.0 
        };
        
        if squeeze_strength > 0.0 {
            // Normalize the squeeze strength to score
            let score = sigmoid_normalize(squeeze_strength, 0.2, 5.0);
            
            // Create raw signal for squeeze
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::BollingerBands.to_indicator_id(),
                SignalKind::Volatility.to_i16(),
                0,
                score as f32,
                squeeze_ratio as f32,
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

/// Calculate Bollinger Bands breakout signals (price moving away from bands)
pub fn calculate_bb_breakout_signals(
    prices: &[f64],
    upper_band: &[f64],
    lower_band: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to detect breakouts
    for i in 1..prices.len() {
        if i >= upper_band.len() || i >= lower_band.len() ||
           i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_price = prices[i - 1];
        let curr_price = prices[i];
        let upper = upper_band[i];
        let lower = lower_band[i];
        
        if prev_price.is_nan() || curr_price.is_nan() || upper.is_nan() || lower.is_nan() {
            continue;
        }
        
        // Detect if there was a breakout from the bands
        let prev_above_upper = prev_price > upper;
        let prev_below_lower = prev_price < lower;
        let curr_above_upper = curr_price > upper;
        let curr_below_lower = curr_price < lower;
        
        let mut breakout_strength = 0.0;
        let mut is_breakout = false;
        let mut side = 0;
        
        // Bullish breakout: price moves above upper band
        if (!prev_above_upper && curr_above_upper) || (prev_below_lower && !curr_below_lower && curr_price > upper * 0.95) {
            breakout_strength = ((curr_price - upper) / (upper - lower)).abs().min(1.0);
            is_breakout = true;
            side = 1;
        }
        // Bearish breakout: price moves below lower band
        else if (!prev_below_lower && curr_below_lower) || (prev_above_upper && !curr_above_upper && curr_price < lower * 1.05) {
            breakout_strength = ((lower - curr_price) / (upper - lower)).abs().min(1.0);
            is_breakout = true;
            side = -1;
        }
        
        if is_breakout {
            // Normalize the breakout strength to score
            let score = sigmoid_normalize(breakout_strength, 0.3, 6.0);
            
            // Create raw signal for breakout
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::BollingerBands.to_indicator_id(),
                SignalKind::Breakout.to_i16(),
                side,
                score as f32,
                breakout_strength as f32,
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