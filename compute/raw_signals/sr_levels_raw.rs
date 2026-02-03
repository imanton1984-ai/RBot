use crate::thresholds::{RawSignal, RawSignalType, SignalConfig, SignalKind};
use crate::scoring::sigmoid_normalize; // Убран unused import: normalize_indicator_to_score
use common::{Symbol, Timeframe};
use std::vec::Vec;

/// Calculate SR levels raw signals for price approaching or bouncing from levels
pub fn calculate_sr_levels_raw_signals(
    prices: &[f64],
    strong_support: f64,
    mid_support: f64,
    light_support: f64,
    strong_resistance: f64,
    mid_resistance: f64,
    light_resistance: f64,
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..prices.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let current_price = prices[i];
        
        if current_price.is_nan() {
            continue;
        }
        
        // Check proximity to each level
        let dist_to_strong_support = (current_price - strong_support).abs();
        let proximity_to_strong_support = calculate_proximity_score(dist_to_strong_support, 0.01); // Assuming 1% tolerance

        // Check proximity to mid support
        let dist_to_mid_support = (current_price - mid_support).abs();
        let proximity_to_mid_support = calculate_proximity_score(dist_to_mid_support, 0.01);

        // Check proximity to light support
        let dist_to_light_support = (current_price - light_support).abs();
        let proximity_to_light_support = calculate_proximity_score(dist_to_light_support, 0.01);

        // Check proximity to strong resistance
        let dist_to_strong_resistance = (current_price - strong_resistance).abs();
        let proximity_to_strong_resistance = calculate_proximity_score(dist_to_strong_resistance, 0.01);

        // Check proximity to mid resistance
        let dist_to_mid_resistance = (current_price - mid_resistance).abs();
        let proximity_to_mid_resistance = calculate_proximity_score(dist_to_mid_resistance, 0.01);

        // Check proximity to light resistance
        let dist_to_light_resistance = (current_price - light_resistance).abs();
        let proximity_to_light_resistance = calculate_proximity_score(dist_to_light_resistance, 0.01);

        // Determine the strongest signal
        let max_proximity = proximity_to_strong_support
            .max(proximity_to_mid_support)
            .max(proximity_to_light_support)
            .max(proximity_to_strong_resistance)
            .max(proximity_to_mid_resistance)
            .max(proximity_to_light_resistance);

        if max_proximity > 0.1 { // Only consider significant proximity
            let strongest_signal = max_proximity; // ИСПРАВЛЕНИЕ: объявляем только здесь

            // Create raw signal
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::SrLevels.to_indicator_id(),
                SignalKind::PriceRelation.to_i16(),
                0,
                strongest_signal as f32,
                strongest_signal as f32,
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

/// Calculate SR levels breakout signals
pub fn calculate_sr_levels_breakout_signals(
    prices: &[f64],
    strong_support: f64,
    mid_support: f64,
    light_support: f64,
    strong_resistance: f64,
    mid_resistance: f64,
    light_resistance: f64,
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to detect breakouts
    for i in 1..prices.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_price = prices[i - 1];
        let curr_price = prices[i];
        
        if prev_price.is_nan() || curr_price.is_nan() {
            continue;
        }
        
        let mut breakout_strength = 0.0;
        let mut is_breakout = false;
        let mut side = 0;
        
        // Check for support breakouts (price moving below support)
        if prev_price >= light_support && curr_price < light_support {
            // Break below light support
            breakout_strength = ((light_support - curr_price) / light_support).abs();
            is_breakout = true;
            side = -1;
        } else if prev_price >= mid_support && curr_price < mid_support {
            // Break below mid support
            breakout_strength = ((mid_support - curr_price) / mid_support).abs();
            is_breakout = true;
            side = -1;
        } else if prev_price >= strong_support && curr_price < strong_support {
            // Break below strong support
            breakout_strength = ((strong_support - curr_price) / strong_support).abs();
            is_breakout = true;
            side = -1;
        }
        // Check for resistance breakouts (price moving above resistance)
        else if prev_price <= light_resistance && curr_price > light_resistance {
            // Break above light resistance
            breakout_strength = ((curr_price - light_resistance) / light_resistance).abs();
            is_breakout = true;
            side = 1;
        } else if prev_price <= mid_resistance && curr_price > mid_resistance {
            // Break above mid resistance
            breakout_strength = ((curr_price - mid_resistance) / mid_resistance).abs();
            is_breakout = true;
            side = 1;
        } else if prev_price <= strong_resistance && curr_price > strong_resistance {
            // Break above strong resistance
            breakout_strength = ((curr_price - strong_resistance) / strong_resistance).abs();
            is_breakout = true;
            side = 1;
        }
        
        if is_breakout {
            // Normalize breakout strength
            let normalized_strength = breakout_strength.min(1.0);
            let score = sigmoid_normalize(normalized_strength, 0.3, 6.0);
            
            // Create raw signal for breakout
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::SrLevels.to_indicator_id(),
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

/// Calculate SR levels bounce signals
pub fn calculate_sr_levels_bounce_signals(
    prices: &[f64],
    strong_support: f64,
    mid_support: f64,
    light_support: f64,
    strong_resistance: f64,
    mid_resistance: f64,
    light_resistance: f64,
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 3 values to detect bounces (approach and reverse)
    for i in 2..prices.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let two_back_price = prices[i - 2];
        let prev_price = prices[i - 1];
        let curr_price = prices[i];
        
        if two_back_price.is_nan() || prev_price.is_nan() || curr_price.is_nan() {
            continue;
        }
        
        let mut bounce_strength = 0.0;
        let mut is_bounce = false;
        let mut side = 0;
        
        // Check for bounce from support levels
        // First check if price approached support and then reversed upward
        if prev_price <= light_support && curr_price > prev_price && two_back_price > prev_price {
            // Bounce from light support
            bounce_strength = ((light_support - prev_price) / light_support).abs();
            is_bounce = true;
            side = 1;
        } else if prev_price <= mid_support && curr_price > prev_price && two_back_price > prev_price {
            // Bounce from mid support
            bounce_strength = ((mid_support - prev_price) / mid_support).abs();
            is_bounce = true;
            side = 1;
        } else if prev_price <= strong_support && curr_price > prev_price && two_back_price > prev_price {
            // Bounce from strong support
            bounce_strength = ((strong_support - prev_price) / strong_support).abs();
            is_bounce = true;
            side = 1;
        }
        // Check for bounce from resistance levels
        else if prev_price >= light_resistance && curr_price < prev_price && two_back_price < prev_price {
            // Bounce from light resistance
            bounce_strength = ((prev_price - light_resistance) / light_resistance).abs();
            is_bounce = true;
            side = -1;
        } else if prev_price >= mid_resistance && curr_price < prev_price && two_back_price < prev_price {
            // Bounce from mid resistance
            bounce_strength = ((prev_price - mid_resistance) / mid_resistance).abs();
            is_bounce = true;
            side = -1;
        } else if prev_price >= strong_resistance && curr_price < prev_price && two_back_price < prev_price {
            // Bounce from strong resistance
            bounce_strength = ((prev_price - strong_resistance) / strong_resistance).abs();
            is_bounce = true;
            side = -1;
        }
        
        if is_bounce {
            // Normalize bounce strength
            let normalized_strength = bounce_strength.min(1.0);
            let score = sigmoid_normalize(normalized_strength, 0.3, 6.0);
            
            // Create raw signal for bounce
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::SrLevels.to_indicator_id(),
                SignalKind::PriceRelation.to_i16(),
                side,
                score as f32,
                bounce_strength as f32,
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

/// Helper function to calculate proximity score based on distance to level
fn calculate_proximity_score(distance: f64, tolerance: f64) -> f64 {
    if distance <= tolerance {
        // Closer to the level means higher score
        let normalized = (1.0 - (distance / tolerance)).max(0.0).min(1.0);
        sigmoid_normalize(normalized, 0.5, 8.0) // Emphasize very close proximity
    } else {
        0.0
    }
}