use crate::thresholds::{RawSignal, RawSignalType, SignalConfig, SignalKind};
use crate::scoring::sigmoid_normalize;
use common::{Symbol, Timeframe};
use std::vec::Vec;

/// Calculate VWAP raw signals based on price vs VWAP relationship
pub fn calculate_vwap_raw_signals(
    prices: &[f64],
    vwap_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
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
        let score = sigmoid_normalize(normalized, 0.3, 6.0);
        let side = if distance_pct > 0.0 { 1 } else { -1 };
        
        // Create raw signal
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::Vwap.to_indicator_id(),
            SignalKind::PriceRelation.to_i16(),
            side,
            score as f32,
            distance_pct as f32,
            None,
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
    symbols: &[Symbol],
    timeframes: &[Timeframe],
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
            let score = sigmoid_normalize(normalized, 0.2, 5.0);
            let side = if is_bullish_cross { 1 } else { -1 };
            
            // Create raw signal for crossover
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::Vwap.to_indicator_id(),
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

/// Calculate VWAP deviation signals (how far price is from VWAP)
pub fn calculate_vwap_deviation_signals(
    prices: &[f64],
    vwap_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
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
        let score = sigmoid_normalize(normalized, 0.4, 7.0); // Higher steepness for deviation
        let side = if deviation_pct > 0.0 { 1 } else { -1 };

        // Create raw signal for deviation
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::Vwap.to_indicator_id(),
            SignalKind::PriceRelation.to_i16(),
            side,
            score as f32,
            deviation_pct as f32,
            None,
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
    symbols: &[Symbol],
    timeframes: &[Timeframe],
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
        let score = sigmoid_normalize(normalized, 0.2, 5.0);
        let side = if momentum_pct > 0.0 { 1 } else { -1 };
        
        // Create raw signal for momentum
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::Vwap.to_indicator_id(),
            SignalKind::Volatility.to_i16(),
            side,
            score as f32,
            momentum_pct as f32,
            None,
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
    symbols: &[Symbol],
    timeframes: &[Timeframe],
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
            let score = sigmoid_normalize(normalized, 0.5, 8.0);
            
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::Vwap.to_indicator_id(),
                SignalKind::PriceRelation.to_i16(),
                1,
                score as f32,
                trend_strength as f32,
                None,
            );
            
            if signal.is_above_threshold(config) {
                signals.push(signal);
            }
        } else if below_count >= trend_period - 1 {
            // Strong bearish trend below VWAP
            let trend_strength = (below_count as f64 / trend_period as f64) * 100.0;
            let normalized = (trend_strength / 100.0).min(1.0);
            let score = sigmoid_normalize(normalized, 0.5, 8.0);
            
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::Vwap.to_indicator_id(),
                SignalKind::PriceRelation.to_i16(),
                -1,
                score as f32,
                -trend_strength as f32,
                None,
            );
            
            if signal.is_above_threshold(config) {
                signals.push(signal);
            }
        }
    }
    
    signals
}