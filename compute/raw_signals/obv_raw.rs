use crate::thresholds::{RawSignal, RawSignalType, SignalConfig, SignalKind};
use crate::scoring::sigmoid_normalize;
use common::{Symbol, Timeframe};
use std::vec::Vec;

/// Calculate OBV raw signals based on OBV values
pub fn calculate_obv_raw_signals(
    obv_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..obv_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let raw_value = obv_values[i];
        
        if raw_value.is_nan() {
            continue;
        }
        
        // Calculate normalized score based on the magnitude of OBV
        // Since OBV can be very large, we'll use a logarithmic approach
        let abs_obv = raw_value.abs();
        let normalized = if abs_obv > 0.0 {
            let log_obv = abs_obv.ln();
            // Assuming max log value of 20 for normalization
            (log_obv / 20.0).min(1.0).max(0.0)
        } else {
            0.0
        };
        
        let score = sigmoid_normalize(normalized, 0.3, 6.0);
        let side = if raw_value > 0.0 { 1 } else { -1 };
        
        // Create raw signal
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::Obv.to_indicator_id(),
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

/// Calculate OBV momentum signals based on rate of change
pub fn calculate_obv_momentum_signals(
    obv_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate momentum
    for i in 1..obv_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_obv = obv_values[i - 1];
        let curr_obv = obv_values[i];
        
        if prev_obv.is_nan() || curr_obv.is_nan() {
            continue;
        }
        
        // Calculate OBV momentum (rate of change)
        let momentum = curr_obv - prev_obv;
        let abs_momentum = momentum.abs();
        
        // Normalize momentum to score using logarithmic scale
        let normalized = if abs_momentum > 0.0 {
            let log_momentum = abs_momentum.ln();
            // Assuming max log momentum of 15 for normalization
            (log_momentum / 15.0).min(1.0).max(0.0)
        } else {
            0.0
        };
        
        let score = sigmoid_normalize(normalized, 0.2, 5.0);
        let side = if momentum > 0.0 { 1 } else { -1 };
        
        // Create raw signal for momentum
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::Obv.to_indicator_id(),
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

/// Calculate OBV trend signals based on OBV moving averages
pub fn calculate_obv_trend_signals(
    obv_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Calculate short and long-term OBV moving averages
    let short_period = 5;
    let long_period = 15;
    
    if obv_values.len() < long_period {
        return signals;
    }
    
    // Calculate OBV moving averages
    let mut short_ma = vec![0.0; obv_values.len()];
    let mut long_ma = vec![0.0; obv_values.len()];
    
    // Calculate initial sums
    let mut short_sum = 0.0;
    let mut long_sum = 0.0;
    
    for i in 0..long_period {
        if i < short_period {
            short_sum += obv_values[i];
        }
        long_sum += obv_values[i];
        
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
    for i in long_period..obv_values.len() {
        // Update short MA
        short_sum = short_sum - obv_values[i - short_period] + obv_values[i];
        short_ma[i] = short_sum / short_period as f64;
        
        // Update long MA
        long_sum = long_sum - obv_values[i - long_period] + obv_values[i];
        long_ma[i] = long_sum / long_period as f64;
    }
    
    // Generate signals based on OBV MA crossovers
    for i in long_period..obv_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let short_avg = short_ma[i];
        let long_avg = long_ma[i];
        
        if short_avg.is_nan() || long_avg.is_nan() {
            continue;
        }
        
        // Bullish signal: short MA crosses above long MA (increasing OBV trend)
        let is_bullish = short_avg > long_avg;
        // Bearish signal: short MA crosses below long MA (decreasing OBV trend)
        let is_bearish = short_avg < long_avg;
        
        if is_bullish || is_bearish {
            // Calculate trend strength based on the difference between MAs
            let trend_strength = ((short_avg - long_avg) / long_avg.abs()).abs();
            let normalized = (trend_strength * 100.0).min(1.0); // Scale appropriately
            let score = sigmoid_normalize(normalized, 0.2, 5.0);
            let side = if is_bullish { 1 } else { -1 };
            
            // Create raw signal for OBV trend
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::Obv.to_indicator_id(),
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

/// Calculate OBV divergence signals (when price and OBV move in opposite directions)
pub fn calculate_obv_divergence_signals(
    prices: &[f64],
    obv_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    // Need at least 2 values to calculate divergence
    for i in 1..prices.len() {
        if i >= obv_values.len() || i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() {
            break;
        }
        
        let prev_price = prices[i - 1];
        let curr_price = prices[i];
        let prev_obv = obv_values[i - 1];
        let curr_obv = obv_values[i];
        
        if prev_price.is_nan() || curr_price.is_nan() || prev_obv.is_nan() || curr_obv.is_nan() {
            continue;
        }
        
        // Calculate price and OBV changes
        let price_change = curr_price - prev_price;
        let obv_change = curr_obv - prev_obv;
        
        // Bullish divergence: price makes lower low, OBV makes higher low
        let is_bullish_div = price_change < 0.0 && obv_change > 0.0;
        
        // Bearish divergence: price makes higher high, OBV makes lower high
        let is_bearish_div = price_change > 0.0 && obv_change < 0.0;
        
        if is_bullish_div || is_bearish_div {
            // Calculate divergence strength
            let divergence_strength = (price_change.abs() + obv_change.abs()) / 2.0;
            let normalized = if divergence_strength > 0.0 {
                let log_divergence = divergence_strength.ln();
                // Assuming max log divergence of 10 for normalization
                (log_divergence / 10.0).min(1.0).max(0.0)
            } else {
                0.0
            };
            
            let score = sigmoid_normalize(normalized, 0.2, 5.0);
            let side = if is_bullish_div { 1 } else { -1 };
            
            // Create raw signal for divergence
            let signal = RawSignal::new(
                symbols[i].clone(),
                timeframes[i],
                timestamps[i],
                RawSignalType::Obv.to_indicator_id(),
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