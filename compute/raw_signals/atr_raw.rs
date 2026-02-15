use crate::thresholds::{RawSignal, RawSignalType, SignalConfig, SignalKind};
use crate::scoring::{normalize_indicator_to_score_with_price, sigmoid_normalize};
use common::{Symbol, Timeframe};
use std::vec::Vec;

/// Calculate ATR-based raw signals.
///
/// `close_prices` is used to normalize ATR as a percentage of the current price,
/// making scores comparable across assets with vastly different price scales
/// (e.g. BTC ≈ 60 000 USD vs DOGE ≈ 0.10 USD).
pub fn calculate_atr_raw_signals(
    atr_values: &[f64],
    close_prices: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    config: &SignalConfig,
) -> Vec<RawSignal> {
    let mut signals = Vec::new();
    
    for i in 0..atr_values.len() {
        if i >= timestamps.len() || i >= symbols.len() || i >= timeframes.len() || i >= close_prices.len() {
            break;
        }
        
        let raw_value = atr_values[i];
        
        if raw_value.is_nan() {
            continue;
        }
        
        // Normalize ATR relative to close price for cross-asset comparability
        let score = normalize_indicator_to_score_with_price(raw_value, "atr", close_prices[i]);
        
        // Create raw signal
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::Atr.to_indicator_id(),
            SignalKind::PriceRelation.to_i16(),
            0,
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

/// Calculate ATR trend signals based on ATR expansion/contraction
pub fn calculate_atr_trend_signals(
    atr_values: &[f64],
    timestamps: &[i64],
    symbols: &[Symbol],
    timeframes: &[Timeframe],
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

        let side = if atr_change > 0.0 { 1 } else { -1 };
        
        // Create raw signal for ATR trend
        let signal = RawSignal::new(
            symbols[i].clone(),
            timeframes[i],
            timestamps[i],
            RawSignalType::Atr.to_indicator_id(),
            SignalKind::Volatility.to_i16(),
            side,
            score as f32,
            atr_change_pct as f32,
            None,
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
    sigmoid_normalize(normalized, 0.2, 5.0)
}