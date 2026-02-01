use std::vec::Vec;

/// Normalizes a signal value to the range [0, 1] based on min/max values
pub fn normalize_signal(value: f64, min_val: f64, max_val: f64) -> f64 {
    if max_val == min_val {
        return 0.5; // Return middle value if min and max are the same
    }
    
    let normalized = (value - min_val) / (max_val - min_val);
    
    // Clamp to [0, 1] range
    normalized.clamp(0.0, 1.0)
}

/// Applies sigmoid normalization to emphasize strong signals
pub fn sigmoid_normalize(value: f64, center: f64, steepness: f64) -> f64 {
    let x = (value - center) * steepness;
    1.0 / (1.0 + (-x).exp())
}

/// Converts raw indicator values to normalized scores in [0, 1] range
pub fn normalize_indicator_to_score(raw_value: f64, indicator_type: &str) -> f64 {
    match indicator_type {
        "rsi" => normalize_rsi_score(raw_value),
        "macd" => normalize_macd_score(raw_value),
        "cci" => normalize_cci_score(raw_value),
        "stoch" => normalize_stoch_score(raw_value),
        "williams" => normalize_williams_score(raw_value),
        "atr" => normalize_atr_score(raw_value), // Volatility measure
        "adx" => normalize_adx_score(raw_value), // Trend strength
        _ => normalize_generic_score(raw_value),
    }
}

/// Normalize RSI values to score (RSI typically ranges 0-100)
fn normalize_rsi_score(rsi_value: f64) -> f64 {
    if rsi_value.is_nan() {
        return 0.0;
    }
    
    // RSI: values near 0 or 100 are strong signals
    // RSI near 50 is neutral
    let distance_from_extremes = ((rsi_value - 0.0).abs().min((rsi_value - 100.0).abs())).min(50.0);
    let normalized = 1.0 - (distance_from_extremes / 50.0);
    
    // Apply sigmoid to emphasize strong signals
    sigmoid_normalize(normalized, 0.5, 8.0)
}

/// Normalize MACD values to score
fn normalize_macd_score(macd_value: f64) -> f64 {
    if macd_value.is_nan() {
        return 0.0;
    }
    
    // Use absolute value to measure signal strength regardless of direction
    let abs_value = macd_value.abs();
    
    // Normalize to 0-1 scale (assuming max meaningful MACD value is around 10)
    let normalized = (abs_value / 10.0).min(1.0);
    
    // Apply sigmoid to emphasize strong signals
    sigmoid_normalize(normalized, 0.3, 6.0)
}

/// Normalize CCI values to score
fn normalize_cci_score(cci_value: f64) -> f64 {
    if cci_value.is_nan() {
        return 0.0;
    }
    
    // CCI: values beyond ±100 are considered strong
    let abs_value = cci_value.abs();
    let normalized = (abs_value / 200.0).min(1.0); // Assuming max meaningful value is 200
    
    // Apply sigmoid to emphasize strong signals
    sigmoid_normalize(normalized, 0.3, 6.0)
}

/// Normalize Stochastic values to score
fn normalize_stoch_score(stoch_value: f64) -> f64 {
    if stoch_value.is_nan() {
        return 0.0;
    }
    
    // Stochastic: values near 0 or 100 are strong signals
    let distance_from_extremes = ((stoch_value - 0.0).abs().min((stoch_value - 100.0).abs())).min(50.0);
    let normalized = 1.0 - (distance_from_extremes / 50.0);
    
    // Apply sigmoid to emphasize strong signals
    sigmoid_normalize(normalized, 0.5, 8.0)
}

/// Normalize Williams %R values to score
fn normalize_williams_score(williams_value: f64) -> f64 {
    if williams_value.is_nan() {
        return 0.0;
    }
    
    // Williams %R: values near -0 or -100 are strong signals
    let abs_value = williams_value.abs();
    let distance_from_extremes = ((abs_value - 0.0).abs().min((abs_value - 100.0).abs())).min(50.0);
    let normalized = 1.0 - (distance_from_extremes / 50.0);
    
    // Apply sigmoid to emphasize strong signals
    sigmoid_normalize(normalized, 0.5, 8.0)
}

/// Normalize ATR values to score (measures volatility)
fn normalize_atr_score(atr_value: f64) -> f64 {
    if atr_value.is_nan() || atr_value <= 0.0 {
        return 0.0;
    }
    
    // ATR: higher values indicate higher volatility (potentially stronger signals)
    // Normalize assuming max meaningful ATR is around 10% of price
    let normalized = (atr_value / 0.1).min(1.0); // Adjust this based on your price scale
    
    // Apply sigmoid to emphasize strong signals
    sigmoid_normalize(normalized, 0.2, 5.0)
}

/// Normalize ADX values to score (measures trend strength)
fn normalize_adx_score(adx_value: f64) -> f64 {
    if adx_value.is_nan() || adx_value <= 0.0 {
        return 0.0;
    }
    
    // ADX: values above 25 indicate trend, above 50 strong trend
    let normalized = (adx_value / 100.0).min(1.0); // ADX typically ranges 0-100
    
    // Apply sigmoid to emphasize strong signals
    sigmoid_normalize(normalized, 0.3, 6.0)
}

/// Generic normalization for unknown indicator types
fn normalize_generic_score(value: f64) -> f64 {
    if value.is_nan() {
        return 0.0;
    }
    
    // Use absolute value and normalize assuming max value of 100
    let abs_value = value.abs();
    let normalized = (abs_value / 100.0).min(1.0);
    
    sigmoid_normalize(normalized, 0.3, 5.0)
}

/// Applies threshold filtering to remove only weak signals (below 0.80)
/// Stronger signals (0.80+) are kept for further processing by predictor
pub fn filter_weak_signals(score: f64) -> bool {
    score >= 0.80  // Only filter out signals below 0.80
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_signal() {
        assert_eq!(normalize_signal(50.0, 0.0, 100.0), 0.5);
        assert_eq!(normalize_signal(0.0, 0.0, 100.0), 0.0);
        assert_eq!(normalize_signal(100.0, 0.0, 100.0), 1.0);
    }

    #[test]
    fn test_normalize_rsi_score() {
        assert!(normalize_rsi_score(0.0) > 0.8); // Strong oversold signal
        assert!(normalize_rsi_score(100.0) > 0.8); // Strong overbought signal
        assert!(normalize_rsi_score(50.0) < 0.3); // Neutral signal
    }

    #[test]
    fn test_filter_weak_signals() {
        assert!(filter_weak_signals(0.85));  // Above threshold
        assert!(!filter_weak_signals(0.75)); // Below threshold
    }
}