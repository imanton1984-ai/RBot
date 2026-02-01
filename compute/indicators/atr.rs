use std::vec::Vec;

pub fn calculate_atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    if high.len() != low.len() || high.len() != close.len() {
        panic!("High, low, and close arrays must have the same length");
    }

    let n = high.len();
    let mut result = vec![f64::NAN; n];

    if n < period {
        return result;
    }

    // Calculate True Range for each period
    let mut tr = vec![0.0; n];
    
    for i in 1..n {
        let hl = high[i] - low[i];
        let h_prev_close = (high[i] - close[i - 1]).abs();
        let l_prev_close = (low[i] - close[i - 1]).abs();
        
        tr[i] = hl.max(h_prev_close).max(l_prev_close);
    }

    // Calculate ATR using SMA for the first value, then EMA-style smoothing
    let mut sum = 0.0;
    for i in 1..=period {
        sum += tr[i];
    }
    
    result[period] = sum / period as f64;

    // Continue with smoothing (like Wilder's RMA - Relative Momentum Average)
    for i in (period + 1)..n {
        result[i] = (result[i - 1] * (period as f64 - 1.0) + tr[i]) / period as f64;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atr_calculation() {
        let high = vec![102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let close = vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0];
        let atr_values = calculate_atr(&high, &low, &close, 5);
        
        assert_eq!(atr_values.len(), high.len());
        // Check that early values are NaN
        assert!(atr_values[0].is_nan());
        assert!(atr_values[1].is_nan());
        assert!(atr_values[2].is_nan());
        assert!(atr_values[3].is_nan());
        assert!(atr_values[4].is_nan());
        // Later values should be calculated
        assert!(!atr_values[5].is_nan());
    }
}