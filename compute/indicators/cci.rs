use std::vec::Vec;

pub fn calculate_cci(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    if high.len() != low.len() || high.len() != close.len() {
        panic!("High, low, and close arrays must have the same length");
    }

    let n = high.len();
    let mut result = vec![f64::NAN; n];

    if n < period {
        return result;
    }

    // Calculate typical price (TP) = (high + low + close) / 3
    let mut tp = vec![0.0; n];
    for i in 0..n {
        tp[i] = (high[i] + low[i] + close[i]) / 3.0;
    }

    // Calculate CCI for each period
    for i in (period - 1)..n {
        // Calculate SMA of TP
        let mut sma_tp = 0.0;
        for j in (i + 1 - period)..=i {
            sma_tp += tp[j];
        }
        sma_tp /= period as f64;

        // Calculate mean deviation
        let mut mean_dev = 0.0;
        for j in (i + 1 - period)..=i {
            mean_dev += (tp[j] - sma_tp).abs();
        }
        mean_dev /= period as f64;

        // Calculate CCI
        if mean_dev != 0.0 {
            result[i] = (tp[i] - sma_tp) / (0.015 * mean_dev);
        } else {
            result[i] = 0.0; // Avoid division by zero
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cci_calculation() {
        let high = vec![102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let close = vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0];
        let cci_values = calculate_cci(&high, &low, &close, 5);
        
        assert_eq!(cci_values.len(), high.len());
        // Check that early values are NaN
        for i in 0..4 {
            if i < cci_values.len() {
                assert!(cci_values[i].is_nan());
            }
        }
        // Later values should be calculated
        if cci_values.len() > 5 {
            assert!(!cci_values[5].is_nan());
        }
    }
}