use std::vec::Vec;

pub fn calculate_williams_r(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Vec<f64> {
    if high.len() != low.len() || high.len() != close.len() {
        panic!("High, low, and close arrays must have the same length");
    }

    let n = high.len();
    let mut williams_r = vec![f64::NAN; n];

    if n < period {
        return williams_r;
    }

    for i in (period - 1)..n {
        let start_idx = i + 1 - period;
        
        let mut highest_high = f64::NEG_INFINITY;
        let mut lowest_low = f64::INFINITY;
        
        for j in start_idx..=i {
            if high[j] > highest_high {
                highest_high = high[j];
            }
            if low[j] < lowest_low {
                lowest_low = low[j];
            }
        }
        
        // Calculate Williams %R
        if highest_high != lowest_low {
            williams_r[i] = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100.0;
        } else {
            williams_r[i] = -50.0; // Avoid division by zero
        }
    }

    williams_r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_williams_r_calculation() {
        let high = vec![102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let low = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        let close = vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0];
        
        let r_values = calculate_williams_r(&high, &low, &close, 4);
        
        assert_eq!(r_values.len(), high.len());
        // Check that early values are NaN
        for i in 0..3 {
            if i < r_values.len() {
                assert!(r_values[i].is_nan());
            }
        }
        // Later values should be calculated (typically between -100 and 0)
        if r_values.len() > 4 {
            assert!(!r_values[4].is_nan());
            assert!(r_values[4] >= -100.0 && r_values[4] <= 0.0);
        }
    }
}