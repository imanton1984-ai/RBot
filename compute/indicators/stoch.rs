use std::vec::Vec;

pub fn calculate_stochastic(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    k_period: usize,
    d_period: usize,
) -> (Vec<f64>, Vec<f64>) {  // (%K, %D)
    if high.len() != low.len() || high.len() != close.len() {
        panic!("High, low, and close arrays must have the same length");
    }

    let n = high.len();
    let mut k_values = vec![f64::NAN; n];
    let mut d_values = vec![f64::NAN; n];

    if n < k_period {
        return (k_values, d_values);
    }

    // Calculate %K
    for i in (k_period - 1)..n {
        let start_idx = i + 1 - k_period;
        
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
        
        if highest_high != lowest_low {
            k_values[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100.0;
        } else {
            k_values[i] = 50.0; // Avoid division by zero
        }
    }

    // Calculate %D (3-period SMA of %K)
    if n >= k_period + d_period - 1 {
        for i in (k_period + d_period - 2)..n {
            let start_idx = i + 1 - d_period;
            let mut sum = 0.0;
            let mut count = 0;

            for j in start_idx..=i {
                if !k_values[j].is_nan() {
                    sum += k_values[j];
                    count += 1;
                }
            }

            if count > 0 {
                d_values[i] = sum / count as f64;
            } else {
                d_values[i] = f64::NAN;
            }
        }
    }

    (k_values, d_values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stochastic_calculation() {
        let high = vec![102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let close = vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0];
        
        let (k_values, d_values) = calculate_stochastic(&high, &low, &close, 5, 3);
        
        assert_eq!(k_values.len(), high.len());
        assert_eq!(d_values.len(), high.len());
        
        // Check that early values are NaN
        for i in 0..4 {
            if i < k_values.len() {
                assert!(k_values[i].is_nan());
            }
        }
    }
}