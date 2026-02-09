use std::vec::Vec;

pub fn calculate_ema(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut result = vec![f64::NAN; n];

    // Find the first valid (non-NaN) index to start calculation
    let first_valid_idx = match prices.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return result, // All NaNs
    };

    // Ensure we have enough data starting from the first valid point
    if n - first_valid_idx < period {
        return result;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);

    // Calculate SMA for the first EMA value, starting from first_valid_idx
    let mut sma_sum = 0.0;
    for i in first_valid_idx..(first_valid_idx + period) {
        sma_sum += prices[i];
    }
    
    let initial_ema = sma_sum / period as f64;
    // Store the initial EMA at the end of the first valid period
    result[first_valid_idx + period - 1] = initial_ema;

    let mut prev_ema = initial_ema;

    // Calculate subsequent EMA values
    for i in (first_valid_idx + period)..n {
        // If we encounter a gap (NaN) in the middle of data, we carry forward previous EMA
        // or effectively skip updates. Here we skip update if current price is NaN.
        if prices[i].is_nan() {
            // Option A: result[i] = NaN; prev_ema = NaN; (Strict)
            // Option B: Carry forward (Resilient)
            // Let's use Option A to avoid inventing data, but restart if needed.
            // For simplicity in financial timeseries, usually intermediate NaNs are rare.
            result[i] = f64::NAN;
        } else {
            let ema = (prices[i] - prev_ema) * multiplier + prev_ema;
            result[i] = ema;
            prev_ema = ema;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_calculation() {
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0];
        let ema_values = calculate_ema(&prices, 5);

        assert_eq!(ema_values.len(), prices.len());
        // Check that first few values are NaN
        for i in 0..4 {
            assert!(ema_values[i].is_nan());
        }
        // Values from index 4 onwards should have EMA values
        for i in 4..ema_values.len() {
            assert!(!ema_values[i].is_nan());
        }
    }

    #[test]
    fn test_ema_with_leading_nans() {
        // MACD line often starts with NaNs. Let's simulate that.
        let input = vec![f64::NAN, f64::NAN, 10.0, 11.0, 12.0, 13.0, 14.0];
        // Period 3. 
        // Valid data starts at index 2.
        // First EMA should be at index 2 + 3 - 1 = 4.
        // SMA of (10, 11, 12) = 11. 
        let ema = calculate_ema(&input, 3);
        
        assert!(ema[0].is_nan());
        assert!(ema[1].is_nan());
        assert!(ema[2].is_nan());
        assert!(ema[3].is_nan()); 
        assert_eq!(ema[4], 11.0); // The first calculated EMA
        assert!(!ema[5].is_nan());
    }
}