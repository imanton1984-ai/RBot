use std::vec::Vec;

pub fn calculate_ema(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut result = vec![f64::NAN; n];

    if n < period {
        return result;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);

    // Calculate SMA for the first EMA value
    let mut sma_sum = 0.0;
    for i in 0..period {
        sma_sum += prices[i];
    }
    let mut ema = sma_sum / period as f64;
    result[period - 1] = ema;

    // Calculate subsequent EMA values
    for i in period..n {
        ema = (prices[i] - ema) * multiplier + ema;
        result[i] = ema;
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
}