use std::vec::Vec;

pub fn calculate_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut result = vec![f64::NAN; n];

    if n < period + 1 {
        return result;
    }

    // Calculate initial average gain and loss
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    for i in 1..=period {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            avg_gain += change;
        } else {
            avg_loss += change.abs();
        }
    }

    avg_gain /= period as f64;
    avg_loss /= period as f64;

    // Calculate RSI for the first valid position
    if avg_loss != 0.0 {
        let rs = avg_gain / avg_loss;
        result[period] = 100.0 - (100.0 / (1.0 + rs));
    } else {
        result[period] = 100.0;
    }

    // Calculate subsequent values
    for i in (period + 1)..n {
        let change = prices[i] - prices[i - 1];
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { change.abs() } else { 0.0 };

        avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;

        if avg_loss != 0.0 {
            let rs = avg_gain / avg_loss;
            result[i] = 100.0 - (100.0 / (1.0 + rs));
        } else {
            result[i] = 100.0;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi_calculation() {
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0];
        let rsi_values = calculate_rsi(&prices, 5);
        
        assert_eq!(rsi_values.len(), prices.len());
        // Check that first few values are NaN
        assert!(rsi_values[0].is_nan());
        assert!(rsi_values[1].is_nan());
        assert!(rsi_values[2].is_nan());
        assert!(rsi_values[3].is_nan());
        assert!(rsi_values[4].is_nan());
        // Values from index 5 onwards should have RSI values
        assert!(!rsi_values[5].is_nan());
    }
}