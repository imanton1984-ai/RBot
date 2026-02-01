use std::vec::Vec;

pub fn calculate_obv(
    close_prices: &[f64],
    volumes: &[f64],
) -> Vec<f64> {
    let n = close_prices.len();
    if n != volumes.len() || n == 0 {
        return vec![];
    }

    let mut obv = vec![0.0f64; n];
    obv[0] = volumes[0];

    for i in 1..n {
        if close_prices[i] > close_prices[i - 1] {
            obv[i] = obv[i - 1] + volumes[i];
        } else if close_prices[i] < close_prices[i - 1] {
            obv[i] = obv[i - 1] - volumes[i];
        } else {
            obv[i] = obv[i - 1];
        }
    }

    obv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obv_calculation() {
        let close_prices = vec![100.0, 101.0, 102.0, 101.5, 103.0];
        let volumes = vec![1000.0, 1200.0, 800.0, 1500.0, 900.0];
        let obv_values = calculate_obv(&close_prices, &volumes);
        
        assert_eq!(obv_values.len(), close_prices.len());
        assert_eq!(obv_values[0], 1000.0); // First value should equal first volume
        assert!(obv_values[1] > obv_values[0]); // Price went up, so OBV should increase
        assert!(obv_values[2] > obv_values[1]); // Price went up again
        assert!(obv_values[3] < obv_values[2]); // Price went down, so OBV should decrease
    }
}