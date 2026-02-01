use std::vec::Vec;

pub fn calculate_sma(prices: &[f64], period: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(prices.len());
    
    for i in 0..prices.len() {
        if i < period - 1 {
            result.push(f64::NAN);
        } else {
            let sum: f64 = prices[(i + 1 - period)..=i].iter().sum();
            result.push(sum / period as f64);
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_calculation() {
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0];
        let sma_values = calculate_sma(&prices, 3);
        
        assert_eq!(sma_values.len(), prices.len());
        // Check that first few values are NaN
        assert!(sma_values[0].is_nan());
        assert!(sma_values[1].is_nan());
        // Value from index 2 onwards should have SMA values
        assert!(!sma_values[2].is_nan());
        assert!(!sma_values[3].is_nan());
    }
}