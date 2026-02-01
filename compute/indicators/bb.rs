use std::vec::Vec;

pub fn calculate_bollinger_bands(
    prices: &[f64],
    period: usize,
    num_std_dev: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = prices.len();
    let mut upper_band = vec![f64::NAN; n];
    let mut middle_band = vec![f64::NAN; n];
    let mut lower_band = vec![f64::NAN; n];

    for i in 0..n {
        if i < period - 1 {
            continue;
        }

        // Calculate SMA (middle band)
        let sum: f64 = prices[(i + 1 - period)..=i].iter().sum();
        middle_band[i] = sum / period as f64;

        // Calculate standard deviation
        let mut sum_sq_diff = 0.0;
        for j in (i + 1 - period)..=i {
            let diff = prices[j] - middle_band[i];
            sum_sq_diff += diff * diff;
        }
        let std_dev = (sum_sq_diff / period as f64).sqrt();

        // Upper and lower bands
        upper_band[i] = middle_band[i] + (num_std_dev * std_dev);
        lower_band[i] = middle_band[i] - (num_std_dev * std_dev);
    }

    (upper_band, middle_band, lower_band)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bollinger_bands_calculation() {
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0];
        let (upper, middle, lower) = calculate_bollinger_bands(&prices, 3, 2.0);
        
        assert_eq!(upper.len(), prices.len());
        assert_eq!(middle.len(), prices.len());
        assert_eq!(lower.len(), prices.len());
    }
}