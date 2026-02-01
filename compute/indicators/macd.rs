use std::vec::Vec;

pub fn calculate_macd(
    prices: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ema_fast = super::calculate_ema(prices, fast_period);
    let ema_slow = super::calculate_ema(prices, slow_period);

    let mut macd_line = Vec::with_capacity(prices.len());
    for i in 0..prices.len() {
        if ema_fast[i].is_nan() || ema_slow[i].is_nan() {
            macd_line.push(f64::NAN);
        } else {
            macd_line.push(ema_fast[i] - ema_slow[i]);
        }
    }

    let signal_line = super::calculate_ema(&macd_line, signal_period);

    let mut histogram = Vec::with_capacity(prices.len());
    for i in 0..prices.len() {
        if macd_line[i].is_nan() || signal_line[i].is_nan() {
            histogram.push(f64::NAN);
        } else {
            histogram.push(macd_line[i] - signal_line[i]);
        }
    }

    (macd_line, signal_line, histogram)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macd_calculation() {
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0];
        let (macd_line, signal_line, histogram) = calculate_macd(&prices, 3, 5, 3);

        assert_eq!(macd_line.len(), prices.len());
        assert_eq!(signal_line.len(), prices.len());
        assert_eq!(histogram.len(), prices.len());
    }
}