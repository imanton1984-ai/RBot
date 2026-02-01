use std::vec::Vec;

// Long-term trend detection based on longer period moving averages
pub fn calculate_long_term_trend(
    prices: &[f64],
    fast_ma_period: usize,  // Typically 20-50 periods
    slow_ma_period: usize,  // Typically 50-200 periods
) -> Vec<i8> {  // 1 for uptrend, -1 for downtrend, 0 for sideways
    if fast_ma_period >= slow_ma_period {
        panic!("Fast MA period must be less than slow MA period");
    }

    let n = prices.len();
    let mut trends = vec![0i8; n];

    if n < slow_ma_period {
        return trends;
    }

    // Calculate moving averages
    let fast_ma = super::sma::calculate_sma(prices, fast_ma_period);
    let slow_ma = super::sma::calculate_sma(prices, slow_ma_period);

    for i in slow_ma_period..n {
        if !fast_ma[i].is_nan() && !slow_ma[i].is_nan() {
            if fast_ma[i] > slow_ma[i] {
                trends[i] = 1;  // Long-term uptrend
            } else if fast_ma[i] < slow_ma[i] {
                trends[i] = -1; // Long-term downtrend
            } else {
                trends[i] = 0;  // Long-term sideways
            }
        }
    }

    trends
}

// Trend strength based on the distance between long-term MAs
pub fn calculate_long_term_trend_strength(
    prices: &[f64],
    fast_ma_period: usize,  // Typically 20-50 periods
    slow_ma_period: usize,  // Typically 50-200 periods
) -> Vec<f64> {
    if fast_ma_period >= slow_ma_period {
        panic!("Fast MA period must be less than slow MA period");
    }

    let n = prices.len();
    let mut strengths = vec![f64::NAN; n];

    if n < slow_ma_period {
        return strengths;
    }

    // Calculate moving averages
    let fast_ma = super::sma::calculate_sma(prices, fast_ma_period);
    let slow_ma = super::sma::calculate_sma(prices, slow_ma_period);

    for i in slow_ma_period..n {
        if !fast_ma[i].is_nan() && !slow_ma[i].is_nan() {
            // Calculate normalized distance between MAs
            strengths[i] = ((fast_ma[i] - slow_ma[i]) / slow_ma[i]) * 100.0;
        }
    }

    strengths
}

// Directional Movement Index for long-term trend confirmation
pub fn calculate_dmi(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {  // (+DI, -DI, ADX)
    if high.len() != low.len() || high.len() != close.len() {
        panic!("High, low, and close arrays must have the same length");
    }

    let n = high.len();
    let mut plus_di = vec![f64::NAN; n];
    let mut minus_di = vec![f64::NAN; n];
    let mut adx = vec![f64::NAN; n];

    if n < period {
        return (plus_di, minus_di, adx);
    }

    // Calculate directional movements
    let mut plus_dm = vec![0.0; n];
    let mut minus_dm = vec![0.0; n];
    let mut tr = vec![0.0; n];

    for i in 1..n {
        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];

        plus_dm[i] = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
        minus_dm[i] = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };

        let h_l = high[i] - low[i];
        let h_pc = (high[i] - close[i - 1]).abs();
        let l_pc = (low[i] - close[i - 1]).abs();

        tr[i] = h_l.max(h_pc).max(l_pc);
    }

    // Smooth the values
    let mut smooth_plus_dm = vec![0.0; n];
    let mut smooth_minus_dm = vec![0.0; n];
    let mut smooth_tr = vec![0.0; n];

    // Initialize first values
    for i in 1..=period {
        if i < n {
            smooth_plus_dm[period - 1] += plus_dm[i];
            smooth_minus_dm[period - 1] += minus_dm[i];
            smooth_tr[period - 1] += tr[i];
        }
    }

    // Continue with smoothing
    for i in period..n {
        smooth_plus_dm[i] = smooth_plus_dm[i - 1] - (smooth_plus_dm[i - 1] / period as f64) + plus_dm[i];
        smooth_minus_dm[i] = smooth_minus_dm[i - 1] - (smooth_minus_dm[i - 1] / period as f64) + minus_dm[i];
        smooth_tr[i] = smooth_tr[i - 1] - (smooth_tr[i - 1] / period as f64) + tr[i];

        if smooth_tr[i] != 0.0 {
            plus_di[i] = (smooth_plus_dm[i] / smooth_tr[i]) * 100.0;
            minus_di[i] = (smooth_minus_dm[i] / smooth_tr[i]) * 100.0;
        }
    }

    // Calculate ADX
    if n >= period * 2 {
        let mut dx = vec![0.0; n];
        for i in period..n {
            let denominator = plus_di[i] + minus_di[i];
            if denominator != 0.0 {
                dx[i] = ((plus_di[i] - minus_di[i]).abs() / denominator) * 100.0;
            }
        }

        // First ADX value
        let mut sum_dx = 0.0;
        let start_adx = period * 2 - 1;
        if start_adx < n {
            for i in period..start_adx {
                sum_dx += dx[i];
            }
            adx[start_adx] = sum_dx / (period - 1) as f64;

            // Continue with smoothing
            for i in (start_adx + 1)..n {
                adx[i] = ((adx[i - 1] * (period as f64 - 1.0)) + dx[i]) / period as f64;
            }
        }
    }

    (plus_di, minus_di, adx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_term_trend_calculation() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0];

        let trends = calculate_long_term_trend(&prices, 3, 6);
        let strengths = calculate_long_term_trend_strength(&prices, 3, 6);

        assert_eq!(trends.len(), prices.len());
        assert_eq!(strengths.len(), prices.len());
    }
}