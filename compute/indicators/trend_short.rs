use std::vec::Vec;

// Short-term trend detection using shorter lookback periods
pub fn calculate_short_term_trend(
    prices: &[f64],
    fast_ma_period: usize,  // Typically 5-15 periods
    slow_ma_period: usize,  // Typically 10-30 periods
) -> Vec<i8> {  // 1 for uptrend, -1 for downtrend, 0 for uncertain
    if fast_ma_period >= slow_ma_period {
        panic!("Fast MA period must be less than slow MA period");
    }

    let n = prices.len();
    let mut trends = vec![0i8; n];

    if n < slow_ma_period {
        return trends;
    }

    // Calculate short-term moving averages
    let fast_ma = super::sma::calculate_sma(prices, fast_ma_period);
    let slow_ma = super::sma::calculate_sma(prices, slow_ma_period);

    for i in slow_ma_period..n {
        if !fast_ma[i].is_nan() && !slow_ma[i].is_nan() {
            if fast_ma[i] > slow_ma[i] {
                trends[i] = 1;  // Short-term uptrend
            } else if fast_ma[i] < slow_ma[i] {
                trends[i] = -1; // Short-term downtrend
            } else {
                trends[i] = 0;  // Short-term sideways
            }
        }
    }

    trends
}

// Very short-term momentum-based trend (typically 2-5 periods)
pub fn calculate_very_short_term_momentum(
    prices: &[f64],
    period: usize,
) -> Vec<i8> {
    let n = prices.len();
    let mut trends = vec![0i8; n];

    if n < period {
        return trends;
    }

    for i in period..n {
        let past_price = prices[i - period];
        let current_price = prices[i];

        if (current_price - past_price).abs() > 0.001 { // Small epsilon to handle floating point precision
            if current_price > past_price {
                trends[i] = 1;  // Positive short-term momentum
            } else {
                trends[i] = -1; // Negative short-term momentum
            }
        } else {
            trends[i] = 0;  // Neutral
        }
    }

    trends
}

// Price action based short-term trend using consecutive higher highs/lower lows
pub fn calculate_price_action_trend(
    high: &[f64],
    low: &[f64],
    period: usize,
) -> Vec<i8> {
    if high.len() != low.len() {
        panic!("High and low arrays must have the same length");
    }

    let n = high.len();
    let mut trends = vec![0i8; n];

    if n < period * 2 {
        return trends;
    }

    for i in (period * 2 - 1)..n {
        let mut higher_highs = 0;
        let mut lower_lows = 0;

        // Compare recent highs and lows
        for j in 1..=period {
            if i >= j && i >= j * 2 {
                if high[i - j + 1] > high[i - j*2 + 1] {
                    higher_highs += 1;
                } else if high[i - j + 1] < high[i - j*2 + 1] {
                    higher_highs -= 1;
                }

                if low[i - j + 1] > low[i - j*2 + 1] {
                    lower_lows += 1;
                } else if low[i - j + 1] < low[i - j*2 + 1] {
                    lower_lows -= 1;
                }
            }
        }

        // Determine trend based on pattern
        if higher_highs > 0 && lower_lows > 0 {
            trends[i] = 1;  // Uptrend confirmed by higher highs and higher lows
        } else if higher_highs < 0 && lower_lows < 0 {
            trends[i] = -1; // Downtrend confirmed by lower highs and lower lows
        } else {
            trends[i] = 0;  // Uncertain
        }
    }

    trends
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_term_trend_calculation() {
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0];

        let trends = calculate_short_term_trend(&prices, 2, 4);
        let momentum_trends = calculate_very_short_term_momentum(&prices, 2);

        assert_eq!(trends.len(), prices.len());
        assert_eq!(momentum_trends.len(), prices.len());
    }
}