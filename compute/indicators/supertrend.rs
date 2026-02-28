// compute/indicators/supertrend.rs
//
// SuperTrend indicator — trend-following based on ATR.
//
// Parameters:
//   period: ATR period (default 10)
//   multiplier: ATR multiplier (default 3.0)
//
// Formula:
//   Basic Upper Band = (High + Low) / 2 + multiplier * ATR
//   Basic Lower Band = (High + Low) / 2 - multiplier * ATR
//
//   Final Upper Band = min(Basic Upper, prev Final Upper) if prev close <= prev Final Upper
//   Final Lower Band = max(Basic Lower, prev Final Lower) if prev close >= prev Final Lower
//
//   SuperTrend = Final Lower Band (uptrend) or Final Upper Band (downtrend)
//   Direction  = 1 (up/bullish) or -1 (down/bearish)
//
// Output: (supertrend_values, direction)
//   Both Vec<f64> of length == input length.
//   First `period` values are NaN / 0.0.

use crate::atr::calculate_atr;

/// Calculate SuperTrend indicator.
///
/// Returns (supertrend_values, direction) where:
/// - supertrend_values: the SuperTrend line values
/// - direction: 1.0 = bullish (price above), -1.0 = bearish (price below)
///
/// First `period` elements are NaN / 0.0.
pub fn calculate_supertrend(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    multiplier: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    debug_assert_eq!(n, low.len());
    debug_assert_eq!(n, close.len());

    let mut st_values = vec![f64::NAN; n];
    let mut direction = vec![0.0f64; n];

    if n < period + 1 {
        return (st_values, direction);
    }

    // Calculate ATR
    let atr = calculate_atr(high, low, close, period);

    let mut final_upper = vec![f64::NAN; n];
    let mut final_lower = vec![f64::NAN; n];

    // First valid bar
    let first = period;
    if first < n && atr[first].is_finite() {
        let hl2 = (high[first] + low[first]) / 2.0;
        final_upper[first] = hl2 + multiplier * atr[first];
        final_lower[first] = hl2 - multiplier * atr[first];
        // Initial direction: bullish
        direction[first] = 1.0;
        st_values[first] = final_lower[first];
    }

    for i in (first + 1)..n {
        if !atr[i].is_finite() {
            continue;
        }

        let hl2 = (high[i] + low[i]) / 2.0;
        let basic_upper = hl2 + multiplier * atr[i];
        let basic_lower = hl2 - multiplier * atr[i];

        // Final Upper Band
        final_upper[i] = if basic_upper < final_upper[i - 1] || close[i - 1] > final_upper[i - 1] {
            basic_upper
        } else {
            final_upper[i - 1]
        };

        // Final Lower Band
        final_lower[i] = if basic_lower > final_lower[i - 1] || close[i - 1] < final_lower[i - 1] {
            basic_lower
        } else {
            final_lower[i - 1]
        };

        // Determine direction
        if direction[i - 1] == 1.0 {
            // Was bullish
            if close[i] < final_lower[i] {
                direction[i] = -1.0; // Flip to bearish
            } else {
                direction[i] = 1.0;
            }
        } else {
            // Was bearish
            if close[i] > final_upper[i] {
                direction[i] = 1.0; // Flip to bullish
            } else {
                direction[i] = -1.0;
            }
        }

        // SuperTrend value
        st_values[i] = if direction[i] == 1.0 {
            final_lower[i]
        } else {
            final_upper[i]
        };
    }

    (st_values, direction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supertrend_basic() {
        // Simple uptrend data
        let high  = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0];
        let low   = vec![ 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0];
        let close = vec![ 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5];

        let (st, dir) = calculate_supertrend(&high, &low, &close, 5, 2.0);
        assert_eq!(st.len(), 12);
        assert_eq!(dir.len(), 12);

        // First 5 values should be NaN / 0
        for i in 0..5 {
            assert!(st[i].is_nan());
        }

        // In a strong uptrend, direction should be bullish (1.0)
        for i in 5..12 {
            assert!(st[i].is_finite());
            assert_eq!(dir[i], 1.0, "Expected bullish at bar {}", i);
        }
    }
}
