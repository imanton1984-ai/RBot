// compute/indicators/mfi.rs
//
// Money Flow Index (MFI) — volume-weighted RSI.
// Uses typical price * volume to measure buying/selling pressure.
//
// Formula:
//   Typical Price = (High + Low + Close) / 3
//   Raw Money Flow = Typical Price * Volume
//   Money Flow Ratio = Positive MF / Negative MF (over `period` bars)
//   MFI = 100 - 100 / (1 + Money Flow Ratio)
//
// Output: Vec<f64> of length == input length.
// First `period` values are NaN.

/// Calculate Money Flow Index (MFI).
///
/// Returns a vector of the same length as inputs.
/// The first `period` elements will be `f64::NAN`.
pub fn calculate_mfi(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    period: usize,
) -> Vec<f64> {
    let n = high.len();
    debug_assert_eq!(n, low.len());
    debug_assert_eq!(n, close.len());
    debug_assert_eq!(n, volume.len());

    if n < period + 1 {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; n];

    // Typical prices
    let tp: Vec<f64> = (0..n)
        .map(|i| (high[i] + low[i] + close[i]) / 3.0)
        .collect();

    // Raw money flow
    let rmf: Vec<f64> = (0..n).map(|i| tp[i] * volume[i]).collect();

    // Classify positive / negative flow
    // Flow at bar i is positive if tp[i] > tp[i-1], negative otherwise
    let mut pos_flow = vec![0.0f64; n];
    let mut neg_flow = vec![0.0f64; n];
    for i in 1..n {
        if tp[i] > tp[i - 1] {
            pos_flow[i] = rmf[i];
        } else if tp[i] < tp[i - 1] {
            neg_flow[i] = rmf[i];
        } else {
            // Equal — split or ignore. Classic approach: ignore.
        }
    }

    // Rolling sum over `period` bars
    let mut sum_pos: f64 = pos_flow[1..=period].iter().sum();
    let mut sum_neg: f64 = neg_flow[1..=period].iter().sum();

    // MFI at bar == period
    let mfr = if sum_neg.abs() > 1e-12 {
        sum_pos / sum_neg
    } else {
        100.0 // All positive → MFI = 100
    };
    result[period] = 100.0 - 100.0 / (1.0 + mfr);

    // Sliding window
    for i in (period + 1)..n {
        // Add new bar, remove oldest bar in window
        sum_pos += pos_flow[i] - pos_flow[i - period];
        sum_neg += neg_flow[i] - neg_flow[i - period];

        let mfr = if sum_neg.abs() > 1e-12 {
            sum_pos / sum_neg
        } else {
            100.0
        };
        result[i] = 100.0 - 100.0 / (1.0 + mfr);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mfi_basic() {
        let high = vec![10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5, 16.0, 15.5, 17.0, 16.5, 18.0];
        let low  = vec![ 9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5, 16.0, 15.5, 17.0];
        let close= vec![ 9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0, 15.5, 15.0, 16.5, 16.0, 17.5];
        let vol  = vec![100.0; 15];

        let mfi = calculate_mfi(&high, &low, &close, &vol, 14);
        assert_eq!(mfi.len(), 15);
        // First 14 are NaN
        for i in 0..14 {
            assert!(mfi[i].is_nan(), "mfi[{}] should be NaN", i);
        }
        // Last value should be valid
        assert!(mfi[14].is_finite());
        assert!(mfi[14] >= 0.0 && mfi[14] <= 100.0);
    }
}
