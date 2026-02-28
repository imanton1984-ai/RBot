// compute/indicators/cmf.rs
//
// Chaikin Money Flow (CMF) — measures accumulation/distribution pressure.
//
// Formula:
//   Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
//   Money Flow Volume = MF Multiplier * Volume
//   CMF = Sum(MF Volume, period) / Sum(Volume, period)
//
// Range: -1.0 to +1.0
//   > 0 = buying pressure
//   < 0 = selling pressure
//
// Output: Vec<f64> of length == input length.
// First `period - 1` values are NaN.

/// Calculate Chaikin Money Flow (CMF).
///
/// Returns a vector of the same length as inputs.
/// The first `period - 1` elements will be `f64::NAN`.
pub fn calculate_cmf(
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

    if n < period {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; n];

    // Money Flow Multiplier for each bar
    let mf_multiplier: Vec<f64> = (0..n)
        .map(|i| {
            let range = high[i] - low[i];
            if range.abs() > 1e-12 {
                ((close[i] - low[i]) - (high[i] - close[i])) / range
            } else {
                0.0
            }
        })
        .collect();

    // Money Flow Volume
    let mf_volume: Vec<f64> = (0..n)
        .map(|i| mf_multiplier[i] * volume[i])
        .collect();

    // Initial rolling sums
    let mut sum_mfv: f64 = mf_volume[..period].iter().sum();
    let mut sum_vol: f64 = volume[..period].iter().sum();

    result[period - 1] = if sum_vol.abs() > 1e-12 {
        sum_mfv / sum_vol
    } else {
        0.0
    };

    // Sliding window
    for i in period..n {
        sum_mfv += mf_volume[i] - mf_volume[i - period];
        sum_vol += volume[i] - volume[i - period];

        result[i] = if sum_vol.abs() > 1e-12 {
            sum_mfv / sum_vol
        } else {
            0.0
        };
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmf_basic() {
        // Bullish scenario: close near highs
        let high  = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
        let low   = vec![ 8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
        let close = vec![ 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8, 16.8, 17.8, 18.8];
        let vol   = vec![100.0; 10];

        let cmf = calculate_cmf(&high, &low, &close, &vol, 5);
        assert_eq!(cmf.len(), 10);

        // First 4 NaN
        for i in 0..4 {
            assert!(cmf[i].is_nan(), "cmf[{}] should be NaN", i);
        }

        // Close near high → positive CMF
        for i in 4..10 {
            assert!(cmf[i].is_finite());
            assert!(cmf[i] > 0.0, "cmf[{}] = {} should be positive", i, cmf[i]);
            assert!(cmf[i] <= 1.0, "cmf[{}] = {} should be <= 1.0", i, cmf[i]);
        }
    }

    #[test]
    fn test_cmf_bearish() {
        // Bearish scenario: close near lows
        let high  = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let low   = vec![ 8.0,  9.0, 10.0, 11.0, 12.0];
        let close = vec![ 8.2,  9.2, 10.2, 11.2, 12.2];
        let vol   = vec![100.0; 5];

        let cmf = calculate_cmf(&high, &low, &close, &vol, 5);
        // CMF should be negative (close near low)
        assert!(cmf[4].is_finite());
        assert!(cmf[4] < 0.0);
    }
}
