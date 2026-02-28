// compute/indicators/fibo.rs
//
// Fibonacci Pivot Levels — dynamic S/R based on rolling high/low/close.
//
// Uses a rolling window of `period` bars to compute:
//   Pivot = (High + Low + Close) / 3
//   R1 = Pivot + 0.382 * (High - Low)
//   S1 = Pivot - 0.382 * (High - Low)
//   R2 = Pivot + 0.618 * (High - Low)
//   S2 = Pivot - 0.618 * (High - Low)
//   R3 = Pivot + 1.000 * (High - Low)
//   S3 = Pivot - 1.000 * (High - Low)
//
// Output: tuple of 7 Vec<f64> (pivot, r1, s1, r2, s2, r3, s3).
// First `period - 1` values are NaN.

/// Fibonacci pivot/S/R level structure for one bar.
pub struct FiboLevels {
    pub pivot: Vec<f64>,
    pub r1: Vec<f64>,
    pub s1: Vec<f64>,
    pub r2: Vec<f64>,
    pub s2: Vec<f64>,
    pub r3: Vec<f64>,
    pub s3: Vec<f64>,
}

/// Calculate Fibonacci Pivot Levels using a rolling window.
///
/// Returns FiboLevels with each vector having the same length as input.
/// First `period - 1` elements are `f64::NAN`.
pub fn calculate_fibo_levels(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> FiboLevels {
    let n = high.len();
    debug_assert_eq!(n, low.len());
    debug_assert_eq!(n, close.len());

    let mut pivot = vec![f64::NAN; n];
    let mut r1 = vec![f64::NAN; n];
    let mut s1 = vec![f64::NAN; n];
    let mut r2 = vec![f64::NAN; n];
    let mut s2 = vec![f64::NAN; n];
    let mut r3 = vec![f64::NAN; n];
    let mut s3 = vec![f64::NAN; n];

    if n < period {
        return FiboLevels { pivot, r1, s1, r2, s2, r3, s3 };
    }

    for i in (period - 1)..n {
        let start = i + 1 - period;

        // Rolling high/low over the window
        let mut window_high = f64::NEG_INFINITY;
        let mut window_low = f64::INFINITY;
        for j in start..=i {
            if high[j] > window_high { window_high = high[j]; }
            if low[j] < window_low { window_low = low[j]; }
        }
        let window_close = close[i];

        let p = (window_high + window_low + window_close) / 3.0;
        let range = window_high - window_low;

        pivot[i] = p;
        r1[i] = p + 0.382 * range;
        s1[i] = p - 0.382 * range;
        r2[i] = p + 0.618 * range;
        s2[i] = p - 0.618 * range;
        r3[i] = p + 1.000 * range;
        s3[i] = p - 1.000 * range;
    }

    FiboLevels { pivot, r1, s1, r2, s2, r3, s3 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibo_basic() {
        let high  = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0];
        let low   = vec![ 8.0,  9.0,  9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0];
        let close = vec![ 9.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0];

        let fibo = calculate_fibo_levels(&high, &low, &close, 5);
        assert_eq!(fibo.pivot.len(), 10);

        // First 4 (period-1) should be NaN
        for i in 0..4 {
            assert!(fibo.pivot[i].is_nan());
        }

        // From index 4 onward, values should be valid
        for i in 4..10 {
            assert!(fibo.pivot[i].is_finite());
            assert!(fibo.r1[i] > fibo.pivot[i]);
            assert!(fibo.s1[i] < fibo.pivot[i]);
            assert!(fibo.r2[i] > fibo.r1[i]);
            assert!(fibo.s2[i] < fibo.s1[i]);
        }
    }
}
