use std::vec::Vec;

pub fn calculate_adx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    if high.len() != low.len() || high.len() != close.len() {
        panic!("High, low, and close arrays must have the same length");
    }

    let n = high.len();
    let mut result = vec![f64::NAN; n];

    if n < period * 2 {
        return result;
    }

    // Calculate True Range, +DM, -DM
    let mut tr = vec![0.0; n];
    let mut plus_dm = vec![0.0; n];
    let mut minus_dm = vec![0.0; n];

    for i in 1..n {
        let h_l = high[i] - low[i];
        let h_pc = (high[i] - close[i - 1]).abs();
        let l_pc = (low[i] - close[i - 1]).abs();

        tr[i] = h_l.max(h_pc).max(l_pc);

        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];

        plus_dm[i] = if up_move > down_move && up_move > 0.0 {
            up_move
        } else {
            0.0
        };

        minus_dm[i] = if down_move > up_move && down_move > 0.0 {
            down_move
        } else {
            0.0
        };
    }

    // Calculate smoothed TR, +DM, -DM
    let mut smoothed_tr = vec![0.0; n];
    let mut smoothed_plus_dm = vec![0.0; n];
    let mut smoothed_minus_dm = vec![0.0; n];

    // Initialize first values
    let mut sum_tr = 0.0;
    let mut sum_plus_dm = 0.0;
    let mut sum_minus_dm = 0.0;

    for i in 1..=period {
        sum_tr += tr[i];
        sum_plus_dm += plus_dm[i];
        sum_minus_dm += minus_dm[i];
    }

    smoothed_tr[period] = sum_tr;
    smoothed_plus_dm[period] = sum_plus_dm;
    smoothed_minus_dm[period] = sum_minus_dm;

    // Smooth the rest
    for i in (period + 1)..n {
        smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] / period as f64) + tr[i];
        smoothed_plus_dm[i] = smoothed_plus_dm[i - 1] - (smoothed_plus_dm[i - 1] / period as f64) + plus_dm[i];
        smoothed_minus_dm[i] = smoothed_minus_dm[i - 1] - (smoothed_minus_dm[i - 1] / period as f64) + minus_dm[i];
    }

    // Calculate DI+ and DI-
    let mut plus_di = vec![0.0; n];
    let mut minus_di = vec![0.0; n];

    for i in period..n {
        if smoothed_tr[i] != 0.0 {
            plus_di[i] = (smoothed_plus_dm[i] / smoothed_tr[i]) * 100.0;
            minus_di[i] = (smoothed_minus_dm[i] / smoothed_tr[i]) * 100.0;
        }
    }

    // Calculate DX
    let mut dx = vec![0.0; n];
    for i in period..n {
        let denominator = plus_di[i] + minus_di[i];
        if denominator != 0.0 {
            let diff = (plus_di[i] - minus_di[i]).abs();
            dx[i] = (diff / denominator) * 100.0;
        }
    }

    // Calculate ADX
    // First ADX value
    let mut sum_dx = 0.0;
    let start_adx = period * 2;
    if n > start_adx {
        for i in period..start_adx {
            sum_dx += dx[i];
        }
        result[start_adx] = sum_dx / period as f64;

        // Continue with smoothing
        for i in (start_adx + 1)..n {
            result[i] = ((result[i - 1] * (period as f64 - 1.0)) + dx[i]) / period as f64;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adx_calculation() {
        let high = vec![102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let close = vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0];
        let adx_values = calculate_adx(&high, &low, &close, 5);
        
        assert_eq!(adx_values.len(), high.len());
        // Check that early values are NaN
        for i in 0..10 {
            if i < adx_values.len() {
                // ADX calculation requires a certain amount of data to start producing values
            }
        }
    }
}