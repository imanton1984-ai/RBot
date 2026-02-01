use std::vec::Vec;

pub fn calculate_vwap(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> Vec<f64> {
    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        panic!("High, low, close, and volume arrays must have the same length");
    }

    let n = high.len();
    let mut vwap = vec![f64::NAN; n];

    if n == 0 {
        return vwap;
    }

    // Calculate typical price and cumulative values
    let mut cum_qty = 0.0;
    let mut cum_bp = 0.0;

    for i in 0..n {
        // Typical price = (high + low + close) / 3
        let typ_price = (high[i] + low[i] + close[i]) / 3.0;
        
        // Cumulative values
        cum_qty += volume[i];
        cum_bp += typ_price * volume[i];
        
        if cum_qty > 0.0 {
            vwap[i] = cum_bp / cum_qty;
        }
    }

    vwap
}

// VWAP with reset periods (e.g., daily VWAP)
pub fn calculate_periodic_vwap(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timestamps: &[i64],
    period_hours: i64,
) -> Vec<f64> {
    if high.len() != low.len() || high.len() != close.len() || 
       high.len() != volume.len() || high.len() != timestamps.len() {
        panic!("All arrays must have the same length");
    }

    let n = high.len();
    let mut vwap = vec![f64::NAN; n];

    if n == 0 {
        return vwap;
    }

    let mut start_idx = 0;
    let period_ms = period_hours * 60 * 60 * 1000; // Convert hours to milliseconds

    for i in 0..n {
        // Check if we need to reset (new period)
        if i > 0 && (timestamps[i] - timestamps[start_idx]) >= period_ms {
            // Find the start of the new period
            start_idx = i;
        }

        // Calculate VWAP for the current period
        let mut cum_qty = 0.0;
        let mut cum_bp = 0.0;

        for j in start_idx..=i {
            let typ_price = (high[j] + low[j] + close[j]) / 3.0;
            cum_qty += volume[j];
            cum_bp += typ_price * volume[j];
        }

        if cum_qty > 0.0 {
            vwap[i] = cum_bp / cum_qty;
        }
    }

    vwap
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vwap_calculation() {
        let high = vec![102.0, 103.0, 104.0, 105.0];
        let low = vec![100.0, 101.0, 102.0, 103.0];
        let close = vec![101.0, 102.0, 103.0, 104.0];
        let volume = vec![100.0, 150.0, 200.0, 250.0];
        
        let vwap_values = calculate_vwap(&high, &low, &close, &volume);
        
        assert_eq!(vwap_values.len(), high.len());
        // Check that first value is calculated (though it equals the first typical price)
        if !vwap_values[0].is_nan() {
            let first_typ = (high[0] + low[0] + close[0]) / 3.0;
            assert!((vwap_values[0] - first_typ).abs() < 0.001);
        }
    }
}