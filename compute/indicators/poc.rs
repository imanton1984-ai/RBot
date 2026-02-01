use std::vec::Vec;
use std::collections::HashMap;

pub fn calculate_poc(prices: &[f64], volumes: &[f64], bucket_size: f64) -> f64 {
    if prices.len() != volumes.len() {
        panic!("Prices and volumes arrays must have the same length");
    }

    // Create price buckets and accumulate volumes
    let mut volume_by_price: HashMap<i64, f64> = HashMap::new();

    for i in 0..prices.len() {
        // Round price to nearest bucket
        let bucket_key = (prices[i] / bucket_size).round() as i64;
        let entry = volume_by_price.entry(bucket_key).or_insert(0.0);
        *entry += volumes[i];
    }

    // Find the bucket with the highest volume
    let mut max_volume = 0.0;
    let mut poc_bucket = 0i64;

    for (bucket, volume) in volume_by_price.iter() {
        if *volume > max_volume {
            max_volume = *volume;
            poc_bucket = *bucket;
        }
    }

    // Convert bucket back to price
    poc_bucket as f64 * bucket_size
}

// For time-based POC calculation (Volume Profile)
pub fn calculate_poc_time_based(
    prices: &[f64],
    volumes: &[f64],
    timestamps: &[i64],
    time_window_minutes: i64,
) -> Vec<(f64, f64)> { // Returns (price, volume) pairs for each time window
    if prices.len() != volumes.len() || prices.len() != timestamps.len() {
        panic!("Prices, volumes, and timestamps arrays must have the same length");
    }

    let mut result = Vec::new();
    
    if prices.is_empty() {
        return result;
    }

    // Group data by time windows
    let start_time = timestamps[0];
    let end_time = *timestamps.iter().max().unwrap_or(&start_time);
    let num_windows = ((end_time - start_time) / (time_window_minutes * 60 * 1000)) + 1;

    for window_idx in 0..num_windows {
        let window_start = start_time + (window_idx * time_window_minutes * 60 * 1000);
        let window_end = window_start + (time_window_minutes * 60 * 1000);

        // Collect data for this window
        let mut window_prices = Vec::new();
        let mut window_volumes = Vec::new();

        for i in 0..timestamps.len() {
            if timestamps[i] >= window_start && timestamps[i] < window_end {
                window_prices.push(prices[i]);
                window_volumes.push(volumes[i]);
            }
        }

        if !window_prices.is_empty() {
            let poc = calculate_poc(&window_prices, &window_volumes, 0.1); // Using 0.1 as bucket size
            result.push((poc, window_volumes.iter().sum())); // Return POC and total volume in window
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poc_calculation() {
        let prices = vec![100.0, 101.0, 100.5, 101.0, 102.0, 101.0];
        let volumes = vec![100.0, 200.0, 150.0, 300.0, 50.0, 250.0];
        let poc = calculate_poc(&prices, &volumes, 0.5);
        
        // Price 101.0 appears 3 times with volumes 200, 300, 250 = 750 total
        // So POC should be around 101.0
        assert!((poc - 101.0).abs() < 0.5); // Within bucket size
    }
}