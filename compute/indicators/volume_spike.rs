use std::vec::Vec;

pub fn calculate_volume_spike(
    volumes: &[f64],
    period: usize,
    threshold_factor: f64,  // Factor by which volume exceeds average to be considered a spike
) -> Vec<bool> {
    let n = volumes.len();
    let mut spikes = vec![false; n];

    if n < period {
        return spikes;
    }

    // Calculate average volume for each position
    for i in (period - 1)..n {
        let start_idx = i + 1 - period;
        
        let sum: f64 = volumes[start_idx..i].iter().sum();
        let avg_vol = sum / (period - 1) as f64;  // Exclude current bar from average
        
        // Check if current volume is significantly higher than average
        if avg_vol > 0.0 && volumes[i] > avg_vol * threshold_factor {
            spikes[i] = true;
        }
    }

    spikes
}

// Alternative: Calculate volume percentile-based spikes
pub fn calculate_volume_percentile_spike(
    volumes: &[f64],
    period: usize,
    percentile: f64,  // Percentile threshold (e.g., 0.95 for 95th percentile)
) -> Vec<bool> {
    let n = volumes.len();
    let mut spikes = vec![false; n];

    if n < period {
        return spikes;
    }

    for i in (period - 1)..n {
        let start_idx = i + 1 - period;
        
        // Get the slice of volumes for comparison
        let mut vol_slice: Vec<f64> = volumes[start_idx..=i].to_vec();
        vol_slice.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate the percentile threshold
        let idx = ((percentile * (vol_slice.len() - 1) as f64).round() as usize).min(vol_slice.len() - 1);
        let threshold = vol_slice[idx];
        
        if volumes[i] > threshold {
            spikes[i] = true;
        }
    }

    spikes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_spike_calculation() {
        let volumes = vec![100.0, 120.0, 110.0, 500.0, 130.0, 140.0, 115.0, 600.0];  // Two spikes at indices 3 and 7
        
        let spikes = calculate_volume_spike(&volumes, 3, 2.0);  // Volume must be 2x average
        
        assert_eq!(spikes.len(), volumes.len());
        // The 4th and 8th elements should be spikes due to high volume
        if spikes.len() > 3 {
            assert!(spikes[3]);  // 500 is much higher than previous volumes
        }
        if spikes.len() > 7 {
            assert!(spikes[7]);  // 600 is much higher than previous volumes
        }
    }
}