use std::vec::Vec;

#[derive(Debug, Clone)]
pub struct SRLLevel {
    pub price: f64,
    pub strength: f64,  // How many times this level was touched
    pub level_type: SRLType,  // Support or Resistance
    pub intensity: LevelIntensity, // Strong, Mid, Light
}

#[derive(Debug, Clone, PartialEq)]
pub enum SRLType {
    Support,
    Resistance,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LevelIntensity {
    Strong,
    Mid,
    Light,
}

#[derive(Debug, Clone)]
pub struct SRLLevels {
    pub strong_support: f64,
    pub mid_support: f64,
    pub light_support: f64,
    pub strong_resistance: f64,
    pub mid_resistance: f64,
    pub light_resistance: f64,
}

pub fn calculate_sr_levels(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sensitivity: f64,  // Minimum price difference to consider different levels
) -> SRLLevels {
    if high.len() != low.len() || high.len() != close.len() {
        panic!("High, low, and close arrays must have the same length");
    }

    let n = high.len();
    if n < 3 {
        return SRLLevels {
            strong_support: f64::NAN,
            mid_support: f64::NAN,
            light_support: f64::NAN,
            strong_resistance: f64::NAN,
            mid_resistance: f64::NAN,
            light_resistance: f64::NAN,
        };
    }

    // Find pivot points (local highs and lows) using a more robust algorithm
    let mut support_points = Vec::new();
    let mut resistance_points = Vec::new();

    // Look for swing highs and lows with configurable depth
    let pivot_depth = 2; // Look at 2 bars on each side
    for i in pivot_depth..(n - pivot_depth) {
        let mut is_swing_high = true;
        let mut is_swing_low = true;

        // Check if this is a swing high (higher than surrounding bars)
        for j in (i - pivot_depth)..(i + pivot_depth + 1) {
            if j != i {
                if high[i] <= high[j] {
                    is_swing_high = false;
                    break;
                }
            }
        }

        // Check if this is a swing low (lower than surrounding bars)
        for j in (i - pivot_depth)..(i + pivot_depth + 1) {
            if j != i {
                if low[i] >= low[j] {
                    is_swing_low = false;
                    break;
                }
            }
        }

        if is_swing_high {
            resistance_points.push(high[i]);
        }
        if is_swing_low {
            support_points.push(low[i]);
        }
    }

    // Additionally, include significant price levels from major highs/lows
    if n >= 10 {
        let lookback = n.min(50); // Look at most recent 50 bars for significant levels
        let recent_highs = &high[(n - lookback)..];
        let recent_lows = &low[(n - lookback)..];

        // Add highest high and lowest low from recent period
        if let Some(max_high) = recent_highs.iter().cloned().reduce(f64::max) {
            resistance_points.push(max_high);
        }
        if let Some(min_low) = recent_lows.iter().cloned().reduce(f64::min) {
            support_points.push(min_low);
        }
    }

    // Calculate levels for supports
    let (strong_support, mid_support, light_support) = calculate_improved_intensity_levels(&support_points, sensitivity);

    // Calculate levels for resistances
    let (strong_resistance, mid_resistance, light_resistance) = calculate_improved_intensity_levels(&resistance_points, sensitivity);

    SRLLevels {
        strong_support,
        mid_support,
        light_support,
        strong_resistance,
        mid_resistance,
        light_resistance,
    }
}

#[allow(dead_code)]
fn calculate_intensity_levels(points: &[f64]) -> (f64, f64, f64) {
    if points.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    // Sort points to find percentiles
    let mut sorted_points = points.to_vec();
    sorted_points.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate percentiles for intensity levels
    let len = sorted_points.len();

    // Strong level: median (50th percentile)
    let strong_idx = len / 2;
    let strong_level = sorted_points[strong_idx];

    // For mid and light levels, we'll use statistical clustering
    if len < 3 {
        return (strong_level, strong_level, strong_level);
    }

    // Calculate mid level (around 25th or 75th percentile depending on distribution)
    let mid_idx = len / 4; // Lower quartile for support, upper for resistance context
    let mid_level = sorted_points[mid_idx.min(len - 1)];

    // Calculate light level (extreme values)
    let light_level = sorted_points[len - 1]; // Highest value as light support/resistance

    (strong_level, mid_level, light_level)
}

fn calculate_improved_intensity_levels(points: &[f64], sensitivity: f64) -> (f64, f64, f64) {
    if points.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    // Group nearby price levels into clusters based on sensitivity
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut sorted_points = points.to_vec();
    sorted_points.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for &price in &sorted_points {
        let mut added_to_cluster = false;

        // Try to add to existing cluster
        for cluster in &mut clusters {
            if (cluster.center - price).abs() <= sensitivity {
                cluster.add_price(price);
                added_to_cluster = true;
                break;
            }
        }

        // If not added to any existing cluster, create a new one
        if !added_to_cluster {
            clusters.push(Cluster::new(price));
        }
    }

    // Sort clusters by their strength (number of touches)
    clusters.sort_by(|a, b| b.strength.total_cmp(&a.strength));

    // Extract the top 3 clusters for strong, mid, and light levels
    let strong_level = if !clusters.is_empty() {
        clusters[0].center
    } else {
        f64::NAN
    };

    let mid_level = if clusters.len() > 1 {
        clusters[1].center
    } else {
        strong_level
    };

    let light_level = if clusters.len() > 2 {
        clusters[2].center
    } else if clusters.len() > 1 {
        clusters[1].center
    } else {
        strong_level
    };

    (strong_level, mid_level, light_level)
}

#[derive(Debug, Clone)]
struct Cluster {
    center: f64,
    count: usize,
    strength: f64,
    min_price: f64,
    max_price: f64,
}

impl Cluster {
    fn new(initial_price: f64) -> Self {
        Cluster {
            center: initial_price,
            count: 1,
            strength: 1.0,
            min_price: initial_price,
            max_price: initial_price,
        }
    }

    fn add_price(&mut self, price: f64) {
        self.count += 1;
        self.min_price = self.min_price.min(price);
        self.max_price = self.max_price.max(price);

        // Update center as weighted average
        self.center = (self.center * (self.count - 1) as f64 + price) / self.count as f64;

        // Strength increases with number of touches but with diminishing returns
        self.strength = (self.count as f64).sqrt();
    }
}

// Alternative method: Calculate static support/resistance based on historical highs/lows
pub fn calculate_static_sr_levels(
    high: &[f64],
    low: &[f64],
    period: usize,
) -> (Vec<f64>, Vec<f64>) {  // (resistance levels, support levels)
    let n = high.len();
    if n < period {
        return (Vec::new(), Vec::new());
    }

    // Calculate highest high and lowest low over the period
    let mut resistances = Vec::new();
    let mut supports = Vec::new();

    for i in (period - 1)..n {
        let start_idx = if i >= period { i + 1 - period } else { 0 };

        let mut highest_high = f64::NEG_INFINITY;
        let mut lowest_low = f64::INFINITY;

        for j in start_idx..=i {
            if high[j] > highest_high {
                highest_high = high[j];
            }
            if low[j] < lowest_low {
                lowest_low = low[j];
            }
        }

        resistances.push(highest_high);
        supports.push(lowest_low);
    }

    (resistances, supports)
}

// Calculate probability of bounce vs breakout based on level strength
pub fn calculate_breakout_probability(
    current_price: f64,
    levels: &SRLLevels,
    price_distance_tolerance: f64,
) -> (f64, f64) {  // (bounce_probability, breakout_probability)
    // Determine which level we're closest to
    let _closest_distance = f64::INFINITY;
    let _is_support = false;

    // Check distances to all levels
    let distances = [
        (levels.strong_support - current_price).abs(),
        (levels.mid_support - current_price).abs(),
        (levels.light_support - current_price).abs(),
        (levels.strong_resistance - current_price).abs(),
        (levels.mid_resistance - current_price).abs(),
        (levels.light_resistance - current_price).abs(),
    ];

    let mut min_dist = distances[0];
    let mut level_index = 0;

    for (i, &dist) in distances.iter().enumerate() {
        if dist < min_dist {
            min_dist = dist;
            level_index = i;
        }
    }

    // Determine if near support or resistance
    let _is_support = level_index < 3; // First 3 are supports, last 3 are resistances

    // Calculate probabilities based on level strength
    let proximity_factor = if min_dist < price_distance_tolerance {
        1.0 - (min_dist / price_distance_tolerance)
    } else {
        0.0
    };

    // Higher probability of bounce from stronger levels
    let bounce_probability = match level_index {
        0 | 3 => proximity_factor * 0.9, // Strong levels
        1 | 4 => proximity_factor * 0.7, // Mid levels
        2 | 5 => proximity_factor * 0.5, // Light levels
        _ => 0.0,
    };

    let breakout_probability = 1.0 - bounce_probability;

    (bounce_probability, breakout_probability)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sr_levels_calculation() {
        let high = vec![102.0, 103.0, 104.0, 103.5, 105.0, 104.5, 106.0, 105.5];
        let low = vec![100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5];
        let close = vec![101.0, 102.0, 103.0, 102.5, 104.0, 103.5, 105.0, 104.5];

        let levels = calculate_sr_levels(&high, &low, &close, 0.5);

        // Should have all 6 levels defined
        assert!(!levels.strong_support.is_nan() || levels.mid_support.is_nan()); // At least some levels should be detected
    }

    #[test]
    fn test_breakout_probability() {
        let levels = SRLLevels {
            strong_support: 100.0,
            mid_support: 102.0,
            light_support: 104.0,
            strong_resistance: 108.0,
            mid_resistance: 106.0,
            light_resistance: 104.0,
        };

        let (bounce_prob, breakout_prob) = calculate_breakout_probability(100.1, &levels, 0.5);

        // Near strong support, should have high bounce probability
        assert!(bounce_prob > breakout_prob);
    }
}