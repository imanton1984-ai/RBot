use std::vec::Vec;

// Calculate the Alligator indicator which consists of three SMAs with different periods and offsets
pub fn calculate_alligator(
    source: &[f64],
    jaw_period: usize,
    teeth_period: usize,
    lips_period: usize,
    jaw_offset: usize,
    teeth_offset: usize,
    lips_offset: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Calculate the three SMAs with respective periods
    let jaw = calculate_sma_with_offset(source, jaw_period, jaw_offset);
    let teeth = calculate_sma_with_offset(source, teeth_period, teeth_offset);
    let lips = calculate_sma_with_offset(source, lips_period, lips_offset);

    (jaw, teeth, lips)
}

fn calculate_sma_with_offset(data: &[f64], period: usize, offset: usize) -> Vec<f64> {
    let mut sma = vec![f64::NAN; data.len()];
    
    if data.len() < period + offset {
        return sma;
    }

    // Calculate SMA normally
    for i in (period - 1)..(data.len() - offset) {
        let sum: f64 = data[(i + 1 - period)..=i].iter().sum();
        sma[i + offset] = sum / period as f64;
    }

    sma
}

// Helper function to calculate plain SMA
#[allow(dead_code)]
fn calculate_sma(data: &[f64], period: usize) -> Vec<f64> {
    let mut sma = vec![f64::NAN; data.len()];

    for i in (period - 1)..data.len() {
        let sum: f64 = data[(i + 1 - period)..=i].iter().sum();
        sma[i] = sum / period as f64;
    }

    sma
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alligator_calculation() {
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 107.0];
        let (jaw, teeth, lips) = calculate_alligator(&prices, 13, 8, 5, 8, 5, 3);
        
        assert_eq!(jaw.len(), prices.len());
        assert_eq!(teeth.len(), prices.len());
        assert_eq!(lips.len(), prices.len());
    }
}