// compute/src/predictions/scoring.rs

pub fn normalize_score(probability: f64) -> f64 {
    probability.max(0.0).min(1.0)
}
