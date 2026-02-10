// compute/predictors/level_view.rs

use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::feature_view::{SrLevel, SrLevelKind};

/// View for parsing and working with support/resistance levels
#[derive(Debug, Clone)]
pub struct LevelView {
    pub levels: Vec<ProcessedLevel>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub symbol: String,
    pub timeframe: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedLevel {
    pub level_hash: String,
    pub level_kind: SrLevelKind,
    pub level_price: f64,
    pub level_strength: f32,
    pub distance_atr: f32,
    pub distance_points: f64,
    pub is_near: bool, // True if within certain threshold
}

impl LevelView {
    /// Creates a new level view by parsing SR levels JSON and calculating distances
    pub fn new(
        sr_levels: Vec<SrLevel>,
        current_price: f64,
        atr: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
        symbol: String,
        timeframe: String,
        max_levels_per_side: usize,
    ) -> Result<Self> {
        let mut processed_levels = Vec::new();
        
        // Calculate distances for each level
        for level in sr_levels {
            let distance_points = (current_price - level.price).abs();
            let distance_atr = if atr > 0.0 {
                (distance_points / atr) as f32
            } else {
                f32::MAX
            };
            
            let is_near = distance_atr <= 1.0; // Within 1 ATR
            
            processed_levels.push(ProcessedLevel {
                level_hash: level.hash,
                level_kind: level.kind,
                level_price: level.price,
                level_strength: level.strength,
                distance_atr,
                distance_points,
                is_near,
            });
        }
        
        // Sort levels by distance and take top N for each side
        let mut support_levels = processed_levels
            .iter()
            .filter(|level| level.level_kind == SrLevelKind::Support)
            .cloned()
            .collect::<Vec<_>>();
        
        let mut resistance_levels = processed_levels
            .iter()
            .filter(|level| level.level_kind == SrLevelKind::Resistance)
            .cloned()
            .collect::<Vec<_>>();
        
        // Sort by distance (closest first)
        support_levels.sort_by(|a, b| a.distance_atr.partial_cmp(&b.distance_atr).unwrap());
        resistance_levels.sort_by(|a, b| a.distance_atr.partial_cmp(&b.distance_atr).unwrap());
        
        // Take top N levels for each side
        support_levels.truncate(max_levels_per_side);
        resistance_levels.truncate(max_levels_per_side);
        
        // Combine the levels
        let mut combined_levels = Vec::new();
        combined_levels.extend(support_levels);
        combined_levels.extend(resistance_levels);
        
        // Sort by distance again
        combined_levels.sort_by(|a, b| a.distance_atr.partial_cmp(&b.distance_atr).unwrap());
        
        Ok(LevelView {
            levels: combined_levels,
            timestamp,
            symbol,
            timeframe,
        })
    }
    
    /// Gets the closest support level
    pub fn get_closest_support(&self) -> Option<&ProcessedLevel> {
        self.levels
            .iter()
            .filter(|level| level.level_kind == SrLevelKind::Support)
            .min_by(|a, b| a.distance_atr.partial_cmp(&b.distance_atr).unwrap())
    }
    
    /// Gets the closest resistance level
    pub fn get_closest_resistance(&self) -> Option<&ProcessedLevel> {
        self.levels
            .iter()
            .filter(|level| level.level_kind == SrLevelKind::Resistance)
            .min_by(|a, b| a.distance_atr.partial_cmp(&b.distance_atr).unwrap())
    }
    
    /// Gets all near levels (within 1 ATR)
    pub fn get_near_levels(&self) -> Vec<&ProcessedLevel> {
        self.levels.iter().filter(|level| level.is_near).collect()
    }
    
    /// Gets levels by kind
    pub fn get_levels_by_kind(&self, kind: SrLevelKind) -> Vec<&ProcessedLevel> {
        self.levels
            .iter()
            .filter(|level| level.level_kind == kind)
            .collect()
    }
    
    /// Finds levels within a certain ATR distance
    pub fn get_levels_within_atr(&self, max_atr_distance: f32) -> Vec<&ProcessedLevel> {
        self.levels
            .iter()
            .filter(|level| level.distance_atr <= max_atr_distance)
            .collect()
    }
    
    /// Calculates the probability of bounce/break for a given level based on various factors
    pub fn calculate_level_probability(
        &self,
        level: &ProcessedLevel,
        trend_strength: f64,
        momentum: f64,
        exhaustion: f64,
        volume_factor: f64,
    ) -> (f64, f64) { // (bounce_prob, break_prob)
        // Base probability influenced by level strength
        let base_prob = level.level_strength as f64;
        
        // Calculate bounce probability
        let bounce_prob = {
            let mut prob = base_prob;
            
            // Higher bounce probability for strong levels
            prob *= 0.7; // Base factor
            
            // If trend is weak, bounce is more likely
            prob *= (1.0 - trend_strength.abs()).max(0.1);
            
            // If momentum is exhausted (overbought/oversold), bounce is more likely
            prob *= exhaustion.max(0.1);
            
            // If volume is high approaching the level, bounce is more likely
            prob *= volume_factor.max(0.5);
            
            // Closer levels have higher probability
            let distance_factor = (1.0 / (level.distance_atr as f64 + 0.1)).min(2.0);
            prob *= distance_factor;
            
            prob.clamp(0.0, 1.0)
        };
        
        // Calculate break probability
        let break_prob = {
            let mut prob = base_prob * 0.3; // Lower base for breaks
            
            // If trend is strong in the direction of break, increase probability
            let trend_factor = if level.level_kind == SrLevelKind::Support {
                // For support, break happens when trending down strongly
                if trend_strength < 0.0 {
                    (trend_strength.abs() * 1.5).min(2.0)
                } else {
                    0.5 // Reduce if trend is opposite
                }
            } else {
                // For resistance, break happens when trending up strongly
                if trend_strength > 0.0 {
                    (trend_strength * 1.5).min(2.0)
                } else {
                    0.5 // Reduce if trend is opposite
                }
            };
            
            prob *= trend_factor;
            
            // High momentum increases break probability
            prob *= momentum.abs().max(0.5);
            
            // If volume is high in the direction of break
            prob *= volume_factor.max(0.5);
            
            // Closer levels have higher probability
            let distance_factor = (1.0 / (level.distance_atr as f64 + 0.1)).min(1.5);
            prob *= distance_factor;
            
            prob.clamp(0.0, 1.0)
        };
        
        (bounce_prob, break_prob)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_level_view_creation() {
        let sr_levels = vec![
            SrLevel {
                price: 95.0,
                kind: SrLevelKind::Support,
                strength: 0.8,
                hash: "support_95".to_string(),
            },
            SrLevel {
                price: 105.0,
                kind: SrLevelKind::Resistance,
                strength: 0.9,
                hash: "resistance_105".to_string(),
            },
            SrLevel {
                price: 90.0,
                kind: SrLevelKind::Support,
                strength: 0.6,
                hash: "support_90".to_string(),
            },
        ];
        
        let timestamp = chrono::Utc::now();
        let level_view = LevelView::new(
            sr_levels,
            100.0, // current price
            2.0,   // ATR
            timestamp,
            "BTCUSDT".to_string(),
            "5m".to_string(),
            2, // max levels per side
        ).expect("Failed to create level view");
        
        assert!(!level_view.levels.is_empty());
        
        // Check that we have both support and resistance levels
        let supports: Vec<_> = level_view.get_levels_by_kind(SrLevelKind::Support);
        let resistances: Vec<_> = level_view.get_levels_by_kind(SrLevelKind::Resistance);
        
        assert!(!supports.is_empty());
        assert!(!resistances.is_empty());
        
        // Check distances are calculated correctly
        for level in &level_view.levels {
            assert!(level.distance_points >= 0.0);
            assert!(level.distance_atr >= 0.0);
        }
    }
    
    #[test]
    fn test_closest_levels() {
        let sr_levels = vec![
            SrLevel {
                price: 99.0, // Closest support
                kind: SrLevelKind::Support,
                strength: 0.8,
                hash: "support_99".to_string(),
            },
            SrLevel {
                price: 101.0, // Closest resistance
                kind: SrLevelKind::Resistance,
                strength: 0.9,
                hash: "resistance_101".to_string(),
            },
            SrLevel {
                price: 95.0, // Further support
                kind: SrLevelKind::Support,
                strength: 0.6,
                hash: "support_95".to_string(),
            },
        ];
        
        let timestamp = chrono::Utc::now();
        let level_view = LevelView::new(
            sr_levels,
            100.0, // current price
            2.0,   // ATR
            timestamp,
            "BTCUSDT".to_string(),
            "5m".to_string(),
            2,
        ).expect("Failed to create level view");
        
        let closest_support = level_view.get_closest_support();
        let closest_resistance = level_view.get_closest_resistance();
        
        assert!(closest_support.is_some());
        assert!(closest_resistance.is_some());
        
        if let Some(closest_sup) = closest_support {
            assert_eq!(closest_sup.level_price, 99.0);
        }
        
        if let Some(closest_res) = closest_resistance {
            assert_eq!(closest_res.level_price, 101.0);
        }
    }
    
    #[test]
    fn test_level_probability_calculation() {
        let sr_levels = vec![
            SrLevel {
                price: 95.0,
                kind: SrLevelKind::Support,
                strength: 0.8,
                hash: "support_95".to_string(),
            },
        ];
        
        let timestamp = chrono::Utc::now();
        let level_view = LevelView::new(
            sr_levels,
            96.0, // current price (close to support)
            2.0,  // ATR
            timestamp,
            "BTCUSDT".to_string(),
            "5m".to_string(),
            2,
        ).expect("Failed to create level view");
        
        if let Some(level) = level_view.levels.first() {
            let (bounce_prob, break_prob) = level_view.calculate_level_probability(
                level,
                -0.3, // Weak downtrend
                -0.2, // Slight negative momentum
                0.8,  // High exhaustion (oversold)
                1.2,  // High volume
            );
            
            // Both probabilities should be between 0 and 1
            assert!(bounce_prob >= 0.0 && bounce_prob <= 1.0);
            assert!(break_prob >= 0.0 && break_prob <= 1.0);
        }
    }
}