// compute/predictors/level_view.rs

use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::feature_view::{SrLevel, SrLevelKind};

/// Optional candle-data context for enriching ProcessedLevel with
/// touch_count, bars_since_last_touch, volume_at_level, approach_velocity.
/// When `None` is passed to [`LevelView::new`], the enrichment fields default
/// to neutral values so the old (level-only) behaviour is preserved.
pub struct CandleContext<'a> {
    pub close_prices: &'a [f64],
    pub volumes: &'a [f64],
    pub high_prices: &'a [f64],
    pub low_prices: &'a [f64],
}

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

    // ── Level-enrichment features ──────────────────────────────────────
    /// How many bars had price within 0.2 × ATR of this level.
    pub touch_count: u32,
    /// Bars elapsed since the last touch (0 = currently touching).
    pub bars_since_last_touch: u32,
    /// Total volume traded on bars that were within 0.2 × ATR of the level.
    pub volume_at_level: f64,
    /// Rate of price approach: positive ⇒ moving toward the level fast.
    /// Calculated as the simple-regression slope of the last 5 closes ÷ ATR.
    pub approach_velocity: f32,
}

impl LevelView {
    /// Creates a new level view by parsing SR levels JSON and calculating distances.
    ///
    /// `candle_ctx` is **optional** — pass `None` when historical candle arrays
    /// are unavailable and the enrichment fields will be left at safe defaults.
    pub fn new(
        sr_levels: Vec<SrLevel>,
        current_price: f64,
        atr: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
        symbol: String,
        timeframe: String,
        max_levels_per_side: usize,
        candle_ctx: Option<CandleContext<'_>>,
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

            let is_near = distance_atr <= 3.0; // Within 3 ATR

            // ── enrichment ─────────────────────────────────────────────
            let (touch_count, bars_since_last_touch, volume_at_level, approach_velocity) =
                if let Some(ref ctx) = candle_ctx {
                    Self::compute_enrichment(level.price, atr, ctx)
                } else {
                    (0, u32::MAX, 0.0, 0.0)
                };

            processed_levels.push(ProcessedLevel {
                level_hash: level.hash,
                level_kind: level.kind,
                level_price: level.price,
                level_strength: level.strength,
                distance_atr,
                distance_points,
                is_near,
                touch_count,
                bars_since_last_touch,
                volume_at_level,
                approach_velocity,
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

    // ── Private helpers for enrichment computation ─────────────────────

    /// Computes (touch_count, bars_since_last_touch, volume_at_level, approach_velocity)
    /// for a single level given the candle context.
    fn compute_enrichment(
        level_price: f64,
        atr: f64,
        ctx: &CandleContext<'_>,
    ) -> (u32, u32, f64, f32) {
        let threshold = 0.2 * atr;
        let n = ctx.close_prices.len()
            .min(ctx.volumes.len())
            .min(ctx.high_prices.len())
            .min(ctx.low_prices.len());

        let mut touch_count: u32 = 0;
        let mut last_touch_idx: Option<usize> = None;
        let mut volume_at_level: f64 = 0.0;

        for i in 0..n {
            let bar_low = ctx.low_prices[i];
            let bar_high = ctx.high_prices[i];

            // A bar "touches" the level when the bar's price range comes
            // within `threshold` of the level price.
            let nearest_bar_price = level_price.clamp(bar_low, bar_high);
            if (nearest_bar_price - level_price).abs() <= threshold {
                touch_count += 1;
                last_touch_idx = Some(i);
                volume_at_level += ctx.volumes[i];
            }
        }

        let bars_since_last_touch = match last_touch_idx {
            Some(idx) => (n.saturating_sub(1).saturating_sub(idx)) as u32,
            None => u32::MAX,
        };

        // approach_velocity: simple linear-regression slope of the last 5 closes
        // divided by ATR.  Positive = moving toward the level.
        let approach_velocity = Self::compute_approach_velocity(
            ctx.close_prices,
            level_price,
            atr,
        );

        (touch_count, bars_since_last_touch, volume_at_level, approach_velocity)
    }

    /// Linear-regression slope over the last `window` close prices, divided by ATR.
    /// The sign is flipped so that a positive value means "approaching the level".
    fn compute_approach_velocity(closes: &[f64], level_price: f64, atr: f64) -> f32 {
        const WINDOW: usize = 5;
        if closes.len() < 2 || atr <= 0.0 {
            return 0.0;
        }
        let tail = if closes.len() >= WINDOW {
            &closes[closes.len() - WINDOW..]
        } else {
            closes
        };
        let n = tail.len() as f64;
        // Simple OLS slope:  slope = (n·Σ(x·y) − Σx·Σy) / (n·Σ(x²) − (Σx)²)
        let mut sum_x: f64 = 0.0;
        let mut sum_y: f64 = 0.0;
        let mut sum_xy: f64 = 0.0;
        let mut sum_x2: f64 = 0.0;
        for (i, &y) in tail.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return 0.0;
        }
        let raw_slope = (n * sum_xy - sum_x * sum_y) / denom;

        // Positive raw_slope means price is rising.
        // If level is *above* current price, rising means approaching → keep positive.
        // If level is *below* current price, rising means moving away → flip sign.
        let last_close = *tail.last().unwrap_or(&0.0);
        let direction = if level_price >= last_close { 1.0 } else { -1.0 };

        ((raw_slope * direction) / atr) as f32
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
            None,
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
            // Without candle context, enrichment should be defaults
            assert_eq!(level.touch_count, 0);
            assert_eq!(level.bars_since_last_touch, u32::MAX);
            assert_eq!(level.volume_at_level, 0.0);
            assert_eq!(level.approach_velocity, 0.0);
        }
    }

    #[test]
    fn test_level_view_with_candle_context() {
        let sr_levels = vec![
            SrLevel {
                price: 100.0,
                kind: SrLevelKind::Support,
                strength: 0.8,
                hash: "support_100".to_string(),
            },
        ];

        // Simulate 10 bars; bars 3,4,5 touch the level at 100.0 (within 0.2*ATR=0.4)
        let close_prices = vec![102.0, 101.5, 101.0, 100.1, 100.2, 100.3, 101.0, 101.5, 102.0, 102.5];
        let high_prices  = vec![102.5, 102.0, 101.5, 100.3, 100.4, 100.5, 101.5, 102.0, 102.5, 103.0];
        let low_prices   = vec![101.5, 101.0, 100.5, 99.9,  100.0, 100.1, 100.5, 101.0, 101.5, 102.0];
        let volumes      = vec![100.0, 110.0, 120.0, 200.0, 180.0, 190.0, 130.0, 120.0, 110.0, 100.0];

        let ctx = CandleContext {
            close_prices: &close_prices,
            volumes: &volumes,
            high_prices: &high_prices,
            low_prices: &low_prices,
        };

        let timestamp = chrono::Utc::now();
        let level_view = LevelView::new(
            sr_levels,
            102.5, // current price (last close)
            2.0,   // ATR => threshold = 0.4
            timestamp,
            "BTCUSDT".to_string(),
            "5m".to_string(),
            2,
            Some(ctx),
        ).expect("Failed to create level view");

        let level = &level_view.levels[0];
        assert!(level.touch_count >= 3, "Expected at least 3 touches, got {}", level.touch_count);
        assert!(level.bars_since_last_touch < 10);
        assert!(level.volume_at_level > 0.0);
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
            None,
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
            None,
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

    #[test]
    fn test_approach_velocity_toward_level() {
        // Price rising toward resistance at 110
        let closes = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let vel = LevelView::compute_approach_velocity(&closes, 110.0, 2.0);
        assert!(vel > 0.0, "Velocity should be positive when approaching: {}", vel);
    }

    #[test]
    fn test_approach_velocity_away_from_level() {
        // Price rising away from support at 90
        let closes = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let vel = LevelView::compute_approach_velocity(&closes, 90.0, 2.0);
        assert!(vel < 0.0, "Velocity should be negative when moving away: {}", vel);
    }
}
