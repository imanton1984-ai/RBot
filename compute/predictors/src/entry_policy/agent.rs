// compute/predictors/src/entry_policy/agent.rs
//
// Entry Agent — Real-time entry timing decision maker
//
// The Entry Agent uses two XGBoost models to make ENTER/WAIT/CANCEL decisions:
//   - entry_enter: Probability that NOW is the optimal entry bar
//   - entry_cancel: Probability that this setup should be cancelled
//
// Decision logic:
//   1. If prob_cancel >= cancel_thr AND prob_cancel > prob_enter → CANCEL
//   2. If prob_enter >= enter_thr AND (prob_enter - prob_cancel) >= min_margin → ENTER
//   3. Otherwise → WAIT
//
// This allows the agent to:
//   - Wait for better entry conditions instead of entering immediately
//   - Cancel setups that are no longer valid (e.g., market conditions changed)
//   - Enter only when probability/reward is maximal

use anyhow::Result;
use crate::ml::model_manager::ModelManager;

/// Entry decision from the agent
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EntryDecision {
    /// Enter the trade now
    Enter {
        /// Confidence (probability from model)
        confidence: f32,
    },
    /// Wait for next bar
    Wait {
        /// Confidence (1 - max(prob_enter, prob_cancel))
        confidence: f32,
    },
    /// Cancel this setup (don't enter)
    Cancel {
        /// Confidence (probability from model)
        confidence: f32,
    },
}

impl EntryDecision {
    /// Check if decision is Enter
    pub fn is_enter(&self) -> bool {
        matches!(self, EntryDecision::Enter { .. })
    }

    /// Check if decision is Wait
    pub fn is_wait(&self) -> bool {
        matches!(self, EntryDecision::Wait { .. })
    }

    /// Check if decision is Cancel
    pub fn is_cancel(&self) -> bool {
        matches!(self, EntryDecision::Cancel { .. })
    }

    /// Get confidence value
    pub fn confidence(&self) -> f32 {
        match self {
            EntryDecision::Enter { confidence } => *confidence,
            EntryDecision::Wait { confidence } => *confidence,
            EntryDecision::Cancel { confidence } => *confidence,
        }
    }
}

/// Configuration for Entry Agent
#[derive(Clone, Copy, Debug)]
pub struct EntryAgentConfig {
    /// Minimum probability threshold for ENTER
    pub enter_threshold: f32,
    /// Minimum probability threshold for CANCEL
    pub cancel_threshold: f32,
    /// Minimum margin between prob_enter and prob_cancel to trigger ENTER
    pub min_margin: f32,
    /// Default window size (bars to wait for entry)
    pub default_window_bars: usize,
}

impl Default for EntryAgentConfig {
    fn default() -> Self {
        Self {
            // V5: DRACONIAN thresholds based on V4.1 backtest failure analysis
            // Problem: V4.1 filtered too little (97% entry on 4h!) → massive losses
            // Solution: Much higher base thresholds + dynamic RR/quality adjustments
            //
            // Base thresholds (overridden per TF via get_*_for_tf methods):
            enter_threshold: 0.65,   // Base (TF-specific: 0.55-0.75)
            cancel_threshold: 0.45,  // Aggressive cancel
            min_margin: 0.20,        // Require clear signal
            default_window_bars: 4,
        }
    }
}

/// Entry Agent — makes real-time entry timing decisions
///
/// # Example
///
/// ```rust,no_run
/// use predictors::entry_policy::{EntryAgent, EntryAgentConfig};
///
/// let config = EntryAgentConfig {
///     enter_threshold: 0.55,
///     cancel_threshold: 0.5,
///     min_margin: 0.15,
///     default_window_bars: 10,
/// };
///
/// let agent = EntryAgent::new(config);
///
/// // On each bar while setup is active:
/// let decision = agent.decide(&model_manager, tf_minutes, &features, elapsed, remaining, use_gpu)?;
///
/// match decision {
///     EntryDecision::Enter { confidence } => { /* Execute trade */ }
///     EntryDecision::Wait { .. } => { /* Wait for next bar */ }
///     EntryDecision::Cancel { .. } => { /* Abort setup */ }
/// }
/// ```
pub struct EntryAgent {
    config: EntryAgentConfig,
}

impl EntryAgent {
    /// Create a new Entry Agent with the given configuration
    pub fn new(config: EntryAgentConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(EntryAgentConfig::default())
    }

    /// Make an entry decision based on current bar features
    ///
    /// # Arguments
    ///
    /// * `mm` - ModelManager with loaded entry_enter and entry_cancel models
    /// * `tf_minutes` - Timeframe in minutes (1, 5, 15, 60, 240)
    /// * `features` - Feature vector (must match model's expected schema)
    /// * `elapsed` - Bars elapsed since setup started
    /// * `remaining` - Bars remaining in entry window
    /// * `use_gpu` - Whether to use GPU for inference
    ///
    /// # Returns
    ///
    /// EntryDecision: Enter, Wait, or Cancel with confidence scores
    pub fn decide(
        &self,
        mm: &ModelManager,
        tf_minutes: i32,
        mut features: Vec<f32>,
        elapsed: u16,
        remaining: u16,
        use_gpu: bool,
    ) -> Result<EntryDecision> {
        // Add temporal features (must match training schema)
        features.push(elapsed as f32);
        features.push(remaining as f32);

        let ncol = features.len();

        // Model keys
        let key_enter = format!("entry_enter_tf{}", tf_minutes);
        let key_cancel = format!("entry_cancel_tf{}", tf_minutes);

        // Get probabilities from both models
        let prob_enter = mm
            .predict_batch(&key_enter, &features, 1, ncol, use_gpu)?
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        let prob_cancel = mm
            .predict_batch(&key_cancel, &features, 1, ncol, use_gpu)?
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        // Decision logic
        let decision = self.make_decision(prob_enter, prob_cancel);

        Ok(decision)
    }

    /// Core decision logic (separated for testing)
    fn make_decision(&self, prob_enter: f32, prob_cancel: f32) -> EntryDecision {
        // CANCEL: high cancel probability AND cancel > enter
        if prob_cancel >= self.config.cancel_threshold && prob_cancel > prob_enter {
            return EntryDecision::Cancel {
                confidence: prob_cancel,
            };
        }

        // ENTER: high enter probability AND sufficient margin over cancel
        if prob_enter >= self.config.enter_threshold
            && (prob_enter - prob_cancel) >= self.config.min_margin
        {
            return EntryDecision::Enter {
                confidence: prob_enter,
            };
        }

        // WAIT: neither condition met
        EntryDecision::Wait {
            confidence: (1.0 - prob_enter.max(prob_cancel)).max(0.0),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &EntryAgentConfig {
        &self.config
    }

    /// Get window size for a given timeframe (matches EntryPolicyConfig::for_timeframe)
    ///
    /// V5: Optimized based on V4.1 backtest failure analysis
    /// 
    /// Key insight: V4.1 allowed 97% entry on 4h → catastrophic losses
    /// Solution: Longer windows for confirmation on high TFs
    pub fn get_window_bars_for_tf(tf_minutes: i32) -> usize {
        match tf_minutes {
            1 => 4,    // 4 min (scalping, quick decision)
            5 => 4,    // 20 min (early entry)
            15 => 5,   // V5: 75 min (was 4) — more time for confirmation
            60 => 6,   // 6 hours (confirmation needed)
            240 => 4,  // V5: 16 hours (was 3) — more patience on 4h
            _ => 4,    // default
        }
    }

    /// Get max hold bars for a given timeframe (matches EntryPolicyConfig::for_timeframe)
    ///
    /// V5: Extended for TP3 capture on high TFs
    pub fn get_max_hold_bars_for_tf(tf_minutes: i32) -> usize {
        match tf_minutes {
            1 => 12,   // V5: 12 min (was 14) — cut losses faster
            5 => 14,   // V5: 70 min (was 16) — balance
            15 => 14,  // V5: 210 min (was 12) — catch TP3
            60 => 12,  // V5: 12 hours (was 10) — more time
            240 => 12, // V5: 48 hours (was 10) — catch full moves
            _ => 12,   // default
        }
    }

    /// Get timeframe-specific enter threshold
    /// V5: DRACONIAN filtering based on V4.1 backtest disaster
    /// 
    /// V4.1 Problem: 4h entered 97% of signals → -0.11% PnL
    /// V5 Solution: Much higher thresholds, especially on 4h/15m
    pub fn get_enter_threshold_for_tf(tf_minutes: i32) -> f32 {
        match tf_minutes {
            1 => 0.52,   // V5: Reduced from 0.56 — 1m needs early entry
            5 => 0.58,   // V5: Increased from 0.57 — more selective
            15 => 0.68,  // V5: DRACONIAN! Was 0.55 → filter 60%+ signals
            60 => 0.70,  // V5: Increased from 0.68 — proven working
            240 => 0.78, // V5: EXTREME! Was 0.65 → filter 70%+ signals
            _ => 0.62,
        }
    }

    /// Get timeframe-specific cancel threshold
    /// V5: More aggressive cancel on problematic TFs
    pub fn get_cancel_threshold_for_tf(tf_minutes: i32) -> f32 {
        match tf_minutes {
            1 => 0.32,   // V5: Reduced from 0.38 — cancel weak scalps fast
            5 => 0.36,   // V5: Reduced from 0.40 — earlier cancel
            15 => 0.44,  // V5: Increased from 0.40 — hold quality setups
            60 => 0.48,  // V5: Increased from 0.46 — proven working
            240 => 0.52, // V5: Increased from 0.45 — cancel weak 4h
            _ => 0.42,
        }
    }

    /// Get timeframe-specific min margin
    /// V5: Require clear signal, especially on high TFs
    pub fn get_min_margin_for_tf(tf_minutes: i32) -> f32 {
        match tf_minutes {
            1 => 0.10,   // V5: Reduced from 0.15 — accept smaller margin for scalps
            5 => 0.12,   // V5: Reduced from 0.16 — earlier entry
            15 => 0.20,  // V5: Increased from 0.16 — require clarity
            60 => 0.24,  // V5: Unchanged — proven working
            240 => 0.30, // V5: Increased from 0.25 — extreme clarity needed
            _ => 0.18,
        }
    }

    /// Get timeframe-specific sl_atr_mult
    /// V5: Tighter on low TFs, wider on high TFs
    pub fn get_sl_atr_mult_for_tf(tf_minutes: i32) -> f64 {
        match tf_minutes {
            1 => 0.65,   // V5: Reduced from 0.70 — very tight for scalps
            5 => 0.70,   // V5: Reduced from 0.75 — tighter
            15 => 0.80,  // V5: Unchanged — working
            60 => 0.90,  // V5: Unchanged — working
            240 => 1.00, // V5: Increased from 0.95 — wider for volatile 4h
            _ => 0.80,
        }
    }

    /// V5: DYNAMIC threshold adjustment based on risk_reward
    /// Call this to get final threshold after RR adjustment
    ///
    /// Training insight: risk_reward = 42.5% gain (MOST IMPORTANT!)
    pub fn adjust_threshold_for_rr(base_threshold: f32, risk_reward: f64) -> f32 {
        let adjustment = if risk_reward >= 3.0 {
            -0.12  // Excellent RR → much earlier entry
        } else if risk_reward >= 2.0 {
            -0.08  // Good RR → earlier entry
        } else if risk_reward >= 1.5 {
            -0.04  // Decent RR → slightly earlier
        } else if risk_reward < 0.8 {
            0.10  // Terrible RR → reject
        } else if risk_reward < 1.0 {
            0.05  // Poor RR → more selective
        } else {
            0.0
        };
        (base_threshold + adjustment).clamp(0.35, 0.85)
    }

    /// V5: DYNAMIC threshold adjustment based on quality_grade
    /// Call this to get final threshold after quality adjustment
    ///
    /// Training insight: quality_grade = 13-32% gain on 15m/4h
    pub fn adjust_threshold_for_quality(base_threshold: f32, quality_grade: f64) -> f32 {
        let adjustment = if quality_grade >= 9.0 {
            -0.10  // Excellent quality → earlier entry
        } else if quality_grade >= 7.0 {
            -0.05  // Good quality → slightly earlier
        } else if quality_grade <= 3.0 {
            0.08  // Terrible quality → reject
        } else if quality_grade <= 5.0 {
            0.04  // Poor quality → more selective
        } else {
            0.0
        };
        (base_threshold + adjustment).clamp(0.35, 0.85)
    }

    /// V5: Combined dynamic adjustment (RR + quality)
    pub fn get_dynamic_enter_threshold(tf_minutes: i32, risk_reward: f64, quality_grade: f64) -> f32 {
        let base = Self::get_enter_threshold_for_tf(tf_minutes);
        let rr_adjusted = Self::adjust_threshold_for_rr(base, risk_reward);
        Self::adjust_threshold_for_quality(rr_adjusted, quality_grade)
    }

    /// V5: Dynamic cancel threshold adjustment based on price movement
    /// Training insight: price10_score critical for 1h/4h cancel decisions
    pub fn adjust_cancel_threshold_for_price(
        base_threshold: f32,
        tf_minutes: i32,
        price10_score: f32,
    ) -> f32 {
        if tf_minutes >= 60 && price10_score < 0.25 {
            // No price movement on high TF → cancel aggressively
            (base_threshold - 0.10).clamp(0.25, 0.70)
        } else {
            base_threshold
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_cancel() {
        let agent = EntryAgent::with_defaults();

        // High cancel, low enter → CANCEL
        let decision = agent.make_decision(0.2, 0.8);
        assert!(decision.is_cancel());
        assert!((decision.confidence() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_decision_enter() {
        let agent = EntryAgent::with_defaults();

        // High enter, low cancel, good margin → ENTER
        let decision = agent.make_decision(0.8, 0.2);
        assert!(decision.is_enter());
        assert!((decision.confidence() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_decision_wait() {
        let agent = EntryAgent::with_defaults();

        // Both low → WAIT
        let decision = agent.make_decision(0.3, 0.2);
        assert!(decision.is_wait());
    }

    #[test]
    fn test_decision_margin_check() {
        let agent = EntryAgent::with_defaults();

        // Enter above threshold but margin too small → WAIT (0.70 - 0.50 = 0.20 < 0.25)
        let decision = agent.make_decision(0.70, 0.50);
        assert!(decision.is_wait());

        // Enter above threshold with good margin → ENTER (0.80 - 0.50 = 0.30 >= 0.25)
        let decision = agent.make_decision(0.80, 0.50);
        assert!(decision.is_enter());
    }

    #[test]
    fn test_window_bars_by_tf() {
        // V5: Optimized values
        assert_eq!(EntryAgent::get_window_bars_for_tf(1), 4);
        assert_eq!(EntryAgent::get_window_bars_for_tf(5), 4);
        assert_eq!(EntryAgent::get_window_bars_for_tf(15), 5);  // V5: increased
        assert_eq!(EntryAgent::get_window_bars_for_tf(60), 6);
        assert_eq!(EntryAgent::get_window_bars_for_tf(240), 4);  // V5: increased
    }

    #[test]
    fn test_max_hold_bars_by_tf() {
        // V5: Extended for TP3
        assert_eq!(EntryAgent::get_max_hold_bars_for_tf(1), 12);
        assert_eq!(EntryAgent::get_max_hold_bars_for_tf(5), 14);
        assert_eq!(EntryAgent::get_max_hold_bars_for_tf(15), 14);  // V5: increased
        assert_eq!(EntryAgent::get_max_hold_bars_for_tf(60), 12);
        assert_eq!(EntryAgent::get_max_hold_bars_for_tf(240), 12);  // V5: increased
    }

    #[test]
    fn test_thresholds_by_tf() {
        // V5: DRACONIAN thresholds
        assert_eq!(EntryAgent::get_enter_threshold_for_tf(1), 0.52);
        assert_eq!(EntryAgent::get_enter_threshold_for_tf(5), 0.58);
        assert_eq!(EntryAgent::get_enter_threshold_for_tf(15), 0.68);  // V5: much higher
        assert_eq!(EntryAgent::get_enter_threshold_for_tf(60), 0.70);
        assert_eq!(EntryAgent::get_enter_threshold_for_tf(240), 0.78);  // V5: extreme

        assert_eq!(EntryAgent::get_cancel_threshold_for_tf(1), 0.32);
        assert_eq!(EntryAgent::get_cancel_threshold_for_tf(5), 0.36);
        assert_eq!(EntryAgent::get_cancel_threshold_for_tf(15), 0.44);
        assert_eq!(EntryAgent::get_cancel_threshold_for_tf(60), 0.48);
        assert_eq!(EntryAgent::get_cancel_threshold_for_tf(240), 0.52);  // V5: higher

        assert_eq!(EntryAgent::get_min_margin_for_tf(1), 0.10);
        assert_eq!(EntryAgent::get_min_margin_for_tf(5), 0.12);
        assert_eq!(EntryAgent::get_min_margin_for_tf(15), 0.20);
        assert_eq!(EntryAgent::get_min_margin_for_tf(60), 0.24);
        assert_eq!(EntryAgent::get_min_margin_for_tf(240), 0.30);  // V5: extreme
    }

    #[test]
    fn test_dynamic_threshold_adjustments() {
        // Test RR adjustment
        let adjusted = EntryAgent::adjust_threshold_for_rr(0.65, 3.5);
        assert!((adjusted - 0.53).abs() < 0.01);  // 0.65 - 0.12

        let adjusted = EntryAgent::adjust_threshold_for_rr(0.65, 0.5);
        assert!((adjusted - 0.75).abs() < 0.01);  // 0.65 + 0.10

        // Test quality adjustment
        let adjusted = EntryAgent::adjust_threshold_for_quality(0.65, 9.5);
        assert!((adjusted - 0.55).abs() < 0.01);  // 0.65 - 0.10

        let adjusted = EntryAgent::adjust_threshold_for_quality(0.65, 2.0);
        assert!((adjusted - 0.73).abs() < 0.01);  // 0.65 + 0.08

        // Test combined
        let dynamic = EntryAgent::get_dynamic_enter_threshold(240, 2.5, 8.0);
        assert!(dynamic < 0.78);  // Should be lower than base 0.78

        // Test price adjustment
        let adjusted = EntryAgent::adjust_cancel_threshold_for_price(0.50, 60, 0.15);
        assert!((adjusted - 0.40).abs() < 0.01);  // 0.50 - 0.10
    }
}
