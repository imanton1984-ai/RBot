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
            enter_threshold: 0.5,
            cancel_threshold: 0.5,
            min_margin: 0.1,
            default_window_bars: 10,
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

    /// Get window size for a given timeframe (matches backtester labeling)
    pub fn get_window_bars_for_tf(tf_minutes: i32) -> usize {
        match tf_minutes {
            1 => 12,   // 12 minutes of opportunities
            5 => 10,   // 50 minutes
            15 => 8,   // 2 hours
            60 => 6,   // 6 hours
            240 => 4,  // 16 hours
            _ => 10,   // default
        }
    }

    /// Get max hold bars for a given timeframe
    pub fn get_max_hold_bars_for_tf(tf_minutes: i32) -> usize {
        match tf_minutes {
            1 => 12,
            5 => 12,
            15 => 10,
            60 => 8,
            240 => 6,
            _ => 10,
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

        // Enter above threshold but margin too small → WAIT
        let decision = agent.make_decision(0.55, 0.50);
        assert!(decision.is_wait());

        // Enter above threshold with good margin → ENTER
        let decision = agent.make_decision(0.70, 0.50);
        assert!(decision.is_enter());
    }

    #[test]
    fn test_window_bars_by_tf() {
        assert_eq!(EntryAgent::get_window_bars_for_tf(1), 12);
        assert_eq!(EntryAgent::get_window_bars_for_tf(5), 10);
        assert_eq!(EntryAgent::get_window_bars_for_tf(15), 8);
        assert_eq!(EntryAgent::get_window_bars_for_tf(60), 6);
        assert_eq!(EntryAgent::get_window_bars_for_tf(240), 4);
    }
}
