// compute/predictors/examples/entry_agent_example.rs
//
// Entry Agent Integration Example
//
// This example shows how to integrate the Entry Agent into your trading pipeline.
// The Entry Agent replaces the "magic ideal candle" with a real online decision maker
// that decides ENTER/WAIT/CANCEL on each bar while the setup is active.
//
// USAGE:
//   1. Load entry_enter and entry_cancel models via ModelManager
//   2. Create EntryAgent with thresholds
//   3. When a setup is detected, start tracking it
//   4. On each new bar, call agent.decide() and act on the result
//
// This fixes the "entering at highs" problem by teaching the model to WAIT
// for the optimal entry moment within the entry window.

use anyhow::Result;
use predictors::entry_policy::{EntryAgent, EntryAgentConfig, EntryDecision};
use predictors::ml::model_manager::ModelManager;

/// Example: How to integrate Entry Agent into your trading pipeline
///
/// This is a conceptual example. Adapt to your actual architecture.
pub struct EntryAgentIntegration {
    /// Model manager with loaded entry models
    model_manager: ModelManager,
    /// Entry agent instance
    agent: EntryAgent,
    /// Whether to use GPU for inference
    use_gpu: bool,
}

/// A pending setup that is being tracked by the Entry Agent
#[derive(Clone, Debug)]
pub struct PendingSetup {
    /// Symbol
    pub symbol: String,
    /// Timeframe in minutes
    pub tf_minutes: i32,
    /// Side: +1 long, -1 short
    pub side: i8,
    /// Bar index when setup started
    pub setup_start_bar: usize,
    /// Current bar index (elapsed bars)
    pub elapsed_bars: usize,
    /// Feature vector for this setup
    pub features: Vec<f32>,
    /// Setup score (from signal scorer)
    pub setup_score: f32,
}

impl EntryAgentIntegration {
    /// Create a new integration instance
    pub fn new(use_gpu: bool) -> Result<Self> {
        // Create model manager
        let mut model_manager = ModelManager::new(use_gpu);

        // Load entry policy models for all timeframes
        let timeframes = vec![1, 5, 15, 60, 240];
        model_manager.load_models_for_timeframes(
            "entry_enter",
            "models/entry_enter_v1_tf{tf}.ubj",
            &timeframes,
            use_gpu,
        )?;

        model_manager.load_models_for_timeframes(
            "entry_cancel",
            "models/entry_cancel_v1_tf{tf}.ubj",
            &timeframes,
            use_gpu,
        )?;

        // Configure entry agent
        let config = EntryAgentConfig {
            enter_threshold: 0.55,      // Minimum probability to enter
            cancel_threshold: 0.50,     // Minimum probability to cancel
            min_margin: 0.15,           // Minimum margin between enter and cancel
            default_window_bars: 10,    // Default window size
        };

        let agent = EntryAgent::new(config);

        Ok(Self {
            model_manager,
            agent,
            use_gpu,
        })
    }

    /// Called when a new setup is detected
    ///
    /// Instead of entering immediately, we start tracking it
    /// and let the Entry Agent decide when to enter.
    pub fn on_setup_detected(
        &mut self,
        symbol: String,
        tf_minutes: i32,
        side: i8,
        features: Vec<f32>,
        setup_score: f32,
    ) -> PendingSetup {
        let window_bars = EntryAgent::get_window_bars_for_tf(tf_minutes);

        tracing::info!(
            "New setup detected: {} {}m side={} score={:.3} window={} bars",
            symbol, tf_minutes, side, setup_score, window_bars
        );

        PendingSetup {
            symbol,
            tf_minutes,
            side,
            setup_start_bar: 0, // Will be set by your bar counter
            elapsed_bars: 0,
            features,
            setup_score,
        }
    }

    /// Called on each new bar for a pending setup
    ///
    /// The Entry Agent decides: ENTER, WAIT, or CANCEL
    pub fn on_new_bar(
        &self,
        setup: &mut PendingSetup,
    ) -> Result<EntryDecision> {
        let window_bars = EntryAgent::get_window_bars_for_tf(setup.tf_minutes);
        let remaining = window_bars.saturating_sub(setup.elapsed_bars);

        // Make decision
        let decision = self.agent.decide(
            &self.model_manager,
            setup.tf_minutes,
            setup.features.clone(),
            setup.elapsed_bars as u16,
            remaining as u16,
            self.use_gpu,
        )?;

        // Log decision
        match &decision {
            EntryDecision::Enter { confidence } => {
                tracing::info!(
                    "ENTER signal for {} {}m (confidence={:.2}, elapsed={}, remaining={})",
                    setup.symbol, setup.tf_minutes, confidence, setup.elapsed_bars, remaining
                );
            }
            EntryDecision::Wait { confidence } => {
                tracing::debug!(
                    "WAIT for {} {}m (confidence={:.2}, elapsed={}, remaining={})",
                    setup.symbol, setup.tf_minutes, confidence, setup.elapsed_bars, remaining
                );
            }
            EntryDecision::Cancel { confidence } => {
                tracing::warn!(
                    "CANCEL setup for {} {}m (confidence={:.2}, elapsed={}, remaining={})",
                    setup.symbol, setup.tf_minutes, confidence, setup.elapsed_bars, remaining
                );
            }
        }

        // Increment elapsed bars
        setup.elapsed_bars += 1;

        Ok(decision)
    }

    /// Check if a setup has expired (exceeded window without ENTER)
    pub fn is_setup_expired(&self, setup: &PendingSetup) -> bool {
        let window_bars = EntryAgent::get_window_bars_for_tf(setup.tf_minutes);
        setup.elapsed_bars >= window_bars
    }
}

/// Example usage in a trading loop
pub async fn example_trading_loop() -> Result<()> {
    // Initialize
    let integration = EntryAgentIntegration::new(false)?; // use_gpu=false for this example

    // Simulate: A setup is detected
    let mut pending_setup = integration.on_setup_detected(
        "BTCUSDT".to_string(),
        5, // 5-minute timeframe
        1, // Long
        vec![0.0; 41], // Feature vector (41 features + 2 temporal = 43 total)
        0.65, // Setup score
    );

    // Simulate: Process each new bar
    for bar_idx in 0..15 {
        pending_setup.elapsed_bars = bar_idx;

        // Get decision from Entry Agent
        let decision = integration.on_new_bar(&mut pending_setup)?;

        match decision {
            EntryDecision::Enter { confidence } => {
                // EXECUTE TRADE
                println!(
                    "Bar {}: ENTER {} {}m with confidence {:.2}",
                    bar_idx, pending_setup.symbol, pending_setup.tf_minutes, confidence
                );
                // Your trade execution logic here:
                //   - Calculate position size
                //   - Place order
                //   - Set stop-loss and take-profit
                break; // Exit loop after entering
            }
            EntryDecision::Wait { .. } => {
                // CONTINUE WAITING
                println!("Bar {}: WAIT for better entry", bar_idx);
                // Continue to next bar
            }
            EntryDecision::Cancel { .. } => {
                // ABORT SETUP
                println!("Bar {}: CANCEL setup", bar_idx);
                break; // Exit loop, don't enter
            }
        }

        // Check if setup expired
        if integration.is_setup_expired(&pending_setup) {
            println!("Bar {}: Setup expired (window exceeded)", bar_idx);
            break;
        }
    }

    Ok(())
}

/// Comparison: Old vs New approach
///
/// OLD (problematic):
/// ```rust,ignore
/// // Signal detected → enter immediately
/// if setup_score >= threshold {
///     execute_trade();  // ❌ Often enters at highs
/// }
/// ```
///
/// NEW (with Entry Agent):
/// ```rust,ignore
/// // Signal detected → start tracking
/// if setup_score >= threshold {
///     pending_setups.push(setup);
/// }
///
/// // On each bar:
/// for setup in &mut pending_setups {
///     match agent.decide(...) {
///         Enter => execute_trade(),  // ✅ Enters at optimal moment
///         Wait => continue,          // Waits for better conditions
///         Cancel => remove(setup),   // Aborts bad setups
///     }
/// }
/// ```
///
/// BENEFITS:
///   - Better entry timing (waits for pullbacks/consolidation)
///   - Higher average PnL per trade
///   - Lower drawdown (avoids entering at extremes)
///   - Improved Sharpe ratio
///   - Fixes the "entering at highs" symptom

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_agent_integration() {
        // This is a conceptual test — actual test would need real models
        let config = EntryAgentConfig::default();
        let agent = EntryAgent::new(config);

        // Test decision logic with mock probabilities
        // (In real usage, these come from XGBoost models)

        // High enter, low cancel → ENTER
        // Low enter, high cancel → CANCEL
        // Both low → WAIT
    }
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Entry Agent Integration Example ===\n");

    // Run the example
    futures::executor::block_on(example_trading_loop())?;

    println!("\n=== Example Complete ===");
    Ok(())
}
