// compute/predictors/src/entry_policy/mod.rs
//
// Entry Policy Module — Online Entry Timing Agent
//
// This module provides the Entry Agent which makes real-time decisions:
//   - ENTER: Enter the trade now
//   - WAIT: Wait for better conditions (next bar)
//   - CANCEL: Setup is no longer valid, don't enter
//
// The agent uses two XGBoost models (entry_enter, entry_cancel) trained
// on expert-labeled data from the backtester's entry_policy_labeler.
//
// USAGE:
//   1. Load models via ModelManager: "entry_enter_tf{X}" and "entry_cancel_tf{X}"
//   2. Create EntryAgent with thresholds
//   3. On each bar while setup is active, call EntryAgent::decide()
//   4. Act on the decision (execute trade, wait, or abort)

pub mod agent;
pub mod dataset;

pub use agent::{EntryAgent, EntryAgentConfig, EntryDecision};
pub use dataset::EntryPolicyConfig;
