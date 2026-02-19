// compute/predictors/src/entry_policy/dataset.rs
//
// Dataset structures for Entry Policy training and inference
//
// This module provides:
//   - EntryPolicyConfig: Configuration for labeling and training
//   - EntryPolicyExample: Single training example
//   - Utilities for exporting/importing datasets
//
// The dataset schema must match between:
//   1. Rust backtester (entry_policy_labeler.rs)
//   2. Python trainer (train_entry_policy.py)
//   3. Rust inference (this module + agent.rs)

use serde::{Deserialize, Serialize};

/// Configuration for Entry Policy labeling and training
///
/// This must match the configuration used in:
///   - backtester/src/entry_policy_labeler.rs (SimCfg)
///   - trainer/src/train_entry_policy.py
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EntryPolicyConfig {
    /// Window size: how many bars to wait for entry
    pub window_bars: usize,
    /// Maximum holding period: bars before forced exit
    pub max_hold_bars: usize,
    /// Stop-loss distance as ATR multiple
    pub sl_atr_mult: f64,
    /// RR targets for TP1/TP2/TP3
    pub rr1: f64,
    pub rr2: f64,
    pub rr3: f64,
    /// Partial close percentages
    pub tp1_close_pct: f64,  // 0.50
    pub tp2_close_pct: f64,  // 0.30
    pub tp3_close_pct: f64,  // 0.20
    /// Minimum setup score threshold (signals below this are ignored)
    pub min_setup_score: f32,
}

impl Default for EntryPolicyConfig {
    fn default() -> Self {
        Self {
            // V4: Unified with EntryAgent for consistent training/inference
            // Goal: Early entry, catch full move, fewer but higher quality trades
            window_bars: 4,
            max_hold_bars: 14,
            sl_atr_mult: 0.85,  // Reduced from 0.9 for tighter stops
            rr1: 0.8,           // Faster TP1 hit (was 1.0)
            rr2: 1.5,
            rr3: 2.5,           // Extended TP3 for full move capture (was 2.0)
            tp1_close_pct: 0.50,  // Close less at TP1 (was 0.70)
            tp2_close_pct: 0.30,  // More at TP2 (was 0.20)
            tp3_close_pct: 0.20,  // More at TP3 (was 0.10)
            min_setup_score: 0.55,
        }
    }
}

impl EntryPolicyConfig {
    /// Create config for a specific timeframe (matches EntryAgent::get_window_bars_for_tf)
    /// V5: UNIFIED with EntryAgent — training/inference must match exactly!
    ///
    /// V5: DRACONIAN filtering based on V4.1 backtest failure
    pub fn for_timeframe(tf_minutes: i32) -> Self {
        let (window_bars, max_hold_bars) = match tf_minutes {
            1 => (4, 12),   // V5: 4 min window, 12 min hold (fast scalping)
            5 => (4, 14),   // V5: 20 min window, 70 min hold (balanced)
            15 => (5, 14),  // V5: 75 min window, 210 min hold (confirmation + TP3)
            60 => (6, 12),  // V5: 6 hour window, 12 hour hold (proven working)
            240 => (4, 12), // V5: 16 hour window, 48 hour hold (patience + TP3)
            _ => (4, 12),
        };

        Self {
            window_bars,
            max_hold_bars,
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.window_bars == 0 {
            return Err("window_bars must be > 0".to_string());
        }
        if self.max_hold_bars == 0 {
            return Err("max_hold_bars must be > 0".to_string());
        }
        if self.sl_atr_mult <= 0.0 {
            return Err("sl_atr_mult must be > 0".to_string());
        }
        if self.tp1_close_pct + self.tp2_close_pct + self.tp3_close_pct != 1.0 {
            return Err("tp_close_pct sum must equal 1.0".to_string());
        }
        Ok(())
    }
}

/// Training example for Entry Policy
///
/// This structure is used for:
///   - Export from Rust backtester
///   - Import in Python trainer
///   - Feature schema validation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntryPolicyExample {
    /// Feature vector (from signal + current bar)
    pub features: Vec<f32>,
    /// Bars elapsed since setup started
    pub elapsed: u16,
    /// Bars remaining in entry window
    pub remaining: u16,
    /// Binary label: should we ENTER now?
    pub label_enter: u8,
    /// Binary label: should we CANCEL this setup?
    pub label_cancel: u8,
    /// Sample weight (based on best_pnl, clamped)
    pub weight: f32,
    /// Metadata: best PnL achieved (for debugging/analysis)
    pub best_pnl: f32,
}

impl EntryPolicyExample {
    /// Create a new example
    pub fn new(
        features: Vec<f32>,
        elapsed: u16,
        remaining: u16,
        label_enter: u8,
        label_cancel: u8,
        weight: f32,
        best_pnl: f32,
    ) -> Self {
        Self {
            features,
            elapsed,
            remaining,
            label_enter,
            label_cancel,
            weight,
            best_pnl,
        }
    }

    /// Get total number of features (including elapsed/remaining)
    pub fn total_features(&self) -> usize {
        self.features.len() + 2 // +2 for elapsed and remaining
    }
}

/// Dataset statistics for analysis
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DatasetStats {
    /// Total number of examples
    pub total_examples: usize,
    /// Number of ENTER examples
    pub enter_examples: usize,
    /// Number of CANCEL examples
    pub cancel_examples: usize,
    /// Number of WAIT examples (total - enter - cancel)
    pub wait_examples: usize,
    /// Average weight
    pub avg_weight: f32,
    /// Average best_pnl
    pub avg_best_pnl: f32,
    /// Examples by timeframe
    pub by_timeframe: std::collections::HashMap<i32, TimeframeStats>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TimeframeStats {
    pub total: usize,
    pub enter: usize,
    pub cancel: usize,
    pub wait: usize,
    pub avg_pnl: f32,
}

impl DatasetStats {
    /// Compute statistics from examples
    pub fn from_examples(examples: &[EntryPolicyExample]) -> Self {
        let mut stats = DatasetStats::default();

        stats.total_examples = examples.len();

        let mut total_weight = 0.0f32;
        let mut total_pnl = 0.0f32;

        for ex in examples {
            if ex.label_enter == 1 {
                stats.enter_examples += 1;
            } else if ex.label_cancel == 1 {
                stats.cancel_examples += 1;
            } else {
                stats.wait_examples += 1;
            }

            total_weight += ex.weight;
            total_pnl += ex.best_pnl;
        }

        stats.avg_weight = if examples.is_empty() {
            0.0
        } else {
            total_weight / examples.len() as f32
        };

        stats.avg_best_pnl = if examples.is_empty() {
            0.0
        } else {
            total_pnl / examples.len() as f32
        };

        stats
    }

    /// Print statistics (for debugging)
    pub fn print(&self) {
        println!("Dataset Statistics:");
        println!("  Total examples: {}", self.total_examples);
        println!("  ENTER: {} ({:.1}%)", self.enter_examples, self.enter_pct());
        println!("  CANCEL: {} ({:.1}%)", self.cancel_examples, self.cancel_pct());
        println!("  WAIT: {} ({:.1}%)", self.wait_examples, self.wait_pct());
        println!("  Avg weight: {:.4}", self.avg_weight);
        println!("  Avg best PnL: {:.4}%", self.avg_best_pnl);
    }

    pub fn enter_pct(&self) -> f32 {
        if self.total_examples == 0 {
            0.0
        } else {
            self.enter_examples as f32 / self.total_examples as f32 * 100.0
        }
    }

    pub fn cancel_pct(&self) -> f32 {
        if self.total_examples == 0 {
            0.0
        } else {
            self.cancel_examples as f32 / self.total_examples as f32 * 100.0
        }
    }

    pub fn wait_pct(&self) -> f32 {
        if self.total_examples == 0 {
            0.0
        } else {
            self.wait_examples as f32 / self.total_examples as f32 * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_for_timeframe() {
        // V5: Unified values matching EntryAgent::get_window_bars_for_tf
        let cfg_1m = EntryPolicyConfig::for_timeframe(1);
        assert_eq!(cfg_1m.window_bars, 4);
        assert_eq!(cfg_1m.max_hold_bars, 12);

        let cfg_5m = EntryPolicyConfig::for_timeframe(5);
        assert_eq!(cfg_5m.window_bars, 4);
        assert_eq!(cfg_5m.max_hold_bars, 14);

        let cfg_15m = EntryPolicyConfig::for_timeframe(15);
        assert_eq!(cfg_15m.window_bars, 5);  // V5: increased
        assert_eq!(cfg_15m.max_hold_bars, 14);  // V5: increased

        let cfg_1h = EntryPolicyConfig::for_timeframe(60);
        assert_eq!(cfg_1h.window_bars, 6);
        assert_eq!(cfg_1h.max_hold_bars, 12);

        let cfg_4h = EntryPolicyConfig::for_timeframe(240);
        assert_eq!(cfg_4h.window_bars, 4);  // V5: increased
        assert_eq!(cfg_4h.max_hold_bars, 12);  // V5: increased
    }

    #[test]
    fn test_config_validate() {
        let cfg = EntryPolicyConfig::default();
        assert!(cfg.validate().is_ok());

        let mut bad_cfg = cfg;
        bad_cfg.window_bars = 0;
        assert!(bad_cfg.validate().is_err());
    }

    #[test]
    fn test_dataset_stats() {
        let examples = vec![
            EntryPolicyExample::new(vec![], 0, 10, 1, 0, 1.0, 0.5),
            EntryPolicyExample::new(vec![], 1, 9, 0, 0, 1.0, 0.5),
            EntryPolicyExample::new(vec![], 2, 8, 0, 1, 1.0, -0.2),
        ];

        let stats = DatasetStats::from_examples(&examples);
        assert_eq!(stats.total_examples, 3);
        assert_eq!(stats.enter_examples, 1);
        assert_eq!(stats.cancel_examples, 1);
        assert_eq!(stats.wait_examples, 1);
    }
}
