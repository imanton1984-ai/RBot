// backtester/src/entry_policy_labeler.rs
//
// Entry Policy Labeler — Expert labeling for training Entry Agent
//
// This module generates training examples for the Entry Policy models.
// It simulates trades at each candidate entry bar within the window and
// selects the optimal entry point that maximizes PnL.
//
// WORKFLOW:
//   1. For each signal (setup) with final_score >= threshold
//   2. Scan window [t0 .. t0+W] of candidate entry bars
//   3. For each candidate t, simulate trade with TP1/TP2/TP3/SL logic
//   4. Select t* = argmax pnl_pct(t)
//   5. If max pnl <= 0 → label CANCEL
//   6. Else: label WAIT for t0..t*-1, ENTER for t*
//
// OUTPUT: CSV with features + elapsed + remaining + label_enter + label_cancel + weight

use serde::{Deserialize, Serialize};

/// OHLCV bar with ATR for simulation
#[derive(Clone, Copy, Debug)]
pub struct OhlcBar {
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub atr: f64,
}

/// Simulation configuration (must match your backtester logic exactly!)
#[derive(Clone, Copy, Debug)]
pub struct SimCfg {
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
}

impl Default for SimCfg {
    fn default() -> Self {
        Self {
            window_bars: 4,
            max_hold_bars: 14,
            sl_atr_mult: 0.9,
            rr1: 1.0,
            rr2: 1.5,
            rr3: 2.0,
            tp1_close_pct: 0.70,
            tp2_close_pct: 0.20,
            tp3_close_pct: 0.10,
        }
    }
}

/// Training example for Entry Policy
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
    /// Metadata for debugging
    pub best_pnl: f32,
}

/// Result of expert labeling for one setup
#[derive(Clone, Debug)]
pub struct SetupLabel {
    /// Optimal entry bar index (relative to t0), or None if CANCEL
    pub best_entry_offset: Option<usize>,
    /// PnL at optimal entry (in %)
    pub best_pnl: f64,
}

/// Simulate PnL for a trade entered at entry_idx with partial close logic
///
/// This MUST match your backtester's exact PnL calculation logic.
/// If there's a mismatch, the model will learn wrong patterns.
#[allow(unused_assignments)]
pub fn simulate_pnl_pct(
    side: i8, // +1 long, -1 short
    entry_idx: usize,
    series: &[OhlcBar],
    cfg: SimCfg,
) -> f64 {
    if entry_idx >= series.len() {
        return 0.0;
    }

    let entry_price = series[entry_idx].close;
    let atr = series[entry_idx].atr.max(1e-9);
    let sl_dist = cfg.sl_atr_mult * atr;

    // Calculate SL and TP levels
    let sl_price = if side > 0 {
        entry_price - sl_dist
    } else {
        entry_price + sl_dist
    };

    let tp1_price = if side > 0 {
        entry_price + cfg.rr1 * sl_dist
    } else {
        entry_price - cfg.rr1 * sl_dist
    };

    let tp2_price = if side > 0 {
        entry_price + cfg.rr2 * sl_dist
    } else {
        entry_price - cfg.rr2 * sl_dist
    };

    let tp3_price = if side > 0 {
        entry_price + cfg.rr3 * sl_dist
    } else {
        entry_price - cfg.rr3 * sl_dist
    };

    // Calculate exit bar
    let exit_idx = (entry_idx + cfg.max_hold_bars).min(series.len() - 1);

    let mut position = 1.0; // 100% position
    let mut pnl = 0.0;
    let mut tp1_hit = false;
    let mut tp2_hit = false;
    let mut tp3_hit = false;

    for i in entry_idx..=exit_idx {
        let bar_high = series[i].high;
        let bar_low = series[i].low;

        // Check SL hit first
        let sl_hit = if side > 0 {
            bar_low <= sl_price
        } else {
            bar_high >= sl_price
        };

        if sl_hit {
            // Close remaining position at SL
            let price_diff = sl_price - entry_price;
            pnl += position * price_diff / entry_price * (side as f64);
            return pnl * 100.0;
        }

        // Check TP1 hit (partial close 50%)
        let tp1_hit_now = if side > 0 {
            bar_high >= tp1_price
        } else {
            bar_low <= tp1_price
        };

        if tp1_hit_now && !tp1_hit && position > 0.0 {
            let price_diff = tp1_price - entry_price;
            pnl += cfg.tp1_close_pct * price_diff / entry_price * (side as f64);
            position -= cfg.tp1_close_pct;
            tp1_hit = true;
        }

        // Check TP2 hit (partial close 30%)
        let tp2_hit_now = if side > 0 {
            bar_high >= tp2_price
        } else {
            bar_low <= tp2_price
        };

        if tp2_hit_now && !tp2_hit && position > 0.0 {
            let price_diff = tp2_price - entry_price;
            pnl += cfg.tp2_close_pct * price_diff / entry_price * (side as f64);
            position -= cfg.tp2_close_pct;
            tp2_hit = true;
        }

        // Check TP3 hit (close remaining 20%)
        let tp3_hit_now = if side > 0 {
            bar_high >= tp3_price
        } else {
            bar_low <= tp3_price
        };

        if tp3_hit_now && !tp3_hit && position > 0.0 {
            let price_diff = tp3_price - entry_price;
            pnl += cfg.tp3_close_pct * price_diff / entry_price * (side as f64);
            position -= cfg.tp3_close_pct;
            tp3_hit = true;
            // Fully closed
            return pnl * 100.0;
        }
    }

    // Expired: close remaining at exit bar's close
    let exit_price = series[exit_idx].close;
    let price_diff = exit_price - entry_price;
    pnl += position * price_diff / entry_price * (side as f64);
    pnl * 100.0
}

/// Find the best entry bar within the window for a given setup
///
/// Returns (best_entry_offset, best_pnl) where:
/// - best_entry_offset = None means CANCEL (no profitable entry)
/// - best_entry_offset = Some(k) means enter at bar t0+k
pub fn find_best_entry(
    side: i8,
    t0: usize,
    series: &[OhlcBar],
    cfg: SimCfg,
) -> SetupLabel {
    let mut best_offset: Option<usize> = None;
    let mut best_pnl: f64 = -1e9;

    let last_entry = (t0 + cfg.window_bars).min(series.len() - 1);

    for offset in 0..=(last_entry - t0) {
        let entry_idx = t0 + offset;
        let pnl = simulate_pnl_pct(side, entry_idx, series, cfg);

        if pnl > best_pnl {
            best_pnl = pnl;
            best_offset = Some(offset);
        }
    }

    // If best PnL is not profitable, recommend CANCEL
    if best_pnl <= 0.0 {
        SetupLabel {
            best_entry_offset: None,
            best_pnl,
        }
    } else {
        SetupLabel {
            best_entry_offset: best_offset,
            best_pnl,
        }
    }
}

/// Generate training examples for one setup
///
/// For CANCEL: creates 1 example at t0 with label_cancel=1
/// For ENTER: creates examples for each bar t0..t* with:
///   - t0..t*-1: label_enter=0 (WAIT)
///   - t*: label_enter=1 (ENTER)
pub fn build_examples_for_setup<F>(
    _side: i8,  // Reserved for future use
    t0: usize,
    label: &SetupLabel,
    window_bars: usize,
    mut feature_extractor: F,
) -> Vec<EntryPolicyExample>
where
    F: FnMut(usize) -> Vec<f32>, // feature extractor at bar index
{
    let mut examples = Vec::new();

    // CANCEL case
    if label.best_entry_offset.is_none() {
        let features = feature_extractor(t0);
        examples.push(EntryPolicyExample {
            features,
            elapsed: 0,
            remaining: window_bars as u16,
            label_enter: 0,
            label_cancel: 1,
            weight: 1.0,
            best_pnl: label.best_pnl as f32,
        });
        return examples;
    }

    // ENTER case: generate sequence WAIT..WAIT..ENTER
    let best_offset = label.best_entry_offset.unwrap();
    let best_pnl = label.best_pnl;

    for offset in 0..=best_offset {
        let bar_idx = t0 + offset;
        let features = feature_extractor(bar_idx);
        let elapsed = offset as u16;
        let remaining = (window_bars - offset) as u16;

        let is_enter_bar = (offset == best_offset) as u8;

        // Weight by PnL (higher PnL = more important sample)
        let weight = (best_pnl.max(0.0) as f32).clamp(0.1, 10.0);

        examples.push(EntryPolicyExample {
            features,
            elapsed,
            remaining,
            label_enter: is_enter_bar,
            label_cancel: 0,
            weight,
            best_pnl: best_pnl as f32,
        });
    }

    examples
}

/// Get window size for a given timeframe (optimized for each TF)
pub fn get_window_bars_for_tf(tf_minutes: i16) -> usize {
    match tf_minutes {
        1 => 3,   // 12 minutes of opportunities
        5 => 2,   // 50 minutes
        15 => 2,   // 2 hours
        60 => 2,   // 6 hours
        240 => 2,  // 16 hours
        _ => 8,    // default
    }
}

/// Get max hold bars for a given timeframe
pub fn get_max_hold_bars_for_tf(tf_minutes: i16) -> usize {
    // Typically 11-12 bars for consistency with backtester
    match tf_minutes {
        1 => 13,
        5 => 13,
        15 => 13,
        60 => 13,
        240 => 6,
        _ => 10,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_pnl_long_win() {
        let series = vec![
            OhlcBar { open: 100.0, high: 102.0, low: 99.0, close: 101.0, atr: 2.0 },
            OhlcBar { open: 101.0, high: 103.0, low: 100.0, close: 102.0, atr: 2.0 },
            OhlcBar { open: 102.0, high: 104.0, low: 101.0, close: 103.0, atr: 2.0 },
        ];

        let cfg = SimCfg {
            window_bars: 2,
            max_hold_bars: 2,
            sl_atr_mult: 0.9,
            rr1: 1.0,
            rr2: 1.5,
            rr3: 2.0,
            tp1_close_pct: 0.70,
            tp2_close_pct: 0.20,
            tp3_close_pct: 0.10,
        };

        // Long entry at bar 0, TP1 at 102 (101 + 1*2)
        let pnl = simulate_pnl_pct(1, 0, &series, cfg);
        assert!(pnl > 0.0, "Should be profitable");
    }

    #[test]
    fn test_simulate_pnl_long_sl_hit() {
        let series = vec![
            OhlcBar { open: 100.0, high: 101.0, low: 97.0, close: 98.0, atr: 2.0 },
            OhlcBar { open: 98.0, high: 99.0, low: 96.0, close: 97.0, atr: 2.0 },
        ];

        let cfg = SimCfg::default();

        // Long entry at bar 0, SL at 98 (100 - 1*2)
        let pnl = simulate_pnl_pct(1, 0, &series, cfg);
        assert!(pnl < 0.0, "Should be loss");
    }
}
