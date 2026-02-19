# Entry Agent — Online Entry Timing System

## Overview

The Entry Agent transforms your entry logic from "magic ideal candle" to a real **online decision-making agent** that makes ENTER/WAIT/CANCEL decisions on each new bar while a setup is active.

This fixes the "entering at highs" symptom by teaching the model to **WAIT** for the optimal entry moment within the entry window.

## Integration Status

### ✅ Completed Components

| Component | Status | Location |
|-----------|--------|----------|
| **Expert Labeling** | ✅ Complete | `backtester/src/entry_policy_labeler.rs` |
| **Dataset Export** | ✅ Complete | `backtester/src/main.rs` (export_entry_policy_dataset) |
| **Python Trainer** | ✅ Complete | `trainer/src/train_entry_policy.py` |
| **Rust EntryAgent** | ✅ Complete | `compute/predictors/src/entry_policy/agent.rs` |
| **Model Loading** | ✅ Complete | `compute/predictors/src/ml/model_manager.rs` |
| **CUDA Acceleration** | ✅ Complete | `cuda/kernels/entry_model.cu` |
| **Backtest Comparison** | ✅ Framework | `backtester/src/entry_agent_evaluator.rs` |

### ⚠️ In Progress

| Component | Status | Notes |
|-----------|--------|-------|
| **Live Trading Integration** | 🔲 TODO | Wire EntryAgent into TradeSignalCalculator |
| **Full Backtest Comparison** | 🔲 Partial | Framework exists, needs model loading |
| **Real-time Feature Extraction** | 🔲 TODO | Add bar-specific features (RSI, etc.) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SETUP DETECTED                              │
│          (signal_score >= threshold, e.g., 0.55)                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ENTRY AGENT (online)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Bar 0: WAIT  │  "Probability not high enough yet"       │  │
│  │  Bar 1: WAIT  │  "Waiting for pullback"                  │  │
│  │  Bar 2: WAIT  │  "Consolidation needed"                  │  │
│  │  Bar 3: ENTER │  "Optimal moment detected!"              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
              EXECUTE TRADE
         (position, SL, TP1/TP2/TP3)
```

## Key Components

### 1. Backtester Labeler (`backtester/src/entry_policy_labeler.rs`)

Generates expert-labeled training data by:
- Scanning the entry window [t0 .. t0+W] for each setup
- Simulating PnL for each candidate entry bar
- Selecting the optimal entry bar (argmax PnL)
- Labeling: WAIT for bars before optimal, ENTER at optimal, CANCEL if none profitable

```rust
use backtester::entry_policy_labeler::*;

let cfg = SimCfg {
    window_bars: 10,
    max_hold_bars: 14,
    sl_atr_mult: 0.9,
    rr1: 1.0,
    rr2: 1.5,
    rr3: 2.0,
    tp1_close_pct: 0.50,
    tp2_close_pct: 0.30,
    tp3_close_pct: 0.20,
};

let label = find_best_entry(side, t0, &ohlc_bars, cfg);
// Returns: (best_entry_offset, best_pnl)
```

### 2. Python Trainer (`trainer/src/train_entry_policy.py`)

Trains two binary XGBoost models per timeframe:
- **entry_enter**: Probability that NOW is the optimal entry bar
- **entry_cancel**: Probability that this setup should be cancelled

```bash
python trainer/src/train_entry_policy.py \
    --csv entry_policy_dataset.csv \
    --output-dir models \
    --gpu
```

Output models:
- `models/entry_enter_v1_tf1.ubj`, `models/entry_enter_v1_tf5.ubj`, etc.
- `models/entry_cancel_v1_tf1.ubj`, `models/entry_cancel_v1_tf5.ubj`, etc.

### 3. Rust Entry Agent (`compute/predictors/src/entry_policy/`)

Real-time inference module:

```rust
use predictors::entry_policy::{EntryAgent, EntryAgentConfig, EntryDecision};

// Configure
let config = EntryAgentConfig {
    enter_threshold: 0.55,
    cancel_threshold: 0.50,
    min_margin: 0.15,
    default_window_bars: 10,
};

let agent = EntryAgent::new(config);

// On each bar:
let decision = agent.decide(&model_manager, tf_minutes, &features, elapsed, remaining, use_gpu)?;

match decision {
    EntryDecision::Enter { confidence } => { /* Execute trade */ }
    EntryDecision::Wait { .. } => { /* Wait for next bar */ }
    EntryDecision::Cancel { .. } => { /* Abort setup */ }
}
```

### 4. CUDA Acceleration (`cuda/kernels/entry_model.cu`)

GPU-accelerated labeling for large datasets:

```bash
# Compiled automatically by cuda/build.rs
nvcc -ptx -o entry_model.ptx kernels/entry_model.cu
```

Useful when labeling millions of signals.

## Decision Logic

```rust
fn make_decision(prob_enter: f32, prob_cancel: f32) -> EntryDecision {
    // CANCEL: high cancel probability AND cancel > enter
    if prob_cancel >= cancel_threshold && prob_cancel > prob_enter {
        return EntryDecision::Cancel { confidence: prob_cancel };
    }
    
    // ENTER: high enter probability AND sufficient margin
    if prob_enter >= enter_threshold && (prob_enter - prob_cancel) >= min_margin {
        return EntryDecision::Enter { confidence: prob_enter };
    }
    
    // WAIT: neither condition met
    EntryDecision::Wait { confidence: 1.0 - max(prob_enter, prob_cancel) }
}
```

## Workflow

### Training Phase

```
1. Run backtester
   └─> cargo run --bin backtester
   └─> Generates: entry_policy_dataset.csv

2. Train models
   └─> python trainer/src/train_entry_policy.py
   └─> Generates: models/entry_enter_v1_tf*.ubj
   └─> Generates: models/entry_cancel_v1_tf*.ubj

3. (Optional) Accelerate labeling with CUDA
   └─> cargo build -p cuda
   └─> Generates: cuda/ptx/entry_model.ptx
```

### Inference Phase

```
1. Load models
   └─> ModelManager::load_models_for_timeframes("entry_enter", ...)
   └─> ModelManager::load_models_for_timeframes("entry_cancel", ...)

2. Create EntryAgent
   └─> EntryAgent::new(config)

3. On setup detected
   └─> Start tracking pending setup

4. On each new bar
   └─> agent.decide(...) → ENTER / WAIT / CANCEL
   └─> Act on decision
```

## Configuration

### Timeframe-Specific Windows

| TF    | Window Bars | Max Hold | Rationale                    |
|-------|-------------|----------|------------------------------|
| 1m    | 12          | 12       | 12 minutes of opportunities  |
| 5m    | 10          | 12       | 50 minutes                   |
| 15m   | 8           | 10       | 2 hours                      |
| 1h    | 6           | 8        | 6 hours                      |
| 4h    | 4           | 6        | 16 hours                     |

### Thresholds

| Parameter         | Default | Description                                    |
|-------------------|---------|------------------------------------------------|
| enter_threshold   | 0.55    | Minimum probability to enter                   |
| cancel_threshold  | 0.50    | Minimum probability to cancel                  |
| min_margin        | 0.15    | Minimum margin (enter - cancel) to trigger enter |

Adjust based on your backtest results:
- **Higher enter_threshold** → More selective entries, fewer trades
- **Higher min_margin** → Avoids ambiguous situations
- **Higher cancel_threshold** → More aggressive at aborting bad setups

## Expected Improvements

Based on your current statistics:

| Metric          | Before (1m/5m/15m) | After (Expected) |
|-----------------|-------------------|------------------|
| Win Rate        | 55-61%            | 58-65%           |
| Avg PnL         | ~0%               | +0.15-0.25%      |
| Avg PnL (wins)  | +0.49%            | +0.55-0.70%      |
| Avg PnL (losses)| -0.66%            | -0.50-0.60%      |
| Sharpe          | ~0                | 0.3-0.5          |
| SL_only %       | High              | -10-15%          |

**Why?**
- Better entry timing → Higher initial buffer → More TP hits, fewer SL hits
- Canceling weak setups → Avoid marginal trades
- Waiting for pullbacks → Enter at better prices

## Integration Example

See `compute/predictors/examples/entry_agent_example.rs` for a complete example.

Key integration points:
1. **TradeSignalCalculator**: Emit `SetupCandidate` instead of immediate `TradeSignal`
2. **EntryAgentStage**: Track pending setups, call `agent.decide()` on each bar
3. **OrderManager**: Only receive signals when `EntryDecision::Enter`

## Troubleshooting

### Model not loading
- Check file paths: `models/entry_enter_v1_tf5.ubj`
- Verify schema: `models/entry_enter_v1_tf5.schema.json`

### Too many WAIT decisions
- Lower `enter_threshold` (e.g., 0.50)
- Lower `min_margin` (e.g., 0.10)
- Retrain with more balanced data

### Too many CANCEL decisions
- Check if setup quality is low (increase min_setup_score)
- Lower `cancel_threshold` (e.g., 0.45)
- Verify feature consistency between training and inference

### Performance issues
- Enable GPU inference: `use_gpu=true`
- Use model pooling (cache frequently-used TF models)
- Reduce feature vector size if >100 features

## Next Steps

1. **Run backtester** to generate entry_policy_dataset.csv
2. **Train models** with train_entry_policy.py
3. **Test in paper trading** before live deployment
4. **Monitor metrics**: Track entry timing improvement vs. baseline
5. **Iterate**: Retrain monthly with fresh data

## Files Reference

| File | Purpose |
|------|---------|
| `backtester/src/entry_policy_labeler.rs` | Expert labeling logic |
| `backtester/src/main.rs` | Dataset export (updated) |
| `trainer/src/train_entry_policy.py` | Python trainer |
| `compute/predictors/src/entry_policy/mod.rs` | Module root |
| `compute/predictors/src/entry_policy/agent.rs` | EntryAgent logic |
| `compute/predictors/src/entry_policy/dataset.rs` | Dataset structures |
| `cuda/kernels/entry_model.cu` | CUDA labeling kernel |
| `cuda/build.rs` | PTX compilation (updated) |
| `compute/predictors/examples/entry_agent_example.rs` | Integration example |
