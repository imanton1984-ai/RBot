// strategies/ml_entry_strategy/src/direction/mod.rs
//
// Direction Model v3 — optimized direction prediction module.
//
// KEY CHANGES FROM v2:
//   - 32 curated features from 6 domains (was 20 with placeholders)
//   - Regression on direction_quality = direction * 1/bars_to_tp (was binary classification)
//   - Inference: sign(prediction) = direction, abs(prediction) = confidence
//   - Confidence gate: skip if abs(prediction) < threshold
//   - HTF hard filter: never LONG against HTF bearish supertrend (and vice versa)
//
// Target: direction accuracy ≥ 0.60 on WFO OOS (honest, no overfitting).
// Previous results: v1 (128 features) ~0.50, v2 (20 features) ~0.49.

pub mod features;
pub mod dataset;
