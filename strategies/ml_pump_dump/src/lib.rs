// strategies/ml_pump_dump/src/lib.rs
//
// ML Pump/Dump Detection Strategy
//
// GOAL:
//   Detect patterns that precede anomalous price movements (pumps/dumps)
//   on crypto markets. The model learns from historical pump/dump events
//   across multiple timeframes to predict the next one BEFORE it happens.
//
// APPROACH:
//   1. Scan daily candles → find candles with >=15% move (anomalous pump/dump)
//   2. Drill down to lower timeframes (4h → 1h → 15m → 5m → 1m) to pinpoint
//      the exact moment of the pump/dump start
//   3. For each pump/dump event, extract features from the N candles BEFORE it:
//      - All indicator values (33 raw from indicators_wide)
//      - Derived features (normalized, ratios, distances)
//      - Temporal/dynamic features (how fast indicators are changing)
//      - Cross-timeframe features (what lower/higher TF look like)
//   4. Train XGBoost to recognize these pre-pump/dump patterns
//   5. In backtest mode, evaluate if the model can predict pumps/dumps
//      1-2 candles before they happen
//
// PHILOSOPHY:
//   - Binary classification: PUMP (1) vs NO_PUMP (0) and DUMP (1) vs NO_DUMP (0)
//   - Two separate models: one for pumps, one for dumps
//   - Multi-timeframe feature aggregation: same event seen through different lenses
//   - WFO training to avoid lookahead bias
//
// MODULES:
//   - pump_dump: Core config, types, pump/dump detection logic
//   - dataset: Multi-TF dataset builder (DB → CSV)

pub mod pump_dump;
pub mod dataset;
