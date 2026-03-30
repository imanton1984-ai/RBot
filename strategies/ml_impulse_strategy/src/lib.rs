// strategies/ml_impulse_strategy/src/lib.rs
//
// ML Impulse Absorption & Engulfing (IAE) Strategy v3
//
// GOAL:
//   Detect strong impulse candles followed by engulfing reversals.
//   The heuristic finds ALL candidate patterns (loose); the ML model
//   filters out the ones unlikely to succeed (hybrid approach).
//
// v3 KEY CHANGES (relaxed heuristic for ML pipeline):
//   1. ATR-based dynamic impulse threshold (1.5× ATR) instead of static %
//   2. Multi-bar absorption: 1-bar OR 2-bar engulfing (piercing line)
//   3. Volume spike on EITHER impulse OR absorption candle
//   4. Micro-gap leniency (0.1%) for continuous 24/7 crypto markets
//   5. Lowered static floors: 15m=1.0%, 1h=1.5%, 4h=3.5%
//   6. Increased max_hold to 8 candles for breathing room
//
// APPROACH:
//   1. Scan target TFs (15m, 1h, 4h) for absorption patterns with:
//      - Impulse candle (body >= 1.5× ATR, with static % floor)
//      - 1-bar or 2-bar engulfing/piercing line
//      - Body-to-shadow ratio > 0.5 on impulse
//      - Volume spike > 1.3× SMA(20) on EITHER impulse or absorption
//   2. For each detected pattern, extract multi-TF indicator features
//      from the lookback window BEFORE the signal bar
//   3. Train XGBoost ONLY on actual engulfing signals (no NONE class!)
//      to predict probability of success (TP hit vs SL hit)
//   4. In production: heuristic finds pattern → ML scores it → trade if prob ≥ 0.60
//
// PHILOSOPHY:
//   - Heuristic = "The Hunter" — finds ALL candidate absorption patterns (LOOSE)
//   - ML = "The Judge" — filters out false signals (STRICT)
//   - Loose heuristic → high volume of noisy candidates → massive training data
//   - XGBoost uses temporal + engulfing-meta features to segment noise from
//     high-probability structural reversals, lifting post-filter WR > 50%
//   - Two directions: LONG (bullish engulfing) and SHORT (bearish engulfing)
//   - Asymmetric TP>SL for favorable risk/reward (R:R ≈ 1.5:1)
//   - Per-TF static floors: 15m=1.0%, 1h=1.5%, 4h=3.5%
//   - 9 engulfing meta-features including absorption_bars indicator
//
// MODULES:
//   - impulse: Core config, types, engulfing detection, feature extraction
//   - dataset: Multi-TF dataset builder (DB → CSV)

pub mod impulse;
pub mod dataset;
