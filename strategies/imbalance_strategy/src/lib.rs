// strategies/imbalance_strategy/src/lib.rs
//
// Imbalance Candle Trading Strategy
//
// GOAL:
//   Trade AFTER a large (imbalance) candle closes on a higher timeframe.
//   Enter on the lower timeframe with confirmation, hold 1-3 candles.
//
// PHILOSOPHY:
//   - We do NOT predict the future. We REACT to a confirmed event.
//   - A large candle = institutional order flow / liquidity event.
//   - After such a candle, price tends to either:
//     a) Continue (momentum) — especially if body is strong and wicks are small
//     b) Pull back (mean reversion) — especially if wicks are large (rejection)
//   - We trade on the lower TF for quick entries/exits with minimal risk exposure.
//
// TIMEFRAME PAIRS:
//   - 1D (≥12% move) → trade on 4H
//   - 4H (≥8% move)  → trade on 1H
//   - 1H (≥5% move)  → trade on 15m
//
// KEY METRICS FOR IMBALANCE CANDLE:
//   - Body ratio (body / range) — >0.6 = strong impulsion, <0.4 = rejection/doji
//   - Wick ratios — upper/lower wick as fraction of range
//   - Volume vs average — confirmation of institutional activity
//   - RSI/Stoch extremes — overbought/oversold context
//   - Trend alignment — is the move with or against the trend?
//
// ENTRY CONFIRMATION (on lower TF):
//   - Continuation: first lower-TF candle closes in same direction
//   - Reversal: first lower-TF candle shows rejection wick + RSI extreme
//
// EXIT:
//   - TP: fraction of parent candle range (dynamic, based on candle strength)
//   - SL: beyond parent candle extreme + ATR buffer
//   - Max hold: 3 candles on lower TF
//
// MODULES:
//   - imbalance: Imbalance candle detection + scoring
//   - confirmation: Entry confirmation + trade management

pub mod imbalance;
pub mod confirmation;
