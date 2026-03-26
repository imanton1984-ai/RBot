// strategies/ml_entry_strategy/src/direction/dataset.rs
//
// Direction Model v3 — Dataset Builder
//
// Builds a CSV dataset with:
//   - 32 direction v3 features (from features.rs)
//   - Labels:
//       * future_return_25: close[t+25]/close[t] - 1 in %
//       * direction: 1=LONG, -1=SHORT
//       * is_super: from TP/SL simulation
//       * bars_to_tp: how many bars until TP hit
//       * magnitude_pct: max of up/down move
//       * direction_quality: direction * (1.0 / bars_to_tp) — regression target
//
// OPTIMIZATION vs super_entry dataset builder:
//   - Computes only 32 features vs 128 → ~4x faster per candle
//   - No per-window 8-metric × 6-window expansion
//   - BTC candles loaded once per TF, reused for all symbols
//   - HTF candles loaded per symbol (for htf_supertrend_dir)
//   - No SR levels loading (removed: sparse, low signal)

use anyhow::Result;
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::io::Write;
use std::sync::Arc;
use tracing::info;

use crate::config::SuperEntryConfig;
use crate::dataset::{CandleWithIndicators, fetch_candles_with_indicators, fetch_active_symbols};
use crate::heuristic::get_higher_tf;
use crate::direction::features::{
    compute_direction_v3_features, resolve_btc_context, resolve_htf_context,
    DIRECTION_V3_FEATURES,
};

/// A single training example for Direction v3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionExample {
    pub symbol: String,
    pub tf_minutes: i32,
    pub timestamp: String, // ISO8601

    /// 32 direction v3 features
    pub features: Vec<f64>,

    // ── Labels ──
    /// Regression target: close[t+25]/close[t] - 1 in %
    pub future_return_25: f64,
    /// 1=LONG, -1=SHORT (sign of future_return, or from TP logic)
    pub direction: i8,
    /// From super model: did price move beyond TF target threshold?
    pub is_super: bool,
    /// How many bars until TP was hit (None if TP never hit within lookahead)
    pub bars_to_tp: Option<i32>,
    /// max of up/down move within lookahead
    pub magnitude_pct: f64,
    /// Quality-weighted direction target:
    ///   direction * (1.0 / bars_to_tp) — fast TP = high |score|
    /// Only meaningful for is_super=true examples.
    pub direction_quality: f64,
    /// Max up move within lookahead (%)
    pub max_up_move_pct: f64,
    /// Max down move within lookahead (%)
    pub max_down_move_pct: f64,
}

/// Build direction labels with TP-first-touch logic + bars_to_tp tracking.
///
/// For each candle at index t ∈ [start_idx, n - lookahead):
///   1. Compute 32 direction v3 features
///   2. Simulate LONG and SHORT trades with TP/SL
///   3. Record labels: direction, is_super, bars_to_tp, direction_quality
pub fn build_direction_labels(
    candles: &[CandleWithIndicators],
    btc_candles: Option<&[CandleWithIndicators]>,
    htf_candles: Option<&[CandleWithIndicators]>,
    start_idx: usize,
    lookahead: usize,
    target_move_pct: f64,
    sl_fraction: f64,
    tf_minutes: i32,
) -> Vec<DirectionExample> {
    let n = candles.len();
    if n < start_idx + lookahead + 1 {
        return Vec::new();
    }

    let end_idx = n - lookahead;
    let mut examples = Vec::with_capacity(end_idx - start_idx);

    for t in start_idx..end_idx {
        let entry_price = candles[t].close;
        if entry_price <= 0.0 {
            continue;
        }

        // ── TP/SL levels ──
        let tp_long = entry_price * (1.0 + target_move_pct / 100.0);
        let sl_long = entry_price * (1.0 - (target_move_pct * sl_fraction) / 100.0);
        let tp_short = entry_price * (1.0 - target_move_pct / 100.0);
        let sl_short = entry_price * (1.0 + (target_move_pct * sl_fraction) / 100.0);

        let mut long_win = false;
        let mut short_win = false;
        let mut long_active = true;
        let mut short_active = true;
        let mut max_up: f64 = 0.0;
        let mut max_down: f64 = 0.0;

        // Track bars_to_tp for each direction
        let mut bars_to_tp_long: Option<i32> = None;
        let mut bars_to_tp_short: Option<i32> = None;

        // Simulate trade candle-by-candle
        for k in 1..=lookahead {
            let idx = t + k;
            if idx >= n {
                break;
            }

            let high = candles[idx].high;
            let low = candles[idx].low;

            let up_move = (high - entry_price) / entry_price * 100.0;
            let down_move = (entry_price - low) / entry_price * 100.0;
            if up_move > max_up {
                max_up = up_move;
            }
            if down_move > max_down {
                max_down = down_move;
            }

            // LONG simulation (conservative: SL checked before TP on same bar)
            if long_active {
                if low <= sl_long {
                    long_active = false;
                } else if high >= tp_long {
                    long_win = true;
                    long_active = false;
                    bars_to_tp_long = Some(k as i32);
                }
            }

            // SHORT simulation
            if short_active {
                if high >= sl_short {
                    short_active = false;
                } else if low <= tp_short {
                    short_win = true;
                    short_active = false;
                    bars_to_tp_short = Some(k as i32);
                }
            }

            if !long_active && !short_active {
                break;
            }
        }

        // Determine direction and super status
        let (is_super, direction, magnitude, bars_to_tp) = if long_win && !short_win {
            (true, 1i8, target_move_pct, bars_to_tp_long)
        } else if short_win && !long_win {
            (true, -1i8, target_move_pct, bars_to_tp_short)
        } else if long_win && short_win {
            // Whipsaw: both TP hit → ambiguous, direction=0
            let mag = max_up.max(max_down);
            (true, 0i8, mag, None)
        } else {
            // Neither won
            if max_up >= max_down {
                (false, 1i8, max_up, None)
            } else {
                (false, -1i8, max_down, None)
            }
        };

        // Regression target: future return at exactly lookahead bars
        let future_close_idx = (t + lookahead).min(n - 1);
        let future_return_25 =
            (candles[future_close_idx].close - entry_price) / entry_price * 100.0;

        // Quality-weighted target: direction * speed_weight
        // Fast TP hit = high weight. No TP = weight ≈ 0.
        let direction_quality = match bars_to_tp {
            Some(bars) if bars > 0 => {
                direction as f64 * (1.0 / bars as f64)
            }
            _ => {
                // Fallback: use direction * small weight for non-super
                direction as f64 * 0.01
            }
        };

        // ── Compute 32 direction v3 features ──
        let btc_ctx = btc_candles
            .and_then(|btc| resolve_btc_context(btc, candles[t].time));

        let htf_ctx = htf_candles
            .and_then(|htf| resolve_htf_context(htf, candles[t].time));

        let features = compute_direction_v3_features(
            candles,
            t,
            btc_ctx.as_ref(),
            htf_ctx.as_ref(),
        );

        examples.push(DirectionExample {
            symbol: candles[t].symbol.clone(),
            tf_minutes,
            timestamp: candles[t].time.to_rfc3339(),
            features,
            future_return_25,
            direction,
            is_super,
            bars_to_tp,
            magnitude_pct: magnitude,
            direction_quality,
            max_up_move_pct: max_up,
            max_down_move_pct: max_down,
        });
    }

    examples
}

/// Export direction dataset to CSV.
///
/// CSV columns:
///   symbol, tf_minutes, timestamp,
///   [32 direction v3 feature columns],
///   future_return_25, direction, is_super, bars_to_tp,
///   magnitude_pct, direction_quality, max_up_move_pct, max_down_move_pct
pub fn export_direction_csv(
    examples: &[DirectionExample],
    output_path: &str,
) -> Result<()> {
    let mut file = std::fs::File::create(output_path)?;

    // Header
    let mut header = String::from("symbol,tf_minutes,timestamp");
    for name in DIRECTION_V3_FEATURES {
        header.push(',');
        header.push_str(name);
    }
    header.push_str(",future_return_25,direction,is_super,bars_to_tp,magnitude_pct,direction_quality,max_up_move_pct,max_down_move_pct");
    writeln!(file, "{}", header)?;

    // Rows — use buffered writer for performance
    let mut buf = std::io::BufWriter::with_capacity(1 << 20, file); // 1MB buffer

    for ex in examples {
        let mut line = format!("{},{},{}", ex.symbol, ex.tf_minutes, ex.timestamp);
        for &val in &ex.features {
            line.push_str(&format!(",{:.6}", val));
        }
        let bars_str = match ex.bars_to_tp {
            Some(b) => b.to_string(),
            None => String::new(), // empty = NULL in CSV
        };
        line.push_str(&format!(
            ",{:.6},{},{},{},{:.6},{:.6},{:.6},{:.6}",
            ex.future_return_25,
            ex.direction,
            if ex.is_super { 1 } else { 0 },
            bars_str,
            ex.magnitude_pct,
            ex.direction_quality,
            ex.max_up_move_pct,
            ex.max_down_move_pct,
        ));
        writeln!(buf, "{}", line)?;
    }

    buf.flush()?;
    Ok(())
}

/// Fetch all data and build the complete direction v3 dataset for one TF.
///
/// Steps:
///   1. Fetch all active symbols
///   2. Load BTC candles for this TF (for relative strength features)
///   3. For each symbol: load candles + HTF candles, compute features + labels
///
/// OPTIMIZATION vs v2:
///   - No SR levels loading (removed — sparse, low signal)
///   - HTF candles loaded per-symbol (for htf_supertrend_dir)
///   - Only 32 features computed vs 128
///   - BufWriter for CSV output
pub async fn build_direction_dataset_for_tf(
    pool: &PgPool,
    tf_minutes: i32,
    config: &SuperEntryConfig,
    limit_per_symbol: usize,
) -> Result<Vec<DirectionExample>> {
    let target_pct = config.target_pct_for_tf(tf_minutes);

    info!("Building direction v3 dataset for TF {}m (target={:.1}%, limit={})",
          tf_minutes, target_pct, limit_per_symbol);

    // 1. Load active symbols
    let symbols = fetch_active_symbols(pool).await?;
    info!("  {} active symbols", symbols.len());

    // 2. Load BTC candles as cross-reference (loaded ONCE, reused for all symbols)
    let btc_candles = fetch_candles_with_indicators(pool, "BTCUSDT", tf_minutes, limit_per_symbol)
        .await
        .unwrap_or_default();
    info!("  BTC candles loaded: {} rows", btc_candles.len());

    let btc_ref = if btc_candles.len() >= 50 {
        Some(btc_candles.as_slice())
    } else {
        info!("  ⚠ Not enough BTC candles (<50). BTC features will be 0.0");
        None
    };

    // Determine HTF for this TF
    let htf = get_higher_tf(tf_minutes);
    if let Some(h) = htf {
        info!("  HTF for {}m = {}m", tf_minutes, h);
    } else {
        info!("  No HTF for {}m (highest TF)", tf_minutes);
    }

    // 3. Process symbols CONCURRENTLY (up to 16 in parallel)
    // This is the key optimization: sequential = ~8s/symbol, concurrent = ~8s/batch_of_16
    let concurrency = std::env::var("DATASET_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16);

    let btc_arc: Arc<Option<Vec<CandleWithIndicators>>> = Arc::new(
        if btc_ref.is_some() { Some(btc_candles) } else { None }
    );

    let warmup_bars = config.warmup_bars;
    let lookahead_bars = config.lookahead_bars;
    let sl_fraction = config.sl_fraction;

    // Filter out BTCUSDT and create owned symbol list
    let symbols_to_process: Vec<String> = symbols
        .into_iter()
        .filter(|s| s != "BTCUSDT")
        .collect();

    info!("  Processing {} symbols with concurrency={}", symbols_to_process.len(), concurrency);

    // Create a stream of futures, each fetching + processing one symbol
    let results: Vec<Vec<DirectionExample>> = stream::iter(symbols_to_process)
        .map(|symbol| {
            let pool = pool.clone();
            let btc_arc = Arc::clone(&btc_arc);
            let min_required = warmup_bars + lookahead_bars + 1;

            async move {
                // Fetch main TF candles
                let candles = match fetch_candles_with_indicators(&pool, &symbol, tf_minutes, limit_per_symbol).await {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!("  Failed to fetch {}: {}", symbol, e);
                        return Vec::new();
                    }
                };

                if candles.len() < min_required {
                    return Vec::new();
                }

                // Fetch HTF candles for htf_supertrend_dir
                let htf_candles = if let Some(htf_tf) = htf {
                    match fetch_candles_with_indicators(&pool, &symbol, htf_tf, limit_per_symbol).await {
                        Ok(data) if data.len() >= 50 => Some(data),
                        _ => None,
                    }
                } else {
                    None
                };

                // CPU-bound feature computation — run in-line since it's fast (~32 features)
                let btc_slice = btc_arc.as_ref().as_deref();
                build_direction_labels(
                    &candles,
                    btc_slice,
                    htf_candles.as_deref(),
                    warmup_bars,
                    lookahead_bars,
                    target_pct,
                    sl_fraction,
                    tf_minutes,
                )
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;

    // Aggregate results
    let mut all_examples: Vec<DirectionExample> = Vec::new();
    let mut n_symbols_ok = 0u32;
    let mut n_super = 0usize;

    for examples in results {
        for ex in &examples {
            if ex.is_super {
                n_super += 1;
            }
        }
        if !examples.is_empty() {
            n_symbols_ok += 1;
        }
        all_examples.extend(examples);
    }

    let super_rate = if !all_examples.is_empty() {
        n_super as f64 / all_examples.len() as f64 * 100.0
    } else {
        0.0
    };

    info!(
        "  TF {}m: {} examples from {} symbols, {} super ({:.1}%)",
        tf_minutes, all_examples.len(), n_symbols_ok, n_super, super_rate
    );

    Ok(all_examples)
}
