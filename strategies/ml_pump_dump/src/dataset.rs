// strategies/ml_pump_dump/src/dataset.rs
//
// Multi-Timeframe Pump/Dump Dataset Builder
//
// PIPELINE:
//   1. Fetch daily candles + indicators → detect pumps/dumps (≥15% moves)
//   2. For each event, drill down through lower TFs to find onset:
//      1440 → 240 → 60 → 15 → 5 → 1
//   3. On each TF, extract features from pre_event_lookback candles BEFORE onset
//   4. Flatten all TFs into a single wide row
//   5. Generate negative (non-pump) examples by sampling normal candles
//   6. Export to CSV for Python WFO trainer
//
// DATA SOURCE:
//   Candles: market.candles_Xm tables
//   Indicators: market.indicators_wide (LEFT JOIN)
//   Same approach as ml_entry_strategy::dataset but with multi-TF drilling.
//
// LABEL:
//   Binary: 1 = pre-pump/dump pattern detected, 0 = normal market conditions
//   We train TWO models: one for pump prediction, one for dump prediction.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::io::Write;
use tracing::info;

use crate::pump_dump::{
    CandleInd, EventType, PumpDumpConfig,
    ANALYSIS_TIMEFRAMES, FULL_FEATURES_PER_CANDLE,
    detect_anomalous_candles, extract_candle_features,
    find_pump_window, validate_sharp_move,
    parent_candle_window,
};

// ═════════════════════════════════════════════════════════════════════════════
// sqlx row struct (mirrors CandleInd)
// ═════════════════════════════════════════════════════════════════════════════

#[derive(sqlx::FromRow)]
struct CandleRow {
    time: DateTime<Utc>,
    symbol: String,
    symbol_id: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    rsi: f64,
    cci: f64,
    stoch_k: f64,
    stoch_d: f64,
    williams: f64,
    macd: f64,
    macd_signal: f64,
    macd_hist: f64,
    adx: f64,
    sma: f64,
    ema_20: f64,
    ema_50: f64,
    ema_200: f64,
    bb_upper: f64,
    bb_mid: f64,
    bb_lower: f64,
    atr: f64,
    obv: f64,
    vwap: f64,
    volume_spike: f64,
    trend: f64,
    trend_short: f64,
    poc: f64,
    alligator_jaw: f64,
    alligator_teeth: f64,
    alligator_lips: f64,
    mfi: f64,
    fibo_pivot: f64,
    fibo_r1: f64,
    fibo_s1: f64,
    supertrend: f64,
    supertrend_dir: f64,
    cmf: f64,
}

impl CandleRow {
    fn into_candle_ind(self) -> CandleInd {
        CandleInd {
            time: self.time, symbol: self.symbol, symbol_id: self.symbol_id,
            open: self.open, high: self.high, low: self.low, close: self.close,
            volume: self.volume,
            rsi: self.rsi, cci: self.cci, stoch_k: self.stoch_k, stoch_d: self.stoch_d,
            williams: self.williams, macd: self.macd, macd_signal: self.macd_signal,
            macd_hist: self.macd_hist, adx: self.adx, sma: self.sma,
            ema_20: self.ema_20, ema_50: self.ema_50, ema_200: self.ema_200,
            bb_upper: self.bb_upper, bb_mid: self.bb_mid, bb_lower: self.bb_lower,
            atr: self.atr, obv: self.obv, vwap: self.vwap,
            volume_spike: self.volume_spike, trend: self.trend, trend_short: self.trend_short,
            poc: self.poc, alligator_jaw: self.alligator_jaw,
            alligator_teeth: self.alligator_teeth, alligator_lips: self.alligator_lips,
            mfi: self.mfi, fibo_pivot: self.fibo_pivot, fibo_r1: self.fibo_r1,
            fibo_s1: self.fibo_s1, supertrend: self.supertrend,
            supertrend_dir: self.supertrend_dir, cmf: self.cmf,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// DB FETCH
// ═════════════════════════════════════════════════════════════════════════════

fn candle_table(tf_minutes: i32) -> &'static str {
    match tf_minutes {
        1 => "market.candles_1m",
        5 => "market.candles_5m",
        15 => "market.candles_15m",
        60 => "market.candles_1h",
        240 => "market.candles_4h",
        1440 => "market.candles_1d",
        _ => "market.candles_1d",
    }
}

/// Fetch candles with indicators for a symbol on a given TF.
pub async fn fetch_candles_with_indicators(
    pool: &PgPool,
    symbol: &str,
    tf_minutes: i32,
    limit: usize,
) -> Result<Vec<CandleInd>> {
    let table = candle_table(tf_minutes);

    let sql = format!(
        r#"
        SELECT
            c.time, c.symbol, p.symbol_id,
            c.open, c.high, c.low, c.close, c.volume,
            COALESCE(i.rsi, 50.0)::FLOAT8 as rsi,
            COALESCE(i.cci, 0.0)::FLOAT8 as cci,
            COALESCE(i.stoch_k, 50.0)::FLOAT8 as stoch_k,
            COALESCE(i.stoch_d, 50.0)::FLOAT8 as stoch_d,
            COALESCE(i.williams, -50.0)::FLOAT8 as williams,
            COALESCE(i.macd, 0.0)::FLOAT8 as macd,
            COALESCE(i.macd_signal, 0.0)::FLOAT8 as macd_signal,
            COALESCE(i.macd_hist, 0.0)::FLOAT8 as macd_hist,
            COALESCE(i.adx, 25.0)::FLOAT8 as adx,
            COALESCE(i.sma, c.close)::FLOAT8 as sma,
            COALESCE(i.ema_20, c.close)::FLOAT8 as ema_20,
            COALESCE(i.ema_50, c.close)::FLOAT8 as ema_50,
            COALESCE(i.ema_200, c.close)::FLOAT8 as ema_200,
            COALESCE(i.bb_upper, c.close)::FLOAT8 as bb_upper,
            COALESCE(i.bb_mid, c.close)::FLOAT8 as bb_mid,
            COALESCE(i.bb_lower, c.close)::FLOAT8 as bb_lower,
            COALESCE(i.atr, 0.001)::FLOAT8 as atr,
            COALESCE(i.obv, 0.0)::FLOAT8 as obv,
            COALESCE(i.vwap, c.close)::FLOAT8 as vwap,
            COALESCE(i.volume_spike, 1.0)::FLOAT8 as volume_spike,
            COALESCE(i.trend, 0.0)::FLOAT8 as trend,
            COALESCE(i.trend_short, 0.0)::FLOAT8 as trend_short,
            COALESCE(i.poc, c.close)::FLOAT8 as poc,
            COALESCE(i.alligator_jaw, c.close)::FLOAT8 as alligator_jaw,
            COALESCE(i.alligator_teeth, c.close)::FLOAT8 as alligator_teeth,
            COALESCE(i.alligator_lips, c.close)::FLOAT8 as alligator_lips,
            COALESCE(i.mfi, 50.0)::FLOAT8 as mfi,
            COALESCE(i.fibo_pivot, c.close)::FLOAT8 as fibo_pivot,
            COALESCE(i.fibo_r1, c.close)::FLOAT8 as fibo_r1,
            COALESCE(i.fibo_s1, c.close)::FLOAT8 as fibo_s1,
            COALESCE(i.supertrend, c.close)::FLOAT8 as supertrend,
            COALESCE(i.supertrend_dir, 0.0)::FLOAT8 as supertrend_dir,
            COALESCE(i.cmf, 0.0)::FLOAT8 as cmf
        FROM {table} c
        JOIN market.pairs p ON p.symbol = c.symbol
        LEFT JOIN market.indicators_wide i
            ON i.symbol_id = p.symbol_id AND i.time = c.time AND i.tf_minutes = $2
        WHERE c.symbol = $1
        ORDER BY c.time ASC
        LIMIT $3
        "#
    );

    let rows = sqlx::query_as::<_, CandleRow>(&sql)
        .bind(symbol)
        .bind(tf_minutes as i16)
        .bind(limit as i64)
        .fetch_all(pool)
        .await?;

    Ok(rows.into_iter().map(|r| r.into_candle_ind()).collect())
}

/// Fetch candles with indicators for ALL active symbols on a given TF.
pub async fn fetch_all_candles_for_tf(
    pool: &PgPool,
    tf_minutes: i32,
    limit_per_symbol: usize,
) -> Result<HashMap<String, Vec<CandleInd>>> {
    let table = candle_table(tf_minutes);

    let sql = format!(
        r#"
        WITH ranked AS (
            SELECT
                c.time, c.symbol, p.symbol_id,
                c.open, c.high, c.low, c.close, c.volume,
                COALESCE(i.rsi, 50.0)::FLOAT8 as rsi,
                COALESCE(i.cci, 0.0)::FLOAT8 as cci,
                COALESCE(i.stoch_k, 50.0)::FLOAT8 as stoch_k,
                COALESCE(i.stoch_d, 50.0)::FLOAT8 as stoch_d,
                COALESCE(i.williams, -50.0)::FLOAT8 as williams,
                COALESCE(i.macd, 0.0)::FLOAT8 as macd,
                COALESCE(i.macd_signal, 0.0)::FLOAT8 as macd_signal,
                COALESCE(i.macd_hist, 0.0)::FLOAT8 as macd_hist,
                COALESCE(i.adx, 25.0)::FLOAT8 as adx,
                COALESCE(i.sma, c.close)::FLOAT8 as sma,
                COALESCE(i.ema_20, c.close)::FLOAT8 as ema_20,
                COALESCE(i.ema_50, c.close)::FLOAT8 as ema_50,
                COALESCE(i.ema_200, c.close)::FLOAT8 as ema_200,
                COALESCE(i.bb_upper, c.close)::FLOAT8 as bb_upper,
                COALESCE(i.bb_mid, c.close)::FLOAT8 as bb_mid,
                COALESCE(i.bb_lower, c.close)::FLOAT8 as bb_lower,
                COALESCE(i.atr, 0.001)::FLOAT8 as atr,
                COALESCE(i.obv, 0.0)::FLOAT8 as obv,
                COALESCE(i.vwap, c.close)::FLOAT8 as vwap,
                COALESCE(i.volume_spike, 1.0)::FLOAT8 as volume_spike,
                COALESCE(i.trend, 0.0)::FLOAT8 as trend,
                COALESCE(i.trend_short, 0.0)::FLOAT8 as trend_short,
                COALESCE(i.poc, c.close)::FLOAT8 as poc,
                COALESCE(i.alligator_jaw, c.close)::FLOAT8 as alligator_jaw,
                COALESCE(i.alligator_teeth, c.close)::FLOAT8 as alligator_teeth,
                COALESCE(i.alligator_lips, c.close)::FLOAT8 as alligator_lips,
                COALESCE(i.mfi, 50.0)::FLOAT8 as mfi,
                COALESCE(i.fibo_pivot, c.close)::FLOAT8 as fibo_pivot,
                COALESCE(i.fibo_r1, c.close)::FLOAT8 as fibo_r1,
                COALESCE(i.fibo_s1, c.close)::FLOAT8 as fibo_s1,
                COALESCE(i.supertrend, c.close)::FLOAT8 as supertrend,
                COALESCE(i.supertrend_dir, 0.0)::FLOAT8 as supertrend_dir,
                COALESCE(i.cmf, 0.0)::FLOAT8 as cmf,
                ROW_NUMBER() OVER (PARTITION BY c.symbol ORDER BY c.time DESC) as rn
            FROM {table} c
            JOIN market.pairs p ON p.symbol = c.symbol AND p.is_active = true
            LEFT JOIN market.indicators_wide i
                ON i.symbol_id = p.symbol_id AND i.time = c.time AND i.tf_minutes = $1
        )
        SELECT time, symbol, symbol_id, open, high, low, close, volume,
               rsi, cci, stoch_k, stoch_d, williams, macd, macd_signal, macd_hist,
               adx, sma, ema_20, ema_50, ema_200, bb_upper, bb_mid, bb_lower, atr,
               obv, vwap, volume_spike, trend, trend_short, poc,
               alligator_jaw, alligator_teeth, alligator_lips,
               mfi, fibo_pivot, fibo_r1, fibo_s1, supertrend, supertrend_dir, cmf
        FROM ranked
        WHERE rn <= $2
        ORDER BY symbol, time ASC
        "#
    );

    let rows = sqlx::query_as::<_, CandleRow>(&sql)
        .bind(tf_minutes as i16)
        .bind(limit_per_symbol as i64)
        .fetch_all(pool)
        .await?;

    let mut grouped: HashMap<String, Vec<CandleInd>> = HashMap::new();
    for r in rows {
        let sym = r.symbol.clone();
        grouped.entry(sym).or_default().push(r.into_candle_ind());
    }

    Ok(grouped)
}

/// Fetch active symbols list.
pub async fn fetch_active_symbols(pool: &PgPool) -> Result<Vec<String>> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT symbol FROM market.pairs WHERE is_active = true ORDER BY symbol"
    )
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(|(s,)| s).collect())
}

// ═════════════════════════════════════════════════════════════════════════════
// DATASET EXAMPLE
// ═════════════════════════════════════════════════════════════════════════════

/// A single training example for the pump/dump model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PumpDumpExample {
    pub symbol: String,
    pub timestamp: String, // ISO8601 of the event onset
    pub event_type: String, // "PUMP", "DUMP", or "NONE"

    /// 1 = pre-pump/dump pattern, 0 = normal
    pub label: i8,

    /// Magnitude of the move (%) — 0 for negative examples
    pub move_pct: f64,

    /// Finest TF the event was located on
    pub finest_tf: i32,

    /// Multi-TF features (flattened)
    /// Structure: [tf1_c0_feat0..tf1_c0_feat81, tf1_c1_feat0..., ..., tfN_cM_feat81]
    pub features: Vec<f64>,
}

// ═════════════════════════════════════════════════════════════════════════════
// MULTI-TF DRILL DOWN + FEATURE COLLECTION
// ═════════════════════════════════════════════════════════════════════════════

/// For a detected daily pump/dump event, drill down through lower TFs and
/// extract features from pre_event_lookback candles before the onset.
///
/// Uses `find_pump_window` to find a window of ≤max_hold_candles where the
/// cumulative move reaches ≥daily_threshold_pct on the finest available TF.
///
/// NOTE: `daily_candle_close` is the **close time** of the daily candle (DB convention).
///
/// # Returns
/// `(features, finest_tf, onset_time, tf_move_pct, hold_candles)` or None.
pub fn drill_down_and_extract(
    all_tf_candles: &HashMap<i32, Vec<CandleInd>>,
    daily_candle_close: DateTime<Utc>,
    event_type: EventType,
    config: &PumpDumpConfig,
) -> Option<(Vec<f64>, i32, DateTime<Utc>, f64, usize)> {
    let lookback = config.pre_event_lookback;
    let n_tfs = ANALYSIS_TIMEFRAMES.len();
    let total_features = n_tfs * lookback * FULL_FEATURES_PER_CANDLE;

    let mut features = vec![0.0f64; total_features];
    let mut finest_tf = 1440;
    let mut onset_time = daily_candle_close;
    let mut best_move_pct = 0.0f64;
    let mut best_hold = 0usize;

    // Correct search window: daily candle covers [open, close]
    let (daily_search_start, daily_search_end) = parent_candle_window(daily_candle_close, 1440);

    // Try each TF from lowest (finest) to highest to find the finest TF
    // where a pump window of ≤max_hold candles reaches ≥threshold%
    // TFs are defined in ANALYSIS_TIMEFRAMES as [1440, 240, 60, 15, 5, 1]
    // We iterate in reverse to check finest first
    let target_tfs: Vec<i32> = ANALYSIS_TIMEFRAMES.iter()
        .filter(|&&tf| tf < 1440) // skip daily itself
        .rev()
        .copied()
        .collect();

    let mut found_finest = false;

    for &tf in &target_tfs {
        let candles = match all_tf_candles.get(&tf) {
            Some(c) if c.len() >= lookback + 10 => c,
            _ => continue,
        };

        let threshold = config.threshold_for_tf(tf);

        // Search within the daily candle's correct time window
        if let Some((onset_idx, move_pct, hold_candles)) = find_pump_window(
            candles, daily_search_start, daily_search_end,
            event_type, threshold, config.max_hold_candles, lookback,
        ) {
            finest_tf = tf;
            onset_time = candles[onset_idx].time;
            best_move_pct = move_pct;
            best_hold = hold_candles;
            found_finest = true;
            break; // prefer the finest TF where the window was found
        }
    }

    if !found_finest {
        // Fallback: try on the daily TF itself
        let candles = match all_tf_candles.get(&1440) {
            Some(c) if c.len() >= lookback + 10 => c,
            _ => return None,
        };
        let daily_idx = candles.iter().position(|c| c.time == daily_candle_close)?;
        if daily_idx < lookback { return None; }
        onset_time = daily_candle_close;
        finest_tf = 1440;
        // Calculate daily move for reporting
        let c = &candles[daily_idx];
        best_move_pct = match event_type {
            EventType::Pump => (c.high - c.open) / c.open * 100.0,
            EventType::Dump => (c.open - c.low) / c.open * 100.0,
        };
        best_hold = 1;
    }

    // Extract features from ALL TFs at the onset time
    for (tf_idx, &tf) in ANALYSIS_TIMEFRAMES.iter().enumerate() {
        let candles = match all_tf_candles.get(&tf) {
            Some(c) if c.len() >= lookback + 10 => c,
            _ => continue,
        };

        // Find the candle at or just before onset_time on this TF
        let idx = if tf == finest_tf {
            // Already found — use onset_time directly
            candles.iter().rposition(|c| c.time <= onset_time)
        } else if tf == 1440 {
            candles.iter().position(|c| c.time == daily_candle_close)
        } else {
            // For other TFs, find the closest candle <= onset_time
            let pp = candles.partition_point(|c| c.time <= onset_time);
            if pp > 0 { Some(pp - 1) } else { None }
        };

        let idx = match idx {
            Some(i) if i >= lookback => i,
            _ => continue,
        };

        // Extract features from lookback candles BEFORE the onset
        let feature_offset = tf_idx * lookback * FULL_FEATURES_PER_CANDLE;
        for c_off in 0..lookback {
            let candle_idx = idx - lookback + c_off;
            if candle_idx >= candles.len() { continue; }

            let candle_feats = extract_candle_features(candles, candle_idx);
            let start = feature_offset + c_off * FULL_FEATURES_PER_CANDLE;
            let end = start + FULL_FEATURES_PER_CANDLE;
            if end <= features.len() {
                features[start..end].copy_from_slice(&candle_feats);
            }
        }
    }

    Some((features, finest_tf, onset_time, best_move_pct, best_hold))
}

/// Generate negative (non-event) examples by sampling normal candles.
///
/// For each TF, at a random normal candle, we extract the same feature structure.
/// Ensures negative examples come from periods WITHOUT nearby pump/dump events.
///
/// Returns examples with label = 0.
pub fn generate_negative_examples(
    all_tf_candles: &HashMap<i32, Vec<CandleInd>>,
    pump_dump_times: &[DateTime<Utc>],
    n_examples: usize,
    config: &PumpDumpConfig,
) -> Vec<PumpDumpExample> {
    let lookback = config.pre_event_lookback;
    let n_tfs = ANALYSIS_TIMEFRAMES.len();
    let total_features = n_tfs * lookback * FULL_FEATURES_PER_CANDLE;

    // We'll sample from the daily candles as anchor points
    let daily_candles = match all_tf_candles.get(&1440) {
        Some(c) if c.len() > lookback + 10 => c,
        _ => return Vec::new(),
    };

    let mut examples = Vec::with_capacity(n_examples);

    // Find safe indices (not near any event)
    let min_distance_days = 3; // stay at least 3 days away from any event
    let safe_indices: Vec<usize> = (lookback..daily_candles.len())
        .filter(|&idx| {
            let t = daily_candles[idx].time;
            pump_dump_times.iter().all(|&event_t| {
                (t - event_t).num_days().unsigned_abs() > min_distance_days as u64
            })
        })
        .collect();

    if safe_indices.is_empty() {
        return Vec::new();
    }

    // Deterministic sampling: evenly space across safe indices
    let step = if safe_indices.len() > n_examples {
        safe_indices.len() / n_examples
    } else {
        1
    };

    let symbol = daily_candles[0].symbol.clone();

    for (count, &safe_idx) in safe_indices.iter().step_by(step).enumerate() {
        if count >= n_examples { break; }

        let anchor_time = daily_candles[safe_idx].time;
        let mut features = vec![0.0f64; total_features];

        // For each TF, find the candle at this time and extract lookback features
        for (tf_idx, &tf) in ANALYSIS_TIMEFRAMES.iter().enumerate() {
            let candles = match all_tf_candles.get(&tf) {
                Some(c) if c.len() >= lookback + 10 => c,
                _ => continue,
            };

            // Find the candle at or just before anchor_time
            let idx = candles.partition_point(|c| c.time <= anchor_time);
            let idx = if idx > 0 { idx - 1 } else { continue };
            if idx < lookback { continue; }

            let feature_offset = tf_idx * lookback * FULL_FEATURES_PER_CANDLE;
            for c_off in 0..lookback {
                let candle_idx = idx - lookback + c_off;
                if candle_idx >= candles.len() { continue; }

                let candle_feats = extract_candle_features(candles, candle_idx);
                let start = feature_offset + c_off * FULL_FEATURES_PER_CANDLE;
                let end = start + FULL_FEATURES_PER_CANDLE;
                if end <= features.len() {
                    features[start..end].copy_from_slice(&candle_feats);
                }
            }
        }

        examples.push(PumpDumpExample {
            symbol: symbol.clone(),
            timestamp: anchor_time.to_rfc3339(),
            event_type: "NONE".to_string(),
            label: 0,
            move_pct: 0.0,
            finest_tf: 1440,
            features,
        });
    }

    examples
}

// ═════════════════════════════════════════════════════════════════════════════
// FULL DATASET BUILD (BULK-LOAD — 5 SQL queries instead of N×5)
// ═════════════════════════════════════════════════════════════════════════════

/// Build the complete pump/dump dataset for all symbols.
///
/// BULK-LOAD approach (same as backtest v4):
///   1. Load ALL candles per TF in 5 queries total
///   2. For each symbol — process entirely in-memory (no per-symbol DB queries)
///   3. Detect pumps/dumps on daily TF
///   4. Validate sharpness against hourly
///   5. Drill down + extract features for each validated event
///   6. Generate negative examples
///   7. Export combined dataset
pub async fn build_pump_dump_dataset(
    pool: &PgPool,
    config: &PumpDumpConfig,
) -> Result<Vec<PumpDumpExample>> {
    info!("Building pump/dump dataset (bulk-load mode)...");
    config.log_summary();

    // Limits per TF (how many candles to fetch per symbol)
    let tf_limits: HashMap<i32, usize> = vec![
        (1440, 3700),  // ~10 years of daily data
        (240, 12000),  // ~8 years of 4h
        (60, 12000),   // ~2 years of 1h
        (15, 12000),   // ~6 months of 15m
        (5, 12000),    // ~2 months of 5m
    ].into_iter().collect();

    // ─── Phase 1: Bulk load ALL data (5 SQL queries total) ───
    info!("  ⏳ Bulk loading all candle data...");
    let total_start = std::time::Instant::now();
    let mut bulk_data: HashMap<i32, HashMap<String, Vec<CandleInd>>> = HashMap::new();
    let mut all_symbols: Vec<String> = Vec::new();

    for &tf in ANALYSIS_TIMEFRAMES {
        let limit = tf_limits.get(&tf).copied().unwrap_or(5000);
        let t0 = std::time::Instant::now();
        let grouped = fetch_all_candles_for_tf(pool, tf, limit).await?;
        let n_syms = grouped.len();
        let n_candles: usize = grouped.values().map(|v| v.len()).sum();
        info!("    TF {:>4}m: {:>3} symbols, {:>8} candles  ({:.1}s)",
              tf, n_syms, n_candles, t0.elapsed().as_secs_f64());

        if all_symbols.is_empty() {
            all_symbols = grouped.keys().cloned().collect();
            all_symbols.sort();
        }

        bulk_data.insert(tf, grouped);
    }

    let load_elapsed = total_start.elapsed().as_secs_f64();
    info!("  ✅ Bulk load complete: {} symbols in {:.1}s", all_symbols.len(), load_elapsed);

    // ─── Phase 2: Process each symbol from in-memory data ───
    info!("  🚀 Processing symbols...");
    let lookback = config.pre_event_lookback;
    let min_history = lookback + 50;

    let mut all_examples: Vec<PumpDumpExample> = Vec::new();
    let mut n_pumps = 0usize;
    let mut n_dumps = 0usize;
    let mut n_negatives = 0usize;
    let mut n_symbols_with_events = 0u32;
    let mut n_rejected_not_sharp = 0usize;
    let mut n_skipped = 0u32;

    let n_symbols = all_symbols.len();
    let progress_interval = (n_symbols / 20).max(1);

    for (si, symbol) in all_symbols.iter().enumerate() {
        // Build per-symbol TF map from bulk data (NO DB queries!)
        let mut sym_tf_candles: HashMap<i32, Vec<CandleInd>> = HashMap::new();
        let mut has_daily = false;

        for &tf in ANALYSIS_TIMEFRAMES {
            if let Some(tf_data) = bulk_data.get(&tf) {
                if let Some(candles) = tf_data.get(symbol) {
                    if candles.len() >= min_history {
                        sym_tf_candles.insert(tf, candles.clone());
                        if tf == 1440 { has_daily = true; }
                    }
                }
            }
        }

        if !has_daily {
            n_skipped += 1;
            continue;
        }

        // Step 1: Detect anomalous candles on daily
        let daily_events = detect_anomalous_candles(
            sym_tf_candles.get(&1440).unwrap(), config.daily_threshold_pct,
        );

        if daily_events.is_empty() { continue; }

        // Collect daily times
        let daily_time_map: Vec<(usize, EventType, f64, DateTime<Utc>)> = daily_events.iter()
            .map(|&(idx, et, mp)| {
                let t = sym_tf_candles.get(&1440).unwrap()[idx].time;
                (idx, et, mp, t)
            })
            .collect();

        // Step 2: Validate sharpness against hourly candles
        let mut validated_events: Vec<(usize, EventType, f64, DateTime<Utc>)> = Vec::new();

        for &(daily_idx, event_type, move_pct, daily_time) in &daily_time_map {
            if let Some(hourly) = sym_tf_candles.get(&60) {
                match validate_sharp_move(
                    hourly, daily_time, event_type, move_pct, config.concentration_pct,
                ) {
                    Some((_hourly_idx, _hourly_move)) => {
                        validated_events.push((daily_idx, event_type, move_pct, daily_time));
                    }
                    None => {
                        n_rejected_not_sharp += 1;
                    }
                }
            } else {
                // No hourly data — accept without sharpness validation
                validated_events.push((daily_idx, event_type, move_pct, daily_time));
            }
        }

        if validated_events.is_empty() { continue; }

        let mut sym_examples = Vec::new();
        let mut event_times = Vec::new();

        // Step 3: For each validated event, drill down and extract features
        for &(_daily_idx, event_type, _move_pct, daily_time) in &validated_events {
            event_times.push(daily_time);

            if let Some((features, finest_tf, onset_time, tf_move_pct, _hold)) =
                drill_down_and_extract(&sym_tf_candles, daily_time, event_type, config)
            {
                sym_examples.push(PumpDumpExample {
                    symbol: symbol.clone(),
                    timestamp: onset_time.to_rfc3339(),
                    event_type: event_type.to_string(),
                    label: 1,
                    move_pct: tf_move_pct,
                    finest_tf,
                    features,
                });
            }
        }

        // Step 4: Generate negative examples
        let n_pos_here = sym_examples.len();
        let n_neg = n_pos_here * config.negative_ratio;

        let neg_examples = generate_negative_examples(
            &sym_tf_candles, &event_times, n_neg, config,
        );
        sym_examples.extend(neg_examples);

        // Accumulate stats
        if n_pos_here > 0 { n_symbols_with_events += 1; }
        for ex in &sym_examples {
            match ex.label {
                1 if ex.event_type == "PUMP" => n_pumps += 1,
                1 if ex.event_type == "DUMP" => n_dumps += 1,
                _ => n_negatives += 1,
            }
        }
        all_examples.extend(sym_examples);

        // Progress bar
        if (si + 1) % progress_interval == 0 || si + 1 == n_symbols {
            info!("  [{:>3}/{}] {:>5.1}% | {} | {} examples so far",
                  si + 1, n_symbols, (si + 1) as f64 / n_symbols as f64 * 100.0,
                  symbol, all_examples.len());
        }
    }

    let total = all_examples.len();
    let elapsed = total_start.elapsed().as_secs_f64();
    info!("  ✅ Dataset built: {} total examples in {:.1}s", total, elapsed);
    info!("    PUMP events:  {} ({:.1}%)", n_pumps, if total > 0 { n_pumps as f64 / total as f64 * 100.0 } else { 0.0 });
    info!("    DUMP events:  {} ({:.1}%)", n_dumps, if total > 0 { n_dumps as f64 / total as f64 * 100.0 } else { 0.0 });
    info!("    Negatives:    {} ({:.1}%)", n_negatives, if total > 0 { n_negatives as f64 / total as f64 * 100.0 } else { 0.0 });
    info!("    Rejected (not sharp): {} | Skipped (no data): {}", n_rejected_not_sharp, n_skipped);
    info!("    Symbols with events: {} | Data load: {:.1}s", n_symbols_with_events, load_elapsed);

    Ok(all_examples)
}

// ═════════════════════════════════════════════════════════════════════════════
// CSV EXPORT
// ═════════════════════════════════════════════════════════════════════════════

/// Export pump/dump dataset to CSV.
pub fn export_dataset_csv(
    examples: &[PumpDumpExample],
    output_path: &str,
    feature_names: &[String],
) -> Result<()> {
    let file = std::fs::File::create(output_path)?;
    let mut buf = std::io::BufWriter::with_capacity(1 << 20, file);

    // Header
    let mut header = String::from("symbol,timestamp,event_type,label,move_pct,finest_tf");
    for name in feature_names {
        header.push(',');
        header.push_str(name);
    }
    writeln!(buf, "{}", header)?;

    // Rows
    for ex in examples {
        let mut line = format!(
            "{},{},{},{},{:.4},{}",
            ex.symbol, ex.timestamp, ex.event_type, ex.label, ex.move_pct, ex.finest_tf,
        );
        for &val in &ex.features {
            line.push_str(&format!(",{:.6}", val));
        }
        writeln!(buf, "{}", line)?;
    }

    buf.flush()?;
    info!("  Exported {} examples to {}", examples.len(), output_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_table() {
        assert_eq!(candle_table(1), "market.candles_1m");
        assert_eq!(candle_table(5), "market.candles_5m");
        assert_eq!(candle_table(15), "market.candles_15m");
        assert_eq!(candle_table(60), "market.candles_1h");
        assert_eq!(candle_table(240), "market.candles_4h");
        assert_eq!(candle_table(1440), "market.candles_1d");
    }
}
