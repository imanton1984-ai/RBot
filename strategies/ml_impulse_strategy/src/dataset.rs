// strategies/ml_impulse_strategy/src/dataset.rs
//
// Multi-TF Impulse Engulfing Dataset Builder
//
// PIPELINE:
//   1. For each active symbol, load candles + indicators on all TFs
//   2. On each target TF (15m, 1h, 4h), detect engulfing signals via heuristic
//   3. For each signal, label: simulate SL/TP forward within max_hold candles
//      → label=1 if TP hit first, label=0 if SL hit first or max_hold expired with loss
//   4. Extract multi-TF features from lookback window BEFORE signal bar
//   5. Generate negative examples from non-signal bars
//   6. Export CSV for Python WFO trainer
//
// MEMORY SAFETY:
//   - Sequential symbol processing with controlled concurrency
//   - Data dropped after each symbol
//   - No bulk loading of all symbols at once

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::io::Write;
use tracing::info;

use crate::impulse::{
    CandleInd, SignalDir, ImpulseConfig, EngulfingSignal,
    ANALYSIS_TIMEFRAMES, FULL_FEATURES_PER_CANDLE,
    ENGULFING_FEATURE_COUNT,
    detect_engulfing_patterns, extract_candle_features,
    extract_engulfing_features,
};

// ═════════════════════════════════════════════════════════════════════════════
// sqlx row struct
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

/// Bulk fetch: load candles + indicators for ALL active symbols on a given TF.
/// Single query per TF — dramatically faster than per-symbol queries.
/// Returns HashMap<symbol, Vec<CandleInd>>.
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
// TRADE LABELING
// ═════════════════════════════════════════════════════════════════════════════

/// Simulate trade forward from signal bar to determine label.
///
/// Entry at candles[signal_bar].close.
/// Walk forward up to max_hold bars.
/// Check SL/TP on each bar's high/low.
/// Conservative: if SL and TP both hit on same bar, SL wins.
///
/// Returns (label, pnl_pct, hold_candles) or None if insufficient future data.
pub fn label_trade(
    candles: &[CandleInd],
    signal_bar: usize,
    direction: SignalDir,
    tp_pct: f64,
    sl_pct: f64,
    max_hold: usize,
) -> Option<(i8, f64, usize)> {
    let entry = candles[signal_bar].close;
    if entry.abs() < 1e-12 { return None; }

    let last_bar = (signal_bar + max_hold).min(candles.len().saturating_sub(1));
    if signal_bar + 1 > last_bar { return None; }

    let tp_price = match direction {
        SignalDir::Long  => entry * (1.0 + tp_pct / 100.0),
        SignalDir::Short => entry * (1.0 - tp_pct / 100.0),
    };
    let sl_price = match direction {
        SignalDir::Long  => entry * (1.0 - sl_pct / 100.0),
        SignalDir::Short => entry * (1.0 + sl_pct / 100.0),
    };

    for bar in (signal_bar + 1)..=last_bar {
        let c = &candles[bar];
        let hold = bar - signal_bar;

        match direction {
            SignalDir::Long => {
                // SL first (conservative)
                if c.low <= sl_price {
                    let pnl = (sl_price - entry) / entry * 100.0;
                    return Some((0, pnl, hold));
                }
                if c.high >= tp_price {
                    let pnl = (tp_price - entry) / entry * 100.0;
                    return Some((1, pnl, hold));
                }
            }
            SignalDir::Short => {
                if c.high >= sl_price {
                    let pnl = (entry - sl_price) / entry * 100.0;
                    return Some((0, pnl, hold));
                }
                if c.low <= tp_price {
                    let pnl = (entry - tp_price) / entry * 100.0;
                    return Some((1, pnl, hold));
                }
            }
        }

        // Max hold expired — label by P&L sign
        if hold >= max_hold {
            let exit = c.close;
            let pnl = match direction {
                SignalDir::Long  => (exit - entry) / entry * 100.0,
                SignalDir::Short => (entry - exit) / entry * 100.0,
            };
            let label = if pnl > 0.0 { 1 } else { 0 };
            return Some((label, pnl, hold));
        }
    }

    None
}

// ═════════════════════════════════════════════════════════════════════════════
// MULTI-TF FEATURE EXTRACTION
// ═════════════════════════════════════════════════════════════════════════════

/// Extract multi-TF features anchored at `bar_idx` on `target_tf`.
///
/// Features come from the lookback window BEFORE bar_idx on each analysis TF.
/// NO look-ahead: only uses candle data with close_time <= anchor bar's time.
///
/// Also appends engulfing meta-features (8).
///
/// Returns feature vector of size
/// (len(ANALYSIS_TIMEFRAMES) × lookback × FULL_FEATURES_PER_CANDLE + ENGULFING_FEATURE_COUNT)
/// or None if insufficient data.
pub fn extract_features_at_signal(
    all_tf_candles: &HashMap<i32, Vec<CandleInd>>,
    target_tf: i32,
    signal: &EngulfingSignal,
    lookback: usize,
) -> Option<Vec<f64>> {
    let n_tfs = ANALYSIS_TIMEFRAMES.len();
    let total_features = n_tfs * lookback * FULL_FEATURES_PER_CANDLE + ENGULFING_FEATURE_COUNT;

    let target_candles = all_tf_candles.get(&target_tf)?;
    let bar_idx = signal.bar_idx;
    if bar_idx >= target_candles.len() || bar_idx < lookback {
        return None;
    }

    let anchor_time = target_candles[bar_idx].time;
    let mut features = vec![0.0f64; total_features];
    let mut valid_tfs = 0u8;

    for (tf_idx, &tf) in ANALYSIS_TIMEFRAMES.iter().enumerate() {
        let candles = match all_tf_candles.get(&tf) {
            Some(c) if c.len() >= lookback + 10 => c,
            _ => continue,
        };

        // Find reference index: latest candle with time <= anchor_time
        let idx = if tf == target_tf {
            bar_idx
        } else {
            let pp = candles.partition_point(|c| c.time <= anchor_time);
            if pp == 0 { continue; }
            pp - 1
        };

        if idx < lookback { continue; }

        valid_tfs += 1;
        let feature_offset = tf_idx * lookback * FULL_FEATURES_PER_CANDLE;

        // Extract features from candles [idx - lookback .. idx - 1]
        // These are BEFORE the signal bar (no look-ahead)
        for c_off in 0..lookback {
            let candle_idx = idx - lookback + c_off;
            let candle_feats = extract_candle_features(candles, candle_idx);
            let start = feature_offset + c_off * FULL_FEATURES_PER_CANDLE;
            let end = start + FULL_FEATURES_PER_CANDLE;
            if end <= features.len() {
                features[start..end].copy_from_slice(&candle_feats);
            }
        }
    }

    // Need at least 2 TFs for meaningful prediction
    if valid_tfs < 2 { return None; }

    // Append engulfing meta-features
    let eng_feats = extract_engulfing_features(target_candles, signal);
    let eng_start = n_tfs * lookback * FULL_FEATURES_PER_CANDLE;
    features[eng_start..eng_start + ENGULFING_FEATURE_COUNT]
        .copy_from_slice(&eng_feats);

    Some(features)
}

// ═════════════════════════════════════════════════════════════════════════════
// DATASET EXAMPLE
// ═════════════════════════════════════════════════════════════════════════════

/// A single training example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpulseExample {
    pub symbol: String,
    pub timestamp: String,
    pub tf_minutes: i32,
    pub direction: String,  // "LONG" or "SHORT" or "NONE"
    /// 1 = TP hit (success), 0 = SL hit or failed
    pub label: i8,
    pub impulse_pct: f64,
    pub pnl_pct: f64,
    pub hold_candles: usize,
    /// Multi-TF features + engulfing meta
    pub features: Vec<f64>,
}

// ═════════════════════════════════════════════════════════════════════════════
// NEGATIVE EXAMPLE GENERATION
// ═════════════════════════════════════════════════════════════════════════════

/// Generate negative examples from bars where NO engulfing signal was detected.
///
/// Samples evenly spaced bars at least `min_distance` bars from any signal.
///
/// DEPRECATED: No longer used. NONE examples poison the ML model — it learns
/// "is engulfing present?" instead of "is this engulfing profitable?".
/// Kept for backward compatibility / manual experimentation.
#[allow(dead_code)]
pub fn generate_negative_examples(
    all_tf_candles: &HashMap<i32, Vec<CandleInd>>,
    target_tf: i32,
    signal_bars: &[usize],
    n_examples: usize,
    lookback: usize,
    symbol: &str,
) -> Vec<ImpulseExample> {
    let candles = match all_tf_candles.get(&target_tf) {
        Some(c) if c.len() > lookback + 10 => c,
        _ => return Vec::new(),
    };

    let min_distance = 5;
    let n_tfs = ANALYSIS_TIMEFRAMES.len();
    let total_features = n_tfs * lookback * FULL_FEATURES_PER_CANDLE + ENGULFING_FEATURE_COUNT;

    // Find safe indices
    let safe_indices: Vec<usize> = (lookback..candles.len().saturating_sub(5))
        .filter(|&idx| {
            signal_bars.iter().all(|&sb| {
                (idx as isize - sb as isize).unsigned_abs() > min_distance
            })
        })
        .collect();

    if safe_indices.is_empty() { return Vec::new(); }

    let step = if safe_indices.len() > n_examples {
        safe_indices.len() / n_examples
    } else {
        1
    };

    let mut examples = Vec::with_capacity(n_examples);

    for (count, &safe_idx) in safe_indices.iter().step_by(step).enumerate() {
        if count >= n_examples { break; }

        let anchor_time = candles[safe_idx].time;
        let mut features = vec![0.0f64; total_features];

        for (tf_idx, &tf) in ANALYSIS_TIMEFRAMES.iter().enumerate() {
            let tf_candles = match all_tf_candles.get(&tf) {
                Some(c) if c.len() >= lookback + 10 => c,
                _ => continue,
            };

            let idx = if tf == target_tf {
                safe_idx
            } else {
                let pp = tf_candles.partition_point(|c| c.time <= anchor_time);
                if pp > 0 { pp - 1 } else { continue }
            };
            if idx < lookback { continue; }

            let feature_offset = tf_idx * lookback * FULL_FEATURES_PER_CANDLE;
            for c_off in 0..lookback {
                let candle_idx = idx - lookback + c_off;
                if candle_idx >= tf_candles.len() { continue; }

                let candle_feats = extract_candle_features(tf_candles, candle_idx);
                let start = feature_offset + c_off * FULL_FEATURES_PER_CANDLE;
                let end = start + FULL_FEATURES_PER_CANDLE;
                if end <= features.len() {
                    features[start..end].copy_from_slice(&candle_feats);
                }
            }
        }

        // Engulfing meta = zeros for negatives (no engulfing pattern)
        // Already initialized to 0.0

        examples.push(ImpulseExample {
            symbol: symbol.to_string(),
            timestamp: anchor_time.to_rfc3339(),
            tf_minutes: target_tf,
            direction: "NONE".to_string(),
            label: 0,
            impulse_pct: 0.0,
            pnl_pct: 0.0,
            hold_candles: 0,
            features,
        });
    }

    examples
}

// ═════════════════════════════════════════════════════════════════════════════
// FULL DATASET BUILD
// ═════════════════════════════════════════════════════════════════════════════

/// Build the complete dataset for all symbols.
///
/// FAST MODE: Bulk-loads all candles per TF in a single SQL query,
/// then processes each symbol from in-memory data (no per-symbol queries).
/// 5 SQL queries total instead of 5 × N_symbols.
pub async fn build_impulse_dataset(
    pool: &PgPool,
    config: &ImpulseConfig,
) -> Result<Vec<ImpulseExample>> {
    info!("Building Impulse Engulfing dataset...");
    config.log_summary();

    let tf_limits: HashMap<i32, usize> = vec![
        (1440, 3700), (240, 12000), (60, 12000),
        (15, 12000), (5, 12000),
    ].into_iter().collect();

    // Phase 1: Bulk load ALL candles per TF (5 queries total)
    let mut all_data: HashMap<i32, HashMap<String, Vec<CandleInd>>> = HashMap::new();
    let mut all_symbols: Vec<String> = Vec::new();

    for &tf in ANALYSIS_TIMEFRAMES {
        let limit = tf_limits.get(&tf).copied().unwrap_or(5000);
        info!("  Loading TF {}m (limit {} per symbol)...", tf, limit);
        let t0 = std::time::Instant::now();
        let grouped = fetch_all_candles_for_tf(pool, tf, limit).await?;
        let n_syms = grouped.len();
        let n_candles: usize = grouped.values().map(|v| v.len()).sum();
        info!("    TF {}m: {} symbols, {} candles in {:.1}s",
              tf, n_syms, n_candles, t0.elapsed().as_secs_f64());

        // Collect all symbol names from the first TF
        if all_symbols.is_empty() {
            all_symbols = grouped.keys().cloned().collect();
            all_symbols.sort();
        }

        all_data.insert(tf, grouped);
    }

    info!("  {} total active symbols loaded", all_symbols.len());

    // Phase 2: Process each symbol from in-memory data (no DB calls)
    let lookback = config.pre_signal_lookback;
    let mut all_examples: Vec<ImpulseExample> = Vec::new();
    let mut n_long = 0usize;
    let mut n_short = 0usize;
    let mut n_negatives = 0usize;
    let mut n_tp_hit = 0usize;
    let mut n_symbols_with_signals = 0u32;

    for (si, symbol) in all_symbols.iter().enumerate() {
        // Build per-symbol TF map from bulk data
        let mut sym_tf_candles: HashMap<i32, Vec<CandleInd>> = HashMap::new();
        for &tf in ANALYSIS_TIMEFRAMES {
            if let Some(tf_data) = all_data.get(&tf) {
                if let Some(candles) = tf_data.get(symbol) {
                    if candles.len() >= lookback + 60 {
                        // Clone only the Vec reference — CandleInd data stays in HashMap
                        sym_tf_candles.insert(tf, candles.clone());
                    }
                }
            }
        }

        let examples = process_symbol_inmem(symbol, &sym_tf_candles, config);

        if examples.iter().any(|e| e.label == 1) {
            n_symbols_with_signals += 1;
        }
        for ex in &examples {
            match (ex.direction.as_str(), ex.label) {
                ("LONG", 1) => { n_long += 1; n_tp_hit += 1; }
                ("LONG", 0) => { n_long += 1; }
                ("SHORT", 1) => { n_short += 1; n_tp_hit += 1; }
                ("SHORT", 0) => { n_short += 1; }
                _ => { n_negatives += 1; }
            }
        }
        all_examples.extend(examples);

        if (si + 1) % 50 == 0 || si + 1 == all_symbols.len() {
            info!("  Progress: {}/{} symbols, {} examples so far",
                  si + 1, all_symbols.len(), all_examples.len());
        }
    }

    let total = all_examples.len();
    let n_signals = n_long + n_short;
    info!("  Dataset built: {} total examples", total);
    info!("    LONG signals:   {} ({:.1}%)", n_long, if total > 0 { n_long as f64 / total as f64 * 100.0 } else { 0.0 });
    info!("    SHORT signals:  {} ({:.1}%)", n_short, if total > 0 { n_short as f64 / total as f64 * 100.0 } else { 0.0 });
    info!("    Negatives:      {} ({:.1}%)", n_negatives, if total > 0 { n_negatives as f64 / total as f64 * 100.0 } else { 0.0 });
    if n_signals > 0 {
        info!("    Win rate (heuristic): {:.1}% ({} TP / {} total signals)",
              n_tp_hit as f64 / n_signals as f64 * 100.0, n_tp_hit, n_signals);
    }
    info!("    Symbols with signals: {}", n_symbols_with_signals);

    Ok(all_examples)
}

/// Process a single symbol from in-memory data (no DB calls).
///
/// NOTE: NONE (random non-signal bars) examples are NO LONGER generated.
/// The model should learn from actual engulfing signals only — some succeed
/// (label=1) and some fail (label=0). This prevents the model from learning
/// the trivial task "is engulfing pattern present?" and forces it to learn
/// "is THIS engulfing pattern likely to succeed?".
fn process_symbol_inmem(
    symbol: &str,
    all_tf_candles: &HashMap<i32, Vec<CandleInd>>,
    config: &ImpulseConfig,
) -> Vec<ImpulseExample> {
    let lookback = config.pre_signal_lookback;
    let mut examples = Vec::new();

    for tf_p in &config.tf_params {
        let target_tf = tf_p.tf_minutes;
        let candles = match all_tf_candles.get(&target_tf) {
            Some(c) if c.len() >= lookback + 60 => c,
            _ => continue,
        };

        let signals = detect_engulfing_patterns(
            candles,
            config,
            tf_p,
            lookback,
        );

        if signals.is_empty() { continue; }

        for signal in &signals {
            if signal.bar_idx + config.max_hold >= candles.len() {
                continue;
            }

            let label_result = label_trade(
                candles,
                signal.bar_idx,
                signal.direction,
                tf_p.tp_pct,
                tf_p.sl_pct,
                config.max_hold,
            );

            let (label, pnl_pct, hold_candles) = match label_result {
                Some(r) => r,
                None => continue,
            };

            let features = match extract_features_at_signal(
                all_tf_candles, target_tf, signal, lookback,
            ) {
                Some(f) => f,
                None => continue,
            };

            examples.push(ImpulseExample {
                symbol: symbol.to_string(),
                timestamp: candles[signal.bar_idx].time.to_rfc3339(),
                tf_minutes: target_tf,
                direction: signal.direction.to_string(),
                label,
                impulse_pct: signal.impulse_pct,
                pnl_pct,
                hold_candles,
                features,
            });
        }

        // NOTE: NONE examples removed — they poison the ML model.
        // With NONE examples, the model learns "is engulfing present?" (trivial task,
        // high accuracy) instead of "is this engulfing profitable?" (useful task).
        // The heuristic already filters non-engulfing bars.
        // Failed engulfing signals (label=0) naturally serve as negative examples.
    }

    examples
}

// ═════════════════════════════════════════════════════════════════════════════
// CSV EXPORT
// ═════════════════════════════════════════════════════════════════════════════

/// Export dataset to CSV.
pub fn export_dataset_csv(
    examples: &[ImpulseExample],
    output_path: &str,
    feature_names: &[String],
) -> Result<()> {
    let file = std::fs::File::create(output_path)?;
    let mut buf = std::io::BufWriter::with_capacity(1 << 20, file);

    // Header
    let mut header = String::from(
        "symbol,timestamp,tf_minutes,direction,label,impulse_pct,pnl_pct,hold_candles"
    );
    for name in feature_names {
        header.push(',');
        header.push_str(name);
    }
    writeln!(buf, "{}", header)?;

    // Rows
    for ex in examples {
        let mut line = format!(
            "{},{},{},{},{},{:.4},{:.4},{}",
            ex.symbol, ex.timestamp, ex.tf_minutes, ex.direction,
            ex.label, ex.impulse_pct, ex.pnl_pct, ex.hold_candles,
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
