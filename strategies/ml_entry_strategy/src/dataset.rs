// strategies/ml_entry_strategy/src/dataset.rs
//
// Dataset Builder for Super Entry Model
//
// For each (pair, timeframe):
//   - First 300 candles are used as warmup for indicator computation
//   - From candle t=301 to t=980, create training examples:
//     * Features: indicator values at candle t
//     * Labels:
//       - max_up_move_pct, max_down_move_pct (over next 20 candles)
//       - direction (LONG if up >= down, else SHORT)
//       - magnitude_pct (max of up/down)
//       - is_super (magnitude >= TF_TARGET_MOVE_PCT)
//
// The dataset is saved as CSV for Python trainer consumption.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::io::Write;

use crate::config::INDICATOR_FEATURES;

/// A single candle row with indicators from the database
#[derive(Debug, Clone)]
pub struct CandleWithIndicators {
    pub time: DateTime<Utc>,
    pub symbol: String,
    pub symbol_id: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    // Indicators (match INDICATOR_FEATURES order)
    pub rsi: f64,
    pub cci: f64,
    pub stoch_k: f64,
    pub stoch_d: f64,
    pub williams: f64,
    pub macd: f64,
    pub macd_signal: f64,
    pub macd_hist: f64,
    pub adx: f64,
    pub sma: f64,
    pub ema_20: f64,
    pub ema_50: f64,
    pub ema_200: f64,
    pub bb_upper: f64,
    pub bb_mid: f64,
    pub bb_lower: f64,
    pub atr: f64,
    pub obv: f64,
    pub vwap: f64,
    pub volume_spike: f64,
    pub trend: f64,
    pub trend_short: f64,
    pub poc: f64,
}

impl CandleWithIndicators {
    /// Extract raw indicator values in the order of INDICATOR_FEATURES
    pub fn indicator_values(&self) -> Vec<f64> {
        vec![
            self.rsi, self.cci, self.stoch_k, self.stoch_d, self.williams,
            self.macd, self.macd_signal, self.macd_hist,
            self.adx, self.sma, self.ema_20, self.ema_50, self.ema_200,
            self.bb_upper, self.bb_mid, self.bb_lower, self.atr,
            self.obv, self.vwap, self.volume_spike,
            self.trend, self.trend_short, self.poc,
        ]
    }

    /// Compute derived features from raw indicators
    pub fn derived_features(&self) -> Vec<f64> {
        let close = self.close;
        let safe_div = |a: f64, b: f64| if b.abs() > 1e-12 { a / b } else { 0.0 };

        let rsi_norm = self.rsi / 100.0;
        let cci_norm = self.cci / 200.0;
        let stoch_norm = self.stoch_k / 100.0;
        let williams_norm = (self.williams + 100.0) / 100.0;

        let bb_range = self.bb_upper - self.bb_lower;
        let bb_position = if bb_range.abs() > 1e-12 {
            (close - self.bb_lower) / bb_range
        } else {
            0.5
        };
        let bb_width_pct = safe_div(bb_range, close) * 100.0;

        let atr_pct = safe_div(self.atr, close) * 100.0;
        let price_vs_sma = safe_div(close - self.sma, close) * 100.0;
        let price_vs_ema20 = safe_div(close - self.ema_20, close) * 100.0;
        let price_vs_ema50 = safe_div(close - self.ema_50, close) * 100.0;
        let price_vs_ema200 = safe_div(close - self.ema_200, close) * 100.0;
        let price_vs_vwap = safe_div(close - self.vwap, close) * 100.0;
        let macd_norm = safe_div(self.macd_hist, close) * 1000.0;
        let obv_change_pct = 0.0; // Requires history; not available in single-row context
        let volume_spike_flag = if self.volume_spike > 2.0 { 1.0 } else { 0.0 };

        vec![
            rsi_norm, cci_norm, stoch_norm, williams_norm,
            bb_position, bb_width_pct, atr_pct,
            price_vs_sma, price_vs_ema20, price_vs_ema50,
            price_vs_ema200, price_vs_vwap,
            macd_norm, obv_change_pct, volume_spike_flag,
        ]
    }

    /// Get full feature vector (indicators + derived)
    pub fn full_features(&self) -> Vec<f64> {
        let mut feats = self.indicator_values();
        feats.extend(self.derived_features());
        feats
    }
}

/// A labeled training example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperEntryExample {
    pub symbol: String,
    pub tf_minutes: i32,
    pub timestamp: String, // ISO8601
    pub features: Vec<f64>,
    pub max_up_move_pct: f64,
    pub max_down_move_pct: f64,
    pub direction: i8,       // 1 = LONG, -1 = SHORT
    pub magnitude_pct: f64,
    pub is_super: bool,
    pub future_return_20: f64, // close[t+20] / close[t] - 1 (in %)
}

/// Build labels for a time series of candles starting from `start_idx`
/// with `lookahead` bars look-forward.
///
/// Returns labeled examples for candles [start_idx .. end_idx].
pub fn build_labels(
    candles: &[CandleWithIndicators],
    start_idx: usize,
    lookahead: usize,
    target_move_pct: f64,
    tf_minutes: i32,
) -> Vec<SuperEntryExample> {
    let n = candles.len();
    if n < start_idx + lookahead + 1 {
        return Vec::new();
    }

    let end_idx = n - lookahead; // last valid index for labeling
    let mut examples = Vec::with_capacity(end_idx - start_idx);

    for t in start_idx..end_idx {
        let entry_price = candles[t].close;
        if entry_price <= 0.0 {
            continue;
        }

        // Look ahead window: [t+1 .. t+lookahead]
        let mut max_up: f64 = 0.0;
        let mut max_down: f64 = 0.0;

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
        }

        // Direction of best movement
        let (direction, magnitude) = if max_up >= max_down {
            (1i8, max_up)    // LONG
        } else {
            (-1i8, max_down) // SHORT
        };

        let is_super = magnitude >= target_move_pct;

        // Future return at exactly lookahead bars
        let future_close_idx = (t + lookahead).min(n - 1);
        let future_return_20 =
            (candles[future_close_idx].close - entry_price) / entry_price * 100.0;

        let features = candles[t].full_features();

        examples.push(SuperEntryExample {
            symbol: candles[t].symbol.clone(),
            tf_minutes,
            timestamp: candles[t].time.to_rfc3339(),
            features,
            max_up_move_pct: max_up,
            max_down_move_pct: max_down,
            direction,
            magnitude_pct: magnitude,
            is_super,
            future_return_20,
        });
    }

    examples
}

/// Fetch candles with indicators for a given symbol and timeframe from DB.
/// Returns up to `limit` rows ordered by time ASC.
pub async fn fetch_candles_with_indicators(
    pool: &PgPool,
    symbol: &str,
    tf_minutes: i32,
    limit: usize,
) -> Result<Vec<CandleWithIndicators>> {
    let candle_table = match tf_minutes {
        1 => "market.candles_1m",
        5 => "market.candles_5m",
        15 => "market.candles_15m",
        60 => "market.candles_1h",
        240 => "market.candles_4h",
        1440 => "market.candles_1d",
        _ => anyhow::bail!("Unsupported timeframe: {}", tf_minutes),
    };

    // NOTE: Indicators in market.indicators_wide are FLOAT4 (f32).
    // We cast them to FLOAT8 (f64) in SQL to match Rust types.
    // Candle OHLCV fields are already FLOAT8 in the candle tables.
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
            COALESCE(i.poc, c.close)::FLOAT8 as poc
        FROM {candle_table} c
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

    let candles = rows
        .into_iter()
        .map(|r| CandleWithIndicators {
            time: r.time,
            symbol: r.symbol,
            symbol_id: r.symbol_id,
            open: r.open,
            high: r.high,
            low: r.low,
            close: r.close,
            volume: r.volume,
            rsi: r.rsi,
            cci: r.cci,
            stoch_k: r.stoch_k,
            stoch_d: r.stoch_d,
            williams: r.williams,
            macd: r.macd,
            macd_signal: r.macd_signal,
            macd_hist: r.macd_hist,
            adx: r.adx,
            sma: r.sma,
            ema_20: r.ema_20,
            ema_50: r.ema_50,
            ema_200: r.ema_200,
            bb_upper: r.bb_upper,
            bb_mid: r.bb_mid,
            bb_lower: r.bb_lower,
            atr: r.atr,
            obv: r.obv,
            vwap: r.vwap,
            volume_spike: r.volume_spike,
            trend: r.trend,
            trend_short: r.trend_short,
            poc: r.poc,
        })
        .collect();

    Ok(candles)
}

/// sqlx compatible row struct
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
}

/// Bulk-fetch ALL candles with indicators for a given TF (all symbols at once).
/// Returns data grouped by symbol. Much faster than per-symbol queries.
pub async fn fetch_all_candles_for_tf(
    pool: &PgPool,
    tf_minutes: i32,
    limit_per_symbol: usize,
) -> Result<std::collections::HashMap<String, Vec<CandleWithIndicators>>> {
    let candle_table = match tf_minutes {
        1 => "market.candles_1m",
        5 => "market.candles_5m",
        15 => "market.candles_15m",
        60 => "market.candles_1h",
        240 => "market.candles_4h",
        1440 => "market.candles_1d",
        _ => anyhow::bail!("Unsupported timeframe: {}", tf_minutes),
    };

    // Single query: get last N candles per symbol with indicators via window function
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
                ROW_NUMBER() OVER (PARTITION BY c.symbol ORDER BY c.time DESC) as rn
            FROM {candle_table} c
            JOIN market.pairs p ON p.symbol = c.symbol AND p.is_active = true
            LEFT JOIN market.indicators_wide i
                ON i.symbol_id = p.symbol_id AND i.time = c.time AND i.tf_minutes = $1
        )
        SELECT time, symbol, symbol_id, open, high, low, close, volume,
               rsi, cci, stoch_k, stoch_d, williams, macd, macd_signal, macd_hist,
               adx, sma, ema_20, ema_50, ema_200, bb_upper, bb_mid, bb_lower, atr,
               obv, vwap, volume_spike, trend, trend_short, poc
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

    // Group by symbol
    let mut grouped: std::collections::HashMap<String, Vec<CandleWithIndicators>> =
        std::collections::HashMap::new();

    for r in rows {
        let candle = CandleWithIndicators {
            time: r.time, symbol: r.symbol.clone(), symbol_id: r.symbol_id,
            open: r.open, high: r.high, low: r.low, close: r.close, volume: r.volume,
            rsi: r.rsi, cci: r.cci, stoch_k: r.stoch_k, stoch_d: r.stoch_d,
            williams: r.williams, macd: r.macd, macd_signal: r.macd_signal,
            macd_hist: r.macd_hist, adx: r.adx, sma: r.sma,
            ema_20: r.ema_20, ema_50: r.ema_50, ema_200: r.ema_200,
            bb_upper: r.bb_upper, bb_mid: r.bb_mid, bb_lower: r.bb_lower,
            atr: r.atr, obv: r.obv, vwap: r.vwap, volume_spike: r.volume_spike,
            trend: r.trend, trend_short: r.trend_short, poc: r.poc,
        };
        grouped.entry(r.symbol).or_default().push(candle);
    }

    Ok(grouped)
}

/// Fetch list of active symbols from market.pairs
pub async fn fetch_active_symbols(pool: &PgPool) -> Result<Vec<String>> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT symbol FROM market.pairs WHERE is_active = true ORDER BY symbol"
    )
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(|(s,)| s).collect())
}

/// Export dataset to CSV file
pub fn export_dataset_csv(
    examples: &[SuperEntryExample],
    output_path: &str,
    feature_names: &[&str],
) -> Result<()> {
    let mut file = std::fs::File::create(output_path)?;

    // Header
    let mut header = String::from("symbol,tf_minutes,timestamp");
    for name in feature_names {
        header.push(',');
        header.push_str(name);
    }
    header.push_str(",max_up_move_pct,max_down_move_pct,direction,magnitude_pct,is_super,future_return_20");
    writeln!(file, "{}", header)?;

    // Rows
    for ex in examples {
        let mut line = format!("{},{},{}", ex.symbol, ex.tf_minutes, ex.timestamp);
        for &val in &ex.features {
            line.push_str(&format!(",{:.6}", val));
        }
        line.push_str(&format!(
            ",{:.6},{:.6},{},{:.6},{},{:.6}",
            ex.max_up_move_pct,
            ex.max_down_move_pct,
            ex.direction,
            ex.magnitude_pct,
            if ex.is_super { 1 } else { 0 },
            ex.future_return_20
        ));
        writeln!(file, "{}", line)?;
    }

    Ok(())
}

/// Get all feature names (indicator + derived) in order
pub fn all_feature_names() -> Vec<&'static str> {
    let mut names: Vec<&str> = INDICATOR_FEATURES.to_vec();
    names.extend_from_slice(crate::config::DERIVED_FEATURES);
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_candle(close: f64, high: f64, low: f64) -> CandleWithIndicators {
        CandleWithIndicators {
            time: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            symbol_id: 1,
            open: close,
            high,
            low,
            close,
            volume: 1000.0,
            rsi: 50.0, cci: 0.0, stoch_k: 50.0, stoch_d: 50.0, williams: -50.0,
            macd: 0.0, macd_signal: 0.0, macd_hist: 0.0,
            adx: 25.0, sma: close, ema_20: close, ema_50: close, ema_200: close,
            bb_upper: close * 1.02, bb_mid: close, bb_lower: close * 0.98,
            atr: close * 0.01, obv: 0.0, vwap: close, volume_spike: 1.0,
            trend: 0.0, trend_short: 0.0, poc: close,
        }
    }

    #[test]
    fn test_build_labels_basic() {
        // Create 25 candles: close=100, but candle 5 has high=110 (10% up)
        let mut candles: Vec<CandleWithIndicators> = (0..25)
            .map(|_| make_test_candle(100.0, 101.0, 99.0))
            .collect();

        // Candle at index 5 has a spike up
        candles[5].high = 110.0;

        let examples = build_labels(&candles, 0, 20, 5.0, 5);

        // We should get examples for indices 0..5 (since 25 - 20 = 5)
        assert_eq!(examples.len(), 5);

        // Example at t=0 should see the spike at t=5 within lookahead
        let ex0 = &examples[0];
        assert!(ex0.max_up_move_pct >= 9.0); // ~10%
        assert_eq!(ex0.direction, 1); // LONG
        assert!(ex0.is_super); // 10% > 5% threshold
    }

    #[test]
    fn test_build_labels_short_direction() {
        let mut candles: Vec<CandleWithIndicators> = (0..25)
            .map(|_| make_test_candle(100.0, 101.0, 99.0))
            .collect();

        // Candle at index 3 has a drop
        candles[3].low = 90.0;

        let examples = build_labels(&candles, 0, 20, 5.0, 5);

        // Example at t=0 should see the drop
        let ex0 = &examples[0];
        assert!(ex0.max_down_move_pct >= 9.0); // ~10%
        assert_eq!(ex0.direction, -1); // SHORT
        assert!(ex0.is_super);
    }

    #[test]
    fn test_feature_count() {
        let candle = make_test_candle(100.0, 101.0, 99.0);
        let features = candle.full_features();
        assert_eq!(features.len(), crate::config::total_feature_count());
    }
}
