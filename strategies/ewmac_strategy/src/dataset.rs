// strategies/ewmac_strategy/src/dataset.rs
//
// Data fetching for EWMAC Strategy.
//
// EWMAC needs only OHLCV data (no indicator JOIN required).
// This is simpler and faster than ml_entry_strategy's dataset.
//
// Uses the same market.candles_{tf} tables and market.pairs.

use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::PgPool;
use std::collections::HashMap;

/// A single candle row with OHLCV data
#[derive(Debug, Clone)]
pub struct Candle {
    pub time: DateTime<Utc>,
    pub symbol: String,
    pub symbol_id: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// sqlx-compatible row struct
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
}

/// Map timeframe minutes to candle table name.
fn candle_table(tf_minutes: i32) -> Result<&'static str> {
    match tf_minutes {
        1 => Ok("market.candles_1m"),
        5 => Ok("market.candles_5m"),
        15 => Ok("market.candles_15m"),
        60 => Ok("market.candles_1h"),
        240 => Ok("market.candles_4h"),
        1440 => Ok("market.candles_1d"),
        _ => anyhow::bail!("Unsupported timeframe: {}", tf_minutes),
    }
}

/// Fetch candles for a given symbol and timeframe from DB.
/// Returns up to `limit` rows ordered by time ASC.
pub async fn fetch_candles(
    pool: &PgPool,
    symbol: &str,
    tf_minutes: i32,
    limit: usize,
) -> Result<Vec<Candle>> {
    let table = candle_table(tf_minutes)?;

    let sql = format!(
        r#"
        SELECT
            c.time, c.symbol, p.symbol_id,
            c.open, c.high, c.low, c.close, c.volume
        FROM {table} c
        JOIN market.pairs p ON p.symbol = c.symbol
        WHERE c.symbol = $1
        ORDER BY c.time ASC
        LIMIT $2
        "#
    );

    let rows = sqlx::query_as::<_, CandleRow>(&sql)
        .bind(symbol)
        .bind(limit as i64)
        .fetch_all(pool)
        .await?;

    Ok(rows.into_iter().map(|r| Candle {
        time: r.time, symbol: r.symbol, symbol_id: r.symbol_id,
        open: r.open, high: r.high, low: r.low, close: r.close, volume: r.volume,
    }).collect())
}

/// Bulk-fetch ALL candles for a given TF (all active symbols at once).
/// Returns data grouped by symbol. Uses ROW_NUMBER window function
/// for efficient batch fetch (1 query per TF instead of N queries per symbol).
pub async fn fetch_all_candles_for_tf(
    pool: &PgPool,
    tf_minutes: i32,
    limit_per_symbol: usize,
) -> Result<HashMap<String, Vec<Candle>>> {
    let table = candle_table(tf_minutes)?;

    let sql = format!(
        r#"
        WITH ranked AS (
            SELECT
                c.time, c.symbol, p.symbol_id,
                c.open, c.high, c.low, c.close, c.volume,
                ROW_NUMBER() OVER (PARTITION BY c.symbol ORDER BY c.time DESC) as rn
            FROM {table} c
            JOIN market.pairs p ON p.symbol = c.symbol AND p.is_active = true
        )
        SELECT time, symbol, symbol_id, open, high, low, close, volume
        FROM ranked
        WHERE rn <= $1
        ORDER BY symbol, time ASC
        "#
    );

    let rows = sqlx::query_as::<_, CandleRow>(&sql)
        .bind(limit_per_symbol as i64)
        .fetch_all(pool)
        .await?;

    let mut grouped: HashMap<String, Vec<Candle>> = HashMap::new();
    for r in rows {
        let candle = Candle {
            time: r.time, symbol: r.symbol.clone(), symbol_id: r.symbol_id,
            open: r.open, high: r.high, low: r.low, close: r.close, volume: r.volume,
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
