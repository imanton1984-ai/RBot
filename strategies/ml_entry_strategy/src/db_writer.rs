// strategies/ml_entry_strategy/src/db_writer.rs
//
// Database Writer for Super Entry signals
//
// Batch-inserts generated signals into trade.super_entry_signals using UNNEST
// for maximum throughput (single RTT per batch).

use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::PgPool;
use tracing::{info, warn};

use crate::signal_generator::SuperEntrySignal;

/// Batch-insert super entry signals into trade.super_entry_signals.
///
/// Uses UNNEST for efficient batch INSERT (single SQL round-trip per batch).
/// ON CONFLICT: updates if a signal with same (symbol_id, tf_minutes, time) exists.
pub async fn insert_signals_batch(pool: &PgPool, signals: &[SuperEntrySignal]) -> Result<usize> {
    if signals.is_empty() {
        return Ok(0);
    }

    let mut inserted = 0;

    // Use larger chunks for bulk inserts (UNNEST supports thousands of rows per RTT)
    for chunk in signals.chunks(2000) {
        let mut time_v: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());
        let mut time_ms_v: Vec<i64> = Vec::with_capacity(chunk.len());
        let mut symbol_v: Vec<String> = Vec::with_capacity(chunk.len());
        let mut symbol_id_v: Vec<i64> = Vec::with_capacity(chunk.len());
        let mut tf_v: Vec<i16> = Vec::with_capacity(chunk.len());
        let mut side_v: Vec<i16> = Vec::with_capacity(chunk.len());
        let mut entry_v: Vec<f64> = Vec::with_capacity(chunk.len());
        let mut sl_v: Vec<f64> = Vec::with_capacity(chunk.len());
        let mut tp_v: Vec<f64> = Vec::with_capacity(chunk.len());
        let mut p_super_v: Vec<f32> = Vec::with_capacity(chunk.len());
        let mut p_long_v: Vec<f32> = Vec::with_capacity(chunk.len());
        let mut score_v: Vec<f32> = Vec::with_capacity(chunk.len());
        let mut dir_conf_v: Vec<f32> = Vec::with_capacity(chunk.len());
        let mut strategy_v: Vec<String> = Vec::with_capacity(chunk.len());
        let mut reason_v: Vec<serde_json::Value> = Vec::with_capacity(chunk.len());

        for sig in chunk {
            time_v.push(sig.time);
            time_ms_v.push(sig.time_ms);
            symbol_v.push(sig.symbol.clone());
            symbol_id_v.push(sig.symbol_id);
            tf_v.push(sig.tf_minutes as i16);
            side_v.push(sig.side);
            entry_v.push(sig.entry_price);
            sl_v.push(sig.sl_price);
            tp_v.push(sig.tp_price);
            p_super_v.push(sig.p_super);
            p_long_v.push(sig.p_long);
            score_v.push(sig.final_score);
            // Recover dir_confidence from p_long
            dir_conf_v.push((sig.p_long - 0.5).abs());
            strategy_v.push("super_entry_v1".to_string());
            reason_v.push(sig.reason.clone());
        }

        let result = sqlx::query(
            r#"
            INSERT INTO trade.super_entry_signals
            (time, time_ms, symbol, symbol_id, tf_minutes, side,
             entry_price, sl_price, tp_price,
             p_super, p_long, combined_score, dir_confidence,
             strategy, reason)
            SELECT * FROM UNNEST(
                $1::timestamptz[], $2::bigint[], $3::text[], $4::bigint[],
                $5::smallint[], $6::smallint[],
                $7::float8[], $8::float8[], $9::float8[],
                $10::real[], $11::real[], $12::real[], $13::real[],
                $14::text[], $15::jsonb[]
            )
            ON CONFLICT (symbol_id, tf_minutes, time) DO UPDATE SET
                side = EXCLUDED.side,
                entry_price = EXCLUDED.entry_price,
                sl_price = EXCLUDED.sl_price,
                tp_price = EXCLUDED.tp_price,
                p_super = EXCLUDED.p_super,
                p_long = EXCLUDED.p_long,
                combined_score = EXCLUDED.combined_score,
                dir_confidence = EXCLUDED.dir_confidence,
                reason = EXCLUDED.reason
            "#,
        )
        .bind(&time_v)
        .bind(&time_ms_v)
        .bind(&symbol_v)
        .bind(&symbol_id_v)
        .bind(&tf_v)
        .bind(&side_v)
        .bind(&entry_v)
        .bind(&sl_v)
        .bind(&tp_v)
        .bind(&p_super_v)
        .bind(&p_long_v)
        .bind(&score_v)
        .bind(&dir_conf_v)
        .bind(&strategy_v)
        .bind(&reason_v)
        .execute(pool)
        .await;

        match result {
            Ok(r) => {
                inserted += r.rows_affected() as usize;
            }
            Err(e) => {
                warn!("Failed to insert super_entry_signals batch: {}", e);
                return Err(e.into());
            }
        }
    }

    Ok(inserted)
}

/// Get the latest signal time for a given (symbol, tf) pair.
/// Returns None if no signals exist yet.
pub async fn get_latest_signal_time(
    pool: &PgPool,
    symbol: &str,
    tf_minutes: i32,
) -> Result<Option<DateTime<Utc>>> {
    let row: Option<(DateTime<Utc>,)> = sqlx::query_as(
        "SELECT time FROM trade.super_entry_signals \
         WHERE symbol = $1 AND tf_minutes = $2 \
         ORDER BY time DESC LIMIT 1"
    )
    .bind(symbol)
    .bind(tf_minutes as i16)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|(t,)| t))
}

/// Count total signals in the table
pub async fn count_signals(pool: &PgPool) -> Result<i64> {
    let (count,): (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM trade.super_entry_signals"
    )
    .fetch_one(pool)
    .await?;

    Ok(count)
}

/// Create the table if it doesn't exist (fallback for when init_db hasn't run)
pub async fn ensure_table_exists(pool: &PgPool) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS trade.super_entry_signals (
            time            TIMESTAMPTZ NOT NULL,
            time_ms         BIGINT NOT NULL,
            symbol          TEXT NOT NULL,
            symbol_id       BIGINT NOT NULL,
            tf_minutes      SMALLINT NOT NULL,
            side            SMALLINT NOT NULL,
            entry_price     FLOAT8 NOT NULL,
            sl_price        FLOAT8 NOT NULL,
            tp_price        FLOAT8 NOT NULL,
            p_super         REAL NOT NULL,
            p_long          REAL NOT NULL,
            combined_score  REAL NOT NULL,
            dir_confidence  REAL NOT NULL DEFAULT 0.0,
            strategy        TEXT NOT NULL DEFAULT 'super_entry_v1',
            reason          JSONB,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol_id, tf_minutes, time)
        )
        "#,
    )
    .execute(pool)
    .await?;

    info!("trade.super_entry_signals table ensured");
    Ok(())
}
