// strategies/ml_pump_dump/src/db_writer.rs
//
// Database Writer for Pump/Dump signals
//
// Batch-inserts generated signals into trade.pump_dump_signals using UNNEST
// for maximum throughput (single RTT per batch).

use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::PgPool;
use std::collections::HashMap;
use tracing::{info, warn};

use crate::signal_generator::PumpDumpSignal;

/// Create the table if it doesn't exist (fallback for when init_db hasn't run).
pub async fn ensure_table_exists(pool: &PgPool) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS trade.pump_dump_signals (
            time            TIMESTAMPTZ NOT NULL,
            time_ms         BIGINT NOT NULL,
            symbol          TEXT NOT NULL,
            symbol_id       BIGINT NOT NULL,
            tf_minutes      SMALLINT NOT NULL,
            side            SMALLINT NOT NULL,
            event_type      TEXT NOT NULL,
            entry_price     FLOAT8 NOT NULL,
            sl_price        FLOAT8 NOT NULL,
            tp_price        FLOAT8 NOT NULL,
            pred            REAL NOT NULL,
            finest_tf       SMALLINT NOT NULL,
            move_pct        FLOAT8 NOT NULL DEFAULT 0.0,
            max_hold_bars   SMALLINT NOT NULL DEFAULT 10,
            strategy        TEXT NOT NULL DEFAULT 'pump_dump_v1',
            reason          JSONB,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol_id, tf_minutes, time)
        )
        "#,
    )
    .execute(pool)
    .await?;

    info!("trade.pump_dump_signals table ensured");
    Ok(())
}

/// Deduplicate signals within a slice: if multiple signals share the same
/// (symbol_id, tf_minutes, time) key (e.g. a PUMP and DUMP fired simultaneously),
/// keep only the one with the highest prediction probability.
///
/// This prevents "ON CONFLICT DO UPDATE command cannot affect row a second time"
/// PostgreSQL error when the same key appears twice in a single UNNEST INSERT.
fn deduplicate_signals(signals: &[PumpDumpSignal]) -> Vec<&PumpDumpSignal> {
    let mut best: HashMap<(i64, i32, i64), &PumpDumpSignal> = HashMap::with_capacity(signals.len());
    for sig in signals {
        let key = (sig.symbol_id, sig.tf_minutes, sig.time_ms);
        let entry = best.entry(key).or_insert(sig);
        if sig.pred > entry.pred {
            *entry = sig;
        }
    }
    best.into_values().collect()
}

/// Batch-insert pump/dump signals into trade.pump_dump_signals.
///
/// Uses UNNEST for efficient batch INSERT (single SQL round-trip per batch).
/// Deduplicates within each batch to avoid "cannot affect row a second time".
/// ON CONFLICT: updates if a signal with same (symbol_id, tf_minutes, time) exists.
pub async fn insert_signals_batch(pool: &PgPool, signals: &[PumpDumpSignal]) -> Result<usize> {
    if signals.is_empty() {
        return Ok(0);
    }

    // Deduplicate: if PUMP + DUMP fire for the same (symbol_id, tf_minutes, time),
    // keep only the signal with the highest pred.
    let deduped = deduplicate_signals(signals);

    let mut inserted = 0;

    // Convert deduped refs to owned slice for chunking
    for chunk in deduped.chunks(2000) {
        let mut time_v: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());
        let mut time_ms_v: Vec<i64> = Vec::with_capacity(chunk.len());
        let mut symbol_v: Vec<String> = Vec::with_capacity(chunk.len());
        let mut symbol_id_v: Vec<i64> = Vec::with_capacity(chunk.len());
        let mut tf_v: Vec<i16> = Vec::with_capacity(chunk.len());
        let mut side_v: Vec<i16> = Vec::with_capacity(chunk.len());
        let mut event_type_v: Vec<String> = Vec::with_capacity(chunk.len());
        let mut entry_v: Vec<f64> = Vec::with_capacity(chunk.len());
        let mut sl_v: Vec<f64> = Vec::with_capacity(chunk.len());
        let mut tp_v: Vec<f64> = Vec::with_capacity(chunk.len());
        let mut pred_v: Vec<f32> = Vec::with_capacity(chunk.len());
        let mut finest_tf_v: Vec<i16> = Vec::with_capacity(chunk.len());
        let mut move_pct_v: Vec<f64> = Vec::with_capacity(chunk.len());
        let mut max_hold_v: Vec<i16> = Vec::with_capacity(chunk.len());
        let mut strategy_v: Vec<String> = Vec::with_capacity(chunk.len());
        let mut reason_v: Vec<serde_json::Value> = Vec::with_capacity(chunk.len());

        for &sig in chunk {
            time_v.push(sig.time);
            time_ms_v.push(sig.time_ms);
            symbol_v.push(sig.symbol.clone());
            symbol_id_v.push(sig.symbol_id);
            tf_v.push(sig.tf_minutes as i16);
            side_v.push(sig.side);
            event_type_v.push(sig.event_type.clone());
            entry_v.push(sig.entry_price);
            sl_v.push(sig.sl_price);
            tp_v.push(sig.tp_price);
            pred_v.push(sig.pred);
            finest_tf_v.push(sig.finest_tf as i16);
            move_pct_v.push(sig.move_pct);
            max_hold_v.push(sig.max_hold_bars);
            strategy_v.push("pump_dump_v1".to_string());
            reason_v.push(sig.reason.clone());
        }

        // Retry up to 3 times on deadlock
        let mut retries = 0;
        let max_retries = 3;
        loop {
            let result = sqlx::query(
                r#"
                INSERT INTO trade.pump_dump_signals
                (time, time_ms, symbol, symbol_id, tf_minutes, side, event_type,
                 entry_price, sl_price, tp_price,
                 pred, finest_tf, move_pct, max_hold_bars,
                 strategy, reason)
                SELECT * FROM UNNEST(
                    $1::timestamptz[], $2::bigint[], $3::text[], $4::bigint[],
                    $5::smallint[], $6::smallint[], $7::text[],
                    $8::float8[], $9::float8[], $10::float8[],
                    $11::real[], $12::smallint[], $13::float8[], $14::smallint[],
                    $15::text[], $16::jsonb[]
                )
                ON CONFLICT (symbol_id, tf_minutes, time) DO UPDATE SET
                    side = EXCLUDED.side,
                    event_type = EXCLUDED.event_type,
                    entry_price = EXCLUDED.entry_price,
                    sl_price = EXCLUDED.sl_price,
                    tp_price = EXCLUDED.tp_price,
                    pred = EXCLUDED.pred,
                    finest_tf = EXCLUDED.finest_tf,
                    move_pct = EXCLUDED.move_pct,
                    max_hold_bars = EXCLUDED.max_hold_bars,
                    reason = EXCLUDED.reason
                "#,
            )
            .bind(&time_v)
            .bind(&time_ms_v)
            .bind(&symbol_v)
            .bind(&symbol_id_v)
            .bind(&tf_v)
            .bind(&side_v)
            .bind(&event_type_v)
            .bind(&entry_v)
            .bind(&sl_v)
            .bind(&tp_v)
            .bind(&pred_v)
            .bind(&finest_tf_v)
            .bind(&move_pct_v)
            .bind(&max_hold_v)
            .bind(&strategy_v)
            .bind(&reason_v)
            .execute(pool)
            .await;

            match result {
                Ok(r) => {
                    inserted += r.rows_affected() as usize;
                    break;
                }
                Err(e) => {
                    let err_str = e.to_string();
                    if err_str.contains("deadlock") && retries < max_retries {
                        retries += 1;
                        warn!("Deadlock on pump_dump_signals insert, retry {}/{}", retries, max_retries);
                        tokio::time::sleep(std::time::Duration::from_millis(50 * retries as u64)).await;
                        continue;
                    }
                    warn!("Failed to insert pump_dump_signals batch: {}", e);
                    return Err(e.into());
                }
            }
        }
    }

    Ok(inserted)
}

/// Count total signals in the table.
pub async fn count_signals(pool: &PgPool) -> Result<i64> {
    let (count,): (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM trade.pump_dump_signals"
    )
    .fetch_one(pool)
    .await?;

    Ok(count)
}
