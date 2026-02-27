// strategies/ewmac_strategy/src/db_writer.rs
//
// Database Writer for EWMAC signals
//
// Batch-inserts generated signals into trade.ewmac_signals using UNNEST
// for maximum throughput (single RTT per batch).

use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::PgPool;
use tracing::{info, warn};

use crate::signal_generator::EwmacSignal;

/// Batch-insert EWMAC signals into trade.ewmac_signals.
///
/// Uses UNNEST for efficient batch INSERT (single SQL round-trip per batch).
/// ON CONFLICT: updates if a signal with same (symbol_id, tf_minutes, time) exists.
pub async fn insert_signals_batch(pool: &PgPool, signals: &[EwmacSignal]) -> Result<usize> {
    if signals.is_empty() {
        return Ok(0);
    }

    let mut inserted = 0;

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
        let mut raw_signal_v: Vec<f64> = Vec::with_capacity(chunk.len());
        let mut norm_signal_v: Vec<f64> = Vec::with_capacity(chunk.len());
        let mut forecast_v: Vec<f64> = Vec::with_capacity(chunk.len());
        let mut strength_v: Vec<f32> = Vec::with_capacity(chunk.len());
        let mut e_2_8_v: Vec<Option<f64>> = Vec::with_capacity(chunk.len());
        let mut e_4_16_v: Vec<Option<f64>> = Vec::with_capacity(chunk.len());
        let mut e_8_32_v: Vec<Option<f64>> = Vec::with_capacity(chunk.len());
        let mut e_16_64_v: Vec<Option<f64>> = Vec::with_capacity(chunk.len());
        let mut e_32_128_v: Vec<Option<f64>> = Vec::with_capacity(chunk.len());
        let mut e_64_256_v: Vec<Option<f64>> = Vec::with_capacity(chunk.len());
        let mut atr_pct_v: Vec<f32> = Vec::with_capacity(chunk.len());
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
            raw_signal_v.push(sig.raw_signal);
            norm_signal_v.push(sig.norm_signal);
            forecast_v.push(sig.forecast);
            strength_v.push(sig.signal_strength);
            e_2_8_v.push(sig.ewmac_2_8);
            e_4_16_v.push(sig.ewmac_4_16);
            e_8_32_v.push(sig.ewmac_8_32);
            e_16_64_v.push(sig.ewmac_16_64);
            e_32_128_v.push(sig.ewmac_32_128);
            e_64_256_v.push(sig.ewmac_64_256);
            atr_pct_v.push(sig.atr_pct);
            strategy_v.push("ewmac_v1".to_string());
            reason_v.push(sig.reason.clone());
        }

        let result = sqlx::query(
            r#"
            INSERT INTO trade.ewmac_signals
            (time, time_ms, symbol, symbol_id, tf_minutes, side,
             entry_price, sl_price, tp_price,
             raw_signal, norm_signal, forecast, signal_strength,
             ewmac_2_8, ewmac_4_16, ewmac_8_32,
             ewmac_16_64, ewmac_32_128, ewmac_64_256,
             atr_pct, strategy, reason)
            SELECT * FROM UNNEST(
                $1::timestamptz[], $2::bigint[], $3::text[], $4::bigint[],
                $5::smallint[], $6::smallint[],
                $7::float8[], $8::float8[], $9::float8[],
                $10::float8[], $11::float8[], $12::float8[], $13::real[],
                $14::float8[], $15::float8[], $16::float8[],
                $17::float8[], $18::float8[], $19::float8[],
                $20::real[], $21::text[], $22::jsonb[]
            )
            ON CONFLICT (symbol_id, tf_minutes, time) DO UPDATE SET
                side = EXCLUDED.side,
                entry_price = EXCLUDED.entry_price,
                sl_price = EXCLUDED.sl_price,
                tp_price = EXCLUDED.tp_price,
                raw_signal = EXCLUDED.raw_signal,
                norm_signal = EXCLUDED.norm_signal,
                forecast = EXCLUDED.forecast,
                signal_strength = EXCLUDED.signal_strength,
                ewmac_2_8 = EXCLUDED.ewmac_2_8,
                ewmac_4_16 = EXCLUDED.ewmac_4_16,
                ewmac_8_32 = EXCLUDED.ewmac_8_32,
                ewmac_16_64 = EXCLUDED.ewmac_16_64,
                ewmac_32_128 = EXCLUDED.ewmac_32_128,
                ewmac_64_256 = EXCLUDED.ewmac_64_256,
                atr_pct = EXCLUDED.atr_pct,
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
        .bind(&raw_signal_v)
        .bind(&norm_signal_v)
        .bind(&forecast_v)
        .bind(&strength_v)
        .bind(&e_2_8_v)
        .bind(&e_4_16_v)
        .bind(&e_8_32_v)
        .bind(&e_16_64_v)
        .bind(&e_32_128_v)
        .bind(&e_64_256_v)
        .bind(&atr_pct_v)
        .bind(&strategy_v)
        .bind(&reason_v)
        .execute(pool)
        .await;

        match result {
            Ok(r) => {
                inserted += r.rows_affected() as usize;
            }
            Err(e) => {
                warn!("Failed to insert ewmac_signals batch: {}", e);
                return Err(e.into());
            }
        }
    }

    Ok(inserted)
}

/// Count total signals in the table
pub async fn count_signals(pool: &PgPool) -> Result<i64> {
    let (count,): (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM trade.ewmac_signals"
    )
    .fetch_one(pool)
    .await?;

    Ok(count)
}

/// Create the table if it doesn't exist (fallback for when init_db hasn't run)
pub async fn ensure_table_exists(pool: &PgPool) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS trade.ewmac_signals (
            time            TIMESTAMPTZ NOT NULL,
            time_ms         BIGINT NOT NULL,
            symbol          TEXT NOT NULL,
            symbol_id       BIGINT NOT NULL,
            tf_minutes      SMALLINT NOT NULL,
            side            SMALLINT NOT NULL,
            entry_price     FLOAT8 NOT NULL,
            sl_price        FLOAT8 NOT NULL,
            tp_price        FLOAT8 NOT NULL,
            raw_signal      FLOAT8 NOT NULL DEFAULT 0.0,
            norm_signal     FLOAT8 NOT NULL DEFAULT 0.0,
            forecast        FLOAT8 NOT NULL DEFAULT 0.0,
            signal_strength REAL NOT NULL DEFAULT 0.0,
            ewmac_2_8      FLOAT8,
            ewmac_4_16     FLOAT8,
            ewmac_8_32     FLOAT8,
            ewmac_16_64    FLOAT8,
            ewmac_32_128   FLOAT8,
            ewmac_64_256   FLOAT8,
            atr_pct         REAL NOT NULL DEFAULT 0.0,
            strategy        TEXT NOT NULL DEFAULT 'ewmac_v1',
            reason          JSONB,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol_id, tf_minutes, time)
        )
        "#,
    )
    .execute(pool)
    .await?;

    info!("trade.ewmac_signals table ensured");
    Ok(())
}
