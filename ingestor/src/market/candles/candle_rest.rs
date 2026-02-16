/*
 * FILE: candle_rest.rs
 *
 * PURPOSE:
 * This module handles historical candle data retrieval from REST API endpoints.
 * It manages the downloading of past candle data for backfilling the database.
 *
 * RESPONSIBILITIES:
 * - Fetches historical candle data from Binance REST API
 * - Manages rate limiting for API requests
 * - Performs backfill operations for missing historical data
 * - Provides direct bulk insertion for initial data loads
 *
 * WORKFLOW:
 * 1. pg_connect() establishes database connection
 * 2. fetch_active_pairs() gets list of trading pairs to process
 * 3. fetch_last_time_per_symbol() determines what data is missing
 * 4. rest_fetch_klines_bytes() downloads data from REST API
 * 5. rest_backfill_one() performs incremental backfill for each pair
 * 6. copy_direct_empty_db() handles fast initial data loading
 */
use crate::market::candles::candle_common::*;
use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};
use common::AppConfig;
use common::timeframe::TimeFrame;
use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio::time::Duration;
use tokio_postgres::NoTls;
use tokio_postgres::binary_copy::BinaryCopyInWriter;
use tokio_postgres::types::Type;

pub async fn pg_connect(db_url: &str) -> Result<tokio_postgres::Client> {
    let (client, connection) = tokio_postgres::connect(db_url, NoTls)
        .await
        .context("tokio_postgres::connect failed")?;

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::error!("postgres connection error: {e}");
        }
    });

    Ok(client)
}

pub async fn fetch_active_pairs(client: &tokio_postgres::Client) -> Result<Vec<PairInfo>> {
    let row = client
        .query_one("SELECT to_regclass('market.pairs')::text", &[])
        .await
        .context("failed to check market.pairs existence")?;
    let reg: Option<String> = row.get(0);
    if reg.is_none() {
        return Err(anyhow::anyhow!(
            "DB schema is not initialized: relation market.pairs does not exist"
        ));
    }

    let rows = client
        .query(
            "SELECT symbol_id, symbol
             FROM market.pairs
             WHERE is_active = TRUE
             ORDER BY symbol_id",
            &[],
        )
        .await
        .context("fetch_active_pairs query failed")?;

    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        out.push(PairInfo {
            symbol_id: r.get::<_, i64>(0),
            symbol: r.get::<_, String>(1),
        });
    }
    Ok(out)
}

pub async fn fetch_last_time_per_symbol(
    client: &tokio_postgres::Client,
    tf: TimeFrame,
) -> Result<HashMap<i64, i64>> {
    let table = build_table_name(tf);
    let q = format!(
        "SELECT symbol_id, COALESCE(MAX(time_ms),0)::bigint AS last_ms
         FROM {}
         GROUP BY symbol_id",
        table
    );

    let rows = client
        .query(&q, &[])
        .await
        .with_context(|| format!("fetch_last_time_per_symbol failed for {}", table))?;

    let mut map = HashMap::with_capacity(rows.len().max(1024));
    for r in rows {
        let sid: i64 = r.get(0);
        let last_ms: i64 = r.get(1);
        map.insert(sid, last_ms);
    }
    Ok(map)
}

pub async fn rest_fetch_klines_bytes(
    http: &reqwest::Client,
    cfg: &AppConfig,
    symbol: &str,
    interval: &str,
    start_time_ms: Option<i64>,
    limit: usize,
    limiter: Option<&WeightLimiter>,
) -> Result<bytes::Bytes> {
    let mut url = format!(
        "{}/fapi/v1/klines?symbol={}&interval={}&limit={}",
        cfg.binance.rest_base_url, symbol, interval, limit
    );
    if let Some(st) = start_time_ms {
        url.push_str(&format!("&startTime={}", st));
    }

    let mut backoff = cfg.binance.http_retry_backoff_ms.max(50);
    for attempt in 1..=cfg.binance.http_retries.max(1) {
        if let Some(lim) = limiter {
            lim.acquire(klines_weight(limit)).await;
        }

        let resp = http.get(&url).send().await;

        match resp {
            Ok(r) => {
                let status = r.status();
                if status.as_u16() == 429 {
                    // Binance: 429 = rate limit hit, рекомендовано backoff; 418 = IP ban.
                    tracing::warn!(
                        "REST 429 on {} {}: attempt={} backoff={:?} body={}",
                        symbol, interval, attempt, backoff, r.text().await.unwrap_or_default()
                    );
                    tokio::time::sleep(Duration::from_millis(backoff)).await;
                    backoff = (backoff * 2).min(cfg.binance.http_retry_backoff_max_ms);
                    continue;
                }
                if status.as_u16() == 418 {
                    return Err(anyhow::anyhow!("REST 418 (ban) on {} {}: {}", symbol, interval, r.text().await.unwrap_or_default()));
                }

                match r.error_for_status() {
                    Ok(ok) => return Ok(ok.bytes().await.context("read bytes failed")?),
                    Err(e) => {
                        tracing::warn!(
                            "REST klines {} {} attempt={} status_err={}",
                            symbol, interval, attempt, e
                        );
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    "REST klines {} {} attempt={} req_err={}",
                    symbol, interval, attempt, e
                );
            }
        }

        tokio::time::sleep(Duration::from_millis(backoff)).await;
        backoff = (backoff * 2).min(cfg.binance.http_retry_backoff_max_ms);
    }

    Err(anyhow::anyhow!("REST klines failed after retries: {} {}", symbol, interval))
}

use crate::market::candles::candle_writer::{TypedWriterMsg, DataSource};

pub async fn rest_backfill_one(
    http: &reqwest::Client,
    cfg: &AppConfig,
    pair: &PairInfo,
    tf: TimeFrame,
    last_ms: i64,
    tx: mpsc::UnboundedSender<TypedWriterMsg>,
    limiter: Option<WeightLimiter>,
) -> Result<()> {
    let interval = tf.as_str();
    // Делаем limit <= 500 => weight=2 вместо weight=5 (если 700).
    // Это обычно выгоднее по "весу на свечу" и снижает вероятность 429.
    // Use env override BACKFILL_CANDLES if set, otherwise config value. Max 1500 (Binance API limit)
    let backfill: usize = std::env::var("BACKFILL_CANDLES")
        .ok().and_then(|v| v.parse().ok())
        .unwrap_or(cfg.runtime.backfill_candles);
    let limit = backfill.min(1500).max(1);

    let max_loops: usize = std::env::var("INGEST_BACKFILL_MAX_LOOPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);

    // Всегда задаём startTime:
    // - если last_ms есть: догоняем с last_ms+1
    // - если last_ms нет/0: берём окно "последние backfill_candles"
    // плюс cap: даже если база отстала на недели — не пытаемся одним запуском догнать всё.
    let now_ms = Utc::now().timestamp_millis();
    let tf_ms = (tf.to_minutes() as i64) * 60_000;
    let cap_start = now_ms - (cfg.runtime.backfill_candles as i64) * tf_ms;
    let mut start = if last_ms > 0 {
        (last_ms + 1).max(cap_start)
    } else {
        cap_start
    };
    if start < 0 { start = 0; }

    let mut loops = 0usize;
    let mut max_close = last_ms;

    while loops < max_loops {
        loops += 1;

        let bytes = rest_fetch_klines_bytes(http, cfg, &pair.symbol, interval, Some(start), limit, limiter.as_ref()).await?;
        let klines: Vec<RestKline> =
            serde_json::from_slice(&bytes).context("parse REST klines failed")?;
        if klines.is_empty() {
            break;
        }

        let mut out: Vec<CandleRow> = Vec::with_capacity(klines.len());

        for k in &klines {
            let close_time = k.6;

            if close_time >= Utc::now().timestamp_millis() {
                continue;
            }

            if close_time <= last_ms {
                continue;
            }
            if close_time > max_close {
                max_close = close_time;
            }

            out.push(CandleRow {
                time_ms: close_time,
                symbol_id: pair.symbol_id,
                symbol: pair.symbol.clone(),
                open: str_f64(k.1.as_ref()),
                high: str_f64(k.2.as_ref()),
                low: str_f64(k.3.as_ref()),
                close: str_f64(k.4.as_ref()),
                volume: str_f64(k.5.as_ref()),
            });
        }

        if !out.is_empty() {
            let _ = tx.send(TypedWriterMsg { rows: out, source: DataSource::Rest });
        }

        if klines.len() < limit {
            break;
        }
        start = max_close + 1;
    }

    Ok(())
}

/// Performs a direct bulk insert of candle data using PostgreSQL's COPY command for empty DB
///
/// This function implements an efficient bulk insert mechanism using PostgreSQL's binary COPY protocol
/// without staging tables. It's designed for the empty database scenario where we don't need
/// to worry about conflicts or upserts.
///
/// # Arguments
/// * `client` - PostgreSQL client connection
/// * `table` - Target table name (e.g., "candles_1m")
/// * `rows` - Mutable reference to vector of CandleRow structs to insert
///
/// # Returns
/// * Result containing the number of rows copied on success, or an error
///
/// # Process
/// 1. Sorts and deduplicates rows to ensure data integrity
/// 2. Uses PostgreSQL's binary COPY protocol to efficiently transfer data directly to main table
pub async fn copy_direct_empty_db(
    client: &tokio_postgres::Client,
    table: &str,
    rows: &mut Vec<CandleRow>,
) -> Result<u64> {
    if rows.is_empty() {
        return Ok(0);
    }

    // Sort and deduplicate to guarantee no duplicates before insertion
    rows.sort_unstable_by(|a, b| {
        match a.symbol_id.cmp(&b.symbol_id) {
            std::cmp::Ordering::Equal => a.time_ms.cmp(&b.time_ms),
            other => other,
        }
    });
    rows.dedup_by(|a, b| a.symbol_id == b.symbol_id && a.time_ms == b.time_ms);

    let stmt = format!(
        "COPY {} (time_ms, time, symbol_id, symbol, open, high, low, close, volume) FROM STDIN BINARY",
        table
    );

    let sink = client.copy_in(&stmt).await
        .context("Failed to start COPY")?;

    let writer = BinaryCopyInWriter::new(
        sink,
        &[
            Type::INT8,        // time_ms
            Type::TIMESTAMPTZ, // time
            Type::INT8,        // symbol_id
            Type::TEXT,        // <--- ДОБАВЛЕНО: symbol
            Type::FLOAT8,      // open
            Type::FLOAT8,      // high
            Type::FLOAT8,      // low
            Type::FLOAT8,      // close
            Type::FLOAT8,      // volume
        ],
    );
    let mut writer = std::pin::pin!(writer);

    // Write all rows to the COPY stream
    for r in rows.iter() {
        let ts = Utc.timestamp_millis_opt(r.time_ms).single().unwrap();
        writer
            .as_mut()
            .write(&[
                &r.time_ms,
                &ts,
                &r.symbol_id,
                &r.symbol,     // <--- ДОБАВЛЕНО: передаем строку символа
                &r.open,
                &r.high,
                &r.low,
                &r.close,
                &r.volume,
            ])
            .await
            .context("Failed to write row to COPY")?;
    }

    let n = writer.as_mut().finish().await.context("Failed to finish COPY")?;
    rows.clear();
    Ok(n)
}