/*
 * FILE: candle_writer.rs
 *
 * PURPOSE:
 * This module handles database writing operations for candle data.
 * It manages the insertion of both historical (REST) and real-time (WebSocket) data into the database.
 *
 * RESPONSIBILITIES:
 * - Writes candle data to PostgreSQL database using binary COPY
 * - Implements different flushing strategies for REST vs WebSocket data
 * - Manages database connections and transactions
 * - Handles data preparation and deduplication
 * - Provides separate buffers for different data sources
 * - Implements live candle writer for real-time updates to market.candles_live table
 *
 * WORKFLOW:
 * 1. writer_task_copy() initializes database connection
 * 2. Sets up separate buffers for REST and WebSocket data
 * 3. Uses different flush thresholds for each data type
 * 4. flush_copy() performs the actual database insertion
 * 5. Data is filtered, sorted, and deduplicated before insertion
 * 6. Separate timers control flushing for each data type
 * 7. LiveWriter handles real-time updates to market.candles_live table
 */
use crate::market::candles::candle_common::*;
use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};
use common::timeframe::TimeFrame;
use std::collections::HashMap;
use tokio::sync::{mpsc, watch};
use tokio::time::{Duration, Instant};
use tokio_postgres::binary_copy::BinaryCopyInWriter;
use tokio_postgres::types::Type;
use sqlx::{PgPool, Postgres, QueryBuilder};
use std::time::{Duration as StdDuration};
use tracing::debug;

// Different message types for REST and WebSocket data to handle them differently
#[derive(Debug, Clone)]
pub enum DataSource {
    Rest,      // Historical data from REST API - use large batches
    WebSocket, // Real-time data from WebSocket - use low latency
}

#[derive(Debug, Clone)]
pub struct TypedWriterMsg {
    pub rows: Vec<CandleRow>,
    pub source: DataSource,
}

#[derive(Debug, Clone)]
pub struct LiveCandle {
    pub symbol: String,
    pub timeframe: String,
    pub open_time_ms: i64,
    pub close_time_ms: i64,

    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,

    pub trades: Option<i32>,
    pub is_final: bool,

    pub event_time_ms: i64,
    pub source: &'static str, // "ws"
}


fn merge_live(dst: &mut LiveCandle, src: LiveCandle) {
    // open_time совпадает по ключу
    dst.close_time_ms = src.close_time_ms;
    // open usually the same, but keep dst.open
    dst.high = dst.high.max(src.high);
    dst.low = dst.low.min(src.low);
    // close/volume — берём самые свежие
    if src.event_time_ms >= dst.event_time_ms {
        dst.close = src.close;
        dst.volume = src.volume;
        dst.trades = src.trades.or(dst.trades);
        dst.event_time_ms = src.event_time_ms;
    } else {
        // если прилетело старое — всё равно high/low уже учли
        dst.volume = dst.volume.max(src.volume);
    }
    dst.is_final = dst.is_final || src.is_final;
}

pub struct LiveWriter {
    pool: PgPool,
    flush_every: StdDuration,
    max_buf: usize,
}

impl LiveWriter {
    pub fn new(pool: PgPool, flush_every_ms: u64, max_buf: usize) -> Self {
        Self {
            pool,
            flush_every: StdDuration::from_millis(flush_every_ms),
            max_buf,
        }
    }

    pub async fn run(self, mut rx: mpsc::Receiver<LiveCandle>) -> Result<()> {
        let mut tick = tokio::time::interval(self.flush_every);
        let mut buf: HashMap<(String, String, i64), LiveCandle> = HashMap::with_capacity(self.max_buf);
        let mut last_flush = Instant::now();

        loop {
            tokio::select! {
                _ = tick.tick() => {
                    if !buf.is_empty() {
                        let n = flush_upsert_live(&self.pool, &mut buf).await?;
                        debug!("candles_live flushed rows={}", n);
                        last_flush = Instant::now();
                    }
                }
                maybe = rx.recv() => {
                    let c = match maybe {
                        Some(v) => v,
                        None => break, // channel closed
                    };
                    let k = (c.symbol.clone(), c.timeframe.clone(), c.open_time_ms);
                    match buf.get_mut(&k) {
                        Some(existing) => merge_live(existing, c),
                        None => { buf.insert(k, c); }
                    }

                    // backpressure guard: если переполнились — принудительный flush
                    if buf.len() >= self.max_buf || last_flush.elapsed() > self.flush_every {
                        let _ = flush_upsert_live(&self.pool, &mut buf).await?;
                        last_flush = Instant::now();
                    }
                }
            }
        }

        // финальный flush
        if !buf.is_empty() {
            let _ = flush_upsert_live(&self.pool, &mut buf).await?;
        }

        Ok(())
    }
}

async fn flush_upsert_live(pool: &PgPool, buf: &mut HashMap<(String, String, i64), LiveCandle>) -> Result<u64> {
    let rows: Vec<LiveCandle> = buf.drain().map(|(_, v)| v).collect();
    if rows.is_empty() {
        return Ok(0);
    }

    // Один большой UPSERT батчен:
    // INSERT ... VALUES (...),(...),...
    // ON CONFLICT (symbol,timeframe,open_time_ms) DO UPDATE SET ...
    let mut qb: QueryBuilder<Postgres> = QueryBuilder::new(
        "INSERT INTO market.candles_live
         (symbol,timeframe,open_time_ms,close_time_ms,open,high,low,close,volume,trades,is_final,last_event_time_ms,updated_at,source) "
    );

    qb.push_values(rows.iter(), |mut b, r| {
        b.push_bind(&r.symbol)
         .push_bind(&r.timeframe)
         .push_bind(r.open_time_ms)
         .push_bind(r.close_time_ms)
         .push_bind(r.open)
         .push_bind(r.high)
         .push_bind(r.low)
         .push_bind(r.close)
         .push_bind(r.volume)
         .push_bind(r.trades)
         .push_bind(r.is_final)
         .push_bind(r.event_time_ms)
         .push_bind(Utc::now())
         .push_bind(r.source);
    });

    qb.push(
        " ON CONFLICT (symbol,timeframe,open_time_ms) DO UPDATE SET
            close_time_ms = EXCLUDED.close_time_ms,
            open          = EXCLUDED.open,
            high          = GREATEST(market.candles_live.high, EXCLUDED.high),
            low           = LEAST(market.candles_live.low, EXCLUDED.low),
            close         = EXCLUDED.close,
            volume        = GREATEST(market.candles_live.volume, EXCLUDED.volume),
            trades        = COALESCE(EXCLUDED.trades, market.candles_live.trades),
            is_final      = (market.candles_live.is_final OR EXCLUDED.is_final),
            last_event_time_ms = GREATEST(market.candles_live.last_event_time_ms, EXCLUDED.last_event_time_ms),
            updated_at    = now(),
            source        = EXCLUDED.source"
    );

    let res = qb.build().execute(pool).await.context("upsert candles_live failed")?;
    Ok(res.rows_affected())
}

pub async fn flush_copy(
    client: &tokio_postgres::Client,
    tf: TimeFrame,
    buf: &mut Vec<CandleRow>,
    last_seen: &mut HashMap<i64, i64>,
) -> Result<u64> {
    if buf.is_empty() {
        return Ok(0);
    }

    prepare_copy_batch(buf, last_seen);
    if buf.is_empty() {
        return Ok(0);
    }

    let mut max_per_symbol: HashMap<i64, i64> = HashMap::new();
    for c in buf.iter() {
        let e = max_per_symbol.entry(c.symbol_id).or_insert(0);
        if c.time_ms > *e {
            *e = c.time_ms;
        }
    }

    let table_name = build_table_name(tf);
    let stmt = format!(
        "COPY {} (time_ms, time, symbol_id, open, high, low, close, volume) FROM STDIN BINARY",
        table_name
    );

    let sink = client.copy_in(&stmt).await.context("copy_in failed")?;
    let writer = BinaryCopyInWriter::new(
        sink,
        &[
            Type::INT8,   // time_ms
            Type::TIMESTAMPTZ, // time
            Type::INT8,   // symbol_id
            Type::FLOAT8, // open
            Type::FLOAT8, // high
            Type::FLOAT8, // low
            Type::FLOAT8, // close
            Type::FLOAT8, // volume
        ],
    );
    let mut writer = std::pin::pin!(writer);

    for r in buf.iter() {
        let ts = Utc.timestamp_millis_opt(r.time_ms).single().unwrap();
        writer
            .as_mut()
            .write(&[
                &r.time_ms,
                &ts,
                &r.symbol_id,
                &r.open,
                &r.high,
                &r.low,
                &r.close,
                &r.volume,
            ])
            .await?;
    }

    let rows = writer.as_mut().finish().await?;

    for (sid, mx) in max_per_symbol {
        let e = last_seen.entry(sid).or_insert(0);
        if mx > *e {
            *e = mx;
        }
    }

    buf.clear();
    Ok(rows)
}

pub async fn writer_task_copy(
    db_url: String,
    tf: TimeFrame,
    mut rx: mpsc::UnboundedReceiver<TypedWriterMsg>, // Using unbounded channel for better performance with WS
    mut last_seen: HashMap<i64, i64>,
    mut shutdown: watch::Receiver<bool>,
) -> Result<()> {
    let flush_rows: usize = std::env::var("INGEST_COPY_FLUSH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30_000);

    let flush_every_ms: u64 = std::env::var("INGEST_COPY_FLUSH_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(250);

    // Different thresholds for REST vs WebSocket data
    let rest_min_flush_rows: usize = std::env::var("INGEST_COPY_REST_MIN_FLUSH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);

    let ws_min_flush_rows: usize = std::env::var("INGEST_COPY_WS_MIN_FLUSH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10); // Much lower threshold for WS data

    let ws_flush_every_ms: u64 = std::env::var("INGEST_COPY_WS_FLUSH_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50); // More frequent flush for WS data

    let mut inserted_total: u64 = 0;
    let mut last_log = Instant::now();

    loop {
        // Reconnect loop
        if *shutdown.borrow() {
            return Ok(());
        }

        let client = match crate::market::candles::candle_rest::pg_connect(&db_url).await {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(?e, "pg_connect failed in writer task");
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
        };

        if std::env::var("INGEST_SYNC_COMMIT_OFF").ok().as_deref() == Some("1") {
            if let Err(e) = client.batch_execute("SET synchronous_commit = OFF;").await {
                tracing::error!(?e, "Failed to set synchronous_commit=OFF");
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue; // reconnect
            }
            tracing::warn!("writer({}): synchronous_commit=OFF", tf.as_str());
        }


        // Separate buffers for REST and WebSocket data to handle them differently
        let mut rest_buf: Vec<CandleRow> = Vec::with_capacity(flush_rows);
        let mut ws_buf: Vec<CandleRow> = Vec::with_capacity(1000); // Smaller buffer for WS

        // Use different intervals for REST and WS data
        let mut rest_tick = tokio::time::interval(Duration::from_millis(flush_every_ms));
        rest_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        let mut ws_tick = tokio::time::interval(Duration::from_millis(ws_flush_every_ms));
        ws_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        let res: Result<()> = loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        break Ok(());
                    }
                }

                _ = rest_tick.tick() => {
                    // Handle REST data with larger batch requirements
                    if rest_buf.len() >= rest_min_flush_rows {
                        match flush_copy(&client, tf, &mut rest_buf, &mut last_seen).await {
                            Ok(n) => {
                                inserted_total += n;
                                if last_log.elapsed() >= Duration::from_secs(2) {
                                    tracing::info!("writer({}): inserted {} REST rows, total {}", tf.as_str(), n, inserted_total);
                                    last_log = Instant::now();
                                }
                            }
                            Err(e) => {
                                tracing::error!(tf=?tf, error=format!("{:#}", e), "writer REST flush failed");
                                break Err(e); // Break inner loop to reconnect
                            }
                        }
                    }
                }

                _ = ws_tick.tick() => {
                    // Handle WebSocket data with low-latency requirements
                    if !ws_buf.is_empty() {
                        match flush_copy(&client, tf, &mut ws_buf, &mut last_seen).await {
                            Ok(n) => {
                                inserted_total += n;
                                if last_log.elapsed() >= Duration::from_secs(2) {
                                    tracing::info!("writer({}): inserted {} WS rows, total {}", tf.as_str(), n, inserted_total);
                                    last_log = Instant::now();
                                }
                            }
                            Err(e) => {
                                tracing::error!(tf=?tf, error=format!("{:#}", e), "writer WS flush failed");
                                break Err(e); // Break inner loop to reconnect
                            }
                        }
                    }
                }

                msg = rx.recv() => {
                    tracing::trace!("writer({}) received message: {:?}", tf.as_str(), msg);
                    match msg {
                        Some(m) => {
                            match m.source {
                                DataSource::Rest => {
                                    rest_buf.extend(m.rows);
                                    if rest_buf.len() >= flush_rows {
                                        match flush_copy(&client, tf, &mut rest_buf, &mut last_seen).await {
                                            Ok(n) => inserted_total += n,
                                            Err(e) => {
                                                tracing::error!(tf=?tf, error=format!("{:#}", e), "writer REST flush failed");
                                                break Err(e); // Break inner loop to reconnect
                                            }
                                        }
                                    }
                                },
                                DataSource::WebSocket => {
                                    ws_buf.extend(m.rows);
                                    // For WebSocket data, flush immediately or when buffer is reasonably full
                                    if ws_buf.len() >= ws_min_flush_rows {
                                        match flush_copy(&client, tf, &mut ws_buf, &mut last_seen).await {
                                            Ok(n) => inserted_total += n,
                                            Err(e) => {
                                                tracing::error!(tf=?tf, error=format!("{:#}", e), "writer WS flush failed");
                                                break Err(e); // Break inner loop to reconnect
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        None => {
                            // Channel closed, probably shutdown
                             break Ok(());
                        }
                    }
                }
            }
        };

        // Final flush before breaking outer loop or reconnecting
        if !rest_buf.is_empty() {
             match flush_copy(&client, tf, &mut rest_buf, &mut last_seen).await {
                Ok(n) => inserted_total += n,
                Err(e) => {
                    tracing::error!(tf=?tf, error=format!("{:#}", e), "writer REST final flush failed");
                }
            }
        }

        if !ws_buf.is_empty() {
             match flush_copy(&client, tf, &mut ws_buf, &mut last_seen).await {
                Ok(n) => inserted_total += n,
                Err(e) => {
                    tracing::error!(tf=?tf, error=format!("{:#}", e), "writer WS final flush failed");
                }
            }
        }

        if res.is_ok() {
            // Channel closed, so exit.
            tracing::info!("writer({}) channel closed. inserted_total={}", tf.as_str(), inserted_total);
            return Ok(());
        }

        // Error happened, sleep before reconnect
        tracing::warn!("writer({}) reconnecting after error...", tf.as_str());
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}