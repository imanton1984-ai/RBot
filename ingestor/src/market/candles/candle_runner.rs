/*
 * FILE: candle_runner.rs
 *
 * PURPOSE:
 * This module orchestrates the entire candle ingestion process.
 * It coordinates between REST data loading, WebSocket real-time updates, and database writing.
 *
 * RESPONSIBILITIES:
 * - Determines if database is empty or needs backfilling
 * - Coordinates historical data loading from REST API
 * - Initiates WebSocket connections for real-time data
 * - Sets up writer tasks for database insertion
 * - Manages overall ingestion workflow
 *
 * WORKFLOW:
 * 1. run_candles_ingest() starts the ingestion process
 * 2. Checks if database is empty to determine ingestion strategy
 * 3. For empty DB: performs fast initial load using REST API
 * 4. For populated DB: performs incremental backfill
 * 5. Starts WebSocket workers for real-time data (after backfill)
 * 6. Coordinates all writer tasks for database insertion
 */
use crate::market::candles::candle_common::*;
use crate::market::candles::candle_rest::*;
use crate::market::candles::candle_writer::*;
use anyhow::{Context, Result};
use common::{load_config, AppConfig};
use common::timeframe::TimeFrame;
use futures::{stream, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tokio::time::Instant;
use tokio::sync::Semaphore;

pub async fn run_candles_ingest() -> Result<()> {
    let cfg: AppConfig = load_config().context("load_config() failed")?;
    let cfg = Arc::new(cfg);

    let db_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| cfg.database.url());

    let http = reqwest::Client::builder()
        .timeout(tokio::time::Duration::from_millis(cfg.binance.http_timeout_ms.max(1_000)))
        .build()
        .context("reqwest build failed")?;

    let meta = crate::market::candles::candle_rest::pg_connect(&db_url).await?;
    let pairs = crate::market::candles::candle_rest::fetch_active_pairs(&meta).await?;
    if pairs.is_empty() {
        anyhow::bail!("No active pairs in market.pairs");
    }
    tracing::info!("Active pairs: {}", pairs.len());

    let mut map = HashMap::with_capacity(pairs.len() * 2);
    for p in &pairs {
        map.insert(p.symbol.clone(), p.symbol_id);
    }
    let symbol_to_id = Arc::new(map);

    let mut tfs: Vec<TimeFrame> = Vec::new();
    for s in &cfg.runtime.timeframes {
        if let Some(tf) = parse_tf(s) {
            tfs.push(tf);
        } else {
            tracing::warn!("Unknown timeframe in runtime.timeframes: {}", s);
        }
    }
    if tfs.is_empty() {
        anyhow::bail!("runtime.timeframes is empty or invalid");
    }

    let mut ws_tfs: Vec<TimeFrame> = Vec::new();
    for s in &cfg.runtime.realtime_ws_timeframes {
        if let Some(tf) = parse_tf(s) {
            ws_tfs.push(tf);
        } else {
            tracing::warn!("Unknown timeframe in runtime.realtime_ws_timeframes: {}", s);
        }
    }
    if ws_tfs.is_empty() {
        ws_tfs = tfs.clone();
    }

    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // writers (binary copy)
    let mut writers: HashMap<TimeFrame, mpsc::UnboundedSender<TypedWriterMsg>> = HashMap::new();
    for tf in tfs.iter().copied() {
        let (tx, rx) = mpsc::unbounded_channel::<TypedWriterMsg>();

        let last_seen = fetch_last_time_per_symbol(&meta, tf).await.unwrap_or_default();
        tracing::info!("last_seen loaded for {}: {}", tf.as_str(), last_seen.len());

        let db_url2 = db_url.clone();
        let shutdown_rx2 = shutdown_rx.clone();

        tokio::spawn(async move {
            if let Err(e) = writer_task_copy(db_url2, tf, rx, last_seen, shutdown_rx2).await {
                tracing::error!("writer({}) failed: {}", tf.as_str(), e);
            }
        });

        writers.insert(tf, tx);
    }
    let writers = Arc::new(writers);

    // NEW: Fast one-shot initial loading for empty database
    // This section handles the fast loading of historical data when the database is empty
    // It's designed to load all required candles quickly by:
    // 1. Making concurrent API requests for all pairs and timeframes
    // 2. Collecting all data in memory by timeframe
    // 3. Performing bulk inserts using PostgreSQL's COPY command
    let last_times = crate::market::candles::candle_rest::fetch_last_time_per_symbol(&meta, tfs[0]).await.unwrap_or_default();
    let is_empty_db = last_times.is_empty();

    if is_empty_db {
        tracing::info!("Empty database detected - performing fast one-shot initial load");

        // Use bounded concurrency semaphore for API requests
        // This limits the number of concurrent HTTP requests to prevent overwhelming the API
        let api_semaphore = Arc::new(Semaphore::new(cfg.runtime.http_concurrency.unwrap_or(16).max(1)));

        // ВНИМАНИЕ: здесь мы трактуем rate_limit_soft_rps как "weight units per second".
        // Для Futures дефолт: 40 weight/sec (2400/min).
        // Рекомендую держать 75-85% от лимита, чтобы не ловить 429/418.
        // This creates a weight-based rate limiter to comply with Binance's rate limits
        let rest_limiter = crate::market::candles::candle_common::WeightLimiter::new(
            cfg.binance.rate_limit_soft_rps,
            cfg.binance.rate_limit_soft_burst,
        );

        // Create a mapping of TimeFrame to collected candles
        // This organizes the fetched data by timeframe for efficient bulk insertion
        let mut tf_candles: HashMap<TimeFrame, Vec<CandleRow>> = HashMap::new();
        for tf in &tfs {
            tf_candles.insert(*tf, Vec::new());
        }

        // Concurrently fetch all candles for all pairs and timeframes
        // This maximizes throughput by making many requests in parallel
        let fetch_start = Instant::now();
        let mut fetch_jobs = Vec::new();

        // Create async jobs to fetch candles for each pair/timeframe combination
        for p in &pairs {
            for tf in &tfs {
                let http_clone = http.clone();
                let cfg_clone = cfg.clone();
                let pair_clone = p.clone();
                let semaphore_clone = api_semaphore.clone();
                let limiter_clone = rest_limiter.clone();

                fetch_jobs.push(async move {
                    // Acquire a permit from the concurrency semaphore
                    let _permit = semaphore_clone.acquire().await.unwrap();

                    // Single request to get the most recent candles for this pair and timeframe
                    // This fetches the configured number of recent candles (e.g., 700) in one request
                    let limit = cfg_clone.runtime.backfill_candles.min(1000).max(1) as usize;
                    let bytes = crate::market::candles::candle_rest::rest_fetch_klines_bytes(
                        &http_clone,
                        &cfg_clone,
                        &pair_clone.symbol,
                        tf.as_str(),
                        None, // No start time - get most recent
                        limit,
                        Some(&limiter_clone),
                    ).await?;

                    // Parse the JSON response into kline structures
                    let klines: Vec<RestKline> = serde_json::from_slice(&bytes).context("parse REST klines failed")?;

                    // Convert klines to CandleRow structures for database insertion
                    let mut results = Vec::new();
                    for k in &klines {
                        results.push(CandleRow {
                            time_ms: k.6, // close_time
                            symbol_id: pair_clone.symbol_id,
                            open: str_f64(k.1.as_ref()),
                            high: str_f64(k.2.as_ref()),
                            low: str_f64(k.3.as_ref()),
                            close: str_f64(k.4.as_ref()),
                            volume: str_f64(k.5.as_ref()),
                        });
                    }

                    // Return both the timeframe and the rows to fix the buffer_unordered issue
                    Ok::<(TimeFrame, Vec<CandleRow>), anyhow::Error>((*tf, results))
                });
            }
        }

        // Execute all fetch jobs concurrently with bounded concurrency
        let fetch_results = stream::iter(fetch_jobs)
            .buffer_unordered(cfg.runtime.http_concurrency.unwrap_or(16).max(1))
            .collect::<Vec<_>>()
            .await;

        // Organize results by timeframe for bulk insertion
        for result in fetch_results {
            match result {
                Ok((tf, rows)) => {
                    // Add the fetched candles to the appropriate timeframe bucket
                    tf_candles.get_mut(&tf).unwrap().extend(rows);
                }
                Err(e) => {
                    tracing::warn!("Failed to fetch candles: {}", e);
                }
            }
        }

        // Bulk insert all candles by timeframe using PostgreSQL's efficient direct COPY command
        let bulk_client = pg_connect(&db_url).await?;
        for tf in &tfs {
            let candles = tf_candles.get_mut(tf).unwrap();
            if !candles.is_empty() {
                let table_name = format!("market.candles_{}", tf.as_str());

                let start = Instant::now();
                // Perform the direct bulk insert using the efficient COPY mechanism (no staging/upsert)
                let inserted = crate::market::candles::candle_rest::copy_direct_empty_db(&bulk_client, &table_name, candles).await?;
                tracing::info!("Direct COPY inserted {} candles for {} in {:?}", inserted, table_name, start.elapsed());
            }
        }

        tracing::info!("Fast initial load completed in {:?}", fetch_start.elapsed());
    } else {
        // OLD: Traditional backfill approach for incremental updates
        // This section handles incremental updates when the database already contains data
        // It fills in gaps between the last stored data and current time
        tracing::info!("Non-empty database detected - performing traditional backfill");

        // REST backfill concurrent
        let http_conc = cfg.runtime.http_concurrency.unwrap_or(1).max(1) as usize;
        tracing::info!(
            "REST backfill: tfs={}, pairs={}, http_concurrency={}",
            tfs.len(),
            pairs.len(),
            http_conc
        );

        // ВНИМАНИЕ: здесь мы трактуем rate_limit_soft_rps как "weight units per second".
        // Для Futures дефолт: 40 weight/sec (2400/min).
        // Рекомендую держать 75-85% от лимита, чтобы не ловить 429/418.
        let rest_limiter = Some(crate::market::candles::candle_common::WeightLimiter::new(
            cfg.binance.rate_limit_soft_rps,
            cfg.binance.rate_limit_soft_burst,
        ));

        let mut jobs = Vec::new();
        for tf in tfs.iter().copied() {
            let tx = writers.get(&tf).cloned().unwrap();
            let last = fetch_last_time_per_symbol(&meta, tf).await.unwrap_or_default();

            for p in &pairs {
                let last_ms = last.get(&p.symbol_id).copied().unwrap_or(0);
                let cfg2 = cfg.clone();
                let http2 = http.clone();
                let p2 = p.clone();
                let tx2 = tx.clone();
                let lim2 = rest_limiter.clone();

                jobs.push(async move {
                    let r = crate::market::candles::candle_rest::rest_backfill_one(&http2, &cfg2, &p2, tf, last_ms, tx2, lim2).await;
                    if let Err(e) = &r {
                        tracing::warn!("backfill {} {} failed: {}", p2.symbol, tf.as_str(), e);
                    }
                    r
                });
            }
        }

        let backfill_started = Instant::now();

        let (ok_cnt, err_cnt) = stream::iter(jobs)
            .buffer_unordered(http_conc)
            .fold((0u64, 0u64), |(ok, err), res| async move {
                match res {
                    Ok(_) => (ok + 1, err),
                    Err(_) => (ok, err + 1),
                }
            })
            .await;

        tracing::info!(
            "REST backfill done: ok={}, err={}, elapsed_ms={}",
            ok_cnt,
            err_cnt,
            backfill_started.elapsed().as_millis()
        );
    }

    // WS combined streams, chunked - STARTED AFTER HISTORY LOADING
    // This ensures that WebSocket real-time data doesn't compete with historical data loading
    let max_streams_per_ws: usize = std::env::var("INGEST_WS_MAX_STREAMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);

    // Create a consolidated map of last seen timestamps per symbol_id across all timeframes
    // This will be used to filter out WebSocket data that's already in the database
    let mut last_seen_per_symbol: HashMap<i64, i64> = HashMap::new();
    for tf in &tfs {
        let tf_last_seen = fetch_last_time_per_symbol(&meta, *tf).await.unwrap_or_default();
        for (symbol_id, last_time) in tf_last_seen {
            let existing = last_seen_per_symbol.entry(symbol_id).or_insert(0);
            if last_time > *existing {
                *existing = last_time; // Keep the latest timestamp across all timeframes
            }
        }
    }
    let last_seen_per_symbol = Arc::new(last_seen_per_symbol);

    let mut streams: Vec<String> = Vec::with_capacity(pairs.len() * ws_tfs.len());
    for p in &pairs {
        let sym = p.symbol.to_lowercase();
        for tf in &ws_tfs {
            streams.push(format!("{}@kline_{}", sym, tf.as_str()));
        }
    }

    let ws_base = cfg.binance.ws_base_url.clone();
    let chunks: Vec<Vec<String>> = streams.chunks(max_streams_per_ws).map(|c| c.to_vec()).collect();
    tracing::info!("WS streams total={}, connections={}", streams.len(), chunks.len());

    for (i, chunk) in chunks.into_iter().enumerate() {
        let url = build_combined_ws_url(&ws_base, &chunk);
        let cfg2 = cfg.clone();
        let writers2 = writers.clone();
        let symbol_to_id2 = symbol_to_id.clone();
        let last_seen_per_symbol2 = last_seen_per_symbol.clone();
        let shutdown_rx2 = shutdown_rx.clone();

        tokio::spawn(async move {
            tracing::info!("WS worker #{} streams={}", i, chunk.len());
            if let Err(e) = crate::market::candles::candle_ws::ws_worker(cfg2, url, symbol_to_id2, last_seen_per_symbol2, writers2, shutdown_rx2).await {
                tracing::error!("WS worker #{} failed: {}", i, e);
            }
        });
    }

    tracing::info!("Candles ingest running (WS realtime + BINARY COPY writers). Press Ctrl+C to stop.");
    tokio::signal::ctrl_c().await.context("ctrl_c failed")?;
    tracing::info!("Shutdown sig...");

    let _ = shutdown_tx.send(true);
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    Ok(())
}