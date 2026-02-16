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
use crate::market::candles::candle_ws::run_kline_ws;
use crate::market::pairs::refresh_universe_pairs;
use anyhow::{Context, Result};
use chrono::Utc;
use common::{load_config, AppConfig};
use common::timeframe::TimeFrame;
use futures::{stream, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tokio::time::Instant;
use tokio::sync::Semaphore;
use tokio::time::interval;
use sqlx::PgPool;
use tracing::info;

// Spawn a task to periodically refresh pairs and potentially restart WebSocket connections
async fn spawn_pair_refresh_task(
    cfg: Arc<AppConfig>,
    symbol_to_id: Arc<HashMap<String, i64>>,
    shutdown_rx: watch::Receiver<bool>,
) {
    let mut shutdown_rx = shutdown_rx;
    let refresh_interval = tokio::time::Duration::from_secs(cfg.universe.refresh_interval_sec as u64);

    tokio::spawn(async move {
        let mut interval = interval(refresh_interval);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    tracing::info!("Periodically refreshing universe pairs...");
                    match refresh_universe_pairs().await {
                        Ok(result) => {
                            tracing::info!("Pairs refreshed: selected_cnt={}, active_cnt={}",
                                         result.selected_cnt, result.active_cnt);

                            // Check if there are new pairs that need to be added to WebSocket connections
                            match crate::market::candles::candle_rest::fetch_active_pairs(&crate::market::candles::candle_rest::pg_connect(&std::env::var("DATABASE_URL").unwrap_or_else(|_| cfg.database.url())).await.unwrap()).await {
                                Ok(updated_pairs) => {
                                    // Check if there are new pairs that weren't in the original set
                                    let current_symbols: std::collections::HashSet<_> = symbol_to_id.keys().cloned().collect();
                                    let updated_symbols: std::collections::HashSet<_> = updated_pairs.iter().map(|p| p.symbol.clone()).collect();

                                    let new_symbols: Vec<_> = updated_symbols.difference(&current_symbols).collect();
                                    if !new_symbols.is_empty() {
                                        tracing::info!("Detected {} new pairs to add to WebSocket connections", new_symbols.len());

                                        // In a production system, we would need to restart WebSocket connections with new pairs
                                        // For now, log the new pairs that need to be added
                                        for symbol in new_symbols {
                                            tracing::info!("New pair detected: {}", symbol);
                                        }
                                    }
                                },
                                Err(e) => {
                                    tracing::error!("Failed to fetch updated pairs after refresh: {}", e);
                                }
                            }
                        },
                        Err(e) => {
                            tracing::error!("Failed to refresh pairs: {}", e);
                        }
                    }
                },
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        tracing::info!("Pair refresh task shutting down");
                        break;
                    }
                }
            }
        }
    });
}

// TODO: твой backfill-функционал
async fn run_rest_backfill(_pool: &PgPool) -> Result<()> {
    // 1) получить список активных пар из market.pairs
    // 2) REST /fapi/v1/klines?symbol=...&interval=...&limit=...
    // 3) bulk insert в исторические таблицы (как сейчас)
    Ok(())
}

pub async fn run_candles_pipeline(
    pool: PgPool,
    ws_base_url: String,
    streams: Vec<String>,
    live_flush_ms: u64,
) -> Result<()> {
    // 1) История
    info!("Backfill start...");
    run_rest_backfill(&pool).await.context("rest backfill failed")?;
    info!("Backfill done.");

    // 2) Канал live свечей (WS → Writer)
    let (tx_live, rx_live) = mpsc::channel::<LiveCandle>(100_000);

    // 3) Writer (UPSERT batched)
    let writer = LiveWriter::new(pool.clone(), live_flush_ms, 50_000);
    tokio::spawn(async move {
        if let Err(e) = writer.run(rx_live).await {
            tracing::error!("LiveWriter crashed: {:?}", e);
        }
    });

    // 4) WS (kline intra updates)
    run_kline_ws(ws_base_url, streams, tx_live).await?;

    Ok(())
}

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

    // Spawn the periodic pair refresh task
    spawn_pair_refresh_task(cfg.clone(), symbol_to_id.clone(), shutdown_rx.clone()).await;

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
        // Use the new parameter if available, otherwise fall back to the old one for backward compatibility
        let weight_per_sec = cfg.binance.soft_weight_per_sec.unwrap_or(40); // Default to 40 weight/sec for futures
        let rest_limiter = crate::market::candles::candle_common::WeightLimiter::new(
            weight_per_sec,
            cfg.binance.rate_limit_soft_burst,
        );

        // Log effective request rate based on the limit and weight
        let backfill_override: Option<usize> = std::env::var("BACKFILL_CANDLES").ok().and_then(|v| v.parse().ok());
        let limit = backfill_override.unwrap_or(cfg.runtime.backfill_candles).min(1500).max(1);
        let weight = klines_weight(limit);
        let effective_req_per_sec = weight_per_sec as f64 / weight as f64;
        tracing::info!(
            "Rate limiting configured: {} weight/sec, limit={}, weight={}, effective ~{:.1} req/sec",
            weight_per_sec, limit, weight, effective_req_per_sec
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

                    // Use a smaller limit (500 instead of 700) to reduce weight from 5 to 2
                    // This allows more requests per second within the same weight budget
                    let limit = backfill_override.unwrap_or(cfg_clone.runtime.backfill_candles).min(1500).max(1);
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
                    let now = Utc::now().timestamp_millis();

                    for k in &klines {
                        // If the close time of the candle is in the future (or is the current second),
                        // it means the candle is not closed. We don't save it to the history table.
                        if k.6 >= now {
                            continue;
                        }

                        results.push(CandleRow {
                            time_ms: k.6, // close_time
                            symbol_id: pair_clone.symbol_id,
                            symbol: pair_clone.symbol.clone(),
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
        // Increase buffer size to allow more concurrent operations
        let fetch_results = stream::iter(fetch_jobs)
            .buffer_unordered(cfg.runtime.http_concurrency.unwrap_or(50).max(1))
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
        // Use the new parameter if available, otherwise fall back to the old one for backward compatibility
        let weight_per_sec = cfg.binance.soft_weight_per_sec.unwrap_or(40); // Default to 40 weight/sec for futures
        let rest_limiter = Some(crate::market::candles::candle_common::WeightLimiter::new(
            weight_per_sec,
            cfg.binance.rate_limit_soft_burst,
        ));

        // Log effective request rate based on the limit and weight
        let backfill_override: Option<usize> = std::env::var("BACKFILL_CANDLES").ok().and_then(|v| v.parse().ok());
        let limit = backfill_override.unwrap_or(cfg.runtime.backfill_candles).min(1500).max(1);
        let weight = klines_weight(limit);
        let effective_req_per_sec = weight_per_sec as f64 / weight as f64;
        tracing::info!(
            "Backfill rate limiting configured: {} weight/sec, limit={}, weight={}, effective ~{:.1} req/sec",
            weight_per_sec, limit, weight, effective_req_per_sec
        );

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

    // Add a delay before starting WebSocket connections to allow the connection to "cool down"
    tracing::info!("Waiting for 5 seconds before starting WebSocket connections...");
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

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

    // NEW: Initialize the live candle writer if enabled in config
    if cfg.runtime.persist_live_candle {
        tracing::info!("Starting live candle writer with flush interval: {}ms", cfg.runtime.persist_live_candle_every_ms);

        // Create a PgPool for the live writer
        let pool = sqlx::PgPool::connect(&db_url).await.context("Failed to connect to database for live writer")?;

        // Create streams for the live candle writer (all pairs and timeframes)
        let mut live_streams: Vec<String> = Vec::with_capacity(pairs.len() * ws_tfs.len());
        for p in &pairs {
            let sym = p.symbol.to_lowercase();
            for tf in &ws_tfs {
                live_streams.push(format!("{}@kline_{}", sym, tf.as_str()));
            }
        }

        // Clone necessary values before moving them into the async block
        let ws_base_clone = ws_base.clone();
        let live_flush_ms = cfg.runtime.persist_live_candle_every_ms;

        // Start the live candle pipeline
        tokio::spawn(async move {
            if let Err(e) = run_candles_pipeline(
                pool,
                ws_base_clone,
                live_streams,
                live_flush_ms,
            ).await {
                tracing::error!("Live candle pipeline failed: {:?}", e);
            }
        });
    }

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