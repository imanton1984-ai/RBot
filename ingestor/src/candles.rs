use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};
use common::{load_config, AppConfig, Candle, RawCandleData, TimeFrame};
use connections_lib::{binance_websocket::BinanceWsConnection, redpanda::RedpandaConnection};
use serde_json::Value;
use sqlx::{PgPool, Row};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{info, warn, error};

#[derive(Debug, Clone)]
struct PairInfo {
    symbol_id: i64,
    symbol: String,
}

pub async fn load_historical_candles() -> Result<()> {
    let cfg: AppConfig = load_config().context("load_config() failed")?;
    let db_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| cfg.database.url());
    let pool = PgPool::connect(&db_url).await.context("connect DB failed")?;

    let timeout = Duration::from_millis(cfg.binance.http_timeout_ms);
    let client = reqwest::Client::builder().timeout(timeout).build()?;

    let rest_base = cfg.binance.rest_base_url.clone();

    // Get all active pairs from the database
    let pairs = get_active_pairs(&pool).await?;
    info!("Found {} active pairs to load candles for", pairs.len());

    // Define the timeframes we want to load
    let timeframes = TimeFrame::all_timeframes();
    info!("Loading candles for timeframes: {:?}", timeframes.iter().map(|tf| tf.as_str()).collect::<Vec<_>>());

    // Use a semaphore to control concurrency - use a very conservative value for historical data loading
    let max_concurrent = std::env::var("MAX_API_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2); // Very conservative for historical loading to avoid rate limits
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent.max(1))); // Minimum of 1 to avoid deadlock

    // Process all pairs and timeframes with controlled concurrency
    let mut tasks = Vec::new();

    // Convert to owned values to avoid borrowing issues
    let pairs_owned: Vec<_> = pairs.into_iter().collect();
    let timeframes_owned: Vec<_> = timeframes.into_iter().collect();

    // Create all tasks with cloned values
    for pair in &pairs_owned {
        for timeframe in &timeframes_owned {
            let client = client.clone();
            let rest_base = rest_base.clone();
            let semaphore = semaphore.clone();
            let db_url = db_url.clone();
            let pair = pair.clone(); // Clone the pair for each task
            let timeframe = *timeframe; // Copy the timeframe

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();

                // Create a new connection for this task
                let pool = PgPool::connect(&db_url).await.context("connect DB failed in task")?;

                info!("Loading {} timeframe for {}", timeframe.as_str(), pair.symbol);

                // Get the last closed candle time for this pair/timeframe combination
                let last_candle_time = get_last_candle_time(&pool, pair.symbol_id, timeframe).await?;

                // Load 700 candles for this pair/timeframe
                let candles_data = fetch_candles_via_api(
                    &client,
                    &rest_base,
                    &pair.symbol,
                    timeframe.as_str(),
                    700,
                    last_candle_time
                ).await?;

                // Convert raw data to Candle structs
                let candles = parse_candle_data(candles_data, pair.symbol_id);

                // Insert candles into the appropriate table
                insert_candles(&pool, &candles, timeframe).await?;

                info!("Inserted {} {} candles for {}", candles.len(), timeframe.as_str(), pair.symbol);

                Ok::<(), anyhow::Error>(())
            });

            tasks.push(task);
        }
    }

    // Wait for all tasks to complete
    for task in tasks {
        match task.await {
            Ok(Ok(())) => {}, // Success
            Ok(Err(e)) => error!("Task error: {}", e),
            Err(e) => error!("Task join error: {}", e),
        }
    }

    info!("Historical candle loading completed");
    Ok(())
}

async fn get_active_pairs(pool: &PgPool) -> Result<Vec<PairInfo>> {
    let rows = sqlx::query(
        "SELECT symbol_id, symbol FROM market.pairs WHERE is_active = TRUE"
    )
    .fetch_all(pool)
    .await?;

    let pairs = rows
        .into_iter()
        .map(|row| PairInfo {
            symbol_id: row.get("symbol_id"),
            symbol: row.get("symbol"),
        })
        .collect();

    Ok(pairs)
}

async fn get_last_candle_time(pool: &PgPool, symbol_id: i64, timeframe: TimeFrame) -> Result<Option<i64>> {
    let table_name = format!("market.candles_{}", timeframe.as_str());
    let query = format!(
        "SELECT MAX(time_ms) as last_time FROM {} WHERE symbol_id = $1",
        table_name
    );

    let row = sqlx::query(&query)
        .bind(symbol_id)
        .fetch_optional(pool)
        .await?;

    if let Some(row) = row {
        let last_time: Option<i64> = row.get("last_time");
        Ok(last_time)
    } else {
        Ok(None)
    }
}

async fn fetch_candles_via_api(
    client: &reqwest::Client,
    base_url: &str,
    symbol: &str,
    interval: &str,
    limit: u32,
    start_time: Option<i64>,
) -> Result<Vec<RawCandleData>> {
    let mut url = format!(
        "{}/fapi/v1/klines?symbol={}&interval={}&limit={}",
        base_url.trim_end_matches('/'),
        symbol.to_uppercase(),
        interval,
        limit
    );

    // Add start time if provided
    if let Some(start) = start_time {
        url.push_str(&format!("&startTime={}", start));
    }

    let response = client.get(&url).send().await?.error_for_status()?;
    let text = response.text().await?;

    // Parse the JSON response
    let json_value: Value = serde_json::from_str(&text)?;

    if let Some(array) = json_value.as_array() {
        let mut candles = Vec::new();

        for item in array {
            if let Some(arr) = item.as_array() {
                if arr.len() >= 11 {
                    let candle = RawCandleData {
                        open_time: arr[0].as_i64().unwrap_or(0),
                        open: arr[1].as_str().unwrap_or("0").to_string(),
                        high: arr[2].as_str().unwrap_or("0").to_string(),
                        low: arr[3].as_str().unwrap_or("0").to_string(),
                        close: arr[4].as_str().unwrap_or("0").to_string(),
                        volume: arr[5].as_str().unwrap_or("0").to_string(),
                        close_time: arr[6].as_i64().unwrap_or(0),
                        quote_asset_volume: arr[7].as_str().unwrap_or("0").to_string(),
                        number_of_trades: arr[8].as_i64().unwrap_or(0),
                        taker_buy_base_asset_volume: arr[9].as_str().unwrap_or("0").to_string(),
                        taker_buy_quote_asset_volume: arr[10].as_str().unwrap_or("0").to_string(),
                    };
                    candles.push(candle);
                }
            }
        }

        Ok(candles)
    } else {
        Ok(Vec::new())
    }
}

fn parse_candle_data(raw_candles: Vec<RawCandleData>, symbol_id: i64) -> Vec<Candle> {
    raw_candles
        .into_iter()
        .map(|raw| {
            let mut candle: Candle = raw.into();
            candle.symbol_id = symbol_id;
            candle
        })
        .collect()
}

async fn insert_candles(pool: &PgPool, candles: &[Candle], timeframe: TimeFrame) -> Result<()> {
    if candles.is_empty() {
        return Ok(());
    }

    let table_name = format!("market.candles_{}", timeframe.as_str());

    // Use bulk insert with a single query for maximum efficiency
    let mut tx = pool.begin().await?;

    // Process in chunks to avoid query size limits
    for chunk in candles.chunks(1000) {
        let mut query_builder: sqlx::QueryBuilder<sqlx::Postgres> = sqlx::QueryBuilder::new(
            format!("INSERT INTO {} (time_ms, time, symbol_id, open, high, low, close, volume) ", table_name)
        );

        query_builder.push_values(chunk, |mut b, candle| {
            let time = Utc.timestamp_opt((candle.time_ms / 1000) as i64, 0)
                .single()
                .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().unwrap());

            b.push_bind(candle.time_ms)
             .push_bind(time)
             .push_bind(candle.symbol_id)
             .push_bind(candle.open)
             .push_bind(candle.high)
             .push_bind(candle.low)
             .push_bind(candle.close)
             .push_bind(candle.volume);
        });

        // Use ON CONFLICT to handle duplicates efficiently
        query_builder.push(" ON CONFLICT (symbol_id, time) DO NOTHING");

        query_builder.build().execute(&mut *tx).await?;
    }

    tx.commit().await?;

    Ok(())
}

pub async fn start_realtime_candle_ingestion() -> Result<()> {
    info!("Starting realtime candle ingestion via WebSocket...");

    // Get configuration
    let cfg: AppConfig = load_config().context("load_config() failed")?;

    // Get active pairs
    let db_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| cfg.database.url());
    let pool = PgPool::connect(&db_url).await.context("connect DB failed")?;
    let pairs = get_active_pairs(&pool).await?;

    // Create Redpanda connection
    let mut redpanda_conn: RedpandaConnection = RedpandaConnection::new_from_env().await?;
    redpanda_conn.connect_producer().await?;

    // Create WebSocket connection for Kline/Candlestick Streams
    let mut ws_streams = Vec::new();
    for pair in &pairs {
        for timeframe in &TimeFrame::all_timeframes() {
            let stream_name = format!("{}@kline_{}", pair.symbol.to_lowercase(), timeframe.as_str());
            ws_streams.push(stream_name);
        }
    }

    // Limit the number of streams due to Binance limits
    // Binance allows up to 200 streams per connection
    if ws_streams.len() > 200 {
        warn!("Too many streams ({}), limiting to 200", ws_streams.len());
        ws_streams.truncate(200);
    }

    info!("Connecting to WebSocket with {} streams", ws_streams.len());

    // Set up the WebSocket connection
    let ws_conn: BinanceWsConnection = BinanceWsConnection::new_from_env().await?;

    // Create channel for receiving WebSocket messages
    let (tx, rx) = mpsc::channel::<bytes::Bytes>(1000);
    let (status_tx, _status_rx) = tokio::sync::watch::channel(false);

    // Spawn WebSocket connection task
    let handle = tokio::spawn({
        let ws_conn = ws_conn;
        let tx = tx.clone();
        let status_tx = status_tx.clone();
        async move {
            if let Err(e) = ws_conn.run_forever(tx, status_tx).await {
                error!("WebSocket connection error: {}", e);
            }
        }
    });

    // Process received WebSocket messages
    let mut rx = rx;
    while let Some(msg_bytes) = rx.recv().await {
        if let Ok(msg_str) = std::str::from_utf8(&msg_bytes) {
            if let Ok(json_value) = serde_json::from_str::<Value>(msg_str) {
                // Check if this is a kline event
                if let Some(event_type) = json_value.get("stream") {
                    if event_type.as_str().unwrap_or("").contains("@kline_") {
                        if let Some(data) = json_value.get("data") {
                            if let Some(kline_data) = data.get("k") {
                                // Parse kline data
                                if let Some(parsed_candle) = parse_kline_data(kline_data, &pairs) {
                                    // Send candle to Redpanda
                                    if let Ok(candle_json) = serde_json::to_string(&parsed_candle) {
                                        if let Err(e) = redpanda_conn.send_message(&parsed_candle.symbol_id.to_string(), candle_json.as_bytes()).await {
                                            error!("Failed to send candle to Redpanda: {}", e);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Wait for the WebSocket task to complete
    let _ = handle.await;

    Ok(())
}

fn parse_kline_data(kline_data: &Value, pairs: &[PairInfo]) -> Option<Candle> {
    // Extract kline data from JSON
    let symbol = kline_data.get("s")?.as_str()?.to_uppercase();
    let close_time = kline_data.get("T")?.as_i64()?;
    let open = kline_data.get("o")?.as_str()?.parse::<f64>().unwrap_or(0.0);
    let high = kline_data.get("h")?.as_str()?.parse::<f64>().unwrap_or(0.0);
    let low = kline_data.get("l")?.as_str()?.parse::<f64>().unwrap_or(0.0);
    let close = kline_data.get("c")?.as_str()?.parse::<f64>().unwrap_or(0.0);
    let volume = kline_data.get("v")?.as_str()?.parse::<f64>().unwrap_or(0.0);

    // Find the corresponding symbol_id
    let symbol_id = pairs.iter()
        .find(|p| p.symbol == symbol)
        .map(|p| p.symbol_id)?;

    Some(Candle {
        time_ms: close_time,
        symbol_id,
        open,
        high,
        low,
        close,
        volume,
    })
}