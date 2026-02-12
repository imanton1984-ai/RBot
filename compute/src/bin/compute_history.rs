use std::sync::Arc;
use dotenvy::dotenv;
use compute_lib::*;
use database_lib;
use common::{Symbol, Timeframe};
use sqlx::PgPool;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());
    database_lib::init_db::initialize_database(&db_url).await?;

    // фиксируем CUDA backend (если бинарь собран с --features cuda)
    let backend_type = if cfg!(feature = "cuda") {
        ComputeBackendType::Cuda
    } else {
        eprintln!("CUDA feature not enabled, falling back to CPU for history");
        ComputeBackendType::Cpu
    };
    
    let compute_backend_manager = ComputeBackendManager::new(backend_type);
    let compute_backend = compute_backend_manager.get_backend();

    // load config
    let config = ComputeConfig {
        batch_size: 100,
        max_concurrent_jobs: 4,
        use_cuda: cfg!(feature = "cuda"),
        cuda_device_id: Some(0),
    };

    let db_pool = sqlx::PgPool::connect(&db_url).await?;
    let candle_fetcher = Arc::new(CandleWindowFetcher::new(db_pool.clone()));

    let (job_scheduler, _result_receiver) = JobScheduler::new(
        compute_backend,
        config.clone(),
        candle_fetcher.clone(),
    );
    let job_scheduler = Arc::new(job_scheduler);

    // Set environment for history mode
    std::env::set_var("DB_PERSIST_MODE", "history");
    std::env::set_var("DB_PERSIST_CHUNK_SIZE_HISTORY", "50000"); // Large chunks for history
    std::env::set_var("DB_PERSIST_FLUSH_MS", "150"); // Slower flushing for history
    std::env::set_var("DB_PERSIST_HISTORY_SKIP_JSON", "1"); // Skip heavy JSON for history
    std::env::set_var("DB_PERSIST_HISTORY_UPSERT", "0"); // Append mode for history

    // 1) получаем список активных пар из БД (market.pairs)
    let symbols = fetch_active_symbols_from_db(&db_pool).await?;

    // 2) для каждого TF запускаем submit_batch(…, window_spec)
    let timeframes = [
        Timeframe::M1,
        Timeframe::M5,
        Timeframe::M15,
        Timeframe::H1,
        Timeframe::H4,
        Timeframe::D1,
    ];

    for timeframe in &timeframes {
        println!("Processing historical data for timeframe: {:?}", timeframe);

        for symbol in &symbols {
            // Determine the historical range to process
            let table_name = format!("market.candles_{}", timeframe.as_str());
            let query = format!(
                "SELECT MIN(time) as first_time, MAX(time) as last_time
                 FROM {} c
                 JOIN market.pairs p ON c.symbol_id = p.symbol_id
                 WHERE p.symbol = $1",
                table_name
            );

            let row = sqlx::query(&query)
                .bind(symbol.as_str())
                .fetch_optional(&db_pool)
                .await?;

            if let Some(row) = row {
                let first_time: Option<chrono::DateTime<chrono::Utc>> = row.get("first_time");
                let last_time: Option<chrono::DateTime<chrono::Utc>> = row.get("last_time");

                if let (Some(start_time), Some(end_time)) = (first_time, last_time) {
                    println!("Processing {} for {} from {} to {}", symbol.as_str(), timeframe.as_str(), start_time, end_time);

                    // Submit batch job for this symbol/timeframe combination
                    let job = ComputeJob {
                        symbol: symbol.clone(),
                        timeframe: *timeframe,
                        window_start: start_time.timestamp_millis(),
                        window_end: end_time.timestamp_millis(),
                        indicators: vec![
                            "adx".to_string(), "atr".to_string(), "bb".to_string(),
                            "cci".to_string(), "ema".to_string(), "macd".to_string(),
                            "obv".to_string(), "rsi".to_string(), "sma".to_string(),
                            "stoch".to_string(), "vwap".to_string(), "williams".to_string(),
                            "alligator".to_string(), "sr_levels".to_string(),
                        ],
                        candle_window: None,
                        is_realtime: false, // Guarantee is_realtime = false for history
                    };

                    if let Err(e) = job_scheduler.submit_batch(vec![job]).await {
                        eprintln!("Error submitting batch job for {} {}: {}", symbol.as_str(), timeframe.as_str(), e);
                    } else {
                        println!("Submitted batch job for {} {}", symbol.as_str(), timeframe.as_str());
                    }
                }
            }
        }
    }

    // Wait a bit to allow jobs to complete
    tokio::time::sleep(Duration::from_secs(10)).await;

    Ok(())
}

// Function to fetch active symbols from the database
async fn fetch_active_symbols_from_db(
    db_pool: &PgPool,
) -> Result<Vec<Symbol>, Box<dyn std::error::Error + Send + Sync>> {
    // First check if the table exists
    let table_exists_result = sqlx::query_scalar::<_, bool>(
        r#"SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'market' AND table_name = 'pairs'
        )"#
    )
    .fetch_one(db_pool)
    .await;

    match table_exists_result {
        Ok(table_exists) => {
            if !table_exists {
                return Ok(Vec::new()); // Return empty vector if table doesn't exist
            }
        },
        Err(_) => {
            return Ok(Vec::new()); // Return empty vector if table check fails
        }
    }

    // If table exists, try to fetch symbols
    let rows_result = sqlx::query(
        r#"SELECT symbol FROM market.pairs WHERE is_active = true"#
    )
    .fetch_all(db_pool)
    .await;

    match rows_result {
        Ok(rows) => {
            let symbols: Vec<Symbol> = rows
                .into_iter()
                .map(|row| {
                    let symbol: String = row.get("symbol");
                    Symbol::from(symbol)
                })
                .collect();
            Ok(symbols)
        },
        Err(e) => {
            eprintln!("Error fetching symbols: {}", e);
            Ok(Vec::new()) // Return empty vector if fetch fails
        }
    }
}