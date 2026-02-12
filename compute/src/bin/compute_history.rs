use std::sync::Arc;
use anyhow::{Context, Result};
use common::{Symbol, Timeframe};
use compute_lib::*;
use database_lib;
use dotenvy::dotenv;
use sqlx::{PgPool, Row};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());
    
    database_lib::init_db::initialize_database(&db_url)
        .await
        .map_err(|e| anyhow::anyhow!("Database init failed: {}", e))?;

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
                 WHERE p.symbol = ",
                table_name
            );

            let row = sqlx::query(&query)
                .bind(symbol.as_str())
                .fetch_optional(&db_pool)
                .await?;

            if let Some(row) = row {
                let first_time_opt: Option<chrono::DateTime<chrono::Utc>> = row.get("first_time");
                let last_time_opt: Option<chrono::DateTime<chrono::Utc>> = row.get("last_time");

                if let (Some(start_time), Some(end_time)) = (first_time_opt, last_time_opt) {
                    println!("Processing {} for {} from {} to {}", symbol.as_str(), timeframe.as_str(), start_time, end_time);

                    let duration_in_minutes = (end_time - start_time).num_minutes();
                    let timeframe_in_minutes = timeframe.to_minutes() as i64;
                    let length = if timeframe_in_minutes > 0 {
                        (duration_in_minutes / timeframe_in_minutes) as usize
                    } else {
                        0
                    };

                    let window_spec = WindowSpec {
                        length,
                        warmup: 0,
                    };

                    if let Err(e) = job_scheduler.submit_batch(*timeframe, vec![symbol.clone()], window_spec).await {
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
) -> Result<Vec<Symbol>> {
    // First check if the table exists
    let table_exists: bool = sqlx::query_scalar(
        r#"SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'market' AND table_name = 'pairs'
        )"#,
    )
    .fetch_one(db_pool)
    .await
    .context("Failed to check for 'market.pairs' table existence")?;

    if !table_exists {
        return Ok(Vec::new()); // Return empty vector if table doesn't exist
    }

    // If table exists, try to fetch symbols
    let rows = sqlx::query("SELECT symbol FROM market.pairs WHERE is_active = true")
        .fetch_all(db_pool)
        .await
        .context("Failed to fetch active symbols from 'market.pairs'")?;

    let symbols: Vec<Symbol> = rows
        .into_iter()
        .map(|row| {
            let symbol: String = row.get("symbol");
            Symbol::from(symbol)
        })
        .collect();
    Ok(symbols)
}