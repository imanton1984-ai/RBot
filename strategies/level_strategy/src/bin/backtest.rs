// strategies/level_strategy/src/bin/backtest.rs
//
// Level Strategy Backtester
//
// Evaluates the level strategy on historical data.
//
// USAGE:
//   cargo run --release -p level_strategy --bin level_strategy_backtest

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::PgPool;
use tracing::info;

use level_strategy::config::LevelStrategyConfig;
use level_strategy::pipeline::LevelStrategyPipeline;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let config = LevelStrategyConfig::from_env();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         LEVEL STRATEGY BACKTESTER                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Config:");
    println!("    Horizon bars:   {}", config.horizon_bars);
    println!("    Min score:      {}", config.min_store_score);
    println!("    Min final:      {}", config.min_final_score);
    println!("    Prefer ML:      {}", config.prefer_ml);
    println!();

    // Initialize pipeline
    let pipeline = LevelStrategyPipeline::new(config.clone());

    if !pipeline.has_models() {
        eprintln!("No level strategy models found!");
        eprintln!("Train models first via trainer/teacher.sh");
        return Ok(());
    }

    let pool = PgPool::connect(&db_url).await?;

    info!("Running level strategy backtest...");
    match pipeline.run_history(&pool).await {
        Ok(count) => {
            println!("✅ Backtest complete: {} signals generated", count);
        }
        Err(e) => {
            eprintln!("Backtest error: {}", e);
        }
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         BACKTEST COMPLETE                                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
