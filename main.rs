mod binance_api;
mod binance_websocket;
mod redpanda;
mod database;
mod connection_check;

use anyhow::Result;
use dotenv::dotenv;
use tracing_subscriber::{self, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    
    // Load environment variables
    dotenv().ok();
    
    println!("Starting connection tests...");
    
    // Run comprehensive connection check
    match connection_check::check_all_connections().await {
        Ok(()) => {
            println!("All connection checks completed!");
        }
        Err(e) => {
            eprintln!("Error during connection checks: {}", e);
        }
    }
    
    Ok(())
}