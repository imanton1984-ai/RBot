use anyhow::Result;
use telemetry::{ConnectionTelemetry, init_tracing};
use binance_rest::BinanceRestClient;
use binance_ws::BinanceWsClient;
use db_lib::DatabaseManager;
use kafka_client::KafkaManager;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    
    println!("Starting connection telemetry service...");
    
    // Initialize connection managers (with mock configurations for testing)
    let binance_rest = BinanceRestClient::new("https://api.binance.com")?;
    let binance_ws = BinanceWsClient::new("wss://stream.binance.com:9443")?;
    let db_manager = DatabaseManager::new("postgresql://trader:secret@localhost:5432/trader_db").await?;
    let kafka_manager = KafkaManager::new("localhost:9092", "test-topic")?;
    
    let mut telemetry = ConnectionTelemetry::new();
    
    // Run continuous monitoring
    telemetry.start_monitoring().await;
    
    // Perform initial health checks
    run_health_checks(&mut telemetry, &binance_rest, &binance_ws, &db_manager, &kafka_manager).await;
    
    // Main monitoring loop
    loop {
        sleep(Duration::from_secs(30)).await;
        
        run_health_checks(&mut telemetry, &binance_rest, &binance_ws, &db_manager, &kafka_manager).await;
        
        // Log current status
        telemetry.log_status();
        
        // Check if all connections are healthy
        if !telemetry.is_everything_connected() {
            println!("WARNING: Some connections are not healthy!");
        } else {
            println!("All connections are healthy.");
        }
    }
}

async fn run_health_checks(
    telemetry: &mut ConnectionTelemetry,
    binance_rest: &BinanceRestClient,
    binance_ws: &BinanceWsClient,
    db_manager: &DatabaseManager,
    kafka_manager: &KafkaManager,
) {
    // Update Binance REST status
    match binance_rest.health_check().await {
        Ok(connected) => {
            if let Err(e) = telemetry.update_binance_rest_status(|| Ok(connected)).await {
                eprintln!("Error updating Binance REST status: {}", e);
            }
        }
        Err(e) => {
            if let Err(update_err) = telemetry.update_binance_rest_status(|| Err(e)).await {
                eprintln!("Error updating Binance REST status: {}", update_err);
            }
        }
    }
    
    // Update Binance WebSocket status
    match binance_ws.health_check().await {
        Ok(connected) => {
            if let Err(e) = telemetry.update_binance_ws_status(|| Ok(connected)).await {
                eprintln!("Error updating Binance WebSocket status: {}", e);
            }
        }
        Err(e) => {
            if let Err(update_err) = telemetry.update_binance_ws_status(|| Err(e)).await {
                eprintln!("Error updating Binance WebSocket status: {}", update_err);
            }
        }
    }
    
    // Update Database status
    match db_manager.health_check().await {
        Ok(connected) => {
            if let Err(e) = telemetry.update_database_status(|| Ok(connected)).await {
                eprintln!("Error updating Database status: {}", e);
            }
        }
        Err(e) => {
            if let Err(update_err) = telemetry.update_database_status(|| Err(e)).await {
                eprintln!("Error updating Database status: {}", update_err);
            }
        }
    }
    
    // Update Kafka status
    match kafka_manager.health_check().await {
        Ok(connected) => {
            if let Err(e) = telemetry.update_kafka_status(|| Ok(connected)).await {
                eprintln!("Error updating Kafka status: {}", e);
            }
        }
        Err(e) => {
            if let Err(update_err) = telemetry.update_kafka_status(|| Err(e)).await {
                eprintln!("Error updating Kafka status: {}", update_err);
            }
        }
    }
}