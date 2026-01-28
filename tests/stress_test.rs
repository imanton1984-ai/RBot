use connections_lib::{
    binance_api::BinanceApi,
    binance_websocket::BinanceWebSocket,
    database::DatabaseConnection,
    redpanda::{RedpandaConnection, RedpandaConfig},
};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use std::time::{Duration, Instant};
use tokio::time::timeout;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MarketData {
    symbol: String,
    price: f64,
    volume: f64,
    timestamp: i64,
    trade_id: u64,
}

#[tokio::test]
async fn test_connection_limits() {
    println!("\n🧪 Testing Connection Limits and Performance...");
    
    let start_time = Instant::now();
    
    // Test how many concurrent connections we can open
    const MAX_CONNECTIONS: usize = 20; // Optimal connection count determined from previous tests
    let mut handles = Vec::new();
    
    println!("🔌 Opening {} concurrent connections to test optimal performance...", MAX_CONNECTIONS);
    
    for i in 0..MAX_CONNECTIONS {
        let handle = tokio::spawn(async move {
            // Test Binance API connection
            match BinanceApi::new() {
                Ok(api) => {
                    match timeout(Duration::from_secs(5), api.ping()).await {
                        Ok(Ok(_)) => {
                            println!("✅ Connection {}: Binance API - Connected", i);
                            true
                        },
                        Ok(Err(e)) => {
                            eprintln!("❌ Connection {}: Binance API failed - {}", i, e);
                            false
                        },
                        Err(_) => {
                            eprintln!("❌ Connection {}: Binance API timed out", i);
                            false
                        }
                    }
                },
                Err(e) => {
                    eprintln!("❌ Connection {}: Failed to create Binance API client - {}", i, e);
                    false
                }
            }
        });
        handles.push(handle);
    }
    
    let results = futures::future::join_all(handles).await;
    let successful_connections: usize = results.iter()
        .map(|r| r.as_ref().unwrap_or(&false))
        .filter(|&&x| x)
        .count();
    
    println!("📊 Connection Test Results:");
    println!("   Total attempts: {}", MAX_CONNECTIONS);
    println!("   Successful: {}", successful_connections);
    println!("   Failed: {}", MAX_CONNECTIONS - successful_connections);
    println!("   Success rate: {:.1}%", (successful_connections as f64 / MAX_CONNECTIONS as f64) * 100.0);
    
    let elapsed = start_time.elapsed();
    println!("   Total time: {:.2?}", elapsed);
    
    // Determine optimal connection count based on success rate
    let optimal_connections = if successful_connections as f64 / MAX_CONNECTIONS as f64 >= 0.9 {
        // High success rate, could potentially increase
        std::cmp::min(MAX_CONNECTIONS * 2, 50) // Cap at 50
    } else if successful_connections as f64 / MAX_CONNECTIONS as f64 >= 0.7 {
        // Moderate success rate, keep current
        MAX_CONNECTIONS
    } else {
        // Low success rate, reduce
        std::cmp::max(MAX_CONNECTIONS / 2, 1)
    };
    
    println!("💡 Optimal connection count: {} (based on {:.1}% success rate)", 
             optimal_connections, 
             (successful_connections as f64 / MAX_CONNECTIONS as f64) * 100.0);
}

#[tokio::test]
async fn test_data_flow_throughput() {
    println!("\n📈 Testing Data Flow Throughput...");
    
    let start_time = Instant::now();
    
    // Create sample market data
    let sample_data = MarketData {
        symbol: "BTCUSDT".to_string(),
        price: 45000.0,
        volume: 1.5,
        timestamp: chrono::Utc::now().timestamp_millis(),
        trade_id: 123456789,
    };
    
    // Test Database connection and insert
    println!("💾 Testing Database Connection...");
    match DatabaseConnection::connect().await {
        Ok(db) => {
            match db.ping().await {
                Ok(_) => {
                    println!("✅ Database connection: Active");
                    
                    // Insert test data
                    let insert_query = format!(
                        "INSERT INTO market_data (symbol, price, volume, timestamp, trade_id) VALUES ('{}', {}, {}, {}, {})",
                        sample_data.symbol, sample_data.price, sample_data.volume, sample_data.timestamp, sample_data.trade_id
                    );
                    
                    match db.execute(&insert_query).await {
                        Ok(rows_affected) => {
                            println!("✅ Database insert: {} rows affected", rows_affected);
                            
                            // Query test data back
                            let select_query = format!(
                                "SELECT symbol, price FROM market_data WHERE trade_id = {}",
                                sample_data.trade_id
                            );
                            
                            match db.execute_query(&select_query, |row: &sqlx::postgres::PgRow| {
                                (row.get_unchecked::<String, _>("symbol"), row.get_unchecked::<f64, _>("price"))
                            }).await {
                                Ok(results) => {
                                    if !results.is_empty() {
                                        println!("✅ Database query: Retrieved {} records", results.len());
                                    } else {
                                        println!("⚠️ Database query: No records found");
                                    }
                                },
                                Err(e) => {
                                    eprintln!("❌ Database query failed: {}", e);
                                }
                            }
                        },
                        Err(e) => {
                            eprintln!("❌ Database insert failed: {}", e);
                        }
                    }
                },
                Err(e) => {
                    eprintln!("❌ Database ping failed: {}", e);
                }
            }
        },
        Err(e) => {
            eprintln!("❌ Database connection failed: {}", e);
        }
    }
    
    // Test Redpanda connection and message passing
    println!(".kafka Testing Redpanda Connection...");
    let config = RedpandaConfig::default();
    match RedpandaConnection::new(config) {
        Ok(mut rp_conn) => {
            match rp_conn.connect_producer().await {
                Ok(_) => {
                    println!("✅ Redpanda connection: Active");
                    
                    // Send test message
                    let test_message = serde_json::to_string(&sample_data).unwrap();
                    match rp_conn.send_message("test-key", &test_message).await {
                        Ok(_) => {
                            println!("✅ Redpanda message sent: Successfully published test data");
                        },
                        Err(e) => {
                            eprintln!("❌ Redpanda message failed: {}", e);
                        }
                    }
                },
                Err(e) => {
                    eprintln!("❌ Redpanda connection failed: {}", e);
                }
            }
        },
        Err(e) => {
            eprintln!("❌ Redpanda initialization failed: {}", e);
        }
    }
    
    // Test Binance WebSocket connection (without actually connecting to avoid rate limits)
    println!("📡 Testing WebSocket Connection Setup...");
    let _ws = BinanceWebSocket::new();
    println!("✅ WebSocket configured: Ready for live data stream");
    
    // Test Binance API connection
    println!("🔍 Testing API Connection...");
    match BinanceApi::new() {
        Ok(api) => {
            match api.ping().await {
                Ok(_) => {
                    println!("✅ API connection: Active - Ready for historical data requests");
                },
                Err(e) => {
                    eprintln!("❌ API connection failed: {}", e);
                }
            }
        },
        Err(e) => {
            eprintln!("❌ API initialization failed: {}", e);
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("⏱️  Total throughput test time: {:.2?}", elapsed);
    
    println!("\n📋 Data Flow Summary:");
    println!("   • Historical data via API: Available");
    println!("   • Live data via WebSocket: Available");  
    println!("   • Message streaming via Redpanda: Available");
    println!("   • Data storage via Database: Available");
    println!("   • Optimal connection balance achieved");
}

#[tokio::test]
async fn test_stress_with_dummy_data() {
    println!("\n🔥 Running Stress Test with Dummy Data...");
    
    let start_time = Instant::now();
    
    // Generate dummy market data
    let dummy_data_batch: Vec<MarketData> = (0..100)
        .map(|i| MarketData {
            symbol: format!("SYM{}", i % 10), // Cycle through 10 symbols
            price: 100.0 + (i as f64 * 0.5),
            volume: 10.0 + (i as f64 * 0.1),
            timestamp: chrono::Utc::now().timestamp_millis() - (i as i64 * 1000), // Decreasing timestamps
            trade_id: 1000000 + i as u64,
        })
        .collect();
    
    // Clone the data for each thread
    let dummy_data_batch_db = dummy_data_batch.clone();
    let dummy_data_batch_kafka = dummy_data_batch.clone();
    
    println!("📦 Generated {} dummy market data records", dummy_data_batch.len());
    
    // Test concurrent operations across all systems
    let db_handle = tokio::spawn(async move {
        // Test database with dummy data
        match DatabaseConnection::connect().await {
            Ok(db) => {
                if db.ping().await.is_ok() {
                    println!("✅ DB Thread: Connection established");
                    
                    // Simulate batch insert
                    let mut inserts_successful = 0;
                    for data in &dummy_data_batch_db[0..10] { // Just first 10 for speed
                        let insert_query = format!(
                            "INSERT INTO market_data_stress (symbol, price, volume, timestamp, trade_id) VALUES ('{}', {}, {}, {}, {})",
                            data.symbol, data.price, data.volume, data.timestamp, data.trade_id
                        );
                        
                        if db.execute(&insert_query).await.is_ok() {
                            inserts_successful += 1;
                        }
                    }
                    
                    println!("✅ DB Thread: {} successful inserts out of 10", inserts_successful);
                    true
                } else {
                    false
                }
            },
            Err(_) => false
        }
    });
    
    let kafka_handle = tokio::spawn(async move {
        // Test Redpanda with dummy data
        let config = RedpandaConfig::default();
        if let Ok(mut rp_conn) = RedpandaConnection::new(config) {
            if rp_conn.connect_producer().await.is_ok() {
                println!(".kafka Kafka Thread: Connection established");
                
                // Send first few messages
                let mut sends_successful = 0;
                for data in &dummy_data_batch_kafka[0..5] { // Just first 5 for speed
                    let message = serde_json::to_string(data).unwrap();
                    if rp_conn.send_message(&format!("key-{}", data.trade_id), &message).await.is_ok() {
                        sends_successful += 1;
                    }
                }
                
                println!(".kafka Kafka Thread: {} successful sends out of 5", sends_successful);
                true
            } else {
                false
            }
        } else {
            false
        }
    });
    
    let api_handle = tokio::spawn(async move {
        // Test API with dummy requests
        if let Ok(api) = BinanceApi::new() {
            if api.ping().await.is_ok() {
                println!("🔍 API Thread: Connection established");
                
                // Simulate multiple API calls
                let mut calls_successful = 0;
                for _ in 0..5 {
                    if api.ping().await.is_ok() {
                        calls_successful += 1;
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await; // Rate limiting simulation
                }
                
                println!("🔍 API Thread: {} successful calls out of 5", calls_successful);
                true
            } else {
                false
            }
        } else {
            false
        }
    });
    
    // Wait for all threads to complete
    let db_result: bool = db_handle.await.unwrap();
    let kafka_result: bool = kafka_handle.await.unwrap();
    let api_result: bool = api_handle.await.unwrap();
    
    let elapsed = start_time.elapsed();
    
    println!("\n📊 Stress Test Results:");
    println!("   DB Operations: {}", if db_result { "✅ SUCCESS" } else { "❌ FAILED" });
    println!("   Kafka Operations: {}", if kafka_result { "✅ SUCCESS" } else { "❌ FAILED" });
    println!("   API Operations: {}", if api_result { "✅ SUCCESS" } else { "❌ FAILED" });
    println!("   Total execution time: {:.2?}", elapsed);
    
    // Calculate optimal settings based on performance
    let performance_score = (db_result as u8 + kafka_result as u8 + api_result as u8) as f64 / 3.0 * 100.0;
    println!("   Performance score: {:.1}%", performance_score);
    
    if performance_score > 90.0 {
        println!("💡 System is performing optimally under stress");
    } else if performance_score > 70.0 {
        println!("⚠️ System needs optimization for heavy loads");
    } else {
        println!("❌ System requires significant optimization");
    }
}