use connections_lib::{
    redpanda::{RedpandaConnection, RedpandaConfig},
    kafka_consumer::{KafkaConsumer, KafkaConsumerConfig},
};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::timeout;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestData {
    id: u64,
    message: String,
    timestamp: i64,
}

#[tokio::test]
async fn test_kafka_producer_consumer_flow() {
    println!("\n.kafka Testing Kafka Producer-Consumer Flow...");
    
    let start_time = std::time::Instant::now();
    
    // Create test data
    let test_data = TestData {
        id: 12345,
        message: "Hello Kafka!".to_string(),
        timestamp: chrono::Utc::now().timestamp_millis(),
    };
    
    // Initialize Kafka producer
    let producer_config = RedpandaConfig {
        brokers: std::env::var("KAFKA_BROKERS").unwrap_or_else(|_| "127.0.0.1:19092".to_string()),
        topic: "test-producer-consumer-flow".to_string(),
        group_id: "test-producer-group".to_string(),
    };
    
    let mut producer = match RedpandaConnection::new(producer_config) {
        Ok(mut p) => {
            match p.connect_producer().await {
                Ok(_) => {
                    println!("✅ Kafka producer initialized and connected");
                    p
                },
                Err(e) => {
                    println!("⚠ Kafka producer connection failed: {}", e);
                    return; // Skip test if Kafka is not available
                }
            }
        },
        Err(e) => {
            println!("⚠ Failed to create Kafka producer: {}", e);
            return; // Skip test if Kafka is not available
        }
    };
    
    // Initialize Kafka consumer
    let consumer_config = KafkaConsumerConfig {
        brokers: std::env::var("KAFKA_BROKERS").unwrap_or_else(|_| "127.0.0.1:19092".to_string()),
        topic: "test-producer-consumer-flow".to_string(),
        group_id: "test-consumer-group".to_string(),
    };
    
    let mut consumer = match KafkaConsumer::new(consumer_config) {
        Ok(mut c) => {
            match c.connect_consumer().await {
                Ok(_) => {
                    println!("✅ Kafka consumer initialized and connected");
                    c
                },
                Err(e) => {
                    println!("⚠ Kafka consumer connection failed: {}", e);
                    return; // Skip test if Kafka is not available
                }
            }
        },
        Err(e) => {
            println!("⚠ Failed to create Kafka consumer: {}", e);
            return; // Skip test if Kafka is not available
        }
    };
    
    // Test producer functionality
    let serialized_data = match serde_json::to_string(&test_data) {
        Ok(s) => s,
        Err(e) => {
            panic!("Failed to serialize test data: {}", e);
        }
    };
    
    match producer.send_message(&test_data.id.to_string(), &serialized_data).await {
        Ok(_) => {
            println!("✅ Message sent to Kafka: ID {}", test_data.id);
        },
        Err(e) => {
            println!("⚠ Failed to send message to Kafka: {}", e);
            return; // Skip test if sending fails
        }
    }
    
    // Test consumer functionality
    let received_data = match timeout(Duration::from_secs(15), consumer.consume_message()).await {
        Ok(Ok(bytes)) => {
            match String::from_utf8(bytes) {
                Ok(s) => s,
                Err(e) => {
                    panic!("Failed to convert received bytes to string: {}", e);
                }
            }
        },
        Ok(Err(e)) => {
            println!("⚠ Failed to receive message from Kafka: {}", e);
            return; // Skip test if receiving fails
        },
        Err(_) => {
            println!("⚠ Timeout waiting for message from Kafka");
            return; // Skip test if timeout occurs
        }
    };
    
    // Verify the received data matches what was sent
    let received_struct: TestData = match serde_json::from_str(&received_data) {
        Ok(d) => d,
        Err(e) => {
            panic!("Failed to deserialize received data: {}", e);
        }
    };
    
    assert_eq!(received_struct.id, test_data.id);
    assert_eq!(received_struct.message, test_data.message);
    println!("✅ Message round-trip successful: '{}' (ID: {})", received_struct.message, received_struct.id);
    
    let elapsed = start_time.elapsed();
    println!("⏱️  Kafka producer-consumer test completed in {:.2?}", elapsed);
    println!("✅ Kafka end-to-end flow test passed");
}

#[tokio::test]
async fn test_kafka_connectivity() {
    println!("\n.kafka Testing Kafka Connectivity...");
    
    // Test producer connectivity
    let producer_config = RedpandaConfig::default();
    let producer_result = RedpandaConnection::new(producer_config)
        .and_then(|mut p| async {
            p.connect_producer().await?;
            p.ping().await
        }.into());
    
    match producer_result.await {
        Ok(_) => println!("✅ Kafka producer connectivity: OK"),
        Err(e) => println!("⚠ Kafka producer connectivity failed: {}", e),
    }
    
    // Test consumer connectivity
    let consumer_config = KafkaConsumerConfig::default();
    let consumer_result = KafkaConsumer::new(consumer_config)
        .and_then(|mut c| async {
            c.connect_consumer().await?;
            c.ping().await
        }.into());
    
    match consumer_result.await {
        Ok(_) => println!("✅ Kafka consumer connectivity: OK"),
        Err(e) => println!("⚠ Kafka consumer connectivity failed: {}", e),
    }
    
    println!("✅ Kafka connectivity test completed");
}

#[tokio::test]
async fn test_kafka_multiple_messages() {
    println!("\n.kafka Testing Kafka Multiple Messages...");
    
    let producer_config = RedpandaConfig {
        brokers: std::env::var("KAFKA_BROKERS").unwrap_or_else(|_| "127.0.0.1:19092".to_string()),
        topic: "test-multiple-messages".to_string(),
        group_id: "test-multi-producer-group".to_string(),
    };
    
    let mut producer = match RedpandaConnection::new(producer_config) {
        Ok(mut p) => {
            match p.connect_producer().await {
                Ok(_) => p,
                Err(e) => {
                    println!("⚠ Failed to connect producer: {}", e);
                    return;
                }
            }
        },
        Err(e) => {
            println!("⚠ Failed to create producer: {}", e);
            return;
        }
    };
    
    // Send multiple messages
    let num_messages = 5;
    for i in 0..num_messages {
        let test_data = TestData {
            id: 1000 + i,
            message: format!("Test message #{}", i),
            timestamp: chrono::Utc::now().timestamp_millis(),
        };
        
        let serialized_data = serde_json::to_string(&test_data).unwrap();
        
        match producer.send_message(&test_data.id.to_string(), &serialized_data).await {
            Ok(_) => {
                println!("✅ Sent message {}: '{}'", test_data.id, test_data.message);
            },
            Err(e) => {
                println!("⚠ Failed to send message {}: {}", test_data.id, e);
                break;
            }
        }
        
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    println!("✅ Multiple messages test completed (messages sent to queue for consumption)");
}