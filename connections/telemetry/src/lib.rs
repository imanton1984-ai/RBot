use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, Duration};
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStatus {
    pub connected: bool,
    pub last_checked: u64,
    pub latency_ms: Option<u64>,
    pub error_msg: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryReport {
    pub timestamp: u64,
    pub binance_rest: ConnectionStatus,
    pub binance_ws: ConnectionStatus,
    pub database: ConnectionStatus,
    pub kafka: ConnectionStatus,
}

pub struct ConnectionTelemetry {
    report: TelemetryReport,
    checks_enabled: bool,
}

impl ConnectionTelemetry {
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            report: TelemetryReport {
                timestamp: now,
                binance_rest: ConnectionStatus {
                    connected: false,
                    last_checked: now,
                    latency_ms: None,
                    error_msg: None,
                },
                binance_ws: ConnectionStatus {
                    connected: false,
                    last_checked: now,
                    latency_ms: None,
                    error_msg: None,
                },
                database: ConnectionStatus {
                    connected: false,
                    last_checked: now,
                    latency_ms: None,
                    error_msg: None,
                },
                kafka: ConnectionStatus {
                    connected: false,
                    last_checked: now,
                    latency_ms: None,
                    error_msg: None,
                },
            },
            checks_enabled: true,
        }
    }

    pub fn with_checks_enabled(mut self, enabled: bool) -> Self {
        self.checks_enabled = enabled;
        self
    }

    pub async fn update_binance_rest_status<F>(&mut self, check_fn: F) -> Result<()>
    where
        F: FnOnce() -> Result<bool>,
    {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64;

        match check_fn() {
            Ok(connected) => {
                let end_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)?
                    .as_millis() as u64;
                
                self.report.binance_rest = ConnectionStatus {
                    connected,
                    last_checked: SystemTime::now()
                        .duration_since(UNIX_EPOCH)?
                        .as_secs(),
                    latency_ms: Some(end_time - start_time),
                    error_msg: None,
                };
            }
            Err(e) => {
                self.report.binance_rest = ConnectionStatus {
                    connected: false,
                    last_checked: SystemTime::now()
                        .duration_since(UNIX_EPOCH)?
                        .as_secs(),
                    latency_ms: None,
                    error_msg: Some(e.to_string()),
                };
                error!("Binance REST health check failed: {}", e);
            }
        }

        Ok(())
    }

    pub async fn update_binance_ws_status<F>(&mut self, check_fn: F) -> Result<()>
    where
        F: FnOnce() -> Result<bool>,
    {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64;

        match check_fn() {
            Ok(connected) => {
                let end_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)?
                    .as_millis() as u64;
                
                self.report.binance_ws = ConnectionStatus {
                    connected,
                    last_checked: SystemTime::now()
                        .duration_since(UNIX_EPOCH)?
                        .as_secs(),
                    latency_ms: Some(end_time - start_time),
                    error_msg: None,
                };
            }
            Err(e) => {
                self.report.binance_ws = ConnectionStatus {
                    connected: false,
                    last_checked: SystemTime::now()
                        .duration_since(UNIX_EPOCH)?
                        .as_secs(),
                    latency_ms: None,
                    error_msg: Some(e.to_string()),
                };
                error!("Binance WebSocket health check failed: {}", e);
            }
        }

        Ok(())
    }

    pub async fn update_database_status<F>(&mut self, check_fn: F) -> Result<()>
    where
        F: FnOnce() -> Result<bool>,
    {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64;

        match check_fn() {
            Ok(connected) => {
                let end_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)?
                    .as_millis() as u64;
                
                self.report.database = ConnectionStatus {
                    connected,
                    last_checked: SystemTime::now()
                        .duration_since(UNIX_EPOCH)?
                        .as_secs(),
                    latency_ms: Some(end_time - start_time),
                    error_msg: None,
                };
            }
            Err(e) => {
                self.report.database = ConnectionStatus {
                    connected: false,
                    last_checked: SystemTime::now()
                        .duration_since(UNIX_EPOCH)?
                        .as_secs(),
                    latency_ms: None,
                    error_msg: Some(e.to_string()),
                };
                error!("Database health check failed: {}", e);
            }
        }

        Ok(())
    }

    pub async fn update_kafka_status<F>(&mut self, check_fn: F) -> Result<()>
    where
        F: FnOnce() -> Result<bool>,
    {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64;

        match check_fn() {
            Ok(connected) => {
                let end_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)?
                    .as_millis() as u64;
                
                self.report.kafka = ConnectionStatus {
                    connected,
                    last_checked: SystemTime::now()
                        .duration_since(UNIX_EPOCH)?
                        .as_secs(),
                    latency_ms: Some(end_time - start_time),
                    error_msg: None,
                };
            }
            Err(e) => {
                self.report.kafka = ConnectionStatus {
                    connected: false,
                    last_checked: SystemTime::now()
                        .duration_since(UNIX_EPOCH)?
                        .as_secs(),
                    latency_ms: None,
                    error_msg: Some(e.to_string()),
                };
                error!("Kafka health check failed: {}", e);
            }
        }

        Ok(())
    }

    pub fn get_report(&self) -> &TelemetryReport {
        &self.report
    }

    pub fn is_everything_connected(&self) -> bool {
        self.report.binance_rest.connected
            && self.report.binance_ws.connected
            && self.report.database.connected
            && self.report.kafka.connected
    }

    pub fn get_overall_status(&self) -> HashMap<String, bool> {
        let mut status = HashMap::new();
        status.insert("binance_rest".to_string(), self.report.binance_rest.connected);
        status.insert("binance_ws".to_string(), self.report.binance_ws.connected);
        status.insert("database".to_string(), self.report.database.connected);
        status.insert("kafka".to_string(), self.report.kafka.connected);
        status
    }

    pub async fn start_monitoring(&mut self) {
        if !self.checks_enabled {
            info!("Connection monitoring is disabled");
            return;
        }

        info!("Starting connection telemetry monitoring...");

        // Spawn a background task to continuously monitor connections
        let report_clone = self.report.clone();
        let checks_enabled = self.checks_enabled;
        
        tokio::spawn(async move {
            let mut local_report = report_clone;
            let mut interval = tokio::time::interval(Duration::from_secs(10)); // Check every 10 seconds
            
            loop {
                interval.tick().await;
                
                if !checks_enabled {
                    continue;
                }

                // Update timestamp
                local_report.timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();

                // Log overall status
                let all_connected = local_report.binance_rest.connected
                    && local_report.binance_ws.connected
                    && local_report.database.connected
                    && local_report.kafka.connected;

                if all_connected {
                    debug!("All connections are healthy");
                } else {
                    warn!(
                        "Some connections are unhealthy: Binance REST={}, Binance WS={}, Database={}, Kafka={}",
                        local_report.binance_rest.connected,
                        local_report.binance_ws.connected,
                        local_report.database.connected,
                        local_report.kafka.connected
                    );
                }
            }
        });
    }

    pub fn log_status(&self) {
        info!(
            "Connection Status - Binance REST: {}, Binance WS: {}, Database: {}, Kafka: {}",
            self.report.binance_rest.connected,
            self.report.binance_ws.connected,
            self.report.database.connected,
            self.report.kafka.connected
        );
    }
}

// Convenience function to initialize tracing
pub fn init_tracing() {
    if tracing_subscriber::fmt().try_init().is_err() {
        warn!("Tracing subscriber already initialized");
    }
}

// Convenience function to initialize metrics (placeholder)
pub fn init_metrics() {
    info!("Metrics collection initialized (placeholder)");
}

// Convenience function to initialize logging (placeholder)
pub fn init_logging() {
    info!("Logging initialized (placeholder)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_telemetry_initialization() {
        let telemetry = ConnectionTelemetry::new();
        assert_eq!(telemetry.get_report().binance_rest.connected, false);
        assert_eq!(telemetry.get_report().binance_ws.connected, false);
        assert_eq!(telemetry.get_report().database.connected, false);
        assert_eq!(telemetry.get_report().kafka.connected, false);
    }

    #[tokio::test]
    async fn test_telemetry_update_functions() {
        let mut telemetry = ConnectionTelemetry::new();
        
        // Test updating binance rest status
        telemetry.update_binance_rest_status(|| Ok(true)).await.unwrap();
        assert_eq!(telemetry.get_report().binance_rest.connected, true);
        
        // Test updating with error
        telemetry.update_binance_rest_status(|| Err(anyhow::anyhow!("Test error"))).await.unwrap();
        assert_eq!(telemetry.get_report().binance_rest.connected, false);
        assert!(telemetry.get_report().binance_rest.error_msg.is_some());
    }
}
