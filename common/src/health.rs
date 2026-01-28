use serde::{Deserialize, Serialize};
// use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub service_name: String,
    pub status: String, // "UP", "DOWN", "DEGRADED"
    pub last_seen_ts: i64,
    pub details: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    pub check_interval_sec: u64,
    pub http_port: u16,
}
