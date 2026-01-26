use anyhow::Result;
use common::config::AppConfig;
use common::health::ServiceStatus; // Берем из common
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use warp::Filter;

// Локальная структура монитора
pub struct HealthMonitor {
    status_map: Arc<RwLock<HashMap<String, ServiceStatus>>>,
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self {
            status_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn update_service(&self, status: ServiceStatus) {
        let mut map = self.status_map.write().unwrap();
        map.insert(status.service_name.clone(), status);
    }

    pub async fn run_server(&self, port: u16) {
        let status_map = self.status_map.clone();

        let health_route = warp::path("health").map(move || {
            let map = status_map.read().unwrap();
            warp::reply::json(&*map)
        });

        tracing::info!("Health Check Service running on port {}", port);
        warp::serve(health_route).run(([0, 0, 0, 0], port)).await;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Инициализация логгера
    tracing_subscriber::fmt().init();

    // Загрузка конфига
    let config = common::config::load_config()?; // Предполагается, что load() существует в common::config
    let health_port = std::env::var("SVC_HEALTH_PORT")
        .unwrap_or_else(|_| "9005".to_string())
        .parse::<u16>()?;

    tracing::info!("Starting svc_health on port {}", health_port);

    let monitor = HealthMonitor::new();

    // Пример: добавляем статус самого себя
    monitor.update_service(ServiceStatus {
        service_name: "svc_health".to_string(),
        status: "UP".to_string(),
        last_seen_ts: chrono::Utc::now().timestamp(),
        details: None,
    });

    monitor.run_server(health_port).await;

    Ok(())
}
