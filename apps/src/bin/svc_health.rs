use anyhow::Result;
use common::config::AppConfig;
use common::health::ServiceStatus;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio_postgres::{Client as PgClient, NoTls};
use warp::Filter;

// Глобальный клиент для проверки БД
struct AppState {
    pg_client: PgClient,
    status_map: Arc<RwLock<HashMap<String, ServiceStatus>>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().init();

    let config = common::config::load_config()?;
    let health_port = std::env::var("SVC_HEALTH_PORT")
        .unwrap_or_else(|_| "9005".to_string())
        .parse::<u16>()?;

    // Подключаемся к БД один раз
    let db_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| {
        "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string()
    });

    let (pg_client, connection) = tokio_postgres::connect(&db_url, NoTls).await?;

    // Спавним задачу для поддержания соединения
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::error!("Database connection error: {}", e);
        }
    });

    let status_map = Arc::new(RwLock::new(HashMap::new()));
    status_map.write().unwrap().insert(
        "svc_health".to_string(),
        ServiceStatus {
            service_name: "svc_health".to_string(),
            status: "UP".to_string(),
            last_seen_ts: chrono::Utc::now().timestamp(),
            details: None,
        },
    );

    let app_state = Arc::new(AppState {
        pg_client,
        status_map,
    });

    // Маршрут /health — текущий
    let state_for_health = app_state.clone();
    let health_route = warp::path!("health").map(move || {
        let map = &state_for_health.status_map.read().unwrap();
        warp::reply::json(&*map)
    });

    // НОВЫЙ маршрут: /health/stage
    let state_for_stage = app_state.clone();
    let stage_route = warp::path!("health" / "stage").map(move || {
        let client = &state_for_stage.pg_client;
        let rt = tokio::runtime::Handle::current();

        // Выполняем синхронный запрос через блокирующий вызов
        let has_pairs = rt.block_on(async {
            match client
                .query_one("SELECT COUNT(*) FROM market.pairs", &[])
                .await
            {
                Ok(row) => row.get::<_, i64>(0) > 0,
                Err(e) => {
                    tracing::warn!("Failed to check pairs: {}", e);
                    false
                }
            }
        });

        let stage = if has_pairs {
            "PAIRS_READY"
        } else {
            "INITIALIZING"
        };

        warp::reply::json(&serde_json::json!({ "stage": stage }))
    });

    let routes = health_route.or(stage_route);

    tracing::info!("Health Check Service running on port {}", health_port);
    warp::serve(routes).run(([0, 0, 0, 0], health_port)).await;

    Ok(())
}
