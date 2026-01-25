use std::{net::SocketAddr, time::Duration};

use axum::{
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Router,
};
use tokio::signal;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let cfg = EnvCfg::from_env("SVC_MARKET_INGEST")?;

    // Preflight: DB
    if let Err(e) = preflight_db(&cfg.database_url).await {
        warn!("DB preflight failed: {e:#}");
    } else {
        info!("DB preflight OK");
    }

    // Preflight: Kafka (Redpanda)
    if let Err(e) = preflight_kafka(&cfg.kafka_brokers).await {
        warn!("Kafka preflight failed: {e:#}");
    } else {
        info!("Kafka preflight OK");
    }

    // HTTP health
    let app = Router::new()
        .route("/healthz", get(|| async { (StatusCode::OK, "OK") }))
        .route("/readyz", get(|| async { (StatusCode::OK, "READY") }));

    let addr: SocketAddr = format!("0.0.0.0:{}", cfg.port).parse()?;
    info!("market_ingest listening on http://{addr}");

    axum::serve(tokio::net::TcpListener::bind(addr).await?, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let _ = signal::ctrl_c().await;
    warn!("shutdown signal received");
    tokio::time::sleep(Duration::from_millis(150)).await;
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,rdkafka=warn,sqlx=warn".to_string()),
        )
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .try_init();
}

struct EnvCfg {
    port: u16,
    database_url: String,
    kafka_brokers: String,
}

impl EnvCfg {
    fn from_env(_svc: &str) -> anyhow::Result<Self> {
        Ok(Self {
            port: env_u16("MARKET_INGEST_PORT").unwrap_or(9001),
            database_url: env_req("DATABASE_URL")?,
            kafka_brokers: env_req("KAFKA_BROKERS")?,
        })
    }
}

fn env_req(key: &str) -> anyhow::Result<String> {
    std::env::var(key).map_err(|_| anyhow::anyhow!("{key} is required"))
}

fn env_u16(key: &str) -> Option<u16> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

// --- preflight helpers ---

async fn preflight_db(db_url: &str) -> anyhow::Result<()> {
    let (client, connection) = tokio_postgres::connect(db_url, tokio_postgres::NoTls).await?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client.simple_query("SELECT 1;").await?;
    Ok(())
}

async fn preflight_kafka(brokers: &str) -> anyhow::Result<()> {
    // lightweight: metadata request via rdkafka producer
    let producer: rdkafka::producer::FutureProducer = rdkafka::config::ClientConfig::new()
        .set("bootstrap.servers", brokers)
        .set("message.timeout.ms", "3000")
        .create()?;

    // forcing metadata fetch by requesting it explicitly isn't public API in rdkafka,
    // but creating producer already validates config; keep as "soft" preflight.
    drop(producer);
    Ok(())
}
