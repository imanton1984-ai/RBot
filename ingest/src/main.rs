use axum::{
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use anyhow::Result;
use tracing::{error, info};

use ingest::health::{healthz, readyz, stagez, StageState};
use ingest::{universe, AppState};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "ingest=info".into()))
        .init();

    let port: u16 = std::env::var("INGEST_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8081);

    let db_url = std::env::var("DATABASE_URL").expect("DATABASE_URL is required");
    let rest_base = std::env::var("BINANCE_REST_BASE")
        .unwrap_or_else(|_| "https://fapi.binance.com".to_string());
    let universe_cfg_path = std::env::var("UNIVERSE_CONFIG")
        .unwrap_or_else(|_| "config/universe.toml".to_string());

    let stage = StageState::new("BOOT");

    let state = AppState {
        stage: stage.clone(),
        db_url: Arc::new(db_url),
        rest_base: Arc::new(rest_base),
        universe_cfg_path: Arc::new(universe_cfg_path),
    };

    // Startup universe refresh
    {
        let st = state.clone();
        tokio::spawn(async move {
            if let Err(e) = universe_startup(st).await {
                error!("universe startup failed: {e:#}");
            }
        });
    }

    let app = Router::new()
    .route("/healthz", get(healthz))
    .route("/readyz", get(readyz))
    .route("/stagez", get(stagez))
    .route("/universe/refresh", post(universe_refresh))
    .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("ingest listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn universe_refresh(
    axum::extract::State(st): axum::extract::State<AppState>,
) -> impl axum::response::IntoResponse {
    match universe::refresh_pairs(&st.db_url, &st.rest_base, &st.universe_cfg_path).await {
        Ok(count) => {
            st.stage.set("PAIRS_READY", format!("pairs loaded: {count}"));
            (
                axum::http::StatusCode::OK,
                axum::Json(serde_json::json!({ "ok": true, "pairs": count })),
            )
        }
        Err(e) => {
            st.stage.set("STUCK", format!("universe refresh failed: {e:#}"));
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({ "ok": false, "error": format!("{e:#}") })),
            )
        }
    }
}

async fn universe_startup(st: AppState) -> Result<()> {
    st.stage.set("LOADING_PAIRS", "startup refresh");

    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        match universe::refresh_pairs(&st.db_url, &st.rest_base, &st.universe_cfg_path).await {
            Ok(count) => {
                st.stage.set("PAIRS_READY", format!("pairs loaded: {count}"));
                info!("PAIRS_READY (pairs={count})");
                return Ok(());
            }
            Err(e) => {
                st.stage.set("LOADING_PAIRS", format!("attempt {attempt}: {e:#}"));
                error!("attempt {attempt}: {e:#}");
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }
    }
}




