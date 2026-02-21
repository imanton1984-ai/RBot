// webui/src/bin/server.rs

use std::net::SocketAddr;
use tower_http::{cors::{Any, CorsLayer}, trace::TraceLayer};
use axum::{routing::get, Router, extract::ws::WebSocketUpgrade, response::IntoResponse, extract::State};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use sqlx::postgres::PgPoolOptions;

use webui::state::AppState;
use webui::api::routes::create_api_router;
use webui::ws::ws_handler;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "webui=warn,warn".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting WebUI server...");

    let database_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());
    let pool = PgPoolOptions::new().max_connections(20).connect(&database_url).await?;
    tracing::info!("Connected to database");

    let state = AppState::new(pool, webui::state::WebUiSettings::default());
    let cors = CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any);
    let app = create_app(state.clone(), cors);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::info!("WebUI server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn create_app(state: AppState, cors: CorsLayer) -> Router {
    let static_files = tower_http::services::ServeDir::new("webui/dist")
        .fallback(tower_http::services::ServeFile::new("webui/dist/index.html"));

    Router::new()
        .nest("/api", create_api_router())
        .route("/ws", get(ws_route))
        .route("/health", get(|| async { "OK" }))
        .fallback_service(static_files)
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn ws_route(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws_handler(ws, State(state)).await
}
