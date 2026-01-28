mod pairs;

use anyhow::Result;
use dotenvy::dotenv;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let min_pairs: i64 = std::env::var("MIN_PAIRS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);

    let res = pairs::refresh_universe_pairs().await?;

    if res.active_cnt < min_pairs {
        anyhow::bail!(
            "pairs_ready=false: active={} < MIN_PAIRS={}",
            res.active_cnt,
            min_pairs
        );
    }

    info!(
        "pairs_ready=true: selected={}, active={} (MIN_PAIRS={})",
        res.selected_cnt,
        res.active_cnt,
        min_pairs
    );

    Ok(())
}

