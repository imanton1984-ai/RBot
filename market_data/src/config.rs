use anyhow::{Context, Result};
use common::config::load_config;
use common::timeframe::Timeframe;
use std::env;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct IngestConfig {
    pub db_url: String,
    pub redpanda_brokers: String, // "host:port,host2:port2"
    pub redpanda_client_id: String,
    pub rest_base_url: String,
    pub ws_base_url: String,
    pub topic_candles_close: String,

    pub health_port: u16,

    pub backfill_candles: usize,
    pub http_concurrency: usize,

    // важно для защиты от 418/ban
    pub http_soft_rps: u32,
    pub http_soft_burst: u32,

    pub timeframes: Vec<Timeframe>,
    pub realtime_ws_timeframes: Vec<Timeframe>,
    pub ws_symbol_streams_max: usize,
    pub ws_max_streams_per_conn: usize,

    pub poll_timeframes: Vec<Timeframe>,
    pub poll_on_close_interval_sec: u64,

    pub universe_refresh_interval_sec: u64,
}

fn parse_brokers_env() -> Option<String> {
    env::var("KAFKA_BROKERS").ok().map(|s| {
        s.split(|c| c == ',' || c == ' ')
            .filter(|x| !x.trim().is_empty())
            .map(|x| x.trim().to_string())
            .collect::<Vec<_>>()
            .join(",")
    })
}

fn env_u16(key: &str) -> Option<u16> {
    env::var(key).ok().and_then(|v| v.parse::<u16>().ok())
}

pub fn env_port_market_ingest() -> u16 {
    env_u16("MARKET_INGEST_PORT")
        .or_else(|| env_u16("MARKET_DATA_PORT"))
        .unwrap_or(9001)
}

impl IngestConfig {
    pub fn load() -> Result<Arc<Self>> {
        let cfg = load_config().context("common::config::load_config failed")?;

        let db_url = env::var("DATABASE_URL")
            .or_else(|_| env::var("DB_URL"))
            .unwrap_or_else(|_| cfg.database.url());

        let redpanda_brokers =
            parse_brokers_env().unwrap_or_else(|| cfg.rust_bot.redpanda_brokers.join(","));

        let health_port = env_port_market_ingest();

        Ok(Arc::new(Self {
            db_url,
            redpanda_brokers,
            redpanda_client_id: cfg.rust_bot.redpanda_client_id.clone(),
            rest_base_url: cfg.binance.rest_base_url.clone(),
            ws_base_url: cfg.binance.ws_base_url.clone(),
            topic_candles_close: cfg.rust_bot.topic_candles_close.clone(),

            health_port,

            backfill_candles: cfg.runtime.backfill_candles,
            http_concurrency: cfg.runtime.http_concurrency.unwrap_or(8),
            http_soft_rps: cfg.binance.rate_limit_soft_rps,
            http_soft_burst: cfg.binance.rate_limit_soft_burst,

            timeframes: cfg
                .runtime
                .timeframes
                .iter()
                .map(|s| Timeframe::parse(s))
                .collect::<Result<Vec<_>, _>>()?,
            realtime_ws_timeframes: cfg
                .runtime
                .realtime_ws_timeframes
                .iter()
                .map(|s| Timeframe::parse(s))
                .collect::<Result<Vec<_>, _>>()?,
            ws_symbol_streams_max: 250, // можно вынести в конфиг позже
            ws_max_streams_per_conn: 180,

            poll_timeframes: cfg
                .runtime
                .realtime_poll_timeframes
                .iter()
                .map(|s| Timeframe::parse(s))
                .collect::<Result<Vec<_>, _>>()?,
            poll_on_close_interval_sec: cfg.runtime.poll_on_close_interval_sec,

            universe_refresh_interval_sec: cfg.universe.refresh_interval_sec,
        }))
    }
}
