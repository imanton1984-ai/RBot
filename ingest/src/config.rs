use anyhow::{anyhow, Context, Result};
use common::config::load_config;
use common::timeframe::Timeframe;
use std::env;

#[derive(Debug, Clone)]
pub struct IngestConfig {
    pub db_url: String,

    pub rest_base_url: String,
    pub ws_base_url: String,

    pub redpanda_brokers: Vec<String>,
    pub redpanda_client_id: String,

    pub topic_candles_close: String,

    pub timeframes: Vec<Timeframe>,
    pub backfill_candles: usize,

    pub ws_timeframes: Vec<Timeframe>,
    pub poll_timeframes: Vec<Timeframe>,
    pub poll_on_close_interval_sec: u64,

    pub ws_max_streams_per_conn: usize,
    pub http_concurrency: usize,

    pub universe_cfg_path: String,

    pub enable_intra_candle_updates: bool,
    pub persist_live_candle: bool,
    pub persist_live_candle_every_ms: u64,
    pub intra_update_sampling_ms: u64,
}

impl IngestConfig {
    pub fn load() -> Result<Self> {
        let cfg = load_config().context("load_config() failed")?;

        // DB url: prefer explicit DATABASE_URL if set (как у тебя уже было)
        let db_url = env::var("DATABASE_URL").unwrap_or_else(|_| cfg.database.url());

        // Binance base urls (env overrides оставляем)
        let rest_base_url = env::var("BINANCE_REST_BASE").unwrap_or_else(|_| cfg.binance.rest_base_url);
        let ws_base_url = env::var("BINANCE_WS_BASE").unwrap_or_else(|_| cfg.binance.ws_base_url);

        let universe_cfg_path = env::var("UNIVERSE_CONFIG").unwrap_or_else(|_| "config/universe.toml".to_string());

        let parse_tfs = |xs: &[String]| -> Result<Vec<Timeframe>> {
            xs.iter()
                .map(|s| Timeframe::parse(s).map_err(|_| anyhow!("bad timeframe: {}", s)))
                .collect()
        };

        let timeframes = parse_tfs(&cfg.runtime.timeframes)?;
        let ws_timeframes = parse_tfs(&cfg.runtime.realtime_ws_timeframes)?;
        let poll_timeframes = parse_tfs(&cfg.runtime.realtime_poll_timeframes)?;

        Ok(Self {
            db_url,
            rest_base_url,
            ws_base_url,

            redpanda_brokers: cfg.rust_bot.redpanda_brokers,
            redpanda_client_id: cfg.rust_bot.redpanda_client_id,

            topic_candles_close: cfg.rust_bot.topic_candles_close,

            timeframes,
            backfill_candles: cfg.runtime.backfill_candles,

            ws_timeframes,
            poll_timeframes,
            poll_on_close_interval_sec: cfg.runtime.poll_on_close_interval_sec,

            // WS лимит у тебя прямо в runtime.toml описан (1024) :contentReference[oaicite:9]{index=9}
            ws_max_streams_per_conn: 950, // безопасный запас < 1024
            http_concurrency: 50,         // чтобы не словить 429 на backfill

            universe_cfg_path,

            enable_intra_candle_updates: cfg.runtime.enable_intra_candle_updates,
            persist_live_candle: cfg.runtime.persist_live_candle,
            persist_live_candle_every_ms: cfg.runtime.persist_live_candle_every_ms,
            intra_update_sampling_ms: cfg.runtime.intra_update_sampling_ms,
        })
    }
}
