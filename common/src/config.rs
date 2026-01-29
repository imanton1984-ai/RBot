use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub database: DatabaseConfig,
    pub binance: BinanceConfig,
    pub universe: UniverseConfig,
    pub runtime: RuntimeConfig,
    pub compute: ComputeConfig,
    pub rust_bot: RustBotConfig,
    pub ports: PortsConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub name: String,
    pub user: String,
    pub password: String,
    pub host: String,
    pub port: u16,
    pub pool_size: u32,
    pub ssl_mode: String, // "disable"|"require"|...
}

impl DatabaseConfig {
    pub fn url(&self) -> String {
        // sslmode: disable/require/verify-ca/verify-full (postgres style)
        format!(
            "postgres://{}:{}@{}:{}/{}?sslmode={}",
            self.user, self.password, self.host, self.port, self.name, self.ssl_mode
        )
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct BinanceConfig {
    pub rest_base_url: String,
    pub ws_base_url: String,

    pub http_timeout_ms: u64,
    pub http_retries: u32,
    pub http_retry_backoff_ms: u64,
    pub http_retry_backoff_max_ms: u64,

    pub ws_ping_interval_sec: u64,
    pub ws_reconnect_backoff_ms: u64,
    pub ws_reconnect_backoff_max_ms: u64,

    pub enable_time_sync: bool,
    pub time_sync_interval_sec: u64,
    pub time_sync_max_skew_ms: u64,

    pub rate_limit_soft_rps: u32,
    pub rate_limit_soft_burst: u32,

    pub futures_default_leverage: u16,
    pub futures_margin_type: String,
    pub recv_window_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UniverseConfig {
    pub quote_asset: String,
    pub futures_only: bool,
    pub perpetual_only: bool,

    pub min_quote_volume_usdt_24h: f64,
    pub refresh_interval_sec: u64,
    pub max_change_ratio: f64,

    pub exclude_base_assets: Vec<String>,
    pub exclude_symbols: Vec<String>,

    pub allowlist: Vec<String>,
    pub denylist: Vec<String>,

    pub apply_mode: String, // "startup_required"|"background"
}

#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeConfig {
    pub timeframes: Vec<String>, // ["1m","5m","..."]
    pub backfill_candles: usize,

    pub realtime_ws_timeframes: Vec<String>,   // для WS
    pub realtime_poll_timeframes: Vec<String>, // для poll/on-close
    pub poll_on_close_interval_sec: u64,

    pub http_concurrency: Option<usize>,

    pub enable_intra_candle_updates: bool,

    pub persist_live_candle: bool,
    pub persist_live_candle_every_ms: u64,

    pub intra_update_sampling_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ComputeConfig {
    pub gpu_enabled: bool,
    pub gpu_preferred: bool,
    pub cpu_enabled: bool,

    pub realtime_default_backend: String, // "cpu"|"gpu"
    pub backfill_default_backend: String, // "cpu"|"gpu"

    pub gpu_catchup_backlog_candles: usize,
    pub gpu_catchup_max_latency_ms: u64,

    pub gpu_batch_symbols: usize,
    pub gpu_batch_timeframes: usize,
    pub gpu_streams: u32,

    pub hot_window_candles: usize,
    pub hot_indicator_window: usize,
    #[serde(default)]
    pub indicators: IndicatorFlags,

    pub raw_signals: RawSignalsConfig,
    pub final_signals: FinalSignalsConfig,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IndicatorFlags {
    pub rsi: bool,
    pub macd: bool,
    pub ema20: bool,
    pub ema50: bool,
    pub ema200: bool,
    pub sma: bool,
    pub bollinger: bool,
    pub stoch: bool,
    pub atr: bool,
    pub adx: bool,
    pub vwap: bool,
    pub obv: bool,
    pub cci: bool,
    pub williams: bool,
    pub alligator: bool,
    pub trend: bool,
    pub volume_spike: bool,
    pub sr_levels: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawSignalsConfig {
    pub min_indicator_score: f64,
    pub min_publish_score: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FinalSignalsConfig {
    pub min_final_score: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RustBotConfig {
    pub env: String,      // dev|prod
    pub run_mode: String, // paper|live
    pub instance_id: String,

    pub log_level: String,
    pub json_logs: bool,

    pub redpanda_brokers: Vec<String>,
    pub redpanda_client_id: String,

    pub topic_candles_close: String,
    pub topic_candles_update: String,
    pub topic_indicators_close: String,
    pub topic_raw_signals: String,
    pub topic_final_signals: String,
    pub topic_features_snapshot: String,
    pub topic_orders_cmd: String,
    pub topic_orders_events: String,
    pub topic_positions_events: String,
    #[serde(default)]
    pub warmup: WarmupConfig,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct WarmupConfig {
    pub require_db_schema_ready: bool,
    pub require_universe_ready: bool,
    pub require_backfill_ready: bool,
    pub require_bootstrap_indicators_ready: bool,
    pub require_model_ready: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PortsConfig {
    pub api_gateway: u16,
    pub market_ingest: u16,
    pub compute_core: u16,
    pub writer: u16,
    pub order_engine: u16,
    pub position_tracker: u16,
}

pub fn load_config() -> Result<AppConfig, config::ConfigError> {
    // Все файлы читаем из ./config/*.toml
    // Env override: BOT__DATABASE__HOST=... (separator "__")
    let builder = config::Config::builder()
        .add_source(config::File::with_name("config/database").required(false))
        .add_source(config::File::with_name("config/binance").required(false))
        .add_source(config::File::with_name("config/universe").required(false))
        .add_source(config::File::with_name("config/runtime").required(false))
        .add_source(config::File::with_name("config/compute").required(false))
        .add_source(config::File::with_name("config/rust_bot").required(false))
        .add_source(
            config::Environment::with_prefix("BOT")
                .separator("__")
                .try_parsing(true)
                .list_separator(","),
        );

    let cfg = builder.build()?;
    cfg.try_deserialize::<AppConfig>()
}
