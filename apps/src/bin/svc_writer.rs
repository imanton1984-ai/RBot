use std::{time::Duration};

use anyhow::{Context, Result};
use bytes::BytesMut;
use rdkafka::{
    consumer::{Consumer, StreamConsumer},
    message::Message,
    ClientConfig,
};
use tokio::sync::mpsc;
use tokio_postgres::NoTls;
use tracing::{error, info, warn};
use ryu;

const BATCH_SIZE: usize = 8000;
const BATCH_MAX_WAIT_MS: u64 = 250;

#[derive(Debug, Clone, Copy)]
enum Timeframe {
    M1,
    M5,
    M15,
    H1,
    H4,
    D1,
}

impl Timeframe {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "1m" => Self::M1,
            "5m" => Self::M5,
            "15m" => Self::M15,
            "1h" => Self::H1,
            "4h" => Self::H4,
            "1d" => Self::D1,
            _ => anyhow::bail!("unknown timeframe: {s}"),
        })
    }

    fn candles_table(&self) -> &'static str {
        match self {
            Self::M1 => "market.candles_1m",
            Self::M5 => "market.candles_5m",
            Self::M15 => "market.candles_15m",
            Self::H1 => "market.candles_1h",
            Self::H4 => "market.candles_4h",
            Self::D1 => "market.candles_1d",
        }
    }

    fn indicators_table(&self) -> &'static str {
        match self {
            Self::M1 => "market.indicators_1m",
            Self::M5 => "market.indicators_5m",
            Self::M15 => "market.indicators_15m",
            Self::H1 => "market.indicators_1h",
            Self::H4 => "market.indicators_4h",
            Self::D1 => "market.indicators_1d",
        }
    }
}

#[derive(Debug)]
enum WriteMsg {
    Candle { symbol_id: i64, _tf: Timeframe, evt: CandleCloseMsg },
    Indicators { symbol_id: i64, _tf: Timeframe, evt: IndicatorsSnapshotMsg },
}

#[derive(Debug, serde::Deserialize)]
struct CandleCloseMsg {
    pub symbol: String,
    pub tf: String,
    pub close_time_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub source_event_time_ms: Option<i64>,
}

#[derive(Debug, serde::Deserialize)]
struct IndicatorsSnapshotMsg {
    pub symbol: String,
    pub tf: String,
    pub close_time_ms: i64,

    pub ema20: Option<f32>,
    pub ema50: Option<f32>,
    pub ema200: Option<f32>,
    pub sma: Option<f32>,

    pub rsi: Option<f32>,
    pub macd: Option<f32>,
    pub macd_signal: Option<f32>,
    pub macd_hist: Option<f32>,

    pub atr: Option<f32>,
    pub adx: Option<f32>,

    pub bb_upper: Option<f32>,
    pub bb_mid: Option<f32>,
    pub bb_lower: Option<f32>,
    pub stoch_k: Option<f32>,
    pub stoch_d: Option<f32>,

    pub vwap: Option<f32>,
    pub obv: Option<f32>,
    pub cci: Option<f32>,
    pub williams: Option<f32>,

    pub alli_jaw: Option<f32>,
    pub alli_teeth: Option<f32>,
    pub alli_lips: Option<f32>,

    pub sr_levels: Option<serde_json::Value>,
    pub features_version: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let cfg = EnvCfg::from_env()?;
    info!(
        "data_writer starting: brokers={}, group={}, topics=[{},{}]",
        cfg.kafka_brokers, cfg.kafka_group, cfg.topic_candles_close, cfg.topic_indicators_close
    );

    // preload symbol cache (symbol -> id)
    let cache = preload_symbols(&cfg.database_url).await?;
    info!("symbol cache loaded: {} symbols", cache.len());

    // channels per TF
    let (tx_1m, rx_1m) = mpsc::channel::<WriteMsg>(cfg.channel_capacity);
    let (tx_5m, rx_5m) = mpsc::channel::<WriteMsg>(cfg.channel_capacity);
    let (tx_15m, rx_15m) = mpsc::channel::<WriteMsg>(cfg.channel_capacity);
    let (tx_1h, rx_1h) = mpsc::channel::<WriteMsg>(cfg.channel_capacity);
    let (tx_4h, rx_4h) = mpsc::channel::<WriteMsg>(cfg.channel_capacity);
    let (tx_1d, rx_1d) = mpsc::channel::<WriteMsg>(cfg.channel_capacity);

    spawn_tf_worker("w_1m", Timeframe::M1, cfg.database_url.clone(), rx_1m);
    spawn_tf_worker("w_5m", Timeframe::M5, cfg.database_url.clone(), rx_5m);
    spawn_tf_worker("w_15m", Timeframe::M15, cfg.database_url.clone(), rx_15m);
    spawn_tf_worker("w_1h", Timeframe::H1, cfg.database_url.clone(), rx_1h);
    spawn_tf_worker("w_4h", Timeframe::H4, cfg.database_url.clone(), rx_4h);
    spawn_tf_worker("w_1d", Timeframe::D1, cfg.database_url.clone(), rx_1d);

    // Kafka consumer
    let consumer: StreamConsumer = ClientConfig::new()
        .set("bootstrap.servers", &cfg.kafka_brokers)
        .set("group.id", &cfg.kafka_group)
        .set("enable.partition.eof", "false")
        .set("session.timeout.ms", "45000")
        .set("enable.auto.commit", "false")
        .set("auto.offset.reset", "latest")
        .create()
        .context("create kafka consumer")?;

    consumer
        .subscribe(&[&cfg.topic_candles_close, &cfg.topic_indicators_close])
        .context("subscribe topics")?;

    info!("data_writer consuming...");

    loop {
        let m = consumer.recv().await.context("consumer.recv")?;
        let topic = m.topic().to_string();

        let payload = match m.payload() {
            Some(p) => p,
            None => {
                consumer.commit_message(&m, rdkafka::consumer::CommitMode::Async).ok();
                continue;
            }
        };

        if topic == cfg.topic_candles_close {
            // MessagePack
            match rmp_serde::from_slice::<CandleCloseMsg>(payload) {
                Ok(evt) => {
                    if let Some(symbol_id) = cache.get(&evt.symbol).copied() {
                        if let Ok(tf) = Timeframe::parse(&evt.tf) {
                            let msg = WriteMsg::Candle { symbol_id, _tf: tf, evt };
                            route_tf(&tf, &tx_1m, &tx_5m, &tx_15m, &tx_1h, &tx_4h, &tx_1d, msg)
                                .await?;
                        }
                    }
                }
                Err(e) => {
                    warn!("decode candles_close msgpack failed: {e}");
                }
            }
        } else if topic == cfg.topic_indicators_close {
            match rmp_serde::from_slice::<IndicatorsSnapshotMsg>(payload) {
                Ok(evt) => {
                    if let Some(symbol_id) = cache.get(&evt.symbol).copied() {
                        if let Ok(tf) = Timeframe::parse(&evt.tf) {
                            let msg = WriteMsg::Indicators { symbol_id, _tf: tf, evt };
                            route_tf(&tf, &tx_1m, &tx_5m, &tx_15m, &tx_1h, &tx_4h, &tx_1d, msg)
                                .await?;
                        }
                    }
                }
                Err(e) => warn!("decode indicators_close msgpack failed: {e}"),
            }
        }

        // ✅ commit after enqueue (bounded channel => backpressure)
        consumer.commit_message(&m, rdkafka::consumer::CommitMode::Async).ok();
    }
}

async fn preload_symbols(db_url: &str) -> Result<std::collections::HashMap<String, i64>> {
    let (client, connection) = tokio_postgres::connect(db_url, NoTls).await?;
    tokio::spawn(async move { let _ = connection.await; });

    // Важно: названия колонок/таблицы подстрой под свою схему
    let rows = client
        .query("SELECT symbol, symbol_id FROM market.pairs WHERE is_active = true", &[])
        .await
        .context("query market.pairs")?;

    Ok(rows
        .into_iter()
        .map(|r| (r.get::<_, String>(0), r.get::<_, i64>(1)))
        .collect())
}

async fn route_tf(
    tf: &Timeframe,
    tx_1m: &mpsc::Sender<WriteMsg>,
    tx_5m: &mpsc::Sender<WriteMsg>,
    tx_15m: &mpsc::Sender<WriteMsg>,
    tx_1h: &mpsc::Sender<WriteMsg>,
    tx_4h: &mpsc::Sender<WriteMsg>,
    tx_1d: &mpsc::Sender<WriteMsg>,
    msg: WriteMsg,
) -> Result<()> {
    match tf {
        Timeframe::M1 => tx_1m.send(msg).await.context("send 1m")?,
        Timeframe::M5 => tx_5m.send(msg).await.context("send 5m")?,
        Timeframe::M15 => tx_15m.send(msg).await.context("send 15m")?,
        Timeframe::H1 => tx_1h.send(msg).await.context("send 1h")?,
        Timeframe::H4 => tx_4h.send(msg).await.context("send 4h")?,
        Timeframe::D1 => tx_1d.send(msg).await.context("send 1d")?,
    }
    Ok(())
}

fn spawn_tf_worker(name: &'static str, tf: Timeframe, db_url: String, mut rx: mpsc::Receiver<WriteMsg>) {
    tokio::spawn(async move {
        if let Err(e) = tf_worker_loop(name, tf, &db_url, &mut rx).await {
            error!("worker {name} crashed: {e:#}");
        }
    });
}

async fn tf_worker_loop(
    _name: &'static str,
    tf: Timeframe,
    db_url: &str,
    rx: &mut mpsc::Receiver<WriteMsg>,
) -> Result<()> {
    let (client, connection) = tokio_postgres::connect(db_url, NoTls).await?;
    tokio::spawn(async move { let _ = connection.await; });

    let candles_table = tf.candles_table();
    let indicators_table = tf.indicators_table();

    let mut candles_buf = BytesMut::with_capacity(1024 * 512);
    let mut ind_buf = BytesMut::with_capacity(1024 * 512);

    let mut candles_n = 0usize;
    let mut ind_n = 0usize;
    let mut last_flush = tokio::time::Instant::now();

    loop {
        let timeout = Duration::from_millis(BATCH_MAX_WAIT_MS);
        let msg = tokio::time::timeout(timeout, rx.recv()).await;

        if let Ok(Some(m)) = msg {
            match m {
                WriteMsg::Candle { symbol_id, _tf: _, evt } => {
                    let mut row = CopyRow::new(&mut candles_buf);
                    row.begin();
                    row.i64(evt.close_time_ms);
                    row.i64(symbol_id);
                    row.f64(evt.open);
                    row.f64(evt.high);
                    row.f64(evt.low);
                    row.f64(evt.close);
                    row.f64(evt.volume);
                    if let Some(x) = evt.source_event_time_ms { row.i64(x) } else { row.null() }
                    row.end();
                    candles_n += 1;
                }
                WriteMsg::Indicators { symbol_id, _tf: _, evt } => {
                    let mut row = CopyRow::new(&mut ind_buf);
                    row.begin();
                    row.i64(evt.close_time_ms);
                    row.i64(symbol_id);

                    row.opt_f32(evt.ema20);
                    row.opt_f32(evt.ema50);
                    row.opt_f32(evt.ema200);
                    row.opt_f32(evt.sma);

                    row.opt_f32(evt.rsi);
                    row.opt_f32(evt.macd);
                    row.opt_f32(evt.macd_signal);
                    row.opt_f32(evt.macd_hist);

                    row.opt_f32(evt.atr);
                    row.opt_f32(evt.adx);

                    row.opt_f32(evt.bb_upper);
                    row.opt_f32(evt.bb_mid);
                    row.opt_f32(evt.bb_lower);
                    row.opt_f32(evt.stoch_k);
                    row.opt_f32(evt.stoch_d);

                    row.opt_f32(evt.vwap);
                    row.opt_f32(evt.obv);
                    row.opt_f32(evt.cci);
                    row.opt_f32(evt.williams);

                    row.opt_f32(evt.alli_jaw);
                    row.opt_f32(evt.alli_teeth);
                    row.opt_f32(evt.alli_lips);

                    row.opt_json(evt.sr_levels.as_ref())?;

                    row.str(evt.features_version.as_deref().unwrap_or("v1"));
                    row.i64(now_ms());

                    row.end();
                    ind_n += 1;
                }
            }
        }

        let need_flush = candles_n >= BATCH_SIZE
            || ind_n >= BATCH_SIZE
            || last_flush.elapsed() >= Duration::from_millis(BATCH_MAX_WAIT_MS);

        if !need_flush {
            continue;
        }

        if candles_n > 0 {
            flush_copy(&client, candles_table, &candles_buf).await?;
            candles_buf.clear();
            candles_n = 0;
        }

        if ind_n > 0 {
            flush_copy(&client, indicators_table, &ind_buf).await?;
            ind_buf.clear();
            ind_n = 0;
        }

        last_flush = tokio::time::Instant::now();
    }
}

async fn flush_copy(client: &tokio_postgres::Client, table: &str, buf: &BytesMut) -> Result<()> {
    // формат: CSV в COPY FROM STDIN (быстро и просто)
    // колонки подстрой под свою DDL
    let copy_stmt = format!(
        "COPY {table} (time_ms, symbol_id, open, high, low, close, volume, source_event_time_ms) FROM STDIN WITH (FORMAT csv)"
    );

    let sink = client.copy_in(&copy_stmt).await?;
    use futures_util::SinkExt;
    use std::pin::pin;
    
    let mut sink = pin!(sink);
    sink.as_mut().send(buf.clone().freeze()).await?;
    sink.as_mut().flush().await?;
    Ok(())
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info,rdkafka=warn".to_string()))
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .try_init();
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

struct EnvCfg {
    database_url: String,
    kafka_brokers: String,
    kafka_group: String,
    topic_candles_close: String,
    topic_indicators_close: String,
    channel_capacity: usize,
}

impl EnvCfg {
    fn from_env() -> Result<Self> {
        Ok(Self {
            database_url: env_req("DATABASE_URL")?,
            kafka_brokers: env_req("KAFKA_BROKERS")?,
            kafka_group: std::env::var("KAFKA_GROUP").unwrap_or_else(|_| "data_writer".to_string()),
            topic_candles_close: std::env::var("TOPIC_CANDLES_CLOSE").unwrap_or_else(|_| "candles_close".to_string()),
            topic_indicators_close: std::env::var("TOPIC_INDICATORS_CLOSE").unwrap_or_else(|_| "indicators_close".to_string()),
            channel_capacity: env_usize("WRITER_CHANNEL_CAP").unwrap_or(50_000),
        })
    }
}

fn env_req(key: &str) -> Result<String> {
    std::env::var(key).map_err(|_| anyhow::anyhow!("{key} is required"))
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

// --- COPY CSV builder ---
struct CopyRow<'a> {
    out: &'a mut BytesMut,
    first: bool,
}

impl<'a> CopyRow<'a> {
    fn new(out: &'a mut BytesMut) -> Self {
        Self { out, first: true }
    }
    fn begin(&mut self) {
        self.first = true;
    }
    fn sep(&mut self) {
        if self.first {
            self.first = false;
        } else {
            self.out.extend_from_slice(b",");
        }
    }
    fn end(&mut self) {
        self.out.extend_from_slice(b"\n");
    }
    fn null(&mut self) {
        self.sep();
        // пустое поле CSV = NULL
    }
    fn i64(&mut self, v: i64) {
        self.sep();
        self.out.extend_from_slice(v.to_string().as_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.sep();
        self.out.extend_from_slice(ryu::Buffer::new().format(v).as_bytes());
    }
    fn str(&mut self, v: &str) {
        self.sep();
        // простая CSV экранировка (если у тебя могут быть запятые/кавычки)
        if v.contains([',', '"', '\n', '\r']) {
            self.out.extend_from_slice(b"\"");
            let escaped = v.replace('"', "\"\"");
            self.out.extend_from_slice(escaped.as_bytes());
            self.out.extend_from_slice(b"\"");
        } else {
            self.out.extend_from_slice(v.as_bytes());
        }
    }
    fn opt_f32(&mut self, v: Option<f32>) {
        match v {
            Some(x) => {
                self.sep();
                self.out.extend_from_slice(ryu::Buffer::new().format_finite(x as f64).as_bytes());
            }
            None => self.null(),
        }
    }
    fn opt_json(&mut self, v: Option<&serde_json::Value>) -> Result<()> {
        match v {
            Some(j) => {
                self.sep();
                let s = serde_json::to_string(j)?;
                self.str(&s);
            }
            None => self.null(),
        }
        Ok(())
    }
}
