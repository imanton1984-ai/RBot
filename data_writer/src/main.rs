use anyhow::{Context, Result};
use bytes::BytesMut;
use dashmap::DashMap;
use rdkafka::consumer::{StreamConsumer};
use rdkafka::message::Message;
use rdkafka::ClientConfig;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{Duration, Instant};
use futures::SinkExt;

use common::config::load_config;
use common::timeframe::Timeframe;

mod messages;
mod copy_row;
use copy_row::CopyRow;
use messages::{CandleCloseMsg, IndicatorsSnapshotMsg};

const CH_CAP: usize = 50_000;
const BATCH_SIZE: usize = 10_000;
const BATCH_MAX_WAIT_MS: u64 = 200;

#[derive(Clone)]
struct SymbolCache {
    map: Arc<DashMap<String, i64>>,
}
impl SymbolCache {
    fn new() -> Self { 
        Self { map: Arc::new(DashMap::new()) } 
    }
    
    fn get(&self, symbol: &str) -> Option<i64> { 
        self.map.get(symbol).map(|v| *v) 
    }
    
    // Меняем id на symbol_id
    fn insert(&self, symbol: String, symbol_id: i64) { 
        self.map.insert(symbol, symbol_id); 
    }
}

#[derive(Debug)]
enum WriteMsg {
    // Используем подчеркивание, чтобы скрыть варнинг о неиспользуемом поле, 
    // но сохраняем его в структуре
    Candle { symbol_id: i64, _tf: Timeframe, evt: CandleCloseMsg },
    Indicators { symbol_id: i64, _tf: Timeframe, evt: IndicatorsSnapshotMsg },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = load_config().context("load_config failed")?;
    let db_url = cfg.database.url();
    println!("DATA_WRITER connecting to: {}", db_url); // для отладки

    // DB connect (single client used for preload; workers will open their own connections)
    let (client, connection) = tokio_postgres::connect(&db_url, tokio_postgres::NoTls)
        .await
        .context("postgres connect failed")?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("postgres connection error: {e}");
        }
    });

    let cache = SymbolCache::new();
    for (sym, id) in preload_symbols(&client).await? {
        cache.insert(sym, id);
    }

    // 6 TF channels
    let (tx_1m, rx_1m) = mpsc::channel::<WriteMsg>(CH_CAP);
    let (tx_5m, rx_5m) = mpsc::channel::<WriteMsg>(CH_CAP);
    let (tx_15m, rx_15m) = mpsc::channel::<WriteMsg>(CH_CAP);
    let (tx_1h, rx_1h) = mpsc::channel::<WriteMsg>(CH_CAP);
    let (tx_4h, rx_4h) = mpsc::channel::<WriteMsg>(CH_CAP);
    let (tx_1d, rx_1d) = mpsc::channel::<WriteMsg>(CH_CAP);

    // Spawn 6 workers (each with its own DB connection + COPY+merge)
    spawn_tf_worker("1m", Timeframe::M1, db_url.clone(), rx_1m);
    spawn_tf_worker("5m", Timeframe::M5, db_url.clone(), rx_5m);
    spawn_tf_worker("15m", Timeframe::M15, db_url.clone(), rx_15m);
    spawn_tf_worker("1h", Timeframe::H1, db_url.clone(), rx_1h);
    spawn_tf_worker("4h", Timeframe::H4, db_url.clone(), rx_4h);
    spawn_tf_worker("1d", Timeframe::D1, db_url.clone(), rx_1d);

    // Kafka consumer
    let consumer: StreamConsumer = ClientConfig::new()
        .set("bootstrap.servers", cfg.rust_bot.redpanda_brokers.join(","))
        .set("group.id", format!("data_writer-{}", cfg.rust_bot.instance_id))
        .set("enable.auto.commit", "false")
        .set("auto.offset.reset", "latest")
        .create()
        .context("create consumer failed")?;

    consumer
        .subscribe(&[
            &cfg.rust_bot.topic_candles_close,
            &cfg.rust_bot.topic_indicators_close,
        ])
        .context("subscribe failed")?;

    loop {
        let m = consumer.recv().await?;
        let topic = m.topic();

        let payload = match m.payload() {
            Some(p) => p,
            None => {
                consumer.commit_message(&m, rdkafka::consumer::CommitMode::Async).ok();
                continue;
            }
        };

        if topic == cfg.rust_bot.topic_candles_close {
            if let Ok(evt) = serde_json::from_slice::<CandleCloseMsg>(payload) {
                if let (Some(symbol_id), Ok(tf)) = (cache.get(&evt.symbol), Timeframe::parse(&evt.tf)) {
                    // Заменяем tf на _tf
                    let msg = WriteMsg::Candle { symbol_id, _tf: tf.clone(), evt };
                    route_tf(&tf, &tx_1m, &tx_5m, &tx_15m, &tx_1h, &tx_4h, &tx_1d, msg).await?;
                }
            }
        } else if topic == cfg.rust_bot.topic_indicators_close {
            if let Ok(evt) = serde_json::from_slice::<IndicatorsSnapshotMsg>(payload) {
                if let (Some(symbol_id), Ok(tf)) = (cache.get(&evt.symbol), Timeframe::parse(&evt.tf)) {
                    // Заменяем tf на _tf
                    let msg = WriteMsg::Indicators { symbol_id, _tf: tf.clone(), evt };
                    route_tf(&tf, &tx_1m, &tx_5m, &tx_15m, &tx_1h, &tx_4h, &tx_1d, msg).await?;
                }
            }
        }

        // ✅ v2/simple: commit after enqueue (bounded channel = backpressure)
        consumer.commit_message(&m, rdkafka::consumer::CommitMode::Async).ok();
    }
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
            eprintln!("worker {name} crashed: {e:?}");
        }
    });
}

async fn tf_worker_loop(
    _name: &'static str,
    tf: Timeframe,
    db_url: &str,
    rx: &mut mpsc::Receiver<WriteMsg>,
) -> Result<()> {
    let (client, connection) = tokio_postgres::connect(db_url, tokio_postgres::NoTls).await?;
    tokio::spawn(async move { let _ = connection.await; });

    let candles_table = tf.candles_table();
    let indicators_table = tf.indicators_table();

    // per-TF buffers
    let mut candles_buf = BytesMut::with_capacity(1024 * 512);
    let mut ind_buf = BytesMut::with_capacity(1024 * 512);
    let mut candles_n = 0usize;
    let mut ind_n = 0usize;

    let mut last_flush = Instant::now();

    loop {
        let timeout = Duration::from_millis(BATCH_MAX_WAIT_MS);
        let msg = tokio::time::timeout(timeout, rx.recv()).await;

        if let Ok(Some(m)) = msg {
            match m {
                // ИСПРАВЛЕННЫЕ match pattern (используем _tf вместо tf)
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
                    match evt.source_event_time_ms { Some(x) => row.i64(x), None => row.null() }
                    row.end();
                    candles_n += 1;
                }
                WriteMsg::Indicators { symbol_id, _tf: _, evt } => {
                    let mut row = CopyRow::new(&mut ind_buf);
                    row.begin();
                    row.i64(evt.close_time_ms);
                    row.i64(symbol_id);

                    row.opt_f32(evt.ema20); row.opt_f32(evt.ema50); row.opt_f32(evt.ema200);
                    row.opt_f32(evt.sma);

                    row.opt_f32(evt.rsi);
                    row.opt_f32(evt.macd); row.opt_f32(evt.macd_signal); row.opt_f32(evt.macd_hist);

                    row.opt_f32(evt.atr); row.opt_f32(evt.adx);

                    row.opt_f32(evt.bb_upper); row.opt_f32(evt.bb_mid); row.opt_f32(evt.bb_lower);
                    row.opt_f32(evt.stoch_k); row.opt_f32(evt.stoch_d);

                    row.opt_f32(evt.vwap); row.opt_f32(evt.obv); row.opt_f32(evt.cci); row.opt_f32(evt.williams);

                    row.opt_f32(evt.alli_jaw); row.opt_f32(evt.alli_teeth); row.opt_f32(evt.alli_lips);

                    row.opt_json(&evt.sr_levels).map_err(|e| anyhow::anyhow!(e))?;

                    row.str(evt.features_version.as_deref().unwrap_or("v1"));
                    row.i64(current_time_ms());
                    row.end();
                    ind_n += 1;
                }
            }
        }

        let need_flush = candles_n >= BATCH_SIZE
            || ind_n >= BATCH_SIZE
            || last_flush.elapsed() >= Duration::from_millis(BATCH_MAX_WAIT_MS);

        if need_flush && (candles_n > 0 || ind_n > 0) {
            // flush both buffers (each into staging then merge into real table)
            if candles_n > 0 {
                flush_candles(&client, candles_table, &mut candles_buf).await?;
                candles_n = 0;
            }
            if ind_n > 0 {
                flush_indicators(&client, indicators_table, &mut ind_buf).await?;
                ind_n = 0;
            }
            last_flush = Instant::now();
            // optional log:
            // eprintln!("worker {name}: flushed");
        }
    }
}

async fn flush_candles(client: &tokio_postgres::Client, dest_table: &str, buf: &mut BytesMut) -> Result<()> {
    // 1) COPY into staging
    let copy_sql = "COPY market.candles_staging (time_ms, symbol_id, open, high, low, close, volume, source_event_time_ms) FROM STDIN WITH (FORMAT text)";
    
    let sink = client.copy_in(copy_sql).await?; 
    let data = buf.split().freeze();
    
    // Используем пиннинг и SinkExt для отправки данных
    tokio::pin!(sink);
    sink.send(data).await.context("failed to send data to postgres copy")?;
    sink.close().await.context("failed to close postgres copy sink")?;

    // 2) merge with dedup
    let merge_sql = format!(
        "INSERT INTO {dest_table} (time_ms, symbol_id, open, high, low, close, volume, source_event_time_ms)
         SELECT time_ms, symbol_id, open, high, low, close, volume, source_event_time_ms
         FROM market.candles_staging
         ON CONFLICT (symbol_id, time_ms) DO NOTHING;
         DELETE FROM market.candles_staging;"
    );
    client.batch_execute(&merge_sql).await?;
    Ok(())
}

async fn flush_indicators(client: &tokio_postgres::Client, dest_table: &str, buf: &mut BytesMut) -> Result<()> {
    let copy_sql = "COPY market.indicators_staging (time_ms, symbol_id, ema20, ema50, ema200, sma, rsi, macd, macd_signal, macd_hist, atr, adx, bb_upper, bb_mid, bb_lower, stoch_k, stoch_d, vwap, obv, cci, williams, alli_jaw, alli_teeth, alli_lips, sr_levels, features_version, updated_at_ms) FROM STDIN WITH (FORMAT text)";
    
    let sink = client.copy_in(copy_sql).await?;
    let data = buf.split().freeze();
    
    tokio::pin!(sink);
    sink.send(data).await.context("failed to send data to postgres copy")?;
    sink.close().await.context("failed to close postgres copy sink")?;

    let merge_sql = format!(
        "INSERT INTO {dest_table} (time_ms, symbol_id, ema20, ema50, ema200, sma, rsi, macd, macd_signal, macd_hist, atr, adx, bb_upper, bb_mid, bb_lower, stoch_k, stoch_d, vwap, obv, cci, williams, alli_jaw, alli_teeth, alli_lips, sr_levels, features_version, updated_at_ms)
         SELECT time_ms, symbol_id, ema20, ema50, ema200, sma, rsi, macd, macd_signal, macd_hist, atr, adx, bb_upper, bb_mid, bb_lower, stoch_k, stoch_d, vwap, obv, cci, williams, alli_jaw, alli_teeth, alli_lips, sr_levels, features_version, updated_at_ms
         FROM market.indicators_staging
         ON CONFLICT (symbol_id, time_ms) DO NOTHING;
         DELETE FROM market.indicators_staging;"
    );
    client.batch_execute(&merge_sql).await?;
    Ok(())
}

async fn preload_symbols(client: &tokio_postgres::Client) -> Result<Vec<(String, i64)>> {
    let rows = client.query("SELECT symbol, symbol_id FROM market.pairs WHERE is_active = true", &[]).await?;
    Ok(rows.into_iter().map(|r| (r.get(0), r.get(1))).collect())
}

fn current_time_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    (d.as_secs() as i64) * 1000 + (d.subsec_millis() as i64)
}
