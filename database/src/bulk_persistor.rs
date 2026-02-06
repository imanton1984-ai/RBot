use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::{PgPool, Row};
use sqlx::types::Json;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub struct BulkPersistorConfig {
    pub database_url: String,
    pub flush_interval_ms: u64,
    pub max_batch: usize,
    pub chunk_size: usize,
}

impl BulkPersistorConfig {
    pub fn from_env() -> Result<Self> {
        let database_url = std::env::var("DATABASE_URL")
            .map_err(|_| anyhow!("DATABASE_URL is not set"))?;
        let flush_interval_ms = std::env::var("DB_PERSIST_FLUSH_MS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(50);
        let max_batch = std::env::var("DB_PERSIST_MAX_BATCH").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(10_000);
        let chunk_size = std::env::var("DB_PERSIST_CHUNK_SIZE").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(2_000);

        Ok(Self { database_url, flush_interval_ms, max_batch, chunk_size })
    }
}

#[derive(Debug, Clone)]
pub enum PersistRecord {
    Indicator {
        symbol: common::Symbol,
        timeframe: i16,        // tf_minutes
        time_ms: i64,

        indicator_name: String,
        value_float: Option<f64>,
        value_json: Option<Value>,

        candle_is_final: bool,
        calc_source: i16,   // indicators DDL: TEXT
        event_time_ms: Option<i64>,
    },

    RawSignal {
        symbol: common::Symbol,
        timeframe: i16, // tf_minutes
        time_ms: i64,

        indicator_id: i16,
        signal_kind: i16,
        signal_sub_id: i16,

        side: i16,
        score: f32,
        value: f32,

        details: Option<Value>,
        candle_is_final: bool,
        calc_source: i16,      // raw_signals DDL: SMALLINT
        event_time_ms: Option<i64>,

        features_json: Option<Value>,
        scores_json: Option<Value>,
        predictions_json: Option<Value>,
    },
}

pub struct BulkPersistor {
    tx: mpsc::Sender<PersistRecord>,
}

impl BulkPersistor {
    pub async fn new_from_env() -> Result<Self> {
        Self::new(BulkPersistorConfig::from_env()?).await
    }

    pub async fn new(cfg: BulkPersistorConfig) -> Result<Self> {
        let pool = PgPool::connect(&cfg.database_url).await?;
        let (tx, mut rx) = mpsc::channel::<PersistRecord>(cfg.max_batch * 2);

        // persistence loop
        let mut tick = tokio::time::interval(Duration::from_millis(cfg.flush_interval_ms));
        let pool2 = pool.clone();
        let cfg2 = cfg.clone();
        tokio::spawn(async move {
            let mut batch: Vec<PersistRecord> = Vec::with_capacity(cfg2.max_batch);
            let cache = Arc::new(tokio::sync::RwLock::new(HashMap::new()));
            // Pre-fill the cache
            {
                let rows = sqlx::query("SELECT symbol_id, symbol FROM market.pairs")
                    .fetch_all(&pool2)
                    .await;
                if let Ok(rows) = rows {
                    let mut cache_write = cache.write().await;
                    for r in rows {
                        if let (Ok(id), Ok(sym)) = (r.try_get("symbol_id"), r.try_get("symbol")) {
                             let id: i64 = id;
                             let sym: String = sym;
                            cache_write.insert(sym, id);
                        }
                    }
                }
            }


            loop {
                tokio::select! {
                    _ = tick.tick() => {
                        if !batch.is_empty() {
                            if let Err(e) = flush_all(&pool2, &cache, &cfg2, &mut batch).await {
                                tracing::error!("BulkPersistor flush error: {}", e);
                            }
                        }
                    }
                    rec = rx.recv() => {
                        match rec {
                            Some(r) => {
                                batch.push(r);
                                if batch.len() >= cfg2.max_batch {
                                    if let Err(e) = flush_all(&pool2, &cache, &cfg2, &mut batch).await {
                                        tracing::error!("BulkPersistor flush error: {}", e);
                                    }
                                }
                            }
                            None => {
                                if !batch.is_empty() {
                                    let _ = flush_all(&pool2, &cache, &cfg2, &mut batch).await;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        });

        Ok(Self {
            tx,
        })
    }

    pub fn sender(&self) -> mpsc::Sender<PersistRecord> {
        self.tx.clone()
    }

    pub async fn enqueue(&self, record: PersistRecord) -> Result<()> {
        self.tx.send(record).await.map_err(|e| anyhow!("persist queue closed: {}", e))
    }


}

async fn get_symbol_id(
    pool: &PgPool,
    cache: &tokio::sync::RwLock<HashMap<String, i64>>,
    symbol: &str,
) -> Result<i64> {
    if let Some(id) = cache.read().await.get(symbol).copied() {
        return Ok(id);
    }

    let row = sqlx::query("SELECT symbol_id FROM market.pairs WHERE symbol = $1")
        .bind(symbol)
        .fetch_optional(pool)
        .await?;

    let id = match row {
        Some(r) => r.try_get::<i64, _>("symbol_id")?,
        None => return Err(anyhow!("symbol not found in market.pairs: {}", symbol)),
    };

    cache.write().await.insert(symbol.to_string(), id);
    Ok(id)
}

fn tf_to_suffix(tf_minutes: i16) -> &'static str {
    match tf_minutes {
        1 => "1m",
        5 => "5m",
        15 => "15m",
        30 => "30m",
        60 => "1h",
        240 => "4h",
        1440 => "1d",
        _ => "1m",
    }
}

async fn flush_all(
    pool: &PgPool,
    cache: &tokio::sync::RwLock<HashMap<String, i64>>,
    cfg: &BulkPersistorConfig,
    batch: &mut Vec<PersistRecord>,
) -> Result<()> {
    // Разделяем по типам
    let mut indicators: Vec<PersistRecord> = Vec::new();
    let mut signals: Vec<PersistRecord> = Vec::new();

    for r in batch.drain(..) {
        match r {
            PersistRecord::Indicator { .. } => indicators.push(r),
            PersistRecord::RawSignal { .. } => signals.push(r),
        }
    }

    if !indicators.is_empty() {
        flush_indicators(pool, cache, cfg, &indicators).await?;
    }
    if !signals.is_empty() {
        flush_raw_signals(pool, cache, cfg, &signals).await?;
    }
    Ok(())
}

async fn flush_indicators(
    pool: &PgPool,
    cache: &tokio::sync::RwLock<HashMap<String, i64>>,
    cfg: &BulkPersistorConfig,
    items: &[PersistRecord],
) -> Result<()> {
    // Дедуп внутри батча
    // key: (symbol_id, tf, time_ms, indicator_name)
    let mut dedup: HashMap<(i64, i16, i64, String), (PersistRecord, i64)> = HashMap::new();

    for it in items {
        if let PersistRecord::Indicator { symbol, timeframe, time_ms, indicator_name, .. } = it {
            let sym_id = get_symbol_id(pool, cache, symbol.as_str()).await?;
            let k = (sym_id, *timeframe, *time_ms, indicator_name.clone());
            dedup.insert(k, (it.clone(), sym_id));
        }
    }

    // Группируем по tf -> table suffix
    let mut by_tf: HashMap<i16, Vec<(PersistRecord, i64)>> = HashMap::new();
    for (_k, (rec, sym_id)) in dedup {
        if let PersistRecord::Indicator { timeframe, .. } = &rec {
            by_tf.entry(*timeframe).or_default().push((rec, sym_id));
        }
    }

    for (tf, mut vec) in by_tf {
        let suffix = tf_to_suffix(tf);
        while !vec.is_empty() {
            let take = vec.len().min(cfg.chunk_size);
            let chunk: Vec<(PersistRecord, i64)> = vec.drain(0..take).collect();
            flush_indicators_chunk(pool, tf, suffix, chunk).await?;
        }
    }

    Ok(())
}

async fn flush_indicators_chunk(
    pool: &PgPool,
    tf: i16,
    suffix: &str,
    chunk: Vec<(PersistRecord, i64)>,
) -> Result<()> {
    if chunk.is_empty() {
        return Ok(());
    }
    tracing::info!("Persistor: Flushing {} indicators...", chunk.len());
    let table = format!("market.indicators_{}", suffix);

    let now: DateTime<Utc> = Utc::now();

    let mut time: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());
    let mut time_ms: Vec<i64> = Vec::with_capacity(chunk.len());
    let mut symbol_id: Vec<i64> = Vec::with_capacity(chunk.len());
    let mut symbol: Vec<String> = Vec::with_capacity(chunk.len());
    let mut tf_minutes: Vec<i16> = Vec::with_capacity(chunk.len());
    let mut indicator_name: Vec<String> = Vec::with_capacity(chunk.len());
    let mut value_float: Vec<Option<f64>> = Vec::with_capacity(chunk.len());
    let mut value_json: Vec<Option<Json<Value>>> = Vec::with_capacity(chunk.len());
    let mut candle_is_final: Vec<bool> = Vec::with_capacity(chunk.len());
    let mut calc_source: Vec<i16> = Vec::with_capacity(chunk.len());
    let mut event_time_ms: Vec<Option<i64>> = Vec::with_capacity(chunk.len());
    let mut created_at: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());
    let mut updated_at: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());

    for (rec, sym_id) in chunk {
        if let PersistRecord::Indicator {
            symbol: sym,
            timeframe: _,
            time_ms: tms,
            indicator_name: name,
            value_float: vf,
            value_json: vj,
            candle_is_final: final_flag,
            calc_source: cs,
            event_time_ms: etm,
        } = rec {
            time.push(ms_to_ts(tms));
            time_ms.push(tms);
            symbol_id.push(sym_id);
            symbol.push(sym.0);
            tf_minutes.push(tf);
            indicator_name.push(name);
            value_float.push(vf);
            value_json.push(vj.map(Json));
            candle_is_final.push(final_flag);
            calc_source.push(cs);
            event_time_ms.push(etm);
            created_at.push(now);
            updated_at.push(now);
        }
    }

    let sql = format!(
        r#"
        INSERT INTO {table}
        (time, time_ms, symbol_id, symbol, tf_minutes, indicator_name, value_float, value_json,
         candle_is_final, calc_source, event_time_ms, created_at, updated_at)
        SELECT * FROM UNNEST(
            $1::timestamptz[],
            $2::bigint[],
            $3::bigint[],
            $4::text[],
            $5::smallint[],
            $6::text[],
            $7::double precision[],
            $8::jsonb[],
            $9::boolean[],
            $10::smallint[],
            $11::bigint[],
            $12::timestamptz[],
            $13::timestamptz[]
        )
        ON CONFLICT (symbol_id, time, indicator_name)
        DO UPDATE SET
            value_float = EXCLUDED.value_float,
            value_json = EXCLUDED.value_json,
            candle_is_final = EXCLUDED.candle_is_final,
            calc_source = EXCLUDED.calc_source,
            event_time_ms = EXCLUDED.event_time_ms,
            updated_at = now()
        "#
    );

    if let Err(e) = sqlx::query(&sql)
        .bind(time)
        .bind(time_ms)
        .bind(symbol_id)
        .bind(symbol)
        .bind(tf_minutes)
        .bind(indicator_name)
        .bind(value_float)
        .bind(value_json)
        .bind(candle_is_final)
        .bind(calc_source)
        .bind(event_time_ms)
        .bind(created_at)
        .bind(updated_at)
        .execute(pool)
        .await {
            tracing::error!("CRITICAL: Failed to insert indicators: {:?}", e);
            return Err(e.into());
        }

    Ok(())
}

async fn flush_raw_signals(
    pool: &PgPool,
    cache: &tokio::sync::RwLock<HashMap<String, i64>>,
    cfg: &BulkPersistorConfig,
    items: &[PersistRecord],
) -> Result<()> {
    // dedup: (symbol_id, tf, time_ms, indicator_id, kind, sub)
    let mut dedup: HashMap<(i64, i16, i64, i16, i16, i16), (PersistRecord, i64)> = HashMap::new();

    for it in items {
        if let PersistRecord::RawSignal { symbol, timeframe, time_ms, indicator_id, signal_kind, signal_sub_id, .. } = it {
            let sym_id = get_symbol_id(pool, cache, symbol.as_str()).await?;
            let k = (sym_id, *timeframe, *time_ms, *indicator_id, *signal_kind, *signal_sub_id);
            dedup.insert(k, (it.clone(), sym_id));
        }
    }

    let mut vec: Vec<(PersistRecord, i64)> = dedup.into_values().collect();

    while !vec.is_empty() {
        let take = vec.len().min(cfg.chunk_size);
        let chunk: Vec<(PersistRecord, i64)> = vec.drain(0..take).collect();
        flush_raw_signals_chunk(pool, chunk).await?;
    }

    Ok(())
}

async fn flush_raw_signals_chunk(pool: &PgPool, chunk: Vec<(PersistRecord, i64)>) -> Result<()> {
    if chunk.is_empty() {
        return Ok(());
    }
    tracing::info!("Persistor: Flushing {} signals...", chunk.len());
    let now: DateTime<Utc> = Utc::now();

    let mut time: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());
    let mut time_ms: Vec<i64> = Vec::with_capacity(chunk.len());
    let mut symbol_id: Vec<i64> = Vec::with_capacity(chunk.len());
    let mut symbol: Vec<String> = Vec::with_capacity(chunk.len());
    let mut tf_minutes: Vec<i16> = Vec::with_capacity(chunk.len());

    let mut indicator_id: Vec<i16> = Vec::with_capacity(chunk.len());
    let mut signal_kind: Vec<i16> = Vec::with_capacity(chunk.len());
    let mut signal_sub_id: Vec<i16> = Vec::with_capacity(chunk.len());

    let mut side: Vec<i16> = Vec::with_capacity(chunk.len());
    let mut score: Vec<f32> = Vec::with_capacity(chunk.len());
    let mut value: Vec<f32> = Vec::with_capacity(chunk.len());

    let mut details: Vec<Option<Json<Value>>> = Vec::with_capacity(chunk.len());
    let mut candle_is_final: Vec<bool> = Vec::with_capacity(chunk.len());
    let mut calc_source: Vec<i16> = Vec::with_capacity(chunk.len());
    let mut event_time_ms: Vec<Option<i64>> = Vec::with_capacity(chunk.len());

    let mut features_json: Vec<Option<Json<Value>>> = Vec::with_capacity(chunk.len());
    let mut scores_json: Vec<Option<Json<Value>>> = Vec::with_capacity(chunk.len());
    let mut predictions_json: Vec<Option<Json<Value>>> = Vec::with_capacity(chunk.len());

    let mut created_at: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());
    let mut updated_at: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());

    for (rec, sym_id) in chunk {
        if let PersistRecord::RawSignal {
            symbol: sym,
            timeframe: tf,
            time_ms: tms,
            indicator_id: iid,
            signal_kind: sk,
            signal_sub_id: ssid,
            side: sd,
            score: sc,
            value: val,
            details: d,
            candle_is_final: fin,
            calc_source: cs,
            event_time_ms: etm,
            features_json: fj,
            scores_json: sj,
            predictions_json: pj,
        } = rec {
            time.push(ms_to_ts(tms));
            time_ms.push(tms);
            symbol_id.push(sym_id);
            symbol.push(sym.0);
            tf_minutes.push(tf);

            indicator_id.push(iid);
            signal_kind.push(sk);
            signal_sub_id.push(ssid);

            side.push(sd);
            score.push(sc);
            value.push(val);

            details.push(d.map(Json));
            candle_is_final.push(fin);
            calc_source.push(cs);
            event_time_ms.push(etm);

            features_json.push(fj.map(Json));
            scores_json.push(sj.map(Json));
            predictions_json.push(pj.map(Json));

            created_at.push(now);
            updated_at.push(now);
        }
    }

    let sql = r#"
        INSERT INTO market.raw_signals
        (time, time_ms, symbol_id, symbol, tf_minutes,
         indicator_id, signal_kind, signal_sub_id,
         side, score, value,
         details, candle_is_final, calc_source, event_time_ms,
         features_json, scores_json, predictions_json,
         created_at, updated_at)
        SELECT * FROM UNNEST(
            $1::timestamptz[],
            $2::bigint[],
            $3::bigint[],
            $4::text[],
            $5::smallint[],
            $6::smallint[],
            $7::smallint[],
            $8::smallint[],
            $9::smallint[],
            $10::real[],
            $11::real[],
            $12::jsonb[],
            $13::boolean[],
            $14::smallint[],
            $15::bigint[],
            $16::jsonb[],
            $17::jsonb[],
            $18::jsonb[],
            $19::timestamptz[],
            $20::timestamptz[]
        )
        ON CONFLICT (symbol_id, tf_minutes, time, indicator_id, signal_kind, signal_sub_id) DO UPDATE SET
            side = EXCLUDED.side,
            score = EXCLUDED.score,
            value = EXCLUDED.value,
            details = EXCLUDED.details,
            candle_is_final = EXCLUDED.candle_is_final,
            calc_source = EXCLUDED.calc_source,
            event_time_ms = EXCLUDED.event_time_ms,
            features_json = EXCLUDED.features_json,
            scores_json = EXCLUDED.scores_json,
            predictions_json = EXCLUDED.predictions_json,
            updated_at = now()
    "#;

    if let Err(e) = sqlx::query(sql)
        .bind(time)
        .bind(time_ms)
        .bind(symbol_id)
        .bind(symbol)
        .bind(tf_minutes)
        .bind(indicator_id)
        .bind(signal_kind)
        .bind(signal_sub_id)
        .bind(side)
        .bind(score)
        .bind(value)
        .bind(details)
        .bind(candle_is_final)
        .bind(calc_source)
        .bind(event_time_ms)
        .bind(features_json)
        .bind(scores_json)
        .bind(predictions_json)
        .bind(created_at)
        .bind(updated_at)
        .execute(pool)
        .await {
            tracing::error!("CRITICAL: Failed to insert signals: {:?}", e);
            return Err(e.into());
        }

    Ok(())
}

fn ms_to_ts(ms: i64) -> DateTime<Utc> {
    let secs = ms / 1000;
    let nsec = ((ms % 1000).max(0) as u32) * 1_000_000;
    DateTime::<Utc>::from_timestamp(secs, nsec).unwrap_or_else(|| Utc::now())
}
