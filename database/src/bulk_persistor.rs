// database/src/bulk_persistor.rs

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::{types::Json, PgPool, Row};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::mpsc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PersistMode {
    History,
    Realtime,
}

impl PersistMode {
    pub fn from_env_var(v: Option<String>) -> Self {
        match v.as_deref() {
            Some("history") | Some("HISTORY") => PersistMode::History,
            Some("realtime") | Some("REALTIME") => PersistMode::Realtime,
            _ => PersistMode::Realtime,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BulkPersistorConfig {
    pub database_url: String,

    // общий
    pub flush_interval_ms: u64,
    pub max_batch: usize,

    // per-mode
    pub mode: PersistMode,
    pub chunk_size: usize,

    // history knobs
    pub history_upsert: bool,      // если 1 — будет DO UPDATE (медленнее, но “перезапись” возможна)
    pub history_skip_json: bool,   // если 1 — не пишем jsonb поля для истории (ускоряет сильно)
}

impl BulkPersistorConfig {
    pub fn from_env() -> Result<Self> {
        let database_url =
            std::env::var("DATABASE_URL").map_err(|_| anyhow!("DATABASE_URL is not set"))?;

        let flush_interval_ms = std::env::var("DB_PERSIST_FLUSH_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50);

        // Increased default from 20_000 to 50_000 to reduce flush frequency for history
        let max_batch = std::env::var("DB_PERSIST_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50_000);

        let mode = PersistMode::from_env_var(std::env::var("DB_PERSIST_MODE").ok());

        // chunk size зависит от режима
        let chunk_size = match mode {
            PersistMode::History => std::env::var("DB_PERSIST_CHUNK_SIZE_HISTORY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50_000),
            PersistMode::Realtime => std::env::var("DB_PERSIST_CHUNK_SIZE_REALTIME")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2_000),
        };

        let history_upsert = std::env::var("DB_PERSIST_HISTORY_UPSERT")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let history_skip_json = std::env::var("DB_PERSIST_HISTORY_SKIP_JSON")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        Ok(Self {
            database_url,
            flush_interval_ms,
            max_batch,
            mode,
            chunk_size,
            history_upsert,
            history_skip_json,
        })
    }

    /// Удобный конструктор для двух стримов: history/realtime из одних env,
    /// но с разными параметрами.
    pub fn from_env_with_mode(mode: PersistMode) -> Result<Self> {
        let mut cfg = Self::from_env()?;
        cfg.mode = mode;

        cfg.chunk_size = match mode {
            PersistMode::History => std::env::var("DB_PERSIST_CHUNK_SIZE_HISTORY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50_000),
            PersistMode::Realtime => std::env::var("DB_PERSIST_CHUNK_SIZE_REALTIME")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2_000),
        };

        Ok(cfg)
    }
}

#[derive(Debug, Clone)]
pub enum PersistRecord {
    Indicator {
        symbol: common::Symbol,
        timeframe: i16,
        time_ms: i64,

        indicator_name: String,
        value_float: Option<f64>,
        value_json: Option<Value>,

        candle_is_final: bool,
        calc_source: i16,
        event_time_ms: Option<i64>,
    },

    RawSignal {
        symbol: common::Symbol,
        timeframe: i16,
        time_ms: i64,

        indicator_id: i16,
        signal_kind: i16,
        signal_sub_id: i16,

        side: i16,
        score: f32,
        value: f32,

        details: Option<Value>,
        candle_is_final: bool,
        calc_source: i16,
        event_time_ms: Option<i64>,

        features_json: Option<Value>,
        scores_json: Option<Value>,
        predictors_json: Option<Value>,
    },

    IndicatorsWide {
        symbol: common::Symbol,
        timeframe: i16,
        time_ms: i64,
        indicators: std::collections::HashMap<String, f64>,
        json_data: std::collections::HashMap<String, serde_json::Value>,
        candle_is_final: bool,
        calc_source: i16,
        event_time_ms: Option<i64>,
    },

    AggregatedSignal {
        symbol: common::Symbol,
        timeframe: i16,
        time_ms: i64,
        signals: Vec<AggregatedSignalItem>,
        features_json: Option<Value>,
        scores_json: Option<Value>,
        predictors_json: Option<Value>,
        candle_is_final: bool,
        calc_source: i16,
        event_time_ms: Option<i64>,
    },

    Predictor {
        symbol: common::Symbol,
        timeframe: i16,
        time_ms: i64,

        horizon_bars: i32,
        aspect: i16,
        calc_source: i16,
        predictor_id: i64,

        score_norm: f32,
        value: f64,
        value_low: Option<f64>,
        value_high: Option<f64>,
        side: Option<i16>,

        level_hash: Option<String>,
        level_kind: Option<i16>,
        level_price: Option<f64>,
        level_strength: Option<f32>,
        level_distance_atr: Option<f32>,

        candle_is_final: bool,
        event_time_ms: Option<i64>,
        details_json: Option<Value>,

        prediction_key: String,
    },

    TradeSignal {
        symbol: common::Symbol,
        timeframe: i16,
        time_ms: i64,
        side: i16,
        final_score: f32,
        ml_score: Option<f32>,
        heur_score: Option<f32>,
        entry_price: Option<f32>,
        sl_price: Option<f32>,
        tp1_price: Option<f32>,
        tp2_price: Option<f32>,
        tp3_price: Option<f32>,
        reason: Option<Value>,
        price10_target: Option<f64>,
        price10_score: Option<f32>,
        bounce_prob: Option<f32>,
        bounce_score: Option<f32>,
        breakout_prob: Option<f32>,
        breakout_score: Option<f32>,
    },
}

#[derive(Debug, Clone)]
pub struct AggregatedSignalItem {
    pub indicator_id: i16,
    pub signal_kind: i16,
    pub side: i16,
    pub score: f32,
    pub value: f32,
    pub details: Option<Value>,
}

pub struct BulkPersistor {
    tx: mpsc::Sender<PersistRecord>,
}

impl BulkPersistor {
    pub async fn new_from_env() -> Result<Self> {
        Self::new(BulkPersistorConfig::from_env()?).await
    }

    pub async fn new_from_env_mode(mode: PersistMode) -> Result<Self> {
        Self::new(BulkPersistorConfig::from_env_with_mode(mode)?).await
    }

    pub async fn new(cfg: BulkPersistorConfig) -> Result<Self> {
        let pool = PgPool::connect(&cfg.database_url).await?;
        let (tx, mut rx) = mpsc::channel::<PersistRecord>(cfg.max_batch * 2);

        let mut tick = tokio::time::interval(Duration::from_millis(cfg.flush_interval_ms));
        let pool2 = pool.clone();
        let cfg2 = cfg.clone();

        tokio::spawn(async move {
            let mut batch: Vec<PersistRecord> = Vec::with_capacity(cfg2.max_batch);

            // symbol cache
            let cache = Arc::new(tokio::sync::RwLock::new(HashMap::<String, i64>::new()));
            {
                let rows = sqlx::query("SELECT symbol_id, symbol FROM market.pairs")
                    .fetch_all(&pool2)
                    .await;
                if let Ok(rows) = rows {
                    let mut w = cache.write().await;
                    for r in rows {
                        if let (Ok(id), Ok(sym)) = (r.try_get("symbol_id"), r.try_get("symbol")) {
                            let id: i64 = id;
                            let sym: String = sym;
                            w.insert(sym, id);
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

        Ok(Self { tx })
    }

    pub fn sender(&self) -> mpsc::Sender<PersistRecord> {
        self.tx.clone()
    }

    pub async fn enqueue(&self, record: PersistRecord) -> Result<()> {
        self.tx
            .send(record)
            .await
            .map_err(|e| anyhow!("persist queue closed: {}", e))
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

async fn flush_all(
    pool: &PgPool,
    cache: &tokio::sync::RwLock<HashMap<String, i64>>,
    cfg: &BulkPersistorConfig,
    batch: &mut Vec<PersistRecord>,
) -> Result<()> {
    let mut signals: Vec<PersistRecord> = Vec::new();
    let mut wide_indicators: Vec<PersistRecord> = Vec::new();
    let mut predictors: Vec<PersistRecord> = Vec::new();
    let mut trade_signals: Vec<PersistRecord> = Vec::new();

    for r in batch.drain(..) {
        match r {
            PersistRecord::RawSignal { .. } => signals.push(r),
            PersistRecord::IndicatorsWide { .. } => wide_indicators.push(r),
            PersistRecord::Predictor { .. } => predictors.push(r),
            PersistRecord::TradeSignal { .. } => trade_signals.push(r),
            PersistRecord::Indicator { .. } => { /* ignore */ }
            PersistRecord::AggregatedSignal { .. } => { /* ignore */ }
        }
    }

    let s1 = async {
        if !signals.is_empty() {
            flush_raw_signals(pool, cache, cfg, &signals).await?;
        }
        Ok::<(), anyhow::Error>(())
    };

    let s2 = async {
        if !wide_indicators.is_empty() {
            flush_wide_indicators(pool, cache, cfg, &wide_indicators).await?;
        }
        Ok::<(), anyhow::Error>(())
    };

    let s3 = async {
        if !predictors.is_empty() {
            flush_predictors(pool, cache, cfg, &predictors).await?;
        }
        Ok::<(), anyhow::Error>(())
    };

    let s4 = async {
        if !trade_signals.is_empty() {
            flush_trade_signals(pool, cache, cfg, &trade_signals).await?;
        }
        Ok::<(), anyhow::Error>(())
    };

    let (r1, r2, r3, r4) = tokio::join!(s1, s2, s3, s4);
    r1?; r2?; r3?; r4?;
    Ok(())
}

async fn flush_predictors(
    pool: &PgPool,
    cache: &tokio::sync::RwLock<HashMap<String, i64>>,
    cfg: &BulkPersistorConfig,
    items: &[PersistRecord],
) -> Result<()> {
    // For history mode: skip dedup (ON CONFLICT DO NOTHING handles it at DB level)
    // This avoids expensive clone + HashMap operations on large batches
    let mut vec: Vec<(PersistRecord, i64)> = Vec::with_capacity(items.len());
    
    for it in items {
        if let PersistRecord::Predictor { symbol, .. } = it {
            let sym_id = get_symbol_id(pool, cache, symbol.as_str()).await?;
            vec.push((it.clone(), sym_id));
        }
    }

    let total = vec.len();
    let mut written = 0;
    while !vec.is_empty() {
        let take = vec.len().min(cfg.chunk_size);
        let chunk: Vec<(PersistRecord, i64)> = vec.drain(0..take).collect();
        let chunk_len = chunk.len();
        flush_predictors_chunk(pool, cfg, chunk).await?;
        written += chunk_len;
    }
    
    if total > 0 {
        tracing::debug!("flush_predictors: wrote {} predictor rows (mode={:?})", written, cfg.mode);
    }

    Ok(())
}

async fn flush_predictors_chunk(
    pool: &PgPool,
    cfg: &BulkPersistorConfig,
    chunk: Vec<(PersistRecord, i64)>,
) -> Result<()> {
    if chunk.is_empty() { return Ok(()); }

    let now: DateTime<Utc> = Utc::now();
    let skip_json = cfg.mode == PersistMode::History && cfg.history_skip_json;

    let mut time: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());
    let mut time_ms: Vec<i64> = Vec::with_capacity(chunk.len());
    let mut symbol_id: Vec<i64> = Vec::with_capacity(chunk.len());
    let mut symbol: Vec<String> = Vec::with_capacity(chunk.len());
    let mut tf_minutes: Vec<i32> = Vec::with_capacity(chunk.len());

    let mut horizon_bars: Vec<i32> = Vec::with_capacity(chunk.len());
    let mut aspect: Vec<i16> = Vec::with_capacity(chunk.len());
    let mut calc_source: Vec<i16> = Vec::with_capacity(chunk.len());
    let mut predictor_id: Vec<i64> = Vec::with_capacity(chunk.len());

    let mut score_norm: Vec<f32> = Vec::with_capacity(chunk.len());
    let mut value: Vec<f64> = Vec::with_capacity(chunk.len());
    let mut value_low: Vec<Option<f64>> = Vec::with_capacity(chunk.len());
    let mut value_high: Vec<Option<f64>> = Vec::with_capacity(chunk.len());
    let mut side: Vec<Option<i16>> = Vec::with_capacity(chunk.len());

    let mut level_hash: Vec<Option<String>> = Vec::with_capacity(chunk.len());
    let mut level_kind: Vec<Option<i16>> = Vec::with_capacity(chunk.len());
    let mut level_price: Vec<Option<f64>> = Vec::with_capacity(chunk.len());
    let mut level_strength: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut level_distance_atr: Vec<Option<f32>> = Vec::with_capacity(chunk.len());

    let mut candle_is_final: Vec<bool> = Vec::with_capacity(chunk.len());
    let mut event_time_ms: Vec<Option<i64>> = Vec::with_capacity(chunk.len());
    let mut details_json: Vec<Option<Json<Value>>> = Vec::with_capacity(chunk.len());
    let mut prediction_key: Vec<String> = Vec::with_capacity(chunk.len());

    let mut created_at: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());
    let mut updated_at: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());

    for (rec, sym_id) in chunk {
        if let PersistRecord::Predictor {
            symbol: sym, timeframe: tf, time_ms: tms,
            horizon_bars: hb, aspect: asp, calc_source: cs, predictor_id: pid,
            score_norm: sn, value: val, value_low: vl, value_high: vh, side: sd,
            level_hash: lh, level_kind: lk, level_price: lp, level_strength: ls, level_distance_atr: lda,
            candle_is_final: fin, event_time_ms: etm, details_json: dj, prediction_key: pk,
        } = rec
        {
            time.push(ms_to_ts(tms));
            time_ms.push(tms);
            symbol_id.push(sym_id);
            symbol.push(sym.0);
            tf_minutes.push(tf as i32);

            horizon_bars.push(hb);
            aspect.push(asp);
            calc_source.push(cs);
            predictor_id.push(pid);

            score_norm.push(sn);
            value.push(val);
            value_low.push(vl);
            value_high.push(vh);
            side.push(sd);

            level_hash.push(lh);
            level_kind.push(lk);
            level_price.push(lp);
            level_strength.push(ls);
            level_distance_atr.push(lda);

            candle_is_final.push(fin);
            event_time_ms.push(etm);

            if skip_json {
                details_json.push(None);
            } else {
                details_json.push(dj.map(Json));
            }

            prediction_key.push(pk);

            created_at.push(now);
            updated_at.push(now);
        }
    }

    let sql = match cfg.mode {
        PersistMode::Realtime => predictors_sql_realtime(),
        PersistMode::History => {
            if cfg.history_upsert { predictors_sql_history_upsert() }
            else { predictors_sql_history_append() }
        }
    };

    sqlx::query(sql)
        .bind(time)
        .bind(time_ms)
        .bind(symbol_id)
        .bind(symbol)
        .bind(tf_minutes)
        .bind(horizon_bars)
        .bind(aspect)
        .bind(calc_source)
        .bind(predictor_id)
        .bind(score_norm)
        .bind(value)
        .bind(value_low)
        .bind(value_high)
        .bind(side)
        .bind(level_hash)
        .bind(level_kind)
        .bind(level_price)
        .bind(level_strength)
        .bind(level_distance_atr)
        .bind(candle_is_final)
        .bind(event_time_ms)
        .bind(details_json)
        .bind(prediction_key)
        .bind(created_at)
        .bind(updated_at)
        .execute(pool)
        .await?;

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
        if let PersistRecord::RawSignal {
            symbol,
            timeframe,
            time_ms,
            indicator_id,
            signal_kind,
            signal_sub_id,
            ..
        } = it
        {
            let sym_id = get_symbol_id(pool, cache, symbol.as_str()).await?;
            let k = (sym_id, *timeframe, *time_ms, *indicator_id, *signal_kind, *signal_sub_id);
            dedup.insert(k, (it.clone(), sym_id));
        }
    }

    let mut vec: Vec<(PersistRecord, i64)> = dedup.into_values().collect();

    while !vec.is_empty() {
        let take = vec.len().min(cfg.chunk_size);
        let chunk: Vec<(PersistRecord, i64)> = vec.drain(0..take).collect();
        flush_raw_signals_chunk(pool, cfg, chunk).await?;
    }

    Ok(())
}

async fn flush_raw_signals_chunk(pool: &PgPool, cfg: &BulkPersistorConfig, chunk: Vec<(PersistRecord, i64)>) -> Result<()> {
    if chunk.is_empty() {
        return Ok(());
    }

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

    // json payloads (можно обнулить в history для скорости)
    let mut features_json: Vec<Option<Json<Value>>> = Vec::with_capacity(chunk.len());
    let mut scores_json: Vec<Option<Json<Value>>> = Vec::with_capacity(chunk.len());
    let mut predictors_json: Vec<Option<Json<Value>>> = Vec::with_capacity(chunk.len());

    let mut created_at: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());
    let mut updated_at: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());

    let skip_json = cfg.mode == PersistMode::History && cfg.history_skip_json;

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
            predictors_json: pj,
        } = rec
        {
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

            if skip_json {
                features_json.push(None);
                scores_json.push(None);
                predictors_json.push(None);
            } else {
                features_json.push(fj.map(Json));
                scores_json.push(sj.map(Json));
                predictors_json.push(pj.map(Json));
            }

            created_at.push(now);
            updated_at.push(now);
        }
    }

    let sql = match cfg.mode {
        PersistMode::Realtime => raw_signals_sql_realtime(),
        PersistMode::History => {
            if cfg.history_upsert {
                raw_signals_sql_history_upsert()
            } else {
                raw_signals_sql_history_append()
            }
        }
    };

    sqlx::query(sql)
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
        .bind(predictors_json)
        .bind(created_at)
        .bind(updated_at)
        .execute(pool)
        .await?;

    Ok(())
}

async fn flush_wide_indicators(
    pool: &PgPool,
    cache: &tokio::sync::RwLock<HashMap<String, i64>>,
    cfg: &BulkPersistorConfig,
    items: &[PersistRecord],
) -> Result<()> {
    // Group by (symbol_id, timeframe, time_ms) and merge
    let mut grouped: HashMap<(i64, i16, i64), (String, String, i16, i64, HashMap<String, f64>, HashMap<String, serde_json::Value>, bool, i16, Option<i64>)> =
        HashMap::new();

    for item in items {
        if let PersistRecord::IndicatorsWide {
            symbol,
            timeframe,
            time_ms,
            indicators,
            json_data,
            candle_is_final,
            calc_source,
            event_time_ms,
        } = item
        {
            let sym_id = get_symbol_id(pool, cache, symbol.as_str()).await?;
            let key = (sym_id, *timeframe, *time_ms);

            match grouped.entry(key) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let (_, _, _, _, existing_indicators, existing_json, _, _, _) = entry.get_mut();
                    for (k, v) in indicators {
                        existing_indicators.insert(k.clone(), *v);
                    }
                    for (k, v) in json_data {
                        existing_json.insert(k.clone(), v.clone());
                    }
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let mut new_ind = HashMap::new();
                    let mut new_json = HashMap::new();
                    for (k, v) in indicators {
                        new_ind.insert(k.clone(), *v);
                    }
                    for (k, v) in json_data {
                        new_json.insert(k.clone(), v.clone());
                    }
                    entry.insert((
                        symbol.0.clone(),
                        symbol.0.clone(),
                        *timeframe,
                        *time_ms,
                        new_ind,
                        new_json,
                        *candle_is_final,
                        *calc_source,
                        *event_time_ms,
                    ));
                }
            }
        }
    }

    let mut records: Vec<_> = grouped.into_values().collect();

    while !records.is_empty() {
        let take = records.len().min(cfg.chunk_size);
        let chunk: Vec<_> = records.drain(0..take).collect();
        flush_wide_indicators_chunk(pool, cache, cfg, chunk).await?;
    }

    Ok(())
}

async fn flush_wide_indicators_chunk(
    pool: &PgPool,
    cache: &tokio::sync::RwLock<HashMap<String, i64>>,
    cfg: &BulkPersistorConfig,
    chunk: Vec<(String, String, i16, i64, HashMap<String, f64>, HashMap<String, serde_json::Value>, bool, i16, Option<i64>)>,
) -> Result<()> {
    if chunk.is_empty() {
        return Ok(());
    }

    let now: DateTime<Utc> = Utc::now();
    let skip_json = cfg.mode == PersistMode::History && cfg.history_skip_json;

    let mut time: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());
    let mut time_ms: Vec<i64> = Vec::with_capacity(chunk.len());
    let mut symbol_id: Vec<i64> = Vec::with_capacity(chunk.len());
    let mut symbol: Vec<String> = Vec::with_capacity(chunk.len());
    let mut tf_minutes: Vec<i16> = Vec::with_capacity(chunk.len());

    let mut rsi: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut cci: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut stoch_k: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut stoch_d: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut williams: Vec<Option<f32>> = Vec::with_capacity(chunk.len());

    let mut macd: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut macd_signal: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut macd_hist: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut adx: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut sma: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut ema_20: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut ema_50: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut ema_200: Vec<Option<f32>> = Vec::with_capacity(chunk.len());

    let mut bb_upper: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut bb_mid: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut bb_lower: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut atr: Vec<Option<f32>> = Vec::with_capacity(chunk.len());

    let mut obv: Vec<Option<f64>> = Vec::with_capacity(chunk.len());
    let mut vwap: Vec<Option<f64>> = Vec::with_capacity(chunk.len());
    let mut volume_spike: Vec<Option<f32>> = Vec::with_capacity(chunk.len());

    let mut alligator_jaw: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut alligator_teeth: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut alligator_lips: Vec<Option<f32>> = Vec::with_capacity(chunk.len());

    let mut trend: Vec<Option<i16>> = Vec::with_capacity(chunk.len());
    let mut trend_short: Vec<Option<i16>> = Vec::with_capacity(chunk.len());
    let mut poc: Vec<Option<f32>> = Vec::with_capacity(chunk.len());

    let mut sr_levels: Vec<Option<Json<Value>>> = Vec::with_capacity(chunk.len());

    let mut candle_is_final: Vec<bool> = Vec::with_capacity(chunk.len());
    let mut calc_source: Vec<i16> = Vec::with_capacity(chunk.len());
    let mut event_time_ms: Vec<Option<i64>> = Vec::with_capacity(chunk.len());
    let mut created_at: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());
    let mut updated_at: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());

    for (sym_id_str, sym_str, tf, tms, indicators, json_data, final_flag, cs, etm) in chunk {
        let sym_id = get_symbol_id(pool, cache, &sym_id_str).await?;

        time.push(ms_to_ts(tms));
        time_ms.push(tms);
        symbol_id.push(sym_id);
        symbol.push(sym_str);
        tf_minutes.push(tf);

        rsi.push(extract_f32(&indicators, "rsi"));
        cci.push(extract_f32(&indicators, "cci"));
        stoch_k.push(extract_f32(&indicators, "stoch_k"));
        stoch_d.push(extract_f32(&indicators, "stoch_d"));
        williams.push(extract_f32(&indicators, "williams"));

        macd.push(extract_f32(&indicators, "macd"));
        macd_signal.push(extract_f32(&indicators, "macd_signal"));
        macd_hist.push(extract_f32(&indicators, "macd_hist"));
        adx.push(extract_f32(&indicators, "adx"));
        sma.push(extract_f32(&indicators, "sma"));
        ema_20.push(extract_f32(&indicators, "ema_20"));
        ema_50.push(extract_f32(&indicators, "ema_50"));
        ema_200.push(extract_f32(&indicators, "ema_200"));

        bb_upper.push(extract_f32(&indicators, "bb_upper"));
        bb_mid.push(extract_f32(&indicators, "bb_mid"));
        bb_lower.push(extract_f32(&indicators, "bb_lower"));
        atr.push(extract_f32(&indicators, "atr"));

        obv.push(extract_f64(&indicators, "obv"));
        vwap.push(extract_f64(&indicators, "vwap"));
        volume_spike.push(extract_f32(&indicators, "volume_spike"));

        alligator_jaw.push(extract_f32(&indicators, "alligator_jaw"));
        alligator_teeth.push(extract_f32(&indicators, "alligator_teeth"));
        alligator_lips.push(extract_f32(&indicators, "alligator_lips"));

        trend.push(extract_i16(&indicators, "trend"));
        trend_short.push(extract_i16(&indicators, "trend_short"));
        poc.push(extract_f32(&indicators, "poc"));

        if skip_json {
            sr_levels.push(None);
        } else {
            sr_levels.push(json_data.get("sr_levels").cloned().map(Json));
        }

        candle_is_final.push(final_flag);
        calc_source.push(cs);
        event_time_ms.push(etm);
        created_at.push(now);
        updated_at.push(now);
    }

    let sql = match cfg.mode {
        PersistMode::Realtime => wide_sql_realtime(),
        PersistMode::History => {
            if cfg.history_upsert {
                wide_sql_history_upsert()
            } else {
                wide_sql_history_append()
            }
        }
    };

    sqlx::query(sql)
        .bind(time)
        .bind(time_ms)
        .bind(symbol_id)
        .bind(symbol)
        .bind(tf_minutes)
        .bind(rsi)
        .bind(cci)
        .bind(stoch_k)
        .bind(stoch_d)
        .bind(williams)
        .bind(macd)
        .bind(macd_signal)
        .bind(macd_hist)
        .bind(adx)
        .bind(sma)
        .bind(ema_20)
        .bind(ema_50)
        .bind(ema_200)
        .bind(bb_upper)
        .bind(bb_mid)
        .bind(bb_lower)
        .bind(atr)
        .bind(obv)
        .bind(vwap)
        .bind(volume_spike)
        .bind(alligator_jaw)
        .bind(alligator_teeth)
        .bind(alligator_lips)
        .bind(trend)
        .bind(trend_short)
        .bind(poc)
        .bind(sr_levels)
        .bind(candle_is_final)
        .bind(calc_source)
        .bind(event_time_ms)
        .bind(created_at)
        .bind(updated_at)
        .execute(pool)
        .await?;

    Ok(())
}

fn ms_to_ts(ms: i64) -> DateTime<Utc> {
    let secs = ms / 1000;
    let nsec = ((ms % 1000).max(0) as u32) * 1_000_000;
    DateTime::<Utc>::from_timestamp(secs, nsec).unwrap_or_else(|| Utc::now())
}

fn extract_f32(ind: &HashMap<String, f64>, key: &str) -> Option<f32> {
    ind.get(key).copied().map(|v| v as f32)
}
fn extract_f64(ind: &HashMap<String, f64>, key: &str) -> Option<f64> {
    ind.get(key).copied()
}
fn extract_i16(ind: &HashMap<String, f64>, key: &str) -> Option<i16> {
    ind.get(key).copied().map(|v| v as i16)
}

/* ---------------- SQL builders ---------------- */

fn raw_signals_sql_realtime() -> &'static str {
    r#"
    INSERT INTO market.raw_signals
    (time, time_ms, symbol_id, symbol, tf_minutes,
     indicator_id, signal_kind, signal_sub_id,
     side, score, value,
     details, candle_is_final, calc_source, event_time_ms,
     features_json, scores_json, predictors_json,
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
        predictors_json = EXCLUDED.predictors_json,
        updated_at = now()
    "#
}

fn raw_signals_sql_history_append() -> &'static str {
    r#"
    INSERT INTO market.raw_signals
    (time, time_ms, symbol_id, symbol, tf_minutes,
     indicator_id, signal_kind, signal_sub_id,
     side, score, value,
     details, candle_is_final, calc_source, event_time_ms,
     features_json, scores_json, predictors_json,
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
    ON CONFLICT (symbol_id, tf_minutes, time, indicator_id, signal_kind, signal_sub_id) DO NOTHING
    "#
}

fn raw_signals_sql_history_upsert() -> &'static str {
    // если хочешь перезаписывать историю при повторном прогоне
    raw_signals_sql_realtime()
}

fn wide_sql_realtime() -> &'static str {
    r#"
    INSERT INTO market.indicators_wide
    (time, time_ms, symbol_id, symbol, tf_minutes,
     rsi, cci, stoch_k, stoch_d, williams,
     macd, macd_signal, macd_hist, adx, sma, ema_20, ema_50, ema_200,
     bb_upper, bb_mid, bb_lower, atr,
     obv, vwap, volume_spike,
     alligator_jaw, alligator_teeth, alligator_lips,
     trend, trend_short, poc,
     sr_levels, candle_is_final, calc_source, event_time_ms, created_at, updated_at)
    SELECT * FROM UNNEST(
        $1::timestamptz[], $2::bigint[], $3::bigint[], $4::text[], $5::smallint[],
        $6::real[], $7::real[], $8::real[], $9::real[], $10::real[],
        $11::real[], $12::real[], $13::real[], $14::real[], $15::real[], $16::real[], $17::real[], $18::real[],
        $19::real[], $20::real[], $21::real[], $22::real[],
        $23::double precision[], $24::double precision[], $25::real[],
        $26::real[], $27::real[], $28::real[],
        $29::smallint[], $30::smallint[], $31::real[],
        $32::jsonb[], $33::boolean[], $34::smallint[], $35::bigint[], $36::timestamptz[], $37::timestamptz[]
    )
    ON CONFLICT (symbol_id, tf_minutes, time) DO UPDATE SET
        rsi = COALESCE(EXCLUDED.rsi, market.indicators_wide.rsi),
        cci = COALESCE(EXCLUDED.cci, market.indicators_wide.cci),
        stoch_k = COALESCE(EXCLUDED.stoch_k, market.indicators_wide.stoch_k),
        stoch_d = COALESCE(EXCLUDED.stoch_d, market.indicators_wide.stoch_d),
        williams = COALESCE(EXCLUDED.williams, market.indicators_wide.williams),
        macd = COALESCE(EXCLUDED.macd, market.indicators_wide.macd),
        macd_signal = COALESCE(EXCLUDED.macd_signal, market.indicators_wide.macd_signal),
        macd_hist = COALESCE(EXCLUDED.macd_hist, market.indicators_wide.macd_hist),
        adx = COALESCE(EXCLUDED.adx, market.indicators_wide.adx),
        sma = COALESCE(EXCLUDED.sma, market.indicators_wide.sma),
        ema_20 = COALESCE(EXCLUDED.ema_20, market.indicators_wide.ema_20),
        ema_50 = COALESCE(EXCLUDED.ema_50, market.indicators_wide.ema_50),
        ema_200 = COALESCE(EXCLUDED.ema_200, market.indicators_wide.ema_200),
        bb_upper = COALESCE(EXCLUDED.bb_upper, market.indicators_wide.bb_upper),
        bb_mid = COALESCE(EXCLUDED.bb_mid, market.indicators_wide.bb_mid),
        bb_lower = COALESCE(EXCLUDED.bb_lower, market.indicators_wide.bb_lower),
        atr = COALESCE(EXCLUDED.atr, market.indicators_wide.atr),
        obv = COALESCE(EXCLUDED.obv, market.indicators_wide.obv),
        vwap = COALESCE(EXCLUDED.vwap, market.indicators_wide.vwap),
        volume_spike = COALESCE(EXCLUDED.volume_spike, market.indicators_wide.volume_spike),
        alligator_jaw = COALESCE(EXCLUDED.alligator_jaw, market.indicators_wide.alligator_jaw),
        alligator_teeth = COALESCE(EXCLUDED.alligator_teeth, market.indicators_wide.alligator_teeth),
        alligator_lips = COALESCE(EXCLUDED.alligator_lips, market.indicators_wide.alligator_lips),
        trend = COALESCE(EXCLUDED.trend, market.indicators_wide.trend),
        trend_short = COALESCE(EXCLUDED.trend_short, market.indicators_wide.trend_short),
        poc = COALESCE(EXCLUDED.poc, market.indicators_wide.poc),
        sr_levels = COALESCE(EXCLUDED.sr_levels, market.indicators_wide.sr_levels),
        candle_is_final = EXCLUDED.candle_is_final,
        calc_source = EXCLUDED.calc_source,
        event_time_ms = EXCLUDED.event_time_ms,
        updated_at = now()
    "#
}

fn wide_sql_history_append() -> &'static str {
    // быстрый режим: вставить один раз и не трогать
    r#"
    INSERT INTO market.indicators_wide
    (time, time_ms, symbol_id, symbol, tf_minutes,
     rsi, cci, stoch_k, stoch_d, williams,
     macd, macd_signal, macd_hist, adx, sma, ema_20, ema_50, ema_200,
     bb_upper, bb_mid, bb_lower, atr,
     obv, vwap, volume_spike,
     alligator_jaw, alligator_teeth, alligator_lips,
     trend, trend_short, poc,
     sr_levels, candle_is_final, calc_source, event_time_ms, created_at, updated_at)
    SELECT * FROM UNNEST(
        $1::timestamptz[], $2::bigint[], $3::bigint[], $4::text[], $5::smallint[],
        $6::real[], $7::real[], $8::real[], $9::real[], $10::real[],
        $11::real[], $12::real[], $13::real[], $14::real[], $15::real[], $16::real[], $17::real[], $18::real[],
        $19::real[], $20::real[], $21::real[], $22::real[],
        $23::double precision[], $24::double precision[], $25::real[],
        $26::real[], $27::real[], $28::real[],
        $29::smallint[], $30::smallint[], $31::real[],
        $32::jsonb[], $33::boolean[], $34::smallint[], $35::bigint[], $36::timestamptz[], $37::timestamptz[]
    )
    ON CONFLICT (symbol_id, tf_minutes, time) DO NOTHING
    "#
}

fn wide_sql_history_upsert() -> &'static str {
    // если надо повторно прогонять историю и "дозаписывать"
    wide_sql_realtime()
}

/* --- SQL builders for predictors --- */

fn predictors_sql_history_append() -> &'static str {
    r#"
    INSERT INTO trade.predictors (
        time, time_ms, symbol_id, symbol, tf_minutes,
        horizon_bars, aspect, calc_source, predictor_id,
        score_norm, value, value_low, value_high, side,
        level_hash, level_kind, level_price, level_strength, level_distance_atr,
        candle_is_final, event_time_ms, details_json, prediction_key,
        created_at, updated_at
    )
    SELECT * FROM UNNEST(
        $1::timestamptz[],
        $2::bigint[],
        $3::bigint[],
        $4::text[],
        $5::int[],
        $6::int[],
        $7::smallint[],
        $8::smallint[],
        $9::bigint[],
        $10::real[],
        $11::double precision[],
        $12::double precision[],
        $13::double precision[],
        $14::smallint[],
        $15::text[],
        $16::smallint[],
        $17::double precision[],
        $18::real[],
        $19::real[],
        $20::boolean[],
        $21::bigint[],
        $22::jsonb[],
        $23::text[],
        $24::timestamptz[],
        $25::timestamptz[]
    )
    ON CONFLICT (symbol_id, tf_minutes, time, prediction_key, predictor_id) DO NOTHING
    "#
}

fn predictors_sql_history_upsert() -> &'static str {
    r#"
    INSERT INTO trade.predictors (
        time, time_ms, symbol_id, symbol, tf_minutes,
        horizon_bars, aspect, calc_source, predictor_id,
        score_norm, value, value_low, value_high, side,
        level_hash, level_kind, level_price, level_strength, level_distance_atr,
        candle_is_final, event_time_ms, details_json, prediction_key,
        created_at, updated_at
    )
    SELECT * FROM UNNEST(
        $1::timestamptz[],
        $2::bigint[],
        $3::bigint[],
        $4::text[],
        $5::int[],
        $6::int[],
        $7::smallint[],
        $8::smallint[],
        $9::bigint[],
        $10::real[],
        $11::double precision[],
        $12::double precision[],
        $13::double precision[],
        $14::smallint[],
        $15::text[],
        $16::smallint[],
        $17::double precision[],
        $18::real[],
        $19::real[],
        $20::boolean[],
        $21::bigint[],
        $22::jsonb[],
        $23::text[],
        $24::timestamptz[],
        $25::timestamptz[]
    )
    ON CONFLICT (symbol_id, tf_minutes, time, prediction_key, predictor_id)
    DO UPDATE SET
        time_ms = EXCLUDED.time_ms,
        symbol = EXCLUDED.symbol,
        horizon_bars = EXCLUDED.horizon_bars,
        aspect = EXCLUDED.aspect,
        calc_source = EXCLUDED.calc_source,
        score_norm = EXCLUDED.score_norm,
        value = EXCLUDED.value,
        value_low = EXCLUDED.value_low,
        value_high = EXCLUDED.value_high,
        side = EXCLUDED.side,
        level_hash = EXCLUDED.level_hash,
        level_kind = EXCLUDED.level_kind,
        level_price = EXCLUDED.level_price,
        level_strength = EXCLUDED.level_strength,
        level_distance_atr = EXCLUDED.level_distance_atr,
        candle_is_final = EXCLUDED.candle_is_final,
        event_time_ms = EXCLUDED.event_time_ms,
        details_json = EXCLUDED.details_json,
        updated_at = EXCLUDED.updated_at
    "#
}

fn predictors_sql_realtime() -> &'static str {
    // в realtime обычно нужен UPSERT (свеча может обновляться)
    predictors_sql_history_upsert()
}

/* --- Trade signals flush --- */

async fn flush_trade_signals(
    pool: &PgPool,
    cache: &tokio::sync::RwLock<HashMap<String, i64>>,
    cfg: &BulkPersistorConfig,
    items: &[PersistRecord],
) -> Result<()> {
    let mut vec: Vec<(PersistRecord, i64)> = Vec::with_capacity(items.len());

    for it in items {
        if let PersistRecord::TradeSignal { symbol, .. } = it {
            let sym_id = get_symbol_id(pool, cache, symbol.as_str()).await?;
            vec.push((it.clone(), sym_id));
        }
    }

    let total = vec.len();
    let mut written = 0;
    while !vec.is_empty() {
        let take = vec.len().min(cfg.chunk_size);
        let chunk: Vec<(PersistRecord, i64)> = vec.drain(0..take).collect();
        let chunk_len = chunk.len();
        flush_trade_signals_chunk(pool, cfg, chunk).await?;
        written += chunk_len;
    }

    if total > 0 {
        tracing::debug!(
            "flush_trade_signals: wrote {} rows (mode={:?})", written, cfg.mode
        );
    }

    Ok(())
}

async fn flush_trade_signals_chunk(
    pool: &PgPool,
    cfg: &BulkPersistorConfig,
    chunk: Vec<(PersistRecord, i64)>,
) -> Result<()> {
    if chunk.is_empty() { return Ok(()); }

    let now: DateTime<Utc> = Utc::now();
    let skip_json = cfg.mode == PersistMode::History && cfg.history_skip_json;

    let mut time: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());
    let mut time_ms: Vec<i64> = Vec::with_capacity(chunk.len());
    let mut symbol_id: Vec<i64> = Vec::with_capacity(chunk.len());
    let mut tf_minutes: Vec<i16> = Vec::with_capacity(chunk.len());
    let mut side: Vec<i16> = Vec::with_capacity(chunk.len());
    let mut final_score: Vec<f32> = Vec::with_capacity(chunk.len());
    let mut ml_score: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut heur_score: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut entry_price: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut sl_price: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut tp1_price: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut tp2_price: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut tp3_price: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut reason: Vec<Option<Json<Value>>> = Vec::with_capacity(chunk.len());
    let mut price10_target: Vec<Option<f64>> = Vec::with_capacity(chunk.len());
    let mut price10_score: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut bounce_prob: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut bounce_score: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut breakout_prob: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut breakout_score: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
    let mut created_at: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());

    for (rec, sym_id) in chunk {
        if let PersistRecord::TradeSignal {
            symbol: _sym, timeframe: tf, time_ms: tms,
            side: sd, final_score: fs, ml_score: mls, heur_score: hs,
            entry_price: ep, sl_price: slp,
            tp1_price: t1, tp2_price: t2, tp3_price: t3,
            reason: rsn,
            price10_target: p10t, price10_score: p10s,
            bounce_prob: bp, bounce_score: bs,
            breakout_prob: brp, breakout_score: brs,
        } = rec
        {
            time.push(ms_to_ts(tms));
            time_ms.push(tms);
            symbol_id.push(sym_id);
            tf_minutes.push(tf);
            side.push(sd);
            final_score.push(fs);
            ml_score.push(mls);
            heur_score.push(hs);
            entry_price.push(ep);
            sl_price.push(slp);
            tp1_price.push(t1);
            tp2_price.push(t2);
            tp3_price.push(t3);
            if skip_json {
                reason.push(None);
            } else {
                reason.push(rsn.map(Json));
            }
            price10_target.push(p10t);
            price10_score.push(p10s);
            bounce_prob.push(bp);
            bounce_score.push(bs);
            breakout_prob.push(brp);
            breakout_score.push(brs);
            created_at.push(now);
        }
    }

    let sql = match cfg.mode {
        PersistMode::Realtime => trade_signals_sql_realtime(),
        PersistMode::History => {
            if cfg.history_upsert { trade_signals_sql_realtime() }
            else { trade_signals_sql_history_append() }
        }
    };

    sqlx::query(sql)
        .bind(time_ms)
        .bind(time)
        .bind(symbol_id)
        .bind(tf_minutes)
        .bind(side)
        .bind(final_score)
        .bind(ml_score)
        .bind(heur_score)
        .bind(entry_price)
        .bind(sl_price)
        .bind(tp1_price)
        .bind(tp2_price)
        .bind(tp3_price)
        .bind(reason)
        .bind(price10_target)
        .bind(price10_score)
        .bind(bounce_prob)
        .bind(bounce_score)
        .bind(breakout_prob)
        .bind(breakout_score)
        .bind(created_at)
        .execute(pool)
        .await?;

    Ok(())
}

/* --- SQL builders for trade_signals --- */

fn trade_signals_sql_realtime() -> &'static str {
    r#"
    INSERT INTO trade.final_signals
    (time_ms, time, symbol_id, tf_minutes,
     side, final_score, ml_score, heur_score,
     entry_price, sl_price, tp1_price, tp2_price, tp3_price,
     reason,
     price10_target, price10_score,
     bounce_prob, bounce_score,
     breakout_prob, breakout_score,
     created_at)
    SELECT * FROM UNNEST(
        $1::bigint[],
        $2::timestamptz[],
        $3::bigint[],
        $4::smallint[],
        $5::smallint[],
        $6::real[],
        $7::real[],
        $8::real[],
        $9::real[],
        $10::real[],
        $11::real[],
        $12::real[],
        $13::real[],
        $14::jsonb[],
        $15::double precision[],
        $16::real[],
        $17::real[],
        $18::real[],
        $19::real[],
        $20::real[],
        $21::timestamptz[]
    )
    ON CONFLICT (symbol_id, tf_minutes, time) DO UPDATE SET
        side = EXCLUDED.side,
        final_score = EXCLUDED.final_score,
        ml_score = EXCLUDED.ml_score,
        heur_score = EXCLUDED.heur_score,
        entry_price = EXCLUDED.entry_price,
        sl_price = EXCLUDED.sl_price,
        tp1_price = EXCLUDED.tp1_price,
        tp2_price = EXCLUDED.tp2_price,
        tp3_price = EXCLUDED.tp3_price,
        reason = EXCLUDED.reason,
        price10_target = EXCLUDED.price10_target,
        price10_score = EXCLUDED.price10_score,
        bounce_prob = EXCLUDED.bounce_prob,
        bounce_score = EXCLUDED.bounce_score,
        breakout_prob = EXCLUDED.breakout_prob,
        breakout_score = EXCLUDED.breakout_score
    "#
}

fn trade_signals_sql_history_append() -> &'static str {
    r#"
    INSERT INTO trade.final_signals
    (time_ms, time, symbol_id, tf_minutes,
     side, final_score, ml_score, heur_score,
     entry_price, sl_price, tp1_price, tp2_price, tp3_price,
     reason,
     price10_target, price10_score,
     bounce_prob, bounce_score,
     breakout_prob, breakout_score,
     created_at)
    SELECT * FROM UNNEST(
        $1::bigint[],
        $2::timestamptz[],
        $3::bigint[],
        $4::smallint[],
        $5::smallint[],
        $6::real[],
        $7::real[],
        $8::real[],
        $9::real[],
        $10::real[],
        $11::real[],
        $12::real[],
        $13::real[],
        $14::jsonb[],
        $15::double precision[],
        $16::real[],
        $17::real[],
        $18::real[],
        $19::real[],
        $20::real[],
        $21::timestamptz[]
    )
    ON CONFLICT (symbol_id, tf_minutes, time) DO NOTHING
    "#
}
