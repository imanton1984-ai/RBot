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
        predictors_json: Option<Value>,
    },

    // NEW VARIANT for wide indicators
    IndicatorsWide {
        symbol: common::Symbol,
        timeframe: i16,
        time_ms: i64,
        // Using a Map for flexibility in the persistor, or a struct if strictly typed
        indicators: std::collections::HashMap<String, f64>, 
        json_data: std::collections::HashMap<String, serde_json::Value>,
        candle_is_final: bool,
        calc_source: i16,
        event_time_ms: Option<i64>,
    },

    // NEW VARIANT for aggregated signals
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

async fn flush_all(
    pool: &PgPool,
    cache: &tokio::sync::RwLock<HashMap<String, i64>>,
    cfg: &BulkPersistorConfig,
    batch: &mut Vec<PersistRecord>,
) -> Result<()> {
    let mut signals: Vec<PersistRecord> = Vec::new();
    let mut wide_indicators: Vec<PersistRecord> = Vec::new();

    for r in batch.drain(..) {
        match r {
            PersistRecord::Indicator { .. } => {
                // Старый формат игнорируем или логируем warning
            }, 
            PersistRecord::RawSignal { .. } => signals.push(r),
            PersistRecord::IndicatorsWide { .. } => wide_indicators.push(r),
            PersistRecord::AggregatedSignal { .. } => {
                 // Игнорируем aggregated, так как используем raw_signals
            },
        }
    }

    if !signals.is_empty() {
        if let Err(e) = flush_raw_signals(pool, cache, cfg, &signals).await {
            tracing::error!("Failed to flush raw signals: {}", e);
        }
    }
    if !wide_indicators.is_empty() {
        if let Err(e) = flush_wide_indicators(pool, cache, cfg, &wide_indicators).await {
            tracing::error!("Failed to flush wide indicators: {}", e);
        }
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
    let mut predictors_json: Vec<Option<Json<Value>>> = Vec::with_capacity(chunk.len());

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
            predictors_json: pj,
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
            predictors_json.push(pj.map(Json));

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
        .bind(predictors_json)
        .bind(created_at)
        .bind(updated_at)
        .execute(pool)
        .await {
            tracing::error!("CRITICAL: Failed to insert signals: {:?}", e);
            return Err(e.into());
        }

    Ok(())
}

async fn flush_wide_indicators(
    pool: &PgPool,
    cache: &tokio::sync::RwLock<HashMap<String, i64>>,
    cfg: &BulkPersistorConfig,
    items: &[PersistRecord],
) -> Result<()> {
    // Group by symbol_id and time to merge indicators for the same candle
    let mut grouped: HashMap<(i64, i16, i64), (String, String, i16, i64, std::collections::HashMap<String, f64>, std::collections::HashMap<String, serde_json::Value>, bool, i16, Option<i64>)> = HashMap::new();

    for item in items {
        if let PersistRecord::IndicatorsWide { symbol, timeframe, time_ms, indicators, json_data, candle_is_final, calc_source, event_time_ms } = item {
            let sym_id = get_symbol_id(pool, cache, symbol.as_str()).await?;
            let key = (sym_id, *timeframe, *time_ms);
            
            match grouped.entry(key) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    // Merge indicators
                    let (_, _, _, _, existing_indicators, existing_json, _, _, _) = entry.get_mut();
                    for (k, v) in indicators {
                        existing_indicators.insert(k.clone(), *v);
                    }
                    for (k, v) in json_data {
                        existing_json.insert(k.clone(), v.clone());
                    }
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let mut new_indicators = std::collections::HashMap::new();
                    let mut new_json_data = std::collections::HashMap::new();
                    for (k, v) in indicators {
                        new_indicators.insert(k.clone(), *v);
                    }
                    for (k, v) in json_data {
                        new_json_data.insert(k.clone(), v.clone());
                    }
                    entry.insert((symbol.0.clone(), symbol.0.clone(), *timeframe, *time_ms, new_indicators, new_json_data, *candle_is_final, *calc_source, *event_time_ms));
                }
            }
        }
    }

    // Convert grouped data to vectors for bulk insert
    let mut records: Vec<_> = grouped.into_values().collect();
    
    while !records.is_empty() {
        let take = records.len().min(cfg.chunk_size);
        let chunk: Vec<_> = records.drain(0..take).collect();
        flush_wide_indicators_chunk(pool, cache, chunk).await?;
    }

    Ok(())
}

async fn flush_wide_indicators_chunk(
    pool: &PgPool,
    cache: &tokio::sync::RwLock<HashMap<String, i64>>,
    chunk: Vec<(String, String, i16, i64, std::collections::HashMap<String, f64>, std::collections::HashMap<String, serde_json::Value>, bool, i16, Option<i64>)>,
) -> Result<()> {
    if chunk.is_empty() { return Ok(()); }
    
    tracing::info!("Persistor: Flushing {} wide indicator rows...", chunk.len());
    let now: DateTime<Utc> = Utc::now();

    let mut time: Vec<DateTime<Utc>> = Vec::with_capacity(chunk.len());
    let mut time_ms: Vec<i64> = Vec::with_capacity(chunk.len());
    let mut symbol_id: Vec<i64> = Vec::with_capacity(chunk.len());
    let mut symbol: Vec<String> = Vec::with_capacity(chunk.len());
    let mut tf_minutes: Vec<i16> = Vec::with_capacity(chunk.len());

    // Vectors for all columns in DDL order
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

    // Log indicator fill stats for first row in chunk (diagnostic)
    if let Some((_, _, _, _, ref first_indicators, _, _, _, _)) = chunk.first() {
        let non_null_count = first_indicators.len();
        tracing::debug!(
            "Wide indicator sample: {} non-null indicators in first row (keys: {:?})",
            non_null_count,
            first_indicators.keys().collect::<Vec<_>>()
        );
    }

    for (sym_id_str, sym_str, tf, tms, indicators, json_data, final_flag, cs, etm) in chunk {
        // Resolve symbol_id (reuse cache logic)
        let sym_id = get_symbol_id(pool, cache, &sym_id_str).await?;

        time.push(ms_to_ts(tms));
        time_ms.push(tms);
        symbol_id.push(sym_id);
        symbol.push(sym_str);
        tf_minutes.push(tf);

        // Extraction matching Compute backend names
        rsi.push(extract_indicator_value(&indicators, "rsi"));
        cci.push(extract_indicator_value(&indicators, "cci"));
        stoch_k.push(extract_indicator_value(&indicators, "stoch_k"));
        stoch_d.push(extract_indicator_value(&indicators, "stoch_d"));
        williams.push(extract_indicator_value(&indicators, "williams"));
        
        macd.push(extract_indicator_value(&indicators, "macd"));
        macd_signal.push(extract_indicator_value(&indicators, "macd_signal"));
        macd_hist.push(extract_indicator_value(&indicators, "macd_hist"));
        adx.push(extract_indicator_value(&indicators, "adx"));
        sma.push(extract_indicator_value(&indicators, "sma"));
        ema_20.push(extract_indicator_value(&indicators, "ema_20"));
        ema_50.push(extract_indicator_value(&indicators, "ema_50"));
        ema_200.push(extract_indicator_value(&indicators, "ema_200"));
        
        bb_upper.push(extract_indicator_value(&indicators, "bb_upper"));
        bb_mid.push(extract_indicator_value(&indicators, "bb_mid"));
        bb_lower.push(extract_indicator_value(&indicators, "bb_lower"));
        atr.push(extract_indicator_value(&indicators, "atr"));
        
        obv.push(extract_indicator_value_as_double(&indicators, "obv"));
        vwap.push(extract_indicator_value_as_double(&indicators, "vwap"));
        volume_spike.push(extract_indicator_value(&indicators, "volume_spike"));
        
        alligator_jaw.push(extract_indicator_value(&indicators, "alligator_jaw"));
        alligator_teeth.push(extract_indicator_value(&indicators, "alligator_teeth"));
        alligator_lips.push(extract_indicator_value(&indicators, "alligator_lips"));
        
        trend.push(extract_indicator_value_as_i16(&indicators, "trend"));
        trend_short.push(extract_indicator_value_as_i16(&indicators, "trend_short"));
        poc.push(extract_indicator_value(&indicators, "poc"));
        
        sr_levels.push(json_data.get("sr_levels").cloned().map(Json));
        
        candle_is_final.push(final_flag);
        calc_source.push(cs);
        event_time_ms.push(etm);
        created_at.push(now);
        updated_at.push(now);
    }

    // SQL MUST match the column list exactly
    let sql = r#"
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
    "#;

    if let Err(e) = sqlx::query(sql)
        .bind(time).bind(time_ms).bind(symbol_id).bind(symbol).bind(tf_minutes)
        .bind(rsi).bind(cci).bind(stoch_k).bind(stoch_d).bind(williams)
        .bind(macd).bind(macd_signal).bind(macd_hist).bind(adx).bind(sma).bind(ema_20).bind(ema_50).bind(ema_200)
        .bind(bb_upper).bind(bb_mid).bind(bb_lower).bind(atr)
        .bind(obv).bind(vwap).bind(volume_spike)
        .bind(alligator_jaw).bind(alligator_teeth).bind(alligator_lips)
        .bind(trend).bind(trend_short).bind(poc)
        .bind(sr_levels).bind(candle_is_final).bind(calc_source).bind(event_time_ms).bind(created_at).bind(updated_at)
        .execute(pool).await 
    {
        tracing::error!("CRITICAL: Failed to insert wide indicators: {:?}", e);
        return Err(e.into());
    }
    Ok(())
}

fn ms_to_ts(ms: i64) -> DateTime<Utc> {
    let secs = ms / 1000;
    let nsec = ((ms % 1000).max(0) as u32) * 1_000_000;
    DateTime::<Utc>::from_timestamp(secs, nsec).unwrap_or_else(|| Utc::now())
}

// Helper functions to extract indicator values
fn extract_indicator_value(indicators: &std::collections::HashMap<String, f64>, key: &str) -> Option<f32> {
    indicators.get(key).copied().map(|v| v as f32)
}

fn extract_indicator_value_as_double(indicators: &std::collections::HashMap<String, f64>, key: &str) -> Option<f64> {
    indicators.get(key).copied()
}

fn extract_indicator_value_as_i16(indicators: &std::collections::HashMap<String, f64>, key: &str) -> Option<i16> {
    indicators.get(key).copied().map(|v| v as i16)
}