// compute/src/raw_signal_persistor.rs

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use sqlx::{PgPool, Postgres, QueryBuilder, Row};
use tokio::sync::{mpsc, RwLock};
use tracing::{error, warn};

use common::Symbol;

use crate::raw_signal_types::RawSignal;

pub struct RawSignalPersistor {
    pub db_pool: PgPool,
    pub persist_queue: Arc<RwLock<Vec<RawSignal>>>,
    pub batch_size: usize,
    pub flush_interval_ms: u64,

    pub symbol_id_cache: Arc<RwLock<HashMap<Symbol, i64>>>,
}

impl RawSignalPersistor {
    pub fn new(
        db_pool: PgPool,
        batch_size: usize,
        flush_interval_ms: u64,
    ) -> (Self, mpsc::UnboundedSender<RawSignal>) {
        let persist_queue = Arc::new(RwLock::new(Vec::new()));
        let symbol_id_cache = Arc::new(RwLock::new(HashMap::new()));

        let (sender, mut receiver) = mpsc::unbounded_channel::<RawSignal>();

        // Receiver -> queue
        {
            let q = Arc::clone(&persist_queue);
            tokio::spawn(async move {
                while let Some(s) = receiver.recv().await {
                    q.write().await.push(s);
                }
            });
        }

        (
            Self {
                db_pool,
                persist_queue,
                batch_size,
                flush_interval_ms,
                symbol_id_cache,
            },
            sender,
        )
    }

    pub async fn start_persistence_loop(self: Arc<Self>) {
        let mut interval = tokio::time::interval(Duration::from_millis(self.flush_interval_ms));

        loop {
            interval.tick().await;

            let batch: Vec<RawSignal> = {
                let mut q = self.persist_queue.write().await;
                if q.is_empty() {
                    continue;
                }
                let take = self.batch_size.min(q.len());
                q.drain(0..take).collect()
            };

            if let Err(e) = self.flush_batch(batch).await {
                error!("Error flushing raw signals batch: {}", e);
            }
        }
    }

    async fn flush_batch(&self, records: Vec<RawSignal>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if records.is_empty() {
            return Ok(());
        }

        // 1) cache symbol_id батчом
        self.prefill_symbol_cache(&records).await?;

        // 2) resolve
        let mut resolved: Vec<(i64, RawSignal)> = Vec::with_capacity(records.len());
        {
            let cache = self.symbol_id_cache.read().await;
            for r in records {
                if let Some(&sid) = cache.get(&r.symbol) {
                    resolved.push((sid, r));
                } else {
                    warn!("RawSignalPersistor: symbol_id not found for {}", r.symbol);
                }
            }
        }
        if resolved.is_empty() {
            return Ok(());
        }

        // 3) DEDUP по точному ключу upsert (включая signal_sub_id!)
        #[derive(Hash, Eq, PartialEq)]
        struct Key {
            symbol_id: i64,
            tf_minutes: i16,
            time_ms: i64,
            indicator_id: i16,
            signal_kind: i16,
            signal_sub_id: i16,
        }

        let mut best: HashMap<Key, (i64, RawSignal)> = HashMap::new();
        for (sid, s) in resolved {
            let k = Key {
                symbol_id: sid,
                tf_minutes: s.timeframe.to_minutes() as i16,
                time_ms: s.timestamp,
                indicator_id: s.indicator_id,
                signal_kind: s.signal_kind,
                signal_sub_id: s.signal_sub_id,
            };

            match best.get(&k) {
                None => {
                    best.insert(k, (sid, s));
                }
                Some((_, prev)) => {
                    // оставляем более “сильный” score
                    if s.score.abs() > prev.score.abs() {
                        best.insert(k, (sid, s));
                    }
                }
            }
        }

        let deduped: Vec<(i64, RawSignal)> = best.into_values().collect();

        // 4) UPSERT чанками
        const CHUNK: usize = 1200; // строка ~ 20 bind

        let mut tx = self.db_pool.begin().await?;
        let now = Utc::now();
        let now_ms = now.timestamp_millis();

        for chunk in deduped.chunks(CHUNK) {
            let mut qb = QueryBuilder::<Postgres>::new(
                "INSERT INTO market.raw_signals \
                (time, time_ms, symbol_id, symbol, tf_minutes, indicator_id, signal_kind, signal_sub_id, side, score, value, details, \
                 candle_is_final, calc_source, event_time_ms, features_json, scores_json, predictions_json, \
                 created_at, created_at_ms, updated_at, updated_at_ms) "
            );

            qb.push_values(chunk, |mut b, (symbol_id, s)| {
                let dt: DateTime<Utc> = DateTime::<Utc>::from_timestamp_millis(s.timestamp)
                    .unwrap_or_else(|| Utc::now());

                b.push_bind(dt);
                b.push_bind(s.timestamp);
                b.push_bind(*symbol_id);
                b.push_bind(s.symbol.to_string());
                b.push_bind(s.timeframe.to_minutes() as i16);

                b.push_bind(s.indicator_id);
                b.push_bind(s.signal_kind);
                b.push_bind(s.signal_sub_id);

                b.push_bind(s.side);
                b.push_bind(s.score);
                b.push_bind(s.value);

                b.push_bind(s.details.clone().map(sqlx::types::Json));

                b.push_bind(s.candle_is_final);
                b.push_bind(s.calc_source);
                b.push_bind(s.event_time_ms);

                b.push_bind(s.features_json.clone().map(sqlx::types::Json));
                b.push_bind(s.scores_json.clone().map(sqlx::types::Json));
                b.push_bind(s.predictions_json.clone().map(sqlx::types::Json));

                b.push_bind(now);
                b.push_bind(now_ms);
                b.push_bind(now);
                b.push_bind(now_ms);
            });

            qb.push(
                " ON CONFLICT (symbol_id, tf_minutes, time, indicator_id, signal_kind, signal_sub_id) DO UPDATE SET \
                  time_ms = EXCLUDED.time_ms, \
                  symbol = EXCLUDED.symbol, \
                  side = EXCLUDED.side, \
                  score = EXCLUDED.score, \
                  value = EXCLUDED.value, \
                  details = EXCLUDED.details, \
                  candle_is_final = EXCLUDED.candle_is_final, \
                  calc_source = EXCLUDED.calc_source, \
                  event_time_ms = EXCLUDED.event_time_ms, \
                  features_json = EXCLUDED.features_json, \
                  scores_json = EXCLUDED.scores_json, \
                  predictions_json = EXCLUDED.predictions_json, \
                  updated_at = EXCLUDED.updated_at, \
                  updated_at_ms = EXCLUDED.updated_at_ms"
            );

            qb.build().execute(&mut *tx).await?;
        }

        tx.commit().await?;
        Ok(())
    }

    async fn prefill_symbol_cache(&self, records: &[RawSignal]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut missing: HashSet<String> = HashSet::new();
        {
            let cache = self.symbol_id_cache.read().await;
            for r in records {
                if !cache.contains_key(&r.symbol) {
                    missing.insert(r.symbol.to_string());
                }
            }
        }
        if missing.is_empty() {
            return Ok(());
        }

        let symbols: Vec<String> = missing.into_iter().collect();
        let rows = sqlx::query("SELECT symbol_id, symbol FROM market.pairs WHERE symbol = ANY($1)")
            .bind(&symbols)
            .fetch_all(&self.db_pool)
            .await?;

        let mut cache = self.symbol_id_cache.write().await;
        for row in rows {
            let sid: i64 = row.get("symbol_id");
            let sym: String = row.get("symbol");
            cache.insert(Symbol::from(sym), sid);
        }

        Ok(())
    }
}