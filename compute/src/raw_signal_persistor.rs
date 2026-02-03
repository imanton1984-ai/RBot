use std::{collections::{HashMap, HashSet}, sync::Arc};

use chrono::{TimeZone, Utc};
use common::Symbol;
use sqlx::{PgPool, Postgres, QueryBuilder};
use tokio::sync::{mpsc, RwLock};

use crate::raw_signal_types::RawSignal;

pub struct RawSignalPersistor {
    db_pool: PgPool,
    persist_queue: Arc<RwLock<Vec<RawSignal>>>,
    batch_size: usize,
    flush_interval_ms: u64,

    // кеш symbol -> symbol_id чтобы не долбить БД каждый flush
    symbol_id_cache: Arc<RwLock<HashMap<Symbol, i64>>>,
}

impl RawSignalPersistor {
    pub fn new(
        db_pool: PgPool,
        batch_size: usize,
        flush_interval_ms: u64,
    ) -> (Self, mpsc::UnboundedSender<RawSignal>) {
        let (sender, mut receiver) = mpsc::unbounded_channel();

        let persistor = Self {
            db_pool,
            persist_queue: Arc::new(RwLock::new(Vec::new())),
            batch_size,
            flush_interval_ms,
            symbol_id_cache: Arc::new(RwLock::new(HashMap::new())),
        };

        let queue = persistor.persist_queue.clone();
        tokio::spawn(async move {
            while let Some(record) = receiver.recv().await {
                let mut q = queue.write().await;
                q.push(record);
            }
        });

        (persistor, sender)
    }

    pub async fn start_persistence_loop(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut last_flush = tokio::time::Instant::now();
        let interval = tokio::time::Duration::from_millis(self.flush_interval_ms);

        loop {
            let should_flush_size = {
                let q = self.persist_queue.read().await;
                q.len() >= self.batch_size
            };

            let should_flush_time = last_flush.elapsed() >= interval;

            if should_flush_size || should_flush_time {
                if let Err(e) = self.flush_batch().await {
                    eprintln!("Error flushing raw signals batch: {}", e);
                }
                last_flush = tokio::time::Instant::now();
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }

    async fn flush_batch(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let records = {
            let mut q = self.persist_queue.write().await;
            if q.is_empty() {
                return Ok(());
            }
            q.drain(..).collect::<Vec<_>>()
        };

        // 1) Собираем уникальные symbols, которые не в кеше
        let mut need_fetch: HashSet<String> = HashSet::new();
        {
            let cache = self.symbol_id_cache.read().await;
            for r in &records {
                if !cache.contains_key(&r.symbol) {
                    need_fetch.insert(r.symbol.as_str().to_string());
                }
            }
        }

        // 2) Догружаем symbol_id одним запросом
        if !need_fetch.is_empty() {
            let symbols: Vec<String> = need_fetch.into_iter().collect();
            let rows = sqlx::query!(
                r#"
                SELECT symbol, symbol_id
                FROM market.pairs
                WHERE symbol = ANY($1)
                "#,
                &symbols
            )
            .fetch_all(&self.db_pool)
            .await?;

            let mut cache = self.symbol_id_cache.write().await;
            for row in rows {
                // Symbol у тебя типизированный, поэтому создаём через Symbol::from если он есть.
                let sym = Symbol::from(row.symbol);
                cache.insert(sym, row.symbol_id);
            }
        }

        // 3) Превращаем RawSignal -> (symbol_id, signal)
        let mut resolved: Vec<(i64, RawSignal)> = Vec::with_capacity(records.len());
        {
            let cache = self.symbol_id_cache.read().await;
            for r in records {
                if let Some(&sid) = cache.get(&r.symbol) {
                    resolved.push((sid, r));
                } else {
                    eprintln!("RawSignalPersistor: symbol_id not found for {}", r.symbol);
                }
            }
        }

        if resolved.is_empty() {
            return Ok(());
        }

        // 4) Bulk UPSERT чанками
        self.upsert_bulk(&resolved).await?;

        Ok(())
    }

    async fn upsert_bulk(
        &self,
        rows: &[(i64, RawSignal)],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Важно: лимит параметров Postgres ~ 65535. У нас 12 bind'ов на строку.
        // 1000 строк = 12000 параметров — норм.
        const CHUNK: usize = 1000;

        let now = Utc::now();
        let now_ms = now.timestamp_millis();

        let mut tx = self.db_pool.begin().await?;

        for chunk in rows.chunks(CHUNK) {
            let mut qb: QueryBuilder<Postgres> = QueryBuilder::new(
                r#"
                INSERT INTO market.raw_signals
                  (time_ms, time, symbol_id, tf_minutes, indicator_id, signal_kind,
                   side, score, value, details, created_at_ms, created_at)
                "#
            );

            qb.push_values(chunk, |mut b, (symbol_id, s)| {
                let time_value = Utc
                    .timestamp_millis_opt(s.timestamp)
                    .single()
                    .unwrap_or(now);

                b.push_bind(s.timestamp)
                    .push_bind(time_value)
                    .push_bind(*symbol_id)
                    .push_bind(s.timeframe.to_minutes() as i16)
                    .push_bind(s.indicator_id)
                    .push_bind(s.signal_kind)
                    .push_bind(s.side)
                    .push_bind(s.score)
                    .push_bind(s.value)
                    .push_bind(&s.details)
                    .push_bind(now_ms)
                    .push_bind(now);
            });

            // Ключ берём по имени constraint (идеально для timescale chunks)
            qb.push(
                r#"
                ON CONFLICT ON CONSTRAINT raw_signals_pkey
                DO UPDATE SET
                  side          = EXCLUDED.side,
                  score         = EXCLUDED.score,
                  value         = EXCLUDED.value,
                  details       = EXCLUDED.details,
                  created_at_ms = EXCLUDED.created_at_ms,
                  created_at    = EXCLUDED.created_at
                "#
            );

            qb.build().execute(&mut *tx).await?;
        }

        tx.commit().await?;
        Ok(())
    }
}
