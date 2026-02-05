// compute/src/indicator_persistor.rs

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use sqlx::{PgPool, Postgres, QueryBuilder, Row};
use tokio::sync::{mpsc, RwLock};
use tracing::{error, warn};

use common::{Symbol, Timeframe};

use crate::FeatureValue;

#[derive(Debug, Clone)]
pub struct IndicatorRecord {
    pub symbol: Symbol,
    pub timeframe: Timeframe,
    pub timestamp: i64,
    pub indicator_name: String,
    pub value: FeatureValue,
}

pub struct IndicatorPersistor {
    pub db_pool: PgPool,
    pub persist_queue: Arc<RwLock<Vec<IndicatorRecord>>>,
    pub batch_size: usize,
    pub flush_interval_ms: u64,

    pub symbol_id_cache: Arc<RwLock<HashMap<Symbol, i64>>>,
}

impl IndicatorPersistor {
    pub fn new(
        db_pool: PgPool,
        batch_size: usize,
        flush_interval_ms: u64,
    ) -> (Self, mpsc::UnboundedSender<IndicatorRecord>) {
        let persist_queue = Arc::new(RwLock::new(Vec::new()));
        let symbol_id_cache = Arc::new(RwLock::new(HashMap::new()));

        // Канал (если где-то захочешь отправлять по одному record)
        let (sender, mut receiver) = mpsc::unbounded_channel::<IndicatorRecord>();

        // Receiver -> persist_queue
        {
            let q = Arc::clone(&persist_queue);
            tokio::spawn(async move {
                while let Some(r) = receiver.recv().await {
                    q.write().await.push(r);
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

    pub async fn queue_records(&self, records: Vec<IndicatorRecord>) {
        if records.is_empty() {
            return;
        }
        self.persist_queue.write().await.extend(records);
    }

    pub async fn start_persistence_loop(self: Arc<Self>) {
        let mut interval = tokio::time::interval(Duration::from_millis(self.flush_interval_ms));

        loop {
            interval.tick().await;

            let batch: Vec<IndicatorRecord> = {
                let mut q = self.persist_queue.write().await;
                if q.is_empty() {
                    continue;
                }
                let take = self.batch_size.min(q.len());
                q.drain(0..take).collect()
            };

            if let Err(e) = self.flush_batch(batch).await {
                error!("Error flushing indicators batch: {}", e);
            }
        }
    }

    async fn flush_batch(&self, records: Vec<IndicatorRecord>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if records.is_empty() {
            return Ok(());
        }

        // 1) Подгружаем symbol_id в кэш (батчом)
        self.prefill_symbol_cache(&records).await?;

        // 2) Resolve symbol_id для каждой записи
        let mut resolved: Vec<(i64, IndicatorRecord)> = Vec::with_capacity(records.len());
        {
            let cache = self.symbol_id_cache.read().await;
            for r in records {
                if let Some(&sid) = cache.get(&r.symbol) {
                    resolved.push((sid, r));
                } else {
                    warn!("IndicatorPersistor: symbol_id not found for {}", r.symbol);
                }
            }
        }
        if resolved.is_empty() {
            return Ok(());
        }

        // 3) Дедуп (чтобы не получить ON CONFLICT ... row a second time)
        #[derive(Hash, Eq, PartialEq)]
        struct Key {
            symbol_id: i64,
            time_ms: i64,
            tf: String,
            name: String,
        }

        let mut best: HashMap<Key, (i64, IndicatorRecord)> = HashMap::new();
        for (sid, r) in resolved {
            let k = Key {
                symbol_id: sid,
                time_ms: r.timestamp,
                tf: r.timeframe.as_str().to_string(),
                name: r.indicator_name.clone(),
            };
            // просто “последний победил”
            best.insert(k, (sid, r));
        }
        let deduped: Vec<(i64, IndicatorRecord)> = best.into_values().collect();

        // 4) Группируем по table_name = market.indicators_{tf}
        let mut by_table: HashMap<String, Vec<(i64, IndicatorRecord)>> = HashMap::new();
        for (sid, r) in deduped {
            let tf = r.timeframe.as_str();
            if !is_safe_ident(tf) {
                warn!("IndicatorPersistor: unsafe timeframe ident: {}", tf);
                continue;
            }
            let table = format!("market.indicators_{}", tf);
            by_table.entry(table).or_default().push((sid, r));
        }

        // 5) UPSERT чанками
        let mut tx = self.db_pool.begin().await?;
        let now = Utc::now();
        let now_ms = now.timestamp_millis();

        const CHUNK: usize = 1500; // 1 строка ~ 15 bind => далеко до лимита 65535

        for (table, rows) in by_table {
            for chunk in rows.chunks(CHUNK) {
                let mut qb = QueryBuilder::<Postgres>::new(format!(
                    "INSERT INTO {} \
                    (time, time_ms, symbol_id, symbol, tf_minutes, indicator_name, value_float, value_json, candle_is_final, calc_source, event_time_ms, created_at, created_at_ms, updated_at, updated_at_ms) ",
                    table
                ));

                qb.push_values(chunk, |mut b, (symbol_id, r)| {
                    let dt: DateTime<Utc> = DateTime::<Utc>::from_timestamp_millis(r.timestamp)
                        .unwrap_or_else(|| Utc::now());

                    let (vf, vj) = match &r.value {
                        FeatureValue::Float(v) => (Some(*v), None),
                        FeatureValue::Json(j) => (None, Some(j.clone())),
                    };

                    b.push_bind(dt);
                    b.push_bind(r.timestamp);
                    b.push_bind(*symbol_id);
                    b.push_bind(r.symbol.to_string());
                    b.push_bind(r.timeframe.to_minutes() as i16);
                    b.push_bind(r.indicator_name.as_str());
                    b.push_bind(vf);
                    b.push_bind(vj.map(sqlx::types::Json));

                    // мета пока дефолт
                    b.push_bind(true);          // candle_is_final
                    b.push_bind(1_i16);         // calc_source
                    b.push_bind(Option::<i64>::None); // event_time_ms

                    b.push_bind(now);
                    b.push_bind(now_ms);
                    b.push_bind(now);
                    b.push_bind(now_ms);
                });

                qb.push(
                    " ON CONFLICT (symbol_id, time, indicator_name) DO UPDATE SET \
                      time_ms = EXCLUDED.time_ms, \
                      symbol = EXCLUDED.symbol, \
                      tf_minutes = EXCLUDED.tf_minutes, \
                      value_float = EXCLUDED.value_float, \
                      value_json = EXCLUDED.value_json, \
                      candle_is_final = EXCLUDED.candle_is_final, \
                      calc_source = EXCLUDED.calc_source, \
                      event_time_ms = EXCLUDED.event_time_ms, \
                      updated_at = EXCLUDED.updated_at, \
                      updated_at_ms = EXCLUDED.updated_at_ms"
                );

                qb.build().execute(&mut *tx).await?;
            }
        }

        tx.commit().await?;
        Ok(())
    }

    async fn prefill_symbol_cache(&self, records: &[IndicatorRecord]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

fn is_safe_ident(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}