use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use common::{Symbol, Timeframe};
use sqlx::PgPool;
use chrono::{Utc, TimeZone};
use std::collections::HashMap;
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
    db_pool: PgPool,
    persist_queue: Arc<RwLock<Vec<IndicatorRecord>>>,
    batch_size: usize,
    flush_interval_ms: u64,
}

impl IndicatorPersistor {
    pub fn new(
        db_pool: PgPool,
        batch_size: usize,
        flush_interval_ms: u64,
    ) -> (Self, mpsc::UnboundedSender<IndicatorRecord>) {
        let (sender, _) = mpsc::unbounded_channel();
        (
            Self {
                db_pool,
                persist_queue: Arc::new(RwLock::new(Vec::new())),
                batch_size,
                flush_interval_ms,
            },
            sender,
        )
    }

    pub async fn start_persistence_loop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut last_flush = tokio::time::Instant::now();
        let interval = tokio::time::Duration::from_millis(self.flush_interval_ms);

        loop {
            let should_flush_size;
            {
                let queue = self.persist_queue.read().await;
                should_flush_size = queue.len() >= self.batch_size;
            }

            let should_flush_time = last_flush.elapsed() >= interval;

            if should_flush_size || should_flush_time {
                if let Err(e) = self.flush_batch().await {
                    eprintln!("Error flushing indicators batch: {}", e);
                }
                last_flush = tokio::time::Instant::now();
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }

    async fn flush_batch(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let records_to_flush = {
            let mut queue = self.persist_queue.write().await;
            if queue.is_empty() {
                return Ok(());
            }
            queue.drain(..).collect::<Vec<_>>()
        };

        let mut grouped_records: HashMap<(Symbol, Timeframe, i64), HashMap<String, FeatureValue>> = HashMap::new();
        
        for record in records_to_flush {
            let key = (record.symbol.clone(), record.timeframe, record.timestamp);
            grouped_records.entry(key).or_insert_with(HashMap::new)
                .insert(record.indicator_name, record.value);
        }

        for ((symbol, timeframe, timestamp), indicators_map) in grouped_records {
            let symbol_id: i64 = match sqlx::query_scalar("SELECT symbol_id FROM market.pairs WHERE symbol = $1")
                .bind(symbol.as_str())
                .fetch_optional(&self.db_pool)
                .await? {
                    Some(id) => id,
                    None => {
                        eprintln!("Unknown symbol for indicator persist: {}", symbol);
                        continue;
                    }
                };

            let table_name = format!("market.indicators_{}", timeframe.as_str());
            let time_utc = Utc.timestamp_millis_opt(timestamp).single().unwrap_or(Utc::now());

            let mut indicator_names = Vec::new();
            let mut indicator_values: Vec<&FeatureValue> = Vec::new();
            
            for (indicator_name, value) in &indicators_map {
                if !indicator_name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    continue;
                }
                indicator_names.push(indicator_name.as_str());
                indicator_values.push(value);
            }

            if indicator_names.is_empty() {
                continue;
            }

            let mut query_builder = sqlx::QueryBuilder::new(format!("INSERT INTO {} (time_ms, time, symbol_id", table_name));
            for indicator_name in &indicator_names {
                query_builder.push(format!(", {}", indicator_name));
            }

            query_builder.push(") VALUES (");
            query_builder.push_bind(timestamp);
            query_builder.push(", ");
            query_builder.push_bind(time_utc);
            query_builder.push(", ");
            query_builder.push_bind(symbol_id);

            for value in &indicator_values {
                query_builder.push(", ");
                match value {
                    FeatureValue::Float(f) => query_builder.push_bind(f),
                    FeatureValue::Json(j) => query_builder.push_bind(j),
                };
            }

            query_builder.push(") ON CONFLICT (symbol_id, time) DO UPDATE SET ");
            
            let mut first_set_item = true;
            for indicator_name in &indicator_names {
                if !first_set_item {
                    query_builder.push(", ");
                }
                query_builder.push(format!("{} = EXCLUDED.{}", indicator_name, indicator_name));
                first_set_item = false;
            }

            query_builder.push(", updated_at_ms = EXCLUDED.time_ms, updated_at = NOW()");
            
            if let Err(e) = query_builder.build().execute(&self.db_pool).await {
                eprintln!("Failed to insert indicators for {}: {}", symbol, e);
            }
        }

        Ok(())
    }

    pub async fn queue_record(&self, record: IndicatorRecord) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut queue = self.persist_queue.write().await;
        queue.push(record);
        Ok(())
    }

    pub async fn queue_records(&self, records: Vec<IndicatorRecord>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut queue = self.persist_queue.write().await;
        queue.extend(records);
        Ok(())
    }
}
