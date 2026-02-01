use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use common::{Symbol, Timeframe};
use sqlx::PgPool;
use chrono::{DateTime, Utc, TimeZone};
use std::collections::HashMap; // Explicitly add this import

#[derive(Debug, Clone)]
pub struct IndicatorRecord {
    pub symbol: Symbol,
    pub timeframe: Timeframe,
    pub timestamp: i64,
    pub indicator_name: String,
    pub value: f64,
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
                // Проверяем, есть ли что сбрасывать, внутри flush_batch, но блокировку на чтение сняли
                if let Err(e) = self.flush_batch().await {
                    eprintln!("Error flushing indicators batch: {}", e);
                }
                last_flush = tokio::time::Instant::now();
            }

            // Короткий сон, чтобы не грузить CPU в цикле
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

        // Group records by symbol, timeframe, and timestamp for aggregation
        // Key: (Symbol, Timeframe, timestamp) -> Value: Map<IndicatorName, Value>
        let mut grouped_records: HashMap<(Symbol, Timeframe, i64), HashMap<String, f64>> = HashMap::new();
        
        for record in records_to_flush {
            let key = (record.symbol.clone(), record.timeframe, record.timestamp);
            grouped_records.entry(key).or_insert_with(HashMap::new)
                .insert(record.indicator_name, record.value);
        }

        // Process each unique (symbol, timeframe, timestamp) combination
        for ((symbol, timeframe, timestamp), indicators_map) in grouped_records {
            // Get the symbol_id from the pairs table
            // OPTIMIZATION: In prod, cache symbol_ids in memory to avoid SELECT on every flush
            let symbol_row = sqlx::query!("SELECT symbol_id FROM market.pairs WHERE symbol = $1", symbol.as_str())
                .fetch_optional(&self.db_pool)
                .await?;
            
            let symbol_id = match symbol_row {
                Some(r) => r.symbol_id,
                None => {
                    eprintln!("Unknown symbol for indicator persist: {}", symbol);
                    continue; 
                }
            };

            // Determine the correct table based on timeframe
            let table_name = format!("market.indicators_{}", timeframe.as_str());
            
            // Convert timestamp to DateTime<Utc> for the 'time' column (Required for Timescale PK)
            let time_utc = Utc.timestamp_millis_opt(timestamp).single().unwrap_or(Utc::now());

            // Build the query
            // We use 'time' column for ON CONFLICT because it's part of the Hypertable PK
            let mut query_builder = sqlx::QueryBuilder::new(format!("INSERT INTO {} (time_ms, time, symbol_id", table_name));

            let mut indicator_names = Vec::new();
            let mut indicator_values = Vec::new();
            
            for (indicator_name, value) in &indicators_map {
                // Basic protection against SQL injection via column names (although they come from internal enums usually)
                if !indicator_name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    continue;
                }
                indicator_names.push(indicator_name.as_str());
                indicator_values.push(*value);
            }

            if indicator_names.is_empty() {
                continue;
            }

            for indicator_name in &indicator_names {
                query_builder.push(format!(", {}", indicator_name));
            }

            query_builder.push(") VALUES (");
            query_builder.push_bind(timestamp);
            query_builder.push_bind(time_utc); // Explicitly bind the timestamptz
            query_builder.push_bind(symbol_id);

            for value in &indicator_values {
                query_builder.push_bind(*value);
            }

            // TimescaleDB PK is (symbol_id, time)
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
