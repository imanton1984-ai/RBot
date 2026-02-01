use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use common::{Symbol, Timeframe};
use sqlx::PgPool;
use serde_json::{json, Value};

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
    // persist_receiver: Arc<RwLock<mpsc::UnboundedReceiver<IndicatorRecord>>>,
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
        loop {
            // Check if we have enough records to batch write
            {
                let queue = self.persist_queue.read().await;
                if queue.len() >= self.batch_size {
                    drop(queue); // Release the read lock before acquiring write lock
                    self.flush_batch().await?;
                }
            }

            // Also flush periodically even if we don't have a full batch
            tokio::time::sleep(tokio::time::Duration::from_millis(self.flush_interval_ms)).await;
        }
    }

    async fn flush_batch(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let records_to_flush = {
            let mut queue = self.persist_queue.write().await;
            let records_to_flush = queue.drain(..).collect::<Vec<_>>();
            records_to_flush
        };

        if records_to_flush.is_empty() {
            return Ok(());
        }

        // Group records by symbol, timeframe, and timestamp for aggregation
        use std::collections::HashMap;
        let mut grouped_records: HashMap<(Symbol, Timeframe, i64), std::collections::HashMap<String, f64>> = HashMap::new();

        for record in records_to_flush {
            let key = (record.symbol.clone(), record.timeframe, record.timestamp);
            grouped_records.entry(key).or_insert_with(std::collections::HashMap::new)
                .insert(record.indicator_name, record.value);
        }

        // Process each unique (symbol, timeframe, timestamp) combination
        for ((symbol, timeframe, timestamp), indicators_map) in grouped_records {
            // Get the symbol_id from the pairs table
            let symbol_row = sqlx::query!("SELECT symbol_id FROM market.pairs WHERE symbol = $1", symbol.as_str())
                .fetch_one(&self.db_pool)
                .await?;
            let symbol_id = symbol_row.symbol_id;

            // Determine the correct table based on timeframe
            let table_name = format!("market.indicators_{}", timeframe.as_str());

            // Build the query with all indicator columns that we have values for
            let mut query_builder = sqlx::QueryBuilder::new(format!("INSERT INTO {} (time_ms, symbol_id", table_name));

            // Add indicator columns based on what's available
            let mut indicator_names = Vec::new();
            let mut indicator_values = Vec::new();
            for (indicator_name, value) in &indicators_map {
                indicator_names.push(indicator_name.as_str());
                indicator_values.push(*value);
            }

            for indicator_name in &indicator_names {
                query_builder.push(format!(", {}", indicator_name));
            }

            query_builder.push(") VALUES (");
            query_builder.push_bind(timestamp);
            query_builder.push_bind(symbol_id);

            // Add the indicator values to the VALUES clause
            for value in &indicator_values {
                query_builder.push_bind(*value);
            }

            query_builder.push(") ON CONFLICT (time_ms, symbol_id) DO UPDATE SET ");

            // Update each indicator column that we have
            let mut first_set_item = true;
            for indicator_name in &indicator_names {
                if !first_set_item {
                    query_builder.push(", ");
                }
                query_builder.push(format!("{} = EXCLUDED.{}", indicator_name, indicator_name));
                first_set_item = false;
            }

            query_builder.push(", updated_at_ms = EXCLUDED.updated_at_ms, updated_at = NOW()");

            query_builder.build().execute(&self.db_pool).await?;
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