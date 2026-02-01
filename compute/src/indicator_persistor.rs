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

        // Write records to the database
        let mut query_builder = sqlx::QueryBuilder::new(
            "INSERT INTO market.raw_signals (time_ms, symbol_id, tf_minutes, indicator_id, signal_kind, side, score, value, details) "
        );

        let mut records_with_strings: Vec<(i64, String, i64, i32, i32, i32, f64, f64, Value)> = Vec::new();
        for rec in records_to_flush {
            records_with_strings.push((
                rec.timestamp,
                rec.symbol.as_str().to_string(), // Convert to owned string
                rec.timeframe.to_minutes().into(),
                1, // placeholder indicator_id
                1, // placeholder signal_kind
                0, // neutral side
                0.0, // score (will be calculated later)
                rec.value,
                json!({"name": rec.indicator_name}),
            ));
        }

        query_builder.push_values(records_with_strings, |mut b, (timestamp, symbol, tf_minutes, indicator_id, signal_kind, side, score, value, details)| {
            b.push_bind(timestamp);
            b.push_bind(symbol);
            b.push_bind(tf_minutes);
            b.push_bind(indicator_id);
            b.push_bind(signal_kind);
            b.push_bind(side);
            b.push_bind(score);
            b.push_bind(value);
            b.push_bind(details);
        });

        query_builder.build().execute(&self.db_pool).await?;

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