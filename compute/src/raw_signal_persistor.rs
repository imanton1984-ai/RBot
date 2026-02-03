use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use common::Symbol; // Убран unused import: Timeframe
use sqlx::PgPool;
use sqlx::Postgres;
use chrono::{Utc, TimeZone};
use std::collections::HashMap;
use crate::raw_signal_types::RawSignal;

pub struct RawSignalPersistor {
    db_pool: PgPool,
    persist_queue: Arc<RwLock<Vec<RawSignal>>>,
    batch_size: usize,
    flush_interval_ms: u64,
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
        };

        let queue = persistor.persist_queue.clone();
        tokio::spawn(async move {
            while let Some(record) = receiver.recv().await {
                let mut queue = queue.write().await;
                queue.push(record);
            }
        });

        (persistor, sender)
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
                    eprintln!("Error flushing raw signals batch: {}", e);
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

        let mut grouped_records: HashMap<Symbol, Vec<RawSignal>> = HashMap::new();
        for record in records_to_flush {
            grouped_records.entry(record.symbol.clone()).or_default().push(record);
        }

        for (symbol, signals) in grouped_records {
            let symbol_id_result = sqlx::query_scalar::<Postgres, i64>("SELECT symbol_id FROM market.pairs WHERE symbol = $1")
                .bind(symbol.as_str())
                .fetch_one(&self.db_pool)
                .await;

            let symbol_id = match symbol_id_result {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("Could not get symbol_id for {}: {}", symbol, e);
                    continue;
                }
            };
            
            // Insert each signal individually
            for signal in signals {
                let time_value = Utc.timestamp_millis_opt(signal.timestamp).single().unwrap_or_else(|| Utc::now());
                let query = sqlx::query("INSERT INTO market.raw_signals (time_ms, time, symbol_id, tf_minutes, indicator_id, signal_kind, side, score, value, details, created_at_ms, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)")
                .bind(signal.timestamp)
                .bind(time_value)
                .bind(symbol_id)
                .bind(signal.timeframe.to_minutes() as i16)
                .bind(signal.indicator_id)
                .bind(signal.signal_kind)
                .bind(signal.side)
                .bind(signal.score)
                .bind(signal.value)
                .bind(signal.details)
                .bind(Utc::now().timestamp_millis())
                .bind(Utc::now());

                if let Err(e) = query.execute(&self.db_pool).await {
                    eprintln!("Failed to insert raw signal for {}: {}", symbol, e);
                }
            }
        }

        Ok(())
    }
}
