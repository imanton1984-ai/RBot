// compute/src/indicator_persistor.rs

use database_lib::PersistRecord;
use tokio::sync::mpsc;
use tracing::error;

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
    bulk_persistor_sender: mpsc::Sender<PersistRecord>,
}

impl IndicatorPersistor {
    pub fn new(
        bulk_persistor_sender: mpsc::Sender<PersistRecord>,
    ) -> (Self, mpsc::UnboundedSender<IndicatorRecord>) {
        let (sender, mut receiver) = mpsc::unbounded_channel::<IndicatorRecord>();
        
        let bulk_sender = bulk_persistor_sender.clone();
        
        // Forward incoming records to the bulk persistor
        tokio::spawn(async move {
            while let Some(record) = receiver.recv().await {
                let tf_minutes: i16 = record.timeframe.to_minutes() as i16;

                let (value_float, value_json) = match record.value {
                    FeatureValue::Float(v) => (Some(v), None),
                    FeatureValue::Json(v) => (None, Some(serde_json::to_value(v).unwrap_or_default())),
                };

                let persist_record = PersistRecord::Indicator {
                    symbol: record.symbol,
                    timeframe: tf_minutes,
                    time_ms: record.timestamp,
                    indicator_name: record.indicator_name.clone(),
                    value_float,
                    value_json,
                    candle_is_final: true, // Assuming true for now, adjust if needed
                    calc_source: 1,      // 1 for "compute"
                    event_time_ms: None, // Adjust if available
                };

                if let Err(e) = bulk_sender.send(persist_record).await {
                    error!("Failed to send indicator record to bulk persistor: {}", e);
                }
            }
        });

        (
            Self {
                bulk_persistor_sender,
            },
            sender,
        )
    }

    pub async fn queue_records(&self, records: Vec<IndicatorRecord>) {
        for record in records {
            let tf_minutes: i16 = record.timeframe.to_minutes() as i16;

            let (value_float, value_json) = match record.value {
                FeatureValue::Float(v) => (Some(v), None),
                FeatureValue::Json(v) => (None, Some(serde_json::to_value(v).unwrap_or_default())),
            };

            let persist_record = PersistRecord::Indicator {
                symbol: record.symbol,
                timeframe: tf_minutes,
                time_ms: record.timestamp,
                indicator_name: record.indicator_name.clone(),
                value_float,
                value_json,
                candle_is_final: true, // Assuming true for now
                calc_source: 1,      // 1 for "compute"
                event_time_ms: None,
            };

            if let Err(e) = self.bulk_persistor_sender.send(persist_record).await {
                error!("Failed to send indicator record to bulk persistor: {}", e);
            }
        }
    }
}