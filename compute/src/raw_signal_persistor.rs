// compute/src/raw_signal_persistor.rs

use database_lib::PersistRecord;
use tokio::sync::mpsc;
use tracing::error;

use crate::raw_signal_types::RawSignal;

pub struct RawSignalPersistor {
    bulk_persistor_sender: mpsc::Sender<PersistRecord>,
}

impl RawSignalPersistor {
    pub fn new(
        bulk_persistor_sender: mpsc::Sender<PersistRecord>,
    ) -> Self {  // Simplified return type since we're not using aggregation
        Self {
            bulk_persistor_sender,
        }
    }

    pub async fn queue_records(&self, records: Vec<RawSignal>) {
        // We send individual signals directly to the database as RawSignal records
        // to be stored in market.raw_signals table
        for signal in records {
            let tf_minutes = signal.timeframe.to_minutes() as i16;
            
            let persist_record = PersistRecord::RawSignal {
                symbol: signal.symbol,
                timeframe: tf_minutes,
                time_ms: signal.timestamp,
                
                indicator_id: signal.indicator_id,
                signal_kind: signal.signal_kind,
                signal_sub_id: signal.signal_sub_id,
                
                side: signal.side,
                score: signal.score,
                value: signal.value,
                
                details: signal.details,
                candle_is_final: signal.candle_is_final,
                calc_source: signal.calc_source,
                event_time_ms: signal.event_time_ms,
                
                // These fields are optional and can be taken from the signal
                features_json: signal.features_json, 
                scores_json: signal.scores_json,
                predictions_json: signal.predictions_json,
            };

            if let Err(e) = self.bulk_persistor_sender.send(persist_record).await {
                error!("Failed to send raw signal: {}", e);
            }
        }
    }
}