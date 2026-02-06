// compute/src/raw_signal_persistor.rs

use database_lib::PersistRecord;
use tokio::sync::mpsc;
use tracing::error;

use crate::raw_signal_types::RawSignal;
use std::collections::HashMap;

// Struct to represent aggregated signals for a single candle
#[derive(Debug, Clone)]
pub struct AggregatedSignalRecord {
    pub symbol: common::Symbol,
    pub timeframe: common::Timeframe,
    pub time_ms: i64,
    pub signals: Vec<RawSignal>,
    pub features_json: Option<serde_json::Value>,
    pub scores_json: Option<serde_json::Value>,
    pub predictions_json: Option<serde_json::Value>,
}

pub struct RawSignalPersistor {
    bulk_persistor_sender: mpsc::Sender<PersistRecord>,
}

impl RawSignalPersistor {
    pub fn new(
        bulk_persistor_sender: mpsc::Sender<PersistRecord>,
    ) -> (Self, mpsc::UnboundedSender<AggregatedSignalRecord>) {
        let (sender, mut receiver) = mpsc::unbounded_channel::<AggregatedSignalRecord>();

        let bulk_sender = bulk_persistor_sender.clone();

        // Forward incoming aggregated signals to the bulk persistor
        tokio::spawn(async move {
            while let Some(aggregated_record) = receiver.recv().await {
                let tf_minutes: i16 = aggregated_record.timeframe.to_minutes() as i16;
                
                // Convert to the new aggregated signal format
                let signals_for_db: Vec<database_lib::AggregatedSignalItem> = aggregated_record.signals.into_iter()
                    .map(|signal| database_lib::AggregatedSignalItem {
                        indicator_id: signal.indicator_id,
                        signal_kind: signal.signal_kind,
                        side: signal.side,
                        score: signal.score,
                        value: signal.value,
                        details: signal.details,
                    })
                    .collect();

                let persist_record = PersistRecord::AggregatedSignal {
                    symbol: aggregated_record.symbol,
                    timeframe: tf_minutes,
                    time_ms: aggregated_record.time_ms,
                    signals: signals_for_db,
                    features_json: aggregated_record.features_json,
                    scores_json: aggregated_record.scores_json,
                    predictions_json: aggregated_record.predictions_json,
                    candle_is_final: true, // Default value
                    calc_source: 1,        // Default value
                    event_time_ms: None,   // Default value
                };

                if let Err(e) = bulk_sender.send(persist_record).await {
                    error!("Failed to send aggregated signal to bulk persistor: {}", e);
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

    pub async fn queue_records(&self, records: Vec<RawSignal>) {
        // Group signals by (symbol, timeframe, timestamp) to create aggregated records
        let mut grouped: HashMap<(common::Symbol, common::Timeframe, i64), Vec<RawSignal>> = HashMap::new();
        
        for signal in records {
            let key = (signal.symbol.clone(), signal.timeframe, signal.timestamp);
            grouped.entry(key).or_default().push(signal);
        }

        for ((symbol, timeframe, time_ms), signals) in grouped {
            // Take features, scores, and predictions from the first signal in the group
            let first_signal = signals.first().unwrap();
            
            // Clone the JSON fields from the first signal before moving the signals vector
            let features_json = first_signal.features_json.clone();
            let scores_json = first_signal.scores_json.clone();
            let predictions_json = first_signal.predictions_json.clone();
            
            let aggregated_record = AggregatedSignalRecord {
                symbol,
                timeframe,
                time_ms,
                signals,
                features_json,
                scores_json,
                predictions_json,
            };

            let tf_minutes: i16 = aggregated_record.timeframe.to_minutes() as i16;
            
            // Convert to the new aggregated signal format
            let signals_for_db: Vec<database_lib::AggregatedSignalItem> = aggregated_record.signals.into_iter()
                .map(|signal| database_lib::AggregatedSignalItem {
                    indicator_id: signal.indicator_id,
                    signal_kind: signal.signal_kind,
                    side: signal.side,
                    score: signal.score,
                    value: signal.value,
                    details: signal.details,
                })
                .collect();

            let persist_record = PersistRecord::AggregatedSignal {
                symbol: aggregated_record.symbol,
                timeframe: tf_minutes,
                time_ms: aggregated_record.time_ms,
                signals: signals_for_db,
                features_json: aggregated_record.features_json,
                scores_json: aggregated_record.scores_json,
                predictions_json: aggregated_record.predictions_json,
                candle_is_final: true, // Default value
                calc_source: 1,        // Default value
                event_time_ms: None,   // Default value
            };

            if let Err(e) = self.bulk_persistor_sender.send(persist_record).await {
                error!("Failed to send aggregated signal to bulk persistor: {}", e);
            }
        }
    }
}