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
    ) -> (Self, mpsc::UnboundedSender<RawSignal>) {
        let (sender, mut receiver) = mpsc::unbounded_channel::<RawSignal>();
        
        let bulk_sender = bulk_persistor_sender.clone();
        
        // Forward incoming signals to the bulk persistor
        tokio::spawn(async move {
            while let Some(signal) = receiver.recv().await {
                let tf_minutes: i16 = signal.timeframe.to_minutes() as i16;
                let persist_record = PersistRecord::RawSignal {
                    symbol: signal.symbol.clone(),
                    timeframe: tf_minutes,
                    time_ms: signal.timestamp,
                    indicator_id: signal.indicator_id,
                    signal_kind: signal.signal_kind,
                    signal_sub_id: signal.signal_sub_id,
                    side: signal.side,
                    score: signal.score,
                    value: signal.value,
                    details: signal.details.clone(),
                    candle_is_final: signal.candle_is_final,
                    calc_source: signal.calc_source,
                    event_time_ms: signal.event_time_ms,
                    features_json: signal.features_json.clone(),
                    scores_json: signal.scores_json.clone(),
                    predictions_json: signal.predictions_json.clone(),
                };
                
                if let Err(e) = bulk_sender.send(persist_record).await {
                    error!("Failed to send raw signal to bulk persistor: {}", e);
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
        for signal in records {
            let tf_minutes: i16 = signal.timeframe.to_minutes() as i16;
            let persist_record = PersistRecord::RawSignal {
                symbol: signal.symbol.clone(),
                timeframe: tf_minutes,
                time_ms: signal.timestamp,
                indicator_id: signal.indicator_id,
                signal_kind: signal.signal_kind,
                signal_sub_id: signal.signal_sub_id,
                side: signal.side,
                score: signal.score,
                value: signal.value,
                details: signal.details.clone(),
                candle_is_final: signal.candle_is_final,
                calc_source: signal.calc_source,
                event_time_ms: signal.event_time_ms,
                features_json: signal.features_json.clone(),
                scores_json: signal.scores_json.clone(),
                predictions_json: signal.predictions_json.clone(),
            };
            
            if let Err(e) = self.bulk_persistor_sender.send(persist_record).await {
                error!("Failed to send raw signal to bulk persistor: {}", e);
            }
        }
    }
}