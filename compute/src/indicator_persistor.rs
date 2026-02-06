// compute/src/indicator_persistor.rs

use database_lib::PersistRecord;
use tokio::sync::mpsc;
use tracing::error;

use common::{Symbol, Timeframe};

use crate::FeatureValue;
use std::collections::HashMap;

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

        // Group incoming records by (symbol, timeframe, timestamp) to create wide format records
        tokio::spawn(async move {
            let mut grouped_records: HashMap<(Symbol, Timeframe, i64), (bool, i16, Option<i64>, HashMap<String, f64>, HashMap<String, serde_json::Value>)> = HashMap::new();

            loop {
                tokio::select! {
                    record_opt = receiver.recv() => {
                        if let Some(record) = record_opt {
                            let key = (record.symbol.clone(), record.timeframe, record.timestamp);
                            
                            // Insert or update the grouped record
                            match grouped_records.entry(key) {
                                std::collections::hash_map::Entry::Occupied(mut entry) => {
                                    let (_, _, _, ref mut indicators, ref mut json_data) = entry.get_mut();
                                    
                                    match record.value {
                                        FeatureValue::Float(v) => {
                                            indicators.insert(record.indicator_name, v);
                                        }
                                        FeatureValue::Json(v) => {
                                            json_data.insert(record.indicator_name, serde_json::to_value(v).unwrap_or_default());
                                        }
                                    }
                                }
                                std::collections::hash_map::Entry::Vacant(entry) => {
                                    let mut indicators = HashMap::new();
                                    let mut json_data = HashMap::new();
                                    
                                    match record.value {
                                        FeatureValue::Float(v) => {
                                            indicators.insert(record.indicator_name, v);
                                        }
                                        FeatureValue::Json(v) => {
                                            json_data.insert(record.indicator_name, serde_json::to_value(v).unwrap_or_default());
                                        }
                                    }
                                    
                                    entry.insert((true, 1, None, indicators, json_data));
                                }
                            }
                        } else {
                            // Channel closed, send remaining records
                            break;
                        }
                    }
                    // Flush periodically to prevent holding data too long
                    _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                        flush_grouped_records(&bulk_sender, &mut grouped_records).await;
                    }
                }
            }
            
            // Send any remaining records
            flush_grouped_records(&bulk_sender, &mut grouped_records).await;
        });

        (
            Self {
                bulk_persistor_sender,
            },
            sender,
        )
    }

    pub async fn queue_records(&self, records: Vec<IndicatorRecord>) {
        // Group records by (symbol, timeframe, timestamp) to create wide format records
        let mut grouped: HashMap<(Symbol, Timeframe, i64), (bool, i16, Option<i64>, HashMap<String, f64>, HashMap<String, serde_json::Value>)> = HashMap::new();
        
        for record in records {
            let key = (record.symbol.clone(), record.timeframe, record.timestamp);
            
            match grouped.entry(key) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let (_, _, _, ref mut indicators, ref mut json_data) = entry.get_mut();
                    
                    match record.value {
                        FeatureValue::Float(v) => {
                            indicators.insert(record.indicator_name, v);
                        }
                        FeatureValue::Json(v) => {
                            json_data.insert(record.indicator_name, serde_json::to_value(v).unwrap_or_default());
                        }
                    }
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let mut indicators = HashMap::new();
                    let mut json_data = HashMap::new();
                    
                    match record.value {
                        FeatureValue::Float(v) => {
                            indicators.insert(record.indicator_name, v);
                        }
                        FeatureValue::Json(v) => {
                            json_data.insert(record.indicator_name, serde_json::to_value(v).unwrap_or_default());
                        }
                    }
                    
                    entry.insert((true, 1, None, indicators, json_data));
                }
            }
        }

        // Send grouped records to bulk persistor
        for ((symbol, timeframe, time_ms), (candle_is_final, calc_source, event_time_ms, indicators, json_data)) in grouped {
            let tf_minutes: i16 = timeframe.to_minutes() as i16;
            
            let persist_record = PersistRecord::IndicatorsWide {
                symbol,
                timeframe: tf_minutes,
                time_ms,
                indicators,
                json_data,
                candle_is_final,
                calc_source,
                event_time_ms,
            };

            if let Err(e) = self.bulk_persistor_sender.send(persist_record).await {
                error!("Failed to send wide indicator record to bulk persistor: {}", e);
            }
        }
    }
}

async fn flush_grouped_records(
    bulk_sender: &mpsc::Sender<PersistRecord>,
    grouped_records: &mut HashMap<(Symbol, Timeframe, i64), (bool, i16, Option<i64>, HashMap<String, f64>, HashMap<String, serde_json::Value>)>
) {
    for ((symbol, timeframe, time_ms), (candle_is_final, calc_source, event_time_ms, indicators, json_data)) in grouped_records.drain() {
        let tf_minutes: i16 = timeframe.to_minutes() as i16;
        
        let persist_record = PersistRecord::IndicatorsWide {
            symbol,
            timeframe: tf_minutes,
            time_ms,
            indicators,
            json_data,
            candle_is_final,
            calc_source,
            event_time_ms,
        };

        if let Err(e) = bulk_sender.send(persist_record).await {
            error!("Failed to send wide indicator record to bulk persistor: {}", e);
        }
    }
}