use std::sync::Arc;
use tokio::sync::mpsc;
use crate::{
    FeatureWindow, FeatureColumn, IndicatorRecord, IndicatorPersistor,
    RawSignalPersistor, RawSignalProcessor, 
};
use crate::predictors::pipeline::FeatureSnapshot;
use crate::predictors::feature_view::IndicatorsWideRow;

pub struct ResultProcessor {
    indicator_persistor: Arc<IndicatorPersistor>,
    raw_signal_persistor: Arc<RawSignalPersistor>,
    raw_signal_processor: Arc<RawSignalProcessor>,
    feature_tx: mpsc::UnboundedSender<FeatureSnapshot>,
}

impl ResultProcessor {
    pub fn new(
        indicator_persistor: Arc<IndicatorPersistor>,
        raw_signal_persistor: Arc<RawSignalPersistor>,
        raw_signal_processor: Arc<RawSignalProcessor>,
        feature_tx: mpsc::UnboundedSender<FeatureSnapshot>,
    ) -> Self {
        Self {
            indicator_persistor,
            raw_signal_persistor,
            raw_signal_processor,
            feature_tx,
        }
    }

    pub async fn run(self, mut rx: mpsc::UnboundedReceiver<Arc<FeatureWindow>>) {
        tracing::info!("Result processor started");
        
        while let Some(feature_window) = rx.recv().await {
            let n = feature_window.batch.timestamps.len();
            if n == 0 { continue; }

            // 1. Determine start index (skip warmup for history)
            let start_idx = if feature_window.is_realtime && n > 2 {
                n - 2
            } else {
                // For history: find first bar with enough valid data
                let mut effective_start = 0;
                // Simplified check: skip first few bars if they are obviously warmup
                // In production, checking for valid indicator values (non-NaN) is better
                if n > 50 { effective_start = 50; } 
                effective_start
            };

            if start_idx >= n { continue; }

            // 2. Persist Indicators
            let mut records = Vec::new();
            let ignored_names = ["open", "high", "low", "close", "volume", "time_ms"];
            
            for (i, &timestamp) in feature_window.batch.timestamps.iter().enumerate().skip(start_idx) {
                for column in &feature_window.batch.columns {
                    match column {
                        FeatureColumn::F64 { name, values } => {
                            if ignored_names.contains(&name.as_str()) { continue; }
                            if let Some(value) = values.get(i) {
                                if value.is_finite() {
                                    records.push(IndicatorRecord {
                                        symbol: feature_window.symbol.clone(),
                                        timeframe: feature_window.timeframe,
                                        timestamp,
                                        indicator_name: name.clone(),
                                        value: crate::FeatureValue::Float(*value),
                                    });
                                }
                            }
                        }
                        FeatureColumn::Json { name, values } => {
                            if let Some(value) = values.get(i) {
                                if !value.is_null() {
                                    records.push(IndicatorRecord {
                                        symbol: feature_window.symbol.clone(),
                                        timeframe: feature_window.timeframe,
                                        timestamp,
                                        indicator_name: name.clone(),
                                        value: crate::FeatureValue::Json(value.clone()),
                                    });
                                }
                            }
                        }
                    }
                }
            }

            if !records.is_empty() {
                self.indicator_persistor.queue_records(records).await;
            }

            // 3. Process & Persist Raw Signals
            let all_raw_signals = self.raw_signal_processor.process_feature_window(&feature_window);
            
            let min_valid_timestamp = feature_window.batch.timestamps[start_idx];
            let signals_to_persist: Vec<_> = all_raw_signals.into_iter()
                .filter(|s| s.timestamp >= min_valid_timestamp)
                .collect();

            if !signals_to_persist.is_empty() {
                self.raw_signal_persistor.queue_records(signals_to_persist).await;
            }

            // 4. Send Snapshot to Predictors Pipeline
            // For history: send ALL valid candles. For realtime: only last.
            let snapshot_range = if feature_window.is_realtime {
                (n - 1)..n
            } else {
                start_idx..n
            };

            let cw = feature_window.candle_window.as_ref().expect("Candle window missing");

            for idx in snapshot_range {
                // Helper to extract f32
                let get_f32 = |name: &str, default: f32| -> f32 {
                    feature_window.batch.get_f64(name)
                        .and_then(|v| v.get(idx))
                        .map(|&v| v as f32)
                        .unwrap_or(default)
                };

                let indicators = IndicatorsWideRow {
                    close: cw.close.get(idx).copied().unwrap_or(0.0) as f32,
                    high: cw.high.get(idx).copied().unwrap_or(0.0) as f32,
                    low: cw.low.get(idx).copied().unwrap_or(0.0) as f32,
                    open: cw.open.get(idx).copied().unwrap_or(0.0) as f32,
                    volume: cw.volume.get(idx).copied().unwrap_or(0.0) as f32,
                    
                    rsi: get_f32("rsi", 50.0),
                    macd_line: get_f32("macd", 0.0),
                    macd_signal: get_f32("macd_signal", 0.0),
                    macd_histogram: get_f32("macd_hist", 0.0),
                    ema_20: get_f32("ema_20", 0.0),
                    ema_50: get_f32("ema_50", 0.0),
                    ema_200: get_f32("ema_200", 0.0),
                    sma: get_f32("sma", 0.0),
                    bb_upper: get_f32("bb_upper", 0.0),
                    bb_lower: get_f32("bb_lower", 0.0),
                    bb_middle: get_f32("bb_mid", 0.0),
                    atr: get_f32("atr", 0.0),
                    adx: get_f32("adx", 0.0),
                    vwap: get_f32("vwap", 0.0),
                    obv: get_f32("obv", 0.0),
                    cci: get_f32("cci", 0.0),
                    stoch_k: get_f32("stoch_k", 50.0),
                    stoch_d: get_f32("stoch_d", 50.0),
                    williams_r: get_f32("williams", -50.0),
                    trend_short: get_f32("trend_short", 0.0),
                    trend_medium: get_f32("trend", 0.0),
                    trend_long: get_f32("trend_long", 0.0),
                    volume_sma: get_f32("volume_sma", 0.0),
                    volume_spike: get_f32("volume_spike", 0.0),
                };

                let sr_levels = feature_window.batch.get_json("sr_levels")
                    .and_then(|v| v.get(idx))
                    .cloned();

                let ts_ms = feature_window.batch.timestamps[idx];
                let timestamp = chrono::DateTime::from_timestamp(ts_ms / 1000, ((ts_ms % 1000) * 1_000_000) as u32)
                    .unwrap_or(chrono::Utc::now());

                let snapshot = FeatureSnapshot {
                    timestamp,
                    symbol: feature_window.symbol.to_string(),
                    timeframe: feature_window.timeframe.as_str().to_string(),
                    indicators,
                    raw_signals_data: None, // Can be populated if needed
                    sr_levels,
                    is_realtime: feature_window.is_realtime,
                };

                if let Err(_) = self.feature_tx.send(snapshot) {
                    break;
                }
            }
        }
    }
}