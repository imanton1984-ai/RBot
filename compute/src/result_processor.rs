use std::sync::Arc;
use tokio::sync::mpsc;
use crate::{
    FeatureWindow, FeatureColumn, IndicatorRecord, IndicatorPersistor,
    RawSignalPersistor, RawSignalProcessor, 
};
use crate::predictors::pipeline::FeatureSnapshot;
use crate::predictors::feature_view::IndicatorsWideRow;

/// Unified result processor for both history and realtime pipelines.
/// Accepts an optional second set of persistors for dual-mode operation (main.rs).
/// When only one persistor set is provided, all data uses those persistors.
pub struct ResultProcessor {
    indicator_persistor: Arc<IndicatorPersistor>,
    raw_signal_persistor: Arc<RawSignalPersistor>,
    // Optional: separate persistors for realtime mode
    indicator_persistor_rt: Option<Arc<IndicatorPersistor>>,
    raw_signal_persistor_rt: Option<Arc<RawSignalPersistor>>,
    raw_signal_processor: Arc<RawSignalProcessor>,
    feature_tx: mpsc::UnboundedSender<FeatureSnapshot>,
}

impl ResultProcessor {
    /// Create a single-mode ResultProcessor (used by compute_history.rs, compute_realtime.rs)
    pub fn new(
        indicator_persistor: Arc<IndicatorPersistor>,
        raw_signal_persistor: Arc<RawSignalPersistor>,
        raw_signal_processor: Arc<RawSignalProcessor>,
        feature_tx: mpsc::UnboundedSender<FeatureSnapshot>,
    ) -> Self {
        Self {
            indicator_persistor,
            raw_signal_persistor,
            indicator_persistor_rt: None,
            raw_signal_persistor_rt: None,
            raw_signal_processor,
            feature_tx,
        }
    }

    /// Create a dual-mode ResultProcessor (used by main.rs that handles both history + realtime)
    pub fn new_dual(
        indicator_persistor_hist: Arc<IndicatorPersistor>,
        raw_signal_persistor_hist: Arc<RawSignalPersistor>,
        indicator_persistor_rt: Arc<IndicatorPersistor>,
        raw_signal_persistor_rt: Arc<RawSignalPersistor>,
        raw_signal_processor: Arc<RawSignalProcessor>,
        feature_tx: mpsc::UnboundedSender<FeatureSnapshot>,
    ) -> Self {
        Self {
            indicator_persistor: indicator_persistor_hist,
            raw_signal_persistor: raw_signal_persistor_hist,
            indicator_persistor_rt: Some(indicator_persistor_rt),
            raw_signal_persistor_rt: Some(raw_signal_persistor_rt),
            raw_signal_processor,
            feature_tx,
        }
    }

    /// Select the appropriate indicator persistor based on realtime flag
    fn select_indicator_persistor(&self, is_realtime: bool) -> &Arc<IndicatorPersistor> {
        if is_realtime {
            self.indicator_persistor_rt.as_ref().unwrap_or(&self.indicator_persistor)
        } else {
            &self.indicator_persistor
        }
    }

    /// Select the appropriate raw signal persistor based on realtime flag
    fn select_raw_signal_persistor(&self, is_realtime: bool) -> &Arc<RawSignalPersistor> {
        if is_realtime {
            self.raw_signal_persistor_rt.as_ref().unwrap_or(&self.raw_signal_persistor)
        } else {
            &self.raw_signal_persistor
        }
    }

    pub async fn run(self, mut rx: mpsc::UnboundedReceiver<Arc<FeatureWindow>>) {
        tracing::info!("Result processor started");
        
        let mut total_indicators_persisted: u64 = 0;
        let mut total_raw_signals_persisted: u64 = 0;
        let mut total_batches_processed: u64 = 0;

        while let Some(feature_window) = rx.recv().await {
            let n = feature_window.batch.timestamps.len();
            if n == 0 { continue; }
            
            total_batches_processed += 1;

            println!(
                "Processing feature batch for {} on {}, bars={}, cols={}, realtime: {}",
                feature_window.symbol,
                feature_window.timeframe,
                n,
                feature_window.batch.columns.len(),
                feature_window.is_realtime
            );
            
            tracing::debug!(
                target: "compute_history",
                "Batch: {} {} bars={} realtime={}",
                feature_window.symbol,
                feature_window.timeframe,
                n,
                feature_window.is_realtime
            );

            // 1. Determine start index (skip warmup for history)
            let start_idx = if feature_window.is_realtime && n > 2 {
                n - 2
            } else {
                // For history: find first bar with enough valid data (>= 50% of indicators finite)
                let f64_columns: Vec<&Vec<f64>> = feature_window.batch.columns.iter()
                    .filter_map(|c| match c {
                        FeatureColumn::F64 { name, values } => {
                            let ignored: [&str; 6] = ["open", "high", "low", "close", "volume", "time_ms"];
                            if ignored.contains(&name.as_str()) { None } else { Some(values) }
                        }
                        _ => None,
                    })
                    .collect();
                let total_cols = f64_columns.len();
                let threshold = (total_cols as f64 * 0.5).ceil() as usize;

                let mut effective_start = 0;
                for bar_idx in 0..n {
                    let valid_count = f64_columns.iter()
                        .filter(|col| col.get(bar_idx).map_or(false, |v| v.is_finite()))
                        .count();
                    if valid_count >= threshold {
                        effective_start = bar_idx;
                        break;
                    }
                    if bar_idx == n - 1 {
                        effective_start = n; // Will skip all bars
                    }
                }
                effective_start
            };

            if start_idx >= n {
                println!(
                    "Skipping feature batch for {} on {} - no bars with enough valid indicators",
                    feature_window.symbol, feature_window.timeframe
                );
                continue;
            }

            // Select the appropriate persistors based on is_realtime
            let selected_indicator_persistor = self.select_indicator_persistor(feature_window.is_realtime);
            let selected_raw_signal_persistor = self.select_raw_signal_persistor(feature_window.is_realtime);

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
                            if ignored_names.contains(&name.as_str()) { continue; }
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
                println!(
                    "  Persisting {} indicator records for {} on {} (bars {}..{}, skipped {} warmup bars)",
                    records.len(),
                    feature_window.symbol,
                    feature_window.timeframe,
                    start_idx,
                    n - 1,
                    start_idx
                );
                total_indicators_persisted += records.len() as u64;
                selected_indicator_persistor.queue_records(records).await;
            }

            // 3. Process & Persist Raw Signals
            let all_raw_signals = self.raw_signal_processor.process_feature_window(&feature_window);
            let all_raw_signals_count = all_raw_signals.len();
            
            let min_valid_timestamp = if start_idx < n {
                feature_window.batch.timestamps[start_idx]
            } else {
                i64::MAX
            };

            let signals_to_persist = if feature_window.is_realtime {
                let last_timestamps: Vec<i64> = feature_window.batch.timestamps.iter().rev().take(2).cloned().collect();
                all_raw_signals.into_iter()
                    .filter(|s| last_timestamps.contains(&s.timestamp))
                    .collect::<Vec<_>>()
            } else {
                all_raw_signals.into_iter()
                    .filter(|s| s.timestamp >= min_valid_timestamp)
                    .collect::<Vec<_>>()
            };

            if !signals_to_persist.is_empty() {
                println!(
                    "  Persisting {} raw signals for {} on {} (filtered from {} total)",
                    signals_to_persist.len(),
                    feature_window.symbol,
                    feature_window.timeframe,
                    all_raw_signals_count
                );
                total_raw_signals_persisted += signals_to_persist.len() as u64;
                selected_raw_signal_persistor.queue_records(signals_to_persist).await;
            }

            // 4. Send Snapshot to Predictors Pipeline
            // For history: send ALL valid candles. For realtime: only last.
            let snapshot_range = if feature_window.is_realtime {
                if n > 0 { (n - 1)..n } else { 0..0 }
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
                    raw_signals_data: None,
                    sr_levels,
                    is_realtime: feature_window.is_realtime,
                };

                if let Err(e) = self.feature_tx.send(snapshot) {
                    // Channel closed is expected in super_entry mode (receiver dropped intentionally).
                    // Silently skip sending snapshots — they're not needed for super_entry backfill.
                    tracing::debug!("Feature snapshot channel closed (expected in super_entry mode): {}", e);
                    break;
                }
            }
        }
        
        // Final summary
        tracing::info!(
            "Result processor complete: {} batches, {} indicators, {} raw signals",
            total_batches_processed,
            total_indicators_persisted,
            total_raw_signals_persisted
        );
    }
}
