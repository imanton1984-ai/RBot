// compute/src/super_entry_stage.rs
//
// Super Entry Stage - integrates Super Entry ML inference into compute_realtime
//
// For REALTIME: Consumes FeatureSnapshot from feature_tx channel, buffers candles
// per (symbol, tf), and runs inference when enough data is accumulated.
//
// For HISTORY: Use run_super_entry_backfill() in compute_history.rs instead —
// it reads directly from DB after indicators are written (much more reliable).

use std::str::FromStr;
use std::sync::Arc;
use anyhow::Result;
use sqlx::PgPool;
use tokio::sync::mpsc;
use tracing::{info, warn, error, debug};

use common::TimeFrame;
use ml_entry_strategy::{
    SuperEntryConfig,
    SuperEntryPipeline,
    dataset::CandleWithIndicators,
};
use crate::predictors::pipeline::FeatureSnapshot;

/// Parse timeframe string to minutes. Returns None on failure.
fn parse_tf_minutes(tf_str: &str) -> Option<i32> {
    TimeFrame::from_str(tf_str)
        .ok()
        .map(|tf| tf.to_minutes())
}

/// Super Entry Stage - consumes feature snapshots and generates signals (REALTIME)
pub struct SuperEntryStage {
    pipeline: Arc<SuperEntryPipeline>,
    db_pool: PgPool,
    feature_rx: mpsc::UnboundedReceiver<FeatureSnapshot>,
    use_gpu: bool,
}

impl SuperEntryStage {
    /// Create a new Super Entry stage
    pub fn new(
        config: SuperEntryConfig,
        db_pool: PgPool,
        feature_rx: mpsc::UnboundedReceiver<FeatureSnapshot>,
        use_gpu: bool,
    ) -> Result<Self> {
        let pipeline = Arc::new(SuperEntryPipeline::new(config, use_gpu)?);
        
        Ok(Self {
            pipeline,
            db_pool,
            feature_rx,
            use_gpu,
        })
    }

    /// Check if models are available
    pub fn has_models(&self) -> bool {
        self.pipeline.has_models()
    }

    /// Run the stage - consumes feature snapshots and generates signals.
    ///
    /// For realtime: processes single candles via process_single().
    /// For history snapshots that arrive through the channel: buffers and batch-processes.
    pub async fn run(mut self) -> Result<()> {
        info!(target: "super_entry_stage", "Super Entry Stage started (realtime mode)");
        
        // Buffer candles per (symbol, tf_minutes) for batch processing
        let mut candle_buffers: std::collections::HashMap<(String, i32), Vec<CandleWithIndicators>> = 
            std::collections::HashMap::new();
        
        let warmup = self.pipeline.config().warmup_bars;
        let max_buffer_size = 1000;
        let mut total_signals_generated = 0u64;

        while let Some(snapshot) = self.feature_rx.recv().await {
            // Parse timeframe string to minutes
            let tf_minutes = match parse_tf_minutes(&snapshot.timeframe) {
                Some(tf) => tf,
                None => {
                    warn!(target: "super_entry_stage", 
                        "Failed to parse timeframe '{}' for {}, skipping",
                        snapshot.timeframe, snapshot.symbol);
                    continue;
                }
            };

            // Check if we have a model for this TF
            if !self.pipeline.has_models() {
                continue;
            }

            // Convert snapshot to CandleWithIndicators
            let candle = snapshot_to_candle(&snapshot);
            
            if snapshot.is_realtime {
                // REALTIME: use process_single for immediate inference
                match self.pipeline.process_single(&candle, tf_minutes, self.use_gpu) {
                    Ok(result) => {
                        if let Some(signal) = result.signal {
                            debug!(target: "super_entry_stage",
                                "RT signal: {} {}m side={} p_super={:.3}",
                                snapshot.symbol, tf_minutes, signal.side, signal.p_super);
                            
                            if let Err(e) = ml_entry_strategy::db_writer::insert_signals_batch(
                                &self.db_pool, &[signal]
                            ).await {
                                error!(target: "super_entry_stage", "Failed to persist RT signal: {}", e);
                            } else {
                                total_signals_generated += 1;
                            }
                        }
                    }
                    Err(e) => {
                        warn!(target: "super_entry_stage", 
                            "process_single failed for {} {}m: {}", 
                            snapshot.symbol, tf_minutes, e);
                    }
                }
            } else {
                // HISTORY: buffer and batch-process
                let key = (snapshot.symbol.clone(), tf_minutes);
                let buffer = candle_buffers.entry(key).or_insert_with(Vec::new);
                buffer.push(candle);
                
                if buffer.len() > warmup + 10 || buffer.len() >= max_buffer_size {
                    let candles_to_process = buffer.clone();
                    let result = self.pipeline.process_candles(
                        &candles_to_process,
                        tf_minutes,
                        self.use_gpu,
                    );
                    
                    match result {
                        Ok(results) => {
                            let signals: Vec<_> = results.into_iter()
                                .filter_map(|r| r.signal)
                                .collect();
                            
                            if !signals.is_empty() {
                                let count = signals.len();
                                if let Err(e) = ml_entry_strategy::db_writer::insert_signals_batch(
                                    &self.db_pool, &signals
                                ).await {
                                    error!(target: "super_entry_stage", "Failed to persist {} signals: {}", count, e);
                                } else {
                                    total_signals_generated += count as u64;
                                    info!(target: "super_entry_stage", 
                                        "Batch: {} {}m → {} signals (total: {})",
                                        snapshot.symbol, tf_minutes, count, total_signals_generated);
                                }
                            }
                        }
                        Err(e) => {
                            warn!(target: "super_entry_stage", "process_candles failed for {} {}m: {}", 
                                snapshot.symbol, tf_minutes, e);
                        }
                    }
                    
                    // Keep only recent candles for next iteration
                    let retain_count = warmup.min(buffer.len());
                    buffer.drain(0..buffer.len() - retain_count);
                }
            }
        }
        
        info!(target: "super_entry_stage", 
            "Super Entry Stage stopped. Total signals generated: {}", total_signals_generated);
        Ok(())
    }
}

/// Convert FeatureSnapshot to CandleWithIndicators
fn snapshot_to_candle(snapshot: &FeatureSnapshot) -> CandleWithIndicators {
    CandleWithIndicators {
        time: snapshot.timestamp,
        symbol: snapshot.symbol.clone(),
        symbol_id: 0, // Will be populated from DB if needed
        open: snapshot.indicators.open as f64,
        high: snapshot.indicators.high as f64,
        low: snapshot.indicators.low as f64,
        close: snapshot.indicators.close as f64,
        volume: snapshot.indicators.volume as f64,
        rsi: snapshot.indicators.rsi as f64,
        cci: snapshot.indicators.cci as f64,
        stoch_k: snapshot.indicators.stoch_k as f64,
        stoch_d: snapshot.indicators.stoch_d as f64,
        williams: snapshot.indicators.williams_r as f64,
        macd: snapshot.indicators.macd_line as f64,
        macd_signal: snapshot.indicators.macd_signal as f64,
        macd_hist: snapshot.indicators.macd_histogram as f64,
        adx: snapshot.indicators.adx as f64,
        sma: snapshot.indicators.sma as f64,
        ema_20: snapshot.indicators.ema_20 as f64,
        ema_50: snapshot.indicators.ema_50 as f64,
        ema_200: snapshot.indicators.ema_200 as f64,
        bb_upper: snapshot.indicators.bb_upper as f64,
        bb_mid: snapshot.indicators.bb_middle as f64,
        bb_lower: snapshot.indicators.bb_lower as f64,
        atr: snapshot.indicators.atr as f64,
        obv: snapshot.indicators.obv as f64,
        vwap: snapshot.indicators.vwap as f64,
        volume_spike: snapshot.indicators.volume_spike as f64,
        trend: snapshot.indicators.trend_short as f64,
        trend_short: snapshot.indicators.trend_short as f64,
        poc: 0.0, // Not in snapshot
    }
}

/// Setup Super Entry stage for REALTIME mode.
/// For HISTORY mode, use run_super_entry_backfill() in compute_history.rs instead.
pub async fn setup_super_entry_stage(
    db_pool: &PgPool,
    feature_rx: mpsc::UnboundedReceiver<FeatureSnapshot>,
    use_cuda: bool,
) -> Option<tokio::task::JoinHandle<Result<()>>> {
    let strategy = std::env::var("ACTIVE_STRATEGY").unwrap_or_else(|_| "level".to_string());
    
    if strategy != "super_entry" && strategy != "combined" {
        info!(target: "super_entry_stage", "Skipping Super Entry stage (strategy: {})", strategy);
        return None;
    }
    
    // Check if Super Entry is explicitly enabled
    let super_entry_enabled = std::env::var("SUPER_ENTRY_ENABLED")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);
    
    if !super_entry_enabled && strategy != "super_entry" {
        info!(target: "super_entry_stage", "Super Entry not enabled for strategy {}", strategy);
        return None;
    }
    
    info!(target: "super_entry_stage", "Setting up Super Entry stage (strategy: {})", strategy);
    
    let config = SuperEntryConfig::from_env();
    let use_gpu = std::env::var("SUPER_ENTRY_USE_GPU")
        .unwrap_or_default()
        .parse::<bool>()
        .unwrap_or(false) || use_cuda;
    
    // Ensure table exists before starting
    if let Err(e) = ml_entry_strategy::db_writer::ensure_table_exists(db_pool).await {
        error!(target: "super_entry_stage", "Failed to ensure super_entry_signals table: {}", e);
        return None;
    }
    
    match SuperEntryStage::new(config, db_pool.clone(), feature_rx, use_gpu) {
        Ok(stage) => {
            if !stage.has_models() {
                error!(target: "super_entry_stage", "Super Entry models not found! Train first or disable SUPER_ENTRY_ENABLED");
                return None;
            }
            
            info!(target: "super_entry_stage", "✅ Super Entry stage initialized (GPU={})", use_gpu);
            Some(tokio::spawn(async move { stage.run().await }))
        }
        Err(e) => {
            error!(target: "super_entry_stage", "Failed to create Super Entry stage: {}", e);
            None
        }
    }
}
