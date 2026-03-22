// compute/src/super_entry_stage.rs
//
// Super Entry Stage — integrates Super Entry ML inference into both
// compute_realtime (single-candle) and compute_history (batch) pipelines.
//
// ZERO-COPY PATH (history):
//   GPU indicators → FeatureSnapshot channel → SuperEntryStage → XGBoost → DB
//   No DB roundtrip for indicator data — direct in-memory processing.
//
// REALTIME PATH:
//   Kafka close event → indicators → FeatureSnapshot → SuperEntryStage → DB
//
// CROSS-TF HEURISTIC FILTER (v2):
//   After ML inference generates a signal, apply cross-TF heuristic direction filter.
//   Uses weighted voting from current + higher + lower TF trend indicators.
//   Mode "filter" (default): reject signals where heuristic disagrees with ML direction.
//   This improves WR from ~63.9% to ~67.4% on 1h backtest.
//   Controlled by env vars: HEURISTIC_DIR_MODE, HEURISTIC_DIR_CONFIDENCE.

use std::collections::HashMap;
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
    heuristic::{
        CrossTfStore, HeuristicMode,
        apply_heuristic_filter, heuristic_min_confidence_from_env,
    },
};
// SuperEntryConfig::timeframes() used to filter production TFs in the run loop
use crate::predictors::pipeline::FeatureSnapshot;

/// Parse timeframe string to minutes. Returns None on failure.
fn parse_tf_minutes(tf_str: &str) -> Option<i32> {
    TimeFrame::from_str(tf_str)
        .ok()
        .map(|tf| tf.to_minutes())
}

/// Super Entry Stage — consumes feature snapshots and generates signals
/// Works in both HISTORY (batch) and REALTIME (single-candle) modes.
///
/// v2: Includes cross-TF heuristic direction filter.
/// Maintains a CrossTfStore updated on EVERY incoming candle (all TFs)
/// to support heuristic lookups for higher/lower TF trend confirmation.
pub struct SuperEntryStage {
    pipeline: Arc<SuperEntryPipeline>,
    config: SuperEntryConfig,
    db_pool: PgPool,
    feature_rx: mpsc::UnboundedReceiver<FeatureSnapshot>,
    use_gpu: bool,
    /// Cache: symbol name → symbol_id from market.pairs
    /// Prevents repeated DB lookups for the same symbol.
    symbol_id_cache: HashMap<String, i64>,
    /// Cross-TF store for heuristic direction filter.
    /// Updated on every incoming FeatureSnapshot (even non-active TFs).
    cross_tf_store: CrossTfStore,
    /// Heuristic filtering mode
    heuristic_mode: HeuristicMode,
    /// Minimum heuristic confidence to trigger filter/override
    heuristic_min_confidence: f32,
}

impl SuperEntryStage {
    /// Create a new Super Entry stage
    pub fn new(
        config: SuperEntryConfig,
        db_pool: PgPool,
        feature_rx: mpsc::UnboundedReceiver<FeatureSnapshot>,
        use_gpu: bool,
    ) -> Result<Self> {
        let pipeline = Arc::new(SuperEntryPipeline::new(config.clone(), use_gpu)?);
        let heuristic_mode = HeuristicMode::from_env();
        let heuristic_min_confidence = heuristic_min_confidence_from_env();

        info!(target: "super_entry_stage",
            "Heuristic filter: mode={:?}, min_confidence={:.2}",
            heuristic_mode, heuristic_min_confidence);

        Ok(Self {
            pipeline,
            config,
            db_pool,
            feature_rx,
            use_gpu,
            symbol_id_cache: HashMap::new(),
            cross_tf_store: CrossTfStore::new(),
            heuristic_mode,
            heuristic_min_confidence,
        })
    }

    /// Check if models are available
    pub fn has_models(&self) -> bool {
        self.pipeline.has_models()
    }

    /// Resolve symbol_id from cache or DB. Caches the result.
    async fn resolve_symbol_id(&mut self, symbol: &str) -> i64 {
        if let Some(&id) = self.symbol_id_cache.get(symbol) {
            return id;
        }
        // Query DB
        let id: i64 = sqlx::query_scalar(
            "SELECT symbol_id FROM market.pairs WHERE symbol = $1 LIMIT 1"
        )
        .bind(symbol)
        .fetch_optional(&self.db_pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0);

        if id > 0 {
            self.symbol_id_cache.insert(symbol.to_string(), id);
        } else {
            warn!(target: "super_entry_stage", "symbol_id not found for '{}', using 0", symbol);
        }
        id
    }

    /// Pre-load all symbol_ids from market.pairs at startup
    async fn preload_symbol_ids(&mut self) {
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT symbol, symbol_id FROM market.pairs WHERE is_active = true"
        )
        .fetch_all(&self.db_pool)
        .await
        .unwrap_or_default();

        for (symbol, id) in rows {
            self.symbol_id_cache.insert(symbol, id);
        }
        info!(target: "super_entry_stage", "Preloaded {} symbol_ids", self.symbol_id_cache.len());
    }

    /// Run the stage — consumes feature snapshots and generates signals.
    ///
    /// For realtime: buffers last 60 candles per (symbol,tf) for dynamic features,
    ///   prefilled from DB at startup to eliminate cold-start.
    ///   Then processes via process_single_with_context().
    /// For history: buffers candles per (symbol, tf) and batch-processes.
    /// When channel closes, flushes ALL remaining buffers.
    ///
    /// v2: Every incoming candle updates cross_tf_store (even non-active TFs).
    /// After signal generation, heuristic filter is applied before DB persistence.
    pub async fn run(mut self) -> Result<()> {
        info!(target: "super_entry_stage", "Super Entry Stage started (heuristic={:?})", self.heuristic_mode);
        
        // Pre-load symbol_ids to avoid per-candle DB lookups
        self.preload_symbol_ids().await;

        // Buffer candles per (symbol, tf_minutes) for batch processing
        let mut candle_buffers: HashMap<(String, i32), Vec<CandleWithIndicators>> =
            HashMap::new();
        
        // Rolling context buffer for realtime (last ~120 candles per symbol/tf)
        // Must be > max_dynamic_lookback (100, for BB squeeze percentile) + margin,
        // otherwise compute_dynamic_features() returns all zeros because t < max_lb.
        // See dataset.rs:200 and config.rs BB_SQUEEZE_LOOKBACK.
        let rt_context_size = 120;
        let mut rt_context: HashMap<(String, i32), Vec<CandleWithIndicators>> =
            HashMap::new();

        // ── PREFILL rt_context from database to avoid cold-start ──────
        // Without this, dynamic features (54 of 106) would be zeros until
        // enough candles accumulate (e.g., 50+ hours for 1h TF!).
        // Loads last `rt_context_size` candles per (symbol, tf) from DB.
        {
            let production_tfs = SuperEntryConfig::timeframes();
            let all_tfs: &[i32] = &[1, 5, 15, 60, 240, 1440];
            let mut total_loaded = 0usize;
            let mut total_pairs = 0usize;

            for &tf in all_tfs {
                let is_production_tf = production_tfs.contains(&tf);
                match ml_entry_strategy::dataset::fetch_all_candles_for_tf(
                    &self.db_pool, tf, rt_context_size
                ).await {
                    Ok(grouped) => {
                        for (symbol, candles) in grouped {
                            if candles.is_empty() { continue; }
                            // Update cross-TF store with the latest candle (all TFs)
                            if let Some(last) = candles.last() {
                                self.cross_tf_store.update(last.clone(), tf);
                            }
                            // Only fill rt_context for production TFs (where we run inference)
                            if is_production_tf {
                                total_pairs += 1;
                                rt_context.insert((symbol, tf), candles);
                            }
                            total_loaded += 1;
                        }
                    }
                    Err(e) => {
                        warn!(target: "super_entry_stage",
                            "Failed to prefill rt_context for TF {}m: {}", tf, e);
                    }
                }
            }

            info!(target: "super_entry_stage",
                "✅ Prefilled rt_context: {} symbol/tf pairs ({} total incl. cross-TF), context_size={}",
                total_pairs, total_loaded, rt_context_size);
        }

        let warmup = self.pipeline.config().warmup_bars;
        // Larger buffer for history mode — processes in bigger batches for throughput
        let max_buffer_size = 5000;
        let mut total_signals_generated = 0u64;
        let mut total_signals_filtered = 0u64;
        let mut total_candles_received = 0u64;
        let mut total_batches_processed = 0u64;

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

            // ── CROSS-TF STORE UPDATE ────────────────────────────────
            // Update cross-TF store with EVERY candle (even non-active TFs).
            // This ensures higher/lower TF data is available for heuristic lookups.
            // Must happen BEFORE the production_tfs filter below.
            let symbol_id = self.resolve_symbol_id(&snapshot.symbol).await;
            let candle = snapshot_to_candle(&snapshot, symbol_id);
            self.cross_tf_store.update(candle.clone(), tf_minutes);

            // Filter: only production timeframes from SUPER_ENTRY_TIMEFRAMES env var.
            // Default: 15m, 1h, 4h, 1d. Configurable via SUPER_ENTRY_TIMEFRAMES="15,60,240,1440"
            let production_tfs = SuperEntryConfig::timeframes();
            if !production_tfs.contains(&tf_minutes) {
                continue;
            }

            // Check if we have a model for this TF
            if !self.pipeline.has_models() {
                continue;
            }

            total_candles_received += 1;

            if snapshot.is_realtime {
                // REALTIME: maintain rolling context for dynamic features,
                // then use process_single_with_context for immediate inference
                let key = (snapshot.symbol.clone(), tf_minutes);
                let ctx = rt_context.entry(key).or_default();
                ctx.push(candle.clone());
                // Keep only last rt_context_size candles
                if ctx.len() > rt_context_size {
                    ctx.drain(0..ctx.len() - rt_context_size);
                }

                match self.pipeline.process_single_with_context(
                    Some(ctx.as_slice()),
                    &candle,
                    tf_minutes,
                    self.use_gpu,
                ) {
                    Ok(result) => {
                        if let Some(signal) = result.signal {
                            // ── HEURISTIC FILTER ──
                            match apply_heuristic_filter(
                                signal,
                                &candle,
                                tf_minutes,
                                &self.cross_tf_store,
                                self.heuristic_mode,
                                self.heuristic_min_confidence,
                                &self.config,
                            ) {
                                Some(filtered_signal) => {
                                    debug!(target: "super_entry_stage",
                                        "RT signal: {} {}m side={} p_super={:.3} (heuristic passed)",
                                        snapshot.symbol, tf_minutes,
                                        filtered_signal.side, filtered_signal.p_super);

                                    if let Err(e) = ml_entry_strategy::db_writer::insert_signals_batch(
                                        &self.db_pool, &[filtered_signal]
                                    ).await {
                                        error!(target: "super_entry_stage", "Failed to persist RT signal: {}", e);
                                    } else {
                                        total_signals_generated += 1;
                                    }
                                }
                                None => {
                                    total_signals_filtered += 1;
                                    debug!(target: "super_entry_stage",
                                        "RT signal REJECTED by heuristic: {} {}m",
                                        snapshot.symbol, tf_minutes);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!(target: "super_entry_stage",
                            "process_single_with_context failed for {} {}m: {}",
                            snapshot.symbol, tf_minutes, e);
                    }
                }
            } else {
                // HISTORY: buffer and batch-process (ZERO-COPY path)
                let key = (snapshot.symbol.clone(), tf_minutes);
                let buffer = candle_buffers.entry(key).or_default();
                buffer.push(candle);
                
                // Process when buffer is large enough (warmup + meaningful batch)
                if buffer.len() >= warmup + 50 || buffer.len() >= max_buffer_size {
                    let (generated, filtered) = self.flush_buffer(
                        &snapshot.symbol, tf_minutes, buffer
                    ).await;
                    total_signals_generated += generated as u64;
                    total_signals_filtered += filtered as u64;
                    total_batches_processed += 1;
                    
                    if total_batches_processed % 10 == 0 {
                        info!(target: "super_entry_stage",
                            "Progress: received={}, batches={}, signals={}, filtered={}",
                            total_candles_received, total_batches_processed,
                            total_signals_generated, total_signals_filtered);
                    }
                }
            }
        }
        
        // ─── FINAL FLUSH — process ALL remaining buffered candles ───
        // This is critical for history mode: when ResultProcessor finishes and
        // drops feature_tx, recv() returns None. We must flush remaining data.
        info!(target: "super_entry_stage", 
            "Channel closed. Flushing {} remaining buffers...",
            candle_buffers.len());

        for ((symbol, tf_minutes), buffer) in candle_buffers.iter_mut() {
            if buffer.len() > warmup {
                let (generated, filtered) = self.flush_buffer(symbol, *tf_minutes, buffer).await;
                total_signals_generated += generated as u64;
                total_signals_filtered += filtered as u64;
                total_batches_processed += 1;
            } else {
                debug!(target: "super_entry_stage",
                    "Skipping final flush for {} {}m — only {} candles (need > {})",
                    symbol, tf_minutes, buffer.len(), warmup);
            }
        }
        
        info!(target: "super_entry_stage", 
            "Super Entry Stage stopped. Total: received={} candles, processed={} batches, \
             generated={} signals, heuristic_filtered={} signals (mode={:?})",
            total_candles_received, total_batches_processed,
            total_signals_generated, total_signals_filtered,
            self.heuristic_mode);
        Ok(())
    }

    /// Flush a buffer: run XGBoost inference on accumulated candles, persist signals.
    /// Returns (signals_generated, signals_filtered_by_heuristic).
    /// After processing, retains only the last `warmup` candles for context.
    async fn flush_buffer(
        &self,
        symbol: &str,
        tf_minutes: i32,
        buffer: &mut Vec<CandleWithIndicators>,
    ) -> (usize, usize) {
        // No clone — process_candles takes &[CandleWithIndicators]
        let result = self.pipeline.process_candles(
            buffer,
            tf_minutes,
            self.use_gpu,
        );

        let (signals_count, filtered_count) = match result {
            Ok(results) => {
                // Collect signals with their corresponding candle data for heuristic filter
                let mut kept_signals = Vec::new();
                let mut n_filtered = 0usize;

                for r in results {
                    if let Some(signal) = r.signal {
                        let candle_idx = r.candle_index;

                        // Get the candle that generated this signal for heuristic analysis
                        if candle_idx < buffer.len() {
                            let candle = &buffer[candle_idx];

                            match apply_heuristic_filter(
                                signal,
                                candle,
                                tf_minutes,
                                &self.cross_tf_store,
                                self.heuristic_mode,
                                self.heuristic_min_confidence,
                                &self.config,
                            ) {
                                Some(filtered_signal) => {
                                    kept_signals.push(filtered_signal);
                                }
                                None => {
                                    n_filtered += 1;
                                }
                            }
                        } else {
                            // Safety fallback: can't find candle, skip filter
                            kept_signals.push(signal);
                        }
                    }
                }

                if !kept_signals.is_empty() {
                    let count = kept_signals.len();
                    match ml_entry_strategy::db_writer::insert_signals_batch(
                        &self.db_pool, &kept_signals
                    ).await {
                        Ok(written) => {
                            info!(target: "super_entry_stage",
                                "{} {}m: {} candles → {} signals ({} heuristic-filtered) → {} written to DB",
                                symbol, tf_minutes, buffer.len(), count + n_filtered,
                                n_filtered, written);
                            (count, n_filtered)
                        }
                        Err(e) => {
                            error!(target: "super_entry_stage",
                                "Failed to persist {} signals for {} {}m: {}",
                                count, symbol, tf_minutes, e);
                            (0, n_filtered)
                        }
                    }
                } else {
                    debug!(target: "super_entry_stage",
                        "{} {}m: {} candles → 0 signals (filtered={})",
                        symbol, tf_minutes, buffer.len(), n_filtered);
                    (0, n_filtered)
                }
            }
            Err(e) => {
                warn!(target: "super_entry_stage",
                    "process_candles failed for {} {}m ({} candles): {}",
                    symbol, tf_minutes, buffer.len(), e);
                (0, 0)
            }
        };

        // Keep only last `warmup` candles for context in next batch
        let warmup = self.pipeline.config().warmup_bars;
        let retain_count = warmup.min(buffer.len());
        if buffer.len() > retain_count {
            buffer.drain(0..buffer.len() - retain_count);
        }

        (signals_count, filtered_count)
    }
}

/// Convert FeatureSnapshot to CandleWithIndicators (zero-copy: no DB roundtrip)
/// symbol_id must be resolved separately (from cache or DB lookup).
fn snapshot_to_candle(snapshot: &FeatureSnapshot, symbol_id: i64) -> CandleWithIndicators {
    CandleWithIndicators {
        time: snapshot.timestamp,
        symbol: snapshot.symbol.clone(),
        symbol_id,
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
        trend: snapshot.indicators.trend_medium as f64,
        trend_short: snapshot.indicators.trend_short as f64,
        poc: snapshot.indicators.poc as f64,
        // Alligator
        alligator_jaw: snapshot.indicators.alligator_jaw as f64,
        alligator_teeth: snapshot.indicators.alligator_teeth as f64,
        alligator_lips: snapshot.indicators.alligator_lips as f64,
        // New indicators (v2)
        mfi: snapshot.indicators.mfi as f64,
        fibo_pivot: snapshot.indicators.fibo_pivot as f64,
        fibo_r1: snapshot.indicators.fibo_r1 as f64,
        fibo_s1: snapshot.indicators.fibo_s1 as f64,
        supertrend: snapshot.indicators.supertrend as f64,
        supertrend_dir: snapshot.indicators.supertrend_dir as f64,
        cmf: snapshot.indicators.cmf as f64,
    }
}

/// Setup Super Entry stage for REALTIME or HISTORY mode.
/// Takes ownership of `feature_rx` — if setup fails, receiver is dropped.
pub async fn setup_super_entry_stage(
    db_pool: &PgPool,
    feature_rx: mpsc::UnboundedReceiver<FeatureSnapshot>,
    use_cuda: bool,
) -> Option<tokio::task::JoinHandle<Result<()>>> {
    let strategy = std::env::var("ACTIVE_STRATEGY").unwrap_or_else(|_| "level".to_string());
    
    if strategy != "super_entry" && strategy != "combined" {
        let explicit = std::env::var("SUPER_ENTRY_ENABLED")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);
        if !explicit {
            info!(target: "super_entry_stage", "Skipping Super Entry stage (strategy: {})", strategy);
            return None;
        }
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
            
            let heuristic_mode = stage.heuristic_mode;
            info!(target: "super_entry_stage",
                "✅ Super Entry stage initialized (GPU={}, heuristic={:?})",
                use_gpu, heuristic_mode);
            Some(tokio::spawn(async move { stage.run().await }))
        }
        Err(e) => {
            error!(target: "super_entry_stage", "Failed to create Super Entry stage: {}", e);
            None
        }
    }
}
