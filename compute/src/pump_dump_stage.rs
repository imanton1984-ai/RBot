// compute/src/pump_dump_stage.rs
//
// Pump/Dump Stage — integrates Pump/Dump ML inference into both
// compute_realtime (trigger-based) and compute_history (batch backfill).
//
// ARCHITECTURE:
//   Unlike SuperEntryStage which processes one (symbol, TF) pair per snapshot,
//   Pump/Dump requires multi-TF candle data for a single symbol to extract
//   cross-timeframe features. Therefore:
//
//   REALTIME PATH:
//     FeatureSnapshot arrives (trigger) → load all TF candles from DB → inference → DB
//     Triggered on candle close of trading TFs (5m, 15m, 1h).
//
//   HISTORY PATH (backfill):
//     After indicators are computed, iterate all symbols:
//       Load all TF candles+indicators from DB → inference → write signals to DB
//     Same as backtest — guarantees identical signal generation.
//
// TABLES:
//   Reads:  market.indicators_wide + market.candles_X (via pump_dump::dataset)
//   Writes: trade.pump_dump_signals (via pump_dump::db_writer)

use std::collections::{HashMap, HashSet};
use anyhow::Result;
use sqlx::PgPool;
use tokio::sync::mpsc;
use tracing::{info, warn, error, debug};

use ml_pump_dump::pump_dump::{CandleInd, PumpDumpConfig, ANALYSIS_TIMEFRAMES};
use ml_pump_dump::signal_generator::PumpDumpSignalConfig;
use ml_pump_dump::pipeline::PumpDumpPipeline;
use ml_pump_dump::dataset::fetch_all_candles_for_tf;
use ml_pump_dump::db_writer;

use crate::predictors::pipeline::FeatureSnapshot;

/// Pump/Dump analysis timeframes as i32 for matching
const PD_ANALYSIS_TFS: &[i32] = ANALYSIS_TIMEFRAMES; // [1440, 240, 60, 15, 5]

/// TFs that trigger inference in realtime mode.
/// Only the LOWEST trading TF triggers inference to avoid redundant processing.
/// Multi-TF data is loaded from DB on trigger, so we don't need triggers from each TF.
const PD_TRIGGER_TFS: &[i32] = &[5];

/// Minimum seconds between processing same symbol (dedup window).
/// Prevents overloading DB with per-candle multi-TF lookups.
const PD_DEDUP_WINDOW_SECS: i64 = 300; // 5 minutes

/// Pump/Dump Stage — consumes feature snapshots as triggers,
/// loads multi-TF data from DB, runs inference, writes signals.
pub struct PumpDumpStage {
    pipeline: PumpDumpPipeline,
    db_pool: PgPool,
    feature_rx: mpsc::UnboundedReceiver<FeatureSnapshot>,
}

impl PumpDumpStage {
    /// Create stage with models loaded.
    pub fn new(
        db_pool: PgPool,
        feature_rx: mpsc::UnboundedReceiver<FeatureSnapshot>,
        use_gpu: bool,
    ) -> Result<Self> {
        let config = PumpDumpConfig::from_env();
        let signal_config = PumpDumpSignalConfig::from_env();

        config.log_summary();
        signal_config.log_summary();

        let pipeline = PumpDumpPipeline::new(config, signal_config, use_gpu)?;

        Ok(Self {
            pipeline,
            db_pool,
            feature_rx,
        })
    }

    /// Check if any models are loaded
    pub fn has_models(&self) -> bool {
        self.pipeline.has_models()
    }

    /// Run the realtime stage.
    ///
    /// Receives FeatureSnapshots as triggers. When a candle closes on a trigger TF,
    /// loads all TF data from DB for that symbol and runs inference.
    /// This guarantees identical feature extraction to backtest (which also reads from DB).
    pub async fn run(mut self) -> Result<()> {
        info!(target: "pump_dump_stage", "Pump/Dump Stage started (trigger TFs: {:?}, dedup={}s)",
            PD_TRIGGER_TFS, PD_DEDUP_WINDOW_SECS);

        let mut total_triggers = 0u64;
        let mut total_signals = 0u64;
        let lookback = self.pipeline.config.pre_event_lookback;

        // Track recently processed symbol → last_processed_time_ms.
        // Uses 5-minute buckets (PD_DEDUP_WINDOW_SECS) to prevent overloading DB
        // with per-candle multi-TF lookups for 300+ symbols.
        let dedup_bucket_ms = PD_DEDUP_WINDOW_SECS * 1000;
        let mut recent_processed: HashSet<(String, i64)> = HashSet::new();
        let mut cycle_count = 0u64;

        while let Some(snapshot) = self.feature_rx.recv().await {
            // Parse TF
            let tf_minutes = match parse_tf_minutes(&snapshot.timeframe) {
                Some(tf) => tf,
                None => continue,
            };

            // Only trigger on the lowest trading TF (5m by default).
            // This is sufficient because we load ALL TF data from DB on each trigger.
            // Processing every TF trigger would multiply DB load by 3x for no benefit.
            if !PD_TRIGGER_TFS.contains(&tf_minutes) {
                continue;
            }

            // Skip history snapshots — those are handled by backfill
            if !snapshot.is_realtime {
                continue;
            }

            // Dedup: skip if we recently processed this symbol (within 5-minute window).
            // With 300+ symbols and 5m candles, this limits to ~1 DB lookup per symbol per 5min.
            let time_ms = snapshot.timestamp.timestamp_millis();
            let dedup_key = (snapshot.symbol.clone(), time_ms / dedup_bucket_ms);
            if recent_processed.contains(&dedup_key) {
                continue;
            }
            recent_processed.insert(dedup_key);

            // Cleanup dedup set periodically
            cycle_count += 1;
            if cycle_count % 500 == 0 {
                let cutoff = chrono::Utc::now().timestamp_millis() / dedup_bucket_ms - 2;
                recent_processed.retain(|(_sym, bucket)| *bucket >= cutoff);
            }

            total_triggers += 1;

            // Load all TF candles from DB for this symbol (same as backtest)
            let all_tf_candles = match PumpDumpPipeline::load_symbol_data(
                &self.db_pool, &snapshot.symbol, lookback
            ).await {
                Ok(d) if !d.is_empty() => d,
                Ok(_) => continue,
                Err(e) => {
                    debug!(target: "pump_dump_stage", "{} load failed: {}", snapshot.symbol, e);
                    continue;
                }
            };

            // Resolve symbol_id
            let symbol_id = all_tf_candles.values()
                .flat_map(|v| v.first())
                .next()
                .map(|c| c.symbol_id)
                .unwrap_or(0);

            // Run inference
            let signals = self.pipeline.process_symbol_candles(
                &all_tf_candles, &snapshot.symbol, symbol_id
            );

            if !signals.is_empty() {
                match db_writer::insert_signals_batch(&self.db_pool, &signals).await {
                    Ok(n) => {
                        total_signals += signals.len() as u64;
                        debug!(target: "pump_dump_stage",
                            "RT: {} {}m → {} signals ({} written)",
                            snapshot.symbol, tf_minutes, signals.len(), n);
                    }
                    Err(e) => {
                        error!(target: "pump_dump_stage",
                            "INSERT failed for {} {}m: {}", snapshot.symbol, tf_minutes, e);
                    }
                }
            }

            if total_triggers % 100 == 0 {
                info!(target: "pump_dump_stage",
                    "RT progress: {} triggers, {} signals generated",
                    total_triggers, total_signals);
            }
        }

        info!(target: "pump_dump_stage",
            "Pump/Dump Stage stopped. Total: {} triggers, {} signals",
            total_triggers, total_signals);
        Ok(())
    }
}

/// Parse timeframe string to minutes
fn parse_tf_minutes(tf_str: &str) -> Option<i32> {
    use std::str::FromStr;
    common::TimeFrame::from_str(tf_str)
        .ok()
        .map(|tf| tf.to_minutes())
}

/// Check if pump_dump strategy is active (from env vars)
pub fn is_pump_dump_enabled() -> bool {
    let strategy = std::env::var("ACTIVE_STRATEGY").unwrap_or_default();
    strategy == "pump_dump" || strategy == "ml_pump_dump"
}

/// Pump/Dump analysis timeframes (for compute TF filtering).
/// All these TFs MUST have indicators computed for feature extraction.
pub fn pump_dump_analysis_tfs() -> &'static [i32] {
    PD_ANALYSIS_TFS
}

/// Setup Pump/Dump stage for REALTIME mode.
/// Returns JoinHandle if setup succeeds, None otherwise.
pub async fn setup_pump_dump_stage(
    db_pool: &PgPool,
    feature_rx: mpsc::UnboundedReceiver<FeatureSnapshot>,
    use_cuda: bool,
) -> Option<tokio::task::JoinHandle<Result<()>>> {
    if !is_pump_dump_enabled() {
        info!(target: "pump_dump_stage", "Pump/Dump strategy not active, skipping stage setup");
        return None;
    }

    info!(target: "pump_dump_stage", "Setting up Pump/Dump stage...");

    // Ensure output table exists
    if let Err(e) = db_writer::ensure_table_exists(db_pool).await {
        error!(target: "pump_dump_stage", "Failed to ensure pump_dump_signals table: {}", e);
        return None;
    }

    let use_gpu = std::env::var("DIRECTION_GPU")
        .map_or(false, |v| v == "1" || v.to_lowercase() == "true") || use_cuda;

    match PumpDumpStage::new(db_pool.clone(), feature_rx, use_gpu) {
        Ok(stage) => {
            if !stage.has_models() {
                error!(target: "pump_dump_stage",
                    "Pump/Dump models not found! Train with pump_dump_dataset + train_pump_dump_wfo.py");
                return None;
            }
            info!(target: "pump_dump_stage",
                "✅ Pump/Dump stage initialized (GPU={}, trigger TFs={:?})",
                use_gpu, PD_TRIGGER_TFS);
            Some(tokio::spawn(async move { stage.run().await }))
        }
        Err(e) => {
            error!(target: "pump_dump_stage", "Failed to create Pump/Dump stage: {}", e);
            None
        }
    }
}

/// Run Pump/Dump backfill — reads candles+indicators from DB, runs inference.
///
/// Called after compute_history finishes indicator computation.
///
/// OPTIMIZED (v2): Uses BULK loading — 5 SQL queries total (one per TF)
/// instead of N×5 per-symbol queries. This eliminates the "slow statement"
/// problem caused by hundreds of individual LEFT JOIN queries on TimescaleDB.
///
/// Flow:
///   1. Load ALL symbols' candles+indicators for each TF in one bulk query (5 total)
///   2. Group by symbol → HashMap<symbol, HashMap<tf, Vec<CandleInd>>>
///   3. Run inference on each symbol from in-memory data (zero additional DB access)
///   4. Batch-insert signals per-symbol
pub async fn run_pump_dump_backfill(pool: &PgPool, use_cuda: bool) -> Result<usize> {
    let config = PumpDumpConfig::from_env();
    let signal_config = PumpDumpSignalConfig::from_env();

    let use_gpu = std::env::var("DIRECTION_GPU")
        .map_or(false, |v| v == "1" || v.to_lowercase() == "true") || use_cuda;

    info!(target: "pump_dump_backfill", "Loading models...");
    let pipeline = PumpDumpPipeline::new(config.clone(), signal_config, use_gpu)?;
    if !pipeline.has_models() {
        warn!(target: "pump_dump_backfill", "No models found, skipping backfill");
        return Ok(0);
    }

    // Ensure table
    db_writer::ensure_table_exists(pool).await?;

    let lookback = config.pre_event_lookback;
    let tf_limits = PumpDumpPipeline::tf_limits();

    info!(target: "pump_dump_backfill",
        "Starting BULK backfill: lookback={}, GPU={}, TFs={:?}",
        lookback, use_gpu, ANALYSIS_TIMEFRAMES);

    let t0 = std::time::Instant::now();

    // ═══════════════════════════════════════════════════════════════════
    // STEP 1: Bulk-load ALL symbols' candles for each TF (5 queries total)
    // ═══════════════════════════════════════════════════════════════════
    // HashMap<symbol, HashMap<tf_minutes, Vec<CandleInd>>>
    let mut all_data: HashMap<String, HashMap<i32, Vec<CandleInd>>> = HashMap::new();

    for &tf in ANALYSIS_TIMEFRAMES {
        let limit = tf_limits.get(&tf).copied().unwrap_or(5000);
        let tf_t0 = std::time::Instant::now();

        match fetch_all_candles_for_tf(pool, tf, limit).await {
            Ok(grouped) => {
                let n_symbols = grouped.len();
                let n_candles: usize = grouped.values().map(|v| v.len()).sum();
                info!(target: "pump_dump_backfill",
                    "  TF {:>5}m: {} symbols, {} candles loaded in {}ms",
                    tf, n_symbols, n_candles, tf_t0.elapsed().as_millis());

                for (symbol, candles) in grouped {
                    if candles.len() >= lookback + 10 {
                        all_data.entry(symbol)
                            .or_default()
                            .insert(tf, candles);
                    }
                }
            }
            Err(e) => {
                warn!(target: "pump_dump_backfill", "  TF {}m bulk fetch failed: {}", tf, e);
            }
        }
    }

    let load_elapsed = t0.elapsed();
    info!(target: "pump_dump_backfill",
        "BULK load complete: {} symbols with data, {:.1}s",
        all_data.len(), load_elapsed.as_secs_f64());

    // ═══════════════════════════════════════════════════════════════════
    // STEP 2: Process each symbol from in-memory data (zero DB access)
    // ═══════════════════════════════════════════════════════════════════
    let mut total_signals = 0usize;
    let mut total_processed = 0usize;

    // Sort symbols for deterministic ordering + progress logging
    let mut symbols: Vec<String> = all_data.keys().cloned().collect();
    symbols.sort();

    for (si, symbol) in symbols.iter().enumerate() {
        let tf_candles = match all_data.get(symbol) {
            Some(d) if !d.is_empty() => d,
            _ => continue,
        };

        total_processed += 1;

        let symbol_id = tf_candles.values()
            .flat_map(|v| v.first())
            .next()
            .map(|c| c.symbol_id)
            .unwrap_or(0);

        let signals = pipeline.process_symbol_candles(tf_candles, symbol, symbol_id);

        if !signals.is_empty() {
            match db_writer::insert_signals_batch(pool, &signals).await {
                Ok(n) => {
                    total_signals += signals.len();
                    debug!(target: "pump_dump_backfill",
                        "{} → {} signals ({} written)", symbol, signals.len(), n);
                }
                Err(e) => {
                    error!(target: "pump_dump_backfill", "{} INSERT failed: {}", symbol, e);
                }
            }
        }

        if (si + 1) % 50 == 0 {
            info!(target: "pump_dump_backfill",
                "Progress: {}/{} symbols, {} signals, {:.1}s",
                si + 1, symbols.len(), total_signals, t0.elapsed().as_secs_f64());
        }
    }

    info!(target: "pump_dump_backfill",
        "✅ Backfill complete: {} signals from {} symbols in {:.1}s (load: {:.1}s, inference: {:.1}s)",
        total_signals, total_processed, t0.elapsed().as_secs_f64(),
        load_elapsed.as_secs_f64(),
        (t0.elapsed() - load_elapsed).as_secs_f64());

    Ok(total_signals)
}
