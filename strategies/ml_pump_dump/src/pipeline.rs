// strategies/ml_pump_dump/src/pipeline.rs
//
// Pipeline for Pump/Dump Detection Strategy (production inference).
//
// Architecture:
//   1. For each symbol, load multi-TF candle data from DB
//   2. Extract features from lookback candles (same structure as training)
//   3. Run both pump and dump XGBoost models
//   4. Apply filters: pred >= 0.65, finest_tf ∈ [1m..1h]
//   5. Generate trade signals → write to trade.pump_dump_signals
//
// REALTIME MODE:
//   Poll DB every N seconds, process latest candle for each symbol.
//   Maintains per-symbol state to avoid reprocessing.
//
// BACKFILL MODE:
//   Process historical candles in bulk.

use anyhow::Result;
use std::collections::HashMap;
use tracing::{info, warn, debug};

use crate::pump_dump::{
    CandleInd, EventType, PumpDumpConfig,
    ANALYSIS_TIMEFRAMES, FULL_FEATURES_PER_CANDLE,
    extract_candle_features,
};
use crate::dataset::fetch_candles_with_indicators;
use crate::signal_generator::{PumpDumpSignalConfig, PumpDumpSignal, generate_signal};

use predictors::ml::xgb_runtime::{Booster, Device, ModelKind};

// ═════════════════════════════════════════════════════════════════════════════
// PIPELINE
// ═════════════════════════════════════════════════════════════════════════════

/// Pump/Dump detection pipeline.
///
/// Holds loaded models and config, provides methods for both
/// batch (backfill) and single-symbol (realtime) processing.
pub struct PumpDumpPipeline {
    pub config: PumpDumpConfig,
    pub signal_config: PumpDumpSignalConfig,
    pump_model: Option<Booster>,
    dump_model: Option<Booster>,
    n_features: usize,
}

impl PumpDumpPipeline {
    /// Create a new pipeline, loading both models from disk.
    ///
    /// # Arguments
    /// * `config` — pump/dump detection config
    /// * `signal_config` — signal generation config (thresholds, targets)
    /// * `use_gpu` — whether to use GPU for XGBoost inference
    pub fn new(
        config: PumpDumpConfig,
        signal_config: PumpDumpSignalConfig,
        use_gpu: bool,
    ) -> Result<Self> {
        let device = if use_gpu { Device::Cuda } else { Device::Cpu };

        let pump_model_path = std::env::var("PD_PUMP_MODEL_PATH")
            .unwrap_or_else(|_| "models/pump_dump_pump_v1.ubj".to_string());
        let dump_model_path = std::env::var("PD_DUMP_MODEL_PATH")
            .unwrap_or_else(|_| "models/pump_dump_dump_v1.ubj".to_string());

        let pump_model = match Booster::load(&pump_model_path, device) {
            Ok(b) => {
                info!("  ✅ Pump model loaded: {}", pump_model_path);
                Some(b)
            }
            Err(e) => {
                warn!("  ❌ Pump model not found: {} — {}", pump_model_path, e);
                None
            }
        };

        let dump_model = match Booster::load(&dump_model_path, device) {
            Ok(b) => {
                info!("  ✅ Dump model loaded: {}", dump_model_path);
                Some(b)
            }
            Err(e) => {
                warn!("  ❌ Dump model not found: {} — {}", dump_model_path, e);
                None
            }
        };

        let n_features = ANALYSIS_TIMEFRAMES.len()
            * config.pre_event_lookback
            * FULL_FEATURES_PER_CANDLE;

        info!("  Feature vector size: {} features", n_features);

        Ok(Self {
            config,
            signal_config,
            pump_model,
            dump_model,
            n_features,
        })
    }

    /// Check if any models are loaded.
    pub fn has_models(&self) -> bool {
        self.pump_model.is_some() || self.dump_model.is_some()
    }

    /// Extract multi-TF features for the current market state of a symbol.
    ///
    /// Treats the latest candle on each TF as the "potential onset" point
    /// and extracts lookback candles before it — same structure as training data.
    ///
    /// # Returns
    /// `(features, finest_tf)` — the flattened feature vector and the finest TF
    /// that had enough data for feature extraction.
    pub fn extract_realtime_features(
        &self,
        all_tf_candles: &HashMap<i32, Vec<CandleInd>>,
    ) -> Option<(Vec<f64>, i32)> {
        let lookback = self.config.pre_event_lookback;
        let n_tfs = ANALYSIS_TIMEFRAMES.len();
        let total_features = n_tfs * lookback * FULL_FEATURES_PER_CANDLE;

        let mut features = vec![0.0f64; total_features];
        let mut finest_tf = 1440; // start pessimistic

        for (tf_idx, &tf) in ANALYSIS_TIMEFRAMES.iter().enumerate() {
            let candles = match all_tf_candles.get(&tf) {
                Some(c) if c.len() >= lookback + 10 => c,
                _ => continue,
            };

            // Use the last candle as "current position" (potential onset)
            let onset_idx = candles.len() - 1;
            if onset_idx < lookback { continue; }

            finest_tf = tf;

            let feature_offset = tf_idx * lookback * FULL_FEATURES_PER_CANDLE;
            for c_off in 0..lookback {
                let candle_idx = onset_idx - lookback + c_off;
                if candle_idx >= candles.len() { continue; }

                let candle_feats = extract_candle_features(candles, candle_idx);
                let start = feature_offset + c_off * FULL_FEATURES_PER_CANDLE;
                let end = start + FULL_FEATURES_PER_CANDLE;
                if end <= features.len() {
                    features[start..end].copy_from_slice(&candle_feats);
                }
            }
        }

        // Only return if we have at least some TF data
        if finest_tf == 1440 {
            // Check if we actually got daily features
            let daily_candles = all_tf_candles.get(&1440);
            if daily_candles.map_or(true, |c| c.len() < lookback + 10) {
                return None;
            }
        }

        Some((features, finest_tf))
    }

    /// Run inference on a single feature vector and generate signals.
    ///
    /// Runs both pump and dump models. Returns 0, 1, or 2 signals
    /// (one pump, one dump if both pass threshold).
    pub fn predict_and_generate(
        &self,
        features: &[f64],
        finest_tf: i32,
        symbol: &str,
        symbol_id: i64,
        time: chrono::DateTime<chrono::Utc>,
        close_price: f64,
    ) -> Vec<PumpDumpSignal> {
        let feats_f32: Vec<f32> = features.iter().map(|&v| v as f32).collect();
        let mut signals = Vec::new();

        // Pump model
        if let Some(ref model) = self.pump_model {
            match model.predict_dense_cpu(&feats_f32, 1, self.n_features, ModelKind::Regressor1) {
                Ok(pred) => {
                    let prob = pred[0].clamp(0.0, 1.0);
                    if let Some(sig) = generate_signal(
                        &self.signal_config,
                        EventType::Pump,
                        prob,
                        finest_tf,
                        symbol,
                        symbol_id,
                        time,
                        close_price,
                    ) {
                        debug!("  PUMP signal: {} pred={:.3} finest={}m",
                            symbol, prob, finest_tf);
                        signals.push(sig);
                    }
                }
                Err(e) => {
                    warn!("Pump model prediction failed for {}: {}", symbol, e);
                }
            }
        }

        // Dump model
        if let Some(ref model) = self.dump_model {
            match model.predict_dense_cpu(&feats_f32, 1, self.n_features, ModelKind::Regressor1) {
                Ok(pred) => {
                    let prob = pred[0].clamp(0.0, 1.0);
                    if let Some(sig) = generate_signal(
                        &self.signal_config,
                        EventType::Dump,
                        prob,
                        finest_tf,
                        symbol,
                        symbol_id,
                        time,
                        close_price,
                    ) {
                        debug!("  DUMP signal: {} pred={:.3} finest={}m",
                            symbol, prob, finest_tf);
                        signals.push(sig);
                    }
                }
                Err(e) => {
                    warn!("Dump model prediction failed for {}: {}", symbol, e);
                }
            }
        }

        signals
    }

    /// Process a single symbol: load multi-TF data, extract features, predict.
    ///
    /// This is the main method for both backfill and realtime processing.
    /// For backfill, call this for each symbol.
    /// For realtime, call with pre-loaded candle data.
    pub fn process_symbol_candles(
        &self,
        all_tf_candles: &HashMap<i32, Vec<CandleInd>>,
        symbol: &str,
        symbol_id: i64,
    ) -> Vec<PumpDumpSignal> {
        let (features, finest_tf) = match self.extract_realtime_features(all_tf_candles) {
            Some(f) => f,
            None => return Vec::new(),
        };

        // Get the latest candle's time and close price from the finest TF
        let (time, close_price) = match all_tf_candles.get(&finest_tf) {
            Some(candles) if !candles.is_empty() => {
                let last = candles.last().unwrap();
                (last.time, last.close)
            }
            _ => return Vec::new(),
        };

        self.predict_and_generate(
            &features,
            finest_tf,
            symbol,
            symbol_id,
            time,
            close_price,
        )
    }

    /// TF limits for per-symbol data loading.
    pub fn tf_limits() -> HashMap<i32, usize> {
        vec![
            (1440, 3700), (240, 12000), (60, 12000),
            (15, 12000), (5, 12000), (1, 5000),
        ].into_iter().collect()
    }

    /// Load all TF candle data for a symbol from DB.
    pub async fn load_symbol_data(
        pool: &sqlx::PgPool,
        symbol: &str,
        lookback: usize,
    ) -> Result<HashMap<i32, Vec<CandleInd>>> {
        let tf_limits = Self::tf_limits();
        let mut all_tf_candles: HashMap<i32, Vec<CandleInd>> = HashMap::new();

        for &tf in ANALYSIS_TIMEFRAMES {
            let limit = tf_limits.get(&tf).copied().unwrap_or(5000);
            match fetch_candles_with_indicators(pool, symbol, tf, limit).await {
                Ok(candles) if candles.len() >= lookback + 10 => {
                    all_tf_candles.insert(tf, candles);
                }
                Ok(_) => {}
                Err(e) => {
                    debug!("  {} TF {}m: {}", symbol, tf, e);
                }
            }
        }

        Ok(all_tf_candles)
    }
}
