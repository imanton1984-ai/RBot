// compute/predictors/pipeline.rs
//
// Optimized pipeline with:
// - In-memory caches for symbol_id and predictor_id (eliminates N DB queries per candle)
// - Batch accumulation for history mode (sends predictions in bulk)
// - Skips details_json for history mode (eliminates expensive JSON serialization)
// - Uses bulk_sender → BulkPersistor for batched DB writes

use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use crate::types::{PredictionRow, PredictionAspect, CalcSource, PredictorMeta};
use crate::feature_view::{FeatureView, IndicatorsWideRow};
use crate::level_view::LevelView;
use crate::config::PredictorsConfig;
use crate::persistence;
use sqlx::PgPool;
use common::MessageBus;
use crate::consensus::ConsensusEngine;

use crate::ml::model_manager::ModelManager;
use database_lib;

/// In-memory cache for symbol_id lookups (avoids DB query per snapshot)
struct SymbolCache {
    cache: HashMap<String, i64>,
}

impl SymbolCache {
    fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    async fn get_or_resolve(&mut self, pool: &PgPool, symbol: &str) -> Result<i64> {
        if let Some(&id) = self.cache.get(symbol) {
            return Ok(id);
        }
        let id = persistence::resolve_symbol_id(pool, symbol).await?;
        self.cache.insert(symbol.to_string(), id);
        Ok(id)
    }
}

/// In-memory cache for predictor_id lookups (avoids DB query per predictor per snapshot)
/// Key: (calc_source_int, name, version)
struct PredictorCache {
    cache: HashMap<(i16, String, String), i64>,
}

impl PredictorCache {
    fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    async fn get_or_register(&mut self, pool: &PgPool, meta: &PredictorMeta) -> Result<i64> {
        let key = (meta.calc_source.as_int(), meta.name.clone(), meta.version.clone());
        if let Some(&id) = self.cache.get(&key) {
            return Ok(id);
        }
        let id = persistence::register_predictor_if_missing(pool, meta).await?.0;
        self.cache.insert(key, id);
        Ok(id)
    }
}

/// Lightweight input for the trade signal stage.
/// Sent from PredictorsPipeline after producing predictions.
#[derive(Debug, Clone)]
pub struct TradeSignalInput {
    pub symbol: String,
    pub symbol_id: i64,
    pub tf_minutes: i16,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub time_ms: i64,
    pub close_price: f64,
    pub predictions: Vec<PredictionRow>,
    pub raw_signals_summary: serde_json::Value,
    pub is_realtime: bool,
}

pub struct PredictorsPipeline {
    config: PredictorsConfig,
    db_pool: PgPool,
    shutdown_rx: tokio::sync::broadcast::Receiver<bool>,
    feature_rx: Option<tokio::sync::mpsc::UnboundedReceiver<FeatureSnapshot>>,
    ml_manager: ModelManager,
    bulk_sender: Option<tokio::sync::mpsc::Sender<database_lib::PersistRecord>>,
    trade_signal_tx: Option<tokio::sync::mpsc::UnboundedSender<TradeSignalInput>>,
    symbol_cache: SymbolCache,
    predictor_cache: PredictorCache,
}

impl PredictorsPipeline {
    pub fn new(
        config: PredictorsConfig,
        db_pool: PgPool,
        _message_bus: MessageBus,
        shutdown_rx: tokio::sync::broadcast::Receiver<bool>,
    ) -> Self {
        let mut ml_manager = ModelManager::new(config.use_cuda);

        // Load models for all timeframes
        let timeframes = vec![1, 5, 15, 60, 240, 1440];

        if let Err(e) = ml_manager.load_models_for_timeframes("price", &config.model_path_price, &timeframes, config.use_gpu_history) {
            tracing::error!("Failed to load price models for timeframes: {}", e);
        }
        if let Err(e) = ml_manager.load_models_for_timeframes("levels", &config.model_path_levels, &timeframes, config.use_gpu_history) {
            tracing::error!("Failed to load level models for timeframes: {}", e);
        }

        Self {
            config,
            db_pool,
            shutdown_rx,
            feature_rx: None,
            ml_manager,
            bulk_sender: None,
            trade_signal_tx: None,
            symbol_cache: SymbolCache::new(),
            predictor_cache: PredictorCache::new(),
        }
    }
    
    pub async fn run(&mut self) -> Result<()> {
        let mut rx = self.feature_rx.take().expect("Feature receiver must be set before calling run()");
        
        // Batch accumulator for history mode: collect predictions before flushing
        let mut history_batch: Vec<database_lib::PersistRecord> = Vec::with_capacity(4096);
        let mut history_batch_count: usize = 0;
        const HISTORY_FLUSH_THRESHOLD: usize = 2048; // Flush every N predictions

        while !self.is_shutdown().await {
            tokio::select! {
                feature_snapshot = rx.recv() => {
                    match feature_snapshot {
                        Some(snapshot) => {
                            let is_history = !snapshot.is_realtime;
                            if let Err(e) = self.process_feature_snapshot(snapshot, &mut history_batch).await {
                                tracing::error!("Error processing feature snapshot: {}", e);
                            }
                            
                            // For history: flush accumulated batch when threshold reached
                            if is_history && history_batch.len() >= HISTORY_FLUSH_THRESHOLD {
                                if let Some(tx) = &self.bulk_sender {
                                    history_batch_count += history_batch.len();
                                    for rec in history_batch.drain(..) {
                                        if let Err(e) = tx.send(rec).await {
                                            tracing::error!(target: "compute_predictors", 
                                                "Failed to flush predictor batch: {}", e);
                                            break;
                                        }
                                    }
                                    if history_batch_count % 10_000 < HISTORY_FLUSH_THRESHOLD {
                                        tracing::info!(target: "compute_predictors",
                                            "Predictors pipeline: flushed {} total predictions to bulk persistor",
                                            history_batch_count);
                                    }
                                }
                                history_batch.clear();
                            }
                        }
                        None => {
                            break;
                        }
                    }
                }
                _ = self.shutdown_rx.recv() => {
                    tracing::info!("Shutdown signal received, stopping predictors pipeline");
                    break;
                }
            }
        }

        // Flush remaining history batch
        if !history_batch.is_empty() {
            if let Some(tx) = &self.bulk_sender {
                history_batch_count += history_batch.len();
                for rec in history_batch.drain(..) {
                    if let Err(e) = tx.send(rec).await {
                        tracing::error!(target: "compute_predictors", 
                            "Failed to flush final predictor batch: {}", e);
                        break;
                    }
                }
                tracing::info!(target: "compute_predictors",
                    "Predictors pipeline: flushed final batch, {} total predictions",
                    history_batch_count);
            }
        }

        Ok(())
    }
    
    pub fn set_input_receiver(&mut self, receiver: tokio::sync::mpsc::UnboundedReceiver<FeatureSnapshot>) {
        self.feature_rx = Some(receiver);
    }

    pub fn set_bulk_sender(&mut self, tx: tokio::sync::mpsc::Sender<database_lib::PersistRecord>) {
        self.bulk_sender = Some(tx);
    }

    pub fn set_trade_signal_sender(&mut self, tx: tokio::sync::mpsc::UnboundedSender<TradeSignalInput>) {
        self.trade_signal_tx = Some(tx);
    }

    async fn is_shutdown(&mut self) -> bool {
        match self.shutdown_rx.try_recv() {
            Ok(_) | Err(tokio::sync::broadcast::error::TryRecvError::Closed) => true,
            _ => false,
        }
    }

    /// Process a single feature snapshot. 
    /// For history mode: predictions are accumulated in `history_batch` (not sent immediately).
    /// For realtime mode: predictions are sent directly via bulk_sender.
    async fn process_feature_snapshot(
        &mut self, 
        snapshot: FeatureSnapshot,
        history_batch: &mut Vec<database_lib::PersistRecord>,
    ) -> Result<()> {
        let is_realtime = snapshot.is_realtime;

        let view = FeatureView::new(
            snapshot.timestamp,
            snapshot.symbol.clone(),
            snapshot.timeframe.clone(),
            snapshot.is_realtime,
            snapshot.indicators,
            snapshot.raw_signals_data.as_ref(),
            snapshot.sr_levels,
        )?;

        let hc_preds = match self.run_hardcode_predictors(&view).await {
            Ok(preds_opt) => preds_opt.unwrap_or_default(),
            Err(e) => {
                tracing::error!(target: "compute_predictors",
                    "Error in hardcode predictors: {} for {}:{}",
                    e, snapshot.symbol, snapshot.timeframe
                );
                Vec::new()
            }
        };

        let ml_preds = match self.run_ml_predictors(&view).await {
            Ok(preds_opt) => preds_opt.unwrap_or_default(),
            Err(e) => {
                tracing::error!(target: "compute_predictors",
                    "Error in ML predictors: {} for {}:{}",
                    e, snapshot.symbol, snapshot.timeframe
                );
                Vec::new()
            }
        };

        let consensus = ConsensusEngine::new();
        let final_preds = match consensus.apply_gate_and_fuse(hc_preds, ml_preds, &view).await {
            Ok(preds) => preds,
            Err(e) => {
                tracing::error!(target: "compute_predictors",
                    "Error in consensus engine: {} for {}:{}",
                    e, snapshot.symbol, snapshot.timeframe
                );
                Vec::new()
            }
        };

        if final_preds.is_empty() {
            return Ok(());
        }

        // Send predictions to TradeSignalStage (if connected)
        if let Some(ref trade_tx) = self.trade_signal_tx {
            let sid = self.symbol_cache.get_or_resolve(&self.db_pool, &snapshot.symbol).await?;
            let tf_min = self.parse_timeframe_minutes(&snapshot.timeframe)? as i16;
            let close_price = view.indicators.close as f64;
            let ts_input = TradeSignalInput {
                symbol: snapshot.symbol.clone(),
                symbol_id: sid,
                tf_minutes: tf_min,
                timestamp: snapshot.timestamp,
                time_ms: snapshot.timestamp.timestamp_millis(),
                close_price,
                predictions: final_preds.clone(),
                raw_signals_summary: build_raw_signals_summary(
                    &view.indicators,
                    view.sr_levels.as_ref(),
                ),
                is_realtime: is_realtime,
            };
            if let Err(e) = trade_tx.send(ts_input) {
                tracing::warn!(
                    target: "compute_predictors",
                    "Failed to send to TradeSignalStage: {} (channel closed?)", e
                );
            }
        }

        // Convert predictions to PersistRecord
        // For history: skip details_json to avoid expensive JSON serialization
        let skip_details = !is_realtime;
        
        let records: Vec<database_lib::PersistRecord> = final_preds.iter().map(|pred| {
            // NaN-safe: f32::NAN.clamp() returns NaN which violates DB CHECK constraint
            let safe_score = if pred.score_norm.is_finite() { pred.score_norm.clamp(0.0, 1.0) } else { 0.0 };
            database_lib::PersistRecord::Predictor {
                symbol: common::Symbol::from(pred.symbol.clone()),
                timeframe: pred.tf_minutes as i16,
                time_ms: pred.time_ms,
                horizon_bars: pred.horizon_bars,
                aspect: pred.aspect.as_int(),
                calc_source: pred.calc_source.as_int(),
                predictor_id: pred.predictor_id,
                score_norm: safe_score,
                value: pred.value,
                value_low: pred.value_low,
                value_high: pred.value_high,
                side: pred.side,
                level_hash: pred.level_hash.clone(),
                level_kind: pred.level_kind,
                level_price: pred.level_price,
                level_strength: pred.level_strength,
                level_distance_atr: pred.level_distance_atr,
                candle_is_final: pred.candle_is_final,
                event_time_ms: pred.event_time_ms,
                details_json: if skip_details { None } else { pred.details_json.clone() },
                prediction_key: pred.prediction_key.clone(),
            }
        }).collect();

        if is_realtime {
            // Realtime: send immediately via bulk_sender or fallback
            if let Some(tx) = &self.bulk_sender {
                for rec in records {
                    if let Err(e) = tx.send(rec).await {
                        tracing::error!(target: "compute_predictors", 
                            "Failed to send predictor to bulk persistor: {}", e);
                    }
                }
            } else {
                // Fallback: direct upsert (slow, only for realtime without bulk_sender)
                let final_preds_len = final_preds.len();
                match persistence::upsert_predictors(&self.db_pool, final_preds).await {
                    Ok(_) => {
                        tracing::debug!(target: "compute_predictors",
                            "Saved {} predictions via direct upsert for {}:{}",
                            final_preds_len, snapshot.symbol, snapshot.timeframe
                        );
                    },
                    Err(e) => {
                        tracing::error!(target: "compute_predictors",
                            "Failed to save predictions: {} for {}:{}",
                            e, snapshot.symbol, snapshot.timeframe
                        );
                    }
                }
            }
        } else {
            // History: accumulate in batch (will be flushed by run() loop)
            history_batch.extend(records);
        }

        Ok(())
    }

    async fn run_hardcode_predictors(&mut self, view: &FeatureView) -> Result<Option<Vec<PredictionRow>>> {
        let mut predictors = Vec::new();
        let view_close = view.indicators.close as f64;
        let is_history = !view.is_realtime();

        // Price Target predictions
        let hc_price = crate::future_predictor::heuristic_predictor::FuturePriceHeuristic::new();
        match hc_price.predict(&view.symbol, &view.timeframe, view) {
            Ok(Some((predicted_prices, score))) => {
                let sid = self.symbol_cache.get_or_resolve(&self.db_pool, &view.symbol).await?;
                let predictor_meta = PredictorMeta {
                    predictor_id: 0,
                    name: "price10_hard".to_string(),
                    version: "1.0".to_string(),
                    aspect: PredictionAspect::PriceTarget,
                    calc_source: CalcSource::Hard,
                    framework: "hardcode".to_string(),
                    artifact_path: None,
                    feature_schema_id: "feature_view".to_string(),
                };
                let predictor_id = self.predictor_cache.get_or_register(&self.db_pool, &predictor_meta).await?;
                
                let last_predicted_price = predicted_prices.last().copied().unwrap_or(view_close);
                let atr = view.indicators.atr as f64;
                let uncertainty_factor = 0.5;
                let value_low = Some((last_predicted_price - atr * uncertainty_factor).max(0.00000001));
                let value_high = Some(last_predicted_price + atr * uncertainty_factor);
                
                let prediction_row = PredictionRow {
                    time: view.timestamp,
                    time_ms: view.timestamp.timestamp_millis(),
                    symbol_id: sid,
                    symbol: view.symbol.clone(),
                    tf_minutes: self.parse_timeframe_minutes(&view.timeframe)?,
                    horizon_bars: self.config.horizon_bars as i32,
                    aspect: PredictionAspect::PriceTarget,
                    calc_source: CalcSource::Hard,
                    predictor_id,
                    score_norm: (score.abs() as f32).clamp(0.0, 1.0),
                    value: last_predicted_price,
                    value_low,
                    value_high,
                    side: self.determine_side(view_close, last_predicted_price),
                    level_hash: None,
                    level_kind: None,
                    level_price: None,
                    level_strength: None,
                    level_distance_atr: None,
                    candle_is_final: true,
                    event_time_ms: None,
                    // Skip JSON for history (big perf win)
                    details_json: if is_history { None } else {
                        Some(serde_json::json!({
                            "method": "hardcode_price10",
                            "predicted_prices": predicted_prices,
                            "confidence_factors": self.calculate_confidence_factors(view)
                        }))
                    },
                    prediction_key: format!("price10_hard_{}_{}", view.symbol, view.timestamp.timestamp()),
                };
                predictors.push(prediction_row);
            },
            Ok(None) => {},
            Err(e) => {
                tracing::error!(target: "compute_predictors",
                    "Price predictor error for {}:{}: {}", view.symbol, view.timeframe, e);
            }
        }

        // Level predictions (Bounce and Breakout)
        let hc_level = crate::level_predictor::future_heruistic_predictor::LevelPredictorHeuristic::new();
        match hc_level.predict(&view.symbol, &view.timeframe, view) {
            Ok(Some((level, prob_bounce, prob_break, score))) => {
                let sid = self.symbol_cache.get_or_resolve(&self.db_pool, &view.symbol).await?;
                let tf_minutes = self.parse_timeframe_minutes(&view.timeframe)?;
                
                // Bounce prediction
                let bounce_meta = PredictorMeta {
                    predictor_id: 0,
                    name: "level_bounce_hard".to_string(),
                    version: "1.0".to_string(),
                    aspect: PredictionAspect::LevelBounce,
                    calc_source: CalcSource::Hard,
                    framework: "hardcode".to_string(),
                    artifact_path: None,
                    feature_schema_id: "feature_view".to_string(),
                };
                let bounce_predictor_id = self.predictor_cache.get_or_register(&self.db_pool, &bounce_meta).await?;
                
                let level_details = if is_history { None } else {
                    Some(serde_json::json!({
                        "method": "hardcode_level_bounce",
                        "level_price": level.level_price,
                        "prob_bounce": prob_bounce,
                        "prob_break": prob_break,
                        "score": score,
                        "level_info": {
                            "hash": &level.level_hash,
                            "kind": level.level_kind.as_i16(),
                            "strength": level.level_strength,
                            "distance_atr": level.distance_atr
                        }
                    }))
                };

                predictors.push(PredictionRow {
                    time: view.timestamp,
                    time_ms: view.timestamp.timestamp_millis(),
                    symbol_id: sid,
                    symbol: view.symbol.clone(),
                    tf_minutes,
                    horizon_bars: self.config.horizon_bars as i32,
                    aspect: PredictionAspect::LevelBounce,
                    calc_source: CalcSource::Hard,
                    predictor_id: bounce_predictor_id,
                    score_norm: (prob_bounce as f32).clamp(0.0, 1.0),
                    value: prob_bounce,
                    value_low: Some((prob_bounce - 0.1).max(0.0)),
                    value_high: Some((prob_bounce + 0.1).min(1.0)),
                    side: Some(if view_close < level.level_price { 1 } else { -1 }),
                    level_hash: Some(level.level_hash.clone()),
                    level_kind: Some(level.level_kind.as_i16()),
                    level_price: Some(level.level_price),
                    level_strength: Some(level.level_strength),
                    level_distance_atr: Some(level.distance_atr),
                    candle_is_final: true,
                    event_time_ms: None,
                    details_json: level_details,
                    prediction_key: format!("bounce_hard_{}_{}_{}", view.symbol, level.level_hash, view.timestamp.timestamp()),
                });
                
                // Breakout prediction
                let break_meta = PredictorMeta {
                    predictor_id: 0,
                    name: "level_breakout_hard".to_string(),
                    version: "1.0".to_string(),
                    aspect: PredictionAspect::LevelBreakout,
                    calc_source: CalcSource::Hard,
                    framework: "hardcode".to_string(),
                    artifact_path: None,
                    feature_schema_id: "feature_view".to_string(),
                };
                let break_predictor_id = self.predictor_cache.get_or_register(&self.db_pool, &break_meta).await?;

                let break_details = if is_history { None } else {
                    Some(serde_json::json!({
                        "method": "hardcode_level_breakout",
                        "level_price": level.level_price,
                        "prob_bounce": prob_bounce,
                        "prob_break": prob_break,
                        "score": score,
                        "level_info": {
                            "hash": &level.level_hash,
                            "kind": level.level_kind.as_i16(),
                            "strength": level.level_strength,
                            "distance_atr": level.distance_atr
                        }
                    }))
                };

                predictors.push(PredictionRow {
                    time: view.timestamp,
                    time_ms: view.timestamp.timestamp_millis(),
                    symbol_id: sid,
                    symbol: view.symbol.clone(),
                    tf_minutes,
                    horizon_bars: self.config.horizon_bars as i32,
                    aspect: PredictionAspect::LevelBreakout,
                    calc_source: CalcSource::Hard,
                    predictor_id: break_predictor_id,
                    score_norm: (prob_break as f32).clamp(0.0, 1.0),
                    value: prob_break,
                    value_low: Some((prob_break - 0.1).max(0.0)),
                    value_high: Some((prob_break + 0.1).min(1.0)),
                    side: Some(if view_close < level.level_price { -1 } else { 1 }),
                    level_hash: Some(level.level_hash.clone()),
                    level_kind: Some(level.level_kind.as_i16()),
                    level_price: Some(level.level_price),
                    level_strength: Some(level.level_strength),
                    level_distance_atr: Some(level.distance_atr),
                    candle_is_final: true,
                    event_time_ms: None,
                    details_json: break_details,
                    prediction_key: format!("breakout_hard_{}_{}_{}", view.symbol, level.level_hash, view.timestamp.timestamp()),
                });
            },
            Ok(None) => {},
            Err(e) => {
                tracing::error!(target: "compute_predictors",
                    "Level predictor error for {}:{}: {}", view.symbol, view.timeframe, e);
            }
        }

        if predictors.is_empty() {
            Ok(None)
        } else {
            Ok(Some(predictors))
        }
    }

    async fn run_ml_predictors(&mut self, view: &FeatureView) -> Result<Option<Vec<PredictionRow>>> {
        let mut predictors = Vec::new();
        let view_close = view.indicators.close as f64;
        let is_history = !view.is_realtime();

        let sid = self.symbol_cache.get_or_resolve(&self.db_pool, &view.symbol).await?;
        let tf_minutes = self.parse_timeframe_minutes(&view.timeframe)?;

        // Price model prediction
        if let Some(schema) = self.ml_manager.get_schema(&format!("price_tf{}", tf_minutes)) {
            let input_vec = schema.build_vector(view);
            let use_gpu = if view.is_realtime() { 
                self.config.use_gpu_realtime 
            } else { 
                self.config.use_gpu_history 
            };
            
            match self.ml_manager.predict_one(&format!("price_tf{}", tf_minutes), &input_vec, use_gpu) {
                Ok(Some(predicted_prices)) => {
                    let predictor_meta = PredictorMeta {
                        predictor_id: 0,
                        name: format!("price10_ml_tf{}", tf_minutes),
                        version: "1.0".to_string(),
                        aspect: PredictionAspect::PriceTarget,
                        calc_source: CalcSource::Ml,
                        framework: "xgboost".to_string(),
                        artifact_path: Some(self.config.model_path_price.replace("{tf}", &tf_minutes.to_string())),
                        feature_schema_id: schema.schema_id.clone(),
                    };
                    let predictor_id = self.predictor_cache.get_or_register(&self.db_pool, &predictor_meta).await?;
                    
                    let score = if !predicted_prices.is_empty() { predicted_prices[0].abs().min(1.0) as f64 } else { 0.5 };
                    let last_predicted_price = predicted_prices.last().copied().unwrap_or(view_close as f32) as f64;
                    let safe_predicted_price = last_predicted_price.max(0.00000001);

                    let atr = view.indicators.atr as f64;
                    let uncertainty_factor = 0.5;
                    let value_low = Some((safe_predicted_price - atr * uncertainty_factor).max(0.00000001));
                    let value_high = Some(safe_predicted_price + atr * uncertainty_factor);

                    let details = if is_history { None } else {
                        Some(serde_json::json!({
                            "method": "ml_price10",
                            "predicted_prices": predicted_prices,
                            "confidence": score,
                            "model_used": self.config.model_path_price.replace("{tf}", &tf_minutes.to_string())
                        }))
                    };

                    predictors.push(PredictionRow {
                        time: view.timestamp,
                        time_ms: view.timestamp.timestamp_millis(),
                        symbol_id: sid,
                        symbol: view.symbol.clone(),
                        tf_minutes,
                        horizon_bars: self.config.horizon_bars as i32,
                        aspect: PredictionAspect::PriceTarget,
                        calc_source: CalcSource::Ml,
                        predictor_id,
                    score_norm: (score as f32).clamp(0.0, 1.0),
                    value: safe_predicted_price,
                        value_low,
                        value_high,
                        side: self.determine_side(view_close, safe_predicted_price),
                        level_hash: None,
                        level_kind: None,
                        level_price: None,
                        level_strength: None,
                        level_distance_atr: None,
                        candle_is_final: true,
                        event_time_ms: None,
                        details_json: details,
                        prediction_key: format!("price10_ml_{}_{}", view.symbol, view.timestamp.timestamp()),
                    });
                },
                Ok(None) => {},
                Err(e) => {
                    tracing::error!(target: "compute_predictors",
                        "ML price prediction error for {}:{}: {}", view.symbol, view.timeframe, e);
                }
            }
        }

        // Level model prediction
        if let Some(schema) = self.ml_manager.get_schema(&format!("level_tf{}", tf_minutes)) {
            let input_vec = schema.build_vector(view);
            let use_gpu = if view.is_realtime() { 
                self.config.use_gpu_realtime 
            } else { 
                self.config.use_gpu_history 
            };
            
            match self.ml_manager.predict_one(&format!("level_tf{}", tf_minutes), &input_vec, use_gpu) {
                Ok(Some(level_outputs)) => {
                    let (prob_bounce, prob_break) = if level_outputs.len() >= 2 {
                        (level_outputs[0].max(0.0).min(1.0) as f64, level_outputs[1].max(0.0).min(1.0) as f64)
                    } else if level_outputs.len() == 1 {
                        let label = level_outputs[0];
                        if label > 0.5 { (0.2, 0.8) } else { (0.8, 0.2) }
                    } else {
                        (0.5, 0.5)
                    };

                    let atr = view.indicators.atr as f64;

                    // Try to get levels and create predictions
                    if let Ok(sr_levels) = view.get_sr_levels() {
                        if !sr_levels.is_empty() {
                            // Extract level info (hash, kind, price, strength, distance_atr)
                            // to avoid lifetime issues with LevelView
                            let level_info_from_view: Option<(String, i16, f64, f32, f32)> = match LevelView::new(sr_levels.clone(), view_close, atr, view.timestamp, view.symbol.clone(), view.timeframe.clone(), 2, None) {
                                Ok(level_view) => {
                                    level_view.get_near_levels().first().map(|level| {
                                        (level.level_hash.clone(), level.level_kind.as_i16(), level.level_price, level.level_strength, level.distance_atr)
                                    })
                                },
                                Err(_) => None,
                            };

                            // If no near level found, find the closest one as fallback
                            let level_info = if let Some(info) = level_info_from_view {
                                Some(info)
                            } else {
                                // Fallback: find closest level
                                sr_levels.iter().min_by(|a, b| {
                                    let da = (a.price - view_close).abs();
                                    let db = (b.price - view_close).abs();
                                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                                }).map(|closest| {
                                    let dist = (closest.price - view_close).abs();
                                    let dist_atr = if atr > 0.0 { (dist / atr) as f32 } else { 0.0 };
                                    let hash = format!("ml_fallback_{}_{}", closest.price, view.timestamp.timestamp());
                                    (hash, 0i16, closest.price, closest.strength, dist_atr)
                                })
                            };

                            if let Some((l_hash, l_kind, l_price, l_strength, dist_atr)) = level_info {
                                // Bounce prediction
                                let bounce_meta = PredictorMeta {
                                    predictor_id: 0,
                                    name: format!("level_bounce_ml_tf{}", tf_minutes),
                                    version: "1.0".to_string(),
                                    aspect: PredictionAspect::LevelBounce,
                                    calc_source: CalcSource::Ml,
                                    framework: "xgboost".to_string(),
                                    artifact_path: Some(self.config.model_path_levels.replace("{tf}", &tf_minutes.to_string())),
                                    feature_schema_id: schema.schema_id.clone(),
                                };
                                let bounce_pid = self.predictor_cache.get_or_register(&self.db_pool, &bounce_meta).await?;

                                let bounce_details = if is_history { None } else {
                                    Some(serde_json::json!({
                                        "method": "ml_level_bounce",
                                        "level_price": l_price,
                                        "prob_bounce": prob_bounce,
                                        "prob_break": prob_break,
                                        "level_info": { "hash": &l_hash, "kind": l_kind, "strength": l_strength, "distance_atr": dist_atr }
                                    }))
                                };

                                predictors.push(PredictionRow {
                                    time: view.timestamp,
                                    time_ms: view.timestamp.timestamp_millis(),
                                    symbol_id: sid,
                                    symbol: view.symbol.clone(),
                                    tf_minutes,
                                    horizon_bars: self.config.horizon_bars as i32,
                                    aspect: PredictionAspect::LevelBounce,
                                    calc_source: CalcSource::Ml,
                                    predictor_id: bounce_pid,
                                    score_norm: (prob_bounce as f32).clamp(0.0, 1.0),
                                    value: prob_bounce,
                                    value_low: Some((prob_bounce - 0.1).max(0.0)),
                                    value_high: Some((prob_bounce + 0.1).min(1.0)),
                                    side: Some(if view_close < l_price { 1 } else { -1 }),
                                    level_hash: Some(l_hash.clone()),
                                    level_kind: Some(l_kind),
                                    level_price: Some(l_price),
                                    level_strength: Some(l_strength),
                                    level_distance_atr: Some(dist_atr),
                                    candle_is_final: true,
                                    event_time_ms: None,
                                    details_json: bounce_details,
                                    prediction_key: format!("bounce_ml_{}_{}_{}", view.symbol, l_hash, view.timestamp.timestamp()),
                                });

                                // Breakout prediction
                                let break_meta = PredictorMeta {
                                    predictor_id: 0,
                                    name: format!("level_breakout_ml_tf{}", tf_minutes),
                                    version: "1.0".to_string(),
                                    aspect: PredictionAspect::LevelBreakout,
                                    calc_source: CalcSource::Ml,
                                    framework: "xgboost".to_string(),
                                    artifact_path: Some(self.config.model_path_levels.replace("{tf}", &tf_minutes.to_string())),
                                    feature_schema_id: schema.schema_id.clone(),
                                };
                                let break_pid = self.predictor_cache.get_or_register(&self.db_pool, &break_meta).await?;

                                let break_details = if is_history { None } else {
                                    Some(serde_json::json!({
                                        "method": "ml_level_breakout",
                                        "level_price": l_price,
                                        "prob_bounce": prob_bounce,
                                        "prob_break": prob_break,
                                        "level_info": { "hash": &l_hash, "kind": l_kind, "strength": l_strength, "distance_atr": dist_atr }
                                    }))
                                };

                                predictors.push(PredictionRow {
                                    time: view.timestamp,
                                    time_ms: view.timestamp.timestamp_millis(),
                                    symbol_id: sid,
                                    symbol: view.symbol.clone(),
                                    tf_minutes,
                                    horizon_bars: self.config.horizon_bars as i32,
                                    aspect: PredictionAspect::LevelBreakout,
                                    calc_source: CalcSource::Ml,
                                    predictor_id: break_pid,
                                    score_norm: (prob_break as f32).clamp(0.0, 1.0),
                                    value: prob_break,
                                    value_low: Some((prob_break - 0.1).max(0.0)),
                                    value_high: Some((prob_break + 0.1).min(1.0)),
                                    side: Some(if view_close < l_price { -1 } else { 1 }),
                                    level_hash: Some(l_hash.clone()),
                                    level_kind: Some(l_kind),
                                    level_price: Some(l_price),
                                    level_strength: Some(l_strength),
                                    level_distance_atr: Some(dist_atr),
                                    candle_is_final: true,
                                    event_time_ms: None,
                                    details_json: break_details,
                                    prediction_key: format!("breakout_ml_{}_{}_{}", view.symbol, l_hash, view.timestamp.timestamp()),
                                });
                            }
                        }
                    }
                },
                Ok(None) => {},
                Err(e) => {
                    tracing::error!(target: "compute_predictors",
                        "ML level prediction error for {}:{}: {}", view.symbol, view.timeframe, e);
                }
            }
        }

        if predictors.is_empty() {
            Ok(None)
        } else {
            Ok(Some(predictors))
        }
    }

    fn determine_side(&self, current_price: f64, target_price: f64) -> Option<i16> {
        if target_price > current_price * 1.001 { Some(1) } 
        else if target_price < current_price * 0.999 { Some(-1) } 
        else { Some(0) }
    }

    fn calculate_confidence_factors(&self, feature_view: &FeatureView) -> Value {
        serde_json::json!({
            "trend_strength": feature_view.indicators.trend_short,
            "momentum_strength": feature_view.indicators.macd_histogram,
            "volatility_regime": if feature_view.indicators.close != 0.0 { 
                Some(feature_view.indicators.atr / feature_view.indicators.close) 
            } else { None },
            "volume_confirmation": feature_view.indicators.volume_spike,
            "oscillator_alignment": self.calculate_oscillator_alignment(feature_view)
        })
    }

    fn calculate_oscillator_alignment(&self, feature_view: &FeatureView) -> f64 {
        let mut alignment_score = 0.0;
        let mut count = 0;
        
        let rsi = feature_view.indicators.rsi;
        if rsi > 30.0 && rsi < 70.0 { alignment_score += 1.0; } else { alignment_score -= 0.5; }
        count += 1;
        
        let stoch_k = feature_view.indicators.stoch_k;
        let stoch_d = feature_view.indicators.stoch_d;
        if (stoch_k > 20.0 && stoch_k < 80.0) && (stoch_d > 20.0 && stoch_d < 80.0) { 
            alignment_score += 1.0; 
        } else { 
            alignment_score -= 0.5; 
        }
        count += 1;
        
        let williams_r = feature_view.indicators.williams_r;
        if williams_r > -80.0 && williams_r < -20.0 { alignment_score += 1.0; } else { alignment_score -= 0.5; }
        count += 1;
        
        if count > 0 { alignment_score / count as f64 } else { 0.0 }
    }

    fn parse_timeframe_minutes(&self, timeframe: &str) -> Result<i32> {
        let num_str = &timeframe[..timeframe.len()-1];
        let unit = &timeframe[timeframe.len()-1..];
        let multiplier = match unit { 
            "m" => 1, 
            "h" => 60, 
            "d" => 24 * 60, 
            _ => anyhow::bail!("Unknown timeframe unit: {}", unit), 
        };
        let num: i32 = num_str.parse()?;
        Ok(num * multiplier)
    }
}

// Helper function to build batch for batch predictions
#[allow(dead_code)]
fn build_batch(schema: &crate::feature_schema::FeatureSchema, views: &[FeatureSnapshot]) -> (Vec<f32>, usize) {
    let ncol = schema.features.len();
    let mut flat = Vec::with_capacity(views.len() * ncol);
    for v in views {
        let view = FeatureView::new(
            v.timestamp,
            v.symbol.clone(),
            v.timeframe.clone(),
            v.is_realtime,
            v.indicators.clone(),
            v.raw_signals_data.as_ref(),
            v.sr_levels.clone(),
        )
        .expect("Failed to create FeatureView in build_batch");
        let fv = schema.build_vector(&view);
        flat.extend_from_slice(&fv);
    }
    (flat, ncol)
}

#[derive(Debug, Clone)]
pub struct FeatureSnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub symbol: String,
    pub timeframe: String,
    pub indicators: IndicatorsWideRow,
    pub raw_signals_data: Option<serde_json::Value>,
    pub sr_levels: Option<serde_json::Value>,
    pub is_realtime: bool,
}

impl FeatureSnapshot {
    pub fn is_realtime(&self) -> bool {
        self.is_realtime
    }
}

/// Build a comprehensive raw_signals_summary from ALL available indicators + SR levels.
/// This feeds into FinalScorer which uses multi-indicator directional scoring.
fn build_raw_signals_summary(
    indicators: &IndicatorsWideRow,
    sr_levels: Option<&serde_json::Value>,
) -> serde_json::Value {
    let close = indicators.close as f64;
    let atr = indicators.atr as f64;
    let atr_pct = if close > 0.0 { atr / close } else { 0.0 };

    let trend_strength = ((indicators.adx as f64 - 15.0) / 25.0).clamp(0.0, 1.0);

    let rsi = indicators.rsi as f64;
    let macd_hist = indicators.macd_histogram as f64;
    let rsi_dev = (rsi - 50.0).abs() / 50.0;
    let macd_norm = macd_hist.abs().min(1.0);
    let momentum_strength = (0.6 * rsi_dev + 0.4 * macd_norm).clamp(0.0, 1.0);

    let volatility_regime = ((atr_pct - 0.006) / 0.034).clamp(0.0, 1.0);
    let volume_spike_score = (indicators.volume_spike as f64).clamp(0.0, 1.0);

    let dominant_side: i64 = if rsi > 55.0 && macd_hist > 0.0 { 1 }
        else if rsi < 45.0 && macd_hist < 0.0 { -1 }
        else { 0 };

    let best_momentum = momentum_strength;
    let best_volume = volume_spike_score;
    let best_levels = ((trend_strength * 0.5 + momentum_strength * 0.3 + volume_spike_score * 0.2) * 1.1).clamp(0.0, 1.0);
    let best_raw = ((trend_strength + momentum_strength) / 2.0).clamp(0.0, 1.0);

    // Extract SR level prices from JSON if available
    let sr = sr_levels.unwrap_or(&serde_json::Value::Null);
    let get_sr = |key: &str| -> serde_json::Value {
        sr.get(key)
            .and_then(|v| v.as_f64())
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| serde_json::json!(v))
            .unwrap_or(serde_json::Value::Null)
    };

    serde_json::json!({
        "atr": indicators.atr,
        "atr_pct": atr_pct,
        "side": dominant_side,
        "dominant_side": dominant_side,
        "trend_strength": trend_strength,
        "momentum_strength": momentum_strength,
        "volatility_regime": volatility_regime,
        "volume_spike_score": volume_spike_score,
        "best_raw_signal_score": best_raw,
        "best_levels_score": best_levels,
        "best_momentum_score": best_momentum,
        "best_volume_score": best_volume,
        "feature_coverage": 1.0,

        // === ALL directional indicators for comprehensive scoring ===
        // Price & EMAs
        "close": indicators.close,
        "open": indicators.open,
        "high": indicators.high,
        "low": indicators.low,
        "ema_20": indicators.ema_20,
        "ema_50": indicators.ema_50,
        "ema_200": indicators.ema_200,
        "sma": indicators.sma,

        // Trend indicators (short → medium → long for multi-TF confirmation)
        "trend_short": indicators.trend_short,
        "trend_medium": indicators.trend_medium,
        "trend_long": indicators.trend_long,
        "adx": indicators.adx,

        // Momentum oscillators
        "rsi": indicators.rsi,
        "macd_hist": indicators.macd_histogram,
        "macd_line": indicators.macd_line,
        "macd_signal": indicators.macd_signal,
        "stoch_k": indicators.stoch_k,
        "stoch_d": indicators.stoch_d,
        "williams_r": indicators.williams_r,
        "cci": indicators.cci,

        // Bollinger Bands
        "bb_upper": indicators.bb_upper,
        "bb_lower": indicators.bb_lower,
        "bb_middle": indicators.bb_middle,

        // Volume indicators
        "volume": indicators.volume,
        "volume_sma": indicators.volume_sma,
        "vwap": indicators.vwap,
        "obv": indicators.obv,

        // Support/Resistance levels (primary strategy component)
        "sr_levels": {
            "strong_support": get_sr("strong_support"),
            "mid_support": get_sr("mid_support"),
            "light_support": get_sr("light_support"),
            "strong_resistance": get_sr("strong_resistance"),
            "mid_resistance": get_sr("mid_resistance"),
            "light_resistance": get_sr("light_resistance"),
        },
    })
}
