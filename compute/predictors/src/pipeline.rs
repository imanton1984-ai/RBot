// compute/predictors/pipeline.rs

use anyhow::Result;
use serde_json::Value;
use crate::types::{PredictionRow, PredictionAspect, CalcSource};
use crate::feature_view::{FeatureView, IndicatorsWideRow};
use crate::level_view::LevelView;
use crate::config::PredictorsConfig;
use crate::persistence::{self, resolve_symbol_id};
use sqlx::PgPool;
use common::MessageBus;
use crate::consensus::ConsensusEngine;

use crate::ml::model_manager::ModelManager;
use crate::ml::model_pool::ModelPool;
use std::sync::Arc;

pub struct PredictorsPipeline {
    config: PredictorsConfig,
    db_pool: PgPool,
    shutdown_rx: tokio::sync::broadcast::Receiver<bool>,
    feature_rx: Option<tokio::sync::mpsc::UnboundedReceiver<FeatureSnapshot>>,
    ml_manager: ModelManager,
}

impl PredictorsPipeline {
    pub fn new(
        config: PredictorsConfig,
        db_pool: PgPool,
        _message_bus: MessageBus,
        shutdown_rx: tokio::sync::broadcast::Receiver<bool>,
    ) -> Self {
        let pool = Arc::new(ModelPool::new());
        let mut ml_manager = ModelManager::new(config.use_cuda, pool);
        // Загружаем модели, пути берем из конфига
        if let Err(e) = ml_manager.load_model("price", &config.model_path_price) {
            tracing::error!("Failed to load price model: {}", e);
        }
        if let Err(e) = ml_manager.load_model("level", &config.model_path_levels) {
            tracing::error!("Failed to load level model: {}", e);
        }

        Self {
            config,
            db_pool,
            shutdown_rx,
            feature_rx: None,
            ml_manager,
        }
    }
    
    pub async fn run(&mut self) -> Result<()> {
        let mut rx = self.feature_rx.take().expect("Feature receiver must be set before calling run()");
        
        while !self.is_shutdown().await {
            tokio::select! {
                feature_snapshot = rx.recv() => {
                    match feature_snapshot {
                        Some(snapshot) => {
                            if let Err(e) = self.process_feature_snapshot(snapshot).await {
                                tracing::error!("Error processing feature snapshot: {}", e);
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

        Ok(())
    }
    
    pub fn set_input_receiver(&mut self, receiver: tokio::sync::mpsc::UnboundedReceiver<FeatureSnapshot>) {
        self.feature_rx = Some(receiver);
    }

    async fn is_shutdown(&mut self) -> bool {
        match self.shutdown_rx.try_recv() {
            Ok(_) | Err(tokio::sync::broadcast::error::TryRecvError::Closed) => true,
            _ => false,
        }
    }

    async fn process_feature_snapshot(&self, snapshot: FeatureSnapshot) -> Result<()> {
        let start = std::time::Instant::now();

        tracing::info!(target: "compute_predictors",
            "Processing feature snapshot: Symbol: {}, TF: {}, Timestamp: {}",
            snapshot.symbol, snapshot.timeframe, snapshot.timestamp
        );

        let view = FeatureView::new(
            snapshot.timestamp,
            snapshot.symbol.clone(),
            snapshot.timeframe.clone(),
            snapshot.indicators,
            snapshot.raw_signals_data.as_ref(),
            snapshot.sr_levels,
        )?;

        tracing::debug!(target: "compute_predictors",
            "Feature view created successfully for {}:{}", snapshot.symbol, snapshot.timeframe
        );

        let (hc_res, ml_res) = tokio::join!(
            self.run_hardcode_predictors(&view),
            self.run_ml_predictors(&view)
        );

        let hc_preds = match hc_res {
            Ok(preds_opt) => {
                let preds = preds_opt.unwrap_or_default();
                tracing::debug!(target: "compute_predictors", 
                    "Hardcode predictors result: {} predictions for {}:{}",
                    preds.len(), snapshot.symbol, snapshot.timeframe
                );
                preds
            },
            Err(e) => {
                tracing::error!(target: "compute_predictors",
                    "Error in hardcode predictors: {} for {}:{}",
                    e, snapshot.symbol, snapshot.timeframe
                );
                Vec::new()
            }
        };

        let ml_preds = match ml_res {
            Ok(preds_opt) => {
                let preds = preds_opt.unwrap_or_default();
                tracing::debug!(target: "compute_predictors",
                    "ML predictors result: {} predictions for {}:{}",
                    preds.len(), snapshot.symbol, snapshot.timeframe
                );
                preds
            },
            Err(e) => {
                tracing::error!(target: "compute_predictors",
                    "Error in ML predictors: {} for {}:{}",
                    e, snapshot.symbol, snapshot.timeframe
                );
                Vec::new()
            }
        };

        tracing::debug!(target: "compute_predictors",
            "Before consensus: HC: {}, ML: {} for {}:{}",
            hc_preds.len(), ml_preds.len(), snapshot.symbol, snapshot.timeframe
        );

        let consensus = ConsensusEngine::new();
        let final_preds = match consensus.apply_gate_and_fuse(hc_preds, ml_preds, &view).await {
            Ok(preds) => {
                tracing::debug!(target: "compute_predictors",
                    "After consensus: {} final predictions for {}:{}",
                    preds.len(), snapshot.symbol, snapshot.timeframe
                );
                preds
            },
            Err(e) => {
                tracing::error!(target: "compute_predictors",
                    "Error in consensus engine: {} for {}:{}",
                    e, snapshot.symbol, snapshot.timeframe
                );
                Vec::new()
            }
        };

        if !final_preds.is_empty() {
            match persistence::upsert_predictors(&self.db_pool, final_preds.clone()).await {
                Ok(_) => {
                    tracing::info!(target: "compute_predictors",
                        "Successfully saved {} predictions to DB for {}:{}, Time: {:?}",
                        final_preds.len(), snapshot.symbol, snapshot.timeframe, start.elapsed()
                    );
                },
                Err(e) => {
                    tracing::error!(target: "compute_predictors",
                        "Failed to save predictions to DB: {} for {}:{}, Time: {:?}",
                        e, snapshot.symbol, snapshot.timeframe, start.elapsed()
                    );
                }
            }
        } else {
            tracing::debug!(target: "compute_predictors",
                "No final predictions to save for {}:{}, Time: {:?}",
                snapshot.symbol, snapshot.timeframe, start.elapsed()
            );
        }

        Ok(())
    }

    async fn run_hardcode_predictors(&self, view: &FeatureView) -> Result<Option<Vec<PredictionRow>>> {
        let mut predictors = Vec::new();
        let view_close = view.indicators.close as f64;

        tracing::debug!(target: "compute_predictors",
            "Running hardcode predictors for {}:{}, close price: {}, ATR: {}",
            view.symbol, view.timeframe, view_close, view.indicators.atr
        );

        // Price Target predictions
        let hc_price = crate::future_predictor::heuristic_predictor::FuturePriceHeuristic::new();
        match hc_price.predict(&view.symbol, &view.timeframe, view) {
            Ok(Some((predicted_prices, score))) => {
                tracing::debug!(target: "compute_predictors",
                    "Price predictor succeeded for {}:{}, score: {}, prices: {:?}",
                    view.symbol, view.timeframe, score, predicted_prices
                );
                
                let sid = resolve_symbol_id(&self.db_pool, &view.symbol).await?;
                let predictor_meta = crate::types::PredictorMeta {
                    predictor_id: 0,
                    name: "price10_hard".to_string(),
                    version: "1.0".to_string(),
                    aspect: PredictionAspect::PriceTarget,
                    calc_source: CalcSource::Hard,
                    framework: "hardcode".to_string(),
                    artifact_path: None,
                    feature_schema_id: "feature_view".to_string(),
                };
                let predictor_id = crate::persistence::register_predictor_if_missing(&self.db_pool, &predictor_meta).await?.0;
                
                // Calculate value_low and value_high based on uncertainty
                let last_predicted_price = predicted_prices.last().copied().unwrap_or(view_close);
                let atr = view.indicators.atr as f64;
                let uncertainty_factor = 0.5; // Half ATR as uncertainty
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
                    score_norm: score.abs() as f32,
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
                    details_json: Some(serde_json::json!({
                        "method": "hardcode_price10",
                        "predicted_prices": predicted_prices,
                        "confidence_factors": self.calculate_confidence_factors(view)
                    })),
                    prediction_key: format!("price10_hard_{}_{}", view.symbol, view.timestamp.timestamp()),
                };
                predictors.push(prediction_row);
            },
            Ok(None) => {
                tracing::debug!(target: "compute_predictors",
                    "Price predictor returned None for {}:{}",
                    view.symbol, view.timeframe
                );
            },
            Err(e) => {
                tracing::error!(target: "compute_predictors",
                    "Price predictor error for {}:{}: {}",
                    view.symbol, view.timeframe, e
                );
            }
        }

        // Level predictions (Bounce and Breakout)
        let hc_level = crate::level_predictor::future_heruistic_predictor::LevelPredictorHeuristic::new();
        match hc_level.predict(&view.symbol, &view.timeframe, view) {
            Ok(Some((level, prob_bounce, prob_break, score))) => {
                tracing::info!(target: "compute_predictors",
                    "Level predictor succeeded for {}:{}, level: {}, bounce_prob: {}, break_prob: {}, score: {}",
                    view.symbol, view.timeframe, level.level_price, prob_bounce, prob_break, score
                );
                
                let sid = resolve_symbol_id(&self.db_pool, &view.symbol).await?;
                
                // Level Bounce prediction
                let bounce_predictor_meta = crate::types::PredictorMeta {
                    predictor_id: 0,
                    name: "level_bounce_hard".to_string(),
                    version: "1.0".to_string(),
                    aspect: PredictionAspect::LevelBounce,
                    calc_source: CalcSource::Hard,
                    framework: "hardcode".to_string(),
                    artifact_path: None,
                    feature_schema_id: "feature_view".to_string(),
                };
                let bounce_predictor_id = crate::persistence::register_predictor_if_missing(&self.db_pool, &bounce_predictor_meta).await?.0;
                
                let bounce_prediction = PredictionRow {
                    time: view.timestamp,
                    time_ms: view.timestamp.timestamp_millis(),
                    symbol_id: sid,
                    symbol: view.symbol.clone(),
                    tf_minutes: self.parse_timeframe_minutes(&view.timeframe)?,
                    horizon_bars: self.config.horizon_bars as i32,
                    aspect: PredictionAspect::LevelBounce,
                    calc_source: CalcSource::Hard,
                    predictor_id: bounce_predictor_id,
                    score_norm: prob_bounce as f32,
                    value: prob_bounce,
                    value_low: Some((prob_bounce - 0.1).max(0.0)), // Add uncertainty range
                    value_high: Some((prob_bounce + 0.1).min(1.0)),
                    side: Some(if view_close < level.level_price { 1 } else { -1 }),
                    level_hash: Some(level.level_hash.clone()),
                    level_kind: Some(level.level_kind.as_i16()),
                    level_price: Some(level.level_price),
                    level_strength: Some(level.level_strength),
                    level_distance_atr: Some(level.distance_atr),
                    candle_is_final: true,
                    event_time_ms: None,
                    details_json: Some(serde_json::json!({
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
                    })),
                    prediction_key: format!("bounce_hard_{}_{}_{}", view.symbol, level.level_hash, view.timestamp.timestamp()),
                };
                predictors.push(bounce_prediction);
                
                // Level Breakout prediction
                let break_predictor_meta = crate::types::PredictorMeta {
                    predictor_id: 0,
                    name: "level_breakout_hard".to_string(),
                    version: "1.0".to_string(),
                    aspect: PredictionAspect::LevelBreakout,
                    calc_source: CalcSource::Hard,
                    framework: "hardcode".to_string(),
                    artifact_path: None,
                    feature_schema_id: "feature_view".to_string(),
                };
                let break_predictor_id = crate::persistence::register_predictor_if_missing(&self.db_pool, &break_predictor_meta).await?.0;
                
                let break_prediction = PredictionRow {
                    time: view.timestamp,
                    time_ms: view.timestamp.timestamp_millis(),
                    symbol_id: sid,
                    symbol: view.symbol.clone(),
                    tf_minutes: self.parse_timeframe_minutes(&view.timeframe)?,
                    horizon_bars: self.config.horizon_bars as i32,
                    aspect: PredictionAspect::LevelBreakout,
                    calc_source: CalcSource::Hard,
                    predictor_id: break_predictor_id,
                    score_norm: prob_break as f32,
                    value: prob_break,
                    value_low: Some((prob_break - 0.1).max(0.0)), // Add uncertainty range
                    value_high: Some((prob_break + 0.1).min(1.0)),
                    side: Some(if view_close < level.level_price { -1 } else { 1 }),
                    level_hash: Some(level.level_hash.clone()),
                    level_kind: Some(level.level_kind.as_i16()),
                    level_price: Some(level.level_price),
                    level_strength: Some(level.level_strength),
                    level_distance_atr: Some(level.distance_atr),
                    candle_is_final: true,
                    event_time_ms: None,
                    details_json: Some(serde_json::json!({
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
                    })),
                    prediction_key: format!("breakout_hard_{}_{}_{}", view.symbol, level.level_hash, view.timestamp.timestamp()),
                };
                predictors.push(break_prediction);
            },
            Ok(None) => {
                tracing::debug!(target: "compute_predictors",
                    "Level predictor returned None for {}:{}",
                    view.symbol, view.timeframe
                );
            },
            Err(e) => {
                tracing::error!(target: "compute_predictors",
                    "Level predictor error for {}:{}: {}",
                    view.symbol, view.timeframe, e
                );
            }
        }

        tracing::debug!(target: "compute_predictors",
            "Hardcode predictors total: {} predictions for {}:{}",
            predictors.len(), view.symbol, view.timeframe
        );

        if predictors.is_empty() {
            Ok(None)
        } else {
            Ok(Some(predictors))
        }
    }

    async fn run_ml_predictors(&self, view: &FeatureView) -> Result<Option<Vec<PredictionRow>>> {
        let mut predictors = Vec::new();
        let view_close = view.indicators.close as f64;

        tracing::debug!(target: "compute_predictors",
            "Running ML predictors for {}:{}, checking if models are loaded",
            view.symbol, view.timeframe
        );

        let sid = resolve_symbol_id(&self.db_pool, &view.symbol).await?;

        if let Some(schema) = self.ml_manager.get_schema("price") {
            tracing::debug!(target: "compute_predictors",
                "Price schema found for {}:{}, proceeding with ML prediction",
                view.symbol, view.timeframe
            );
            
            let input_vec = schema.build_vector(view);
            match self.ml_manager.predict("price", &input_vec) {
                Ok(Some(predicted_prices)) => {
                    tracing::info!(target: "compute_predictors",
                        "ML price prediction succeeded for {}:{}, prices: {:?}",
                        view.symbol, view.timeframe, predicted_prices
                    );
                    
                    let predictor_meta = crate::types::PredictorMeta {
                        predictor_id: 0,
                        name: "price10_ml".to_string(),
                        version: "1.0".to_string(),
                        aspect: PredictionAspect::PriceTarget,
                        calc_source: CalcSource::Ml,
                        framework: "onnx".to_string(),
                        artifact_path: Some(self.config.model_path_price.clone()),
                        feature_schema_id: schema.schema_id.clone(),
                    };
                    let predictor_id = crate::persistence::register_predictor_if_missing(&self.db_pool, &predictor_meta).await?.0;
                    let score = if !predicted_prices.is_empty() { predicted_prices[0].abs().min(1.0) as f64 } else { 0.5 };
                    let last_predicted_price = predicted_prices.last().copied().unwrap_or(view_close as f32) as f64;
                    // Ensure the predicted price is positive
                    let safe_predicted_price = last_predicted_price.max(0.00000001);
                    
                    // Calculate value_low and value_high based on uncertainty
                    let atr = view.indicators.atr as f64;
                    let uncertainty_factor = 0.5; // Half ATR as uncertainty
                    let value_low = Some((safe_predicted_price - atr * uncertainty_factor).max(0.00000001));
                    let value_high = Some(safe_predicted_price + atr * uncertainty_factor);

                    let prediction_row = PredictionRow {
                        time: view.timestamp,
                        time_ms: view.timestamp.timestamp_millis(),
                        symbol_id: sid,
                        symbol: view.symbol.clone(),
                        tf_minutes: self.parse_timeframe_minutes(&view.timeframe)?,
                        horizon_bars: self.config.horizon_bars as i32,
                        aspect: PredictionAspect::PriceTarget,
                        calc_source: CalcSource::Ml,
                        predictor_id,
                        score_norm: score as f32,
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
                        details_json: Some(serde_json::json!({
                            "method": "ml_price10",
                            "predicted_prices": predicted_prices,
                            "confidence": score,
                            "model_used": &self.config.model_path_price
                        })),
                        prediction_key: format!("price10_ml_{}_{}", view.symbol, view.timestamp.timestamp()),
                    };
                    predictors.push(prediction_row);
                },
                Ok(None) => {
                    tracing::debug!(target: "compute_predictors",
                        "ML price prediction returned None for {}:{}",
                        view.symbol, view.timeframe
                    );
                },
                Err(e) => {
                    tracing::error!(target: "compute_predictors",
                        "ML price prediction error for {}:{}: {}",
                        view.symbol, view.timeframe, e
                    );
                }
            }
        } else {
            tracing::debug!(target: "compute_predictors",
                "No price schema found for {}:{}, ML price predictor disabled",
                view.symbol, view.timeframe
            );
        }

        if let Some(schema) = self.ml_manager.get_schema("level") {
            tracing::debug!(target: "compute_predictors",
                "Level schema found for {}:{}, proceeding with ML level prediction",
                view.symbol, view.timeframe
            );
            
            let input_vec = schema.build_vector(view);
            match self.ml_manager.predict("level", &input_vec) {
                Ok(Some(level_outputs)) => {
                    if level_outputs.len() >= 2 {
                        let prob_bounce = level_outputs[0].max(0.0).min(1.0) as f64;
                        let prob_break = level_outputs[1].max(0.0).min(1.0) as f64;
                        let score = ((prob_bounce + prob_break) / 2.0).min(1.0) as f64;
                        
                        match view.get_sr_levels() {
                            Ok(sr_levels) => {
                                if !sr_levels.is_empty() {
                                    let atr = view.indicators.atr as f64;
                                    match LevelView::new(sr_levels, view_close, atr, view.timestamp, view.symbol.clone(), view.timeframe.clone(), 2) {
                                        Ok(level_view) => {
                                            let levels = level_view.get_near_levels();
                                            if let Some(level) = levels.first() {
                                                tracing::info!(target: "compute_predictors",
                                                    "ML level prediction succeeded for {}:{}, level: {}, bounce_prob: {}, break_prob: {}",
                                                    view.symbol, view.timeframe, level.level_price, prob_bounce, prob_break
                                                );
                                                
                                                // Bounce
                                                let bounce_predictor_meta = crate::types::PredictorMeta { 
                                                    predictor_id: 0, 
                                                    name: "level_bounce_ml".to_string(), 
                                                    version: "1.0".to_string(), 
                                                    aspect: PredictionAspect::LevelBounce, 
                                                    calc_source: CalcSource::Ml, 
                                                    framework: "onnx".to_string(), 
                                                    artifact_path: Some(self.config.model_path_levels.clone()), 
                                                    feature_schema_id: schema.schema_id.clone() 
                                                };
                                                let bounce_predictor_id = crate::persistence::register_predictor_if_missing(&self.db_pool, &bounce_predictor_meta).await?.0;
                                                let bounce_prediction = PredictionRow { 
                                                    time: view.timestamp, 
                                                    time_ms: view.timestamp.timestamp_millis(), 
                                                    symbol_id: sid, 
                                                    symbol: view.symbol.clone(), 
                                                    tf_minutes: self.parse_timeframe_minutes(&view.timeframe)?, 
                                                    horizon_bars: self.config.horizon_bars as i32, 
                                                    aspect: PredictionAspect::LevelBounce, 
                                                    calc_source: CalcSource::Ml, 
                                                    predictor_id: bounce_predictor_id, 
                                                    score_norm: prob_bounce as f32, 
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
                                                    details_json: Some(serde_json::json!({ 
                                                        "method": "ml_level_bounce", 
                                                        "level_price": level.level_price, 
                                                        "prob_bounce": prob_bounce, 
                                                        "prob_break": prob_break, 
                                                        "score": score, 
                                                        "model_used": &self.config.model_path_levels 
                                                    })), 
                                                    prediction_key: format!("bounce_ml_{}_{}_{}", view.symbol, level.level_hash, view.timestamp.timestamp()) 
                                                };
                                                predictors.push(bounce_prediction);
                                                
                                                // Breakout
                                                let break_predictor_meta = crate::types::PredictorMeta { 
                                                    predictor_id: 0, 
                                                    name: "level_breakout_ml".to_string(), 
                                                    version: "1.0".to_string(), 
                                                    aspect: PredictionAspect::LevelBreakout, 
                                                    calc_source: CalcSource::Ml, 
                                                    framework: "onnx".to_string(), 
                                                    artifact_path: Some(self.config.model_path_levels.clone()), 
                                                    feature_schema_id: schema.schema_id.clone() 
                                                };
                                                let break_predictor_id = crate::persistence::register_predictor_if_missing(&self.db_pool, &break_predictor_meta).await?.0;
                                                let break_prediction = PredictionRow { 
                                                    time: view.timestamp, 
                                                    time_ms: view.timestamp.timestamp_millis(), 
                                                    symbol_id: sid, 
                                                    symbol: view.symbol.clone(), 
                                                    tf_minutes: self.parse_timeframe_minutes(&view.timeframe)?, 
                                                    horizon_bars: self.config.horizon_bars as i32, 
                                                    aspect: PredictionAspect::LevelBreakout, 
                                                    calc_source: CalcSource::Ml, 
                                                    predictor_id: break_predictor_id, 
                                                    score_norm: prob_break as f32, 
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
                                                    details_json: Some(serde_json::json!({ 
                                                        "method": "ml_level_breakout", 
                                                        "level_price": level.level_price, 
                                                        "prob_bounce": prob_bounce, 
                                                        "prob_break": prob_break, 
                                                        "score": score, 
                                                        "model_used": &self.config.model_path_levels 
                                                    })), 
                                                    prediction_key: format!("breakout_ml_{}_{}_{}", view.symbol, level.level_hash, view.timestamp.timestamp()) 
                                                };
                                                predictors.push(break_prediction);
                                            } else {
                                                tracing::debug!(target: "compute_predictors",
                                                    "No near levels found for ML level prediction for {}:{}",
                                                    view.symbol, view.timeframe
                                                );
                                            }
                                        },
                                        Err(e) => {
                                            tracing::error!(target: "compute_predictors",
                                                "Failed to create LevelView for {}:{}: {}",
                                                view.symbol, view.timeframe, e
                                            );
                                        }
                                    }
                                } else {
                                    tracing::debug!(target: "compute_predictors",
                                        "No SR levels available for ML level prediction for {}:{}",
                                        view.symbol, view.timeframe
                                    );
                                }
                            },
                            Err(e) => {
                                tracing::error!(target: "compute_predictors",
                                    "Failed to get SR levels for {}:{}: {}",
                                    view.symbol, view.timeframe, e
                                );
                            }
                        }
                    } else {
                        tracing::warn!(target: "compute_predictors",
                            "ML level prediction returned insufficient outputs ({} < 2) for {}:{}",
                            level_outputs.len(), view.symbol, view.timeframe
                        );
                    }
                },
                Ok(None) => {
                    tracing::debug!(target: "compute_predictors",
                        "ML level prediction returned None for {}:{}",
                        view.symbol, view.timeframe
                    );
                },
                Err(e) => {
                    tracing::error!(target: "compute_predictors",
                        "ML level prediction error for {}:{}: {}",
                        view.symbol, view.timeframe, e
                    );
                }
            }
        } else {
            tracing::debug!(target: "compute_predictors",
                "No level schema found for {}:{}, ML level predictor disabled",
                view.symbol, view.timeframe
            );
        }

        tracing::debug!(target: "compute_predictors",
            "ML predictors total: {} predictions for {}:{}",
            predictors.len(), view.symbol, view.timeframe
        );

        if predictors.is_empty() {
            Ok(None)
        } else {
            Ok(Some(predictors))
        }
    }


    fn determine_side(&self, current_price: f64, target_price: f64) -> Option<i16> {
        if target_price > current_price * 1.001 { Some(1) } else if target_price < current_price * 0.999 { Some(-1) } else { Some(0) }
    }

    fn calculate_confidence_factors(&self, feature_view: &FeatureView) -> Value {
        serde_json::json!({
            "trend_strength": feature_view.indicators.trend_short,
            "momentum_strength": feature_view.indicators.macd_histogram,
            "volatility_regime": if feature_view.indicators.close != 0.0 { Some(feature_view.indicators.atr / feature_view.indicators.close) } else { None },
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
        if (stoch_k > 20.0 && stoch_k < 80.0) && (stoch_d > 20.0 && stoch_d < 80.0) { alignment_score += 1.0; } else { alignment_score -= 0.5; }
        count += 1;
        let williams_r = feature_view.indicators.williams_r;
        if williams_r > -80.0 && williams_r < -20.0 { alignment_score += 1.0; } else { alignment_score -= 0.5; }
        count += 1;
        if count > 0 { alignment_score / count as f64 } else { 0.0 }
    }

    fn parse_timeframe_minutes(&self, timeframe: &str) -> Result<i32> {
        let num_str = &timeframe[..timeframe.len()-1];
        let unit = &timeframe[timeframe.len()-1..];
        let multiplier = match unit { "m" => 1, "h" => 60, "d" => 24 * 60, _ => anyhow::bail!("Unknown timeframe unit: {}", unit), };
        let num: i32 = num_str.parse()?;
        Ok(num * multiplier)
    }
}

#[derive(Debug, Clone)]
pub struct FeatureSnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub symbol: String,
    pub timeframe: String,
    pub indicators: IndicatorsWideRow,
    pub raw_signals_data: Option<serde_json::Value>,
    pub sr_levels: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse_timeframe_minutes() {
    }
}