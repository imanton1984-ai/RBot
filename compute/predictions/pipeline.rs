// compute/predictions/pipeline.rs

use anyhow::Result;
use tokio::sync::mpsc;
use serde_json::Value;
use crate::predictions::types::{PredictionRow, PredictionAspect, CalcSource};
use crate::predictions::feature_view::{FeatureView, FeatureVector};
use crate::predictions::level_view::LevelView;
use crate::predictions::config::PredictionsConfig;
use crate::predictions::persistence;
use database_lib::PgPool;
use connections_lib::RedpandaClient;

pub struct PredictionsPipeline {
    config: PredictionsConfig,
    db_pool: PgPool,
    redpanda_client: RedpandaClient,
    shutdown_rx: tokio::sync::broadcast::Receiver<bool>,
}

impl PredictionsPipeline {
    pub fn new(
        config: PredictionsConfig,
        db_pool: PgPool,
        redpanda_client: RedpandaClient,
        shutdown_rx: tokio::sync::broadcast::Receiver<bool>,
    ) -> Self {
        Self {
            config,
            db_pool,
            redpanda_client,
            shutdown_rx,
        }
    }

    pub async fn run(&mut self) -> Result<()> {
        // Subscribe to feature snapshots
        let (tx, mut rx) = mpsc::unbounded_channel();

        // Start consumer for feature snapshots
        self.start_feature_consumer(tx).await?;

        // Main processing loop
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
                            // Channel closed, break the loop
                            break;
                        }
                    }
                }
                _ = self.shutdown_rx.recv() => {
                    tracing::info!("Shutdown signal received, stopping predictions pipeline");
                    break;
                }
            }
        }

        Ok(())
    }

    async fn is_shutdown(&mut self) -> bool {
        // Check if shutdown signal was sent
        match self.shutdown_rx.try_recv() {
            Ok(_) => true,
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => true,
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => false,
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => false,
        }
    }

    async fn start_feature_consumer(&self, tx: mpsc::UnboundedSender<FeatureSnapshot>) -> Result<()> {
        // This would connect to the feature snapshot topic and forward messages to the channel
        // Implementation depends on your Redpanda setup

        // For now, we'll simulate this by spawning a task that would consume from the topic
        let db_pool = self.db_pool.clone();
        let redpanda_client = self.redpanda_client.clone();
        let topic = self.redpanda_client.config.topic_features_snapshot.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            // In a real implementation, this would consume from the features snapshot topic
            // and forward the data to the tx channel
            tracing::info!("Started feature consumer for topic: {}", topic);
        });

        Ok(())
    }

    async fn process_feature_snapshot(&self, snapshot: FeatureSnapshot) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Create feature view from the snapshot
        let feature_view = FeatureView::new(
            snapshot.timestamp,
            snapshot.symbol.clone(),
            snapshot.timeframe.clone(),
            &snapshot.indicators_data,
            snapshot.raw_signals_data.as_ref(),
        )?;

        // Convert to feature vector for ML models
        let feature_vector = feature_view.to_feature_vector();

        // Run all predictors
        let mut all_predictions = Vec::new();

        // Run hardcode predictors
        if let Some(hard_predictions) = self.run_hardcode_predictors(&feature_view, &feature_vector).await? {
            all_predictions.extend(hard_predictions);
        }

        // Run ML predictors
        if let Some(ml_predictions) = self.run_ml_predictors(&feature_view, &feature_vector).await? {
            all_predictions.extend(ml_predictions);
        }

        // Apply consensus logic if both hardcode and ML predictions exist
        let final_predictions = self.apply_consensus_logic(all_predictions, &feature_view).await?;

        // Filter predictions by minimum score
        let filtered_predictions: Vec<PredictionRow> = final_predictions
            .into_iter()
            .filter(|pred| pred.score_norm as f64 >= self.config.min_store_score)
            .collect();

        if !filtered_predictions.is_empty() {
            // Persist predictions to database
            persistence::upsert_predictions(&self.db_pool, filtered_predictions).await?;

            // Optionally publish to predictions topic
            self.publish_predictions(&snapshot, &feature_view).await?;
        }

        Ok(())
    }

    async fn run_hardcode_predictors(
        &self,
        feature_view: &FeatureView,
        feature_vector: &FeatureVector,
    ) -> Result<Option<Vec<PredictionRow>>> {
        if !self.config.enabled {
            return Ok(None);
        }

        let mut predictions = Vec::new();

        // Run price target predictor (hardcode)
        if let Some(price_pred) = self.run_hardcode_price_predictor(feature_view).await? {
            predictions.push(price_pred);
        }

        // Run level predictors (bounce/breakout)
        if let Some(level_preds) = self.run_hardcode_level_predictors(feature_view).await? {
            predictions.extend(level_preds);
        }

        Ok(Some(predictions))
    }

    async fn run_hardcode_price_predictor(&self, feature_view: &FeatureView) -> Result<Option<PredictionRow>> {
        // Import the heuristic predictor
        use crate::predictions::future_price::heuristic_predictor::FuturePriceHeuristic;
        use crate::predictions::future_price::heuristic_predictor::Backend;

        // Create a dummy backend for now (would use actual compute backend in real implementation)
        let backend = Backend::Cpu; // Placeholder
        let predictor = FuturePriceHeuristic::new(backend);

        // Run prediction
        if let Some((predicted_prices, score)) = predictor.predict(
            &feature_view.symbol,
            &feature_view.timeframe,
            feature_view,
        ).await? {
            if score >= self.config.min_store_score {
                // Calculate target price for horizon_bars (10)
                let target_price = if predicted_prices.len() > self.config.horizon_bars {
                    predicted_prices[self.config.horizon_bars - 1]
                } else {
                    predicted_prices.last().copied().unwrap_or(feature_view.close)
                };

                // Register predictor if missing
                let predictor_meta = persistence::PredictorMeta {
                    predictor_id: 0, // Will be filled by register_predictor_if_missing
                    name: "price10_hard".to_string(),
                    version: "1.0".to_string(),
                    aspect: PredictionAspect::PriceTarget,
                    calc_source: CalcSource::Hard,
                    framework: "hardcode".to_string(),
                    artifact_path: None,
                    feature_schema_id: feature_vector.schema_id.clone(),
                };

                let predictor_id = persistence::register_predictor_if_missing(&self.db_pool, &predictor_meta).await?.0;

                // Create prediction row
                let prediction_row = PredictionRow {
                    time: feature_view.timestamp,
                    time_ms: feature_view.timestamp.timestamp_millis(),
                    symbol_id: 0, // Would come from symbol lookup
                    symbol: feature_view.symbol.clone(),
                    tf_minutes: self.parse_timeframe_minutes(&feature_view.timeframe)?,
                    horizon_bars: self.config.horizon_bars as i32,
                    aspect: PredictionAspect::PriceTarget,
                    calc_source: CalcSource::Hard,
                    predictor_id,
                    score_norm: score as f32,
                    value: target_price,
                    value_low: None, // Could be calculated from bands
                    value_high: None, // Could be calculated from bands
                    side: self.determine_side(feature_view.close, target_price),
                    level_hash: None,
                    level_kind: None,
                    level_price: None,
                    level_strength: None,
                    level_distance_atr: None,
                    candle_is_final: true, // Assuming final candle
                    event_time_ms: None,
                    details_json: Some(serde_json::json!({
                        "method": "hardcode_price10",
                        "predicted_prices": predicted_prices,
                        "confidence_factors": self.calculate_confidence_factors(feature_view)
                    })),
                    prediction_key: format!("price10_hard_{}_{}", feature_view.symbol, feature_view.timestamp.timestamp()),
                };

                return Ok(Some(prediction_row));
            }
        }

        Ok(None)
    }

    async fn run_hardcode_level_predictors(&self, feature_view: &FeatureView) -> Result<Option<Vec<PredictionRow>>> {
        // Parse SR levels from JSON
        let sr_levels = feature_view.get_sr_levels()?;
        if sr_levels.is_empty() {
            return Ok(None);
        }

        // Create level view
        let atr = feature_view.atr.unwrap_or(0.01); // Default to small value if no ATR
        let level_view = LevelView::new(
            sr_levels,
            feature_view.close,
            atr,
            feature_view.timestamp,
            feature_view.symbol.clone(),
            feature_view.timeframe.clone(),
            self.config.max_levels_per_side,
        )?;

        let mut predictions = Vec::new();

        // Process each near level
        for level in level_view.get_near_levels() {
            // Calculate trend strength
            let trend_strength = feature_view.trend_short.unwrap_or(0.0);

            // Calculate momentum
            let momentum = feature_view.macd_histogram.unwrap_or(0.0);

            // Calculate exhaustion (using RSI or Williams %R)
            let exhaustion = {
                let rsi = feature_view.rsi.unwrap_or(50.0);
                let williams = feature_view.williams_r.unwrap_or(-50.0);

                // Normalize both to 0-1 scale where higher means more exhausted
                let rsi_exhaustion = if rsi > 70.0 { (rsi - 70.0) / 30.0 } else if rsi < 30.0 { (30.0 - rsi) / 30.0 } else { 0.0 };
                let williams_exhaustion = if williams < -80.0 { (williams.abs() - 80.0) / 20.0 } else if williams > -20.0 { (williams + 20.0) / 20.0 } else { 0.0 };

                rsi_exhaustion.max(williams_exhaustion).min(1.0)
            };

            // Calculate volume factor
            let volume_factor = if let (Some(volume_sma), volume) = (feature_view.volume_sma, feature_view.volume) {
                if volume_sma > 0.0 {
                    (volume / volume_sma).min(2.0) // Cap at 2x average
                } else {
                    1.0
                }
            } else {
                1.0
            };

            // Calculate bounce/break probabilities
            let (bounce_prob, break_prob) = level_view.calculate_level_probability(
                level,
                trend_strength,
                momentum,
                exhaustion,
                volume_factor,
            );

            // Create predictions for both bounce and break
            if bounce_prob >= self.config.min_store_score {
                let predictor_meta = persistence::PredictorMeta {
                    predictor_id: 0,
                    name: "level_bounce_hard".to_string(),
                    version: "1.0".to_string(),
                    aspect: PredictionAspect::LevelBounce,
                    calc_source: CalcSource::Hard,
                    framework: "hardcode".to_string(),
                    artifact_path: None,
                    feature_schema_id: "sr_level_features".to_string(), // Fixed schema for level features
                };

                let predictor_id = persistence::register_predictor_if_missing(&self.db_pool, &predictor_meta).await?.0;

                let bounce_prediction = PredictionRow {
                    time: feature_view.timestamp,
                    time_ms: feature_view.timestamp.timestamp_millis(),
                    symbol_id: 0,
                    symbol: feature_view.symbol.clone(),
                    tf_minutes: self.parse_timeframe_minutes(&feature_view.timeframe)?,
                    horizon_bars: self.config.horizon_bars as i32,
                    aspect: PredictionAspect::LevelBounce,
                    calc_source: CalcSource::Hard,
                    predictor_id,
                    score_norm: bounce_prob as f32,
                    value: bounce_prob,
                    value_low: None,
                    value_high: None,
                    side: Some(if level.level_kind == crate::predictions::feature_view::SrLevelKind::Support { 1 } else { -1 }),
                    level_hash: Some(level.level_hash.clone()),
                    level_kind: Some(level.level_kind.as_i16()),
                    level_price: Some(level.level_price),
                    level_strength: Some(level.level_strength),
                    level_distance_atr: Some(level.distance_atr),
                    candle_is_final: true,
                    event_time_ms: None,
                    details_json: Some(serde_json::json!({
                        "method": "hardcode_level_bounce",
                        "level_info": {
                            "price": level.level_price,
                            "kind": level.level_kind.as_i16(),
                            "strength": level.level_strength,
                            "distance_atr": level.distance_atr,
                        },
                        "factors": {
                            "trend_strength": trend_strength,
                            "momentum": momentum,
                            "exhaustion": exhaustion,
                            "volume_factor": volume_factor,
                        }
                    })),
                    prediction_key: format!("bounce_hard_{}_{}_{}", feature_view.symbol, level.level_hash, feature_view.timestamp.timestamp()),
                };

                predictions.push(bounce_prediction);
            }

            if break_prob >= self.config.min_store_score {
                let predictor_meta = persistence::PredictorMeta {
                    predictor_id: 0,
                    name: "level_breakout_hard".to_string(),
                    version: "1.0".to_string(),
                    aspect: PredictionAspect::LevelBreakout,
                    calc_source: CalcSource::Hard,
                    framework: "hardcode".to_string(),
                    artifact_path: None,
                    feature_schema_id: "sr_level_features".to_string(),
                };

                let predictor_id = persistence::register_predictor_if_missing(&self.db_pool, &predictor_meta).await?.0;

                let break_prediction = PredictionRow {
                    time: feature_view.timestamp,
                    time_ms: feature_view.timestamp.timestamp_millis(),
                    symbol_id: 0,
                    symbol: feature_view.symbol.clone(),
                    tf_minutes: self.parse_timeframe_minutes(&feature_view.timeframe)?,
                    horizon_bars: self.config.horizon_bars as i32,
                    aspect: PredictionAspect::LevelBreakout,
                    calc_source: CalcSource::Hard,
                    predictor_id,
                    score_norm: break_prob as f32,
                    value: break_prob,
                    value_low: None,
                    value_high: None,
                    side: Some(if level.level_kind == crate::predictions::feature_view::SrLevelKind::Support { -1 } else { 1 }), // Opposite direction for break
                    level_hash: Some(level.level_hash.clone()),
                    level_kind: Some(level.level_kind.as_i16()),
                    level_price: Some(level.level_price),
                    level_strength: Some(level.level_strength),
                    level_distance_atr: Some(level.distance_atr),
                    candle_is_final: true,
                    event_time_ms: None,
                    details_json: Some(serde_json::json!({
                        "method": "hardcode_level_breakout",
                        "level_info": {
                            "price": level.level_price,
                            "kind": level.level_kind.as_i16(),
                            "strength": level.level_strength,
                            "distance_atr": level.distance_atr,
                        },
                        "factors": {
                            "trend_strength": trend_strength,
                            "momentum": momentum,
                            "exhaustion": exhaustion,
                            "volume_factor": volume_factor,
                        }
                    })),
                    prediction_key: format!("breakout_hard_{}_{}_{}", feature_view.symbol, level.level_hash, feature_view.timestamp.timestamp()),
                };

                predictions.push(break_prediction);
            }
        }

        if predictions.is_empty() {
            Ok(None)
        } else {
            Ok(Some(predictions))
        }
    }

    async fn run_ml_predictors(
        &self,
        feature_view: &FeatureView,
        feature_vector: &FeatureVector,
    ) -> Result<Option<Vec<PredictionRow>>> {
        if !self.config.enabled {
            return Ok(None);
        }

        let mut predictions = Vec::new();

        // Run ML price predictor
        if let Some(ml_price_pred) = self.run_ml_price_predictor(feature_view, feature_vector).await? {
            predictions.push(ml_price_pred);
        }

        // Run ML level predictor
        if let Some(ml_level_preds) = self.run_ml_level_predictor(feature_view, feature_vector).await? {
            predictions.extend(ml_level_preds);
        }

        if predictions.is_empty() {
            Ok(None)
        } else {
            Ok(Some(predictions))
        }
    }

    async fn run_ml_price_predictor(
        &self,
        feature_view: &FeatureView,
        feature_vector: &FeatureVector,
    ) -> Result<Option<PredictionRow>> {
        // Import the ML predictor
        use crate::predictions::future_price::ml_predictor::FuturePriceMl;
        use crate::predictions::future_price::ml_predictor::Backend;

        // Create a dummy backend for now
        let backend = Backend::Cpu; // Placeholder
        let model_path = "models/price10.onnx"; // Would come from config
        let predictor = FuturePriceMl::new(model_path, backend)?;

        // Run prediction
        if let Some((predicted_prices, score)) = predictor.predict(
            &feature_view.symbol,
            &feature_view.timeframe,
            feature_vector,
        ).await? {
            if score >= self.config.min_store_score {
                // Calculate target price for horizon_bars (10)
                let target_price = if predicted_prices.len() > self.config.horizon_bars {
                    predicted_prices[self.config.horizon_bars - 1]
                } else {
                    predicted_prices.last().copied().unwrap_or(feature_view.close)
                };

                // Register predictor if missing
                let predictor_meta = persistence::PredictorMeta {
                    predictor_id: 0,
                    name: "price10_ml".to_string(),
                    version: "1.0".to_string(),
                    aspect: PredictionAspect::PriceTarget,
                    calc_source: CalcSource::Ml,
                    framework: "onnx".to_string(),
                    artifact_path: Some(model_path.to_string()),
                    feature_schema_id: feature_vector.schema_id.clone(),
                };

                let predictor_id = persistence::register_predictor_if_missing(&self.db_pool, &predictor_meta).await?.0;

                // Create prediction row
                let prediction_row = PredictionRow {
                    time: feature_view.timestamp,
                    time_ms: feature_view.timestamp.timestamp_millis(),
                    symbol_id: 0,
                    symbol: feature_view.symbol.clone(),
                    tf_minutes: self.parse_timeframe_minutes(&feature_view.timeframe)?,
                    horizon_bars: self.config.horizon_bars as i32,
                    aspect: PredictionAspect::PriceTarget,
                    calc_source: CalcSource::Ml,
                    predictor_id,
                    score_norm: score as f32,
                    value: target_price,
                    value_low: None, // Could be calculated from quantile outputs
                    value_high: None, // Could be calculated from quantile outputs
                    side: self.determine_side(feature_view.close, target_price),
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
                        "feature_vector_length": feature_vector.values.len(),
                        "model_used": model_path
                    })),
                    prediction_key: format!("price10_ml_{}_{}", feature_view.symbol, feature_view.timestamp.timestamp()),
                };

                return Ok(Some(prediction_row));
            }
        }

        Ok(None)
    }

    async fn run_ml_level_predictor(
        &self,
        feature_view: &FeatureView,
        feature_vector: &FeatureVector,
    ) -> Result<Option<Vec<PredictionRow>>> {
        // Parse SR levels from JSON
        let sr_levels = feature_view.get_sr_levels()?;
        if sr_levels.is_empty() {
            return Ok(None);
        }

        // Create level view
        let atr = feature_view.atr.unwrap_or(0.01);
        let level_view = LevelView::new(
            sr_levels,
            feature_view.close,
            atr,
            feature_view.timestamp,
            feature_view.symbol.clone(),
            feature_view.timeframe.clone(),
            self.config.max_levels_per_side,
        )?;

        let mut predictions = Vec::new();

        // Import the ML level predictor
        use crate::predictions::level_predictor::ml_predictor::LevelPredictorMl;
        use crate::predictions::level_predictor::ml_predictor::Backend;

        // Create a dummy backend for now
        let backend = Backend::Cpu; // Placeholder
        let model_path = "models/levels.onnx"; // Would come from config
        let predictor = LevelPredictorMl::new(model_path, backend)?;

        // Process each near level
        for level in level_view.get_near_levels() {
            // Run prediction for this level
            if let Some((level_price, prob_bounce, prob_break, score)) = predictor.predict_for_level(
                &feature_view.symbol,
                &feature_view.timeframe,
                feature_vector,
                level,
            ).await? {
                // Determine which prediction to use based on score
                if prob_bounce >= self.config.min_store_score {
                    let predictor_meta = persistence::PredictorMeta {
                        predictor_id: 0,
                        name: "level_bounce_ml".to_string(),
                        version: "1.0".to_string(),
                        aspect: PredictionAspect::LevelBounce,
                        calc_source: CalcSource::Ml,
                        framework: "onnx".to_string(),
                        artifact_path: Some(model_path.to_string()),
                        feature_schema_id: feature_vector.schema_id.clone(),
                    };

                    let predictor_id = persistence::register_predictor_if_missing(&self.db_pool, &predictor_meta).await?.0;

                    let bounce_prediction = PredictionRow {
                        time: feature_view.timestamp,
                        time_ms: feature_view.timestamp.timestamp_millis(),
                        symbol_id: 0,
                        symbol: feature_view.symbol.clone(),
                        tf_minutes: self.parse_timeframe_minutes(&feature_view.timeframe)?,
                        horizon_bars: self.config.horizon_bars as i32,
                        aspect: PredictionAspect::LevelBounce,
                        calc_source: CalcSource::Ml,
                        predictor_id,
                        score_norm: prob_bounce as f32,
                        value: prob_bounce,
                        value_low: None,
                        value_high: None,
                        side: Some(if level.level_kind == crate::predictions::feature_view::SrLevelKind::Support { 1 } else { -1 }),
                        level_hash: Some(level.level_hash.clone()),
                        level_kind: Some(level.level_kind.as_i16()),
                        level_price: Some(level_price),
                        level_strength: Some(level.level_strength),
                        level_distance_atr: Some(level.distance_atr),
                        candle_is_final: true,
                        event_time_ms: None,
                        details_json: Some(serde_json::json!({
                            "method": "ml_level_bounce",
                            "level_info": {
                                "price": level_price,
                                "kind": level.level_kind.as_i16(),
                                "strength": level.level_strength,
                                "distance_atr": level.distance_atr,
                            },
                            "model_used": model_path
                        })),
                        prediction_key: format!("bounce_ml_{}_{}_{}", feature_view.symbol, level.level_hash, feature_view.timestamp.timestamp()),
                    };

                    predictions.push(bounce_prediction);
                }

                if prob_break >= self.config.min_store_score {
                    let predictor_meta = persistence::PredictorMeta {
                        predictor_id: 0,
                        name: "level_breakout_ml".to_string(),
                        version: "1.0".to_string(),
                        aspect: PredictionAspect::LevelBreakout,
                        calc_source: CalcSource::Ml,
                        framework: "onnx".to_string(),
                        artifact_path: Some(model_path.to_string()),
                        feature_schema_id: feature_vector.schema_id.clone(),
                    };

                    let predictor_id = persistence::register_predictor_if_missing(&self.db_pool, &predictor_meta).await?.0;

                    let break_prediction = PredictionRow {
                        time: feature_view.timestamp,
                        time_ms: feature_view.timestamp.timestamp_millis(),
                        symbol_id: 0,
                        symbol: feature_view.symbol.clone(),
                        tf_minutes: self.parse_timeframe_minutes(&feature_view.timeframe)?,
                        horizon_bars: self.config.horizon_bars as i32,
                        aspect: PredictionAspect::LevelBreakout,
                        calc_source: CalcSource::Ml,
                        predictor_id,
                        score_norm: prob_break as f32,
                        value: prob_break,
                        value_low: None,
                        value_high: None,
                        side: Some(if level.level_kind == crate::predictions::feature_view::SrLevelKind::Support { -1 } else { 1 }),
                        level_hash: Some(level.level_hash.clone()),
                        level_kind: Some(level.level_kind.as_i16()),
                        level_price: Some(level_price),
                        level_strength: Some(level.level_strength),
                        level_distance_atr: Some(level.distance_atr),
                        candle_is_final: true,
                        event_time_ms: None,
                        details_json: Some(serde_json::json!({
                            "method": "ml_level_breakout",
                            "level_info": {
                                "price": level_price,
                                "kind": level.level_kind.as_i16(),
                                "strength": level.level_strength,
                                "distance_atr": level.distance_atr,
                            },
                            "model_used": model_path
                        })),
                        prediction_key: format!("breakout_ml_{}_{}_{}", feature_view.symbol, level.level_hash, feature_view.timestamp.timestamp()),
                    };

                    predictions.push(break_prediction);
                }
            }
        }

        if predictions.is_empty() {
            Ok(None)
        } else {
            Ok(Some(predictions))
        }
    }

    async fn apply_consensus_logic(
        &self,
        mut predictions: Vec<PredictionRow>,
        feature_view: &FeatureView,
    ) -> Result<Vec<PredictionRow>> {
        // If we have both hardcode and ML predictions for the same aspect and level,
        // apply consensus logic
        if !self.config.prefer_ml || predictions.len() <= 1 {
            return Ok(predictions);
        }

        // Group predictions by aspect and level (if applicable)
        use std::collections::HashMap;
        let mut grouped: HashMap<String, Vec<PredictionRow>> = HashMap::new();

        for pred in predictions {
            let key = match &pred.level_hash {
                Some(hash) => format!("{}_{}_{}", pred.aspect.as_int(), hash, pred.symbol),
                None => format!("{}_{}", pred.aspect.as_int(), pred.symbol),
            };

            grouped.entry(key).or_insert_with(Vec::new).push(pred);
        }

        let mut final_predictions = Vec::new();

        for (_key, mut group) in grouped {
            if group.len() == 1 {
                // Single prediction, just add it
                final_predictions.push(group.remove(0));
            } else {
                // Multiple predictions for the same aspect/level, apply consensus
                let consensus_pred = self.apply_gate_and_fuse_logic(&group, feature_view).await?;
                if let Some(consensus_pred) = consensus_pred {
                    final_predictions.push(consensus_pred);
                }
            }
        }

        Ok(final_predictions)
    }

    async fn apply_gate_and_fuse_logic(
        &self,
        predictions: &[PredictionRow],
        feature_view: &FeatureView,
    ) -> Result<Option<PredictionRow>> {
        // Separate hardcode and ML predictions
        let mut hard_preds: Vec<&PredictionRow> = predictions.iter()
            .filter(|p| p.calc_source == CalcSource::Hard)
            .collect();

        let ml_preds: Vec<&PredictionRow> = predictions.iter()
            .filter(|p| p.calc_source == CalcSource::Ml)
            .collect();

        if hard_preds.is_empty() && ml_preds.is_empty() {
            return Ok(None);
        }

        // If only one type exists, return the best of that type
        if hard_preds.is_empty() {
            if let Some(best_ml) = ml_preds.iter().max_by(|a, b| a.score_norm.partial_cmp(&b.score_norm).unwrap()) {
                return Ok(Some((*best_ml).clone()));
            }
        } else if ml_preds.is_empty() {
            if let Some(best_hard) = hard_preds.iter().max_by(|a, b| a.score_norm.partial_cmp(&b.score_norm).unwrap()) {
                return Ok(Some((*best_hard).clone()));
            }
        }

        // Both hardcode and ML exist, apply gate and fuse logic
        if let (Some(hard_pred), Some(ml_pred)) = (hard_preds.pop(), ml_preds.first()) {
            // Gate logic: hardcode determines if there's a valid setup
            let gate = ((hard_pred.score_norm as f64 - 0.5) / 0.5).clamp(0.0, 1.0);

            // Fuse logic: combine scores with weights
            let s_ml = ml_pred.score_norm as f64;
            let s_hc = hard_pred.score_norm as f64;

            // Simple fusion formula: gate * (1 - (1 - s_ml)^w_ml * (1 - s_hc)^w_hc)
            // Using equal weights for now (w_ml = w_hc = 1)
            let s_fused = gate * (1.0 - (1.0 - s_ml) * (1.0 - s_hc));

            // Create fused prediction
            let mut fused_pred = if s_ml >= s_hc { ml_pred.clone() } else { hard_pred.clone() };
            fused_pred.score_norm = s_fused as f32;

            // Update details to reflect consensus
            if let Some(ref mut details) = fused_pred.details_json {
                if let Value::Object(ref mut obj) = details {
                    obj.insert("consensus_applied".to_string(), Value::Bool(true));
                    obj.insert("gate_value".to_string(), Value::Number(serde_json::Number::from_f64(gate).unwrap()));
                    obj.insert("original_ml_score".to_string(), Value::Number(serde_json::Number::from_f64(s_ml).unwrap()));
                    obj.insert("original_hard_score".to_string(), Value::Number(serde_json::Number::from_f64(s_hc).unwrap()));
                    obj.insert("fused_score".to_string(), Value::Number(serde_json::Number::from_f64(s_fused).unwrap()));
                }
            }

            Ok(Some(fused_pred))
        } else {
            Ok(None)
        }
    }

    async fn publish_predictions(&self, snapshot: &FeatureSnapshot, feature_view: &FeatureView) -> Result<()> {
        // Publish to predictions topic
        let payload = serde_json::json!({
            "symbol": &feature_view.symbol,
            "timeframe": &feature_view.timeframe,
            "timestamp": feature_view.timestamp,
            "predictions_reference": format!("{}_{}", feature_view.symbol, feature_view.timestamp.timestamp()),
        });

        self.redpanda_client
            .produce(&self.redpanda_client.config.topic_predictions, &payload)
            .await?;

        Ok(())
    }

    fn determine_side(&self, current_price: f64, target_price: f64) -> Option<i16> {
        if target_price > current_price * 1.001 { // Small buffer for floating point comparison
            Some(1) // Long/up
        } else if target_price < current_price * 0.999 {
            Some(-1) // Short/down
        } else {
            Some(0) // Neutral
        }
    }

    fn calculate_confidence_factors(&self, feature_view: &FeatureView) -> Value {
        serde_json::json!({
            "trend_strength": feature_view.trend_short.unwrap_or(0.0),
            "momentum_strength": feature_view.macd_histogram.unwrap_or(0.0),
            "volatility_regime": feature_view.atr.map(|atr| atr / feature_view.close),
            "volume_confirmation": feature_view.volume_spike.unwrap_or(0.0),
            "oscillator_alignment": self.calculate_oscillator_alignment(feature_view)
        })
    }

    fn calculate_oscillator_alignment(&self, feature_view: &FeatureView) -> f64 {
        // Calculate how aligned the oscillators are with the expected direction
        let mut alignment_score = 0.0;
        let mut count = 0;

        if let Some(rsi) = feature_view.rsi {
            // If expecting upward movement and RSI is in reasonable range
            if rsi > 30.0 && rsi < 70.0 {
                alignment_score += 1.0;
            } else {
                alignment_score -= 0.5; // Penalty for extreme values
            }
            count += 1;
        }

        if let Some(stoch_k) = feature_view.stoch_k {
            if let Some(stoch_d) = feature_view.stoch_d {
                if (stoch_k > 20.0 && stoch_k < 80.0) && (stoch_d > 20.0 && stoch_d < 80.0) {
                    alignment_score += 1.0;
                } else {
                    alignment_score -= 0.5;
                }
                count += 1;
            }
        }

        if let Some(williams_r) = feature_view.williams_r {
            if williams_r > -80.0 && williams_r < -20.0 {
                alignment_score += 1.0;
            } else {
                alignment_score -= 0.5;
            }
            count += 1;
        }

        if count > 0 {
            alignment_score / count as f64
        } else {
            0.0
        }
    }

    fn parse_timeframe_minutes(&self, timeframe: &str) -> Result<i32> {
        // Parse timeframe string like "1m", "5m", "1h", etc.
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

#[derive(Debug, Clone)]
pub struct FeatureSnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub symbol: String,
    pub timeframe: String,
    pub indicators_data: serde_json::Value,
    pub raw_signals_data: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_timeframe_minutes() {
        let pipeline = PredictionsPipeline {
            config: PredictionsConfig {
                enabled: true,
                horizon_bars: 10,
                min_store_score: 0.80,
                min_final_score: 0.90,
                prefer_ml: true,
                max_levels_per_side: 2,
            },
            db_pool: todo!(), // Mock
            redpanda_client: todo!(), // Mock
            shutdown_rx: tokio::sync::broadcast::channel(1).1,
        };

        assert_eq!(pipeline.parse_timeframe_minutes("1m").unwrap(), 1);
        assert_eq!(pipeline.parse_timeframe_minutes("5m").unwrap(), 5);
        assert_eq!(pipeline.parse_timeframe_minutes("1h").unwrap(), 60);
        assert_eq!(pipeline.parse_timeframe_minutes("4h").unwrap(), 240);
        assert_eq!(pipeline.parse_timeframe_minutes("1d").unwrap(), 1440);
    }
}
