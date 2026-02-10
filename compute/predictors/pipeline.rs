// compute/predictors/pipeline.rs

use anyhow::Result;
use tokio::sync::mpsc;
use serde_json::Value;
use crate::types::{PredictionRow, PredictionAspect, CalcSource};
use crate::feature_view::{FeatureView, FeatureVector};
use crate::level_view::LevelView;
use crate::config::PredictionsConfig;
use crate::persistence;
use database_lib::PgPool;
use connections_lib::RedpandaClient;

pub struct PredictionsPipeline {
    config: PredictionsConfig,
    db_pool: PgPool,
    redpanda_client: RedpandaClient,
    shutdown_rx: tokio::sync::broadcast::Receiver<bool>,
    feature_rx: Option<tokio::sync::mpsc::UnboundedReceiver<FeatureSnapshot>>,
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
            feature_rx: None,
        }
    }

    // Add a field to hold the receiver
    // We'll add this as a method to allow setting the receiver externally
    
    pub async fn run(&mut self) -> Result<()> {
        // Get the receiver - it must be set before calling run
        let mut rx = self.feature_rx.take().expect("Feature receiver must be set before calling run()");
        
        // Main processing loop - now using the external receiver
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
    
    pub fn set_input_receiver(&mut self, receiver: tokio::sync::mpsc::UnboundedReceiver<FeatureSnapshot>) {
        self.feature_rx = Some(receiver);
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
        let start = std::time::Instant::now();

        // 1. Create FeatureView
        let view = FeatureView::new(
            snapshot.timestamp,
            snapshot.symbol.clone(),
            snapshot.timeframe.clone(),
            &snapshot.indicators_data,
            snapshot.raw_signals_data.as_ref(),
        )?;

        let vector = view.to_feature_vector();

        // 2. Run predictors in parallel (join!)
        let (hc_res, ml_res) = tokio::join!(
            self.run_hardcode_predictors(&view),
            self.run_ml_predictors(&view, &vector)
        );

        // 3. Collect results
        let mut all_preds = Vec::new();
        if let Ok(Some(p)) = hc_res { all_preds.extend(p); }
        if let Ok(Some(p)) = ml_res { all_preds.extend(p); }

        // 4. Consensus
        let consensus = crate::consensus::ConsensusEngine::new();
        let final_preds = consensus.apply_gate_and_fuse(all_preds, &view).await?;

        // 5. Filter and write
        if !final_preds.is_empty() {
            persistence::upsert_predictions(&self.db_pool, final_preds).await?;

            // LOGGING TO NEW FILE
            tracing::info!(target: "compute_predictors", 
                "Symbol: {}, TF: {}, Predictions: {}, Time: {:?}", 
                snapshot.symbol, snapshot.timeframe, final_preds.len(), start.elapsed()
            );
        }

        Ok(())
    }

    async fn run_hardcode_predictors(&self, view: &FeatureView) -> Result<Option<Vec<PredictionRow>>> {
        let mut predictions = Vec::new();

        // Run FuturePriceHeuristic
        let hc_price = crate::future_predictor::heuristic_predictor::FuturePriceHeuristic::new();
        if let Some((predicted_prices, score)) = hc_price.predict(&view.symbol, &view.timeframe, view)? {
            // Create prediction row for price prediction
            let predictor_meta = persistence::PredictorMeta {
                predictor_id: 0,
                name: "price10_hard".to_string(),
                version: "1.0".to_string(),
                aspect: PredictionAspect::PriceTarget,
                calc_source: CalcSource::Hard,
                framework: "hardcode".to_string(),
                artifact_path: None,
                feature_schema_id: "feature_view".to_string(),
            };

            let predictor_id = persistence::register_predictor_if_missing(&self.db_pool, &predictor_meta).await?.0;

            let prediction_row = PredictionRow {
                time: view.timestamp,
                time_ms: view.timestamp.timestamp_millis(),
                symbol_id: 0,
                symbol: view.symbol.clone(),
                tf_minutes: self.parse_timeframe_minutes(&view.timeframe)?,
                horizon_bars: self.config.horizon_bars as i32,
                aspect: PredictionAspect::PriceTarget,
                calc_source: CalcSource::Hard,
                predictor_id,
                score_norm: score as f32,
                value: predicted_prices.last().copied().unwrap_or(view.close),
                value_low: None,
                value_high: None,
                side: self.determine_side(view.close, predicted_prices.last().copied().unwrap_or(view.close)),
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

            predictions.push(prediction_row);
        }

        // Run LevelPredictorHeuristic
        let level_predictor = crate::level_predictor::heuristic_predictor::LevelPredictorHeuristic::new();
        if let Some((level_price, prob_bounce, prob_break, score)) = level_predictor.predict(&view.symbol, &view.timeframe, view)? {
            // Create prediction row for bounce
            let bounce_predictor_meta = persistence::PredictorMeta {
                predictor_id: 0,
                name: "level_bounce_hard".to_string(),
                version: "1.0".to_string(),
                aspect: PredictionAspect::LevelBounce,
                calc_source: CalcSource::Hard,
                framework: "hardcode".to_string(),
                artifact_path: None,
                feature_schema_id: "feature_view".to_string(),
            };

            let bounce_predictor_id = persistence::register_predictor_if_missing(&self.db_pool, &bounce_predictor_meta).await?.0;

            let bounce_prediction = PredictionRow {
                time: view.timestamp,
                time_ms: view.timestamp.timestamp_millis(),
                symbol_id: 0,
                symbol: view.symbol.clone(),
                tf_minutes: self.parse_timeframe_minutes(&view.timeframe)?,
                horizon_bars: self.config.horizon_bars as i32,
                aspect: PredictionAspect::LevelBounce,
                calc_source: CalcSource::Hard,
                predictor_id: bounce_predictor_id,
                score_norm: prob_bounce as f32,
                value: prob_bounce,
                value_low: None,
                value_high: None,
                side: Some(if view.close < level_price { 1 } else { -1 }),
                level_hash: Some(format!("level_{:.2}", level_price)),
                level_kind: Some(1), // Support
                level_price: Some(level_price),
                level_strength: Some(score as f32),
                level_distance_atr: None,
                candle_is_final: true,
                event_time_ms: None,
                details_json: Some(serde_json::json!({
                    "method": "hardcode_level_bounce",
                    "level_price": level_price,
                    "prob_bounce": prob_bounce,
                    "prob_break": prob_break,
                    "score": score
                })),
                prediction_key: format!("bounce_hard_{}_{}_{}", view.symbol, level_price, view.timestamp.timestamp()),
            };

            predictions.push(bounce_prediction);

            // Create prediction row for break
            let break_predictor_meta = persistence::PredictorMeta {
                predictor_id: 0,
                name: "level_breakout_hard".to_string(),
                version: "1.0".to_string(),
                aspect: PredictionAspect::LevelBreakout,
                calc_source: CalcSource::Hard,
                framework: "hardcode".to_string(),
                artifact_path: None,
                feature_schema_id: "feature_view".to_string(),
            };

            let break_predictor_id = persistence::register_predictor_if_missing(&self.db_pool, &break_predictor_meta).await?.0;

            let break_prediction = PredictionRow {
                time: view.timestamp,
                time_ms: view.timestamp.timestamp_millis(),
                symbol_id: 0,
                symbol: view.symbol.clone(),
                tf_minutes: self.parse_timeframe_minutes(&view.timeframe)?,
                horizon_bars: self.config.horizon_bars as i32,
                aspect: PredictionAspect::LevelBreakout,
                calc_source: CalcSource::Hard,
                predictor_id: break_predictor_id,
                score_norm: prob_break as f32,
                value: prob_break,
                value_low: None,
                value_high: None,
                side: Some(if view.close < level_price { -1 } else { 1 }),
                level_hash: Some(format!("level_{:.2}", level_price)),
                level_kind: Some(1), // Support
                level_price: Some(level_price),
                level_strength: Some(score as f32),
                level_distance_atr: None,
                candle_is_final: true,
                event_time_ms: None,
                details_json: Some(serde_json::json!({
                    "method": "hardcode_level_breakout",
                    "level_price": level_price,
                    "prob_bounce": prob_bounce,
                    "prob_break": prob_break,
                    "score": score
                })),
                prediction_key: format!("breakout_hard_{}_{}_{}", view.symbol, level_price, view.timestamp.timestamp()),
            };

            predictions.push(break_prediction);
        }

        if predictions.is_empty() {
            Ok(None)
        } else {
            Ok(Some(predictions))
        }
    }

    async fn run_ml_predictors(&self, view: &FeatureView, vector: &FeatureVector) -> Result<Option<Vec<PredictionRow>>> {
        let mut predictions = Vec::new();

        // Run FuturePriceMl
        let use_cuda = self.config.use_cuda; // from config
        let ml_price = crate::future_predictor::ml_predictor::FuturePriceMl::new("models/price10.onnx", use_cuda)?;
        if let Some((predicted_prices, score)) = ml_price.predict(&view.symbol, &view.timeframe, vector).await? {
            // Create prediction row for ML price prediction
            let predictor_meta = persistence::PredictorMeta {
                predictor_id: 0,
                name: "price10_ml".to_string(),
                version: "1.0".to_string(),
                aspect: PredictionAspect::PriceTarget,
                calc_source: CalcSource::Ml,
                framework: "onnx".to_string(),
                artifact_path: Some("models/price10.onnx".to_string()),
                feature_schema_id: vector.schema_id.clone(),
            };

            let predictor_id = persistence::register_predictor_if_missing(&self.db_pool, &predictor_meta).await?.0;

            let prediction_row = PredictionRow {
                time: view.timestamp,
                time_ms: view.timestamp.timestamp_millis(),
                symbol_id: 0,
                symbol: view.symbol.clone(),
                tf_minutes: self.parse_timeframe_minutes(&view.timeframe)?,
                horizon_bars: self.config.horizon_bars as i32,
                aspect: PredictionAspect::PriceTarget,
                calc_source: CalcSource::Ml,
                predictor_id,
                score_norm: score as f32,
                value: predicted_prices.last().copied().unwrap_or(view.close),
                value_low: None,
                value_high: None,
                side: self.determine_side(view.close, predicted_prices.last().copied().unwrap_or(view.close)),
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
                    "model_used": "models/price10.onnx"
                })),
                prediction_key: format!("price10_ml_{}_{}", view.symbol, view.timestamp.timestamp()),
            };

            predictions.push(prediction_row);
        }

        // Run LevelPredictorMl
        let sr_levels = view.get_sr_levels()?;
        if !sr_levels.is_empty() {
            let atr = view.atr.unwrap_or(1.0);
            let level_view = LevelView::new(sr_levels, view.close, atr, view.timestamp, view.symbol.clone(), view.timeframe.clone(), 2)?;
            let levels = level_view.get_near_levels();

            if let Some(level) = levels.first() {
                let ml_level = crate::level_predictor::ml_predictor::LevelPredictorMl::new("models/levels.onnx", use_cuda)?;
                if let Some((prob_bounce, prob_break, score)) = ml_level.predict(&view.symbol, &view.timeframe, vector, level.level_price).await? {
                    // Create prediction row for ML bounce
                    let bounce_predictor_meta = persistence::PredictorMeta {
                        predictor_id: 0,
                        name: "level_bounce_ml".to_string(),
                        version: "1.0".to_string(),
                        aspect: PredictionAspect::LevelBounce,
                        calc_source: CalcSource::Ml,
                        framework: "onnx".to_string(),
                        artifact_path: Some("models/levels.onnx".to_string()),
                        feature_schema_id: vector.schema_id.clone(),
                    };

                    let bounce_predictor_id = persistence::register_predictor_if_missing(&self.db_pool, &bounce_predictor_meta).await?.0;

                    let bounce_prediction = PredictionRow {
                        time: view.timestamp,
                        time_ms: view.timestamp.timestamp_millis(),
                        symbol_id: 0,
                        symbol: view.symbol.clone(),
                        tf_minutes: self.parse_timeframe_minutes(&view.timeframe)?,
                        horizon_bars: self.config.horizon_bars as i32,
                        aspect: PredictionAspect::LevelBounce,
                        calc_source: CalcSource::Ml,
                        predictor_id: bounce_predictor_id,
                        score_norm: prob_bounce as f32,
                        value: prob_bounce,
                        value_low: None,
                        value_high: None,
                        side: Some(if view.close < level.level_price { 1 } else { -1 }),
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
                            "model_used": "models/levels.onnx"
                        })),
                        prediction_key: format!("bounce_ml_{}_{}_{}", view.symbol, level.level_hash, view.timestamp.timestamp()),
                    };

                    predictions.push(bounce_prediction);

                    // Create prediction row for ML break
                    let break_predictor_meta = persistence::PredictorMeta {
                        predictor_id: 0,
                        name: "level_breakout_ml".to_string(),
                        version: "1.0".to_string(),
                        aspect: PredictionAspect::LevelBreakout,
                        calc_source: CalcSource::Ml,
                        framework: "onnx".to_string(),
                        artifact_path: Some("models/levels.onnx".to_string()),
                        feature_schema_id: vector.schema_id.clone(),
                    };

                    let break_predictor_id = persistence::register_predictor_if_missing(&self.db_pool, &break_predictor_meta).await?.0;

                    let break_prediction = PredictionRow {
                        time: view.timestamp,
                        time_ms: view.timestamp.timestamp_millis(),
                        symbol_id: 0,
                        symbol: view.symbol.clone(),
                        tf_minutes: self.parse_timeframe_minutes(&view.timeframe)?,
                        horizon_bars: self.config.horizon_bars as i32,
                        aspect: PredictionAspect::LevelBreakout,
                        calc_source: CalcSource::Ml,
                        predictor_id: break_predictor_id,
                        score_norm: prob_break as f32,
                        value: prob_break,
                        value_low: None,
                        value_high: None,
                        side: Some(if view.close < level.level_price { -1 } else { 1 }),
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
                            "model_used": "models/levels.onnx"
                        })),
                        prediction_key: format!("breakout_ml_{}_{}_{}", view.symbol, level.level_hash, view.timestamp.timestamp()),
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
                use_cuda: false,
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