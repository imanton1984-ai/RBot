// compute/scoring/multi_strategy_processor.rs
//
// MultiStrategyTradeSignalStage — генерирует сигналы для ВСЕХ 6 стратегий одновременно
//
// Для каждого input создаёт 6 сигналов:
// 1. indicator_ml_strategy
// 2. indicator_heuristic_strategy
// 3. indicator_consensus_strategy
// 4. level_ml_strategy
// 5. level_heuristic_strategy
// 6. level_consensus_strategy

use anyhow::Result;
use serde_json::{json, Value};
use sqlx::PgPool;
use tokio::sync::mpsc;
use common::Symbol;

use crate::predictors::types::{PredictionAspect, CalcSource};
pub use crate::predictors::pipeline::TradeSignalInput;

// Import all 6 strategies
use indicator_ml_strategy::{IndicatorMlScorer, TradeSignalCalculator as IndicatorMlCalculator};
use indicator_heuristic_strategy::{IndicatorHeuristicScorer, TradeSignalCalculator as IndicatorHeuristicCalculator};
use indicator_consensus_strategy::{IndicatorConsensusScorer, TradeSignalCalculator as IndicatorConsensusCalculator};
use level_ml_strategy::{LevelMlScorer, TradeSignalCalculator as LevelMlCalculator};
use level_heuristic_strategy::{LevelHeuristicScorer, TradeSignalCalculator as LevelHeuristicCalculator};
use level_consensus_strategy::{LevelConsensusScorer, TradeSignalCalculator as LevelConsensusCalculator};

use super::market_params_calculator::MarketParamsCalculator;

/// Multi-strategy pipeline stage that produces trade signals for ALL 6 strategies
pub struct MultiStrategyTradeSignalStage {
    db_pool: PgPool,
    market_params_calc: MarketParamsCalculator,
    
    // Scorers for each strategy
    indicator_ml_scorer: IndicatorMlScorer,
    indicator_heuristic_scorer: IndicatorHeuristicScorer,
    indicator_consensus_scorer: IndicatorConsensusScorer,
    level_ml_scorer: LevelMlScorer,
    level_heuristic_scorer: LevelHeuristicScorer,
    level_consensus_scorer: LevelConsensusScorer,
    
    // Calculators for each strategy
    indicator_ml_calc: IndicatorMlCalculator,
    indicator_heuristic_calc: IndicatorHeuristicCalculator,
    indicator_consensus_calc: IndicatorConsensusCalculator,
    level_ml_calc: LevelMlCalculator,
    level_heuristic_calc: LevelHeuristicCalculator,
    level_consensus_calc: LevelConsensusCalculator,
    
    bulk_sender: mpsc::Sender<database_lib::PersistRecord>,
    prediction_rx: mpsc::UnboundedReceiver<TradeSignalInput>,
}

impl MultiStrategyTradeSignalStage {
    pub fn new(
        db_pool: PgPool,
        market_params_calc: MarketParamsCalculator,
        bulk_sender: mpsc::Sender<database_lib::PersistRecord>,
        prediction_rx: mpsc::UnboundedReceiver<TradeSignalInput>,
        min_score: f64,
    ) -> Self {
        tracing::info!("Initializing MultiStrategyTradeSignalStage with min_score: {}", min_score);

        Self {
            db_pool,
            market_params_calc,
            
            // Initialize all 6 scorers
            indicator_ml_scorer: IndicatorMlScorer::new(min_score),
            indicator_heuristic_scorer: IndicatorHeuristicScorer::new(min_score),
            indicator_consensus_scorer: IndicatorConsensusScorer::new(min_score),
            level_ml_scorer: LevelMlScorer::new(min_score),
            level_heuristic_scorer: LevelHeuristicScorer::new(min_score),
            level_consensus_scorer: LevelConsensusScorer::new(min_score),
            
            // Initialize all 6 calculators
            indicator_ml_calc: IndicatorMlCalculator::new(min_score),
            indicator_heuristic_calc: IndicatorHeuristicCalculator::new(min_score),
            indicator_consensus_calc: IndicatorConsensusCalculator::new(min_score),
            level_ml_calc: LevelMlCalculator::new(min_score),
            level_heuristic_calc: LevelHeuristicCalculator::new(min_score),
            level_consensus_calc: LevelConsensusCalculator::new(min_score),
            
            bulk_sender,
            prediction_rx,
        }
    }

    /// Main loop: receive prediction inputs, score with ALL 6 strategies, and produce trade signals
    pub async fn run(mut self) -> Result<()> {
        tracing::info!(target: "multi_strategy_stage", "MultiStrategyTradeSignalStage started");
        let mut signals_produced: u64 = 0;
        let mut inputs_processed: u64 = 0;

        while let Some(input) = self.prediction_rx.recv().await {
            inputs_processed += 1;

            if let Err(e) = self.process_input(input).await {
                tracing::error!(
                    target: "multi_strategy_stage",
                    "Error processing multi-strategy signal input: {}", e
                );
            }

            if inputs_processed % 1000 == 0 {
                tracing::info!(
                    target: "multi_strategy_stage",
                    "MultiStrategyTradeSignalStage: processed {} inputs, produced {} signals",
                    inputs_processed, signals_produced
                );
            }
        }

        tracing::info!(
            target: "multi_strategy_stage",
            "MultiStrategyTradeSignalStage shutting down: processed {} inputs, produced {} signals",
            inputs_processed, signals_produced
        );

        Ok(())
    }

    async fn process_input(&self, input: TradeSignalInput) -> Result<()> {
        if input.predictions.is_empty() {
            return Ok(());
        }

        // 1. Compute BTC market params
        let market_params = self
            .market_params_calc
            .get_or_compute(&self.db_pool, input.timestamp)
            .await?;

        let symbol = Symbol::from(input.symbol.clone());
        let mut signals_count = 0;

        // ═══════════════════════════════════════════════════════════════
        // Strategy 1: Indicator ML
        // ═══════════════════════════════════════════════════════════════
        if let Ok(Some(signal)) = self.indicator_ml_calc.build_trade_signal(
            &self.db_pool,
            &self.indicator_ml_scorer,
            input.timestamp,
            input.time_ms,
            input.symbol_id,
            &symbol,
            input.tf_minutes,
            input.close_price,
            &input.raw_signals_summary,
            &input.predictions,
        ).await {
            self.persist_signal(signal.into(), &input).await?;
            signals_count += 1;
        }

        // ═══════════════════════════════════════════════════════════════
        // Strategy 2: Indicator Heuristic
        // ═══════════════════════════════════════════════════════════════
        if let Ok(Some(signal)) = self.indicator_heuristic_calc.build_trade_signal(
            &self.db_pool,
            &self.indicator_heuristic_scorer,
            input.timestamp,
            input.time_ms,
            input.symbol_id,
            &symbol,
            input.tf_minutes,
            input.close_price,
            &input.raw_signals_summary,
            &input.predictions,
        ).await {
            self.persist_signal(signal.into(), &input).await?;
            signals_count += 1;
        }

        // ═══════════════════════════════════════════════════════════════
        // Strategy 3: Indicator Consensus
        // ═══════════════════════════════════════════════════════════════
        if let Ok(Some(signal)) = self.indicator_consensus_calc.build_trade_signal(
            &self.db_pool,
            &self.indicator_consensus_scorer,
            input.timestamp,
            input.time_ms,
            input.symbol_id,
            &symbol,
            input.tf_minutes,
            input.close_price,
            &input.raw_signals_summary,
            &input.predictions,
        ).await {
            self.persist_signal(signal.into(), &input).await?;
            signals_count += 1;
        }

        // ═══════════════════════════════════════════════════════════════
        // Strategy 4: Level ML
        // ═══════════════════════════════════════════════════════════════
        if let Ok(Some(signal)) = self.level_ml_calc.build_trade_signal(
            &self.db_pool,
            &self.level_ml_scorer,
            input.timestamp,
            input.time_ms,
            input.symbol_id,
            &symbol,
            input.tf_minutes,
            input.close_price,
            &input.raw_signals_summary,
            &input.predictions,
        ).await {
            self.persist_signal(signal.into(), &input).await?;
            signals_count += 1;
        }

        // ═══════════════════════════════════════════════════════════════
        // Strategy 5: Level Heuristic
        // ═══════════════════════════════════════════════════════════════
        if let Ok(Some(signal)) = self.level_heuristic_calc.build_trade_signal(
            &self.db_pool,
            &self.level_heuristic_scorer,
            input.timestamp,
            input.time_ms,
            input.symbol_id,
            &symbol,
            input.tf_minutes,
            input.close_price,
            &input.raw_signals_summary,
            &input.predictions,
        ).await {
            self.persist_signal(signal.into(), &input).await?;
            signals_count += 1;
        }

        // ═══════════════════════════════════════════════════════════════
        // Strategy 6: Level Consensus (original)
        // ═══════════════════════════════════════════════════════════════
        if let Ok(Some(signal)) = self.level_consensus_calc.build_trade_signal(
            &self.db_pool,
            &self.level_consensus_scorer,
            input.timestamp,
            input.time_ms,
            input.symbol_id,
            &symbol,
            input.tf_minutes,
            input.close_price,
            &input.raw_signals_summary,
            &input.predictions,
        ).await {
            self.persist_signal(signal.into(), &input).await?;
            signals_count += 1;
        }

        if signals_count > 0 {
            tracing::debug!(
                target: "multi_strategy_stage",
                "Generated {} signals for {} tf={}",
                signals_count, input.symbol, input.tf_minutes
            );
        }

        Ok(())
    }

    async fn persist_signal(
        &self,
        signal: StrategyTradeSignal,
        input: &TradeSignalInput,
    ) -> Result<()> {
        // Extract prediction aspect scores for the DB columns
        let mut price10_target: Option<f64> = None;
        let mut price10_score: Option<f32> = None;
        let mut bounce_prob: Option<f32> = None;
        let mut bounce_score: Option<f32> = None;
        let mut breakout_prob: Option<f32> = None;
        let mut breakout_score: Option<f32> = None;
        let mut ml_score: Option<f32> = None;
        let mut heur_score: Option<f32> = None;

        for p in &input.predictions {
            match p.aspect {
                PredictionAspect::PriceTarget => {
                    match p.calc_source {
                        CalcSource::Ml => {
                            price10_target = Some(p.value);
                            price10_score = Some(p.score_norm);
                            ml_score = Some(p.score_norm);
                        }
                        CalcSource::Hard => {
                            if price10_target.is_none() {
                                price10_target = Some(p.value);
                                price10_score = Some(p.score_norm);
                            }
                            heur_score = Some(p.score_norm);
                        }
                    }
                }
                PredictionAspect::LevelBounce => {
                    bounce_prob = Some(p.value as f32);
                    bounce_score = Some(p.score_norm);
                }
                PredictionAspect::LevelBreakout => {
                    breakout_prob = Some(p.value as f32);
                    breakout_score = Some(p.score_norm);
                }
            }
        }

        let record = database_lib::PersistRecord::TradeSignal {
            symbol: Symbol::from(signal.symbol.clone()),
            timeframe: signal.tf_minutes,
            time_ms: signal.time_ms,
            side: signal.side as i16,
            final_score: signal.final_score as f32,
            ml_score,
            heur_score,
            entry_price: Some(signal.entry as f32),
            sl_price: Some(signal.stop_loss as f32),
            tp1_price: Some(signal.tp1 as f32),
            tp2_price: Some(signal.tp2 as f32),
            tp3_price: Some(signal.tp3 as f32),
            reason: Some(signal.breakdown_json),
            price10_target,
            price10_score,
            bounce_prob,
            bounce_score,
            breakout_prob,
            breakout_score,
            strategy_id: signal.strategy_id,
            strategy_name: signal.strategy_name,
        };

        if let Err(e) = self.bulk_sender.send(record).await {
            tracing::error!(
                target: "multi_strategy_stage",
                "Failed to send TradeSignal to BulkPersistor: {}", e
            );
            return Ok(());
        }

        Ok(())
    }
}

// Common trade signal structure that all strategies produce
#[derive(Debug, Clone)]
pub struct StrategyTradeSignal {
    pub time: chrono::DateTime<chrono::Utc>,
    pub time_ms: i64,
    pub symbol_id: i64,
    pub symbol: String,
    pub tf_minutes: i16,
    pub side: i8,
    pub entry: f64,
    pub stop_loss: f64,
    pub tp1: f64,
    pub tp2: f64,
    pub tp3: f64,
    pub leverage: i16,
    pub final_score: f64,
    pub breakdown_json: Value,
    pub strategy_id: i16,
    pub strategy_name: String,
}

// Implement From for each strategy's TradeSignal
impl From<indicator_ml_strategy::TradeSignal> for StrategyTradeSignal {
    fn from(s: indicator_ml_strategy::TradeSignal) -> Self {
        StrategyTradeSignal {
            time: s.time,
            time_ms: s.time_ms,
            symbol_id: s.symbol_id,
            symbol: s.symbol,
            tf_minutes: s.tf_minutes,
            side: s.side,
            entry: s.entry,
            stop_loss: s.stop_loss,
            tp1: s.tp1,
            tp2: s.tp2,
            tp3: s.tp3,
            leverage: s.leverage,
            final_score: s.final_score,
            breakdown_json: s.breakdown_json,
            strategy_id: indicator_ml_strategy::STRATEGY_ID,
            strategy_name: indicator_ml_strategy::STRATEGY_NAME.to_string(),
        }
    }
}

impl From<indicator_heuristic_strategy::TradeSignal> for StrategyTradeSignal {
    fn from(s: indicator_heuristic_strategy::TradeSignal) -> Self {
        StrategyTradeSignal {
            time: s.time,
            time_ms: s.time_ms,
            symbol_id: s.symbol_id,
            symbol: s.symbol,
            tf_minutes: s.tf_minutes,
            side: s.side,
            entry: s.entry,
            stop_loss: s.stop_loss,
            tp1: s.tp1,
            tp2: s.tp2,
            tp3: s.tp3,
            leverage: s.leverage,
            final_score: s.final_score,
            breakdown_json: s.breakdown_json,
            strategy_id: indicator_heuristic_strategy::STRATEGY_ID,
            strategy_name: indicator_heuristic_strategy::STRATEGY_NAME.to_string(),
        }
    }
}

impl From<indicator_consensus_strategy::TradeSignal> for StrategyTradeSignal {
    fn from(s: indicator_consensus_strategy::TradeSignal) -> Self {
        StrategyTradeSignal {
            time: s.time,
            time_ms: s.time_ms,
            symbol_id: s.symbol_id,
            symbol: s.symbol,
            tf_minutes: s.tf_minutes,
            side: s.side,
            entry: s.entry,
            stop_loss: s.stop_loss,
            tp1: s.tp1,
            tp2: s.tp2,
            tp3: s.tp3,
            leverage: s.leverage,
            final_score: s.final_score,
            breakdown_json: s.breakdown_json,
            strategy_id: indicator_consensus_strategy::STRATEGY_ID,
            strategy_name: indicator_consensus_strategy::STRATEGY_NAME.to_string(),
        }
    }
}

impl From<level_ml_strategy::TradeSignal> for StrategyTradeSignal {
    fn from(s: level_ml_strategy::TradeSignal) -> Self {
        StrategyTradeSignal {
            time: s.time,
            time_ms: s.time_ms,
            symbol_id: s.symbol_id,
            symbol: s.symbol,
            tf_minutes: s.tf_minutes,
            side: s.side,
            entry: s.entry,
            stop_loss: s.stop_loss,
            tp1: s.tp1,
            tp2: s.tp2,
            tp3: s.tp3,
            leverage: s.leverage,
            final_score: s.final_score,
            breakdown_json: s.breakdown_json,
            strategy_id: level_ml_strategy::STRATEGY_ID,
            strategy_name: level_ml_strategy::STRATEGY_NAME.to_string(),
        }
    }
}

impl From<level_heuristic_strategy::TradeSignal> for StrategyTradeSignal {
    fn from(s: level_heuristic_strategy::TradeSignal) -> Self {
        StrategyTradeSignal {
            time: s.time,
            time_ms: s.time_ms,
            symbol_id: s.symbol_id,
            symbol: s.symbol,
            tf_minutes: s.tf_minutes,
            side: s.side,
            entry: s.entry,
            stop_loss: s.stop_loss,
            tp1: s.tp1,
            tp2: s.tp2,
            tp3: s.tp3,
            leverage: s.leverage,
            final_score: s.final_score,
            breakdown_json: s.breakdown_json,
            strategy_id: level_heuristic_strategy::STRATEGY_ID,
            strategy_name: level_heuristic_strategy::STRATEGY_NAME.to_string(),
        }
    }
}

impl From<level_consensus_strategy::TradeSignal> for StrategyTradeSignal {
    fn from(s: level_consensus_strategy::TradeSignal) -> Self {
        StrategyTradeSignal {
            time: s.time,
            time_ms: s.time_ms,
            symbol_id: s.symbol_id,
            symbol: s.symbol,
            tf_minutes: s.tf_minutes,
            side: s.side,
            entry: s.entry,
            stop_loss: s.stop_loss,
            tp1: s.tp1,
            tp2: s.tp2,
            tp3: s.tp3,
            leverage: s.leverage,
            final_score: s.final_score,
            breakdown_json: s.breakdown_json,
            strategy_id: level_consensus_strategy::STRATEGY_ID,
            strategy_name: level_consensus_strategy::STRATEGY_NAME.to_string(),
        }
    }
}
