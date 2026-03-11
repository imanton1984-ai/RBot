// strategies/super_level_strategy/src/pipeline.rs
//
// Orchestration Pipeline for Super Level Strategy
//
// Объединяет:
//   1. ML inference (SuperEntryPipeline) — batch zero-copy CUDA
//   2. EWMAC calculation (EwmacCalculator) — incremental O(1) per bar
//   3. 5-phase filter (phases.rs) — inline logic
//
// Весь горячий путь:
//   - Одна загрузка данных из БД (reuse ml_entry_strategy::dataset)
//   - Один batch ML inference (zero-copy f32 features → CUDA)
//   - Один проход EWMAC (incremental EMA state)
//   - Фазовый фильтр per-bar (inline, no alloc)

use anyhow::Result;
use tracing::info;

use crate::config::SuperLevelConfig;
use crate::phases::{self, PhaseResult, Scenario, RejectPhase};

use ml_entry_strategy::config::SuperEntryConfig;
use ml_entry_strategy::dataset::CandleWithIndicators;
use ml_entry_strategy::model::SuperEntryPrediction;
use ml_entry_strategy::pipeline::SuperEntryPipeline;

use ewmac_strategy::ewmac::EwmacCalculator;
use ewmac_strategy::config::EwmacConfig;

/// Результат обработки одной свечи пайплайном
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// Индекс свечи в массиве
    pub candle_index: usize,
    /// Результат 5-фазной фильтрации
    pub phase_result: PhaseResult,
    /// ML prediction (raw)
    pub ml_prediction: Option<SuperEntryPrediction>,
    /// EWMAC forecast
    pub ewmac_forecast: f64,
}

impl PipelineOutput {
    /// Прошёл ли все 5 фаз (сделка разрешена)
    pub fn is_signal(&self) -> bool {
        self.phase_result.passed
    }
}

/// Super Level Pipeline
///
/// Полный пайплайн: DB → ML (batch) + EWMAC (incremental) → 5 phases → signals
pub struct SuperLevelPipeline {
    config: SuperLevelConfig,
    ml_pipeline: SuperEntryPipeline,
    ewmac_config: EwmacConfig,
}

impl SuperLevelPipeline {
    /// Create a new pipeline.
    ///
    /// Загружает ML модели (SuperEntry) и конфигурирует EWMAC.
    pub fn new(
        config: SuperLevelConfig,
        ml_config: SuperEntryConfig,
        use_gpu: bool,
    ) -> Result<Self> {
        let ml_pipeline = SuperEntryPipeline::new(ml_config, use_gpu)?;
        let ewmac_config = EwmacConfig::default();

        Ok(Self {
            config,
            ml_pipeline,
            ewmac_config,
        })
    }

    /// Check if ML models are loaded
    pub fn has_models(&self) -> bool {
        self.ml_pipeline.has_models()
    }

    /// Process a batch of candles for one (symbol, tf) pair.
    ///
    /// Workflow:
    ///   1. Run ML batch inference (zero-copy feature extraction → CUDA)
    ///   2. Initialize EWMAC calculator & run incrementally
    ///   3. For each bar after warmup: run 5-phase filter
    ///
    /// Returns: Vec of PipelineOutput for each processed candle
    pub fn process_candles(
        &self,
        candles: &[CandleWithIndicators],
        tf_minutes: i32,
        use_gpu: bool,
    ) -> Result<Vec<PipelineOutput>> {
        let warmup = self.config.warmup_bars;
        if candles.len() <= warmup {
            return Ok(Vec::new());
        }

        // ── STEP 1: ML batch inference (reuse SuperEntryPipeline) ──
        // Возвращает PipelineResult per candle after warmup,
        // каждый содержит prediction (p_super, p_long, direction)
        let ml_results = self.ml_pipeline.process_candles(candles, tf_minutes, use_gpu)?;

        // ── STEP 2: EWMAC incremental computation ──
        let mut ewmac_calc = EwmacCalculator::new(&self.ewmac_config);
        let mut ewmac_forecasts = Vec::with_capacity(candles.len());

        for c in candles {
            let forecast = ewmac_calc.update(c.high, c.low, c.close)
                .map(|r| r.forecast)
                .unwrap_or(0.0);
            ewmac_forecasts.push(forecast);
        }

        // ── STEP 3: 5-phase filter per bar ──
        let n_processed = ml_results.len();
        let mut outputs = Vec::with_capacity(n_processed);

        // Создаём маппинг: ml_results[i].candle_index → ml_results[i]
        // ml_results уже содержат candle_index для каждого результата
        for ml_res in &ml_results {
            let idx = ml_res.candle_index;

            // Нужно оставить зазор для entry_window_bars вперёд
            let need_after = self.config.entry_window_bars + self.config.max_hold_bars;
            if idx + need_after >= candles.len() {
                continue;
            }

            // Извлекаем ML prediction
            let (p_super, ml_direction) = match ml_res.prediction {
                Some(pred) => (pred.p_super, pred.direction),
                None => (0.0, 0),
            };

            let ewmac_forecast = ewmac_forecasts[idx];

            // Прогоняем 5 фаз
            let phase_result = phases::run_all_phases(
                candles,
                idx,
                ewmac_forecast,
                p_super,
                ml_direction,
                &self.config,
            );

            outputs.push(PipelineOutput {
                candle_index: idx,
                phase_result,
                ml_prediction: ml_res.prediction,
                ewmac_forecast,
            });
        }

        Ok(outputs)
    }

    /// Get config reference
    pub fn config(&self) -> &SuperLevelConfig {
        &self.config
    }

    /// Get ML config reference
    pub fn ml_config(&self) -> &SuperEntryConfig {
        self.ml_pipeline.config()
    }
}

/// Статистика по фазам (для отладки и отчёта)
#[derive(Debug, Default)]
pub struct PhaseStats {
    pub total_candles: usize,
    pub phase1_passed: usize,
    pub phase2_passed: usize,
    pub phase3_passed: usize,
    pub phase4_passed: usize,
    pub phase5_signals: usize,
    pub reject_no_level: usize,
    pub reject_context: usize,
    pub reject_ml: usize,
    pub reject_entry: usize,
    pub bounce_signals: usize,
    pub breakout_signals: usize,
}

impl PhaseStats {
    /// Collect stats from pipeline outputs
    pub fn from_outputs(outputs: &[PipelineOutput]) -> Self {
        let mut stats = Self::default();
        stats.total_candles = outputs.len();

        for out in outputs {
            if out.phase_result.passed {
                stats.phase5_signals += 1;
                stats.phase1_passed += 1;
                stats.phase2_passed += 1;
                stats.phase3_passed += 1;
                stats.phase4_passed += 1;

                match out.phase_result.scenario {
                    Some(Scenario::Bounce) => stats.bounce_signals += 1,
                    Some(Scenario::Breakout) => stats.breakout_signals += 1,
                    None => {}
                }
            } else if let Some(reason) = out.phase_result.reject_reason {
                match reason {
                    RejectPhase::NoLevel => stats.reject_no_level += 1,
                    RejectPhase::ContextUnclear => {
                        stats.phase1_passed += 1;
                        stats.reject_context += 1;
                    }
                    RejectPhase::MlRejected => {
                        stats.phase1_passed += 1;
                        stats.phase2_passed += 1;
                        stats.reject_ml += 1;
                    }
                    RejectPhase::EntryCancelled => {
                        stats.phase1_passed += 1;
                        stats.phase2_passed += 1;
                        stats.phase3_passed += 1;
                        stats.reject_entry += 1;
                    }
                }
            }
        }

        stats
    }

    pub fn print_report(&self, tf_name: &str) {
        info!("═══ Phase Stats [{}] ═══", tf_name);
        info!("  Total candles processed: {}", self.total_candles);
        info!("  Phase 1 (Radar):   {} passed ({:.1}%)", self.phase1_passed,
            pct(self.phase1_passed, self.total_candles));
        info!("  Phase 2 (Context): {} passed ({:.1}%)", self.phase2_passed,
            pct(self.phase2_passed, self.total_candles));
        info!("  Phase 3 (ML):      {} passed ({:.1}%)", self.phase3_passed,
            pct(self.phase3_passed, self.total_candles));
        info!("  Phase 4 (Entry):   {} passed ({:.1}%)", self.phase4_passed,
            pct(self.phase4_passed, self.total_candles));
        info!("  → Final signals:   {} ({:.2}%)", self.phase5_signals,
            pct(self.phase5_signals, self.total_candles));
        info!("    Bounce: {} | Breakout: {}", self.bounce_signals, self.breakout_signals);
        info!("  Rejects: NoLevel={} Context={} ML={} Entry={}",
            self.reject_no_level, self.reject_context, self.reject_ml, self.reject_entry);
    }
}

fn pct(part: usize, total: usize) -> f64 {
    if total > 0 { part as f64 / total as f64 * 100.0 } else { 0.0 }
}
