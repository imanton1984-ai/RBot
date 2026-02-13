use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::sync::RwLock;

use common::{Symbol, Timeframe};
use crate::{JobScheduler, WindowSpec};
use compute_indicators::FeatureStore;
use tracing;

#[derive(Debug, Clone)]
pub struct HistoryStatus {
    pub last_timestamp: i64,
    pub bars_written: usize,
    pub history_ready: bool,
}

pub struct BootstrapCoordinator {
    history_tracker: Arc<RwLock<HashMap<(Symbol, Timeframe), HistoryStatus>>>,
    _feature_store: Arc<FeatureStore>,
    job_scheduler: Arc<JobScheduler>,

    required_lookback: usize,
    warmup_bars: usize,

    // NEW: минимальный порог баров, после которого мы считаем историю "готовой"
    min_bars_ready: usize,
}

impl BootstrapCoordinator {
    pub fn new(
        _feature_store: Arc<FeatureStore>,
        job_scheduler: Arc<JobScheduler>,
        required_lookback: usize,
        warmup_bars: usize,
        min_bars_ready: usize,
    ) -> Self {
        Self {
            history_tracker: Arc::new(RwLock::new(HashMap::new())),
            _feature_store,
            job_scheduler,
            required_lookback,
            warmup_bars,
            min_bars_ready,
        }
    }

    pub fn min_bars_ready(&self) -> usize {
        self.min_bars_ready
    }

    /// Возвращает true, если история теперь ready
    pub async fn update_history_status(
        &self,
        symbol: Symbol,
        timeframe: Timeframe,
        last_timestamp: i64,
        bars_written: usize,
    ) -> bool {
        let history_ready = bars_written >= self.min_bars_ready;

        let mut tracker = self.history_tracker.write().await;
        tracker.insert(
            (symbol, timeframe),
            HistoryStatus {
                last_timestamp,
                bars_written,
                history_ready,
            },
        );

        history_ready
    }

    pub async fn is_history_ready(&self, symbol: &Symbol, timeframe: &Timeframe) -> bool {
        let tracker = self.history_tracker.read().await;
        tracker
            .get(&(symbol.clone(), *timeframe))
            .map(|s| s.history_ready)
            .unwrap_or(false)
    }

    /// Сабмит только для одного таймфрейма, чтобы НЕ было дубликатов сабмита при цикле по TF
    pub async fn start_bootstrap_compute_for_timeframe(
        &self,
        timeframe: Timeframe,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        // Сначала собираем список без удержания lock во время await
        let to_submit: Vec<(Symbol, WindowSpec)> = {
            let tracker = self.history_tracker.read().await;

            tracker
                .iter()
                .filter(|((_, tf), status)| *tf == timeframe && status.history_ready)
                .map(|((sym, _), status)| {
                    let length = std::cmp::min(
                        status.bars_written,
                        self.required_lookback + self.warmup_bars,
                    );

                    // warmup не должен быть >= length
                    let warmup = std::cmp::min(self.warmup_bars, length.saturating_sub(1));

                    (
                        sym.clone(),
                        WindowSpec {
                            length,
                            warmup,
                        },
                    )
                })
                .collect()
        };

        let mut submitted = 0usize;

        for (symbol, window_spec) in to_submit {
            // если внезапно слишком мало баров — не сабмитим мусор
            if window_spec.length <= window_spec.warmup {
                continue;
            }

            // Minimum required length for ATR and other indicators to work properly
            // ATR typically needs at least period (14) + 1 data points
            let min_required_length = 15; // 14 + 1 for ATR calculation
            if window_spec.length < min_required_length {
                tracing::warn!(
                    "Skipping symbol {} on timeframe {} due to insufficient data: {} < {}", 
                    symbol.as_str(), 
                    timeframe.as_str(), 
                    window_spec.length, 
                    min_required_length
                );
                continue; // Skip this symbol-timeframe combination
            }

            self.job_scheduler
                .submit_batch(timeframe, vec![symbol], window_spec)
                .await?;

            submitted += 1;
        }

        Ok(submitted)
    }

    /// Ждём пока будет хотя бы min_symbols ready (для запуска realtime).
    /// Таймаут 5 минут — если за это время не набралось min_symbols,
    /// всё равно стартуем realtime (Kafka consumer), чтобы не зависнуть навсегда.
    pub async fn wait_for_readiness_threshold(
        &self,
        min_symbols: usize,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut last_log = Instant::now();
        let deadline = Instant::now() + Duration::from_secs(300); // 5 min max wait

        loop {
            let (ready_count, total) = {
                let tracker = self.history_tracker.read().await;
                (tracker.values().filter(|s| s.history_ready).count(), tracker.len())
            };

            if ready_count >= min_symbols {
                println!(
                    "Readiness threshold reached: ready={}/{} (min_symbols={})",
                    ready_count, total, min_symbols
                );
                return Ok(());
            }

            if Instant::now() >= deadline {
                println!(
                    "Readiness timeout after 5 min: ready={}/{} (min_symbols={}). Starting real-time anyway.",
                    ready_count, total, min_symbols
                );
                return Ok(());
            }

            if last_log.elapsed() >= Duration::from_secs(10) {
                println!(
                    "Waiting readiness: ready={}/{} (min_symbols={}, min_bars_ready={}, timeout in {:.0}s)",
                    ready_count, total, min_symbols, self.min_bars_ready,
                    (deadline - Instant::now()).as_secs_f64()
                );
                last_log = Instant::now();
            }

            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
}