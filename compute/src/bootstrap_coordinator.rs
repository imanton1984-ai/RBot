use std::sync::Arc;
use tokio::sync::RwLock;
use common::{Symbol, Timeframe};
use crate::{JobScheduler};
use compute_indicators::FeatureStore;

#[derive(Debug, Clone)]
pub struct HistoryStatus {
    pub last_timestamp: i64,
    pub bars_written: usize,
    pub history_ready: bool,
}

pub struct BootstrapCoordinator {
    history_tracker: Arc<RwLock<std::collections::HashMap<(Symbol, Timeframe), HistoryStatus>>>,
    _feature_store: Arc<FeatureStore>,
    job_scheduler: Arc<JobScheduler>,
    required_lookback: usize,
    warmup_bars: usize,
}

impl BootstrapCoordinator {
    pub fn new(
        _feature_store: Arc<FeatureStore>,
        job_scheduler: Arc<JobScheduler>,
        required_lookback: usize,
        warmup_bars: usize,
    ) -> Self {
        Self {
            history_tracker: Arc::new(RwLock::new(std::collections::HashMap::new())),
            _feature_store,
            job_scheduler,
            required_lookback,
            warmup_bars,
        }
    }

    pub async fn update_history_status(
        &self,
        symbol: Symbol,
        timeframe: Timeframe,
        last_timestamp: i64,
        bars_written: usize,
    ) {
        let mut tracker = self.history_tracker.write().await;
        
        let history_ready = bars_written >= (self.required_lookback + self.warmup_bars);
        
        tracker.insert(
            (symbol, timeframe),
            HistoryStatus {
                last_timestamp,
                bars_written,
                history_ready,
            },
        );
    }

    pub async fn is_history_ready(&self, symbol: &Symbol, timeframe: &Timeframe) -> bool {
        let tracker = self.history_tracker.read().await;
        
        if let Some(status) = tracker.get(&(symbol.clone(), *timeframe)) {
            status.history_ready
        } else {
            false
        }
    }

    pub async fn are_all_histories_ready(&self) -> bool {
        let tracker = self.history_tracker.read().await;
        
        for (_, status) in tracker.iter() {
            if !status.history_ready {
                return false;
            }
        }
        
        true
    }

    pub async fn start_bootstrap_compute(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let tracker = self.history_tracker.read().await;
        
        // Collect all ready symbols/timeframes
        let ready_pairs: Vec<(Symbol, Timeframe)> = tracker
            .iter()
            .filter(|(_, status)| status.history_ready)
            .map(|((symbol, timeframe), _)| (symbol.clone(), *timeframe))
            .collect();

        // Submit compute jobs for all ready pairs
        for (symbol, timeframe) in ready_pairs {
            let window_spec = crate::WindowSpec {
                length: self.required_lookback + self.warmup_bars,
                warmup: self.warmup_bars,
            };

            self.job_scheduler
                .submit_batch(timeframe, vec![symbol], window_spec)
                .await?;
        }

        Ok(())
    }

    pub async fn wait_for_readiness_threshold(&self, min_symbols: usize) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        loop {
            let tracker = self.history_tracker.read().await;
            let ready_count = tracker.values().filter(|status| status.history_ready).count();
            
            if ready_count >= min_symbols {
                break;
            }

            // Wait a bit before checking again
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        Ok(())
    }
}