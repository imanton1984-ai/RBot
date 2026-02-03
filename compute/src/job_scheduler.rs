use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use common::{Symbol, Timeframe};
use crate::{ComputeBackend, ComputeJob, FeatureWindow, ComputeConfig, WindowSpec, CandleWindowFetcher};


pub struct JobSchedulerInner {
    compute_backend: Arc<dyn ComputeBackend>,
    config: ComputeConfig,
    job_queue: Arc<RwLock<Vec<ComputeJob>>>,
    result_sender: mpsc::UnboundedSender<Arc<FeatureWindow>>,
    candle_fetcher: Arc<CandleWindowFetcher>,
}

#[derive(Clone)]
pub struct JobScheduler {
    inner: Arc<JobSchedulerInner>,
}

impl JobScheduler {
    pub fn new(
        compute_backend: Arc<dyn ComputeBackend>,
        config: ComputeConfig,
        candle_fetcher: Arc<CandleWindowFetcher>,
    ) -> (Self, mpsc::UnboundedReceiver<Arc<FeatureWindow>>) {
        let (result_sender, result_receiver) = mpsc::unbounded_channel();

        let inner = JobSchedulerInner {
            compute_backend,
            config,
            job_queue: Arc::new(RwLock::new(Vec::new())),
            result_sender,
            candle_fetcher,
        };

        let job_scheduler = Self {
            inner: Arc::new(inner),
        };

        (job_scheduler, result_receiver)
    }

    pub async fn submit_batch(
        &self,
        timeframe: Timeframe,
        symbols: Vec<Symbol>,
        window_spec: WindowSpec,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut jobs = Vec::new();

        // For historical data processing, we need to get the actual time range of available data
        let all_windows = self.inner.candle_fetcher.fetch_bulk_candles(&symbols, timeframe, 0, chrono::Utc::now().timestamp_millis()).await?;

        // Create jobs for each symbol
        for symbol in symbols {
            // Use the actual window that was fetched, or create a job with the calculated time range
            let (window_start, window_end, candle_window) = if let Some(window) = all_windows.get(&symbol) {
                if !window.close.is_empty() {
                    // Use the actual time range of the data
                    let start_time = window.timestamps.first().copied().unwrap_or_else(|| chrono::Utc::now().timestamp_millis() - (window_spec.length as i64 * timeframe.to_minutes() as i64 * 60 * 1000));
                    let end_time = window.timestamps.last().copied().unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
                    (start_time, end_time, Some(window.clone()))
                } else {
                    // If no data, use default range
                    let end_time = chrono::Utc::now().timestamp_millis();
                    let start_time = end_time - (window_spec.length as i64 * timeframe.to_minutes() as i64 * 60 * 1000);
                    (start_time, end_time, None)
                }
            } else {
                // If no window found, use default range
                let end_time = chrono::Utc::now().timestamp_millis();
                let start_time = end_time - (window_spec.length as i64 * timeframe.to_minutes() as i64 * 60 * 1000);
                (start_time, end_time, None)
            };

            let job = ComputeJob {
                symbol: symbol.clone(),
                timeframe,
                window_start,
                window_end,
                indicators: vec![
                    "adx".to_string(),
                    "atr".to_string(),
                    "bb".to_string(),
                    "cci".to_string(),
                    "ema".to_string(),
                    "macd".to_string(),
                    "obv".to_string(),
                    "rsi".to_string(),
                    "sma".to_string(),
                    "stoch".to_string(),
                    "vwap".to_string(),
                    "williams".to_string(),
                    "alligator".to_string(),
                ],
                candle_window,
            };
            jobs.push(job);
        }

        // Add jobs to queue
        {
            let mut queue = self.inner.job_queue.write().await;
            queue.extend(jobs);
        }

        // Process the queue
        self.process_queue().await?;

        Ok(())
    }

    async fn process_queue(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let jobs_to_process = {
            let mut queue = self.inner.job_queue.write().await;
            let jobs_to_process = queue.drain(..).collect::<Vec<_>>();
            jobs_to_process
        };

        if jobs_to_process.is_empty() {
            return Ok(());
        }

        // Process jobs in batches
        for chunk in jobs_to_process.chunks(self.inner.config.batch_size) {
            let results = self.inner.compute_backend.compute_indicators(chunk.to_vec()).await?;

            for result in results {
                if let Err(_) = self.inner.result_sender.send(result) {
                    // Handle error if receiver dropped
                }
            }
        }

        Ok(())
    }

    pub async fn process_single_job(
        &self,
        mut job: ComputeJob,
    ) -> Result<Arc<FeatureWindow>, Box<dyn std::error::Error + Send + Sync>> {
        let window = self.inner.candle_fetcher.fetch_candle_window(&job.symbol, job.timeframe, job.window_start, job.window_end).await?;
        job.candle_window = Some(window);
        let results = self.inner.compute_backend.compute_indicators(vec![job]).await?;

        if let Some(result) = results.into_iter().next() {
            Ok(result)
        } else {
            Err("No results returned from compute backend".into())
        }
    }
}