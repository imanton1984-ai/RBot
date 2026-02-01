use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use common::{Symbol, Timeframe};
use crate::{ComputeBackend, ComputeJob, FeatureWindow, ComputeConfig, WindowSpec};

pub struct JobScheduler {
    compute_backend: Arc<dyn ComputeBackend>,
    config: ComputeConfig,
    job_queue: Arc<RwLock<Vec<ComputeJob>>>,
    result_sender: mpsc::UnboundedSender<Arc<FeatureWindow>>,
}

impl JobScheduler {
    pub fn new(
        compute_backend: Arc<dyn ComputeBackend>,
        config: ComputeConfig,
    ) -> (Self, mpsc::UnboundedReceiver<Arc<FeatureWindow>>) {
        let (result_sender, result_receiver) = mpsc::unbounded_channel();
        
        let job_scheduler = Self {
            compute_backend,
            config,
            job_queue: Arc::new(RwLock::new(Vec::new())),
            result_sender,
        };
        
        (job_scheduler, result_receiver)
    }

    pub async fn submit_batch(
        &self,
        timeframe: Timeframe,
        symbols: Vec<Symbol>,
        _window_spec: WindowSpec,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut jobs = Vec::new();

        // Create jobs for each symbol
        for symbol in symbols {
            let job = ComputeJob {
                symbol,
                timeframe,
                window_start: 0, // This would come from DB
                window_end: 0,   // This would come from DB
                indicators: vec!["rsi".to_string(), "ema".to_string(), "macd".to_string()], // Default indicators
            };
            jobs.push(job);
        }

        // Add jobs to queue
        {
            let mut queue = self.job_queue.write().await;
            queue.extend(jobs);
        }

        // Process the queue
        self.process_queue().await?;

        Ok(())
    }

    async fn process_queue(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let jobs_to_process = {
            let mut queue = self.job_queue.write().await;
            let jobs_to_process = queue.drain(..).collect::<Vec<_>>();
            jobs_to_process
        };

        if jobs_to_process.is_empty() {
            return Ok(());
        }

        // Process jobs in batches
        for chunk in jobs_to_process.chunks(self.config.batch_size) {
            let results = self.compute_backend.compute_indicators(chunk.to_vec()).await?;
            
            for result in results {
                if let Err(_) = self.result_sender.send(result) {
                    // Handle error if receiver dropped
                }
            }
        }

        Ok(())
    }

    pub async fn process_single_job(
        &self,
        job: ComputeJob,
    ) -> Result<Arc<FeatureWindow>, Box<dyn std::error::Error + Send + Sync>> {
        let results = self.compute_backend.compute_indicators(vec![job]).await?;
        
        if let Some(result) = results.into_iter().next() {
            Ok(result)
        } else {
            Err("No results returned from compute backend".into())
        }
    }
}