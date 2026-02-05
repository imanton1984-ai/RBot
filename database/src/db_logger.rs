use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;
use tracing::info;

/// Database logger to track insertions to indicators and raw_signals tables
#[derive(Clone)]
pub struct DatabaseLogger {
    indicators_count: Arc<AtomicU64>,
    raw_signals_count: Arc<AtomicU64>,
}

impl DatabaseLogger {
    pub fn new() -> Self {
        Self {
            indicators_count: Arc::new(AtomicU64::new(0)),
            raw_signals_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Increment the counter for indicators inserted
    pub fn increment_indicators(&self, count: u64) {
        self.indicators_count.fetch_add(count, Ordering::Relaxed);
    }

    /// Increment the counter for raw signals inserted
    pub fn increment_raw_signals(&self, count: u64) {
        self.raw_signals_count.fetch_add(count, Ordering::Relaxed);
    }

    /// Get the current count of indicators inserted
    pub fn get_indicators_count(&self) -> u64 {
        self.indicators_count.load(Ordering::Relaxed)
    }

    /// Get the current count of raw signals inserted
    pub fn get_raw_signals_count(&self) -> u64 {
        self.raw_signals_count.load(Ordering::Relaxed)
    }

    /// Reset the counters to zero
    pub fn reset_counters(&self) {
        self.indicators_count.store(0, Ordering::Relaxed);
        self.raw_signals_count.store(0, Ordering::Relaxed);
    }

    /// Start the periodic logging task
    pub async fn start_logging(&self) {
        let logger = self.clone();
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Log every minute
            loop {
                interval.tick().await;
                
                let indicators_added = logger.get_indicators_count();
                let raw_signals_added = logger.get_raw_signals_count();
                
                info!(
                    "database.out - Indicators added: {}, Raw signals added: {}",
                    indicators_added, raw_signals_added
                );
            }
        });
    }
}

// Global logger instance
static mut DATABASE_LOGGER: Option<DatabaseLogger> = None;
static LOGGER_INIT: std::sync::Once = std::sync::Once::new();

pub fn get_database_logger() -> &'static DatabaseLogger {
    unsafe {
        LOGGER_INIT.call_once(|| {
            DATABASE_LOGGER = Some(DatabaseLogger::new());
        });
        DATABASE_LOGGER.as_ref().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_logger() {
        let logger = DatabaseLogger::new();
        
        logger.increment_indicators(5);
        logger.increment_raw_signals(3);
        
        assert_eq!(logger.get_indicators_count(), 5);
        assert_eq!(logger.get_raw_signals_count(), 3);
        
        logger.reset_counters();
        
        assert_eq!(logger.get_indicators_count(), 0);
        assert_eq!(logger.get_raw_signals_count(), 0);
    }
}