use std::sync::atomic::{AtomicI64, Ordering};
use tokio::time::{sleep, Duration};
use chrono::Utc;

#[derive(Debug, Default)]
pub struct HttpGate {
    banned_until_ms: AtomicI64,
}

impl HttpGate {
    pub fn new() -> Self {
        Self { banned_until_ms: AtomicI64::new(0) }
    }

    pub fn set_banned_until_ms(&self, until_ms: i64) {
        let mut cur = self.banned_until_ms.load(Ordering::Relaxed);
        while until_ms > cur {
            match self.banned_until_ms.compare_exchange(
                cur, until_ms, Ordering::SeqCst, Ordering::SeqCst
            ) {
                Ok(_) => break,
                Err(v) => cur = v,
            }
        }
    }

    pub async fn wait_if_banned(&self) {
        loop {
            let until = self.banned_until_ms.load(Ordering::Relaxed);
            let now = Utc::now().timestamp_millis();
            if until <= now { return; }
            let wait_ms = (until - now + 250).max(250) as u64;
            sleep(Duration::from_millis(wait_ms)).await;
        }
    }
}
