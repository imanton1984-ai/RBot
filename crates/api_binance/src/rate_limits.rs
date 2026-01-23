use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone)]
pub struct RateLimiter {
    sem: Arc<Semaphore>,
    max: usize,
    refill_per_sec: usize,
}

pub struct RateGuard {
    _permit: OwnedSemaphorePermit,
}

impl RateLimiter {
    /// Soft limiter: rps + burst.
    /// Ровно то, что у тебя в config/binance.toml: rps=8 burst=16 (дефолты зададим выше уровнем).
    pub fn new(rps: u32, burst: u32) -> Self {
        let max = burst.max(1) as usize;
        let refill_per_sec = rps.max(1) as usize;

        let sem = Arc::new(Semaphore::new(max));
        let sem_bg = sem.clone();

        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(1));
            loop {
                tick.tick().await;
                let avail = sem_bg.available_permits();
                if avail < max {
                    // добавляем не больше rps и не выходим за burst
                    let add = (max - avail).min(refill_per_sec);
                    sem_bg.add_permits(add);
                }
            }
        });

        Self { sem, max, refill_per_sec }
    }

    /// weight — “стоимость” запроса (можешь позже маппить на Binance weight).
    pub async fn acquire(&self, weight: u32) -> RateGuard {
        let mut w = weight.max(1) as usize;

        // если кто-то попросит weight > burst, режем на burst-чанки
        // (это простая страховка, чтобы не зависнуть навсегда)
        while w > self.max {
            let _ = self
                .sem
                .clone()
                .acquire_many_owned(self.max as u32)
                .await
                .expect("semaphore closed");
            // permit дропается сразу => “съели” burst токенов и отпустили;
            // дальше берем остаток.
            w -= self.max;
        }

        let permit = self
            .sem
            .clone()
            .acquire_many_owned(w as u32)
            .await
            .expect("semaphore closed");

        RateGuard { _permit: permit }
    }

    #[allow(dead_code)]
    pub fn debug_available(&self) -> usize {
        self.sem.available_permits()
    }

    #[allow(dead_code)]
    pub fn debug_refill_per_sec(&self) -> usize {
        self.refill_per_sec
    }
}
