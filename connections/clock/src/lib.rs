use chrono::{DateTime, Utc};

pub trait ClockProvider {
    fn now(&self) -> DateTime<Utc>;
}

pub struct RealClock;

impl ClockProvider for RealClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}
