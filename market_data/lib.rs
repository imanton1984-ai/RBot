pub mod backfill;
pub mod live_ws;
pub mod universe;
pub mod normalizer;

pub struct MarketDataCollector;

impl MarketDataCollector {
    pub fn new() -> Self {
        Self {}
    }
}
