pub mod backfill;
pub mod candle_builder;
pub mod config;
pub mod gap_fill;
pub mod health;
pub mod producer;
pub mod ws_manager;
pub mod universe;

use std::sync::Arc;

use anyhow::Result;
use connections::{BinanceRestClient, RestRateLimitCfg};
use rdkafka::producer::FutureProducer;

use crate::config::IngestConfig;
use crate::health::StageState;
use crate::producer::build_producer;

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<IngestConfig>,
    pub stage: StageState,
    pub rest: BinanceRestClient,
    pub producer: FutureProducer,
}

impl AppState {
    pub fn new(cfg: IngestConfig) -> Result<Self> {
        let cfg = Arc::new(cfg);

        let rest = {
            // используем лимитер из connections (он умеет и rate-limit и anti-ban gate)
            // http_concurrency кладём в max_in_flight
            let mut rl = RestRateLimitCfg::default();
            rl.rps = cfg.http_soft_rps.max(1);
            rl.burst = cfg.http_soft_burst.max(1);
            rl.max_in_flight = cfg.http_concurrency.max(1);

            BinanceRestClient::new(cfg.rest_base_url.clone(), rl)?
        };

        let producer = build_producer(&cfg)?;

        Ok(Self {
            cfg,
            stage: StageState::new("INIT"),
            rest,
            producer,
        })
    }
}
