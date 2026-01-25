use anyhow::{Context, Result};
use std::env;

use common::config::{load_config, AppConfig};
use connections::{KafkaManager, DatabaseManager};

#[derive(Clone)]
pub struct AppCtx {
    pub cfg: AppConfig,
    pub db: DatabaseManager,
    pub kafka: KafkaManager,
}

/// ЕДИНСТВЕННАЯ точка настройки логов для всех svc_*
/// (никаких копий в bin-файлах)
pub fn init_tracing() {
    // безопасно вызывается многократно
    let _ = if env::var("RUST_LOG").is_err() {
        // RUST_LOG не обязателен; но если есть cfg.rust_bot.log_level — пусть будет в конфиге
        Ok(())
    } else {
        Ok(())
    };

    let fmt = tracing_subscriber::fmt()
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true);

    // если у тебя есть json_logs в cfg — можно будет расширить,
    // но чтобы не усложнять: оставим plain fmt, как сейчас.
    let _ = fmt.try_init();
}

pub mod envx {
    use super::*;

    pub fn opt(key: &str) -> Option<String> {
        env::var(key).ok().filter(|s| !s.trim().is_empty())
    }

    pub fn req(key: &str) -> Result<String> {
        opt(key).with_context(|| format!("missing env {key}"))
    }

    pub fn u16(key: &str, default: u16) -> u16 {
        opt(key)
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(default)
    }

    pub fn usize(key: &str, default: usize) -> usize {
        opt(key)
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default)
    }

    pub fn bool(key: &str, default: bool) -> bool {
        opt(key)
            .map(|v| {
                let v = v.to_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "on"
            })
            .unwrap_or(default)
    }
}

/// ВАЖНО: даём env-override поверх toml.
/// Это критично, чтобы один и тот же билд работал и локально, и в docker/CI.
fn resolve_database_url(cfg: &AppConfig) -> String {
    envx::opt("DATABASE_URL").unwrap_or_else(|| cfg.database.url.clone())
}

fn resolve_kafka_brokers(cfg: &AppConfig) -> String {
    envx::opt("KAFKA_BROKERS").unwrap_or_else(|| cfg.rust_bot.redpanda_brokers.join(","))
}

fn resolve_topic(cfg_val: &str, env_key: &str) -> String {
    envx::opt(env_key).unwrap_or_else(|| cfg_val.to_string())
}

/// Общий контекст: cfg + DB + Kafka
pub async fn build_ctx() -> Result<AppCtx> {
    let cfg = load_config().context("load_config failed")?;

    let db_url = resolve_database_url(&cfg);
    let kafka_brokers = resolve_kafka_brokers(&cfg);

    // минимум необходимых топиков: close (остальные — сервисы могут брать сами из cfg/env)
    let topic_candles_close = resolve_topic(&cfg.rust_bot.topic_candles_close, "TOPIC_CANDLES_CLOSE");

    let db = DatabaseManager::connect(&db_url).await.context("db connect")?;
    let kafka = KafkaManager::new(&kafka_brokers, &topic_candles_close).context("kafka init")?;

    Ok(AppCtx { cfg, db, kafka })
}

