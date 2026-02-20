// risk_manager/src/main.rs
//
// Risk Manager Service
//
// Запускает все мониторы в реал-тайме:
//   1. Подписывается на indicators.close (Kafka/Redpanda)
//   2. Подписывается на positions.events (Kafka/Redpanda)
//   3. Обрабатывает события через VolumePriceMonitor, TrendChangeMonitor, PositionMonitor
//   4. Публикует алерты в risk.alerts
//   5. При включённом PositionCloser — отправляет команды закрытия в orders.cmd

use anyhow::Result;
use futures::StreamExt;
use risk_manager::*;
use common::{MessageBus, MessageBusConfig, Codec};
use tracing::{info, error, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Инициализация логирования
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!("═══════════════════════════════════════════════════");
    info!("  Risk Manager Service starting...");
    info!("═══════════════════════════════════════════════════");

    // Загружаем конфигурацию
    let config = RiskManagerConfig::load_with_env()?;

    if !config.enabled {
        warn!("Risk Manager is DISABLED in config. Exiting.");
        return Ok(());
    }

    info!("Config loaded:");
    info!("  BTC threshold: {:.1}%", config.btc_alert_threshold_pct);
    info!("  Alt threshold: {:.1}%", config.alt_alert_threshold_pct);
    info!("  Volume spike threshold: {:.1}x", config.volume_spike_threshold);
    info!("  Monitor TFs: {:?}", config.monitor_timeframes);
    info!("  Position closer: {}", if config.position_closer_enabled { "ENABLED" } else { "DISABLED" });
    info!("  Kafka brokers: {}", config.kafka_brokers);

    // Инициализация мониторов
    let mut volume_price_monitor = VolumePriceMonitor::new(config.clone());
    let mut trend_change_monitor = TrendChangeMonitor::new(config.clone());
    let position_monitor = PositionMonitor::new(config.clone());
    let mut position_closer = PositionCloser::new(config.position_closer_enabled);

    // Инициализация Kafka
    let bus_config = MessageBusConfig {
        brokers: config.kafka_brokers.clone(),
        topic_prefix: String::new(), // Без префикса — используем полные имена топиков
        message_timeout_ms: 5000,
        retry_backoff_ms: 100,
        max_in_flight: 10_000,
        flush_interval_ms: 25,
        batch_max_messages: 1000,
        auto_offset_reset: "latest".to_string(),
        enable_auto_commit: true,
    };

    let bus = MessageBus::new(bus_config)?;

    // Подписка на индикаторы
    let mut indicator_stream = bus
        .subscribe::<IndicatorEvent>(
            &config.topic_indicators,
            &config.kafka_group_id,
            Codec::Json,
        )
        .await?;

    info!("Subscribed to topic: {}", config.topic_indicators);
    info!("Risk Manager is running. Waiting for events...");

    // Основной цикл обработки событий
    let mut total_events: u64 = 0;
    let mut total_alerts: u64 = 0;

    while let Some(msg) = indicator_stream.next().await {
        match msg {
            Ok(incoming) => {
                let event = incoming.payload;
                total_events += 1;

                if total_events % 10_000 == 0 {
                    info!(
                        "Processed {} events, {} alerts generated, {} positions tracked",
                        total_events,
                        total_alerts,
                        position_monitor.total_positions(),
                    );
                }

                // 1. Volume + Price Monitor
                let price_alerts = volume_price_monitor.process_indicator(&event);

                // 2. Trend Change Monitor
                let trend_alerts = trend_change_monitor.process_indicator(&event);

                // 3. Position Monitor (проверяем позиции при ценовых алертах)
                let position_alerts = position_monitor.check_against_price_alerts(&price_alerts);

                // 4. Position Closer (если включён)
                let close_commands = position_closer.process_position_alerts(&position_alerts);

                // Собираем все алерты
                let mut all_alerts = Vec::new();
                all_alerts.extend(price_alerts);
                all_alerts.extend(trend_alerts);
                all_alerts.extend(position_alerts);

                // Публикуем алерты в Kafka
                for alert in &all_alerts {
                    if let Err(e) = bus
                        .publish(
                            &config.topic_alerts,
                            alert.symbol.as_bytes(),
                            alert,
                            Codec::Json,
                        )
                        .await
                    {
                        error!("Failed to publish risk alert: {}", e);
                    }
                }

                // Публикуем команды закрытия
                for cmd in &close_commands {
                    if let Err(e) = bus
                        .publish(
                            &config.topic_orders_cmd,
                            cmd.symbol.as_bytes(),
                            cmd,
                            Codec::Json,
                        )
                        .await
                    {
                        error!("Failed to publish close command: {}", e);
                    }
                }

                total_alerts += all_alerts.len() as u64;
            }
            Err(e) => {
                error!("Error receiving indicator event: {}", e);
            }
        }
    }

    warn!("Indicator stream ended. Risk Manager shutting down.");
    Ok(())
}
