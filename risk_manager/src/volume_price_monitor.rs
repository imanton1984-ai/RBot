// risk_manager/src/volume_price_monitor.rs
//
// Volume + Price Monitor
//
// Мониторинг BTC: алерт при резких движениях 0.5-0.7%+
// Мониторинг Top 20 altcoins: алерт при 1.5-2%+
// Источник: market.indicators_wide (volume_spike, trend, trend_short) на TF 1m, 5m, 15m
// Работает в реал-тайме через Redpanda/Kafka

use crate::types::*;
use std::collections::HashMap;
use tracing::{info, warn};

/// Состояние цены для отслеживания изменений
#[derive(Debug, Clone)]
pub struct PriceState {
    /// Последняя известная цена
    pub last_price: f64,
    /// Время последнего обновления (ms)
    pub last_time_ms: i64,
    /// Последний volume_spike
    pub last_volume_spike: f32,
    /// Последний trend
    pub last_trend: i16,
    /// Последний trend_short
    pub last_trend_short: i16,
}

/// Volume + Price Monitor
///
/// Отслеживает резкие движения цены и объёма на BTC и Top альткоинах.
/// Генерирует алерты при превышении порогов.
pub struct VolumePriceMonitor {
    config: RiskManagerConfig,
    /// Состояние по символу + таймфрейму: (symbol, tf_minutes) -> PriceState
    states: HashMap<(String, i16), PriceState>,
}

impl VolumePriceMonitor {
    pub fn new(config: RiskManagerConfig) -> Self {
        Self {
            config,
            states: HashMap::new(),
        }
    }

    /// Обработать новый индикатор и вернуть алерты (если есть)
    pub fn process_indicator(&mut self, event: &IndicatorEvent) -> Vec<RiskAlert> {
        let mut alerts = Vec::new();

        // Проверяем только BTC и Top альты
        if !self.config.is_btc(&event.symbol) && !self.config.is_top_alt(&event.symbol) {
            return alerts;
        }

        // Проверяем только мониторируемые таймфреймы
        if !self.config.monitor_timeframes.contains(&event.tf_minutes) {
            return alerts;
        }

        let key = (event.symbol.clone(), event.tf_minutes);

        // Получаем текущую цену
        let current_price = match event.close {
            Some(p) if p > 0.0 => p,
            _ => return alerts,
        };

        // Проверяем volume spike
        if let Some(vs) = event.volume_spike {
            if vs >= self.config.volume_spike_threshold {
                let severity = if vs >= self.config.volume_spike_threshold * 2.0 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                };

                alerts.push(
                    RiskAlert::new(
                        AlertSource::VolumePrice,
                        severity,
                        &event.symbol,
                        event.tf_minutes,
                        format!(
                            "Volume spike detected: {:.1}x (threshold: {:.1}x)",
                            vs, self.config.volume_spike_threshold
                        ),
                    )
                    .with_price(current_price),
                );
            }
        }

        // Проверяем изменение цены
        if let Some(prev_state) = self.states.get(&key) {
            let change_pct = (current_price - prev_state.last_price) / prev_state.last_price * 100.0;
            let abs_change = change_pct.abs();
            let threshold = self.config.alert_threshold_for(&event.symbol);

            if abs_change >= threshold {
                let direction = if change_pct > 0.0 { "UP" } else { "DOWN" };
                let severity = if abs_change >= threshold * 1.5 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                };

                alerts.push(
                    RiskAlert::new(
                        AlertSource::VolumePrice,
                        severity,
                        &event.symbol,
                        event.tf_minutes,
                        format!(
                            "Sharp price movement {}: {:.2}% (threshold: {:.1}%)",
                            direction, change_pct, threshold
                        ),
                    )
                    .with_price(current_price)
                    .with_change_pct(change_pct),
                );
            }
        }

        // Обновляем состояние
        self.states.insert(
            key,
            PriceState {
                last_price: current_price,
                last_time_ms: event.time_ms,
                last_volume_spike: event.volume_spike.unwrap_or(0.0),
                last_trend: event.trend.unwrap_or(0),
                last_trend_short: event.trend_short.unwrap_or(0),
            },
        );

        if !alerts.is_empty() {
            for alert in &alerts {
                match alert.severity {
                    AlertSeverity::Critical => warn!("🚨 RISK ALERT: {}", alert),
                    AlertSeverity::Warning => warn!("⚠️  RISK ALERT: {}", alert),
                    AlertSeverity::Info => info!("ℹ️  RISK INFO: {}", alert),
                }
            }
        }

        alerts
    }

    /// Получить текущее состояние для символа
    pub fn get_state(&self, symbol: &str, tf_minutes: i16) -> Option<&PriceState> {
        self.states.get(&(symbol.to_string(), tf_minutes))
    }

    /// Сбросить состояние (при переподключении)
    pub fn reset(&mut self) {
        self.states.clear();
        info!("VolumePriceMonitor: state reset");
    }

    /// Количество отслеживаемых пар (symbol, tf)
    pub fn tracked_count(&self) -> usize {
        self.states.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(symbol: &str, tf: i16, close: f64, volume_spike: f32) -> IndicatorEvent {
        IndicatorEvent {
            symbol: symbol.to_string(),
            tf_minutes: tf,
            time_ms: 1000,
            volume_spike: Some(volume_spike),
            trend: Some(1),
            trend_short: Some(1),
            close: Some(close),
            rsi: Some(50.0),
            atr: Some(100.0),
        }
    }

    #[test]
    fn test_no_alert_for_unknown_symbol() {
        let config = RiskManagerConfig::default();
        let mut monitor = VolumePriceMonitor::new(config);
        let event = make_event("RANDOMUSDT", 1, 100.0, 1.0);
        let alerts = monitor.process_indicator(&event);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_volume_spike_alert() {
        let config = RiskManagerConfig::default();
        let mut monitor = VolumePriceMonitor::new(config);

        // Volume spike >= 2.0 на BTC
        let event = make_event("BTCUSDT", 1, 50000.0, 3.0);
        let alerts = monitor.process_indicator(&event);
        assert!(!alerts.is_empty());
        assert!(alerts.iter().any(|a| a.source == AlertSource::VolumePrice));
    }

    #[test]
    fn test_price_movement_alert_btc() {
        let config = RiskManagerConfig::default();
        let mut monitor = VolumePriceMonitor::new(config);

        // Первое обновление — устанавливаем baseline
        let event1 = make_event("BTCUSDT", 1, 50000.0, 1.0);
        let alerts1 = monitor.process_indicator(&event1);
        // Может быть volume spike alert, но не price movement
        assert!(alerts1.iter().all(|a| !a.message.contains("Sharp price")));

        // Второе обновление — движение 0.6% (> 0.5% порог)
        let event2 = make_event("BTCUSDT", 1, 50300.0, 1.0);
        let alerts2 = monitor.process_indicator(&event2);
        assert!(alerts2.iter().any(|a| a.message.contains("Sharp price")));
    }

    #[test]
    fn test_price_movement_no_alert_below_threshold() {
        let config = RiskManagerConfig::default();
        let mut monitor = VolumePriceMonitor::new(config);

        let event1 = make_event("BTCUSDT", 1, 50000.0, 1.0);
        monitor.process_indicator(&event1);

        // Движение 0.1% (< 0.5% порог)
        let event2 = make_event("BTCUSDT", 1, 50050.0, 1.0);
        let alerts = monitor.process_indicator(&event2);
        assert!(alerts.iter().all(|a| !a.message.contains("Sharp price")));
    }

    #[test]
    fn test_alt_higher_threshold() {
        let config = RiskManagerConfig::default();
        let mut monitor = VolumePriceMonitor::new(config);

        let event1 = make_event("ETHUSDT", 1, 3000.0, 1.0);
        monitor.process_indicator(&event1);

        // Движение 1.0% — ниже порога 1.5% для альтов
        let event2 = make_event("ETHUSDT", 1, 3030.0, 1.0);
        let alerts = monitor.process_indicator(&event2);
        assert!(alerts.iter().all(|a| !a.message.contains("Sharp price")));

        // Движение 2.0% — выше порога 1.5%
        let event3 = make_event("ETHUSDT", 1, 3090.6, 1.0);
        let alerts = monitor.process_indicator(&event3);
        assert!(alerts.iter().any(|a| a.message.contains("Sharp price")));
    }

    #[test]
    fn test_reset() {
        let config = RiskManagerConfig::default();
        let mut monitor = VolumePriceMonitor::new(config);

        let event = make_event("BTCUSDT", 1, 50000.0, 1.0);
        monitor.process_indicator(&event);
        assert_eq!(monitor.tracked_count(), 1);

        monitor.reset();
        assert_eq!(monitor.tracked_count(), 0);
    }
}
