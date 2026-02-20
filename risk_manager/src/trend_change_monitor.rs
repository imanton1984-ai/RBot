// risk_manager/src/trend_change_monitor.rs
//
// Trend Change Monitor
//
// Отслеживание смены тренда (trend + trend_short индикаторы из indicators_wide)
// На BTC и Top 20 альтов.
//
// trend:       1 = uptrend, -1 = downtrend, 0 = neutral
// trend_short: 1 = short-term up, -1 = short-term down, 0 = neutral

use crate::types::*;
use std::collections::HashMap;
use tracing::{info, warn};

/// Состояние тренда для символа
#[derive(Debug, Clone)]
struct TrendState {
    /// Предыдущий trend
    prev_trend: i16,
    /// Предыдущий trend_short
    prev_trend_short: i16,
    /// Время последнего обновления
    #[allow(dead_code)]
    last_time_ms: i64,
    /// Количество последовательных смен тренда (для фильтрации шума)
    consecutive_changes: u32,
}

/// Тип смены тренда
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendChangeType {
    /// Основной тренд сменился
    MainTrend,
    /// Краткосрочный тренд сменился
    ShortTrend,
    /// Оба тренда сменились одновременно (сильный сигнал)
    BothTrends,
}

impl std::fmt::Display for TrendChangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MainTrend => write!(f, "MAIN_TREND"),
            Self::ShortTrend => write!(f, "SHORT_TREND"),
            Self::BothTrends => write!(f, "BOTH_TRENDS"),
        }
    }
}

/// Trend Change Monitor
///
/// Отслеживает смену тренда на BTC и Top альткоинах.
/// Генерирует алерты при смене направления trend/trend_short.
pub struct TrendChangeMonitor {
    config: RiskManagerConfig,
    /// Состояние по символу + таймфрейму
    states: HashMap<(String, i16), TrendState>,
}

impl TrendChangeMonitor {
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

        let current_trend = match event.trend {
            Some(t) => t,
            None => return alerts,
        };
        let current_trend_short = match event.trend_short {
            Some(t) => t,
            None => return alerts,
        };

        let key = (event.symbol.clone(), event.tf_minutes);

        if let Some(prev_state) = self.states.get(&key) {
            let main_changed = current_trend != prev_state.prev_trend
                && prev_state.prev_trend != 0
                && current_trend != 0;
            let short_changed = current_trend_short != prev_state.prev_trend_short
                && prev_state.prev_trend_short != 0
                && current_trend_short != 0;

            if main_changed && short_changed {
                // Оба тренда сменились — сильный сигнал
                let direction = trend_direction_str(current_trend);
                alerts.push(
                    RiskAlert::new(
                        AlertSource::TrendChange,
                        AlertSeverity::Critical,
                        &event.symbol,
                        event.tf_minutes,
                        format!(
                            "BOTH trends reversed to {} (main: {} → {}, short: {} → {})",
                            direction,
                            trend_direction_str(prev_state.prev_trend),
                            trend_direction_str(current_trend),
                            trend_direction_str(prev_state.prev_trend_short),
                            trend_direction_str(current_trend_short),
                        ),
                    )
                    .with_price(event.close.unwrap_or(0.0)),
                );
            } else if main_changed {
                let direction = trend_direction_str(current_trend);
                alerts.push(
                    RiskAlert::new(
                        AlertSource::TrendChange,
                        AlertSeverity::Warning,
                        &event.symbol,
                        event.tf_minutes,
                        format!(
                            "Main trend changed: {} → {}",
                            trend_direction_str(prev_state.prev_trend),
                            direction,
                        ),
                    )
                    .with_price(event.close.unwrap_or(0.0)),
                );
            } else if short_changed {
                let direction = trend_direction_str(current_trend_short);
                alerts.push(
                    RiskAlert::new(
                        AlertSource::TrendChange,
                        AlertSeverity::Info,
                        &event.symbol,
                        event.tf_minutes,
                        format!(
                            "Short-term trend changed: {} → {}",
                            trend_direction_str(prev_state.prev_trend_short),
                            direction,
                        ),
                    )
                    .with_price(event.close.unwrap_or(0.0)),
                );
            }
        }

        // Обновляем состояние
        let consecutive = self
            .states
            .get(&key)
            .map(|s| {
                if current_trend != s.prev_trend || current_trend_short != s.prev_trend_short {
                    s.consecutive_changes + 1
                } else {
                    0
                }
            })
            .unwrap_or(0);

        self.states.insert(
            key,
            TrendState {
                prev_trend: current_trend,
                prev_trend_short: current_trend_short,
                last_time_ms: event.time_ms,
                consecutive_changes: consecutive,
            },
        );

        if !alerts.is_empty() {
            for alert in &alerts {
                match alert.severity {
                    AlertSeverity::Critical => warn!("🔄 TREND ALERT: {}", alert),
                    AlertSeverity::Warning => warn!("🔄 TREND ALERT: {}", alert),
                    AlertSeverity::Info => info!("🔄 TREND INFO: {}", alert),
                }
            }
        }

        alerts
    }

    /// Получить текущий тренд для символа
    pub fn get_trend(&self, symbol: &str, tf_minutes: i16) -> Option<(i16, i16)> {
        self.states
            .get(&(symbol.to_string(), tf_minutes))
            .map(|s| (s.prev_trend, s.prev_trend_short))
    }

    /// Сбросить состояние
    pub fn reset(&mut self) {
        self.states.clear();
        info!("TrendChangeMonitor: state reset");
    }

    /// Количество отслеживаемых пар
    pub fn tracked_count(&self) -> usize {
        self.states.len()
    }
}

/// Преобразовать числовой тренд в строку
fn trend_direction_str(trend: i16) -> &'static str {
    match trend {
        1 => "UP",
        -1 => "DOWN",
        _ => "NEUTRAL",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(symbol: &str, tf: i16, trend: i16, trend_short: i16) -> IndicatorEvent {
        IndicatorEvent {
            symbol: symbol.to_string(),
            tf_minutes: tf,
            time_ms: 1000,
            volume_spike: Some(1.0),
            trend: Some(trend),
            trend_short: Some(trend_short),
            close: Some(50000.0),
            rsi: Some(50.0),
            atr: Some(100.0),
        }
    }

    #[test]
    fn test_no_alert_first_event() {
        let config = RiskManagerConfig::default();
        let mut monitor = TrendChangeMonitor::new(config);

        let event = make_event("BTCUSDT", 1, 1, 1);
        let alerts = monitor.process_indicator(&event);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_main_trend_change() {
        let config = RiskManagerConfig::default();
        let mut monitor = TrendChangeMonitor::new(config);

        // Устанавливаем baseline: uptrend
        let event1 = make_event("BTCUSDT", 1, 1, 1);
        monitor.process_indicator(&event1);

        // Смена main trend на downtrend
        let event2 = make_event("BTCUSDT", 1, -1, 1);
        let alerts = monitor.process_indicator(&event2);
        assert_eq!(alerts.len(), 1);
        assert!(alerts[0].message.contains("Main trend changed"));
        assert_eq!(alerts[0].severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_short_trend_change() {
        let config = RiskManagerConfig::default();
        let mut monitor = TrendChangeMonitor::new(config);

        let event1 = make_event("BTCUSDT", 1, 1, 1);
        monitor.process_indicator(&event1);

        let event2 = make_event("BTCUSDT", 1, 1, -1);
        let alerts = monitor.process_indicator(&event2);
        assert_eq!(alerts.len(), 1);
        assert!(alerts[0].message.contains("Short-term trend"));
        assert_eq!(alerts[0].severity, AlertSeverity::Info);
    }

    #[test]
    fn test_both_trends_change() {
        let config = RiskManagerConfig::default();
        let mut monitor = TrendChangeMonitor::new(config);

        let event1 = make_event("BTCUSDT", 1, 1, 1);
        monitor.process_indicator(&event1);

        // Оба тренда сменились
        let event2 = make_event("BTCUSDT", 1, -1, -1);
        let alerts = monitor.process_indicator(&event2);
        assert_eq!(alerts.len(), 1);
        assert!(alerts[0].message.contains("BOTH trends"));
        assert_eq!(alerts[0].severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_no_alert_same_trend() {
        let config = RiskManagerConfig::default();
        let mut monitor = TrendChangeMonitor::new(config);

        let event1 = make_event("BTCUSDT", 1, 1, 1);
        monitor.process_indicator(&event1);

        // Тот же тренд — нет алерта
        let event2 = make_event("BTCUSDT", 1, 1, 1);
        let alerts = monitor.process_indicator(&event2);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_ignores_non_monitored_symbol() {
        let config = RiskManagerConfig::default();
        let mut monitor = TrendChangeMonitor::new(config);

        let event1 = make_event("RANDOMUSDT", 1, 1, 1);
        monitor.process_indicator(&event1);

        let event2 = make_event("RANDOMUSDT", 1, -1, -1);
        let alerts = monitor.process_indicator(&event2);
        assert!(alerts.is_empty());
    }
}
