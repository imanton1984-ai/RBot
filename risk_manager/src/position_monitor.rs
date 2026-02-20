// risk_manager/src/position_monitor.rs
//
// Position Monitor
//
// Если открытые шорт-позиции + цена резко пошла UP → ALERT
// Если открытые лонг-позиции + цена резко пошла DOWN → ALERT
//
// Работает в связке с VolumePriceMonitor: получает алерты о резких движениях
// и проверяет, есть ли открытые позиции в противоположном направлении.

use crate::types::*;
use std::collections::HashMap;
use tracing::{info, warn};

/// Position Monitor
///
/// Отслеживает открытые позиции и генерирует алерты,
/// если цена движется против позиции.
pub struct PositionMonitor {
    config: RiskManagerConfig,
    /// Открытые позиции по символу
    positions: HashMap<String, Vec<OpenPosition>>,
}

impl PositionMonitor {
    pub fn new(config: RiskManagerConfig) -> Self {
        Self {
            config,
            positions: HashMap::new(),
        }
    }

    /// Обновить список открытых позиций
    pub fn update_positions(&mut self, positions: Vec<OpenPosition>) {
        self.positions.clear();
        for pos in positions {
            self.positions
                .entry(pos.symbol.clone())
                .or_default()
                .push(pos);
        }
        info!(
            "PositionMonitor: updated {} positions across {} symbols",
            self.total_positions(),
            self.positions.len()
        );
    }

    /// Добавить/обновить одну позицию
    pub fn upsert_position(&mut self, position: OpenPosition) {
        let entry = self.positions.entry(position.symbol.clone()).or_default();
        // Удаляем старую позицию с тем же side
        entry.retain(|p| p.side != position.side);
        if position.quantity.abs() > 1e-12 {
            entry.push(position);
        }
    }

    /// Удалить позицию (закрыта)
    pub fn remove_position(&mut self, symbol: &str, side: PositionSide) {
        if let Some(positions) = self.positions.get_mut(symbol) {
            positions.retain(|p| p.side != side);
            if positions.is_empty() {
                self.positions.remove(symbol);
            }
        }
    }

    /// Проверить позиции при получении алерта о резком движении цены
    ///
    /// Принимает алерты от VolumePriceMonitor и проверяет,
    /// есть ли открытые позиции в противоположном направлении.
    pub fn check_against_price_alerts(&self, price_alerts: &[RiskAlert]) -> Vec<RiskAlert> {
        let mut position_alerts = Vec::new();

        for alert in price_alerts {
            // Нас интересуют только алерты о резком движении цены
            if alert.source != AlertSource::VolumePrice {
                continue;
            }
            let change_pct = match alert.change_pct {
                Some(pct) => pct,
                None => continue,
            };

            // Проверяем позиции по этому символу
            if let Some(positions) = self.positions.get(&alert.symbol) {
                for pos in positions {
                    let is_against = match pos.side {
                        PositionSide::Long => change_pct < 0.0,  // Лонг + цена DOWN
                        PositionSide::Short => change_pct > 0.0, // Шорт + цена UP
                    };

                    if is_against {
                        let severity = if change_pct.abs() >= self.config.alert_threshold_for(&alert.symbol) * 1.5 {
                            AlertSeverity::Critical
                        } else {
                            AlertSeverity::Warning
                        };

                        let direction = if change_pct > 0.0 { "UP" } else { "DOWN" };

                        position_alerts.push(
                            RiskAlert::new(
                                AlertSource::Position,
                                severity,
                                &alert.symbol,
                                alert.tf_minutes,
                                format!(
                                    "⚠️ {} position at risk! Price moved {} by {:.2}%, \
                                     entry={:.4}, notional={:.2} USDT, unrealized_pnl={:.2}",
                                    pos.side,
                                    direction,
                                    change_pct.abs(),
                                    pos.entry_price,
                                    pos.notional_usdt,
                                    pos.unrealized_pnl,
                                ),
                            )
                            .with_price(alert.price.unwrap_or(0.0))
                            .with_change_pct(change_pct),
                        );
                    }
                }
            }
        }

        if !position_alerts.is_empty() {
            for alert in &position_alerts {
                warn!("🔴 POSITION RISK: {}", alert);
            }
        }

        position_alerts
    }

    /// Проверить позиции напрямую по индикатору (без промежуточных алертов)
    pub fn check_indicator(&self, event: &IndicatorEvent) -> Vec<RiskAlert> {
        let mut alerts = Vec::new();

        let positions = match self.positions.get(&event.symbol) {
            Some(p) if !p.is_empty() => p,
            _ => return alerts,
        };

        let current_price = match event.close {
            Some(p) if p > 0.0 => p,
            _ => return alerts,
        };

        for pos in positions {
            let change_from_entry = (current_price - pos.entry_price) / pos.entry_price * 100.0;
            let threshold = self.config.alert_threshold_for(&event.symbol);

            let is_against = match pos.side {
                PositionSide::Long => change_from_entry < -threshold,
                PositionSide::Short => change_from_entry > threshold,
            };

            if is_against {
                let severity = if change_from_entry.abs() >= threshold * 2.0 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                };

                alerts.push(
                    RiskAlert::new(
                        AlertSource::Position,
                        severity,
                        &event.symbol,
                        event.tf_minutes,
                        format!(
                            "{} position against trend: price moved {:.2}% from entry {:.4}",
                            pos.side, change_from_entry, pos.entry_price,
                        ),
                    )
                    .with_price(current_price)
                    .with_change_pct(change_from_entry),
                );
            }
        }

        alerts
    }

    /// Получить все открытые позиции
    pub fn all_positions(&self) -> Vec<&OpenPosition> {
        self.positions.values().flatten().collect()
    }

    /// Получить позиции для символа
    pub fn positions_for(&self, symbol: &str) -> Option<&Vec<OpenPosition>> {
        self.positions.get(symbol)
    }

    /// Общее количество открытых позиций
    pub fn total_positions(&self) -> usize {
        self.positions.values().map(|v| v.len()).sum()
    }

    /// Сбросить все позиции
    pub fn reset(&mut self) {
        self.positions.clear();
        info!("PositionMonitor: positions reset");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_position(symbol: &str, side: PositionSide, entry_price: f64) -> OpenPosition {
        OpenPosition {
            symbol: symbol.to_string(),
            side,
            entry_price,
            quantity: 0.1,
            notional_usdt: entry_price * 0.1,
            leverage: 10,
            open_time: Utc::now(),
            unrealized_pnl: 0.0,
        }
    }

    fn make_price_alert(symbol: &str, change_pct: f64) -> RiskAlert {
        RiskAlert::new(
            AlertSource::VolumePrice,
            AlertSeverity::Warning,
            symbol,
            1,
            "Sharp price movement",
        )
        .with_price(50000.0)
        .with_change_pct(change_pct)
    }

    #[test]
    fn test_long_position_price_down_alert() {
        let config = RiskManagerConfig::default();
        let mut monitor = PositionMonitor::new(config);

        monitor.upsert_position(make_position("BTCUSDT", PositionSide::Long, 50000.0));

        let price_alerts = vec![make_price_alert("BTCUSDT", -0.7)];
        let alerts = monitor.check_against_price_alerts(&price_alerts);

        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].source, AlertSource::Position);
        assert!(alerts[0].message.contains("LONG"));
    }

    #[test]
    fn test_short_position_price_up_alert() {
        let config = RiskManagerConfig::default();
        let mut monitor = PositionMonitor::new(config);

        monitor.upsert_position(make_position("BTCUSDT", PositionSide::Short, 50000.0));

        let price_alerts = vec![make_price_alert("BTCUSDT", 0.7)];
        let alerts = monitor.check_against_price_alerts(&price_alerts);

        assert_eq!(alerts.len(), 1);
        assert!(alerts[0].message.contains("SHORT"));
    }

    #[test]
    fn test_no_alert_same_direction() {
        let config = RiskManagerConfig::default();
        let mut monitor = PositionMonitor::new(config);

        // Лонг + цена UP — нет алерта
        monitor.upsert_position(make_position("BTCUSDT", PositionSide::Long, 50000.0));

        let price_alerts = vec![make_price_alert("BTCUSDT", 0.7)];
        let alerts = monitor.check_against_price_alerts(&price_alerts);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_no_alert_no_positions() {
        let config = RiskManagerConfig::default();
        let monitor = PositionMonitor::new(config);

        let price_alerts = vec![make_price_alert("BTCUSDT", -0.7)];
        let alerts = monitor.check_against_price_alerts(&price_alerts);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_upsert_and_remove() {
        let config = RiskManagerConfig::default();
        let mut monitor = PositionMonitor::new(config);

        monitor.upsert_position(make_position("BTCUSDT", PositionSide::Long, 50000.0));
        assert_eq!(monitor.total_positions(), 1);

        // Upsert same side — replaces
        monitor.upsert_position(make_position("BTCUSDT", PositionSide::Long, 51000.0));
        assert_eq!(monitor.total_positions(), 1);

        // Add different side
        monitor.upsert_position(make_position("BTCUSDT", PositionSide::Short, 52000.0));
        assert_eq!(monitor.total_positions(), 2);

        // Remove one
        monitor.remove_position("BTCUSDT", PositionSide::Long);
        assert_eq!(monitor.total_positions(), 1);
    }
}
