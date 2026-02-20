// risk_manager/src/position_closer.rs
//
// Position Closer (togglable)
//
// Включаемый/отключаемый модуль.
// Автоматически закрывает позиции, открытые против тренда при резком движении.
//
// Логика:
//   1. Получает алерты от PositionMonitor (позиция против тренда)
//   2. Проверяет, что position_closer_enabled = true
//   3. Отправляет команду закрытия через Kafka topic orders.cmd

use crate::types::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{info, warn, error};
use serde::{Deserialize, Serialize};

/// Команда закрытия позиции (отправляется в orders.cmd)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosePositionCommand {
    /// Тип команды
    pub cmd_type: String,
    /// Символ
    pub symbol: String,
    /// Сторона позиции для закрытия
    pub side: String,
    /// Причина закрытия
    pub reason: String,
    /// Время команды (ms)
    pub timestamp_ms: i64,
    /// Источник команды
    pub source: String,
}

/// Position Closer
///
/// Автоматически закрывает позиции при резком движении против тренда.
/// Может быть включён/отключён в runtime через toggle().
pub struct PositionCloser {
    /// Включён ли closer (атомарный для потокобезопасности)
    enabled: Arc<AtomicBool>,
    /// Счётчик закрытых позиций
    closed_count: u64,
    /// Последние команды закрытия (для аудита)
    recent_commands: Vec<ClosePositionCommand>,
    /// Максимальный размер истории команд
    max_history: usize,
}

impl PositionCloser {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled: Arc::new(AtomicBool::new(enabled)),
            closed_count: 0,
            recent_commands: Vec::new(),
            max_history: 100,
        }
    }

    /// Включить/отключить position closer
    pub fn toggle(&self, enabled: bool) {
        let prev = self.enabled.swap(enabled, Ordering::SeqCst);
        if prev != enabled {
            if enabled {
                warn!("🟢 Position Closer ENABLED — positions against trend will be auto-closed");
            } else {
                info!("🔴 Position Closer DISABLED — manual management only");
            }
        }
    }

    /// Проверить, включён ли closer
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Получить Arc для шаринга между задачами
    pub fn enabled_flag(&self) -> Arc<AtomicBool> {
        self.enabled.clone()
    }

    /// Обработать алерты от PositionMonitor и сгенерировать команды закрытия
    ///
    /// Возвращает список команд для отправки в Kafka (orders.cmd)
    pub fn process_position_alerts(&mut self, alerts: &[RiskAlert]) -> Vec<ClosePositionCommand> {
        if !self.is_enabled() {
            return Vec::new();
        }

        let mut commands = Vec::new();

        for alert in alerts {
            if alert.source != AlertSource::Position {
                continue;
            }

            // Только Critical алерты вызывают автоматическое закрытие
            if alert.severity != AlertSeverity::Critical {
                info!(
                    "Position alert for {} is {:?}, skipping auto-close (only Critical triggers close)",
                    alert.symbol, alert.severity
                );
                continue;
            }

            // Определяем сторону позиции из сообщения
            let side = if alert.message.contains("LONG") {
                "LONG"
            } else if alert.message.contains("SHORT") {
                "SHORT"
            } else {
                error!("Cannot determine position side from alert: {}", alert.message);
                continue;
            };

            let cmd = ClosePositionCommand {
                cmd_type: "close_position".to_string(),
                symbol: alert.symbol.clone(),
                side: side.to_string(),
                reason: format!(
                    "Risk Manager auto-close: {} (change: {:.2}%)",
                    alert.message,
                    alert.change_pct.unwrap_or(0.0)
                ),
                timestamp_ms: chrono::Utc::now().timestamp_millis(),
                source: "risk_manager.position_closer".to_string(),
            };

            warn!(
                "🔴 AUTO-CLOSE: {} {} position on {} — {}",
                side, alert.symbol, alert.tf_minutes, cmd.reason
            );

            self.closed_count += 1;

            // Сохраняем в историю
            if self.recent_commands.len() >= self.max_history {
                self.recent_commands.remove(0);
            }
            self.recent_commands.push(cmd.clone());

            commands.push(cmd);
        }

        commands
    }

    /// Получить количество автоматически закрытых позиций
    pub fn closed_count(&self) -> u64 {
        self.closed_count
    }

    /// Получить последние команды закрытия
    pub fn recent_commands(&self) -> &[ClosePositionCommand] {
        &self.recent_commands
    }

    /// Сбросить счётчик и историю
    pub fn reset_stats(&mut self) {
        self.closed_count = 0;
        self.recent_commands.clear();
        info!("PositionCloser: stats reset");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_critical_position_alert(symbol: &str, side: &str) -> RiskAlert {
        RiskAlert::new(
            AlertSource::Position,
            AlertSeverity::Critical,
            symbol,
            1,
            format!("{} position at risk! Price moved sharply", side),
        )
        .with_price(50000.0)
        .with_change_pct(-1.0)
    }

    fn make_warning_position_alert(symbol: &str, side: &str) -> RiskAlert {
        RiskAlert::new(
            AlertSource::Position,
            AlertSeverity::Warning,
            symbol,
            1,
            format!("{} position at risk! Price moved", side),
        )
        .with_price(50000.0)
        .with_change_pct(-0.5)
    }

    #[test]
    fn test_disabled_no_commands() {
        let mut closer = PositionCloser::new(false);
        let alerts = vec![make_critical_position_alert("BTCUSDT", "LONG")];
        let commands = closer.process_position_alerts(&alerts);
        assert!(commands.is_empty());
    }

    #[test]
    fn test_enabled_critical_generates_command() {
        let mut closer = PositionCloser::new(true);
        let alerts = vec![make_critical_position_alert("BTCUSDT", "LONG")];
        let commands = closer.process_position_alerts(&alerts);
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].symbol, "BTCUSDT");
        assert_eq!(commands[0].side, "LONG");
        assert_eq!(commands[0].cmd_type, "close_position");
        assert_eq!(closer.closed_count(), 1);
    }

    #[test]
    fn test_warning_does_not_trigger_close() {
        let mut closer = PositionCloser::new(true);
        let alerts = vec![make_warning_position_alert("BTCUSDT", "LONG")];
        let commands = closer.process_position_alerts(&alerts);
        assert!(commands.is_empty());
    }

    #[test]
    fn test_toggle() {
        let closer = PositionCloser::new(false);
        assert!(!closer.is_enabled());

        closer.toggle(true);
        assert!(closer.is_enabled());

        closer.toggle(false);
        assert!(!closer.is_enabled());
    }

    #[test]
    fn test_short_position_close() {
        let mut closer = PositionCloser::new(true);
        let alerts = vec![make_critical_position_alert("ETHUSDT", "SHORT")];
        let commands = closer.process_position_alerts(&alerts);
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].side, "SHORT");
    }

    #[test]
    fn test_history_limit() {
        let mut closer = PositionCloser::new(true);
        closer.max_history = 3;

        for i in 0..5 {
            let alerts = vec![make_critical_position_alert(
                &format!("SYM{}USDT", i),
                "LONG",
            )];
            closer.process_position_alerts(&alerts);
        }

        assert_eq!(closer.recent_commands().len(), 3);
        assert_eq!(closer.closed_count(), 5);
    }
}
