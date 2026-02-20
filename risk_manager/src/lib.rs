// risk_manager/src/lib.rs
//
// Risk Manager — модуль управления рисками торгового бота.
//
// Компоненты:
//   1. VolumePriceMonitor — мониторинг резких движений цены/объёма
//   2. TrendChangeMonitor — отслеживание смены тренда
//   3. PositionMonitor — мониторинг открытых позиций против тренда
//   4. PositionCloser — автоматическое закрытие позиций (togglable)
//
// Все мониторы работают в реал-тайме через Kafka/Redpanda,
// получая данные из market.indicators_wide.

pub mod types;
pub mod volume_price_monitor;
pub mod trend_change_monitor;
pub mod position_monitor;
pub mod position_closer;

pub use types::*;
pub use volume_price_monitor::VolumePriceMonitor;
pub use trend_change_monitor::TrendChangeMonitor;
pub use position_monitor::PositionMonitor;
pub use position_closer::PositionCloser;
