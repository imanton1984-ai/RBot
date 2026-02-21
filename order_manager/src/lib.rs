// order_manager/src/lib.rs
//
// Order Manager — полноценная система создания, отслеживания
// и закрытия позиций на Binance Futures.
//
// Компоненты:
//   1. SignalScanner — сканирует trade.super_entry_signals на свежие сигналы
//   2. OrderExecutor — открывает/закрывает позиции, записывает в БД
//   3. PositionTracker — отслеживает PnL, candles_left, отправляет в WebUI
//
// Пропорции одновременных позиций (при 10 ордерах):
//   - 1h:  70% → 7 слотов
//   - 4h:  20% → 2 слота
//   - 15m: 10% → 1 слот

pub mod types;
pub mod config;
pub mod exchange_info;
pub mod signal_scanner;
pub mod order_executor;
pub mod position_tracker;

#[cfg(test)]
mod tests_synthetic;

pub use types::*;
pub use config::OrderManagerConfig;
pub use exchange_info::ExchangeInfoCache;
pub use signal_scanner::SignalScanner;
pub use order_executor::OrderExecutor;
pub use position_tracker::PositionTracker;
