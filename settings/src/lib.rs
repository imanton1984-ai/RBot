// settings/src/lib.rs
//
// Модуль настроек торгового бота.
// Содержит:
//   - order_settings: leverage, размер позиции, тип стратегии
//   - exchange_settings: API ключи, комиссии, режим аккаунта

pub mod order_settings;
pub mod exchange_settings;

pub use order_settings::*;
pub use exchange_settings::*;

/// Загрузить все настройки разом
pub struct AllSettings {
    pub order: OrderSettings,
    pub exchange: ExchangeSettings,
}

impl AllSettings {
    /// Загрузить все настройки из стандартных путей
    pub fn load() -> anyhow::Result<Self> {
        let order = OrderSettings::load()?;
        let exchange = ExchangeSettings::load()?;
        Ok(Self { order, exchange })
    }

    /// Загрузить с override из env-переменных
    pub fn load_with_env() -> anyhow::Result<Self> {
        let order = OrderSettings::load_with_env()?;
        let exchange = ExchangeSettings::load_with_env()?;
        Ok(Self { order, exchange })
    }
}
