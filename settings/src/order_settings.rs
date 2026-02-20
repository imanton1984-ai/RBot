// settings/src/order_settings.rs
//
// Настройки ордеров: leverage, размер позиции, тип стратегии, тип ордера.
// Загружаются из config/order_settings.toml или env-переменных.

use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{info, warn};

/// Тип размера позиции
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TradeSizeType {
    /// Фиксированная сумма в USDT
    FixedUsdt,
    /// Процент от депозита
    PercentDepo,
}

impl Default for TradeSizeType {
    fn default() -> Self {
        Self::FixedUsdt
    }
}

impl std::fmt::Display for TradeSizeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FixedUsdt => write!(f, "fixed_usdt"),
            Self::PercentDepo => write!(f, "percent_depo"),
        }
    }
}

/// Тип стратегии
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyType {
    /// ML Super Entry Strategy
    MlSuperEntry,
    /// Level-based Strategy
    LevelStrategy,
}

impl Default for StrategyType {
    fn default() -> Self {
        Self::MlSuperEntry
    }
}

impl std::fmt::Display for StrategyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MlSuperEntry => write!(f, "ml_super_entry"),
            Self::LevelStrategy => write!(f, "level_strategy"),
        }
    }
}

/// Тип ордера
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderType {
    /// Futures OCO (One-Cancels-Other): TP + SL
    FuturesOco,
}

impl Default for OrderType {
    fn default() -> Self {
        Self::FuturesOco
    }
}

impl std::fmt::Display for OrderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FuturesOco => write!(f, "futures_oco"),
        }
    }
}

/// Режим торговли (переключаемый в runtime)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TradingMode {
    /// Автоматическая торговля — бот открывает/закрывает позиции самостоятельно
    Auto,
    /// Ручной режим — бот генерирует сигналы, но не открывает позиции.
    /// Пользователь подтверждает каждую сделку вручную (через WebUI/API).
    Manual,
    /// Выключен — бот не генерирует сигналы и не торгует.
    /// Используется для обслуживания или паузы.
    Off,
}

impl Default for TradingMode {
    fn default() -> Self {
        Self::Manual // Безопасный дефолт — не торгуем без явного включения
    }
}

impl std::fmt::Display for TradingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Manual => write!(f, "manual"),
            Self::Off => write!(f, "off"),
        }
    }
}

impl TradingMode {
    /// Можно ли автоматически открывать позиции
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    /// Генерируются ли сигналы (auto или manual)
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::Off)
    }

    /// Полностью выключен
    pub fn is_off(&self) -> bool {
        matches!(self, Self::Off)
    }
}

/// Настройки ордеров
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderSettings {
    /// Кредитное плечо (1-125 для Binance Futures)
    pub leverage: u16,

    /// Максимальное количество одновременно открытых ордеров
    pub max_orders_at_a_time: u16,

    /// Тип размера позиции
    pub trade_size_type: TradeSizeType,

    /// Значение размера позиции:
    /// - Если trade_size_type = FixedUsdt → сумма в USDT (напр. 100.0)
    /// - Если trade_size_type = PercentDepo → процент от депозита (напр. 10.0 = 10%)
    pub trade_size_value: f64,

    /// Тип стратегии
    pub strategy_type: StrategyType,

    /// Тип ордера
    pub order_type: OrderType,

    /// Режим торговли: auto / manual / off
    /// - auto: бот торгует автоматически
    /// - manual: бот генерирует сигналы, пользователь подтверждает
    /// - off: бот не торгует (пауза/обслуживание)
    /// Переключается через ENV: ORDER_TRADING_MODE=auto|manual|off
    pub trading_mode: TradingMode,
}

impl Default for OrderSettings {
    fn default() -> Self {
        Self {
            leverage: 10,
            max_orders_at_a_time: 10,
            trade_size_type: TradeSizeType::FixedUsdt,
            trade_size_value: 100.0,
            strategy_type: StrategyType::MlSuperEntry,
            order_type: OrderType::FuturesOco,
            trading_mode: TradingMode::Manual,
        }
    }
}

impl OrderSettings {
    /// Загрузить из TOML-файла. Если файл не найден — вернуть дефолт.
    pub fn load_from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            warn!("Order settings file not found at {:?}, using defaults", path);
            return Ok(Self::default());
        }

        let content = std::fs::read_to_string(path)?;
        let settings: Self = toml::from_str(&content)?;
        info!(
            "Loaded order settings: leverage={}, max_orders={}, size_type={}, size_value={}, strategy={}, order_type={}, trading_mode={}",
            settings.leverage,
            settings.max_orders_at_a_time,
            settings.trade_size_type,
            settings.trade_size_value,
            settings.strategy_type,
            settings.order_type,
            settings.trading_mode,
        );
        settings.validate()?;
        Ok(settings)
    }

    /// Загрузить из стандартного пути config/order_settings.toml
    pub fn load() -> anyhow::Result<Self> {
        Self::load_from_file("config/order_settings.toml")
    }

    /// Загрузить с override из env-переменных
    pub fn load_with_env() -> anyhow::Result<Self> {
        let mut settings = Self::load()?;

        if let Ok(v) = std::env::var("ORDER_LEVERAGE") {
            if let Ok(n) = v.parse() {
                settings.leverage = n;
            }
        }
        if let Ok(v) = std::env::var("ORDER_MAX_ORDERS") {
            if let Ok(n) = v.parse() {
                settings.max_orders_at_a_time = n;
            }
        }
        if let Ok(v) = std::env::var("ORDER_TRADE_SIZE_VALUE") {
            if let Ok(n) = v.parse() {
                settings.trade_size_value = n;
            }
        }
        if let Ok(v) = std::env::var("ORDER_TRADE_SIZE_TYPE") {
            match v.as_str() {
                "fixed_usdt" => settings.trade_size_type = TradeSizeType::FixedUsdt,
                "percent_depo" => settings.trade_size_type = TradeSizeType::PercentDepo,
                _ => warn!("Unknown ORDER_TRADE_SIZE_TYPE: {}", v),
            }
        }
        if let Ok(v) = std::env::var("ORDER_STRATEGY_TYPE") {
            match v.as_str() {
                "ml_super_entry" => settings.strategy_type = StrategyType::MlSuperEntry,
                "level_strategy" => settings.strategy_type = StrategyType::LevelStrategy,
                _ => warn!("Unknown ORDER_STRATEGY_TYPE: {}", v),
            }
        }
        if let Ok(v) = std::env::var("ORDER_TRADING_MODE") {
            match v.to_lowercase().as_str() {
                "auto" => settings.trading_mode = TradingMode::Auto,
                "manual" => settings.trading_mode = TradingMode::Manual,
                "off" => settings.trading_mode = TradingMode::Off,
                _ => warn!("Unknown ORDER_TRADING_MODE: '{}'. Use auto/manual/off", v),
            }
        }

        settings.validate()?;
        Ok(settings)
    }

    /// Валидация настроек
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.leverage == 0 || self.leverage > 125 {
            anyhow::bail!(
                "Invalid leverage: {}. Must be 1-125 for Binance Futures",
                self.leverage
            );
        }
        if self.max_orders_at_a_time == 0 {
            anyhow::bail!("max_orders_at_a_time must be > 0");
        }
        if self.trade_size_value <= 0.0 {
            anyhow::bail!("trade_size_value must be > 0.0");
        }
        if self.trade_size_type == TradeSizeType::PercentDepo && self.trade_size_value > 100.0 {
            anyhow::bail!(
                "trade_size_value as percent_depo cannot exceed 100.0%, got {}",
                self.trade_size_value
            );
        }
        Ok(())
    }

    /// Сохранить в TOML-файл
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_order_settings() {
        let settings = OrderSettings::default();
        assert_eq!(settings.leverage, 10);
        assert_eq!(settings.max_orders_at_a_time, 10);
        assert_eq!(settings.trade_size_type, TradeSizeType::FixedUsdt);
        assert!((settings.trade_size_value - 100.0).abs() < 1e-8);
        assert_eq!(settings.strategy_type, StrategyType::MlSuperEntry);
        assert_eq!(settings.order_type, OrderType::FuturesOco);
        assert_eq!(settings.trading_mode, TradingMode::Manual);
        assert!(!settings.trading_mode.is_auto());
        assert!(settings.trading_mode.is_active());
    }

    #[test]
    fn test_trading_mode() {
        assert!(TradingMode::Auto.is_auto());
        assert!(TradingMode::Auto.is_active());
        assert!(!TradingMode::Auto.is_off());

        assert!(!TradingMode::Manual.is_auto());
        assert!(TradingMode::Manual.is_active());
        assert!(!TradingMode::Manual.is_off());

        assert!(!TradingMode::Off.is_auto());
        assert!(!TradingMode::Off.is_active());
        assert!(TradingMode::Off.is_off());
    }

    #[test]
    fn test_validate_leverage() {
        let mut settings = OrderSettings::default();
        settings.leverage = 0;
        assert!(settings.validate().is_err());

        settings.leverage = 126;
        assert!(settings.validate().is_err());

        settings.leverage = 50;
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn test_validate_percent_depo() {
        let mut settings = OrderSettings::default();
        settings.trade_size_type = TradeSizeType::PercentDepo;
        settings.trade_size_value = 150.0;
        assert!(settings.validate().is_err());

        settings.trade_size_value = 10.0;
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn test_serde_roundtrip() {
        let settings = OrderSettings::default();
        let toml_str = toml::to_string_pretty(&settings).unwrap();
        let deserialized: OrderSettings = toml::from_str(&toml_str).unwrap();
        assert_eq!(deserialized.leverage, settings.leverage);
        assert_eq!(deserialized.trade_size_type, settings.trade_size_type);
    }
}
