// order_manager/src/config.rs
//
// Конфигурация Order Manager.
// Загружается из config/order_manager.toml + env-переменных.
//
// Содержит:
//   - Параметры signal_scanner (score range, price drift, lookback)
//   - Параметры order_executor (пропорции таймфреймов, leverage)
//   - Параметры position_tracker (scan interval, force-close)
//   - Kafka/Database endpoints

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::types::TimeframeAllocation;

/// Полная конфигурация Order Manager (единый файл config/order_manager.toml)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderManagerConfig {
    // ─── Trading Settings (WebUI Trading Options) ──────────
    /// Кредитное плечо (1-125)
    pub leverage: u16,
    /// Максимальное количество одновременно открытых ордеров
    #[serde(alias = "max_open_positions")]
    pub max_orders_at_a_time: u16,
    /// Тип размера позиции: "fixed_usdt" или "percent_depo"
    #[serde(default = "default_trade_size_type")]
    pub trade_size_type: String,
    /// Значение размера позиции (USDT или %)
    #[serde(alias = "trade_size_usdt")]
    pub trade_size_value: f64,
    /// Тип стратегии: "ml_super_entry" или "level_strategy"
    #[serde(default = "default_strategy_type")]
    pub strategy_type: String,
    /// Тип ордера: "futures_oco"
    #[serde(default = "default_order_type")]
    pub order_type: String,
    /// Режим торговли: "auto" / "manual" / "off"
    #[serde(default = "default_trading_mode")]
    pub trading_mode: String,

    // ─── Signal Scanner (per-timeframe score ranges) ────────────────────────────────────
    /// Минимальный combined_score для 1m сигнала (inclusive)
    pub signal_score_min_1m: f32,
    /// Максимальный combined_score для 1m сигнала (inclusive)
    pub signal_score_max_1m: f32,
    /// Минимальный combined_score для 5m сигнала (inclusive)
    pub signal_score_min_5m: f32,
    /// Максимальный combined_score для 5m сигнала (inclusive)
    pub signal_score_max_5m: f32,
    /// Минимальный combined_score для 15m сигнала (inclusive)
    pub signal_score_min_15m: f32,
    /// Максимальный combined_score для 15m сигнала (inclusive)
    pub signal_score_max_15m: f32,
    /// Минимальный combined_score для 1h сигнала (inclusive)
    pub signal_score_min_1h: f32,
    /// Максимальный combined_score для 1h сигнала (inclusive)
    pub signal_score_max_1h: f32,
    /// Минимальный combined_score для 4h сигнала (inclusive)
    pub signal_score_min_4h: f32,
    /// Максимальный combined_score для 4h сигнала (inclusive)
    pub signal_score_max_4h: f32,
    /// Минимальный combined_score для 1d сигнала (inclusive)
    pub signal_score_min_1d: f32,
    /// Максимальный combined_score для 1d сигнала (inclusive)
    pub signal_score_max_1d: f32,
    /// Максимальный дрифт цены (%), чтобы сигнал считался актуальным
    pub max_price_drift_pct: f64,
    /// Cooldown (часы) — запрещает повторное открытие ордера на ту же пару в течение этого времени.
    /// Предотвращает ситуации типа BANANAS31USDT × 7 подряд за одну сессию.
    #[serde(default = "default_symbol_cooldown_hours")]
    pub symbol_cooldown_hours: f64,
    /// Интервал сканирования сигналов (секунды)
    pub scan_interval_secs: u64,

    // ─── Order Executor ────────────────────────────────────
    /// Максимальное количество баров удержания позиции
    pub max_hold_bars: i16,

    /// Пропорция слотов для 1m (процент, напр. 0 = 0%)
    pub tf_1m_pct: u16,
    /// Пропорция слотов для 5m (процент, напр. 0 = 0%)
    pub tf_5m_pct: u16,
    /// Пропорция слотов для 15m (процент, напр. 10 = 10%)
    pub tf_15m_pct: u16,
    /// Пропорция слотов для 1h (процент, напр. 70 = 70%)
    pub tf_1h_pct: u16,
    /// Пропорция слотов для 4h (процент, напр. 20 = 20%)
    pub tf_4h_pct: u16,
    /// Пропорция слотов для 1d (процент, напр. 0 = 0%)
    pub tf_1d_pct: u16,

    // ─── Position Tracker ──────────────────────────────────
    /// Интервал обновления позиций / PnL (секунды)
    pub tracker_interval_secs: u64,

    // ─── Infrastructure ────────────────────────────────────
    /// Kafka brokers
    pub kafka_brokers: String,
    /// Kafka topic для событий позиций (→ WebUI)
    pub topic_positions: String,
    /// Kafka topic для команд ордеров (← risk_manager)
    pub topic_orders_cmd: String,
    /// Kafka group ID
    pub kafka_group_id: String,
    /// Database URL
    pub database_url: String,
    /// Использовать тестнет Binance
    pub binance_testnet: bool,
}

fn default_trade_size_type() -> String { "fixed_usdt".to_string() }
fn default_strategy_type() -> String { "ml_super_entry".to_string() }
fn default_order_type() -> String { "futures_oco".to_string() }
fn default_trading_mode() -> String { "off".to_string() }
fn default_symbol_cooldown_hours() -> f64 { 12.0 }

impl Default for OrderManagerConfig {
    fn default() -> Self {
        Self {
            // Trading Settings
            leverage: 10,
            max_orders_at_a_time: 10,
            trade_size_type: "fixed_usdt".to_string(),
            trade_size_value: 100.0,
            strategy_type: "ml_super_entry".to_string(),
            order_type: "futures_oco".to_string(),
            trading_mode: "off".to_string(),

            // Scanner (per-timeframe score ranges)
            signal_score_min_1m: 0.70,
            signal_score_max_1m: 0.80,
            signal_score_min_5m: 0.70,
            signal_score_max_5m: 0.80,
            signal_score_min_15m: 0.70,
            signal_score_max_15m: 0.80,
            signal_score_min_1h: 0.70,
            signal_score_max_1h: 0.80,
            signal_score_min_4h: 0.70,
            signal_score_max_4h: 0.80,
            signal_score_min_1d: 0.70,
            signal_score_max_1d: 0.80,
            max_price_drift_pct: 0.2,
            symbol_cooldown_hours: 12.0,
            scan_interval_secs: 30,

            // Executor
            max_hold_bars: 25,
            tf_1m_pct: 0,
            tf_5m_pct: 0,
            tf_15m_pct: 10,
            tf_1h_pct: 70,
            tf_4h_pct: 20,
            tf_1d_pct: 0,

            // Tracker
            tracker_interval_secs: 10,

            // Infra
            kafka_brokers: std::env::var("KAFKA_BROKERS")
                .unwrap_or_else(|_| "localhost:19092".to_string()),
            topic_positions: "positions.updates".to_string(),
            topic_orders_cmd: "orders.cmd".to_string(),
            kafka_group_id: "order-manager".to_string(),
            database_url: std::env::var("DATABASE_URL").unwrap_or_else(|_|
                "postgres://postgres:postgres@127.0.0.1:5433/timescaledb_binance".to_string()
            ),
            binance_testnet: false,
        }
    }
}

impl OrderManagerConfig {
    /// Is trading mode auto?
    pub fn is_auto(&self) -> bool {
        self.trading_mode == "auto"
    }
}

impl OrderManagerConfig {
    /// Загрузить из TOML-файла с fallback на дефолт
    pub fn load() -> anyhow::Result<Self> {
        let path = "config/order_manager.toml";
        if std::path::Path::new(path).exists() {
            let content = std::fs::read_to_string(path)?;
            let config: Self = toml::from_str(&content)?;
            info!("OrderManagerConfig loaded from {}", path);
            config.validate()?;
            Ok(config)
        } else {
            warn!("Order manager config not found at {}, using defaults", path);
            Ok(Self::default())
        }
    }

    /// Загрузить с override из env переменных
    pub fn load_with_env() -> anyhow::Result<Self> {
        let mut cfg = Self::load()?;

        // Per-timeframe score overrides (optional)
        if let Ok(v) = std::env::var("OM_SCORE_MIN_1H") {
            if let Ok(n) = v.parse() { cfg.signal_score_min_1h = n; }
        }
        if let Ok(v) = std::env::var("OM_SCORE_MAX_1H") {
            if let Ok(n) = v.parse() { cfg.signal_score_max_1h = n; }
        }
        if let Ok(v) = std::env::var("OM_SCORE_MIN_4H") {
            if let Ok(n) = v.parse() { cfg.signal_score_min_4h = n; }
        }
        if let Ok(v) = std::env::var("OM_SCORE_MAX_4H") {
            if let Ok(n) = v.parse() { cfg.signal_score_max_4h = n; }
        }
        if let Ok(v) = std::env::var("OM_SCORE_MIN_15M") {
            if let Ok(n) = v.parse() { cfg.signal_score_min_15m = n; }
        }
        if let Ok(v) = std::env::var("OM_SCORE_MAX_15M") {
            if let Ok(n) = v.parse() { cfg.signal_score_max_15m = n; }
        }
        if let Ok(v) = std::env::var("OM_PRICE_DRIFT_PCT") {
            if let Ok(n) = v.parse() { cfg.max_price_drift_pct = n; }
        }
        if let Ok(v) = std::env::var("OM_SYMBOL_COOLDOWN_HOURS") {
            if let Ok(n) = v.parse() { cfg.symbol_cooldown_hours = n; }
        }
        if let Ok(v) = std::env::var("OM_SCAN_INTERVAL") {
            if let Ok(n) = v.parse() { cfg.scan_interval_secs = n; }
        }
        if let Ok(v) = std::env::var("OM_MAX_POSITIONS") {
            if let Ok(n) = v.parse() { cfg.max_orders_at_a_time = n; }
        }
        if let Ok(v) = std::env::var("OM_LEVERAGE") {
            if let Ok(n) = v.parse() { cfg.leverage = n; }
        }
        if let Ok(v) = std::env::var("OM_TRADE_SIZE_USDT") {
            if let Ok(n) = v.parse() { cfg.trade_size_value = n; }
        }
        if let Ok(v) = std::env::var("OM_MAX_HOLD_BARS") {
            if let Ok(n) = v.parse() { cfg.max_hold_bars = n; }
        }
        if let Ok(v) = std::env::var("KAFKA_BROKERS") {
            cfg.kafka_brokers = v;
        }
        if let Ok(v) = std::env::var("DATABASE_URL") {
            cfg.database_url = v;
        }
        if let Ok(v) = std::env::var("BINANCE_TESTNET") {
            cfg.binance_testnet = v == "true" || v == "1";
        }

        cfg.validate()?;
        Ok(cfg)
    }

    /// Валидация конфигурации
    pub fn validate(&self) -> anyhow::Result<()> {
        // Validate per-timeframe score ranges
        if self.signal_score_min_1m >= self.signal_score_max_1m {
            anyhow::bail!(
                "signal_score_min_1m ({}) must be < signal_score_max_1m ({})",
                self.signal_score_min_1m,
                self.signal_score_max_1m
            );
        }
        if self.signal_score_min_5m >= self.signal_score_max_5m {
            anyhow::bail!(
                "signal_score_min_5m ({}) must be < signal_score_max_5m ({})",
                self.signal_score_min_5m,
                self.signal_score_max_5m
            );
        }
        if self.signal_score_min_15m >= self.signal_score_max_15m {
            anyhow::bail!(
                "signal_score_min_15m ({}) must be < signal_score_max_15m ({})",
                self.signal_score_min_15m,
                self.signal_score_max_15m
            );
        }
        if self.signal_score_min_1h >= self.signal_score_max_1h {
            anyhow::bail!(
                "signal_score_min_1h ({}) must be < signal_score_max_1h ({})",
                self.signal_score_min_1h,
                self.signal_score_max_1h
            );
        }
        if self.signal_score_min_4h >= self.signal_score_max_4h {
            anyhow::bail!(
                "signal_score_min_4h ({}) must be < signal_score_max_4h ({})",
                self.signal_score_min_4h,
                self.signal_score_max_4h
            );
        }
        if self.signal_score_min_1d >= self.signal_score_max_1d {
            anyhow::bail!(
                "signal_score_min_1d ({}) must be < signal_score_max_1d ({})",
                self.signal_score_min_1d,
                self.signal_score_max_1d
            );
        }
        if self.max_orders_at_a_time == 0 {
            anyhow::bail!("max_orders_at_a_time must be > 0");
        }
        if self.leverage == 0 || self.leverage > 125 {
            anyhow::bail!("leverage must be 1..125, got {}", self.leverage);
        }
        if self.trade_size_value <= 0.0 {
            anyhow::bail!("trade_size_value must be > 0");
        }
        if self.trade_size_type != "fixed_usdt" && self.trade_size_type != "percent_depo" {
            anyhow::bail!(
                "trade_size_type must be 'fixed_usdt' or 'percent_depo', got '{}'",
                self.trade_size_type
            );
        }
        if self.trade_size_type == "percent_depo" && self.trade_size_value > 100.0 {
            anyhow::bail!(
                "trade_size_value in percent_depo mode must be <= 100, got {}",
                self.trade_size_value
            );
        }
        if self.max_hold_bars <= 0 {
            anyhow::bail!("max_hold_bars must be > 0");
        }
        let total_pct = self.tf_1m_pct + self.tf_5m_pct + self.tf_15m_pct + self.tf_1h_pct + self.tf_4h_pct + self.tf_1d_pct;
        if total_pct != 100 {
            anyhow::bail!(
                "Timeframe percentages must sum to 100, got {} (1m={}%, 5m={}%, 15m={}%, 1h={}%, 4h={}%, 1d={}%)",
                total_pct,
                self.tf_1m_pct,
                self.tf_5m_pct,
                self.tf_15m_pct,
                self.tf_1h_pct,
                self.tf_4h_pct,
                self.tf_1d_pct
            );
        }
        Ok(())
    }

    /// Вычислить распределение слотов по таймфреймам
    pub fn timeframe_allocation(&self) -> TimeframeAllocation {
        let total = self.max_orders_at_a_time;
        // Рассчитываем слоты: каждый таймфрейм получает floor, 1h — остаток
        let slots_1m = (total as f64 * self.tf_1m_pct as f64 / 100.0).floor() as u16;
        let slots_5m = (total as f64 * self.tf_5m_pct as f64 / 100.0).floor() as u16;
        let slots_15m = (total as f64 * self.tf_15m_pct as f64 / 100.0).floor() as u16;
        let slots_4h = (total as f64 * self.tf_4h_pct as f64 / 100.0).floor() as u16;
        let slots_1d = (total as f64 * self.tf_1d_pct as f64 / 100.0).floor() as u16;
        let slots_1h = total
            .saturating_sub(slots_1m)
            .saturating_sub(slots_5m)
            .saturating_sub(slots_15m)
            .saturating_sub(slots_4h)
            .saturating_sub(slots_1d);

        TimeframeAllocation {
            slots: vec![
                (1, slots_1m),
                (5, slots_5m),
                (15, slots_15m),
                (60, slots_1h),
                (240, slots_4h),
                (1440, slots_1d),
            ],
            total,
        }
    }

    /// Lookback-окно для таймфрейма (в минутах).
    /// Для 15m → 15 мин, для 1h → 60 мин, для 4h → 240 мин
    pub fn lookback_minutes_for_tf(&self, tf_minutes: i16) -> i64 {
        tf_minutes as i64
    }
}

/// Сохранить дефолтный конфиг в файл (для инициализации)
pub fn write_default_config() -> anyhow::Result<()> {
    let cfg = OrderManagerConfig::default();
    let content = toml::to_string_pretty(&cfg)?;
    std::fs::write("config/order_manager.toml", content)?;
    info!("Default order_manager config written to config/order_manager.toml");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let cfg = OrderManagerConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_allocation_10_slots() {
        let cfg = OrderManagerConfig::default();
        let alloc = cfg.timeframe_allocation();
        assert_eq!(alloc.total, 10);
        assert_eq!(alloc.slots_for_tf(1), 0);   // 1m: 0%
        assert_eq!(alloc.slots_for_tf(5), 0);   // 5m: 0%
        assert_eq!(alloc.slots_for_tf(15), 1);  // 15m: 10%
        assert_eq!(alloc.slots_for_tf(60), 7);  // 1h: 70%
        assert_eq!(alloc.slots_for_tf(240), 2); // 4h: 20%
        assert_eq!(alloc.slots_for_tf(1440), 0); // 1d: 0%
    }

    #[test]
    fn test_invalid_score_range_1h() {
        let mut cfg = OrderManagerConfig::default();
        cfg.signal_score_min_1h = 0.9;
        cfg.signal_score_max_1h = 0.8;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_pct_sum() {
        let mut cfg = OrderManagerConfig::default();
        cfg.tf_1h_pct = 50;
        cfg.tf_4h_pct = 20;
        cfg.tf_15m_pct = 10;
        cfg.tf_1m_pct = 0;
        cfg.tf_5m_pct = 0;
        cfg.tf_1d_pct = 0;
        assert!(cfg.validate().is_err()); // sum = 80 ≠ 100
    }

    #[test]
    fn test_serde_roundtrip() {
        let cfg = OrderManagerConfig::default();
        let toml_str = toml::to_string_pretty(&cfg).unwrap();
        let deserialized: OrderManagerConfig = toml::from_str(&toml_str).unwrap();
        assert!((deserialized.signal_score_min_1h - cfg.signal_score_min_1h).abs() < 1e-6);
        assert_eq!(deserialized.max_orders_at_a_time, cfg.max_orders_at_a_time);
    }
}
