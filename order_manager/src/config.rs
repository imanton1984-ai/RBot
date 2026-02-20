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

/// Полная конфигурация Order Manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderManagerConfig {
    // ─── Signal Scanner ────────────────────────────────────
    /// Минимальный combined_score для сигнала (inclusive)
    pub signal_score_min: f32,
    /// Максимальный combined_score для сигнала (inclusive)
    pub signal_score_max: f32,
    /// Максимальный дрифт цены (%), чтобы сигнал считался актуальным
    pub max_price_drift_pct: f64,
    /// Интервал сканирования сигналов (секунды)
    pub scan_interval_secs: u64,

    // ─── Order Executor ────────────────────────────────────
    /// Максимальное количество одновременно открытых ордеров
    pub max_open_positions: u16,
    /// Кредитное плечо
    pub leverage: u16,
    /// Размер позиции в USDT
    pub trade_size_usdt: f64,
    /// Максимальное количество баров удержания позиции
    pub max_hold_bars: i16,

    /// Пропорция слотов для 1h (процент, напр. 70 = 70%)
    pub tf_1h_pct: u16,
    /// Пропорция слотов для 4h (процент, напр. 20 = 20%)
    pub tf_4h_pct: u16,
    /// Пропорция слотов для 15m (процент, напр. 10 = 10%)
    pub tf_15m_pct: u16,

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

impl Default for OrderManagerConfig {
    fn default() -> Self {
        Self {
            // Scanner
            signal_score_min: 0.70,
            signal_score_max: 0.80,
            max_price_drift_pct: 0.2,
            scan_interval_secs: 30,

            // Executor
            max_open_positions: 10,
            leverage: 10,
            trade_size_usdt: 100.0,
            max_hold_bars: 25,
            tf_1h_pct: 70,
            tf_4h_pct: 20,
            tf_15m_pct: 10,

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

        if let Ok(v) = std::env::var("OM_SCORE_MIN") {
            if let Ok(n) = v.parse() { cfg.signal_score_min = n; }
        }
        if let Ok(v) = std::env::var("OM_SCORE_MAX") {
            if let Ok(n) = v.parse() { cfg.signal_score_max = n; }
        }
        if let Ok(v) = std::env::var("OM_PRICE_DRIFT_PCT") {
            if let Ok(n) = v.parse() { cfg.max_price_drift_pct = n; }
        }
        if let Ok(v) = std::env::var("OM_SCAN_INTERVAL") {
            if let Ok(n) = v.parse() { cfg.scan_interval_secs = n; }
        }
        if let Ok(v) = std::env::var("OM_MAX_POSITIONS") {
            if let Ok(n) = v.parse() { cfg.max_open_positions = n; }
        }
        if let Ok(v) = std::env::var("OM_LEVERAGE") {
            if let Ok(n) = v.parse() { cfg.leverage = n; }
        }
        if let Ok(v) = std::env::var("OM_TRADE_SIZE_USDT") {
            if let Ok(n) = v.parse() { cfg.trade_size_usdt = n; }
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
        if self.signal_score_min >= self.signal_score_max {
            anyhow::bail!(
                "signal_score_min ({}) must be < signal_score_max ({})",
                self.signal_score_min,
                self.signal_score_max
            );
        }
        if self.max_open_positions == 0 {
            anyhow::bail!("max_open_positions must be > 0");
        }
        if self.leverage == 0 || self.leverage > 125 {
            anyhow::bail!("leverage must be 1..125, got {}", self.leverage);
        }
        if self.trade_size_usdt <= 0.0 {
            anyhow::bail!("trade_size_usdt must be > 0");
        }
        if self.max_hold_bars <= 0 {
            anyhow::bail!("max_hold_bars must be > 0");
        }
        let total_pct = self.tf_1h_pct + self.tf_4h_pct + self.tf_15m_pct;
        if total_pct != 100 {
            anyhow::bail!(
                "Timeframe percentages must sum to 100, got {} (1h={}%, 4h={}%, 15m={}%)",
                total_pct,
                self.tf_1h_pct,
                self.tf_4h_pct,
                self.tf_15m_pct
            );
        }
        Ok(())
    }

    /// Вычислить распределение слотов по таймфреймам
    pub fn timeframe_allocation(&self) -> TimeframeAllocation {
        let total = self.max_open_positions;
        // Рассчитываем слоты: 15m получает floor, 4h получает floor, 1h — остаток
        let slots_15m = (total as f64 * self.tf_15m_pct as f64 / 100.0).floor() as u16;
        let slots_4h = (total as f64 * self.tf_4h_pct as f64 / 100.0).floor() as u16;
        let slots_1h = total.saturating_sub(slots_15m).saturating_sub(slots_4h);

        TimeframeAllocation {
            slots: vec![
                (60, slots_1h),
                (240, slots_4h),
                (15, slots_15m),
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
        assert_eq!(alloc.slots_for_tf(60), 7);
        assert_eq!(alloc.slots_for_tf(240), 2);
        assert_eq!(alloc.slots_for_tf(15), 1);
    }

    #[test]
    fn test_invalid_score_range() {
        let mut cfg = OrderManagerConfig::default();
        cfg.signal_score_min = 0.9;
        cfg.signal_score_max = 0.8;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_pct_sum() {
        let mut cfg = OrderManagerConfig::default();
        cfg.tf_1h_pct = 50;
        cfg.tf_4h_pct = 20;
        cfg.tf_15m_pct = 10;
        assert!(cfg.validate().is_err()); // sum = 80 ≠ 100
    }

    #[test]
    fn test_serde_roundtrip() {
        let cfg = OrderManagerConfig::default();
        let toml_str = toml::to_string_pretty(&cfg).unwrap();
        let deserialized: OrderManagerConfig = toml::from_str(&toml_str).unwrap();
        assert!((deserialized.signal_score_min - cfg.signal_score_min).abs() < 1e-6);
        assert_eq!(deserialized.max_open_positions, cfg.max_open_positions);
    }
}
