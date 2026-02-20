// risk_manager/src/types.rs
//
// Общие типы для Risk Manager: алерты, позиции, конфигурация мониторов.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

// ═══════════════════════════════════════════════════════════
// АЛЕРТЫ
// ═══════════════════════════════════════════════════════════

/// Уровень серьёзности алерта
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertSeverity {
    /// Информационный (логируется)
    Info,
    /// Предупреждение (логируется + уведомление)
    Warning,
    /// Критический (логируется + уведомление + возможное действие)
    Critical,
}

/// Источник алерта
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertSource {
    /// Volume + Price Monitor
    VolumePrice,
    /// Trend Change Monitor
    TrendChange,
    /// Position Monitor
    Position,
    /// Position Closer
    PositionCloser,
}

/// Алерт от Risk Manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAlert {
    /// Время алерта
    pub timestamp: DateTime<Utc>,
    /// Источник
    pub source: AlertSource,
    /// Серьёзность
    pub severity: AlertSeverity,
    /// Символ (напр. "BTCUSDT")
    pub symbol: String,
    /// Таймфрейм (минуты)
    pub tf_minutes: i16,
    /// Сообщение
    pub message: String,
    /// Текущая цена (если доступна)
    pub price: Option<f64>,
    /// Изменение в процентах (если доступно)
    pub change_pct: Option<f64>,
}

impl RiskAlert {
    pub fn new(
        source: AlertSource,
        severity: AlertSeverity,
        symbol: impl Into<String>,
        tf_minutes: i16,
        message: impl Into<String>,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            source,
            severity,
            symbol: symbol.into(),
            tf_minutes,
            message: message.into(),
            price: None,
            change_pct: None,
        }
    }

    pub fn with_price(mut self, price: f64) -> Self {
        self.price = Some(price);
        self
    }

    pub fn with_change_pct(mut self, pct: f64) -> Self {
        self.change_pct = Some(pct);
        self
    }
}

impl std::fmt::Display for RiskAlert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{:?}][{:?}] {} ({}m): {}",
            self.severity, self.source, self.symbol, self.tf_minutes, self.message
        )?;
        if let Some(price) = self.price {
            write!(f, " | price={:.4}", price)?;
        }
        if let Some(pct) = self.change_pct {
            write!(f, " | change={:.2}%", pct)?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════
// ПОЗИЦИИ
// ═══════════════════════════════════════════════════════════

/// Направление позиции
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionSide {
    Long,
    Short,
}

impl std::fmt::Display for PositionSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Long => write!(f, "LONG"),
            Self::Short => write!(f, "SHORT"),
        }
    }
}

/// Открытая позиция (для мониторинга)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenPosition {
    /// Символ
    pub symbol: String,
    /// Направление
    pub side: PositionSide,
    /// Цена входа
    pub entry_price: f64,
    /// Размер позиции (в базовом активе)
    pub quantity: f64,
    /// Нотионал (в USDT)
    pub notional_usdt: f64,
    /// Плечо
    pub leverage: u16,
    /// Время открытия
    pub open_time: DateTime<Utc>,
    /// Нереализованный PnL
    pub unrealized_pnl: f64,
}

// ═══════════════════════════════════════════════════════════
// ДАННЫЕ ИНДИКАТОРОВ (из market.indicators_wide)
// ═══════════════════════════════════════════════════════════

/// Строка из market.indicators_wide (только нужные поля для risk manager)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorRow {
    pub time_ms: i64,
    pub symbol: String,
    pub tf_minutes: i16,
    pub close: Option<f64>,
    pub volume_spike: Option<f32>,
    pub trend: Option<i16>,
    pub trend_short: Option<i16>,
    pub rsi: Option<f32>,
    pub atr: Option<f32>,
}

/// Событие индикатора из Kafka (indicators.close)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorEvent {
    pub symbol: String,
    pub tf_minutes: i16,
    pub time_ms: i64,
    pub volume_spike: Option<f32>,
    pub trend: Option<i16>,
    pub trend_short: Option<i16>,
    pub close: Option<f64>,
    pub rsi: Option<f32>,
    pub atr: Option<f32>,
}

// ═══════════════════════════════════════════════════════════
// КОНФИГУРАЦИЯ RISK MANAGER
// ═══════════════════════════════════════════════════════════

/// Конфигурация Risk Manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskManagerConfig {
    /// Включён ли risk manager
    pub enabled: bool,

    /// Пороги для BTC (резкое движение, %)
    pub btc_alert_threshold_pct: f64,

    /// Пороги для альткоинов (резкое движение, %)
    pub alt_alert_threshold_pct: f64,

    /// Порог volume_spike для алерта
    pub volume_spike_threshold: f32,

    /// Таймфреймы для мониторинга
    pub monitor_timeframes: Vec<i16>,

    /// Top N альткоинов для мониторинга
    pub top_alts_count: usize,

    /// Список символов Top альткоинов (если пустой — берётся из universe)
    pub top_alt_symbols: Vec<String>,

    /// Включён ли Position Closer
    pub position_closer_enabled: bool,

    /// Kafka brokers
    pub kafka_brokers: String,

    /// Kafka group ID для risk manager
    pub kafka_group_id: String,

    /// Топик индикаторов
    pub topic_indicators: String,

    /// Топик позиций
    pub topic_positions: String,

    /// Топик алертов (куда публикуем)
    pub topic_alerts: String,

    /// Топик команд ордеров (для position closer)
    pub topic_orders_cmd: String,

    /// Database URL
    pub database_url: String,
}

impl Default for RiskManagerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            btc_alert_threshold_pct: 0.5,
            alt_alert_threshold_pct: 1.5,
            volume_spike_threshold: 2.0,
            monitor_timeframes: vec![1, 5, 15],
            top_alts_count: 20,
            top_alt_symbols: vec![
                "ETHUSDT".into(), "BNBUSDT".into(), "SOLUSDT".into(),
                "XRPUSDT".into(), "DOGEUSDT".into(), "ADAUSDT".into(),
                "AVAXUSDT".into(), "DOTUSDT".into(), "LINKUSDT".into(),
                "MATICUSDT".into(), "SHIBUSDT".into(), "LTCUSDT".into(),
                "TRXUSDT".into(), "ATOMUSDT".into(), "UNIUSDT".into(),
                "NEARUSDT".into(), "APTUSDT".into(), "ARBUSDT".into(),
                "OPUSDT".into(), "FILUSDT".into(),
            ],
            position_closer_enabled: false,
            kafka_brokers: std::env::var("KAFKA_BROKERS")
                .unwrap_or_else(|_| "localhost:19092".to_string()),
            kafka_group_id: "risk-manager".to_string(),
            topic_indicators: "indicators.close".to_string(),
            topic_positions: "positions.events".to_string(),
            topic_alerts: "risk.alerts".to_string(),
            topic_orders_cmd: "orders.cmd".to_string(),
            database_url: std::env::var("DATABASE_URL").unwrap_or_else(|_|
                "postgres://postgres:postgres@127.0.0.1:5433/timescaledb_binance".to_string()
            ),
        }
    }
}

impl RiskManagerConfig {
    /// Загрузить из JSON-файла с fallback на дефолт
    pub fn load() -> anyhow::Result<Self> {
        let path = "config/risk_manager.json";
        if std::path::Path::new(path).exists() {
            let content = std::fs::read_to_string(path)?;
            let config: Self = serde_json::from_str(&content)?;
            Ok(config)
        } else {
            tracing::warn!("Risk manager config not found at {}, using defaults", path);
            Ok(Self::default())
        }
    }

    /// Загрузить с override из env
    pub fn load_with_env() -> anyhow::Result<Self> {
        let mut config = Self::load()?;

        if let Ok(v) = std::env::var("RISK_BTC_THRESHOLD") {
            if let Ok(n) = v.parse() { config.btc_alert_threshold_pct = n; }
        }
        if let Ok(v) = std::env::var("RISK_ALT_THRESHOLD") {
            if let Ok(n) = v.parse() { config.alt_alert_threshold_pct = n; }
        }
        if let Ok(v) = std::env::var("RISK_POSITION_CLOSER") {
            config.position_closer_enabled = v == "true" || v == "1";
        }
        if let Ok(v) = std::env::var("KAFKA_BROKERS") {
            config.kafka_brokers = v;
        }
        if let Ok(v) = std::env::var("DATABASE_URL") {
            config.database_url = v;
        }

        Ok(config)
    }

    /// Проверить, является ли символ BTC
    pub fn is_btc(&self, symbol: &str) -> bool {
        symbol == "BTCUSDT" || symbol == "BTCBUSD"
    }

    /// Проверить, является ли символ Top альткоином
    pub fn is_top_alt(&self, symbol: &str) -> bool {
        self.top_alt_symbols.iter().any(|s| s == symbol)
    }

    /// Получить порог алерта для символа
    pub fn alert_threshold_for(&self, symbol: &str) -> f64 {
        if self.is_btc(symbol) {
            self.btc_alert_threshold_pct
        } else {
            self.alt_alert_threshold_pct
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_alert_display() {
        let alert = RiskAlert::new(
            AlertSource::VolumePrice,
            AlertSeverity::Warning,
            "BTCUSDT",
            1,
            "Sharp price movement detected",
        )
        .with_price(50000.0)
        .with_change_pct(-0.7);

        let s = format!("{}", alert);
        assert!(s.contains("BTCUSDT"));
        assert!(s.contains("Sharp price movement"));
        assert!(s.contains("-0.70%"));
    }

    #[test]
    fn test_config_thresholds() {
        let config = RiskManagerConfig::default();
        assert!((config.alert_threshold_for("BTCUSDT") - 0.5).abs() < 1e-8);
        assert!((config.alert_threshold_for("ETHUSDT") - 1.5).abs() < 1e-8);
        assert!((config.alert_threshold_for("RANDOMUSDT") - 1.5).abs() < 1e-8);
    }

    #[test]
    fn test_config_is_top_alt() {
        let config = RiskManagerConfig::default();
        assert!(config.is_top_alt("ETHUSDT"));
        assert!(config.is_top_alt("SOLUSDT"));
        assert!(!config.is_top_alt("RANDOMUSDT"));
        assert!(!config.is_btc("ETHUSDT"));
        assert!(config.is_btc("BTCUSDT"));
    }
}
