// order_manager/src/types.rs
//
// Общие типы для Order Manager: сигналы, позиции, статусы, события.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════
// DIRECTION & STATUS ENUMS
// ═══════════════════════════════════════════════════════════

/// Направление позиции / ордера
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Long,
    Short,
}

impl Side {
    /// DB representation: 1 = LONG, -1 = SHORT
    pub fn as_db_i16(self) -> i16 {
        match self {
            Side::Long => 1,
            Side::Short => -1,
        }
    }

    pub fn from_db_i16(val: i16) -> Option<Self> {
        match val {
            1 => Some(Side::Long),
            -1 => Some(Side::Short),
            _ => None,
        }
    }

    /// Binance API order side to OPEN this position
    pub fn entry_side_str(self) -> &'static str {
        match self {
            Side::Long => "BUY",
            Side::Short => "SELL",
        }
    }

    /// Binance API order side to CLOSE this position
    pub fn close_side_str(self) -> &'static str {
        match self {
            Side::Long => "SELL",
            Side::Short => "BUY",
        }
    }

    /// For logging
    pub fn as_str(self) -> &'static str {
        match self {
            Side::Long => "LONG",
            Side::Short => "SHORT",
        }
    }
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Статус позиции в trade.positions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionStatus {
    /// Позиция открыта, ордера на бирже активны
    Open = 1,
    /// Позиция закрыта (все ордера исполнены/отменены)
    Closed = 2,
    /// Ошибка при открытии / частично открыта
    Error = 3,
}

impl PositionStatus {
    pub fn as_i16(self) -> i16 {
        self as i16
    }

    pub fn from_i16(val: i16) -> Option<Self> {
        match val {
            1 => Some(Self::Open),
            2 => Some(Self::Closed),
            3 => Some(Self::Error),
            _ => None,
        }
    }
}

/// Причина закрытия позиции
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloseReason {
    /// Take-profit сработал
    TpHit,
    /// Stop-loss сработал
    SlHit,
    /// Лимит по свечам (max_hold_bars) исчерпан
    MaxBars,
    /// Risk Manager принудительно закрыл
    RiskManager,
    /// Ручное закрытие оператором
    Manual,
    /// Emergency stop - принудительное закрытие всех позиций
    EmerClosed,
}

impl CloseReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TpHit => "tp_hit",
            Self::SlHit => "sl_hit",
            Self::MaxBars => "max_bars",
            Self::RiskManager => "risk_manager",
            Self::Manual => "manual",
            Self::EmerClosed => "emer_closed",
        }
    }
}

impl std::fmt::Display for CloseReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ═══════════════════════════════════════════════════════════
// QUALIFIED SIGNAL (output from signal_scanner)
// ═══════════════════════════════════════════════════════════

/// Квалифицированный сигнал, прошедший все фильтры scanner'а.
/// Передаётся из signal_scanner → order_executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualifiedSignal {
    /// Время свечи сигнала
    pub signal_time: DateTime<Utc>,
    pub signal_time_ms: i64,
    /// Символ
    pub symbol: String,
    pub symbol_id: i64,
    /// Таймфрейм (минуты): 15 | 60 | 240
    pub tf_minutes: i16,
    /// Направление: Long / Short
    pub side: Side,
    /// Цена входа из сигнала
    pub entry_price: f64,
    /// Стоп-лосс из сигнала
    pub sl_price: f64,
    /// Тейк-профит из сигнала
    pub tp_price: f64,
    /// P(super) вероятность от модели
    pub p_super: f32,
    /// Combined score (главный фильтр: 0.70..0.80)
    pub combined_score: f32,
    /// Текущая рыночная цена на момент сканирования
    pub current_price: f64,
    /// Отклонение текущей цены от entry_price (%)
    pub price_drift_pct: f64,
}

// ═══════════════════════════════════════════════════════════
// MANAGED POSITION (runtime state in position_tracker)
// ═══════════════════════════════════════════════════════════

/// Полное состояние управляемой позиции (in-memory + DB)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedPosition {
    /// ID из trade.positions
    pub position_id: i64,
    /// Символ
    pub symbol: String,
    pub symbol_id: i64,
    /// Таймфрейм сигнала
    pub tf_minutes: i16,
    /// Направление
    pub side: Side,
    /// Цена входа
    pub entry_price: f64,
    /// SL / TP цены
    pub sl_price: f64,
    pub tp_price: f64,
    /// Количество (в базовом активе)
    pub qty: f64,
    /// Плечо
    pub leverage: u16,
    /// Сколько баров осталось до принудительного закрытия
    pub candles_left: i16,
    /// Максимальное количество баров
    pub max_hold_bars: i16,
    /// Нереализованный PnL (USDT)
    pub unrealized_pnl: f64,
    /// Нереализованный PnL (%)
    pub unrealized_pnl_pct: f64,
    /// Текущая рыночная цена
    pub current_price: f64,
    /// Время открытия
    pub opened_at: DateTime<Utc>,
    /// ML combined score при входе
    pub combined_score: f32,
    /// P(super) при входе
    pub p_super: f32,
    /// Binance order IDs
    pub entry_order_id: Option<String>,
    pub sl_order_id: Option<String>,
    pub tp_order_id: Option<String>,
}

impl ManagedPosition {
    /// Рассчитать нереализованный PnL для текущей цены
    pub fn calc_unrealized_pnl(&self, current_price: f64) -> (f64, f64) {
        let direction = match self.side {
            Side::Long => 1.0,
            Side::Short => -1.0,
        };
        let pnl = direction * (current_price - self.entry_price) * self.qty;
        let pnl_pct = direction * (current_price - self.entry_price) / self.entry_price * 100.0;
        (pnl, pnl_pct)
    }

    /// Проверить, достиг ли SL
    pub fn is_sl_hit(&self, current_price: f64) -> bool {
        match self.side {
            Side::Long => current_price <= self.sl_price,
            Side::Short => current_price >= self.sl_price,
        }
    }

    /// Проверить, достиг ли TP
    pub fn is_tp_hit(&self, current_price: f64) -> bool {
        match self.side {
            Side::Long => current_price >= self.tp_price,
            Side::Short => current_price <= self.tp_price,
        }
    }

    /// Исчерпаны ли свечи (candles_left <= 0)
    pub fn is_expired(&self) -> bool {
        self.candles_left <= 0
    }
}

// ═══════════════════════════════════════════════════════════
// POSITION UPDATE EVENT (sent to WebUI via Redpanda)
// ═══════════════════════════════════════════════════════════

/// Событие обновления позиции для отправки в WebUI через Kafka/Redpanda
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionUpdateEvent {
    pub event_type: String,           // "position_update" | "position_opened" | "position_closed"
    pub position_id: i64,
    pub symbol: String,
    pub tf_minutes: i16,
    pub side: String,                 // "LONG" | "SHORT"
    pub entry_price: f64,
    pub current_price: f64,
    pub sl_price: f64,
    pub tp_price: f64,
    pub qty: f64,
    pub leverage: u16,
    pub unrealized_pnl: f64,
    pub unrealized_pnl_pct: f64,
    pub realized_pnl: Option<f64>,
    pub realized_pnl_pct: Option<f64>,
    pub candles_left: i16,
    pub max_hold_bars: i16,
    pub combined_score: f32,
    pub close_reason: Option<String>,
    pub opened_at: String,            // ISO 8601
    pub closed_at: Option<String>,    // ISO 8601
    pub timestamp_ms: i64,
}

impl From<&ManagedPosition> for PositionUpdateEvent {
    fn from(pos: &ManagedPosition) -> Self {
        Self {
            event_type: "position_update".to_string(),
            position_id: pos.position_id,
            symbol: pos.symbol.clone(),
            tf_minutes: pos.tf_minutes,
            side: pos.side.as_str().to_string(),
            entry_price: pos.entry_price,
            current_price: pos.current_price,
            sl_price: pos.sl_price,
            tp_price: pos.tp_price,
            qty: pos.qty,
            leverage: pos.leverage,
            unrealized_pnl: pos.unrealized_pnl,
            unrealized_pnl_pct: pos.unrealized_pnl_pct,
            realized_pnl: None,
            realized_pnl_pct: None,
            candles_left: pos.candles_left,
            max_hold_bars: pos.max_hold_bars,
            combined_score: pos.combined_score,
            close_reason: None,
            opened_at: pos.opened_at.to_rfc3339(),
            closed_at: None,
            timestamp_ms: Utc::now().timestamp_millis(),
        }
    }
}

// ═══════════════════════════════════════════════════════════
// TIMEFRAME SLOT ALLOCATION
// ═══════════════════════════════════════════════════════════

/// Распределение слотов по таймфреймам.
/// При 10 одновременных ордерах: 1h=7, 4h=2, 15m=1
#[derive(Debug, Clone)]
pub struct TimeframeAllocation {
    /// tf_minutes → количество слотов
    pub slots: Vec<(i16, u16)>,
    /// Общее количество ордеров
    pub total: u16,
}

impl TimeframeAllocation {
    /// Стандартное распределение: 1h=70%, 4h=20%, 15m=10%
    pub fn default_10() -> Self {
        Self {
            slots: vec![
                (1, 0),    // 1m: 0%
                (5, 0),    // 5m: 0%
                (15, 1),   // 15m: 10%
                (60, 7),   // 1h: 70%
                (240, 2),  // 4h: 20%
                (1440, 0), // 1d: 0%
            ],
            total: 10,
        }
    }

    /// Получить количество слотов для таймфрейма
    pub fn slots_for_tf(&self, tf_minutes: i16) -> u16 {
        self.slots
            .iter()
            .find(|(tf, _)| *tf == tf_minutes)
            .map(|(_, count)| *count)
            .unwrap_or(0)
    }

    /// Все торгуемые таймфреймы
    pub fn timeframes(&self) -> Vec<i16> {
        self.slots.iter().map(|(tf, _)| *tf).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_side_conversions() {
        assert_eq!(Side::Long.as_db_i16(), 1);
        assert_eq!(Side::Short.as_db_i16(), -1);
        assert_eq!(Side::from_db_i16(1), Some(Side::Long));
        assert_eq!(Side::from_db_i16(-1), Some(Side::Short));
        assert_eq!(Side::from_db_i16(0), None);
    }

    #[test]
    fn test_side_binance_strings() {
        assert_eq!(Side::Long.entry_side_str(), "BUY");
        assert_eq!(Side::Long.close_side_str(), "SELL");
        assert_eq!(Side::Short.entry_side_str(), "SELL");
        assert_eq!(Side::Short.close_side_str(), "BUY");
    }

    #[test]
    fn test_allocation_default() {
        let alloc = TimeframeAllocation::default_10();
        assert_eq!(alloc.total, 10);
        assert_eq!(alloc.slots_for_tf(60), 7);
        assert_eq!(alloc.slots_for_tf(240), 2);
        assert_eq!(alloc.slots_for_tf(15), 1);
        assert_eq!(alloc.slots_for_tf(5), 0); // не торгуем
    }

    #[test]
    fn test_managed_position_pnl() {
        let pos = ManagedPosition {
            position_id: 1,
            symbol: "BTCUSDT".to_string(),
            symbol_id: 1,
            tf_minutes: 60,
            side: Side::Long,
            entry_price: 50000.0,
            sl_price: 49000.0,
            tp_price: 52000.0,
            qty: 0.01,
            leverage: 10,
            candles_left: 20,
            max_hold_bars: 25,
            unrealized_pnl: 0.0,
            unrealized_pnl_pct: 0.0,
            current_price: 50000.0,
            opened_at: Utc::now(),
            combined_score: 0.75,
            p_super: 0.8,
            entry_order_id: None,
            sl_order_id: None,
            tp_order_id: None,
        };

        // Long +2%
        let (pnl, pnl_pct) = pos.calc_unrealized_pnl(51000.0);
        assert!((pnl - 10.0).abs() < 0.01);   // 0.01 BTC * 1000 = 10 USDT
        assert!((pnl_pct - 2.0).abs() < 0.01);

        // SL check
        assert!(!pos.is_sl_hit(50000.0));
        assert!(pos.is_sl_hit(48999.0));

        // TP check
        assert!(!pos.is_tp_hit(51000.0));
        assert!(pos.is_tp_hit(52000.0));
    }

    #[test]
    fn test_managed_position_short() {
        let pos = ManagedPosition {
            position_id: 2,
            symbol: "ETHUSDT".to_string(),
            symbol_id: 2,
            tf_minutes: 240,
            side: Side::Short,
            entry_price: 3000.0,
            sl_price: 3100.0,
            tp_price: 2800.0,
            qty: 1.0,
            leverage: 10,
            candles_left: 15,
            max_hold_bars: 25,
            unrealized_pnl: 0.0,
            unrealized_pnl_pct: 0.0,
            current_price: 3000.0,
            opened_at: Utc::now(),
            combined_score: 0.72,
            p_super: 0.65,
            entry_order_id: None,
            sl_order_id: None,
            tp_order_id: None,
        };

        // Short, price dropped → profit
        let (pnl, pnl_pct) = pos.calc_unrealized_pnl(2900.0);
        assert!((pnl - 100.0).abs() < 0.01);
        assert!((pnl_pct - 3.33).abs() < 0.1);

        // SL for short: price goes UP
        assert!(!pos.is_sl_hit(3050.0));
        assert!(pos.is_sl_hit(3100.0));

        // TP for short: price goes DOWN
        assert!(!pos.is_tp_hit(2900.0));
        assert!(pos.is_tp_hit(2800.0));
    }
}
