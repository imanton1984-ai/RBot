use serde::{Deserialize, Serialize};

/// 1 long, -1 short, 0 neutral
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i16)]
pub enum Side {
    Short = -1,
    Neutral = 0,
    Long = 1,
}

/// kind: 1=entry, 2=filter, 3=warning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i16)]
pub enum SignalKind {
    Entry = 1,
    Filter = 2,
    Warning = 3,
}

/// Индикаторы — фиксируем стабильные ID (важно для storage и аналитики)
/// Порядок: самые важные — меньше ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i16)]
pub enum IndicatorId {
    Rsi = 1,
    Macd = 2,
    Ema20 = 3,
    Ema50 = 4,
    Ema200 = 5,
    Sma = 6,
    Bollinger = 7,
    Stoch = 8,
    Atr = 9,
    Adx = 10,
    Vwap = 11,
    Obv = 12,
    Cci = 13,
    Williams = 14,
    Alligator = 15,
    Trend = 16,
    VolumeSpike = 17,
    SrLevels = 18,
}

/// Для order_engine (futures)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i16)]
pub enum OrderType {
    Market = 1,
    Limit = 2,
    StopMarket = 3,
    TakeProfitMarket = 4,
    StopLimit = 5,
    TakeProfitLimit = 6,
}

/// Статус — упрощенный (дальше можно расширить)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i16)]
pub enum OrderStatus {
    New = 1,
    PartiallyFilled = 2,
    Filled = 3,
    Canceled = 4,
    Rejected = 5,
    Expired = 6,
}

/// Для position_tracker
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i16)]
pub enum PositionStatus {
    Open = 1,
    Closed = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i16)]
pub enum PositionEventType {
    Open = 1,
    Fill = 2,
    Tp1 = 3,
    Tp2 = 4,
    Tp3 = 5,
    SlMoved = 6,
    Close = 7,
    Reconcile = 8,
}
