use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(i16)]
pub enum Timeframe {
    M1   = 1,
    M5   = 5,
    M15  = 15,
    H1   = 60,
    H4   = 240,
    D1   = 1440,
}

impl Timeframe {
    #[inline]
    pub const fn minutes(self) -> i16 {
        self as i16
    }

    #[inline]
    pub const fn as_binance_interval(self) -> &'static str {
        match self {
            Timeframe::M1  => "1m",
            Timeframe::M5  => "5m",
            Timeframe::M15 => "15m",
            Timeframe::H1  => "1h",
            Timeframe::H4  => "4h",
            Timeframe::D1  => "1d",
        }
    }

    #[inline]
    pub const fn duration_secs(self) -> i64 {
        (self.minutes() as i64) * 60
    }

    /// Таблица candles для конкретного TF (у нас candles_{tf})
    #[inline]
    pub const fn candles_table(self) -> &'static str {
        match self {
            Timeframe::M1  => "market.candles_1m",
            Timeframe::M5  => "market.candles_5m",
            Timeframe::M15 => "market.candles_15m",
            Timeframe::H1  => "market.candles_1h",
            Timeframe::H4  => "market.candles_4h",
            Timeframe::D1  => "market.candles_1d",
        }
    }

    /// Таблица indicators snapshot для конкретного TF (у нас indicators_{tf})
    #[inline]
    pub const fn indicators_table(self) -> &'static str {
        match self {
            Timeframe::M1  => "market.indicators_1m",
            Timeframe::M5  => "market.indicators_5m",
            Timeframe::M15 => "market.indicators_15m",
            Timeframe::H1  => "market.indicators_1h",
            Timeframe::H4  => "market.indicators_4h",
            Timeframe::D1  => "market.indicators_1d",
        }
    }

    /// Парсинг из строки "1m"/"5m"/"15m"/"1h"/"4h"/"1d"
    pub fn parse(s: &str) -> Result<Self, TimeframeParseError> {
        match s {
            "1m"  => Ok(Timeframe::M1),
            "5m"  => Ok(Timeframe::M5),
            "15m" => Ok(Timeframe::M15),
            "1h"  => Ok(Timeframe::H1),
            "4h"  => Ok(Timeframe::H4),
            "1d"  => Ok(Timeframe::D1),
            _ => Err(TimeframeParseError(s.to_string())),
        }
    }
}

impl fmt::Display for Timeframe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_binance_interval())
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Invalid timeframe: {0}")]
pub struct TimeframeParseError(pub String);
