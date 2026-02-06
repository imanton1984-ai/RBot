use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TimeFrame {
    M1,
    M5,
    M15,
    H1,
    H4,
    D1,
}

impl TimeFrame {
    pub fn as_str(&self) -> &'static str {
        match self {
            TimeFrame::M1 => "1m",
            TimeFrame::M5 => "5m",
            TimeFrame::M15 => "15m",
            TimeFrame::H1 => "1h",
            TimeFrame::H4 => "4h",
            TimeFrame::D1 => "1d",
        }
    }

    pub fn all_timeframes() -> Vec<TimeFrame> {
        vec![
            TimeFrame::M1,
            TimeFrame::M5,
            TimeFrame::M15,
            TimeFrame::H1,
            TimeFrame::H4,
            TimeFrame::D1,
        ]
    }

    pub fn to_minutes(&self) -> i32 {
        match self {
            TimeFrame::M1 => 1,
            TimeFrame::M5 => 5,
            TimeFrame::M15 => 15,
            TimeFrame::H1 => 60,
            TimeFrame::H4 => 240,
            TimeFrame::D1 => 1440,
        }
    }
}

impl fmt::Display for TimeFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for TimeFrame {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "1m" | "m1" => Ok(TimeFrame::M1),
            "5m" | "m5" => Ok(TimeFrame::M5),
            "15m" | "m15" => Ok(TimeFrame::M15),
            "1h" | "h1" => Ok(TimeFrame::H1),
            "4h" | "h4" => Ok(TimeFrame::H4),
            "1d" | "d1" => Ok(TimeFrame::D1),
            _ => Err(()),
        }
    }
}
