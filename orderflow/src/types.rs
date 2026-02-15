/// Core data types for real-time order flow analysis.

/// Complete snapshot of order flow metrics for a single symbol.
///
/// Updated on every `@depth20@100ms` tick. Trade-flow fields are
/// computed over a sliding window (default 60 s, configurable via
/// `ORDERFLOW_WINDOW_SECS`).
#[derive(Debug, Clone)]
pub struct OrderFlowSnapshot {
    pub symbol: String,
    pub timestamp_ms: i64,

    // ── Order book metrics ──────────────────────────────────────────
    /// (total_bid_qty − total_ask_qty) / (total_bid_qty + total_ask_qty).
    /// Range: \[−1, 1\].
    pub bid_ask_imbalance: f64,
    /// Total bid depth in USD (top 20 levels).
    pub bid_depth_usd: f64,
    /// Total ask depth in USD (top 20 levels).
    pub ask_depth_usd: f64,
    /// Spread in basis points.
    pub spread_bps: f64,
    /// Price of the largest bid concentration in top 20 levels.
    pub bid_wall_price: Option<f64>,
    /// Price of the largest ask concentration in top 20 levels.
    pub ask_wall_price: Option<f64>,

    // ── Trade flow metrics (sliding window) ─────────────────────────
    /// buy_volume − sell_volume over the window.
    pub volume_delta: f64,
    /// volume_delta / total_volume.  Range: \[−1, 1\].
    pub volume_delta_ratio: f64,
    /// Total taker buy volume (USD) over the window.
    pub buy_volume: f64,
    /// Total taker sell volume (USD) over the window.
    pub sell_volume: f64,
    /// Trades whose notional > 10× the running average in the window.
    pub large_trade_count: u32,
    /// Net direction of large trades.  Range: \[−1, 1\].
    pub large_trade_bias: f64,

    // ── Derived ─────────────────────────────────────────────────────
    /// Combined buy/sell pressure score.
    /// `0.4 * bid_ask_imbalance + 0.6 * volume_delta_ratio`.
    /// Range: \[−1, 1\].
    pub pressure_score: f64,
}

impl Default for OrderFlowSnapshot {
    fn default() -> Self {
        Self {
            symbol: String::new(),
            timestamp_ms: 0,
            bid_ask_imbalance: 0.0,
            bid_depth_usd: 0.0,
            ask_depth_usd: 0.0,
            spread_bps: 0.0,
            bid_wall_price: None,
            ask_wall_price: None,
            volume_delta: 0.0,
            volume_delta_ratio: 0.0,
            buy_volume: 0.0,
            sell_volume: 0.0,
            large_trade_count: 0,
            large_trade_bias: 0.0,
            pressure_score: 0.0,
        }
    }
}

/// A single price-quantity level in the order book.
#[derive(Debug, Clone, Copy)]
pub struct BookLevel {
    pub price: f64,
    pub qty: f64,
}

/// Parsed aggregate trade from Binance `@aggTrade` stream.
#[derive(Debug, Clone, Copy)]
pub struct AggTrade {
    /// Event time (ms).
    pub timestamp_ms: i64,
    /// Trade price.
    pub price: f64,
    /// Trade quantity (base asset).
    pub qty: f64,
    /// `true` → buyer is maker → this is a **sell** (taker sold into bid).
    pub is_buyer_maker: bool,
}

impl AggTrade {
    /// Notional value in quote asset (≈ USD for *USDT pairs).
    #[inline]
    pub fn notional(&self) -> f64 {
        self.price * self.qty
    }

    /// `true` when the taker **bought** (i.e. buyer is *not* the maker).
    #[inline]
    pub fn is_buy(&self) -> bool {
        !self.is_buyer_maker
    }
}
