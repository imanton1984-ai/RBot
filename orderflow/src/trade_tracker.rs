/// Sliding-window aggregator for `@aggTrade` events.
///
/// Maintains a circular buffer ([`VecDeque`]) of recent trades and
/// computes volume-delta, large-trade detection, and directional bias
/// over a configurable window (`ORDERFLOW_WINDOW_SECS`, default 60).

use std::collections::VecDeque;

use crate::types::AggTrade;

/// Computed trade-flow metrics over the sliding window.
#[derive(Debug, Clone)]
pub struct TradeMetrics {
    /// buy_volume − sell_volume (USD).
    pub volume_delta: f64,
    /// volume_delta / total_volume.  Range: \[−1, 1\].
    pub volume_delta_ratio: f64,
    /// Total taker-buy volume (USD) in the window.
    pub buy_volume: f64,
    /// Total taker-sell volume (USD) in the window.
    pub sell_volume: f64,
    /// Number of trades whose notional > 10× average in the window.
    pub large_trade_count: u32,
    /// Net direction of large trades:
    /// (large_buy_notional − large_sell_notional) /
    /// (large_buy_notional + large_sell_notional).
    /// Range: \[−1, 1\].  0.0 when no large trades exist.
    pub large_trade_bias: f64,
}

/// Per-symbol trade window tracker.
pub struct TradeTracker {
    /// Sliding window of recent trades.
    trades: VecDeque<AggTrade>,
    /// Window length in milliseconds.
    window_ms: i64,
}

impl TradeTracker {
    /// Create a new tracker with the given window in **seconds**.
    pub fn new(window_secs: u64) -> Self {
        Self {
            trades: VecDeque::with_capacity(4096),
            window_ms: window_secs as i64 * 1000,
        }
    }

    // ── Update ──────────────────────────────────────────────────────

    /// Push a new aggregate trade and evict stale entries.
    pub fn push(&mut self, trade: AggTrade) {
        self.trades.push_back(trade);
        self.evict(trade.timestamp_ms);
    }

    /// Remove trades that are older than the window relative to `now_ms`.
    fn evict(&mut self, now_ms: i64) {
        let cutoff = now_ms - self.window_ms;
        while self.trades.front().is_some_and(|t| t.timestamp_ms < cutoff) {
            self.trades.pop_front();
        }
    }

    // ── Compute ─────────────────────────────────────────────────────

    pub fn compute_metrics(&self) -> TradeMetrics {
        if self.trades.is_empty() {
            return TradeMetrics {
                volume_delta: 0.0,
                volume_delta_ratio: 0.0,
                buy_volume: 0.0,
                sell_volume: 0.0,
                large_trade_count: 0,
                large_trade_bias: 0.0,
            };
        }

        // First pass: accumulate buy/sell volumes and total notional for
        // computing the average trade size.
        let mut buy_vol = 0.0_f64;
        let mut sell_vol = 0.0_f64;
        let mut total_notional = 0.0_f64;

        for t in &self.trades {
            let n = t.notional();
            total_notional += n;
            if t.is_buy() {
                buy_vol += n;
            } else {
                sell_vol += n;
            }
        }

        let avg_notional = total_notional / self.trades.len() as f64;
        let large_threshold = avg_notional * 10.0;

        // Second pass: detect large trades.
        let mut large_count: u32 = 0;
        let mut large_buy = 0.0_f64;
        let mut large_sell = 0.0_f64;

        for t in &self.trades {
            let n = t.notional();
            if n > large_threshold {
                large_count += 1;
                if t.is_buy() {
                    large_buy += n;
                } else {
                    large_sell += n;
                }
            }
        }

        let total_vol = buy_vol + sell_vol;
        let delta = buy_vol - sell_vol;
        let delta_ratio = if total_vol > 0.0 {
            delta / total_vol
        } else {
            0.0
        };

        let large_sum = large_buy + large_sell;
        let large_bias = if large_sum > 0.0 {
            (large_buy - large_sell) / large_sum
        } else {
            0.0
        };

        TradeMetrics {
            volume_delta: delta,
            volume_delta_ratio: delta_ratio,
            buy_volume: buy_vol,
            sell_volume: sell_vol,
            large_trade_count: large_count,
            large_trade_bias: large_bias,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn trade(ts: i64, price: f64, qty: f64, buyer_maker: bool) -> AggTrade {
        AggTrade {
            timestamp_ms: ts,
            price,
            qty,
            is_buyer_maker: buyer_maker,
        }
    }

    #[test]
    fn empty_window() {
        let tt = TradeTracker::new(60);
        let m = tt.compute_metrics();
        assert_eq!(m.volume_delta, 0.0);
        assert_eq!(m.large_trade_count, 0);
    }

    #[test]
    fn basic_delta() {
        let mut tt = TradeTracker::new(60);
        // Buy: price=100, qty=1 → notional=100
        tt.push(trade(1000, 100.0, 1.0, false));
        // Sell: price=100, qty=2 → notional=200
        tt.push(trade(2000, 100.0, 2.0, true));

        let m = tt.compute_metrics();
        assert!((m.buy_volume - 100.0).abs() < 1e-8);
        assert!((m.sell_volume - 200.0).abs() < 1e-8);
        assert!((m.volume_delta - (-100.0)).abs() < 1e-8);
        // ratio = -100/300
        assert!((m.volume_delta_ratio - (-100.0 / 300.0)).abs() < 1e-8);
    }

    #[test]
    fn eviction() {
        let mut tt = TradeTracker::new(5); // 5-second window
        tt.push(trade(1_000, 100.0, 1.0, false));
        tt.push(trade(3_000, 100.0, 1.0, false));
        // Push at t=7_000 → cutoff = 2_000 → t=1_000 is evicted
        tt.push(trade(7_000, 100.0, 1.0, true));

        let m = tt.compute_metrics();
        // Only 2 trades survive: the one at 3_000 (buy) and 7_000 (sell).
        assert!((m.buy_volume - 100.0).abs() < 1e-8);
        assert!((m.sell_volume - 100.0).abs() < 1e-8);
    }

    #[test]
    fn large_trade_detection() {
        let mut tt = TradeTracker::new(60);
        // 99 small buys: notional = 100 each → total 9_900
        for i in 0..99 {
            tt.push(trade(1000 + i * 10, 100.0, 1.0, false));
        }
        // 1 large sell: notional = 100 * 20 = 2_000
        // avg = (9_900 + 2_000) / 100 = 119
        // threshold = 10 × 119 = 1_190
        // 2_000 > 1_190 → detected as large
        tt.push(trade(2000, 100.0, 20.0, true));

        let m = tt.compute_metrics();
        assert!(m.large_trade_count >= 1);
        // large_trade_bias should be negative (sell)
        assert!(m.large_trade_bias < 0.0);
    }
}
