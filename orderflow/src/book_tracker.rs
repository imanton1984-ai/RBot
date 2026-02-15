/// Maintains the latest order-book state for a single symbol using
/// Binance Futures `@depth20@100ms` snapshots (full top-20 refresh
/// every 100 ms — no local book maintenance required).

use crate::types::BookLevel;

/// Computed order-book metrics derived from a single depth snapshot.
#[derive(Debug, Clone)]
pub struct BookMetrics {
    /// (total_bid_qty − total_ask_qty) / (total_bid_qty + total_ask_qty).
    pub bid_ask_imbalance: f64,
    /// Total bid depth in USD (price × qty summed over 20 levels).
    pub bid_depth_usd: f64,
    /// Total ask depth in USD.
    pub ask_depth_usd: f64,
    /// Spread between best ask and best bid in basis points.
    pub spread_bps: f64,
    /// Price of the level with the highest quantity on the bid side.
    pub bid_wall_price: Option<f64>,
    /// Price of the level with the highest quantity on the ask side.
    pub ask_wall_price: Option<f64>,
}

/// Tracks a single symbol's top-20 order-book snapshot.
pub struct BookTracker {
    bids: Vec<BookLevel>,
    asks: Vec<BookLevel>,
}

impl BookTracker {
    pub fn new() -> Self {
        Self {
            bids: Vec::with_capacity(20),
            asks: Vec::with_capacity(20),
        }
    }

    // ── Update ──────────────────────────────────────────────────────

    /// Replace the entire book snapshot.  Called on every
    /// `@depth20@100ms` message.
    pub fn update(&mut self, bids: Vec<BookLevel>, asks: Vec<BookLevel>) {
        self.bids = bids;
        self.asks = asks;
    }

    // ── Compute ─────────────────────────────────────────────────────

    /// Derive all order-book metrics from the current snapshot.
    pub fn compute_metrics(&self) -> BookMetrics {
        let (bid_depth_usd, total_bid_qty, bid_wall) = Self::side_stats(&self.bids);
        let (ask_depth_usd, total_ask_qty, ask_wall) = Self::side_stats(&self.asks);

        let sum_qty = total_bid_qty + total_ask_qty;
        let bid_ask_imbalance = if sum_qty > 0.0 {
            (total_bid_qty - total_ask_qty) / sum_qty
        } else {
            0.0
        };

        let spread_bps = Self::compute_spread_bps(&self.bids, &self.asks);

        BookMetrics {
            bid_ask_imbalance,
            bid_depth_usd,
            ask_depth_usd,
            spread_bps,
            bid_wall_price: bid_wall,
            ask_wall_price: ask_wall,
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────

    /// Returns (total_depth_usd, total_qty, wall_price) for one side.
    fn side_stats(levels: &[BookLevel]) -> (f64, f64, Option<f64>) {
        let mut total_usd = 0.0_f64;
        let mut total_qty = 0.0_f64;
        let mut max_qty = 0.0_f64;
        let mut wall_price: Option<f64> = None;

        for lvl in levels {
            let notional = lvl.price * lvl.qty;
            total_usd += notional;
            total_qty += lvl.qty;
            if lvl.qty > max_qty {
                max_qty = lvl.qty;
                wall_price = Some(lvl.price);
            }
        }
        (total_usd, total_qty, wall_price)
    }

    /// Spread in basis points between best ask and best bid.
    /// Returns 0.0 if either side is empty.
    fn compute_spread_bps(bids: &[BookLevel], asks: &[BookLevel]) -> f64 {
        match (bids.first(), asks.first()) {
            (Some(best_bid), Some(best_ask)) if best_bid.price > 0.0 => {
                (best_ask.price - best_bid.price) / best_bid.price * 10_000.0
            }
            _ => 0.0,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(price: f64, qty: f64) -> BookLevel {
        BookLevel { price, qty }
    }

    #[test]
    fn basic_metrics() {
        let mut bt = BookTracker::new();
        bt.update(
            vec![mk(100.0, 5.0), mk(99.0, 10.0)],
            vec![mk(101.0, 3.0), mk(102.0, 8.0)],
        );
        let m = bt.compute_metrics();

        // bid depth = 100*5 + 99*10 = 1490
        assert!((m.bid_depth_usd - 1490.0).abs() < 1e-8);
        // ask depth = 101*3 + 102*8 = 1119
        assert!((m.ask_depth_usd - 1119.0).abs() < 1e-8);
        // total_bid_qty=15, total_ask_qty=11 → imbalance = 4/26
        assert!((m.bid_ask_imbalance - 4.0 / 26.0).abs() < 1e-8);
        // spread = (101 - 100)/100 * 10000 = 100 bps
        assert!((m.spread_bps - 100.0).abs() < 1e-8);
        // bid wall = 99.0 (qty 10 > 5)
        assert_eq!(m.bid_wall_price, Some(99.0));
        // ask wall = 102.0 (qty 8 > 3)
        assert_eq!(m.ask_wall_price, Some(102.0));
    }

    #[test]
    fn empty_book() {
        let bt = BookTracker::new();
        let m = bt.compute_metrics();
        assert_eq!(m.bid_ask_imbalance, 0.0);
        assert_eq!(m.spread_bps, 0.0);
        assert!(m.bid_wall_price.is_none());
    }
}
