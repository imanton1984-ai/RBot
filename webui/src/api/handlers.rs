// webui/src/api/handlers.rs — Full WebUI API Handlers
//
// Endpoints for:
//   - Market data (pairs, candles, indicators, signals)
//   - Balance (from Binance Futures account via settings)
//   - PnL (calculated without leverage = real PnL)
//   - Connections (DB, Redpanda, REST API, WebSocket, Account)
//   - Trading Options (order_settings.toml)
//   - Order Options (order_manager.toml + risk_manager.toml)
//   - Alerts (from risk.alerts Kafka topic / stub)
//   - Trading control (start/stop, emergency stop)
//   - Manual order placement with TP/SL

use axum::{extract::{Query, State}, Json, http::StatusCode};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use crate::state::{
    AppState, BalanceInfo, PnlOverview,
    ConnectionStatus, TradingOptionsPayload,
    AutoTradingState, ManualOrderRequest, CandlesLeftUpdateRequest,
};

// ─── Query parameters ─────────────────────────────────────────────────
#[derive(Debug, Deserialize)]
pub struct PairsQuery { pub search: Option<String>, pub limit: Option<i64> }

#[derive(Debug, Deserialize)]
pub struct CandlesQuery { pub pair: String, pub tf: i32, pub limit: Option<i64> }

#[derive(Debug, Deserialize)]
pub struct IndicatorsQuery {
    pub pair: String,
    pub tf: i32,
    #[serde(rename = "type")]
    pub indicator_type: String,
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct SignalsQuery { pub pair: Option<String>, pub tf: Option<i32>, pub limit: Option<i64> }

#[derive(Debug, Deserialize)]
pub struct AlertsQuery { pub limit: Option<i64> }

#[derive(Debug, Deserialize)]
pub struct PnlQuery { pub range: Option<String> }

#[derive(Debug, Deserialize)]
pub struct StatsQuery {
    pub range: Option<String>,
    pub interval: Option<String>,
}

// ─── Response types ────────────────────────────────────────────────────
#[derive(Debug, Serialize)]
pub struct PairInfo { pub symbol: String, pub symbol_id: i64 }

#[derive(Debug, Serialize)]
pub struct CandleResponse { pub t: i64, pub o: f64, pub h: f64, pub l: f64, pub c: f64, pub v: f64 }

#[derive(Debug, Serialize)]
pub struct IndicatorResponse { pub t: i64, pub value: Option<f64> }

#[derive(Debug, Serialize)]
pub struct MarketSummary {
    pub price: f64,
    pub volume_24h: f64,
    pub change_24h: f64,
    pub change_1h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
}

#[derive(Debug, Serialize)]
pub struct SignalRow {
    pub id: String,
    pub time: String,
    pub pair: String,
    pub tf: i32,
    pub side: i16,
    pub score: f32,
    pub entry_price: f64,
    pub sl_price: f64,
    pub tp_price: f64,
    pub strategy: String,
    pub status: String,
    pub p_super: f32,
}

#[derive(Debug, Serialize)]
pub struct PositionRow {
    pub id: i64,
    pub pair: String,
    pub side: String,
    pub qty: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub pnl_usdt: f64,
    pub pnl_pct: f64,
    pub status: String,
    pub open_time: String,
    pub candles_left: i16,
    pub leverage: u16,
    pub tf_minutes: i16,
}

#[derive(Debug, Serialize)]
pub struct HistoryRow {
    pub id: i64,
    pub pair: String,
    pub side: String,
    pub qty: f64,
    pub entry_price: f64,
    pub close_price: f64,
    pub close_type: String,
    pub pnl_usdt: f64,
    pub pnl_pct: f64,
    pub open_time: String,
    pub close_time: String,
}

// ─── Statistics Response Types ────────────────────────────────────────
#[derive(Debug, Serialize)]
pub struct StatsSummary {
    pub total_trades: i64,
    pub winning_trades: i64,
    pub losing_trades: i64,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub total_pnl_pct: f64,       // PnL as % of balance
    pub avg_win: f64,
    pub avg_win_pct: f64,         // avg win as %
    pub avg_loss: f64,
    pub avg_loss_pct: f64,        // avg loss as %
    pub profit_factor: f64,
    pub best_trade: f64,
    pub best_trade_pct: f64,
    pub worst_trade: f64,
    pub worst_trade_pct: f64,
    pub avg_trade_duration_hours: f64,
    pub max_consecutive_wins: i64,
    pub max_consecutive_losses: i64,
    pub expectancy: f64,          // (win_rate * avg_win + (1-win_rate) * avg_loss)
    pub sharpe_approx: f64,       // rough Sharpe ratio approximation
}

#[derive(Debug, Serialize)]
pub struct PnlTimePoint {
    pub t: i64,
    pub cumulative_pnl: f64,
    pub balance: f64,
}

#[derive(Debug, Serialize)]
pub struct PnlTimeline {
    pub points: Vec<PnlTimePoint>,
    pub start_date: String,
    pub end_date: String,
}

#[derive(Debug, Serialize)]
pub struct TradeDistribution {
    pub by_pair: Vec<PairStats>,
    pub by_timeframe: Vec<TfStats>,
    pub by_close_type: Vec<CloseTypeStats>,
    pub by_side: Vec<SideStats>,
}

#[derive(Debug, Serialize)]
pub struct PairStats {
    pub pair: String,
    pub trades: i64,
    pub wins: i64,
    pub pnl: f64,
    pub win_rate: f64,
}

#[derive(Debug, Serialize)]
pub struct TfStats {
    pub tf: i16,
    pub trades: i64,
    pub wins: i64,
    pub pnl: f64,
    pub win_rate: f64,
}

#[derive(Debug, Serialize)]
pub struct CloseTypeStats {
    pub close_type: String,
    pub count: i64,
    pub pnl: f64,
    pub win_rate: f64,
}

#[derive(Debug, Serialize)]
pub struct SideStats {
    pub side: String,
    pub trades: i64,
    pub wins: i64,
    pub pnl: f64,
    pub win_rate: f64,
}

#[derive(Debug, Serialize)]
pub struct BestWorstTrades {
    pub best: Vec<HistoryRow>,
    pub worst: Vec<HistoryRow>,
}

#[derive(Debug, Serialize)]
pub struct TimeBasedStats {
    pub period: String,
    pub data: Vec<TimeSlotStats>,
}

#[derive(Debug, Serialize)]
pub struct TimeSlotStats {
    pub label: String,
    pub trades: i64,
    pub wins: i64,
    pub losses: i64,
    pub pnl: f64,
    pub win_rate: f64,
    pub avg_pnl: f64,
}

#[derive(Debug, Serialize)]
pub struct MonthlyStats {
    pub month: String,
    pub trades: i64,
    pub wins: i64,
    pub pnl: f64,
    pub win_rate: f64,
}

#[derive(Debug, Serialize)]
pub struct DayPartStats {
    pub bucket: String,      // "Morning", "Day", "Evening", "Night"
    pub hours: String,        // "06:00–12:00"
    pub trades: i64,
    pub wins: i64,
    pub losses: i64,
    pub pnl: f64,
    pub pnl_pct: f64,       // PnL as % of balance
    pub win_rate: f64,
    pub avg_pnl: f64,
    pub best_trade: f64,
    pub worst_trade: f64,
}

#[derive(Debug, Serialize)]
pub struct FullStatistics {
    pub summary: StatsSummary,
    pub timeline: PnlTimeline,
    pub distribution: TradeDistribution,
    pub best_worst: BestWorstTrades,
    pub hourly: TimeBasedStats,
    pub daily: TimeBasedStats,
    pub weekly: TimeBasedStats,
    pub monthly: Vec<MonthlyStats>,
    pub day_parts: Vec<DayPartStats>,
}

#[derive(Debug, Serialize)]
pub struct AlertRow {
    pub id: i64,
    pub pair: String,
    pub alert_type: String,
    pub time_ago: String,
    pub message: String,
    pub severity: String,
    pub source: String,
    pub timestamp: String,
}

// ─── Helpers ───────────────────────────────────────────────────────────

/// Map tf_minutes → table name
fn candle_table(tf: i32) -> &'static str {
    match tf {
        1    => "market.candles_1m",
        5    => "market.candles_5m",
        15   => "market.candles_15m",
        60   => "market.candles_1h",
        240  => "market.candles_4h",
        1440 => "market.candles_1d",
        _    => "market.candles_1m",
    }
}

/// Map tf_minutes → timeframe string used in market.candles_live
fn tf_live_label(tf: i32) -> &'static str {
    match tf {
        1    => "1m",
        5    => "5m",
        15   => "15m",
        60   => "1h",
        240  => "4h",
        1440 => "1d",
        _    => "1m",
    }
}

// ─── GET /api/pairs ────────────────────────────────────────────────────
pub async fn get_pairs(
    Query(query): Query<PairsQuery>,
    State(state): State<AppState>,
) -> Result<Json<Vec<PairInfo>>, StatusCode> {
    let limit = query.limit.unwrap_or(500); // Default 500 for full list
    let search = query.search.unwrap_or_default();

    let rows = sqlx::query(
        "SELECT symbol, symbol_id FROM market.pairs
         WHERE is_active = true AND ($1 = '' OR symbol ILIKE $1)
         ORDER BY volume_24h_usdt DESC NULLS LAST, symbol
         LIMIT $2"
    )
    .bind(format!("%{}%", search))
    .bind(limit)
    .fetch_all(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("Pairs error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let pairs = rows.iter().map(|r| PairInfo {
        symbol: r.get("symbol"),
        symbol_id: r.get("symbol_id"),
    }).collect();

    Ok(Json(pairs))
}

// ─── GET /api/market/summary ───────────────────────────────────────────
pub async fn get_market_summary(
    Query(query): Query<CandlesQuery>,
    State(state): State<AppState>,
) -> Result<Json<MarketSummary>, StatusCode> {
    let row = sqlx::query(
        "WITH latest AS (
            SELECT close FROM market.candles_1m
            WHERE symbol = $1 ORDER BY time DESC LIMIT 1
        ), stats AS (
            SELECT
                COALESCE(MAX(high), 0) as high_24h,
                COALESCE(MIN(low), 0) as low_24h,
                COALESCE(SUM(volume), 0) as volume_24h
            FROM market.candles_1m
            WHERE symbol = $1 AND time >= now() - INTERVAL '24 hours'
        ), price_24h_ago AS (
            SELECT close as price_24h FROM market.candles_1m
            WHERE symbol = $1 AND time <= now() - INTERVAL '24 hours'
            ORDER BY time DESC LIMIT 1
        ), price_1h_ago AS (
            SELECT close as price_1h FROM market.candles_1m
            WHERE symbol = $1 AND time <= now() - INTERVAL '1 hour'
            ORDER BY time DESC LIMIT 1
        )
        SELECT
            (SELECT close FROM latest) as last_price,
            (SELECT high_24h FROM stats),
            (SELECT low_24h FROM stats),
            (SELECT volume_24h FROM stats),
            (SELECT price_24h FROM price_24h_ago) as price_24h,
            (SELECT price_1h FROM price_1h_ago) as price_1h"
    )
    .bind(&query.pair)
    .fetch_optional(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("Market summary error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    match row {
        Some(r) => {
            let price: f64 = r.try_get("last_price").unwrap_or(0.0);
            let price_24h: f64 = r.try_get("price_24h").unwrap_or(price);
            let price_1h: f64 = r.try_get("price_1h").unwrap_or(price);
            let change_24h = if price_24h > 0.0 { (price - price_24h) / price_24h * 100.0 } else { 0.0 };
            let change_1h = if price_1h > 0.0 { (price - price_1h) / price_1h * 100.0 } else { 0.0 };
            Ok(Json(MarketSummary {
                price,
                volume_24h: r.try_get("volume_24h").unwrap_or(0.0),
                change_24h,
                change_1h,
                high_24h: r.try_get("high_24h").unwrap_or(0.0),
                low_24h: r.try_get("low_24h").unwrap_or(0.0),
            }))
        }
        None => Ok(Json(MarketSummary {
            price: 0.0, volume_24h: 0.0, change_24h: 0.0,
            change_1h: 0.0, high_24h: 0.0, low_24h: 0.0,
        })),
    }
}

// ─── GET /api/candles ──────────────────────────────────────────────────
pub async fn get_candles(
    Query(query): Query<CandlesQuery>,
    State(state): State<AppState>,
) -> Result<Json<Vec<CandleResponse>>, StatusCode> {
    let limit = query.limit.unwrap_or(500).min(5000);
    let table = candle_table(query.tf);

    let sql = format!(
        "SELECT c.time_ms as t, c.open as o, c.high as h, c.low as l, c.close as c, c.volume as v
         FROM {} c
         JOIN market.pairs p ON c.symbol_id = p.symbol_id
         WHERE p.symbol = $1
         ORDER BY c.time DESC
         LIMIT $2",
        table
    );

    let rows = sqlx::query(&sql)
        .bind(&query.pair)
        .bind(limit)
        .fetch_all(&state.db_pool)
        .await
        .map_err(|e| { tracing::error!("Candles error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let mut candles: Vec<CandleResponse> = rows.iter().rev().map(|r| CandleResponse {
        t: r.get("t"),
        o: r.get("o"),
        h: r.get("h"),
        l: r.get("l"),
        c: r.get("c"),
        v: r.get("v"),
    }).collect();

    // FIX #4: Append/update the current FORMING candle from market.candles_live.
    // The ingestor writes forming candles to candles_live via WebSocket UPSERT.
    //
    // IMPORTANT: Per runtime.toml, only 1m/5m/15m/1h have WS streams.
    // 4h and 1d use poll-on-close (no intra-candle updates in candles_live).
    // Fallback for 4h/1d: use the latest 1m live candle's close as current price
    // and update the last candle in the chart.
    //
    // KEY: candles_live stores `open_time_ms` while candles_1h stores `time_ms` (close time).
    // We convert open_time → close_time for consistent chart x-axis.
    {
        let tf_label = tf_live_label(query.tf);
        let tf_ms = query.tf as i64 * 60_000;

        // Try the exact TF from candles_live first
        let live_result = sqlx::query(
            "SELECT open_time_ms,
                    open as o, high as h, low as l, close as close_val, volume as v
             FROM market.candles_live
             WHERE symbol = $1 AND timeframe = $2
             ORDER BY open_time_ms DESC
             LIMIT 1"
        )
        .bind(&query.pair)
        .bind(tf_label)
        .fetch_optional(&state.db_pool)
        .await;

        // Extract live OHLCV data
        let live_data: Option<(i64, f64, f64, f64, f64, f64)> = match &live_result {
            Ok(Some(r)) => {
                let open_time_ms: i64 = r.get("open_time_ms");
                let live_t = open_time_ms + tf_ms;
                Some((live_t, r.get("o"), r.get("h"), r.get("l"), r.get("close_val"), r.get("v")))
            }
            _ => None,
        };

        // For 4h/1d (no WS stream), fallback: get current price from 1m candles_live
        // and just update the close of the last chart candle
        let fallback_price: Option<f64> = if live_data.is_none() && query.tf > 60 {
            sqlx::query(
                "SELECT close as close_val FROM market.candles_live
                 WHERE symbol = $1 AND timeframe = '1m'
                 ORDER BY open_time_ms DESC LIMIT 1"
            )
            .bind(&query.pair)
            .fetch_optional(&state.db_pool)
            .await
            .ok()
            .flatten()
            .map(|r| r.get::<f64, _>("close_val"))
        } else {
            None
        };

        if let Some((live_t, live_o, live_h, live_l, live_c, live_v)) = live_data {
            if live_o > 0.0 && live_c > 0.0 {
                if let Some(last) = candles.last_mut() {
                    if last.t == live_t {
                        // Same candle period — update OHLCV with live data
                        last.h = live_h.max(last.h);
                        last.l = if last.l > 0.0 { live_l.min(last.l) } else { live_l };
                        last.c = live_c;
                        last.v = live_v;
                    } else if live_t > last.t {
                        // New forming candle — append
                        candles.push(CandleResponse {
                            t: live_t, o: live_o, h: live_h, l: live_l, c: live_c, v: live_v,
                        });
                    }
                } else {
                    candles.push(CandleResponse {
                        t: live_t, o: live_o, h: live_h, l: live_l, c: live_c, v: live_v,
                    });
                }
            }
        } else if let Some(current_price) = fallback_price {
            // Fallback for 4h/1d: update close of the last candle with current 1m price
            if current_price > 0.0 {
                if let Some(last) = candles.last_mut() {
                    last.c = current_price;
                    // Also update high/low if current price extends the range
                    if current_price > last.h { last.h = current_price; }
                    if current_price < last.l { last.l = current_price; }
                }
            }
        }
    }

    Ok(Json(candles))
}

// ─── GET /api/indicators ──────────────────────────────────────────────
pub async fn get_indicators(
    Query(query): Query<IndicatorsQuery>,
    State(state): State<AppState>,
) -> Result<Json<Vec<IndicatorResponse>>, StatusCode> {
    let column = match query.indicator_type.as_str() {
        "ema" | "ema20" | "ema_20"   => "ema_20",
        "ema50" | "ema_50"           => "ema_50",
        "ema200" | "ema_200"         => "ema_200",
        "rsi"                        => "rsi",
        "macd"                       => "macd",
        "macd_signal"                => "macd_signal",
        "macd_hist"                  => "macd_hist",
        "adx"                        => "adx",
        "cci"                        => "cci",
        "atr"                        => "atr",
        "bb_upper"                   => "bb_upper",
        "bb_mid"                     => "bb_mid",
        "bb_lower"                   => "bb_lower",
        "stoch_k"                    => "stoch_k",
        "stoch_d"                    => "stoch_d",
        "obv"                        => "obv",
        "vwap"                       => "vwap",
        "volume_spike"               => "volume_spike",
        "sma"                        => "sma",
        _                            => "rsi",
    };

    let limit = query.limit.unwrap_or(500).min(5000);

    let sql = format!(
        "SELECT time_ms as t, {}::double precision as value
         FROM market.indicators_wide
         WHERE symbol = $1 AND tf_minutes = $2
         ORDER BY time DESC
         LIMIT $3",
        column
    );

    let rows = sqlx::query(&sql)
        .bind(&query.pair)
        .bind(query.tf as i16)
        .bind(limit)
        .fetch_all(&state.db_pool)
        .await
        .map_err(|e| { tracing::error!("Indicators error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let indicators: Vec<IndicatorResponse> = rows.iter().rev().map(|r| IndicatorResponse {
        t: r.get("t"),
        value: r.try_get("value").ok(),
    }).collect();

    Ok(Json(indicators))
}

// ─── GET /api/signals ─────────────────────────────────────────────────
pub async fn get_signals(
    Query(query): Query<SignalsQuery>,
    State(state): State<AppState>,
) -> Result<Json<Vec<SignalRow>>, StatusCode> {
    let limit = query.limit.unwrap_or(100);

    let rows = sqlx::query(
        "SELECT
            time, time_ms, symbol, tf_minutes, side, combined_score,
            entry_price, sl_price, tp_price, strategy, p_super
         FROM trade.super_entry_signals
         WHERE ($1::text IS NULL OR symbol = $1)
           AND ($2::int IS NULL OR tf_minutes = $2)
         ORDER BY time DESC
         LIMIT $3"
    )
    .bind(&query.pair)
    .bind(query.tf)
    .bind(limit)
    .fetch_all(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("Signals error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let signals: Vec<SignalRow> = rows.iter().map(|r| {
        let time: chrono::DateTime<chrono::Utc> = r.get("time");
        let time_ms: i64 = r.get("time_ms");
        SignalRow {
            id: format!("{}", time_ms),
            time: time.to_rfc3339(),
            pair: r.get("symbol"),
            tf: r.get::<i16, _>("tf_minutes") as i32,
            side: r.get("side"),
            score: r.get("combined_score"),
            entry_price: r.get("entry_price"),
            sl_price: r.get("sl_price"),
            tp_price: r.get("tp_price"),
            strategy: r.try_get("strategy").unwrap_or("super_entry_v1".to_string()),
            status: "active".to_string(),
            p_super: r.get("p_super"),
        }
    }).collect();

    Ok(Json(signals))
}

// ─── GET /api/positions/open ───────────────────────────────────────────
pub async fn get_open_positions(
    State(state): State<AppState>,
) -> Result<Json<Vec<PositionRow>>, StatusCode> {
    // FIX #6: Calculate pnl_pct dynamically.
    // FIX #11: Priority for current_price:
    //   1. candles_live (real-time from WebSocket ingestor, updated every ~200ms)
    //   2. positions.current_price (from position_tracker, updated every 2-3s)
    //   3. entry_price (fallback — only if nothing else available)
    //
    // PERF: Replaced LEFT JOIN LATERAL (N+1 subqueries) with a single CTE
    // that uses DISTINCT ON to get the latest 1m candle per symbol in one pass.
    let rows = sqlx::query(
        "WITH latest_prices AS (
             SELECT DISTINCT ON (symbol) symbol, close
             FROM market.candles_live
             WHERE timeframe = '1m'
             ORDER BY symbol, open_time_ms DESC
         )
         SELECT p.id,
                mp.symbol as pair,
                CASE WHEN p.side = 1 THEN 'LONG' ELSE 'SHORT' END as side,
                p.qty,
                p.entry_price,
                COALESCE(
                    lp.close,
                    p.current_price,
                    p.entry_price,
                    0
                ) as current_price,
                COALESCE(p.sl_price, 0) as stop_loss,
                COALESCE(p.tp_price, 0) as take_profit,
                COALESCE(p.unrealized_pnl, 0) as pnl_usdt,
                CASE
                    WHEN p.entry_price > 0 AND p.qty > 0 THEN
                        CASE WHEN p.side = 1
                            THEN (COALESCE(lp.close, p.current_price, p.entry_price) - p.entry_price) / p.entry_price * 100.0
                            ELSE (p.entry_price - COALESCE(lp.close, p.current_price, p.entry_price)) / p.entry_price * 100.0
                        END
                    ELSE 0
                END::double precision as pnl_pct,
                COALESCE(p.candles_left, 0)::smallint as candles_left,
                COALESCE(p.leverage, 10)::smallint as leverage,
                COALESCE(p.tf_minutes, 60)::smallint as tf_minutes,
                p.opened_at as open_time
         FROM trade.positions p
         JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
         LEFT JOIN latest_prices lp ON lp.symbol = mp.symbol
         WHERE p.status = 1
         ORDER BY p.opened_at DESC"
    )
    .fetch_all(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("Positions error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let positions: Vec<PositionRow> = rows.iter().map(|r| {
        let open_time: chrono::DateTime<chrono::Utc> = r.get("open_time");
        let leverage: i16 = r.try_get("leverage").unwrap_or(10);
        PositionRow {
            id: r.get("id"),
            pair: r.get("pair"),
            side: r.get("side"),
            qty: r.get("qty"),
            entry_price: r.get("entry_price"),
            current_price: r.get("current_price"),
            stop_loss: r.get("stop_loss"),
            take_profit: r.get("take_profit"),
            pnl_usdt: r.get("pnl_usdt"),
            pnl_pct: r.get("pnl_pct"),
            status: "open".to_string(),
            open_time: open_time.to_rfc3339(),
            candles_left: r.get("candles_left"),
            leverage: leverage as u16,
            tf_minutes: r.get("tf_minutes"),
        }
    }).collect();

    Ok(Json(positions))
}

// ─── GET /api/positions/history ────────────────────────────────────────
pub async fn get_positions_history(
    State(state): State<AppState>,
) -> Result<Json<Vec<HistoryRow>>, StatusCode> {
    let rows = sqlx::query(
        "SELECT id, position_id, symbol, side, qty, entry_price,
                COALESCE(exit_price, entry_price) as close_price,
                COALESCE(close_reason, 'manual') as close_type,
                COALESCE(realized_pnl, 0) as pnl_usdt,
                COALESCE(realized_pnl_pct, 0) as pnl_pct,
                opened_at as open_time,
                closed_at as close_time
         FROM trade.position_history
         ORDER BY closed_at DESC
         LIMIT 500"
    )
    .fetch_all(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("History error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let history: Vec<HistoryRow> = rows.iter().map(|r| {
        let open_time: chrono::DateTime<chrono::Utc> = r.get("open_time");
        let close_time: chrono::DateTime<chrono::Utc> = r.get("close_time");
        let side_i16: i16 = r.get("side");
        HistoryRow {
            id: r.get("id"),
            pair: r.get("symbol"),
            side: if side_i16 == 1 { "LONG".to_string() } else { "SHORT".to_string() },
            qty: r.get("qty"),
            entry_price: r.get("entry_price"),
            close_price: r.get("close_price"),
            close_type: r.get("close_type"),
            pnl_usdt: r.get("pnl_usdt"),
            pnl_pct: r.get("pnl_pct"),
            open_time: open_time.to_rfc3339(),
            close_time: close_time.to_rfc3339(),
        }
    }).collect();

    Ok(Json(history))
}

// ─── GET /api/statistics ──────────────────────────────────────────────
pub async fn get_statistics(
    Query(query): Query<StatsQuery>,
    State(state): State<AppState>,
) -> Result<Json<FullStatistics>, StatusCode> {
    let range = query.range.as_deref().unwrap_or("30d");
    let interval = query.interval.as_deref().unwrap_or("daily");

    // Build date range
    let (start_date, end_date) = match range {
        "24h" => ("NOW() - INTERVAL '24 hours'", "NOW()"),
        "7d" => ("NOW() - INTERVAL '7 days'", "NOW()"),
        "30d" => ("NOW() - INTERVAL '30 days'", "NOW()"),
        "90d" => ("NOW() - INTERVAL '90 days'", "NOW()"),
        "all" => ("'2024-01-01'::timestamptz", "NOW()"),
        _ => ("NOW() - INTERVAL '30 days'", "NOW()"),
    };

    // Use position_history if available, fallback to closed positions from trade.positions
    // This ensures we get data from both sources
    let history_table = format!(
        "(SELECT id, position_id, symbol, symbol_id, side, qty, entry_price, exit_price,
                leverage, tf_minutes, sl_price, tp_price, realized_pnl, realized_pnl_pct,
                combined_score, p_super, close_reason, opened_at, closed_at
         FROM trade.position_history
         WHERE closed_at BETWEEN {} AND {}
         UNION ALL
         SELECT id, id as position_id, mp.symbol as symbol, p.symbol_id, side, qty,
                entry_price, COALESCE(exit_price, current_price) as exit_price,
                leverage, tf_minutes, sl_price, tp_price,
                COALESCE(realized_pnl, unrealized_pnl, 0) as realized_pnl,
                COALESCE(realized_pnl_pct, 0) as realized_pnl_pct,
                combined_score, p_super, COALESCE(close_reason, 'manual') as close_reason,
                opened_at, COALESCE(closed_at, now()) as closed_at
         FROM trade.positions p
         JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
         WHERE p.status = 2 AND COALESCE(closed_at, now()) BETWEEN {} AND {}
           AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
        ) as history_data",
        start_date, end_date, start_date, end_date
    );

    // ─── SUMMARY STATISTICS ────────────────────────────────────────────
    let summary_row = sqlx::query(&format!(
        "SELECT
            COUNT(*) as total_trades,
            COUNT(*) FILTER (WHERE realized_pnl > 0) as winning_trades,
            COUNT(*) FILTER (WHERE realized_pnl <= 0) as losing_trades,
            COALESCE(SUM(realized_pnl), 0) as total_pnl,
            COALESCE(AVG(realized_pnl) FILTER (WHERE realized_pnl > 0), 0) as avg_win,
            COALESCE(AVG(realized_pnl) FILTER (WHERE realized_pnl <= 0), 0) as avg_loss,
            COALESCE(MAX(realized_pnl), 0) as best_trade,
            COALESCE(MIN(realized_pnl), 0) as worst_trade,
            COALESCE(AVG(EXTRACT(EPOCH FROM (closed_at - opened_at)) / 3600)::float8, 0) as avg_duration_hours
         FROM {}
         WHERE closed_at BETWEEN {} AND {}",
        history_table, start_date, end_date
    ))
    .fetch_one(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("Stats summary error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let total_trades: i64 = summary_row.get("total_trades");
    let winning_trades: i64 = summary_row.get("winning_trades");
    let losing_trades: i64 = summary_row.get("losing_trades");
    let total_pnl: f64 = summary_row.get("total_pnl");
    let avg_win: f64 = summary_row.get("avg_win");
    let avg_loss: f64 = summary_row.get("avg_loss");
    let best_trade: f64 = summary_row.get("best_trade");
    let worst_trade: f64 = summary_row.get("worst_trade");
    let avg_duration_hours: f64 = summary_row.get("avg_duration_hours");
    
    let win_rate = if total_trades > 0 { (winning_trades as f64 / total_trades as f64) * 100.0 } else { 0.0 };
    let gross_profit = if winning_trades > 0 { avg_win * winning_trades as f64 } else { 0.0 };
    let gross_loss = if losing_trades > 0 { avg_loss.abs() * losing_trades as f64 } else { 0.0 };
    let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else { if gross_profit > 0.0 { f64::INFINITY } else { 0.0 } };

    // Get balance for % calculations
    let balance_for_pct = get_binance_balance_internal().await.wallet_balance.max(1.0);

    // % calculations
    let total_pnl_pct = (total_pnl / balance_for_pct) * 100.0;
    let avg_win_pct = (avg_win / balance_for_pct) * 100.0;
    let avg_loss_pct = (avg_loss / balance_for_pct) * 100.0;
    let best_trade_pct = (best_trade / balance_for_pct) * 100.0;
    let worst_trade_pct = (worst_trade / balance_for_pct) * 100.0;

    // Expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    let expectancy = (win_rate / 100.0) * avg_win + (1.0 - win_rate / 100.0) * avg_loss;

    // Rough Sharpe approximation: expectancy / stddev(trade_pnl)
    let sharpe_approx = calculate_sharpe_approx(&state.db_pool, start_date, end_date, expectancy).await.unwrap_or(0.0);

    // Calculate max consecutive wins/losses
    let (max_consecutive_wins, max_consecutive_losses) = calculate_consecutive(&state.db_pool, start_date, end_date).await.unwrap_or((0, 0));

    let summary = StatsSummary {
        total_trades,
        winning_trades,
        losing_trades,
        win_rate,
        total_pnl,
        total_pnl_pct,
        avg_win,
        avg_win_pct,
        avg_loss,
        avg_loss_pct,
        profit_factor,
        best_trade,
        best_trade_pct,
        worst_trade,
        worst_trade_pct,
        avg_trade_duration_hours: avg_duration_hours,
        max_consecutive_wins,
        max_consecutive_losses,
        expectancy,
        sharpe_approx,
    };

    // ─── PNL TIMELINE ──────────────────────────────────────────────────
    let timeline = calculate_pnl_timeline(&state.db_pool, start_date, end_date, interval).await?;

    // ─── DISTRIBUTION ──────────────────────────────────────────────────
    let distribution = calculate_distribution(&state.db_pool, start_date, end_date).await?;

    // ─── BEST/WORST TRADES ─────────────────────────────────────────────
    let best_worst = calculate_best_worst(&state.db_pool, start_date, end_date).await?;

    // ─── TIME-BASED STATS (hourly, daily, weekly) ──────────────────────
    let hourly = calculate_hourly_stats(&state.db_pool, start_date, end_date).await?;
    let daily = calculate_daily_stats(&state.db_pool, start_date, end_date).await?;
    let weekly = calculate_weekly_stats(&state.db_pool, start_date, end_date).await?;
    let monthly = calculate_monthly_stats(&state.db_pool, start_date, end_date).await?;

    // ─── DAY PART STATS (Morning/Day/Evening/Night) ────────────────────
    let day_parts = calculate_day_part_stats(&state.db_pool, start_date, end_date, balance_for_pct).await?;

    Ok(Json(FullStatistics {
        summary,
        timeline,
        distribution,
        best_worst,
        hourly,
        daily,
        weekly,
        monthly,
        day_parts,
    }))
}

async fn calculate_consecutive(
    pool: &sqlx::PgPool,
    start_date: &str,
    end_date: &str,
) -> Result<(i64, i64), StatusCode> {
    let rows = sqlx::query(&format!(
        "SELECT CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END as is_win
         FROM (
           SELECT realized_pnl, closed_at FROM trade.position_history WHERE closed_at BETWEEN {0} AND {1}
           UNION ALL
           SELECT COALESCE(realized_pnl, unrealized_pnl, 0) as realized_pnl, COALESCE(closed_at, now()) as closed_at
           FROM trade.positions p
           JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
           WHERE p.status = 2 AND COALESCE(closed_at, now()) BETWEEN {0} AND {1}
             AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
         ) as all_history
         ORDER BY closed_at ASC",
        start_date, end_date
    ))
    .fetch_all(pool)
    .await
    .map_err(|e| { tracing::error!("Consecutive error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let mut max_wins = 0i64;
    let mut max_losses = 0i64;
    let mut cur_wins = 0i64;
    let mut cur_losses = 0i64;

    for row in rows {
        let is_win: i32 = row.get("is_win");
        if is_win == 1 {
            cur_wins += 1;
            cur_losses = 0;
            max_wins = max_wins.max(cur_wins);
        } else {
            cur_losses += 1;
            cur_wins = 0;
            max_losses = max_losses.max(cur_losses);
        }
    }

    Ok((max_wins, max_losses))
}

async fn calculate_pnl_timeline(
    pool: &sqlx::PgPool,
    start_date: &str,
    end_date: &str,
    interval: &str,
) -> Result<PnlTimeline, StatusCode> {
    let date_trunc = match interval {
        "hourly" => "hour",
        "daily" => "day",
        "weekly" => "week",
        _ => "day",
    };

    let rows = sqlx::query(&format!(
        "SELECT
            DATE_TRUNC('{}', closed_at) as period,
            COALESCE(SUM(realized_pnl), 0) as period_pnl,
            COALESCE(SUM(SUM(realized_pnl)) OVER (ORDER BY DATE_TRUNC('{}', closed_at)), 0) as cumulative_pnl
         FROM (
           SELECT realized_pnl, closed_at FROM trade.position_history WHERE closed_at BETWEEN {} AND {}
           UNION ALL
           SELECT COALESCE(realized_pnl, unrealized_pnl, 0) as realized_pnl, COALESCE(closed_at, now()) as closed_at
           FROM trade.positions p
           JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
           WHERE p.status = 2 AND COALESCE(closed_at, now()) BETWEEN {} AND {}
             AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
         ) as all_history
         GROUP BY DATE_TRUNC('{}', closed_at)
         ORDER BY period ASC",
        date_trunc, date_trunc, start_date, end_date, start_date, end_date, date_trunc
    ))
    .fetch_all(pool)
    .await
    .map_err(|e| { tracing::error!("Timeline error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let mut points = Vec::new();
    let mut cumulative = 0.0;
    let mut start_str = String::new();
    let mut end_str = String::new();

    for row in rows {
        let period: chrono::DateTime<chrono::Utc> = row.get("period");
        let period_pnl: f64 = row.get("period_pnl");
        cumulative += period_pnl;
        
        if start_str.is_empty() {
            start_str = period.to_rfc3339();
        }
        end_str = period.to_rfc3339();

        points.push(PnlTimePoint {
            t: period.timestamp_millis(),
            cumulative_pnl: cumulative,
            balance: cumulative,
        });
    }

    Ok(PnlTimeline {
        points,
        start_date: start_str,
        end_date: end_str,
    })
}

async fn calculate_distribution(
    pool: &sqlx::PgPool,
    start_date: &str,
    end_date: &str,
) -> Result<TradeDistribution, StatusCode> {
    // By pair
    let pair_rows = sqlx::query(&format!(
        "SELECT
            pair,
            COUNT(*) as trades,
            COUNT(*) FILTER (WHERE realized_pnl > 0) as wins,
            COALESCE(SUM(realized_pnl), 0) as pnl
         FROM (
           SELECT symbol as pair, realized_pnl FROM trade.position_history WHERE closed_at BETWEEN {0} AND {1}
           UNION ALL
           SELECT mp.symbol as pair, COALESCE(p.realized_pnl, p.unrealized_pnl, 0) as realized_pnl
           FROM trade.positions p
           JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
           WHERE p.status = 2 AND COALESCE(p.closed_at, now()) BETWEEN {0} AND {1}
             AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
         ) as all_history
         GROUP BY pair
         ORDER BY pnl DESC",
        start_date, end_date
    ))
    .fetch_all(pool)
    .await
    .map_err(|e| { tracing::error!("Pair stats error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let by_pair: Vec<PairStats> = pair_rows.iter().map(|r| {
        let trades: i64 = r.get("trades");
        let wins: i64 = r.get("wins");
        PairStats {
            pair: r.get("pair"),
            trades,
            wins,
            pnl: r.get("pnl"),
            win_rate: if trades > 0 { (wins as f64 / trades as f64) * 100.0 } else { 0.0 },
        }
    }).collect();

    // By timeframe
    let tf_rows = sqlx::query(&format!(
        "SELECT
            tf,
            COUNT(*) as trades,
            COUNT(*) FILTER (WHERE realized_pnl > 0) as wins,
            COALESCE(SUM(realized_pnl), 0) as pnl
         FROM (
           SELECT tf_minutes as tf, realized_pnl FROM trade.position_history WHERE closed_at BETWEEN {0} AND {1}
           UNION ALL
           SELECT p.tf_minutes as tf, COALESCE(p.realized_pnl, p.unrealized_pnl, 0) as realized_pnl
           FROM trade.positions p
           JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
           WHERE p.status = 2 AND COALESCE(p.closed_at, now()) BETWEEN {0} AND {1}
             AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
         ) as all_history
         GROUP BY tf
         ORDER BY tf ASC",
        start_date, end_date
    ))
    .fetch_all(pool)
    .await
    .map_err(|e| { tracing::error!("TF stats error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let by_timeframe: Vec<TfStats> = tf_rows.iter().map(|r| {
        let trades: i64 = r.get("trades");
        let wins: i64 = r.get("wins");
        TfStats {
            tf: r.get("tf"),
            trades,
            wins,
            pnl: r.get("pnl"),
            win_rate: if trades > 0 { (wins as f64 / trades as f64) * 100.0 } else { 0.0 },
        }
    }).collect();

    // By close type
    let close_type_rows = sqlx::query(&format!(
        "SELECT
            close_type,
            COUNT(*) as count,
            COALESCE(SUM(realized_pnl), 0) as pnl,
            COUNT(*) FILTER (WHERE realized_pnl > 0) as wins
         FROM (
           SELECT close_reason as close_type, realized_pnl FROM trade.position_history WHERE closed_at BETWEEN {0} AND {1}
           UNION ALL
           SELECT COALESCE(p.close_reason, 'manual') as close_type, COALESCE(p.realized_pnl, p.unrealized_pnl, 0) as realized_pnl
           FROM trade.positions p
           JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
           WHERE p.status = 2 AND COALESCE(p.closed_at, now()) BETWEEN {0} AND {1}
             AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
         ) as all_history
         GROUP BY close_type
         ORDER BY count DESC",
        start_date, end_date
    ))
    .fetch_all(pool)
    .await
    .map_err(|e| { tracing::error!("Close type stats error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let by_close_type: Vec<CloseTypeStats> = close_type_rows.iter().map(|r| {
        let count: i64 = r.get("count");
        let wins: i64 = r.get("wins");
        CloseTypeStats {
            close_type: r.get("close_type"),
            count,
            pnl: r.get("pnl"),
            win_rate: if count > 0 { (wins as f64 / count as f64) * 100.0 } else { 0.0 },
        }
    }).collect();

    // By side
    let side_rows = sqlx::query(&format!(
        "SELECT
            side,
            COUNT(*) as trades,
            COUNT(*) FILTER (WHERE realized_pnl > 0) as wins,
            COALESCE(SUM(realized_pnl), 0) as pnl
         FROM (
           SELECT CASE WHEN side = 1 THEN 'LONG' ELSE 'SHORT' END as side, realized_pnl FROM trade.position_history WHERE closed_at BETWEEN {0} AND {1}
           UNION ALL
           SELECT CASE WHEN p.side = 1 THEN 'LONG' ELSE 'SHORT' END as side, COALESCE(p.realized_pnl, p.unrealized_pnl, 0) as realized_pnl
           FROM trade.positions p
           JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
           WHERE p.status = 2 AND COALESCE(p.closed_at, now()) BETWEEN {0} AND {1}
             AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
         ) as all_history
         GROUP BY side
         ORDER BY pnl DESC",
        start_date, end_date
    ))
    .fetch_all(pool)
    .await
    .map_err(|e| { tracing::error!("Side stats error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let by_side: Vec<SideStats> = side_rows.iter().map(|r| {
        let trades: i64 = r.get("trades");
        let wins: i64 = r.get("wins");
        SideStats {
            side: r.get("side"),
            trades,
            wins,
            pnl: r.get("pnl"),
            win_rate: if trades > 0 { (wins as f64 / trades as f64) * 100.0 } else { 0.0 },
        }
    }).collect();

    Ok(TradeDistribution {
        by_pair,
        by_timeframe,
        by_close_type,
        by_side,
    })
}

async fn calculate_best_worst(
    pool: &sqlx::PgPool,
    start_date: &str,
    end_date: &str,
) -> Result<BestWorstTrades, StatusCode> {
    let best_rows = sqlx::query(&format!(
        "SELECT id, position_id, symbol, side, qty, entry_price,
                COALESCE(exit_price, entry_price) as close_price,
                COALESCE(close_reason, 'manual') as close_type,
                COALESCE(realized_pnl, 0) as pnl_usdt,
                COALESCE(realized_pnl_pct, 0) as pnl_pct,
                opened_at as open_time,
                closed_at as close_time
         FROM (
           SELECT id, position_id, symbol, side, qty, entry_price, exit_price, close_reason, realized_pnl, realized_pnl_pct, opened_at, closed_at
           FROM trade.position_history WHERE closed_at BETWEEN {0} AND {1}
           UNION ALL
           SELECT p.id, p.id as position_id, mp.symbol as symbol, p.side, p.qty, p.entry_price,
                  COALESCE(p.exit_price, p.current_price) as exit_price,
                  COALESCE(p.close_reason, 'manual') as close_type,
                  COALESCE(p.realized_pnl, p.unrealized_pnl, 0) as pnl_usdt,
                  COALESCE(p.realized_pnl_pct, 0) as pnl_pct,
                  p.opened_at, COALESCE(p.closed_at, now()) as close_time
           FROM trade.positions p
           JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
           WHERE p.status = 2 AND COALESCE(p.closed_at, now()) BETWEEN {0} AND {1}
             AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
         ) as all_history
         ORDER BY pnl_usdt DESC
         LIMIT 5",
        start_date, end_date
    ))
    .fetch_all(pool)
    .await
    .map_err(|e| { tracing::error!("Best trades error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let worst_rows = sqlx::query(&format!(
        "SELECT id, position_id, symbol, side, qty, entry_price,
                COALESCE(exit_price, entry_price) as close_price,
                COALESCE(close_reason, 'manual') as close_type,
                COALESCE(realized_pnl, 0) as pnl_usdt,
                COALESCE(realized_pnl_pct, 0) as pnl_pct,
                opened_at as open_time,
                closed_at as close_time
         FROM (
           SELECT id, position_id, symbol, side, qty, entry_price, exit_price, close_reason, realized_pnl, realized_pnl_pct, opened_at, closed_at
           FROM trade.position_history WHERE closed_at BETWEEN {0} AND {1}
           UNION ALL
           SELECT p.id, p.id as position_id, mp.symbol as symbol, p.side, p.qty, p.entry_price,
                  COALESCE(p.exit_price, p.current_price) as exit_price,
                  COALESCE(p.close_reason, 'manual') as close_type,
                  COALESCE(p.realized_pnl, p.unrealized_pnl, 0) as pnl_usdt,
                  COALESCE(p.realized_pnl_pct, 0) as pnl_pct,
                  p.opened_at, COALESCE(p.closed_at, now()) as close_time
           FROM trade.positions p
           JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
           WHERE p.status = 2 AND COALESCE(p.closed_at, now()) BETWEEN {0} AND {1}
             AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
         ) as all_history
         ORDER BY pnl_usdt ASC
         LIMIT 5",
        start_date, end_date
    ))
    .fetch_all(pool)
    .await
    .map_err(|e| { tracing::error!("Worst trades error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let parse_row = |r: &sqlx::postgres::PgRow| -> HistoryRow {
        let open_time: chrono::DateTime<chrono::Utc> = r.get("open_time");
        let close_time: chrono::DateTime<chrono::Utc> = r.get("close_time");
        let side_i16: i16 = r.get("side");
        HistoryRow {
            id: r.get("id"),
            pair: r.get("symbol"),
            side: if side_i16 == 1 { "LONG".to_string() } else { "SHORT".to_string() },
            qty: r.get("qty"),
            entry_price: r.get("entry_price"),
            close_price: r.get("close_price"),
            close_type: r.get("close_type"),
            pnl_usdt: r.get("pnl_usdt"),
            pnl_pct: r.get("pnl_pct"),
            open_time: open_time.to_rfc3339(),
            close_time: close_time.to_rfc3339(),
        }
    };

    Ok(BestWorstTrades {
        best: best_rows.iter().map(parse_row).collect(),
        worst: worst_rows.iter().map(parse_row).collect(),
    })
}

async fn calculate_hourly_stats(
    pool: &sqlx::PgPool,
    start_date: &str,
    end_date: &str,
) -> Result<TimeBasedStats, StatusCode> {
    let rows = sqlx::query(&format!(
        "SELECT
            EXTRACT(HOUR FROM closed_at)::bigint as hour,
            COUNT(*) as trades,
            COUNT(*) FILTER (WHERE realized_pnl > 0) as wins,
            COUNT(*) FILTER (WHERE realized_pnl <= 0) as losses,
            COALESCE(SUM(realized_pnl), 0) as pnl,
            COALESCE(AVG(realized_pnl), 0) as avg_pnl
         FROM (
           SELECT realized_pnl, closed_at FROM trade.position_history WHERE closed_at BETWEEN {0} AND {1}
           UNION ALL
           SELECT COALESCE(p.realized_pnl, p.unrealized_pnl, 0) as realized_pnl, COALESCE(p.closed_at, now()) as closed_at
           FROM trade.positions p
           JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
           WHERE p.status = 2 AND COALESCE(p.closed_at, now()) BETWEEN {0} AND {1}
             AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
         ) as all_history
         GROUP BY EXTRACT(HOUR FROM closed_at)
         ORDER BY hour ASC",
        start_date, end_date
    ))
    .fetch_all(pool)
    .await
    .map_err(|e| { tracing::error!("Hourly stats error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let data: Vec<TimeSlotStats> = rows.iter().map(|r| {
        let trades: i64 = r.get("trades");
        let wins: i64 = r.get("wins");
        let losses: i64 = r.get("losses");
        let hour: i64 = r.get("hour");
        TimeSlotStats {
            label: format!("{:02}:00", hour),
            trades,
            wins,
            losses,
            pnl: r.get("pnl"),
            win_rate: if trades > 0 { (wins as f64 / trades as f64) * 100.0 } else { 0.0 },
            avg_pnl: r.get("avg_pnl"),
        }
    }).collect();

    Ok(TimeBasedStats {
        period: "hourly".to_string(),
        data,
    })
}

async fn calculate_daily_stats(
    pool: &sqlx::PgPool,
    start_date: &str,
    end_date: &str,
) -> Result<TimeBasedStats, StatusCode> {
    let rows = sqlx::query(&format!(
        "SELECT
            TO_CHAR(closed_at, 'Day') as day_name,
            EXTRACT(DOW FROM closed_at)::bigint as day_num,
            COUNT(*) as trades,
            COUNT(*) FILTER (WHERE realized_pnl > 0) as wins,
            COUNT(*) FILTER (WHERE realized_pnl <= 0) as losses,
            COALESCE(SUM(realized_pnl), 0) as pnl,
            COALESCE(AVG(realized_pnl), 0) as avg_pnl
         FROM (
           SELECT realized_pnl, closed_at FROM trade.position_history WHERE closed_at BETWEEN {0} AND {1}
           UNION ALL
           SELECT COALESCE(p.realized_pnl, p.unrealized_pnl, 0) as realized_pnl, COALESCE(p.closed_at, now()) as closed_at
           FROM trade.positions p
           JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
           WHERE p.status = 2 AND COALESCE(p.closed_at, now()) BETWEEN {0} AND {1}
             AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
         ) as all_history
         GROUP BY EXTRACT(DOW FROM closed_at), TO_CHAR(closed_at, 'Day')
         ORDER BY day_num ASC",
        start_date, end_date
    ))
    .fetch_all(pool)
    .await
    .map_err(|e| { tracing::error!("Daily stats error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let data: Vec<TimeSlotStats> = rows.iter().map(|r| {
        let trades: i64 = r.get("trades");
        let wins: i64 = r.get("wins");
        let losses: i64 = r.get("losses");
        let day_name: String = r.get("day_name");
        TimeSlotStats {
            label: day_name.trim().to_string(),
            trades,
            wins,
            losses,
            pnl: r.get("pnl"),
            win_rate: if trades > 0 { (wins as f64 / trades as f64) * 100.0 } else { 0.0 },
            avg_pnl: r.get("avg_pnl"),
        }
    }).collect();

    Ok(TimeBasedStats {
        period: "daily".to_string(),
        data,
    })
}

async fn calculate_weekly_stats(
    pool: &sqlx::PgPool,
    start_date: &str,
    end_date: &str,
) -> Result<TimeBasedStats, StatusCode> {
    let rows = sqlx::query(&format!(
        "SELECT
            TO_CHAR(closed_at, 'IYYY-IW') as week,
            COUNT(*) as trades,
            COUNT(*) FILTER (WHERE realized_pnl > 0) as wins,
            COUNT(*) FILTER (WHERE realized_pnl <= 0) as losses,
            COALESCE(SUM(realized_pnl), 0) as pnl,
            COALESCE(AVG(realized_pnl), 0) as avg_pnl
         FROM (
           SELECT realized_pnl, closed_at FROM trade.position_history WHERE closed_at BETWEEN {0} AND {1}
           UNION ALL
           SELECT COALESCE(p.realized_pnl, p.unrealized_pnl, 0) as realized_pnl, COALESCE(p.closed_at, now()) as closed_at
           FROM trade.positions p
           JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
           WHERE p.status = 2 AND COALESCE(p.closed_at, now()) BETWEEN {0} AND {1}
             AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
         ) as all_history
         GROUP BY TO_CHAR(closed_at, 'IYYY-IW')
         ORDER BY week ASC",
        start_date, end_date
    ))
    .fetch_all(pool)
    .await
    .map_err(|e| { tracing::error!("Weekly stats error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let data: Vec<TimeSlotStats> = rows.iter().map(|r| {
        let trades: i64 = r.get("trades");
        let wins: i64 = r.get("wins");
        let losses: i64 = r.get("losses");
        let week: String = r.get("week");
        TimeSlotStats {
            label: format!("Week {}", week),
            trades,
            wins,
            losses,
            pnl: r.get("pnl"),
            win_rate: if trades > 0 { (wins as f64 / trades as f64) * 100.0 } else { 0.0 },
            avg_pnl: r.get("avg_pnl"),
        }
    }).collect();

    Ok(TimeBasedStats {
        period: "weekly".to_string(),
        data,
    })
}

async fn calculate_monthly_stats(
    pool: &sqlx::PgPool,
    start_date: &str,
    end_date: &str,
) -> Result<Vec<MonthlyStats>, StatusCode> {
    let rows = sqlx::query(&format!(
        "SELECT
            TO_CHAR(closed_at, 'YYYY-MM') as month,
            COUNT(*) as trades,
            COUNT(*) FILTER (WHERE realized_pnl > 0) as wins,
            COALESCE(SUM(realized_pnl), 0) as pnl
         FROM (
           SELECT realized_pnl, closed_at FROM trade.position_history WHERE closed_at BETWEEN {0} AND {1}
           UNION ALL
           SELECT COALESCE(realized_pnl, unrealized_pnl, 0) as realized_pnl, COALESCE(closed_at, now()) as closed_at
           FROM trade.positions p
           JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
           WHERE p.status = 2 AND COALESCE(closed_at, now()) BETWEEN {0} AND {1}
             AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
         ) as all_history
         GROUP BY TO_CHAR(closed_at, 'YYYY-MM')
         ORDER BY month DESC",
        start_date, end_date
    ))
    .fetch_all(pool)
    .await
    .map_err(|e| { tracing::error!("Monthly stats error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let monthly: Vec<MonthlyStats> = rows.iter().map(|r| {
        let trades: i64 = r.get("trades");
        let wins: i64 = r.get("wins");
        MonthlyStats {
            month: r.get("month"),
            trades,
            wins,
            pnl: r.get("pnl"),
            win_rate: if trades > 0 { (wins as f64 / trades as f64) * 100.0 } else { 0.0 },
        }
    }).collect();

    Ok(monthly)
}

/// Calculate day-part (Morning/Day/Evening/Night) statistics.
/// Buckets: Morning=06-12, Day=12-18, Evening=18-00, Night=00-06 (UTC).
async fn calculate_day_part_stats(
    pool: &sqlx::PgPool,
    start_date: &str,
    end_date: &str,
    balance: f64,
) -> Result<Vec<DayPartStats>, StatusCode> {
    let rows = sqlx::query(&format!(
        "SELECT * FROM (
            SELECT
                CASE
                    WHEN EXTRACT(HOUR FROM closed_at) >= 6  AND EXTRACT(HOUR FROM closed_at) < 12 THEN 'Morning'
                    WHEN EXTRACT(HOUR FROM closed_at) >= 12 AND EXTRACT(HOUR FROM closed_at) < 18 THEN 'Day'
                    WHEN EXTRACT(HOUR FROM closed_at) >= 18 THEN 'Evening'
                    ELSE 'Night'
                END as bucket,
                COUNT(*) as trades,
                COUNT(*) FILTER (WHERE realized_pnl > 0) as wins,
                COUNT(*) FILTER (WHERE realized_pnl <= 0) as losses,
                COALESCE(SUM(realized_pnl), 0) as pnl,
                COALESCE(AVG(realized_pnl), 0) as avg_pnl,
                COALESCE(MAX(realized_pnl), 0) as best_trade,
                COALESCE(MIN(realized_pnl), 0) as worst_trade
             FROM (
               SELECT realized_pnl, closed_at FROM trade.position_history WHERE closed_at BETWEEN {0} AND {1}
               UNION ALL
               SELECT COALESCE(p.realized_pnl, p.unrealized_pnl, 0) as realized_pnl, COALESCE(p.closed_at, now()) as closed_at
               FROM trade.positions p
               JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
               WHERE p.status = 2 AND COALESCE(p.closed_at, now()) BETWEEN {0} AND {1}
                 AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
             ) as all_history
             GROUP BY 1
        ) sub
        ORDER BY
            CASE sub.bucket
                WHEN 'Morning' THEN 1
                WHEN 'Day' THEN 2
                WHEN 'Evening' THEN 3
                WHEN 'Night' THEN 4
            END",
        start_date, end_date
    ))
    .fetch_all(pool)
    .await
    .map_err(|e| { tracing::error!("Day part stats error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let hours_map: std::collections::HashMap<&str, &str> = [
        ("Morning", "06:00–12:00"),
        ("Day", "12:00–18:00"),
        ("Evening", "18:00–00:00"),
        ("Night", "00:00–06:00"),
    ].into_iter().collect();

    let data: Vec<DayPartStats> = rows.iter().map(|r| {
        let trades: i64 = r.get("trades");
        let wins: i64 = r.get("wins");
        let losses: i64 = r.get("losses");
        let pnl: f64 = r.get("pnl");
        let bucket: String = r.get("bucket");
        DayPartStats {
            hours: hours_map.get(bucket.as_str()).unwrap_or(&"").to_string(),
            bucket,
            trades,
            wins,
            losses,
            pnl,
            pnl_pct: if balance > 0.0 { (pnl / balance) * 100.0 } else { 0.0 },
            win_rate: if trades > 0 { (wins as f64 / trades as f64) * 100.0 } else { 0.0 },
            avg_pnl: r.get("avg_pnl"),
            best_trade: r.get("best_trade"),
            worst_trade: r.get("worst_trade"),
        }
    }).collect();

    Ok(data)
}

/// Calculate rough Sharpe ratio approximation: expectancy / stddev(trade_pnl).
async fn calculate_sharpe_approx(
    pool: &sqlx::PgPool,
    start_date: &str,
    end_date: &str,
    expectancy: f64,
) -> Result<f64, StatusCode> {
    let row = sqlx::query(&format!(
        "SELECT COALESCE(STDDEV(realized_pnl), 0) as stddev_pnl
         FROM (
           SELECT realized_pnl FROM trade.position_history WHERE closed_at BETWEEN {0} AND {1}
           UNION ALL
           SELECT COALESCE(realized_pnl, unrealized_pnl, 0) as realized_pnl
           FROM trade.positions p
           JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
           WHERE p.status = 2 AND COALESCE(closed_at, now()) BETWEEN {0} AND {1}
             AND NOT EXISTS (SELECT 1 FROM trade.position_history ph WHERE ph.position_id = p.id)
         ) as all_history",
        start_date, end_date
    ))
    .fetch_one(pool)
    .await
    .map_err(|e| { tracing::error!("Sharpe error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let stddev: f64 = row.get("stddev_pnl");
    if stddev > 0.0 {
        Ok(expectancy / stddev)
    } else {
        Ok(0.0)
    }
}

// ─── GET /api/pnl/overview ─────────────────────────────────────────────
pub async fn get_pnl_overview(
    Query(_query): Query<PnlQuery>,
    State(state): State<AppState>,
) -> Result<Json<PnlOverview>, StatusCode> {
    // Closed PnL — real PnL without leverage effect
    // realized_pnl in position_history is already the real USDT PnL
    let row = sqlx::query(
        "SELECT
            COALESCE(SUM(realized_pnl), 0) as closed_pnl,
            COUNT(*) FILTER (WHERE realized_pnl > 0) as wins,
            COUNT(*) as total_trades
         FROM trade.position_history
         WHERE closed_at > NOW() - INTERVAL '30 days'"
    )
    .fetch_one(&state.db_pool)
    .await
    .map_err(|e| { tracing::error!("PnL error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;

    let closed_pnl: f64 = row.get("closed_pnl");
    let wins: i64 = row.get("wins");
    let total_trades: i64 = row.get("total_trades");
    let win_rate = if total_trades > 0 { (wins as f64 / total_trades as f64) * 100.0 } else { 0.0 };

    // Unrealized PnL from open positions (real PnL, not leveraged)
    // unrealized_pnl = direction * (current_price - entry_price) * qty
    // This is already in USDT without leverage amplification
    let unrealized_row = sqlx::query(
        "SELECT COALESCE(SUM(unrealized_pnl), 0) as unrealized
         FROM trade.positions WHERE status = 1"
    )
    .fetch_one(&state.db_pool)
    .await
    .ok();

    let unrealized_pnl = unrealized_row
        .and_then(|r| r.try_get::<f64, _>("unrealized").ok())
        .unwrap_or(0.0);

    // Get overall balance from Binance account (wallet_balance)
    // This is the real balance without leverage
    let balance_info = get_binance_balance_internal().await;
    let overall_balance = balance_info.wallet_balance;

    Ok(Json(PnlOverview {
        closed_pnl,
        unrealized_pnl,
        win_rate,
        today_trades: total_trades,
        equity_points: vec![],
        overall_balance,
    }))
}

// ─── GET /api/balance ──────────────────────────────────────────────────
pub async fn get_balance(
    State(state): State<AppState>,
) -> Result<Json<BalanceInfo>, StatusCode> {
    // Read real balance from Binance Futures account
    let mut balance = get_binance_balance_internal().await;

    // Also include in_orders from our DB (real margin used, not leveraged notional)
    let in_orders_row = sqlx::query(
        "SELECT COALESCE(SUM(qty * entry_price / GREATEST(COALESCE(leverage, 10), 1)), 0) as in_orders
         FROM trade.positions WHERE status = 1"
    )
    .fetch_optional(&state.db_pool)
    .await
    .ok()
    .flatten();

    if let Some(row) = in_orders_row {
        let db_in_orders: f64 = row.try_get("in_orders").unwrap_or(0.0);
        if db_in_orders > 0.0 {
            balance.in_orders = db_in_orders;
            balance.available = balance.wallet_balance - balance.in_orders;
        }
    }

    // Overall = wallet_balance (real deposit + realized PnL, WITHOUT leverage)
    balance.overall = balance.wallet_balance;

    Ok(Json(balance))
}

/// Internal function to get Binance balance
async fn get_binance_balance_internal() -> BalanceInfo {
    // Try to load credentials and fetch real balance
    match settings::ExchangeSettings::load() {
        Ok(exchange_settings) => {
            if exchange_settings.has_credentials() {
                let testnet = std::env::var("BINANCE_TESTNET")
                    .map(|v| v == "true" || v == "1")
                    .unwrap_or(false);
                match connections_lib::BinanceFuturesClient::new(
                    exchange_settings.api_key(),
                    exchange_settings.api_secret(),
                    testnet,
                ) {
                    Ok(client) => {
                        match client.account_info().await {
                            Ok(info) => {
                                let wallet_balance = info.total_wallet_balance
                                    .parse::<f64>().unwrap_or(0.0);
                                let unrealized_pnl = info.total_unrealized_profit
                                    .parse::<f64>().unwrap_or(0.0);
                                let available = info.available_balance
                                    .parse::<f64>().unwrap_or(0.0);

                                // Calculate in_orders = wallet - available
                                let in_orders = (wallet_balance - available).max(0.0);

                                return BalanceInfo {
                                    overall: wallet_balance,
                                    in_orders,
                                    available,
                                    wallet_balance,
                                    unrealized_pnl,
                                };
                            }
                            Err(e) => {
                                tracing::debug!("Binance account unavailable: {:?}", e);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::debug!("Failed to create Binance client: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            tracing::debug!("No exchange settings: {}", e);
        }
    }

    // Fallback — return zeros
    BalanceInfo {
        overall: 0.0,
        in_orders: 0.0,
        available: 0.0,
        wallet_balance: 0.0,
        unrealized_pnl: 0.0,
    }
}

// ─── GET /api/connections ──────────────────────────────────────────────
pub async fn get_connections(
    State(state): State<AppState>,
) -> Result<Json<ConnectionStatus>, StatusCode> {
    // Check all connections in parallel
    let db_ok = check_database(&state.db_pool).await;
    let (api_ok, ws_ok, account_ok) = check_binance().await;
    let redpanda_ok = check_redpanda().await;

    Ok(Json(ConnectionStatus {
        database: db_ok,
        redpanda: redpanda_ok,
        rest_api: api_ok,
        websocket: ws_ok,
        account: account_ok,
    }))
}

async fn check_database(pool: &sqlx::PgPool) -> bool {
    sqlx::query("SELECT 1")
        .fetch_one(pool)
        .await
        .is_ok()
}

async fn check_redpanda() -> bool {
    // Try TCP connect to Kafka brokers
    let brokers = std::env::var("KAFKA_BROKERS")
        .unwrap_or_else(|_| "127.0.0.1:19092".to_string());
    let addr = brokers.split(',').next().unwrap_or("127.0.0.1:19092");
    tokio::net::TcpStream::connect(addr)
        .await
        .is_ok()
}

async fn check_binance() -> (bool, bool, bool) {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    // REST API ping
    let rest_ok = client.get("https://fapi.binance.com/fapi/v1/ping")
        .send().await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    // WebSocket check (quick TCP/TLS handshake to fstream)
    let ws_ok = tokio::net::TcpStream::connect("fstream.binance.com:443")
        .await
        .is_ok();

    // Account check
    let account_ok = match settings::ExchangeSettings::load() {
        Ok(es) if es.has_credentials() => {
            let testnet = std::env::var("BINANCE_TESTNET")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false);
            match connections_lib::BinanceFuturesClient::new(es.api_key(), es.api_secret(), testnet) {
                Ok(c) => c.ping().await.is_ok(),
                Err(_) => false,
            }
        }
        _ => false,
    };

    (rest_ok, ws_ok, account_ok)
}

// ─── GET /api/alerts ──────────────────────────────────────────────────
pub async fn get_alerts(
    Query(query): Query<AlertsQuery>,
    State(state): State<AppState>,
) -> Result<Json<Vec<AlertRow>>, StatusCode> {
    let limit = query.limit.unwrap_or(50);

    // Try to read from risk.alerts table if it exists
    let rows = sqlx::query(
        "SELECT id, symbol, source, severity, message, price, change_pct, timestamp
         FROM risk.alerts
         ORDER BY timestamp DESC
         LIMIT $1"
    )
    .bind(limit)
    .fetch_all(&state.db_pool)
    .await;

    match rows {
        Ok(rows) => {
            let alerts: Vec<AlertRow> = rows.iter().map(|r| {
                let ts: chrono::DateTime<chrono::Utc> = r.get("timestamp");
                let now = chrono::Utc::now();
                let diff = now - ts;
                let time_ago = if diff.num_hours() > 0 {
                    format!("{}h ago", diff.num_hours())
                } else if diff.num_minutes() > 0 {
                    format!("{}m ago", diff.num_minutes())
                } else {
                    "just now".to_string()
                };

                AlertRow {
                    id: r.get("id"),
                    pair: r.get("symbol"),
                    alert_type: r.try_get::<String, _>("source").unwrap_or("Info".to_string()),
                    time_ago,
                    message: r.get("message"),
                    severity: r.try_get::<String, _>("severity").unwrap_or("info".to_string()),
                    source: r.try_get::<String, _>("source").unwrap_or("system".to_string()),
                    timestamp: ts.to_rfc3339(),
                }
            }).collect();
            Ok(Json(alerts))
        }
        Err(_) => {
            // Table doesn't exist yet — return empty
            Ok(Json(vec![]))
        }
    }
}

// ─── GET /api/trading-options ─────────────────────────────────────────
pub async fn get_trading_options(
) -> Result<Json<TradingOptionsPayload>, StatusCode> {
    // Read from unified config/order_manager.toml
    let path = "config/order_manager.toml";
    if let Ok(content) = std::fs::read_to_string(path) {
        if let Ok(val) = content.parse::<toml::Value>() {
            return Ok(Json(TradingOptionsPayload {
                leverage: val.get("leverage").and_then(|v| v.as_integer()).unwrap_or(10) as u16,
                max_orders_at_a_time: val.get("max_orders_at_a_time").and_then(|v| v.as_integer()).unwrap_or(10) as u16,
                trade_size_type: val.get("trade_size_type").and_then(|v| v.as_str()).unwrap_or("fixed_usdt").to_string(),
                trade_size_value: val.get("trade_size_value").and_then(|v| v.as_float()).unwrap_or(100.0),
                strategy_type: val.get("strategy_type").and_then(|v| v.as_str()).unwrap_or("ml_super_entry").to_string(),
                order_type: val.get("order_type").and_then(|v| v.as_str()).unwrap_or("futures_oco").to_string(),
                trading_mode: val.get("trading_mode").and_then(|v| v.as_str()).unwrap_or("off").to_string(),
            }));
        }
    }
    // Fallback defaults
    Ok(Json(TradingOptionsPayload {
        leverage: 10, max_orders_at_a_time: 10,
        trade_size_type: "fixed_usdt".to_string(), trade_size_value: 100.0,
        strategy_type: "ml_super_entry".to_string(), order_type: "futures_oco".to_string(),
        trading_mode: "off".to_string(),
    }))
}

// ─── POST /api/trading-options ────────────────────────────────────────
pub async fn save_trading_options(
    State(state): State<AppState>,
    Json(options): Json<TradingOptionsPayload>,
) -> Result<StatusCode, StatusCode> {
    // Update trading fields in order_manager.toml (preserve other fields)
    let path = "config/order_manager.toml";
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut val: toml::Value = content.parse().unwrap_or(toml::Value::Table(Default::default()));

    if let Some(table) = val.as_table_mut() {
        table.insert("leverage".to_string(), toml::Value::Integer(options.leverage as i64));
        table.insert("max_orders_at_a_time".to_string(), toml::Value::Integer(options.max_orders_at_a_time as i64));
        table.insert("trade_size_type".to_string(), toml::Value::String(options.trade_size_type));
        table.insert("trade_size_value".to_string(), toml::Value::Float(options.trade_size_value));
        table.insert("strategy_type".to_string(), toml::Value::String(options.strategy_type));
        table.insert("order_type".to_string(), toml::Value::String(options.order_type));
        table.insert("trading_mode".to_string(), toml::Value::String(options.trading_mode));
    }

    std::fs::write(path, toml::to_string_pretty(&val).unwrap_or_default())
        .map_err(|e| {
            tracing::error!("Failed to save trading options: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    state.broadcast(crate::state::WsMessage::OptionsUpdated);
    Ok(StatusCode::OK)
}

// ─── GET /api/order-options ───────────────────────────────────────────
pub async fn get_order_options(
) -> Result<Json<crate::state::OrderOptionsPayload>, StatusCode> {
    // Load order_manager.toml
    let om = load_order_manager_toml();
    // Load risk_manager.toml
    let rm = load_risk_manager_toml();

    Ok(Json(crate::state::OrderOptionsPayload {
        order_manager: crate::state::OrderManagerOptionsPayload {
            signal_score_min_1m: om.score_min_1m,
            signal_score_max_1m: om.score_max_1m,
            signal_score_min_5m: om.score_min_5m,
            signal_score_max_5m: om.score_max_5m,
            signal_score_min_15m: om.score_min_15m,
            signal_score_max_15m: om.score_max_15m,
            signal_score_min_1h: om.score_min_1h,
            signal_score_max_1h: om.score_max_1h,
            signal_score_min_4h: om.score_min_4h,
            signal_score_max_4h: om.score_max_4h,
            signal_score_min_1d: om.score_min_1d,
            signal_score_max_1d: om.score_max_1d,
            max_hold_bars: om.max_hold,
            tf_1m_pct: om.tf_1m,
            tf_5m_pct: om.tf_5m,
            tf_15m_pct: om.tf_15m,
            tf_1h_pct: om.tf_1h,
            tf_4h_pct: om.tf_4h,
            tf_1d_pct: om.tf_1d,
        },
        risk_manager: crate::state::RiskManagerOptionsPayload {
            btc_alert_threshold_pct: rm.0,
            alt_alert_threshold_pct: rm.1,
            volume_spike_threshold: rm.2,
        },
    }))
}

struct OrderManagerSubset {
    score_min_1m: f64,
    score_max_1m: f64,
    score_min_5m: f64,
    score_max_5m: f64,
    score_min_15m: f64,
    score_max_15m: f64,
    score_min_1h: f64,
    score_max_1h: f64,
    score_min_4h: f64,
    score_max_4h: f64,
    score_min_1d: f64,
    score_max_1d: f64,
    max_hold: i32,
    tf_1m: u16,
    tf_5m: u16,
    tf_15m: u16,
    tf_1h: u16,
    tf_4h: u16,
    tf_1d: u16,
}

fn load_order_manager_toml() -> OrderManagerSubset {
    let path = "config/order_manager.toml";
    if let Ok(content) = std::fs::read_to_string(path) {
        if let Ok(val) = content.parse::<toml::Value>() {
            let score_min_1m = val.get("signal_score_min_1m").and_then(|v| v.as_float()).unwrap_or(0.70);
            let score_max_1m = val.get("signal_score_max_1m").and_then(|v| v.as_float()).unwrap_or(0.80);
            let score_min_5m = val.get("signal_score_min_5m").and_then(|v| v.as_float()).unwrap_or(0.70);
            let score_max_5m = val.get("signal_score_max_5m").and_then(|v| v.as_float()).unwrap_or(0.80);
            let score_min_15m = val.get("signal_score_min_15m").and_then(|v| v.as_float()).unwrap_or(0.70);
            let score_max_15m = val.get("signal_score_max_15m").and_then(|v| v.as_float()).unwrap_or(0.80);
            let score_min_1h = val.get("signal_score_min_1h").and_then(|v| v.as_float()).unwrap_or(0.70);
            let score_max_1h = val.get("signal_score_max_1h").and_then(|v| v.as_float()).unwrap_or(0.80);
            let score_min_4h = val.get("signal_score_min_4h").and_then(|v| v.as_float()).unwrap_or(0.70);
            let score_max_4h = val.get("signal_score_max_4h").and_then(|v| v.as_float()).unwrap_or(0.80);
            let score_min_1d = val.get("signal_score_min_1d").and_then(|v| v.as_float()).unwrap_or(0.70);
            let score_max_1d = val.get("signal_score_max_1d").and_then(|v| v.as_float()).unwrap_or(0.80);
            let max_hold = val.get("max_hold_bars").and_then(|v| v.as_integer()).unwrap_or(25) as i32;
            let tf_1m = val.get("tf_1m_pct").and_then(|v| v.as_integer()).unwrap_or(0) as u16;
            let tf_5m = val.get("tf_5m_pct").and_then(|v| v.as_integer()).unwrap_or(0) as u16;
            let tf_15m = val.get("tf_15m_pct").and_then(|v| v.as_integer()).unwrap_or(10) as u16;
            let tf_1h = val.get("tf_1h_pct").and_then(|v| v.as_integer()).unwrap_or(70) as u16;
            let tf_4h = val.get("tf_4h_pct").and_then(|v| v.as_integer()).unwrap_or(20) as u16;
            let tf_1d = val.get("tf_1d_pct").and_then(|v| v.as_integer()).unwrap_or(0) as u16;
            return OrderManagerSubset {
                score_min_1m, score_max_1m,
                score_min_5m, score_max_5m,
                score_min_15m, score_max_15m,
                score_min_1h, score_max_1h,
                score_min_4h, score_max_4h,
                score_min_1d, score_max_1d,
                max_hold,
                tf_1m, tf_5m, tf_15m, tf_1h, tf_4h, tf_1d,
            };
        }
    }
    OrderManagerSubset {
        score_min_1m: 0.70, score_max_1m: 0.80,
        score_min_5m: 0.70, score_max_5m: 0.80,
        score_min_15m: 0.70, score_max_15m: 0.80,
        score_min_1h: 0.70, score_max_1h: 0.80,
        score_min_4h: 0.70, score_max_4h: 0.80,
        score_min_1d: 0.70, score_max_1d: 0.80,
        max_hold: 25,
        tf_1m: 0, tf_5m: 0, tf_15m: 10, tf_1h: 70, tf_4h: 20, tf_1d: 0,
    }
}

fn load_risk_manager_toml() -> (f64, f64, f64) {
    let path = "config/risk_manager.toml";
    if let Ok(content) = std::fs::read_to_string(path) {
        if let Ok(val) = content.parse::<toml::Value>() {
            let btc = val.get("btc_alert_threshold_pct").and_then(|v| v.as_float()).unwrap_or(0.5);
            let alt = val.get("alt_alert_threshold_pct").and_then(|v| v.as_float()).unwrap_or(1.5);
            let vol = val.get("volume_spike_threshold").and_then(|v| v.as_float()).unwrap_or(2.0);
            return (btc, alt, vol);
        }
    }
    (0.5, 1.5, 2.0)
}

// ─── POST /api/order-options ──────────────────────────────────────────
pub async fn save_order_options(
    State(state): State<AppState>,
    Json(options): Json<crate::state::OrderOptionsPayload>,
) -> Result<StatusCode, StatusCode> {
    // Update order_manager.toml — preserve existing fields, update subset
    save_order_manager_subset(&options.order_manager)
        .map_err(|e| {
            tracing::error!("Failed to save order manager opts: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Update risk_manager.toml — preserve existing fields, update subset
    save_risk_manager_subset(&options.risk_manager)
        .map_err(|e| {
            tracing::error!("Failed to save risk manager opts: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    state.broadcast(crate::state::WsMessage::OptionsUpdated);
    tracing::info!("Order options saved");
    Ok(StatusCode::OK)
}

fn save_order_manager_subset(opts: &crate::state::OrderManagerOptionsPayload) -> anyhow::Result<()> {
    let path = "config/order_manager.toml";
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut val: toml::Value = content.parse().unwrap_or(toml::Value::Table(Default::default()));

    if let Some(table) = val.as_table_mut() {
        // Per-timeframe score ranges
        table.insert("signal_score_min_1m".to_string(), toml::Value::Float(opts.signal_score_min_1m));
        table.insert("signal_score_max_1m".to_string(), toml::Value::Float(opts.signal_score_max_1m));
        table.insert("signal_score_min_5m".to_string(), toml::Value::Float(opts.signal_score_min_5m));
        table.insert("signal_score_max_5m".to_string(), toml::Value::Float(opts.signal_score_max_5m));
        table.insert("signal_score_min_15m".to_string(), toml::Value::Float(opts.signal_score_min_15m));
        table.insert("signal_score_max_15m".to_string(), toml::Value::Float(opts.signal_score_max_15m));
        table.insert("signal_score_min_1h".to_string(), toml::Value::Float(opts.signal_score_min_1h));
        table.insert("signal_score_max_1h".to_string(), toml::Value::Float(opts.signal_score_max_1h));
        table.insert("signal_score_min_4h".to_string(), toml::Value::Float(opts.signal_score_min_4h));
        table.insert("signal_score_max_4h".to_string(), toml::Value::Float(opts.signal_score_max_4h));
        table.insert("signal_score_min_1d".to_string(), toml::Value::Float(opts.signal_score_min_1d));
        table.insert("signal_score_max_1d".to_string(), toml::Value::Float(opts.signal_score_max_1d));
        // Max hold bars
        table.insert("max_hold_bars".to_string(), toml::Value::Integer(opts.max_hold_bars as i64));
        // Timeframe percentages
        table.insert("tf_1m_pct".to_string(), toml::Value::Integer(opts.tf_1m_pct as i64));
        table.insert("tf_5m_pct".to_string(), toml::Value::Integer(opts.tf_5m_pct as i64));
        table.insert("tf_15m_pct".to_string(), toml::Value::Integer(opts.tf_15m_pct as i64));
        table.insert("tf_1h_pct".to_string(), toml::Value::Integer(opts.tf_1h_pct as i64));
        table.insert("tf_4h_pct".to_string(), toml::Value::Integer(opts.tf_4h_pct as i64));
        table.insert("tf_1d_pct".to_string(), toml::Value::Integer(opts.tf_1d_pct as i64));
    }

    std::fs::write(path, toml::to_string_pretty(&val)?)?;
    Ok(())
}

fn save_risk_manager_subset(opts: &crate::state::RiskManagerOptionsPayload) -> anyhow::Result<()> {
    let path = "config/risk_manager.toml";
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut val: toml::Value = content.parse().unwrap_or(toml::Value::Table(Default::default()));

    if let Some(table) = val.as_table_mut() {
        table.insert("btc_alert_threshold_pct".to_string(), toml::Value::Float(opts.btc_alert_threshold_pct));
        table.insert("alt_alert_threshold_pct".to_string(), toml::Value::Float(opts.alt_alert_threshold_pct));
        table.insert("volume_spike_threshold".to_string(), toml::Value::Float(opts.volume_spike_threshold));
    }

    std::fs::write(path, toml::to_string_pretty(&val)?)?;
    Ok(())
}

// ─── POST /api/trade/order (manual with TP/SL) ───────────────────────
pub async fn place_order(
    State(state): State<AppState>,
    Json(order): Json<ManualOrderRequest>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Manual order: {:?}", order);

    // Broadcast order event
    state.broadcast(crate::state::WsMessage::OrderEvent(crate::state::OrderEvent {
        order_id: uuid::Uuid::new_v4().to_string(),
        pair: order.pair.clone(),
        side: order.side.clone(),
        order_type: order.order_type.clone(),
        status: "created".into(),
        price: order.entry_price,
        qty: order.amount_usdt,
        ts: chrono::Utc::now().timestamp_millis(),
    }));

    // TODO: Actually place order via BinanceFuturesClient
    // 1. Set leverage
    // 2. Place market/limit entry
    // 3. Place stop-loss order
    // 4. Place take-profit order
    // 5. Record in trade.positions

    Ok(StatusCode::OK)
}

// ─── POST /api/trade/close ────────────────────────────────────────────
pub async fn close_position(
    State(state): State<AppState>,
    Json(request): Json<crate::state::ClosePositionRequest>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Close position request: id={}", request.position_id);

    // 1. Fetch position details from DB
    let row = sqlx::query(
        "SELECT p.id, mp.symbol as pair,
                CASE WHEN p.side = 1 THEN 'LONG' ELSE 'SHORT' END as side,
                p.qty, p.symbol_id
         FROM trade.positions p
         JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
         WHERE p.id = $1 AND p.status = 1"
    )
    .bind(request.position_id)
    .fetch_optional(&state.db_pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to query position {}: {}", request.position_id, e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let row = match row {
        Some(r) => r,
        None => {
            tracing::warn!("Position {} not found or already closed", request.position_id);
            return Err(StatusCode::NOT_FOUND);
        }
    };

    let pair: String = row.get("pair");
    let side: String = row.get("side");
    let qty: f64 = row.get("qty");

    // 2. Load Binance client
    let exchange_settings = settings::ExchangeSettings::load()
        .map_err(|e| {
            tracing::error!("Cannot load exchange settings for close: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if !exchange_settings.has_credentials() {
        tracing::error!("No API credentials for closing position");
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    let testnet = std::env::var("BINANCE_TESTNET")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    let client = connections_lib::BinanceFuturesClient::new(
        exchange_settings.api_key(),
        exchange_settings.api_secret(),
        testnet,
    ).map_err(|e| {
        tracing::error!("Failed to create Binance client for close: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // 3. Cancel any open SL/TP orders for this symbol
    if let Err(e) = client.cancel_all_orders(&pair).await {
        tracing::warn!("Failed to cancel orders for {} (may be none): {}", pair, e);
    }

    // 4. Close position via market order
    if qty > 0.0 {
        match client.close_position(&pair, &side, qty).await {
            Ok(order) => {
                tracing::info!(
                    "Position #{} {} {} closed → orderId={}, status={}",
                    request.position_id, pair, side, order.order_id, order.status
                );
            }
            Err(e) => {
                tracing::error!(
                    "Failed to close position #{} {} {}: {}",
                    request.position_id, pair, side, e
                );
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
        }
    }

    // 5. Update DB status
    let _ = sqlx::query(
        "UPDATE trade.positions SET status = 2, close_reason = 'manual_webui', closed_at = now() WHERE id = $1"
    )
    .bind(request.position_id)
    .execute(&state.db_pool)
    .await;

    // 6. Record in position_history
    let _ = sqlx::query(
        "INSERT INTO trade.position_history (position_id, symbol, symbol_id, side, qty, entry_price, exit_price, close_reason, realized_pnl, realized_pnl_pct, opened_at, closed_at, leverage, tf_minutes)
         SELECT id, COALESCE(symbol, ''), symbol_id, side, qty, entry_price,
                COALESCE(current_price, entry_price), 'manual_webui',
                COALESCE(unrealized_pnl, 0),
                CASE WHEN entry_price > 0 THEN
                    CASE WHEN side = 1
                        THEN (COALESCE(current_price, entry_price) - entry_price) / entry_price * 100.0
                        ELSE (entry_price - COALESCE(current_price, entry_price)) / entry_price * 100.0
                    END
                ELSE 0 END,
                opened_at, now(), COALESCE(leverage, 10), COALESCE(tf_minutes, 60)
         FROM trade.positions WHERE id = $1"
    )
    .bind(request.position_id)
    .execute(&state.db_pool)
    .await;

    // 7. Broadcast update
    state.broadcast(crate::state::WsMessage::OptionsUpdated);

    tracing::info!("✅ Position #{} closed successfully via WebUI", request.position_id);
    Ok(StatusCode::OK)
}

// ─── POST /api/trade/update-candles-left ──────────────────────────────
pub async fn update_candles_left(
    State(state): State<AppState>,
    Json(request): Json<CandlesLeftUpdateRequest>,
) -> Result<StatusCode, StatusCode> {
    sqlx::query(
        "UPDATE trade.positions SET candles_left = $1 WHERE id = $2"
    )
    .bind(request.candles_left)
    .bind(request.position_id)
    .execute(&state.db_pool)
    .await
    .map_err(|e| {
        tracing::error!("Update candles_left error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    tracing::info!("Candles left updated: position={} candles_left={}", request.position_id, request.candles_left);
    Ok(StatusCode::OK)
}

// ─── POST /api/control/emergency_stop ─────────────────────────────────
pub async fn emergency_stop(
    State(state): State<AppState>,
) -> Result<StatusCode, StatusCode> {
    tracing::warn!("🚨 EMERGENCY STOP triggered from WebUI!");

    // 1. Stop auto trading in memory
    {
        let mut at = state.auto_trading.write().await;
        at.is_running = false;
        at.trading_mode = "off".to_string();
    }

    // 2. Persist trading_mode = "off" to config file
    update_trading_mode_in_config("off");

    // 3. Close ALL open positions on Binance
    let close_results = emergency_close_all_positions(&state.db_pool).await;
    match &close_results {
        Ok(closed) => {
            tracing::warn!("🚨 Emergency closed {} positions", closed);
        }
        Err(e) => {
            tracing::error!("🚨 Emergency close failed: {}", e);
        }
    }

    // 4. Send Kafka close_all command to order_manager (best-effort)
    if let Err(e) = send_kafka_close_all().await {
        tracing::warn!("Failed to send Kafka close_all (order_manager may not receive): {}", e);
    }

    state.broadcast(crate::state::WsMessage::TradingStateUpdate(
        state.auto_trading.read().await.clone()
    ));

    Ok(StatusCode::OK)
}

/// Close all open positions via Binance Futures API.
/// Returns the number of positions closed.
async fn emergency_close_all_positions(pool: &sqlx::PgPool) -> Result<usize, StatusCode> {
    // Load Binance client
    let exchange_settings = settings::ExchangeSettings::load()
        .map_err(|e| {
            tracing::error!("Cannot load exchange settings for emergency close: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if !exchange_settings.has_credentials() {
        tracing::error!("No API credentials for emergency close");
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    let testnet = std::env::var("BINANCE_TESTNET")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    let client = connections_lib::BinanceFuturesClient::new(
        exchange_settings.api_key(),
        exchange_settings.api_secret(),
        testnet,
    ).map_err(|e| {
        tracing::error!("Failed to create Binance client for emergency close: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Query open positions from DB
    let rows = sqlx::query(
        "SELECT p.id, mp.symbol as pair,
                CASE WHEN p.side = 1 THEN 'LONG' ELSE 'SHORT' END as side,
                p.qty
         FROM trade.positions p
         JOIN market.pairs mp ON p.symbol_id = mp.symbol_id
         WHERE p.status = 1"
    )
    .fetch_all(pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to query open positions: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let mut closed = 0usize;

    for row in &rows {
        let pair: String = row.get("pair");
        let side: String = row.get("side");
        let qty: f64 = row.get("qty");
        let position_id: i64 = row.get("id");

        // Cancel all open orders first (SL/TP)
        if let Err(e) = client.cancel_all_orders(&pair).await {
            tracing::warn!("Failed to cancel orders for {} (may be none): {}", pair, e);
        }

        // If qty=0 in DB (entry wasn't confirmed), skip closing via DB qty.
        // The safety net below will find and close the Binance position.
        if qty <= 0.0 {
            tracing::warn!(
                "🚨 Position #{} {} {} has qty=0 in DB (entry price bug). Skipping DB close, will use Binance safety net.",
                position_id, pair, side
            );
            // Mark as closed in DB so it doesn't show up anymore
            let _ = sqlx::query(
                "UPDATE trade.positions SET status = 2, close_reason = 'emer_closed', closed_at = now() WHERE id = $1"
            )
            .bind(position_id)
            .execute(pool)
            .await;
            // Record in position_history even for qty=0 positions
            let _ = sqlx::query(
                "INSERT INTO trade.position_history (position_id, symbol, symbol_id, side, qty, entry_price, exit_price, close_reason, realized_pnl, realized_pnl_pct, opened_at, closed_at, leverage, tf_minutes, combined_score, p_super)
                 SELECT id, COALESCE(symbol, ''), symbol_id, side, qty, entry_price,
                        entry_price, 'emer_closed',
                        COALESCE(unrealized_pnl, 0),
                        CASE WHEN entry_price > 0 THEN
                            CASE WHEN side = 1
                                THEN (entry_price - entry_price) / entry_price * 100.0
                                ELSE (entry_price - entry_price) / entry_price * 100.0
                            END
                        ELSE 0 END,
                        opened_at, now(), COALESCE(leverage, 10), COALESCE(tf_minutes, 60),
                        combined_score, p_super
                 FROM trade.positions WHERE id = $1"
            )
            .bind(position_id)
            .execute(pool)
            .await;
            continue;
        }

        tracing::warn!("🚨 Emergency closing: {} {} qty={:.8} (position #{})", pair, side, qty, position_id);

        // Close position via market order
        match client.close_position(&pair, &side, qty).await {
            Ok(order) => {
                tracing::warn!(
                    "🚨 Position #{} {} {} closed → orderId={}, status={}",
                    position_id, pair, side, order.order_id, order.status
                );

                // Update DB status
                let _ = sqlx::query(
                    "UPDATE trade.positions SET status = 2, close_reason = 'emer_closed', closed_at = now() WHERE id = $1"
                )
                .bind(position_id)
                .execute(pool)
                .await;

                // Record in position_history with pnl and all details
                let _ = sqlx::query(
                    "INSERT INTO trade.position_history (position_id, symbol, symbol_id, side, qty, entry_price, exit_price, close_reason, realized_pnl, realized_pnl_pct, opened_at, closed_at, leverage, tf_minutes, combined_score, p_super)
                     SELECT id, COALESCE(symbol, ''), symbol_id, side, qty, entry_price,
                            COALESCE(current_price, entry_price), 'emer_closed',
                            COALESCE(unrealized_pnl, 0),
                            CASE WHEN entry_price > 0 THEN
                                CASE WHEN side = 1
                                    THEN (COALESCE(current_price, entry_price) - entry_price) / entry_price * 100.0
                                    ELSE (entry_price - COALESCE(current_price, entry_price)) / entry_price * 100.0
                                END
                            ELSE 0 END,
                            opened_at, now(), COALESCE(leverage, 10), COALESCE(tf_minutes, 60),
                            combined_score, p_super
                     FROM trade.positions WHERE id = $1"
                )
                .bind(position_id)
                .execute(pool)
                .await;

                closed += 1;
            }
            Err(e) => {
                tracing::error!("🚨 Failed to close position #{} {} {}: {}", position_id, pair, side, e);
            }
        }
    }

    // Also close any positions on Binance that aren't in our DB (safety net)
    if let Ok(binance_positions) = client.open_positions().await {
        for bp in &binance_positions {
            let amt: f64 = bp.position_amt.parse().unwrap_or(0.0);
            if amt.abs() < 0.001 { continue; }

            let side = if amt > 0.0 { "LONG" } else { "SHORT" };
            tracing::warn!(
                "🚨 Found untracked Binance position: {} {} qty={:.8}, closing...",
                bp.symbol, side, amt.abs()
            );

            match client.close_position(&bp.symbol, side, amt.abs()).await {
                Ok(order) => {
                    tracing::warn!("🚨 Untracked position {} closed → orderId={}", bp.symbol, order.order_id);
                    closed += 1;
                }
                Err(e) => {
                    tracing::error!("🚨 Failed to close untracked position {}: {}", bp.symbol, e);
                }
            }
        }
    }

    Ok(closed)
}

/// Send close_all command to order_manager via Kafka (best-effort).
async fn send_kafka_close_all() -> anyhow::Result<()> {
    let brokers = std::env::var("KAFKA_BROKERS")
        .unwrap_or_else(|_| "127.0.0.1:19092".to_string());

    let producer: rdkafka::producer::FutureProducer = rdkafka::config::ClientConfig::new()
        .set("bootstrap.servers", &brokers)
        .set("message.timeout.ms", "5000")
        .create()?;

    let payload = serde_json::json!({
        "cmd_type": "close_all",
        "source": "emergency_stop",
        "timestamp": chrono::Utc::now().timestamp_millis()
    });

    let payload_str = serde_json::to_string(&payload)?;
    let record = rdkafka::producer::FutureRecord::to("orders.cmd")
        .key("emergency")
        .payload(&payload_str);

    producer.send(record, std::time::Duration::from_secs(3)).await
        .map_err(|(e, _)| anyhow::anyhow!("Kafka send error: {}", e))?;

    tracing::info!("📤 Sent close_all command to orders.cmd topic");
    Ok(())
}

// ─── POST /api/control/start_trading ──────────────────────────────────
pub async fn start_trading(
    State(state): State<AppState>,
) -> Result<Json<AutoTradingState>, StatusCode> {
    tracing::info!("START TRADING triggered from WebUI");

    let new_state = {
        let mut at = state.auto_trading.write().await;
        at.is_running = true;
        at.trading_mode = "auto".to_string();
        at.clone()
    };

    // Update order_manager.toml trading_mode to auto
    update_trading_mode_in_config("auto");

    state.broadcast(crate::state::WsMessage::TradingStateUpdate(new_state.clone()));
    Ok(Json(new_state))
}

// ─── POST /api/control/stop_trading ───────────────────────────────────
pub async fn stop_trading(
    State(state): State<AppState>,
) -> Result<Json<AutoTradingState>, StatusCode> {
    tracing::info!("STOP TRADING triggered from WebUI");

    let new_state = {
        let mut at = state.auto_trading.write().await;
        at.is_running = false;
        at.trading_mode = "off".to_string();
        at.clone()
    };

    // Update order_manager.toml trading_mode to off
    update_trading_mode_in_config("off");

    state.broadcast(crate::state::WsMessage::TradingStateUpdate(new_state.clone()));
    Ok(Json(new_state))
}

// ─── GET /api/control/trading_state ───────────────────────────────────
pub async fn get_trading_state(
    State(state): State<AppState>,
) -> Result<Json<AutoTradingState>, StatusCode> {
    let at = state.auto_trading.read().await.clone();
    Ok(Json(at))
}

/// Helper: update trading_mode in order_manager.toml
fn update_trading_mode_in_config(mode: &str) {
    let path = "config/order_manager.toml";
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut val: toml::Value = content.parse().unwrap_or(toml::Value::Table(Default::default()));
    if let Some(table) = val.as_table_mut() {
        table.insert("trading_mode".to_string(), toml::Value::String(mode.to_string()));
    }
    let _ = std::fs::write(path, toml::to_string_pretty(&val).unwrap_or_default());
}

// ─── GET /api/strategies ──────────────────────────────────────────────
pub async fn get_strategies(
    _state: State<AppState>,
) -> Result<Json<Vec<crate::state::StrategyInfo>>, StatusCode> {
    // Read current strategy from order_manager.toml
    let current_strategy = std::fs::read_to_string("config/order_manager.toml")
        .ok()
        .and_then(|c| c.parse::<toml::Value>().ok())
        .and_then(|v| v.get("strategy_type").and_then(|s| s.as_str()).map(|s| s.to_string()))
        .unwrap_or_else(|| "ml_super_entry".to_string());

    Ok(Json(vec![
        crate::state::StrategyInfo {
            id: "ml_super_entry".into(),
            name: "ML Super Entry".into(),
            enabled: current_strategy == "ml_super_entry",
            priority: 0,
            description: "ML-модель поиска super moves с P(super) > threshold".into(),
        },
        crate::state::StrategyInfo {
            id: "level_strategy".into(),
            name: "Level Strategy".into(),
            enabled: current_strategy == "level_strategy",
            priority: 1,
            description: "ML predictors + trade signals (в разработке)".into(),
        },
    ]))
}

// ─── POST /api/strategy/toggle ────────────────────────────────────────
pub async fn toggle_strategy(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<StatusCode, StatusCode> {
    let id = payload.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let enabled = payload.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
    tracing::info!("Toggle strategy: {} -> {}", id, enabled);
    state.broadcast(crate::state::WsMessage::StrategyUpdated);
    Ok(StatusCode::OK)
}
