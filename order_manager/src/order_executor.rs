// order_manager/src/order_executor.rs
//
// Order Executor — открытие и закрытие позиций на Binance Futures.
//
// Логика:
//   1. Получает QualifiedSignal от SignalScanner
//   2. Открывает MARKET ордер (entry)
//   3. Размещает STOP_MARKET (SL) и TAKE_PROFIT_MARKET (TP) ордера
//   4. Записывает позицию в trade.positions и ордера в trade.orders
//   5. Поддерживает пропорцию 1h:4h:15m = 7:2:1 при 10 открытых позициях
//   6. При закрытии — записывает в trade.position_history

use anyhow::{Context, Result};
use chrono::Utc;
use sqlx::{PgPool, Row};
use std::time::Duration;
use tracing::{info, warn, error, debug};

use connections_lib::BinanceFuturesClient;

use crate::config::OrderManagerConfig;
use crate::exchange_info::ExchangeInfoCache;
use crate::types::{
    CloseReason, ManagedPosition, PositionStatus, PositionUpdateEvent, QualifiedSignal, Side,
};

/// Order Executor — управляет жизненным циклом ордеров на бирже.
pub struct OrderExecutor {
    pool: PgPool,
    config: OrderManagerConfig,
    client: BinanceFuturesClient,
    exchange_info: ExchangeInfoCache,
}

impl OrderExecutor {
    pub fn new(
        pool: PgPool,
        config: OrderManagerConfig,
        client: BinanceFuturesClient,
        exchange_info: ExchangeInfoCache,
    ) -> Self {
        Self {
            pool,
            config,
            client,
            exchange_info,
        }
    }

    /// Открыть новую позицию по квалифицированному сигналу.
    pub async fn open_position(&self, signal: &QualifiedSignal) -> Result<ManagedPosition> {
        let symbol = &signal.symbol;
        let side = signal.side;
        let entry_side = side.entry_side_str();
        let close_side = side.close_side_str();

        // 1. Рассчитать количество (с учётом Exchange Info stepSize)
        let qty = self.calculate_quantity(signal.current_price, symbol).await?;

        info!(
            "🔵 Opening {} {} position: price={:.4}, qty={:.8}, SL={:.4}, TP={:.4}, score={:.3}",
            symbol, side, signal.current_price, qty, signal.sl_price, signal.tp_price, signal.combined_score
        );

        // 2. Set leverage
        if let Err(e) = self.client.set_leverage(symbol, self.config.leverage).await {
            warn!("Failed to set leverage for {} (may already be set): {}", symbol, e);
        }

        // 3. Place MARKET entry order
        let entry_order = self
            .client
            .place_market_order(symbol, entry_side, qty, false)
            .await
            .context("Failed to place entry market order")?;

        // 3a. Wait for FILLED status — Binance may return status=NEW initially
        let (filled_price, filled_qty) = self
            .wait_for_fill(symbol, &entry_order, signal.current_price, qty)
            .await;

        info!(
            "✅ Entry filled: {} {} avg_price={:.8}, qty={:.8}, orderId={}",
            symbol, entry_side, filled_price, filled_qty, entry_order.order_id
        );

        // 4. Round SL/TP prices by exchange tickSize to avoid -1111 precision errors
        let sl_price_rounded = self.exchange_info.round_price(symbol, signal.sl_price).await;
        let tp_price_rounded = self.exchange_info.round_price(symbol, signal.tp_price).await;

        // 4a. Place SL order (with -4120/-1111 fallback to client-side monitoring)
        let sl_order = match self
            .client
            .place_stop_market(symbol, close_side, filled_qty, sl_price_rounded)
            .await
        {
            Ok(order) => {
                info!("✅ SL placed: {} stopPrice={:.8}, orderId={}", symbol, sl_price_rounded, order.order_id);
                Some(order)
            }
            Err(e) => {
                let err_str = format!("{}", e);
                if err_str.contains("-4120") || err_str.contains("-1111") {
                    warn!(
                        "⚠️ {} SL order rejected ({}). Client-side SL monitoring active via PositionTracker.",
                        symbol, if err_str.contains("-4120") { "-4120: unsupported order type" } else { "-1111: precision error" }
                    );
                } else {
                    error!("❌ Failed to place SL for {}: {}", symbol, e);
                }
                None
            }
        };

        // 5. Place TP order (with -4120/-1111 fallback to client-side monitoring)
        let tp_order = match self
            .client
            .place_take_profit_market(symbol, close_side, filled_qty, tp_price_rounded)
            .await
        {
            Ok(order) => {
                info!("✅ TP placed: {} stopPrice={:.8}, orderId={}", symbol, tp_price_rounded, order.order_id);
                Some(order)
            }
            Err(e) => {
                let err_str = format!("{}", e);
                if err_str.contains("-4120") || err_str.contains("-1111") {
                    warn!(
                        "⚠️ {} TP order rejected ({}). Client-side TP monitoring active via PositionTracker.",
                        symbol, if err_str.contains("-4120") { "-4120: unsupported order type" } else { "-1111: precision error" }
                    );
                } else {
                    error!("❌ Failed to place TP for {}: {}", symbol, e);
                }
                None
            }
        };

        // Log if both SL and TP are client-side managed
        if sl_order.is_none() && tp_order.is_none() {
            warn!(
                "🛡️ {} SL={:.4} / TP={:.4} managed entirely client-side. \
                 PositionTracker checks every {}s. Bot downtime = unprotected!",
                symbol, signal.sl_price, signal.tp_price, self.config.tracker_interval_secs
            );
        }

        // 6. Записать позицию в БД
        let now = Utc::now();
        let entry_oid = entry_order.order_id.to_string();
        let sl_oid = sl_order.as_ref().map(|o| o.order_id.to_string());
        let tp_oid = tp_order.as_ref().map(|o| o.order_id.to_string());

        let position_id = self
            .insert_position(signal, filled_price, filled_qty, &entry_oid, sl_oid.as_deref(), tp_oid.as_deref())
            .await?;

        // 7. Записать ордера в trade.orders
        self.insert_order_record(signal, position_id, &entry_order, "MARKET", false).await.ok();
        if let Some(ref sl) = sl_order {
            self.insert_order_record(signal, position_id, sl, "STOP_MARKET", true).await.ok();
        }
        if let Some(ref tp) = tp_order {
            self.insert_order_record(signal, position_id, tp, "TAKE_PROFIT_MARKET", true).await.ok();
        }

        let managed = ManagedPosition {
            position_id,
            symbol: symbol.clone(),
            symbol_id: signal.symbol_id,
            tf_minutes: signal.tf_minutes,
            side,
            entry_price: filled_price,
            sl_price: signal.sl_price,
            tp_price: signal.tp_price,
            qty: filled_qty,
            leverage: self.config.leverage,
            candles_left: self.config.max_hold_bars,
            max_hold_bars: self.config.max_hold_bars,
            unrealized_pnl: 0.0,
            unrealized_pnl_pct: 0.0,
            current_price: filled_price,
            opened_at: now,
            combined_score: signal.combined_score,
            p_super: signal.p_super,
            entry_order_id: Some(entry_oid),
            sl_order_id: sl_oid,
            tp_order_id: tp_oid,
        };

        info!(
            "📊 Position #{} opened: {} {} entry={:.4} SL={:.4} TP={:.4} tf={}m candles_left={}",
            position_id, symbol, side, filled_price, signal.sl_price, signal.tp_price,
            signal.tf_minutes, self.config.max_hold_bars
        );

        Ok(managed)
    }

    /// Закрыть позицию по рыночной цене.
    pub async fn close_position(
        &self,
        pos: &ManagedPosition,
        reason: CloseReason,
        current_price: f64,
    ) -> Result<PositionUpdateEvent> {
        let symbol = &pos.symbol;

        info!(
            "🔴 Closing {} {} position #{}: reason={}, current_price={:.4}",
            symbol, pos.side, pos.position_id, reason, current_price
        );

        // 1. Отменить SL/TP ордера на бирже
        if let Err(e) = self.client.cancel_all_orders(symbol).await {
            warn!("Failed to cancel orders for {} (position #{}): {}", symbol, pos.position_id, e);
        }

        // 2. Close position
        let close_order = self
            .client
            .close_position(symbol, pos.side.as_str(), pos.qty)
            .await
            .context("Failed to place close market order")?;

        // Wait for FILLED to get real exit price
        let (exit_price, _) = self
            .wait_for_fill(symbol, &close_order, current_price, pos.qty)
            .await;

        // 3. Рассчитать PnL
        let direction = match pos.side {
            Side::Long => 1.0,
            Side::Short => -1.0,
        };
        let realized_pnl = direction * (exit_price - pos.entry_price) * pos.qty;
        let realized_pnl_pct = direction * (exit_price - pos.entry_price) / pos.entry_price * 100.0;

        // Комиссии (taker 0.04% Binance Futures)
        let taker_fee_pct = 0.04;
        let entry_notional = pos.entry_price * pos.qty;
        let exit_notional = exit_price * pos.qty;
        let fees_total = (entry_notional + exit_notional) * taker_fee_pct / 100.0;

        let bars_lived = pos.max_hold_bars - pos.candles_left;

        info!(
            "📊 Position #{} closed: {} {} entry={:.4} exit={:.4} PnL={:.4} USDT ({:.2}%) fees={:.4} reason={}",
            pos.position_id, symbol, pos.side, pos.entry_price, exit_price,
            realized_pnl, realized_pnl_pct, fees_total, reason
        );

        // 4. Обновить trade.positions
        self.update_position_closed(pos.position_id, exit_price, realized_pnl, realized_pnl_pct, fees_total, reason).await?;

        // 5. Записать в trade.position_history
        self.insert_position_history(pos, exit_price, realized_pnl, realized_pnl_pct, fees_total, reason, bars_lived).await?;

        // 6. Событие для WebUI
        let mut event = PositionUpdateEvent::from(pos);
        event.event_type = "position_closed".to_string();
        event.current_price = exit_price;
        event.realized_pnl = Some(realized_pnl);
        event.realized_pnl_pct = Some(realized_pnl_pct);
        event.close_reason = Some(reason.to_string());
        event.closed_at = Some(Utc::now().to_rfc3339());

        Ok(event)
    }

    /// Wait for a market order to be FILLED.
    ///
    /// Binance sometimes returns status="NEW" for MARKET orders, especially
    /// on low-liquidity tokens. We poll `get_order_status` until we see
    /// status="FILLED" with a valid avg_price and executed_qty.
    ///
    /// Returns (avg_price, executed_qty). Falls back to signal price / qty on timeout.
    async fn wait_for_fill(
        &self,
        symbol: &str,
        initial_order: &connections_lib::NewOrderResponse,
        fallback_price: f64,
        fallback_qty: f64,
    ) -> (f64, f64) {
        // Try initial response first
        let avg_price: f64 = initial_order.avg_price.parse().unwrap_or(0.0);
        let exec_qty: f64 = initial_order.executed_qty.parse().unwrap_or(0.0);

        if initial_order.status == "FILLED" && avg_price > 0.0 && exec_qty > 0.0 {
            return (avg_price, exec_qty);
        }

        // If avg_price came through but status is not FILLED yet (partial?), still check
        if avg_price > 0.0 && exec_qty > 0.0 {
            info!(
                "📋 Order {} status={} but has price={:.8}, qty={:.8} — accepting",
                initial_order.order_id, initial_order.status, avg_price, exec_qty
            );
            return (avg_price, exec_qty);
        }

        info!(
            "⏳ Order {} returned status={}, avg_price={}, polling for FILLED...",
            initial_order.order_id, initial_order.status, initial_order.avg_price
        );

        // Poll for fill status
        const MAX_RETRIES: u32 = 15;
        const RETRY_DELAY_MS: u64 = 300;

        for attempt in 1..=MAX_RETRIES {
            tokio::time::sleep(Duration::from_millis(RETRY_DELAY_MS)).await;

            match self.client.get_order_status(symbol, initial_order.order_id).await {
                Ok(status) => {
                    let price: f64 = status.avg_price.parse().unwrap_or(0.0);
                    let qty: f64 = status.executed_qty.parse().unwrap_or(0.0);

                    if status.status == "FILLED" && price > 0.0 && qty > 0.0 {
                        info!(
                            "✅ Order {} FILLED (poll #{}, ~{}ms): avg_price={:.8}, qty={:.8}",
                            initial_order.order_id,
                            attempt,
                            attempt as u64 * RETRY_DELAY_MS,
                            price,
                            qty
                        );
                        return (price, qty);
                    }

                    // Check terminal states that mean the order won't fill
                    if status.status == "CANCELED"
                        || status.status == "EXPIRED"
                        || status.status == "REJECTED"
                    {
                        warn!(
                            "⚠️ Order {} terminated with status={}, not filled",
                            initial_order.order_id, status.status
                        );
                        break;
                    }

                    debug!(
                        "⏳ Order {} status={} price={} qty={} (attempt {}/{})",
                        initial_order.order_id, status.status, status.avg_price,
                        status.executed_qty, attempt, MAX_RETRIES
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to poll order status (attempt {}/{}): {}",
                        attempt, MAX_RETRIES, e
                    );
                }
            }
        }

        // Fallback: use signal's current price
        warn!(
            "⚠️ Order {} not confirmed FILLED after {}ms polling. Using fallback price={:.8}, qty={:.8}",
            initial_order.order_id,
            MAX_RETRIES as u64 * RETRY_DELAY_MS,
            fallback_price,
            fallback_qty
        );
        (fallback_price, fallback_qty)
    }

    /// Рассчитать количество базового актива.
    /// qty = trade_size_usdt * leverage / current_price
    /// Количество округляется по stepSize из Exchange Info.
    async fn calculate_quantity(&self, current_price: f64, symbol: &str) -> Result<f64> {
        if current_price <= 0.0 {
            anyhow::bail!("Invalid price: {}", current_price);
        }
        let notional = self.config.trade_size_value * self.config.leverage as f64;
        let raw_qty = notional / current_price;

        // Округление по Exchange Info (stepSize)
        let qty = self.exchange_info.round_quantity(symbol, raw_qty).await;

        if qty <= 0.0 {
            anyhow::bail!(
                "Calculated quantity is 0 after rounding (price={}, notional={}, raw_qty={:.8})",
                current_price, notional, raw_qty
            );
        }

        // Проверка minNotional
        if !self.exchange_info.check_min_notional(symbol, qty, current_price).await {
            anyhow::bail!(
                "Order below minNotional for {} (qty={:.8}, price={:.4}, notional={:.4})",
                symbol, qty, current_price, qty * current_price
            );
        }

        Ok(qty)
    }

    /// INSERT в trade.positions
    async fn insert_position(
        &self,
        signal: &QualifiedSignal,
        filled_price: f64,
        filled_qty: f64,
        entry_order_id: &str,
        sl_order_id: Option<&str>,
        tp_order_id: Option<&str>,
    ) -> Result<i64> {
        let row = sqlx::query(
            r#"
            INSERT INTO trade.positions (
                symbol_id, symbol, side, qty, entry_price, leverage, margin_type,
                status, tf_minutes, signal_time, candles_left, max_hold_bars,
                sl_price, tp_price, combined_score, p_super,
                entry_order_id_exchange, sl_order_id_exchange, tp_order_id_exchange,
                opened_at, meta
            ) VALUES (
                $1, $2, $3, $4, $5, $6, 1,
                $7, $8, $9, $10, $11,
                $12, $13, $14, $15,
                $16, $17, $18,
                now(), '{}'::jsonb
            )
            RETURNING id
            "#,
        )
        .bind(signal.symbol_id)
        .bind(&signal.symbol)
        .bind(signal.side.as_db_i16())
        .bind(filled_qty)
        .bind(filled_price)
        .bind(self.config.leverage as f32)
        .bind(PositionStatus::Open.as_i16())
        .bind(signal.tf_minutes)
        .bind(signal.signal_time)
        .bind(self.config.max_hold_bars)
        .bind(self.config.max_hold_bars)
        .bind(signal.sl_price)
        .bind(signal.tp_price)
        .bind(signal.combined_score)
        .bind(signal.p_super)
        .bind(entry_order_id)
        .bind(sl_order_id)
        .bind(tp_order_id)
        .fetch_one(&self.pool)
        .await
        .context("Failed to insert position")?;

        let id: i64 = row.get("id");
        Ok(id)
    }

    /// INSERT в trade.orders
    async fn insert_order_record(
        &self,
        signal: &QualifiedSignal,
        position_id: i64,
        order: &connections_lib::NewOrderResponse,
        order_type_str: &str,
        reduce_only: bool,
    ) -> Result<()> {
        let order_type_code: i16 = match order_type_str {
            "MARKET" => 1,
            "STOP_MARKET" => 2,
            "TAKE_PROFIT_MARKET" => 3,
            _ => 0,
        };
        let status_code: i16 = match order.status.as_str() {
            "NEW" => 1,
            "FILLED" => 2,
            "PARTIALLY_FILLED" => 3,
            "CANCELED" => 4,
            "EXPIRED" => 5,
            _ => 0,
        };

        let filled_price: Option<f64> = order.avg_price.parse().ok();
        let filled_qty: Option<f64> = order.executed_qty.parse().ok();
        let stop_price: Option<f64> = if order.stop_price.is_empty() { None } else { order.stop_price.parse().ok() };

        sqlx::query(
            r#"
            INSERT INTO trade.orders (
                order_id_exchange, client_order_id, symbol_id, symbol,
                side, order_type, reduce_only, qty, price, stop_price,
                status, tf_minutes, position_id, filled_price, filled_qty,
                meta
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15,
                '{}'::jsonb
            )
            "#,
        )
        .bind(order.order_id.to_string())
        .bind(&order.client_order_id)
        .bind(signal.symbol_id)
        .bind(&signal.symbol)
        .bind(signal.side.as_db_i16())
        .bind(order_type_code)
        .bind(reduce_only)
        .bind(order.orig_qty.parse::<f64>().unwrap_or(0.0))
        .bind(order.price.parse::<f64>().unwrap_or(0.0))
        .bind(stop_price)
        .bind(status_code)
        .bind(signal.tf_minutes)
        .bind(position_id)
        .bind(filled_price)
        .bind(filled_qty)
        .execute(&self.pool)
        .await
        .context("Failed to insert order")?;

        Ok(())
    }

    /// UPDATE trade.positions при закрытии
    async fn update_position_closed(
        &self,
        position_id: i64,
        exit_price: f64,
        realized_pnl: f64,
        realized_pnl_pct: f64,
        fees_total: f64,
        reason: CloseReason,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE trade.positions SET
                status = $1,
                exit_price = $2,
                realized_pnl = $3,
                realized_pnl_pct = $4,
                fees_total = $5,
                close_reason = $6,
                closed_at = now(),
                updated_at = now()
            WHERE id = $7
            "#,
        )
        .bind(PositionStatus::Closed.as_i16())
        .bind(exit_price)
        .bind(realized_pnl)
        .bind(realized_pnl_pct)
        .bind(fees_total)
        .bind(reason.as_str())
        .bind(position_id)
        .execute(&self.pool)
        .await
        .context("Failed to update position as closed")?;

        Ok(())
    }

    /// INSERT в trade.position_history
    async fn insert_position_history(
        &self,
        pos: &ManagedPosition,
        exit_price: f64,
        realized_pnl: f64,
        realized_pnl_pct: f64,
        fees_total: f64,
        reason: CloseReason,
        duration_bars: i16,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO trade.position_history (
                position_id, symbol, symbol_id, tf_minutes, side,
                entry_price, exit_price, qty, leverage,
                sl_price, tp_price,
                realized_pnl, realized_pnl_pct, fees_total,
                combined_score, p_super, close_reason,
                opened_at, closed_at, duration_bars, max_hold_bars,
                meta
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10, $11,
                $12, $13, $14,
                $15, $16, $17,
                $18, now(), $19, $20,
                '{}'::jsonb
            )
            "#,
        )
        .bind(pos.position_id)
        .bind(&pos.symbol)
        .bind(pos.symbol_id)
        .bind(pos.tf_minutes)
        .bind(pos.side.as_db_i16())
        .bind(pos.entry_price)
        .bind(exit_price)
        .bind(pos.qty)
        .bind(pos.leverage as f32)
        .bind(pos.sl_price)
        .bind(pos.tp_price)
        .bind(realized_pnl)
        .bind(realized_pnl_pct)
        .bind(fees_total)
        .bind(pos.combined_score)
        .bind(pos.p_super)
        .bind(reason.as_str())
        .bind(pos.opened_at)
        .bind(duration_bars)
        .bind(pos.max_hold_bars)
        .execute(&self.pool)
        .await
        .context("Failed to insert position history")?;

        Ok(())
    }

    /// Загрузить все открытые позиции из БД (при старте сервиса).
    pub async fn load_open_positions(&self) -> Result<Vec<ManagedPosition>> {
        let rows = sqlx::query(
            r#"
            SELECT
                id, symbol, symbol_id, tf_minutes, side,
                entry_price, sl_price, tp_price, qty, leverage,
                candles_left, max_hold_bars, combined_score, p_super,
                entry_order_id_exchange, sl_order_id_exchange, tp_order_id_exchange,
                opened_at
            FROM trade.positions
            WHERE status = 1
              AND tf_minutes IS NOT NULL
              AND symbol IS NOT NULL
            ORDER BY opened_at ASC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let positions: Vec<ManagedPosition> = rows
            .iter()
            .filter_map(|r| {
                let side_val: i16 = r.get("side");
                let side = Side::from_db_i16(side_val)?;
                let entry_price: f64 = r.get("entry_price");

                Some(ManagedPosition {
                    position_id: r.get("id"),
                    symbol: r.get("symbol"),
                    symbol_id: r.get("symbol_id"),
                    tf_minutes: r.get("tf_minutes"),
                    side,
                    entry_price,
                    sl_price: r.get("sl_price"),
                    tp_price: r.get("tp_price"),
                    qty: r.get("qty"),
                    leverage: {
                        let lev: f32 = r.get("leverage");
                        lev as u16
                    },
                    candles_left: r.get("candles_left"),
                    max_hold_bars: r.get("max_hold_bars"),
                    unrealized_pnl: 0.0,
                    unrealized_pnl_pct: 0.0,
                    current_price: entry_price,
                    opened_at: r.get("opened_at"),
                    combined_score: r.get("combined_score"),
                    p_super: r.get("p_super"),
                    entry_order_id: r.try_get("entry_order_id_exchange").ok(),
                    sl_order_id: r.try_get("sl_order_id_exchange").ok(),
                    tp_order_id: r.try_get("tp_order_id_exchange").ok(),
                })
            })
            .collect();

        info!("📂 Loaded {} open positions from database", positions.len());

        Ok(positions)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_quantity_calculation() {
        let price: f64 = 50000.0;
        let leverage = 10u16;
        let trade_size: f64 = 100.0;
        let notional = trade_size * leverage as f64;
        let qty = (notional / price * 1e8).floor() / 1e8;
        assert!((qty - 0.02_f64).abs() < 1e-8);
    }

    #[test]
    fn test_pnl_long() {
        let entry: f64 = 50000.0;
        let exit: f64 = 51000.0;
        let qty: f64 = 0.02;
        let pnl = 1.0_f64 * (exit - entry) * qty;
        let pnl_pct = 1.0_f64 * (exit - entry) / entry * 100.0;
        assert!((pnl - 20.0_f64).abs() < 0.01);
        assert!((pnl_pct - 2.0_f64).abs() < 0.01);
    }

    #[test]
    fn test_pnl_short() {
        let entry: f64 = 50000.0;
        let exit: f64 = 49000.0;
        let qty: f64 = 0.02;
        let pnl = -1.0_f64 * (exit - entry) * qty;
        let pnl_pct = -1.0_f64 * (exit - entry) / entry * 100.0;
        assert!((pnl - 20.0_f64).abs() < 0.01);
        assert!((pnl_pct - 2.0_f64).abs() < 0.01);
    }

    #[test]
    fn test_fee_calculation() {
        let entry_notional: f64 = 50000.0 * 0.02;
        let exit_notional: f64 = 51000.0 * 0.02;
        let taker_fee_pct: f64 = 0.04;
        let fees = (entry_notional + exit_notional) * taker_fee_pct / 100.0;
        assert!((fees - 0.808_f64).abs() < 0.001);
    }
}
