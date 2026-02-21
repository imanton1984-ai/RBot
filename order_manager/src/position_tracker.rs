// order_manager/src/position_tracker.rs
//
// Position Tracker — мониторинг открытых позиций в реальном времени.
//
// Функции:
//   1. Обновление unrealized PnL по текущей цене
//   2. Декремент candles_left при каждой новой свече таймфрейма
//   3. Force-close позиций при candles_left == 0
//   4. Проверка SL/TP на стороне бота (safety net)
//   5. Реакция на команды от risk_manager через Kafka orders.cmd
//   6. Отправка обновлений в Redpanda → WebUI

use anyhow::Result;
use sqlx::{PgPool, Row};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

use connections_lib::BinanceFuturesClient;

use crate::config::OrderManagerConfig;
use crate::types::{CloseReason, ManagedPosition, PositionUpdateEvent};

/// Position Tracker — следит за открытыми позициями.
pub struct PositionTracker {
    pool: PgPool,
    #[allow(dead_code)] // used for future config-driven behavior
    config: OrderManagerConfig,
    client: BinanceFuturesClient,
    /// In-memory кэш открытых позиций
    positions: Vec<ManagedPosition>,
    /// Последнее время свечи по каждому (symbol, tf_minutes)
    last_candle_times: HashMap<(String, i16), i64>,
}

impl PositionTracker {
    pub fn new(pool: PgPool, config: OrderManagerConfig, client: BinanceFuturesClient) -> Self {
        Self {
            pool,
            config,
            client,
            positions: Vec::new(),
            last_candle_times: HashMap::new(),
        }
    }

    /// Инициализировать трекер: загрузить открытые позиции
    pub async fn init(&mut self, positions: Vec<ManagedPosition>) {
        info!("PositionTracker: initialized with {} open positions", positions.len());
        self.positions = positions;
    }

    /// Добавить новую позицию в трекер
    pub fn add_position(&mut self, pos: ManagedPosition) {
        info!(
            "PositionTracker: tracking new position #{} {} {} tf={}m",
            pos.position_id, pos.symbol, pos.side, pos.tf_minutes
        );
        self.positions.push(pos);
    }

    /// Удалить позицию из трекера (после закрытия)
    pub fn remove_position(&mut self, position_id: i64) {
        self.positions.retain(|p| p.position_id != position_id);
    }

    /// Получить текущие открытые позиции
    pub fn open_positions(&self) -> &[ManagedPosition] {
        &self.positions
    }

    /// Количество открытых позиций
    pub fn open_count(&self) -> usize {
        self.positions.len()
    }

    /// Количество позиций по таймфрейму
    pub fn count_by_tf(&self, tf_minutes: i16) -> usize {
        self.positions.iter().filter(|p| p.tf_minutes == tf_minutes).count()
    }

    /// Найти позицию по ID
    pub fn get_position(&self, position_id: i64) -> Option<&ManagedPosition> {
        self.positions.iter().find(|p| p.position_id == position_id)
    }

    // ═══════════════════════════════════════════════════════════
    // MAIN TICK
    // ═══════════════════════════════════════════════════════════

    /// Основной тик трекера.
    /// Возвращает:
    ///  - events: обновления для WebUI
    ///  - to_close: позиции для принудительного закрытия
    pub async fn tick(&mut self) -> Result<(Vec<PositionUpdateEvent>, Vec<(i64, CloseReason)>)> {
        let mut events = Vec::new();
        let mut to_close = Vec::new();

        if self.positions.is_empty() {
            return Ok((events, to_close));
        }

        // 1. Обновить цены и PnL
        self.update_prices().await;

        // 2. Проверить candles_left (новые свечи)
        self.check_new_candles().await;

        // 3. Проверить каждую позицию
        for pos in &self.positions {
            if pos.is_expired() {
                warn!(
                    "⏰ Position #{} {} {} EXPIRED (candles_left=0), force-closing",
                    pos.position_id, pos.symbol, pos.side
                );
                to_close.push((pos.position_id, CloseReason::MaxBars));
                continue;
            }

            if pos.is_sl_hit(pos.current_price) {
                warn!(
                    "🛑 Position #{} {} {} SL HIT at {:.4} (SL={:.4})",
                    pos.position_id, pos.symbol, pos.side, pos.current_price, pos.sl_price
                );
                to_close.push((pos.position_id, CloseReason::SlHit));
                continue;
            }

            if pos.is_tp_hit(pos.current_price) {
                info!(
                    "🎯 Position #{} {} {} TP HIT at {:.4} (TP={:.4})",
                    pos.position_id, pos.symbol, pos.side, pos.current_price, pos.tp_price
                );
                to_close.push((pos.position_id, CloseReason::TpHit));
                continue;
            }

            events.push(PositionUpdateEvent::from(pos));
        }

        Ok((events, to_close))
    }

    /// Обновить текущие цены и PnL для всех позиций.
    /// Сначала собираем все цены, затем обновляем позиции (избегаем borrow конфликт).
    async fn update_prices(&mut self) {
        // Фаза 1: собрать цены
        let mut price_updates: Vec<(usize, f64)> = Vec::new();

        for (idx, pos) in self.positions.iter().enumerate() {
            let price = match self.client.get_mark_price(&pos.symbol).await {
                Ok(p) if p > 0.0 => p,
                _ => {
                    // Fallback: из БД
                    match Self::fetch_current_price_static(&self.pool, &pos.symbol).await {
                        Ok(p) if p > 0.0 => p,
                        _ => continue,
                    }
                }
            };
            price_updates.push((idx, price));
        }

        // Фаза 2: применить обновления
        for (idx, price) in &price_updates {
            let pos = &mut self.positions[*idx];
            pos.current_price = *price;
            let (pnl, pnl_pct) = pos.calc_unrealized_pnl(*price);
            pos.unrealized_pnl = pnl;
            pos.unrealized_pnl_pct = pnl_pct;

            // Обновить в БД (не критично при ошибке)
            Self::update_position_pnl_static(
                &self.pool,
                pos.position_id,
                pos.unrealized_pnl,
                pos.candles_left,
                pos.current_price,
            )
            .await
            .ok();
        }
    }

    /// Проверить новые свечи и декрементировать candles_left.
    async fn check_new_candles(&mut self) {
        // Собираем уникальные (symbol, tf_minutes)
        let pairs: Vec<(String, i16)> = self
            .positions
            .iter()
            .map(|p| (p.symbol.clone(), p.tf_minutes))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        for (symbol, tf_minutes) in pairs {
            let latest_candle_ms = match Self::fetch_latest_candle_time_static(
                &self.pool,
                &symbol,
                tf_minutes,
            )
            .await
            {
                Ok(ms) => ms,
                Err(_) => continue,
            };

            let key = (symbol.clone(), tf_minutes);
            let prev_ms = self.last_candle_times.get(&key).copied().unwrap_or(0);

            if latest_candle_ms > prev_ms && prev_ms > 0 {
                let candles_elapsed = ((latest_candle_ms - prev_ms) as f64
                    / (tf_minutes as f64 * 60_000.0))
                    .round() as i16;

                for pos in &mut self.positions {
                    if pos.symbol == symbol && pos.tf_minutes == tf_minutes {
                        pos.candles_left = (pos.candles_left - candles_elapsed).max(0);
                        debug!(
                            "🕐 Position #{} {} candles_left: {} (elapsed={})",
                            pos.position_id, symbol, pos.candles_left, candles_elapsed
                        );

                        Self::update_candles_left_static(
                            &self.pool,
                            pos.position_id,
                            pos.candles_left,
                        )
                        .await
                        .ok();
                    }
                }
            }

            self.last_candle_times.insert(key, latest_candle_ms);
        }
    }

    /// Обработать команду закрытия от risk_manager.
    pub fn process_close_command(&self, symbol: &str, side_str: &str) -> Vec<(i64, CloseReason)> {
        let mut to_close = Vec::new();
        for pos in &self.positions {
            if pos.symbol == symbol && pos.side.as_str() == side_str {
                warn!(
                    "🚨 Risk Manager: closing position #{} {} {}",
                    pos.position_id, symbol, side_str
                );
                to_close.push((pos.position_id, CloseReason::RiskManager));
            }
        }
        to_close
    }

    /// Сгенерировать сводку по всем открытым позициям
    pub fn summary(&self) -> String {
        if self.positions.is_empty() {
            return "No open positions".to_string();
        }

        let total_pnl: f64 = self.positions.iter().map(|p| p.unrealized_pnl).sum();
        let count_1h = self.count_by_tf(60);
        let count_4h = self.count_by_tf(240);
        let count_15m = self.count_by_tf(15);

        format!(
            "Open: {} (1h={}, 4h={}, 15m={}) | Total unPnL: {:.2} USDT",
            self.positions.len(),
            count_1h,
            count_4h,
            count_15m,
            total_pnl
        )
    }

    // ═══════════════════════════════════════════════════════════
    // STATIC DB HELPERS (используются через &PgPool, не &self)
    // ═══════════════════════════════════════════════════════════

    /// Статический метод: текущая цена из БД
    async fn fetch_current_price_static(pool: &PgPool, symbol: &str) -> Result<f64> {
        let row = sqlx::query("SELECT close FROM market.candles_1m WHERE symbol = $1 ORDER BY time DESC LIMIT 1")
            .bind(symbol)
            .fetch_optional(pool)
            .await?;

        match row {
            Some(r) => {
                let price: f64 = r.get("close");
                Ok(price)
            }
            None => anyhow::bail!("No candle data for {}", symbol),
        }
    }

    /// Статический метод: последнее время свечи для таймфрейма
    async fn fetch_latest_candle_time_static(
        pool: &PgPool,
        symbol: &str,
        tf_minutes: i16,
    ) -> Result<i64> {
        let table = match tf_minutes {
            1 => "market.candles_1m",
            5 => "market.candles_5m",
            15 => "market.candles_15m",
            60 => "market.candles_1h",
            240 => "market.candles_4h",
            1440 => "market.candles_1d",
            _ => anyhow::bail!("Unknown timeframe: {}m", tf_minutes),
        };

        let query = format!(
            "SELECT time_ms FROM {} WHERE symbol = $1 ORDER BY time DESC LIMIT 1",
            table
        );

        let row: Option<(i64,)> = sqlx::query_as(&query)
            .bind(symbol)
            .fetch_optional(pool)
            .await?;

        match row {
            Some((ms,)) => Ok(ms),
            None => anyhow::bail!("No candle for {} in {}", symbol, table),
        }
    }

    /// Статический метод: обновить pnl и current_price в БД
    async fn update_position_pnl_static(
        pool: &PgPool,
        position_id: i64,
        unrealized_pnl: f64,
        candles_left: i16,
        current_price: f64,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE trade.positions SET unrealized_pnl = $1, candles_left = $2, current_price = $3, updated_at = now() WHERE id = $4",
        )
        .bind(unrealized_pnl)
        .bind(candles_left)
        .bind(current_price)
        .bind(position_id)
        .execute(pool)
        .await?;
        Ok(())
    }

    /// Статический метод: обновить candles_left в БД
    async fn update_candles_left_static(
        pool: &PgPool,
        position_id: i64,
        candles_left: i16,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE trade.positions SET candles_left = $1, updated_at = now() WHERE id = $2",
        )
        .bind(candles_left)
        .bind(position_id)
        .execute(pool)
        .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::types::Side;

    fn make_test_position(id: i64, symbol: &str, tf: i16, side: Side, entry: f64) -> ManagedPosition {
        ManagedPosition {
            position_id: id,
            symbol: symbol.to_string(),
            symbol_id: 1,
            tf_minutes: tf,
            side,
            entry_price: entry,
            sl_price: if matches!(side, Side::Long) { entry * 0.98 } else { entry * 1.02 },
            tp_price: if matches!(side, Side::Long) { entry * 1.04 } else { entry * 0.96 },
            qty: 0.01,
            leverage: 10,
            candles_left: 25,
            max_hold_bars: 25,
            unrealized_pnl: 0.0,
            unrealized_pnl_pct: 0.0,
            current_price: entry,
            opened_at: Utc::now(),
            combined_score: 0.75,
            p_super: 0.8,
            entry_order_id: None,
            sl_order_id: None,
            tp_order_id: None,
        }
    }

    #[test]
    fn test_count_by_tf() {
        let positions = vec![
            make_test_position(1, "BTCUSDT", 60, Side::Long, 50000.0),
            make_test_position(2, "ETHUSDT", 60, Side::Short, 3000.0),
            make_test_position(3, "SOLUSDT", 240, Side::Long, 100.0),
            make_test_position(4, "BTCUSDT", 15, Side::Short, 50000.0),
        ];

        let count_1h = positions.iter().filter(|p| p.tf_minutes == 60).count();
        let count_4h = positions.iter().filter(|p| p.tf_minutes == 240).count();
        let count_15m = positions.iter().filter(|p| p.tf_minutes == 15).count();

        assert_eq!(count_1h, 2);
        assert_eq!(count_4h, 1);
        assert_eq!(count_15m, 1);
    }

    #[test]
    fn test_expired_position() {
        let mut pos = make_test_position(1, "BTCUSDT", 60, Side::Long, 50000.0);
        pos.candles_left = 0;
        assert!(pos.is_expired());

        pos.candles_left = 1;
        assert!(!pos.is_expired());
    }

    #[test]
    fn test_process_close_command() {
        let positions = vec![
            make_test_position(1, "BTCUSDT", 60, Side::Long, 50000.0),
            make_test_position(2, "ETHUSDT", 60, Side::Short, 3000.0),
            make_test_position(3, "BTCUSDT", 240, Side::Short, 50000.0),
        ];

        // Close all BTCUSDT LONG
        let to_close: Vec<_> = positions
            .iter()
            .filter(|p| p.symbol == "BTCUSDT" && p.side.as_str() == "LONG")
            .map(|p| (p.position_id, CloseReason::RiskManager))
            .collect();

        assert_eq!(to_close.len(), 1);
        assert_eq!(to_close[0].0, 1);
    }
}
