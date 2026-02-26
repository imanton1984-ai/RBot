// order_manager/src/signal_scanner.rs
//
// Signal Scanner — сканирует trade.super_entry_signals на предмет свежих,
// квалифицированных сигналов для открытия позиций.
//
// Логика фильтрации:
//   1. combined_score в диапазоне [signal_score_min_X, signal_score_max_X] для каждого таймфрейма
//   2. Таймфреймы: 1m, 5m, 15m, 1h, 4h, 1d
//   3. Только сигналы в окне lookback (для 4h смотрим -4ч от текущего времени)
//   4. Проверка дрифта цены: |current_price - entry_price| / entry_price <= max_price_drift_pct
//   5. Не используем сигнал, если для него уже открыта позиция

use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{PgPool, Row};
use tracing::{debug, info, warn};

use crate::config::OrderManagerConfig;
use crate::types::{QualifiedSignal, Side, TimeframeAllocation};

/// Signal Scanner — периодически опрашивает БД на предмет новых сигналов.
pub struct SignalScanner {
    pool: PgPool,
    config: OrderManagerConfig,
}

/// Сырая строка из trade.super_entry_signals
#[derive(Debug)]
struct RawSignalRow {
    time: DateTime<Utc>,
    time_ms: i64,
    symbol: String,
    symbol_id: i64,
    tf_minutes: i16,
    side: i16,
    entry_price: f64,
    sl_price: f64,
    tp_price: f64,
    p_super: f32,
    combined_score: f32,
}

impl SignalScanner {
    pub fn new(pool: PgPool, config: OrderManagerConfig) -> Self {
        Self { pool, config }
    }

    /// Основной метод: найти все квалифицированные сигналы для указанных таймфреймов.
    ///
    /// Возвращает вектор `QualifiedSignal`, отсортированный по combined_score DESC.
    pub async fn scan_for_signals(&self, needed_timeframes: &[(i16, u16)]) -> Result<Vec<QualifiedSignal>> {
        let mut all_signals = Vec::new();

        for &(tf_minutes, needed_count) in needed_timeframes {
            if needed_count == 0 {
                continue;
            }

            let signals = self.scan_timeframe(tf_minutes, needed_count).await?;
            all_signals.extend(signals);
        }

        // Сортировка по убыванию combined_score (лучшие сигналы первыми)
        all_signals.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if !all_signals.is_empty() {
            info!(
                "📡 SignalScanner: found {} qualified signals across timeframes",
                all_signals.len()
            );
        }

        Ok(all_signals)
    }

    /// Сканировать один таймфрейм.
    async fn scan_timeframe(&self, tf_minutes: i16, max_count: u16) -> Result<Vec<QualifiedSignal>> {
        let lookback_minutes = self.config.lookback_minutes_for_tf(tf_minutes);

        // 1. Получить сырые сигналы из БД
        let raw_signals = self.fetch_raw_signals(tf_minutes, lookback_minutes, max_count * 3).await?;

        if raw_signals.is_empty() {
            debug!(
                "SignalScanner: no raw signals for tf={}m in last {} min",
                tf_minutes, lookback_minutes
            );
            return Ok(Vec::new());
        }

        // 2. Get symbols that already have ANY open position (across ALL timeframes)
        let used_symbols = self.fetch_open_position_symbols().await?;

        // 3. Для каждого сигнала проверить актуальность цены
        let mut qualified = Vec::new();

        for raw in &raw_signals {
            // Skip if this symbol already has an open position on ANY timeframe
            if used_symbols.contains(&raw.symbol) {
                debug!(
                    "SignalScanner: skipping {} tf={}m — already has open position for this symbol",
                    raw.symbol, tf_minutes
                );
                continue;
            }

            // Текущая цена из последней 1m свечи
            let current_price = match self.fetch_current_price(&raw.symbol).await {
                Ok(price) if price > 0.0 => price,
                Ok(_) => {
                    warn!(
                        "SignalScanner: zero price for {}, skipping signal",
                        raw.symbol
                    );
                    continue;
                }
                Err(e) => {
                    warn!(
                        "SignalScanner: failed to get price for {}: {}",
                        raw.symbol, e
                    );
                    continue;
                }
            };

            // Проверка дрифта цены
            let drift_pct = ((current_price - raw.entry_price) / raw.entry_price * 100.0).abs();
            if drift_pct > self.config.max_price_drift_pct {
                debug!(
                    "SignalScanner: {} tf={}m price drift {:.3}% > max {:.1}%, skipping",
                    raw.symbol, tf_minutes, drift_pct, self.config.max_price_drift_pct
                );
                continue;
            }

            let side = match Side::from_db_i16(raw.side) {
                Some(s) => s,
                None => {
                    warn!(
                        "SignalScanner: unknown side={} for {} tf={}m",
                        raw.side, raw.symbol, tf_minutes
                    );
                    continue;
                }
            };

            qualified.push(QualifiedSignal {
                signal_time: raw.time,
                signal_time_ms: raw.time_ms,
                symbol: raw.symbol.clone(),
                symbol_id: raw.symbol_id,
                tf_minutes: raw.tf_minutes,
                side,
                entry_price: raw.entry_price,
                sl_price: raw.sl_price,
                tp_price: raw.tp_price,
                p_super: raw.p_super,
                combined_score: raw.combined_score,
                current_price,
                price_drift_pct: drift_pct,
            });

            if qualified.len() >= max_count as usize {
                break;
            }
        }

        debug!(
            "SignalScanner: tf={}m → {} raw → {} qualified",
            tf_minutes,
            raw_signals.len(),
            qualified.len()
        );

        Ok(qualified)
    }

    /// Получить сырые сигналы из trade.super_entry_signals
    async fn fetch_raw_signals(
        &self,
        tf_minutes: i16,
        lookback_minutes: i64,
        limit: u16,
    ) -> Result<Vec<RawSignalRow>> {
        // Get per-timeframe score range
        let (score_min, score_max) = self.get_score_range_for_tf(tf_minutes);

        let rows = sqlx::query(
            r#"
            SELECT
                time, time_ms, symbol, symbol_id, tf_minutes,
                side, entry_price, sl_price, tp_price,
                p_super, combined_score
            FROM trade.super_entry_signals
            WHERE tf_minutes = $1
              AND combined_score >= $2
              AND combined_score <= $3
              AND time >= now() - make_interval(mins => $4::int)
            ORDER BY combined_score DESC, time DESC
            LIMIT $5
            "#,
        )
        .bind(tf_minutes)
        .bind(score_min)
        .bind(score_max)
        .bind(lookback_minutes as i32)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let signals: Vec<RawSignalRow> = rows
            .iter()
            .map(|row| RawSignalRow {
                time: row.get("time"),
                time_ms: row.get("time_ms"),
                symbol: row.get("symbol"),
                symbol_id: row.get("symbol_id"),
                tf_minutes: row.get("tf_minutes"),
                side: row.get("side"),
                entry_price: row.get("entry_price"),
                sl_price: row.get("sl_price"),
                tp_price: row.get("tp_price"),
                p_super: row.get("p_super"),
                combined_score: row.get("combined_score"),
            })
            .collect();

        Ok(signals)
    }

    /// Get per-timeframe score range from config
    fn get_score_range_for_tf(&self, tf_minutes: i16) -> (f32, f32) {
        match tf_minutes {
            1 => (self.config.signal_score_min_1m, self.config.signal_score_max_1m),
            5 => (self.config.signal_score_min_5m, self.config.signal_score_max_5m),
            15 => (self.config.signal_score_min_15m, self.config.signal_score_max_15m),
            60 => (self.config.signal_score_min_1h, self.config.signal_score_max_1h),
            240 => (self.config.signal_score_min_4h, self.config.signal_score_max_4h),
            1440 => (self.config.signal_score_min_1d, self.config.signal_score_max_1d),
            _ => (0.70, 0.80), // default fallback
        }
    }

    /// Получить все символы, для которых уже есть ОТКРЫТАЯ позиция (на ЛЮБОМ TF).
    /// Prevents opening duplicate positions for the same coin.
    async fn fetch_open_position_symbols(&self) -> Result<Vec<String>> {
        let rows = sqlx::query(
            r#"
            SELECT DISTINCT COALESCE(symbol, '') as symbol
            FROM trade.positions
            WHERE status = 1
              AND symbol IS NOT NULL
              AND symbol != ''
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| {
                let s: String = r.get("symbol");
                s
            })
            .collect())
    }

    /// Получить текущую рыночную цену для символа (close последней 1m свечи)
    async fn fetch_current_price(&self, symbol: &str) -> Result<f64> {
        let row = sqlx::query(
            "SELECT close FROM market.candles_1m WHERE symbol = $1 ORDER BY time DESC LIMIT 1",
        )
        .bind(symbol)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(r) => {
                let price: f64 = r.get("close");
                Ok(price)
            }
            None => anyhow::bail!("No candle data found for {}", symbol),
        }
    }

    /// Получить количество открытых позиций по таймфрейму
    pub async fn count_open_positions_by_tf(&self) -> Result<Vec<(i16, i64)>> {
        let rows = sqlx::query(
            r#"
            SELECT tf_minutes, COUNT(*) as cnt
            FROM trade.positions
            WHERE status = 1
              AND tf_minutes IS NOT NULL
            GROUP BY tf_minutes
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| {
                let tf: i16 = r.get("tf_minutes");
                let cnt: i64 = r.get("cnt");
                (tf, cnt)
            })
            .collect())
    }

    /// Определить, для каких таймфреймов нужны новые позиции.
    pub async fn needed_slots(
        &self,
        allocation: &TimeframeAllocation,
    ) -> Result<Vec<(i16, u16)>> {
        let current = self.count_open_positions_by_tf().await?;
        let mut needed = Vec::new();

        for &(tf, target_slots) in &allocation.slots {
            let current_count = current
                .iter()
                .find(|(t, _)| *t == tf)
                .map(|(_, c)| *c as u16)
                .unwrap_or(0);

            if current_count < target_slots {
                let gap = target_slots - current_count;
                needed.push((tf, gap));
                debug!(
                    "Slot gap: tf={}m → have={}, target={}, need={}",
                    tf, current_count, target_slots, gap
                );
            }
        }

        Ok(needed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_side_mapping() {
        assert_eq!(Side::from_db_i16(1), Some(Side::Long));
        assert_eq!(Side::from_db_i16(-1), Some(Side::Short));
        assert_eq!(Side::from_db_i16(0), None);
    }

    #[test]
    fn test_price_drift_calculation() {
        let entry: f64 = 50000.0;
        let current: f64 = 50080.0;
        let drift: f64 = ((current - entry) / entry * 100.0).abs();
        assert!((drift - 0.16_f64).abs() < 0.01);
        assert!(drift <= 0.2);
    }

    #[test]
    fn test_price_drift_too_large() {
        let entry: f64 = 50000.0;
        let current: f64 = 50150.0;
        let drift: f64 = ((current - entry) / entry * 100.0).abs();
        assert!((drift - 0.3_f64).abs() < 0.01);
        assert!(drift > 0.2);
    }
}
