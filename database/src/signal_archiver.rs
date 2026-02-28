// database/src/signal_archiver.rs
//
// Signal Archiver — автоматическая джоба для управления таблицей trade.super_entry_signals.
//
// Задачи:
//   1. Каждые 5 минут: перемещает сигналы старше 24 часов из trade.super_entry_signals
//      в trade.super_entry_outdated, используя DELETE..RETURNING + INSERT (zero-copy move).
//   2. Retention policy: в trade.super_entry_outdated хранит максимум 60,000 сигналов
//      на каждый таймфрейм (tf_minutes). Старые удаляются.
//   3. Очистка candles_live: удаляет устаревшие свечи старше 6 часов (таблица-кеш).

use sqlx::PgPool;
use tracing::{info, warn, debug};

/// Переместить сигналы старше 24 часов в trade.super_entry_outdated.
/// Использует DELETE...RETURNING + INSERT вместо INSERT ON CONFLICT DO UPDATE
/// для избежания повторного обновления уже перемещённых строк.
///
/// PERF: Предыдущая версия делала INSERT ON CONFLICT DO UPDATE для ВСЕХ строк
/// старше 24ч (включая те, что уже были перемещены), вызывая 356K+ обновлений
/// за ~9 секунд. Новая версия перемещает только свежие строки за один проход.
pub async fn archive_old_signals(pool: &PgPool) -> Result<u64, sqlx::Error> {
    // Атомарное перемещение: DELETE из active → INSERT в outdated.
    // CTE with DELETE...RETURNING гарантирует zero-copy move.
    // LIMIT 10000 — батчинг для избежания блокировки таблицы на длительное время.
    let result = sqlx::query(
        "WITH moved AS (
             DELETE FROM trade.super_entry_signals
             WHERE ctid IN (
                 SELECT ctid FROM trade.super_entry_signals
                 WHERE time < now() - INTERVAL '24 hours'
                 LIMIT 10000
             )
             RETURNING time, time_ms, symbol, symbol_id, tf_minutes, side,
                       entry_price, sl_price, tp_price, p_super, p_long,
                       combined_score, dir_confidence, strategy, reason, created_at
         )
         INSERT INTO trade.super_entry_outdated
             (time, time_ms, symbol, symbol_id, tf_minutes, side,
              entry_price, sl_price, tp_price, p_super, p_long,
              combined_score, dir_confidence, strategy, reason, created_at, moved_at)
         SELECT time, time_ms, symbol, symbol_id, tf_minutes, side,
                entry_price, sl_price, tp_price, p_super, p_long,
                combined_score, dir_confidence, strategy, reason, created_at, now()
         FROM moved
         ON CONFLICT (symbol_id, tf_minutes, time) DO NOTHING"
    )
    .execute(pool)
    .await?;

    let moved_count = result.rows_affected();

    if moved_count > 0 {
        info!(
            "📦 Signal archiver: moved {} signals to outdated (batch)",
            moved_count
        );
    }

    Ok(moved_count)
}

/// Retention policy: удалить старые сигналы из trade.super_entry_outdated,
/// оставив не более `max_per_tf` (= 60,000) записей на каждый таймфрейм.
///
/// Стратегия: для каждого tf_minutes определяем time 60,000-й записи (по убыванию),
/// и удаляем всё, что старше.
pub async fn enforce_retention_policy(pool: &PgPool, max_per_tf: i64) -> Result<u64, sqlx::Error> {
    // Получить все уникальные таймфреймы
    let tfs: Vec<(i16,)> = sqlx::query_as(
        "SELECT DISTINCT tf_minutes FROM trade.super_entry_outdated ORDER BY tf_minutes"
    )
    .fetch_all(pool)
    .await?;

    let mut total_deleted: u64 = 0;

    for (tf,) in tfs {
        // Найти порог времени: time 60,000-й записи
        let cutoff: Option<(chrono::DateTime<chrono::Utc>,)> = sqlx::query_as(
            "SELECT time FROM trade.super_entry_outdated
             WHERE tf_minutes = $1
             ORDER BY time DESC
             OFFSET $2
             LIMIT 1"
        )
        .bind(tf)
        .bind(max_per_tf)
        .fetch_optional(pool)
        .await?;

        if let Some((cutoff_time,)) = cutoff {
            let deleted = sqlx::query(
                "DELETE FROM trade.super_entry_outdated
                 WHERE tf_minutes = $1 AND time < $2"
            )
            .bind(tf)
            .bind(cutoff_time)
            .execute(pool)
            .await?;

            let count = deleted.rows_affected();
            if count > 0 {
                info!(
                    "🗑️  Retention: deleted {} outdated signals for tf={}m (cutoff={})",
                    count, tf, cutoff_time
                );
                total_deleted += count;
            }
        }
    }

    Ok(total_deleted)
}

/// Очистка candles_live — удаляет свечи старше 6 часов.
/// Таблица candles_live — горячий кеш, не историческая таблица.
/// Без очистки она растёт бесконтрольно, вызывая медленные запросы.
///
/// PERF: Без этой очистки таблица candles_live за месяц накапливает миллионы строк,
/// и любой запрос к ней (LATERAL JOIN, broadcaster poll) занимает >2 секунд.
pub async fn cleanup_candles_live(pool: &PgPool) -> Result<u64, sqlx::Error> {
    // 6 часов достаточно: UI показывает только текущую свечу (1m-1h),
    // а 4h/1d используют fallback через последнюю 1m свечу.
    let cutoff_ms = chrono::Utc::now().timestamp_millis() - 6 * 3600 * 1000;

    let deleted = sqlx::query(
        "DELETE FROM market.candles_live
         WHERE open_time_ms < $1"
    )
    .bind(cutoff_ms)
    .execute(pool)
    .await?;

    let count = deleted.rows_affected();
    if count > 0 {
        info!(
            "🧹 Cleaned {} old candles from candles_live (cutoff < 6h ago)",
            count
        );
    }

    Ok(count)
}

/// Запустить фоновую задачу, которая каждые 5 минут:
///   1. Перемещает сигналы старше 24 часов (батчами по 10K)
///   2. Применяет retention policy (60,000 на tf)
///   3. Очищает candles_live (удаляет свечи старше 6 часов)
///
/// Батчинг: archive_old_signals перемещает до 10,000 строк за вызов.
/// Если за 24ч накопилось больше — следующий цикл (через 5 минут) подхватит остаток.
pub async fn run_signal_archiver_loop(pool: PgPool) {
    info!("🔄 Signal archiver started (every 5 min)");

    loop {
        // 1. Archive signals > 24h old (repeat until all moved)
        let mut total_archived = 0u64;
        loop {
            match archive_old_signals(&pool).await {
                Ok(count) => {
                    total_archived += count;
                    if count < 10_000 {
                        // < batch size means no more rows to move
                        break;
                    }
                    // More rows may exist — continue immediately
                    debug!("📦 Batch archived {}; continuing...", count);
                }
                Err(e) => {
                    warn!("⚠️ Signal archiver: archive error: {}", e);
                    break;
                }
            }
        }
        if total_archived > 0 {
            info!("📦 Total archived: {} signals this cycle", total_archived);
        }

        // 2. Enforce retention (60,000 per tf)
        match enforce_retention_policy(&pool, 60_000).await {
            Ok(count) => {
                if count > 0 {
                    info!("🗑️  Cleaned {} signals exceeding retention limit", count);
                }
            }
            Err(e) => {
                warn!("⚠️ Signal archiver: retention error: {}", e);
            }
        }

        // 3. Cleanup candles_live (remove stale candles > 6h old)
        match cleanup_candles_live(&pool).await {
            Ok(_) => {}
            Err(e) => {
                warn!("⚠️ Signal archiver: candles_live cleanup error: {}", e);
            }
        }

        // Sleep for 5 minutes
        tokio::time::sleep(std::time::Duration::from_secs(300)).await;
    }
}
