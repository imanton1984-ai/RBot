// database/src/signal_archiver.rs
//
// Signal Archiver — автоматическая джоба для управления таблицей trade.super_entry_signals.
//
// Задачи:
//   1. Каждый час: перемещает сигналы старше 24 часов из trade.super_entry_signals
//      в trade.super_entry_outdated.
//   2. Retention policy: в trade.super_entry_outdated хранит максимум 60,000 сигналов
//      на каждый таймфрейм (tf_minutes). Старые удаляются.

use sqlx::PgPool;
use tracing::{info, warn};

/// Переместить сигналы старше 24 часов в trade.super_entry_outdated.
/// Возвращает количество перемещённых строк.
pub async fn archive_old_signals(pool: &PgPool) -> Result<u64, sqlx::Error> {
    // 1. INSERT INTO outdated SELECT FROM signals WHERE time < now() - 24h
    //    ON CONFLICT — обновляем moved_at (перезаписываем если дубль).
    let inserted = sqlx::query(
        "INSERT INTO trade.super_entry_outdated 
            (time, time_ms, symbol, symbol_id, tf_minutes, side,
             entry_price, sl_price, tp_price, p_super, p_long,
             combined_score, dir_confidence, strategy, reason, created_at, moved_at)
         SELECT 
             time, time_ms, symbol, symbol_id, tf_minutes, side,
             entry_price, sl_price, tp_price, p_super, p_long,
             combined_score, dir_confidence, strategy, reason, created_at, now()
         FROM trade.super_entry_signals
         WHERE time < now() - INTERVAL '24 hours'
         ON CONFLICT (symbol_id, tf_minutes, time) DO UPDATE SET moved_at = now()"
    )
    .execute(pool)
    .await?;

    let moved_count = inserted.rows_affected();

    if moved_count > 0 {
        // 2. DELETE from active signals table
        let deleted = sqlx::query(
            "DELETE FROM trade.super_entry_signals
             WHERE time < now() - INTERVAL '24 hours'"
        )
        .execute(pool)
        .await?;

        info!(
            "📦 Signal archiver: moved {} signals to outdated, deleted {} from active",
            moved_count,
            deleted.rows_affected()
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

/// Запустить фоновую задачу, которая каждые 5 минут:
///   1. Перемещает сигналы старше 24 часов
///   2. Применяет retention policy (60,000 на tf)
///
/// Интервал 5 мин (вместо 1 часа) гарантирует, что super_entry_signals
/// содержит только свежие сигналы (<24h), а старые быстро попадают в outdated.
pub async fn run_signal_archiver_loop(pool: PgPool) {
    info!("🔄 Signal archiver started (every 5 min)");

    loop {
        // 1. Archive signals > 24h old
        match archive_old_signals(&pool).await {
            Ok(count) => {
                if count > 0 {
                    info!("📦 Archived {} old signals", count);
                }
            }
            Err(e) => {
                warn!("⚠️ Signal archiver: archive error: {}", e);
            }
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

        // Sleep for 5 minutes (frequent enough to keep super_entry_signals clean)
        tokio::time::sleep(std::time::Duration::from_secs(300)).await;
    }
}
