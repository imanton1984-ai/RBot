# Plan: Super Entry Production Integration

## Текущее состояние

Super Entry стратегия работает ТОЛЬКО как оффлайн-бектест:
- Датасет строится из БД
- Модели обучаются
- Бектест запускается, показывает метрики
- **Сигналы НЕ пишутся в БД**
- **Нет realtime-сервиса**

## Цель

При `./startGPU.sh --super-entry` или `SUPER_ENTRY_ENABLED=true`:
1. Базовый пайплайн запускается как обычно (connections → ingestor → compute)
2. Дополнительно запускается `super_entry_service` — новый сервис который:
   - Читает индикаторы из `market.indicators_wide`
   - Запускает ML inference (P(super) + P(direction))
   - Пишет сигналы в `trade.super_entry_signals` (новая таблица)
   - Работает как в history (backfill), так и в realtime (Kafka events)

## Архитектура

### Что НЕ меняем
- `connections` — как есть
- `ingestor` — как есть
- `compute_history` / `compute_realtime` — как есть
  Они уже производят всё что Super Entry нужно: candles + indicators_wide

### Что создаём

#### 1. Новая таблица: `trade.super_entry_signals`

```sql
CREATE TABLE trade.super_entry_signals (
    time            TIMESTAMPTZ NOT NULL,
    time_ms         BIGINT NOT NULL,
    symbol          TEXT NOT NULL,
    symbol_id       BIGINT NOT NULL,
    tf_minutes      SMALLINT NOT NULL,
    side            SMALLINT NOT NULL,       -- 1=LONG, -1=SHORT
    entry_price     FLOAT8 NOT NULL,
    sl_price        FLOAT8 NOT NULL,
    tp_price        FLOAT8 NOT NULL,
    p_super         REAL NOT NULL,           -- P(super move)
    p_long          REAL NOT NULL,           -- P(direction=LONG)
    combined_score  REAL NOT NULL,           -- combined scorer output
    dir_confidence  REAL NOT NULL,           -- directional confidence
    strategy        TEXT NOT NULL DEFAULT 'super_entry_v1',
    reason          JSONB,                   -- metadata
    created_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (symbol_id, tf_minutes, time)
);
```

#### 2. Новый бинарник: `super_entry_service`

Путь: `strategies/ml_entry_strategy/src/bin/service.rs`

Два режима:
- **History mode**: Сканирует `market.indicators_wide` за весь период, генерит сигналы
- **Realtime mode**: Слушает Kafka topic `candles.close`, на каждое событие запускает inference

```
super_entry_service
  ├── startup: load XGBoost models
  ├── history_backfill(): scan indicators_wide → inference → write super_entry_signals
  └── realtime_loop(): kafka consumer → on candle close → fetch indicators → inference → write
```

#### 3. Изменения в `scripts/run.sh`

Добавить условный запуск:
```bash
# 8. Super Entry Service (if enabled)
if [ "${SUPER_ENTRY_ENABLED:-false}" = "true" ]; then
    log "Starting super_entry_service..."
    start_svc "super_entry_service" "super_entry_service"
fi
```

### Что Super Entry НЕ использует (можно пропускать)
- Raw signals (`trade.raw_signals`) — не нужны
- Predictors (`trade.predictors` — price/levels ML) — не нужны
- Entry Policy / Signal Quality — не нужны
- Trade Signals (`trade.final_signals`) — Super Entry пишет в свою таблицу

### Что Super Entry ИСПОЛЬЗУЕТ
- `market.pairs` — список активных пар
- `market.candles_*` — OHLCV данные
- `market.indicators_wide` — рассчитанные индикаторы (RSI, MACD, ATR и т.д.)
- `models/super_entry_v1_tf*.ubj` — обученные модели

## Задачи реализации

1. Создать таблицу `trade.super_entry_signals` (в database/init_db)
2. Создать `strategies/ml_entry_strategy/src/bin/service.rs`:
   - History backfill mode: scan indicators → inference → batch INSERT
   - Realtime mode: Kafka consumer → inference → INSERT
3. Создать `strategies/ml_entry_strategy/src/db_writer.rs`:
   - Batch INSERT в trade.super_entry_signals через UNNEST
4. Обновить `scripts/run.sh` — запуск super_entry_service при флаге
5. Обновить `scripts/stop.sh` — остановка super_entry_service (уже есть pgrep)
6. Обновить `strategy_switcher.rs` — интеграция с run.sh через env vars

## Преимущества подхода
- **Не ломает существующий пайплайн** — всё работает как раньше
- **Простое добавление новых стратегий** — каждая стратегия = отдельный сервис
- **Независимость** — Super Entry можно включать/выключать без перезапуска основного pipeline
- **Масштабируемость** — в будущем Combined mode просто запускает оба сервиса

## Вопросы для рассмотрения
- Нужно ли Super Entry для 1d TF? (при текущих порогах 99.7% super — бесполезно)
- Kafka topic для super entry signals? (для Order Manager)
- Throttling/dedup: не генерить сигнал если такой же уже есть в последних N минутах
