# Стратегии трейдинга - Rust Trader

## Обзор

Этот модуль содержит 6 независимых стратегий для генерации торговых сигналов. Каждая стратегия имеет свой собственный `scorer.rs` (расчет качества сигнала) и `trade_signal_calculator.rs` (расчет TP/SL и параметров сделки).

## Стратегии

### 1. Indicator ML Strategy (`indicator_ml_strategy`)
**Strategy ID:** 1

**Описание:** Стратегия полагается на ML предсказания + индикаторы с равными весами.

**Источники данных:**
- ML предсказания (PriceTarget от ML)
- Индикаторы: RSI, MACD, Stochastic, ADX, ATR, Volume, Trend (EMA)

**Веса индикаторов:**
- RSI: 0.20
- MACD: 0.20
- Stochastic: 0.15
- ADX: 0.15
- ATR (volatility): 0.10
- Volume: 0.10
- Trend (EMA): 0.10

**TP/SL:**
- TP берется из ML price10_target
- SL рассчитывается через ATR (0.75 ATR)
- Минимальный R:R = 1.5

---

### 2. Indicator Heuristic Strategy (`indicator_heuristic_strategy`)
**Strategy ID:** 2

**Описание:** Стратегия полагается на эвристические предсказания + индикаторы.

**Источники данных:**
- Эвристические предсказания (PriceTarget от Hard/Heuristic)
- Индикаторы: те же веса что и в Indicator ML

**TP/SL:**
- TP берется из heuristic price10_target
- SL рассчитывается через ATR
- Минимальный R:R = 1.5

---

### 3. Indicator Consensus Strategy (`indicator_consensus_strategy`)
**Strategy ID:** 3

**Описание:** Стратегия использует консенсус между ML и heuristic предсказаниями + индикаторы.

**Источники данных:**
- Консенсус предсказаний (average ML + heuristic)
- Индикаторы

**Особенности:**
- Бонус за согласие ML и heuristic (agreement boost)
- Если ML и heuristic не согласны - penalty

**TP/SL:**
- TP = average(ML target, heuristic target)
- SL через ATR

---

### 4. Level ML Strategy (`level_ml_strategy`)
**Strategy ID:** 4

**Описание:** Стратегия использует ТОЛЬКО ML предсказания с уровнями поддержки/сопротивления.

**Источники данных:**
- ML предсказания (PriceTarget, LevelBounce, LevelBreakout)
- Уровни из raw_signals_summary + predictors

**TP/SL:**
- Использует level-aware targets
- Для Bounce setup: SL = 0.55 ATR, TP1 = 0.75 ATR
- Для Breakout setup: SL = 0.75 ATR, TP1 = 1.1 ATR

---

### 5. Level Heuristic Strategy (`level_heuristic_strategy`)
**Strategy ID:** 5

**Описание:** Стратегия использует ТОЛЬКО эвристические предсказания с уровнями.

**Источники данных:**
- Эвристические предсказания
- Уровни поддержки/сопротивления

**TP/SL:**
- Аналогично Level ML но с heuristic targets

---

### 6. Level Consensus Strategy (`level_consensus_strategy`)
**Strategy ID:** 6

**Описание:** Эталонная стратегия - копия текущей логики final_scorer.

**Источники данных:**
- Консенсус предсказаний (ML + heuristic fused)
- Уровни поддержки/сопротивления
- Raw signals
- Индикаторы
- Market params (BTC regime)

**Особенности:**
- SR alignment boost (1.08x)
- Synergy boost (1.12x)
- Trend conflict penalty (0.75x)
- Correction penalty для overextension

**TP/SL:**
- Полная логика из оригинального final_scorer

---

## База данных

### Таблица `trade.final_signals`

Новые колонки:
```sql
strategy_id     SMALLINT NOT NULL DEFAULT 6,  -- 1-6 для стратегий
strategy_name   TEXT NOT NULL DEFAULT 'level_consensus'
```

### Таблица `trade.backtest_results`

Новые колонки:
```sql
strategy_id     SMALLINT NOT NULL DEFAULT 6,
strategy_name   TEXT NOT NULL DEFAULT 'level_consensus'
```

### Таблица `trade.strategy_registry`

Справочник стратегий:
```sql
strategy_id     SMALLINT PRIMARY KEY,
strategy_name   TEXT NOT NULL UNIQUE,
description     TEXT,
is_active       BOOLEAN NOT NULL DEFAULT true
```

---

## Backtester

Backtester теперь группирует результаты по стратегиям.

### Запуск

```bash
cd backtester
cargo run
```

### Вывод

Backtester выводит:
1. **Общий summary** по всем сигналам
2. **Summary по стратегиям:**
   - Win rate по каждой стратегии
   - Average PnL
   - Sharpe ratio
   - TP level distribution
   - Score breakdown по bucket'ам

### CSV экспорт

Экспортируемые файлы:
- `backtest_results.csv` - глобальный файл со всеми сигналами
- `backtest_results_1m.csv`, `backtest_results_5m.csv`, etc. - по таймфреймам
- `backtest_results_15m.csv`, etc.

Новые колонки в CSV:
- `strategy_id`
- `strategy_name`

---

## Архитектура

```
strategies/
├── indicator_ml_strategy/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── scorer.rs
│       └── trade_signal_calculator.rs
├── indicator_heuristic_strategy/
├── indicator_consensus_strategy/
├── level_ml_strategy/
├── level_heuristic_strategy/
└── level_consensus_strategy/
```

### Scorers

Каждый scorer реализует:
- `score_signal()` - возвращает только final score
- `score_signal_verbose()` - возвращает breakdown с деталями

### Trade Signal Calculators

Каждый calculator реализует:
- `build_trade_signal()` - создает TradeSignal с TP/SL/leverage

---

## Индикаторы

Все стратегии используют следующие индикаторы из `market.indicators_wide`:

**Momentum:**
- RSI (Relative Strength Index)
- CCI (Commodity Channel Index)
- Stochastic (%K, %D)
- Williams %R

**Trend:**
- MACD (line, signal, histogram)
- ADX (Average Directional Index)
- EMA (20, 50, 200)
- SMA

**Volatility:**
- Bollinger Bands (upper, middle, lower)
- ATR (Average True Range)

**Volume:**
- OBV (On Balance Volume)
- VWAP (Volume Weighted Average Price)
- Volume Spike

**Level:**
- Support/Resistance levels (JSON)
- Alligator (jaw, teeth, lips)

---

## Использование

### Добавление новой стратегии

1. Создать папку в `strategies/`
2. Добавить `Cargo.toml`
3. Реализовать `scorer.rs` и `trade_signal_calculator.rs`
4. Добавить в `Cargo.toml` workspace
5. Добавить запись в `trade.strategy_registry`

### Интеграция с pipeline

```rust
use indicator_ml_strategy::{IndicatorMlScorer, TradeSignalCalculator, STRATEGY_ID, STRATEGY_NAME};

let scorer = IndicatorMlScorer::new(0.55);
let calculator = TradeSignalCalculator::new(0.55);

let signal = calculator.build_trade_signal(
    &pool, &scorer,
    time, time_ms, symbol_id, &symbol, tf_minutes,
    entry_price, &raw_signals_summary, &predictors,
).await?;
```

---

## Tuning параметров

Параметры TP/SL настраиваются в `config/signal_params.toml`:

```toml
[timeframe_15m]
tp1_pct = 1.5
tp2_pct = 2.8
tp3_pct = 4.0
sl_min_pct = 1.0
sl_pct = 3.0
max_pred_deviation_pct = 15.0

[atr_bounce]
sl_mult = 0.55
tp1_mult = 0.75
tp2_mult = 1.4
tp3_mult = 2.2

[atr_breakout]
sl_mult = 0.75
tp1_mult = 1.1
tp2_mult = 1.8
tp3_mult = 2.8
```

---

## Анализ результатов

После запуска backtester:

1. **Сравнить win rate по стратегиям:**
   - Indicator ML vs Indicator Heuristic - что лучше?
   - Level ML vs Level Heuristic - что лучше?
   - Consensus vs отдельные - есть ли benefit?

2. **Сравнить PnL:**
   - Какая стратегия самая прибыльная?
   - Какая стратегия самая стабильная (Sharpe)?

3. **Анализ по таймфреймам:**
   - На каких TF стратегии работают лучше?

4. **Score buckets:**
   - Какие score threshold дают лучший win rate?

---

## Миграции

Применить миграцию:
```bash
psql -d timescaledb_binance -f database/ddl/090_strategy_support.sql
```
