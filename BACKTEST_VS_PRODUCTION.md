# 🔍 Таблица соответствия: Лучший бектест vs Продакшен

**Бектест (лучший):** `HEURISTIC_DIR_MODE=off SUPER_ENTRY_TIMEFRAMES="5,15,60,240,1440" ./target/release/super_entry_backtest`

Продакшен пайплайн состоит из **3 слоёв**, каждый может фильтровать/менять сигналы:
1. **Pipeline** (`pipeline.rs` + `scorer.rs`) — ML inference + overheated/danger zone фильтры
2. **Super Entry Stage** (`super_entry_stage.rs`) — heuristic direction filter
3. **Order Manager** (`order_manager.toml` + `signal_scanner.rs`) — финальная фильтрация перед ордером

---

## 🟥 СЛОЙ 1: ML Pipeline (super_entry_stage.rs / pipeline.rs)

| Параметр | Бектест (best) | Продакшен (default) | Совпадает? | Импакт |
|----------|---------------|--------------------|-----------:|--------|
| `HEURISTIC_DIR_MODE` | **`off`** | **`filter`** (default в коде) | ❌ **НЕТ** | 🔴 **КРИТИЧЕСКИЙ**: heuristic фильтр отсеивает сделки где тренд-индикаторы (supertrend, trend) не согласны с ML direction. Бектест показал что это УХУДШАЕТ результат: WR 13-16% на override, filter тоже режет прибыльные сделки |
| `HEURISTIC_DIR_CONFIDENCE` | N/A (off) | **`0.3`** (default) | ❌ | Мин. уверенность эвристики для срабатывания фильтра |
| `SUPER_ENTRY_TIMEFRAMES` (env) | **`5,15,60,240,1440`** | **`15,60,240,1440`** (startCPU.sh) | ❌ **НЕТ** | 🟡 5m отключен в продакшене (TF с худшим WR=32%, но всё равно положительный PnL) |
| `SUPER_ENTRY_TIMEFRAMES` (code default) | `5,15,60,240,1440` | `5,15,60,240,1440` | ✅ | Code default совпадает, но startCPU.sh перезаписывает |
| Direction model | **v3** (direction_v3_tf{X}.ubj загружается, если есть) | **v3** (та же логика загрузки) | ✅ | Оба используют v3 → legacy fallback |
| Модели на диске | v3 есть для 5,15,60,240,1440 + legacy для всех | Те же файлы | ✅ | |
| `p_threshold` (SuperEntryConfig) | **0.55** | **0.55** | ✅ | Мин. P(super) на уровне pipeline |
| `sl_fraction` | **0.65** | **0.65** | ✅ | SL = 65% от TP target |
| `max_hold_bars` | **25** | **25** | ✅ | |
| `lookahead_bars` | **25** | **25** | ✅ | |
| `enable_danger_zone_filter` | **true** (default) | **true** (default) | ✅ | Скипает agrees_count 3-4 |
| Overheated filter | **true** (scorer default) | **true** (scorer default) | ✅ | |
| `min_dir_confidence` | **0.05** (scorer default) | **0.05** | ✅ | Мин. confidence direction model |

---

## 🟥 СЛОЙ 2: Order Manager (config/order_manager.toml) — финал перед ордером

| Параметр | Бектест (best) | Продакшен (order_manager.toml) | Совпадает? | Импакт |
|----------|---------------|-------------------------------|----------:|--------|
| `p_super_min_5m` | 0.55 (pipeline default) | **0.0** (disabled) | ❌ | 5m фактически отключен |
| `p_super_min_15m` | 0.55 | **0.80** | ❌ **НЕТ** | 🔴 **СИЛЬНЫЙ**: Продакшен требует p≥0.80 для 15m, а бектест работал с 0.55. Bucket 0.55-0.80 теряется! |
| `p_super_min_1h` | 0.55 | **0.73** | ❌ **НЕТ** | 🔴 **СИЛЬНЫЙ**: Бектест показал WR 59% уже с bucket 0.60-0.70, а продакшен режет до 0.73 |
| `p_super_min_4h` | 0.55 | **0.55** | ✅ | |
| `p_super_min_1d` | 0.55 | **0.55** | ✅ | |
| `signal_score_max_5m` | нет лимита | **1.1** | ❌ | Cap на combined_score |
| `signal_score_max_15m` | нет лимита | **1.1** | ❌ | 🟡 Обрезает high-confidence сигналы |
| `signal_score_max_1h` | нет лимита | **1.1** | ❌ | 🟡 combined = p_super × (1+dir_conf), при p_super=0.9 и conf=0.3 → score=1.17 → ОБРЕЗАН! |
| `signal_score_max_4h` | нет лимита | **1.1** | ❌ | 🟡 |
| `signal_score_max_1d` | нет лимита | **1.1** | ❌ | 🟡 |
| `signal_score_min_*` | 0.0 | **0.0** | ✅ | |

---

## 🟥 СЛОЙ 3: Аллокация капитала (config/order_manager.toml)

| Параметр | Бектест (best) | Продакшен (order_manager.toml) | Совпадает? | Импакт |
|----------|---------------|-------------------------------|----------:|--------|
| `tf_5m_pct` | активен | **0%** | ❌ | 5m выключен (WR=32%, Sharpe=0.20 — слабый, норм) |
| `tf_15m_pct` | активен | **60%** | — | На 15m WR=50.7%, AvgPnL=1.22% |
| `tf_1h_pct` | активен | **40%** | — | На 1h WR=62.3%, AvgPnL=2.18%, Sharpe=0.576 |
| `tf_4h_pct` | активен | **0%** | ❌ **НЕТ** | 🔴 **КРИТИЧЕСКИЙ**: 4h — ЛУЧШИЙ TF! WR=66.6%, Sharpe=0.622 — но 0% аллокации! |
| `tf_1d_pct` | активен | **0%** | ❌ **НЕТ** | 🔴 **КРИТИЧЕСКИЙ**: 1d — АБСОЛЮТНО ЛУЧШИЙ! WR=84.5%, Sharpe=1.281 — но 0% аллокации! |
| `tf_1m_pct` | не тестировался | **0%** | — | |
| `trading_mode` | N/A (бектест) | **`"off"`** | ⚠️ | 🔴 Торговля ПОЛНОСТЬЮ ВЫКЛЮЧЕНА в order_manager |
| `max_orders_at_a_time` | N/A | **10** | — | |
| `leverage` | N/A | **5** | — | |
| `symbol_cooldown_hours` | N/A | **7.0** | — | Кулдаун между сделками по одному символу |

---

## 📊 Сводка критических расхождений

| # | Расхождение | Где | Рекомендация (из бектеста) |
|---|------------|-----|---------------------------|
| 1 | **Heuristic filter = ON** | `.env` / env не задан → default `filter` | `export HEURISTIC_DIR_MODE=off` |
| 2 | **4h = 0% аллокации** | `config/order_manager.toml` `tf_4h_pct=0` | `tf_4h_pct = 35` |
| 3 | **1d = 0% аллокации** | `config/order_manager.toml` `tf_1d_pct=0` | `tf_1d_pct = 25` |
| 4 | **p_super_min_1h = 0.73** | `config/order_manager.toml` | Снизить до `0.60` (WR 59% в bucket 0.60-0.70) |
| 5 | **p_super_min_15m = 0.80** | `config/order_manager.toml` | Снизить до `0.60` или оставить (15m слабее) |
| 6 | **signal_score_max = 1.1** | `config/order_manager.toml` все TF | Поднять до `2.0` — не обрезать high-confidence |
| 7 | **SUPER_ENTRY_TIMEFRAMES** | `startCPU.sh` / `startGPU.sh` | Добавить `5` если нужен (или оставить без 5m) |
| 8 | **trading_mode = "off"** | `config/order_manager.toml` | Включить перед торговлей |

---

## 🎯 Рекомендуемые изменения для продакшена

### `.env` или env переменные перед стартом:
```bash
export HEURISTIC_DIR_MODE=off
export SUPER_ENTRY_TIMEFRAMES="60,240,1440"  # или "15,60,240,1440"
```

### `config/order_manager.toml`:
```toml
# P(super) пороги — из бектеста
p_super_min_15m = 0.60    # было 0.80, bucket 0.60+ уже 50% WR
p_super_min_1h = 0.60     # было 0.73, bucket 0.60-0.70 = 59% WR
p_super_min_4h = 0.60     # было 0.55
p_super_min_1d = 0.55     # было 0.55 (уже ок)

# Убрать обрезку combined_score
signal_score_max_15m = 2.0  # было 1.1
signal_score_max_1h = 2.0   # было 1.1
signal_score_max_4h = 2.0   # было 1.1
signal_score_max_1d = 2.0   # было 1.1

# Аллокация — лучшие TF получают больше
tf_15m_pct = 0              # опционально, 15m слабоват
tf_1h_pct = 40              # было 40 ✅
tf_4h_pct = 35              # было 0 ❌→35
tf_1d_pct = 25              # было 0 ❌→25

# Включить торговлю
trading_mode = "live"       # было "off"
```

### Итого: 3 главных блокера в продакшене
1. **Heuristic filter = ON** → режет прибыльные сделки (WR 13-16% при override)
2. **4h и 1d — 0% аллокации** → лучшие TF (WR 66-84%, Sharpe 0.62-1.28) не торгуются!
3. **signal_score_max = 1.1** → обрезает сильнейшие сигналы (p_super × confidence > 1.1)
