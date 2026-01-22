#!/usr/bin/env bash
set -euo pipefail

: "${DATABASE_URL:?DATABASE_URL is required}"

PSQL="psql -X -v ON_ERROR_STOP=1 -tA"

fail() {
  echo "❌ DB HEALTH FAILED: $*" >&2
  exit 1
}

ok() {
  echo "✅ $*"
}

# --- helpers ---
q() {
  # run query, return raw output
  $PSQL "$DATABASE_URL" -c "$1" | tr -d '\r'
}

require_true() {
  local name="$1"
  local sql="$2"
  local out
  out="$(q "$sql")"
  [[ "$out" == "t" || "$out" == "true" || "$out" == "1" ]] || fail "$name expected TRUE, got: $out"
  ok "$name"
}

require_notnull() {
  local name="$1"
  local sql="$2"
  local out
  out="$(q "$sql")"
  [[ -n "$out" && "$out" != "f" && "$out" != "false" ]] || fail "$name expected NOT NULL/TRUE, got: $out"
  ok "$name"
}

require_min_int() {
  local name="$1"
  local sql="$2"
  local min="$3"
  local out
  out="$(q "$sql")"
  [[ "$out" =~ ^[0-9]+$ ]] || fail "$name expected int, got: $out"
  (( out >= min )) || fail "$name expected >= $min, got: $out"
  ok "$name ($out >= $min)"
}

# --- H00 ping ---
require_min_int "H00 ping" "SELECT 1;" 1

# --- H01 Timescale ---
require_true "H01 timescale installed" "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname='timescaledb');"

# --- H02 Schemas ---
require_true "H02 schema market" "SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname='market');"
require_true "H02 schema trade"  "SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname='trade');"

# --- H03 Core tables exist ---
require_true "H03 market.pairs"             "SELECT to_regclass('market.pairs') IS NOT NULL;"
require_true "H03 market.collected_candles" "SELECT to_regclass('market.collected_candles') IS NOT NULL;"
require_true "H03 market.candles_live"      "SELECT to_regclass('market.candles_live') IS NOT NULL;"
require_true "H03 market.raw_signals"       "SELECT to_regclass('market.raw_signals') IS NOT NULL;"
require_true "H03 trade.final_signals"      "SELECT to_regclass('trade.final_signals') IS NOT NULL;"
require_true "H03 trade.orders"             "SELECT to_regclass('trade.orders') IS NOT NULL;"
require_true "H03 trade.positions"          "SELECT to_regclass('trade.positions') IS NOT NULL;"
require_true "H03 trade.position_events"    "SELECT to_regclass('trade.position_events') IS NOT NULL;"
require_true "H03 trade.ml_train_examples"  "SELECT to_regclass('trade.ml_train_examples') IS NOT NULL;"

# --- H04 Candle TF tables ---
require_true "H04 candles_1m"  "SELECT to_regclass('market.candles_1m')  IS NOT NULL;"
require_true "H04 candles_5m"  "SELECT to_regclass('market.candles_5m')  IS NOT NULL;"
require_true "H04 candles_15m" "SELECT to_regclass('market.candles_15m') IS NOT NULL;"
require_true "H04 candles_1h"  "SELECT to_regclass('market.candles_1h')  IS NOT NULL;"
require_true "H04 candles_4h"  "SELECT to_regclass('market.candles_4h')  IS NOT NULL;"
require_true "H04 candles_1d"  "SELECT to_regclass('market.candles_1d')  IS NOT NULL;"

# --- H05 Indicators TF tables ---
require_true "H05 indicators_1m"  "SELECT to_regclass('market.indicators_1m')  IS NOT NULL;"
require_true "H05 indicators_5m"  "SELECT to_regclass('market.indicators_5m')  IS NOT NULL;"
require_true "H05 indicators_15m" "SELECT to_regclass('market.indicators_15m') IS NOT NULL;"
require_true "H05 indicators_1h"  "SELECT to_regclass('market.indicators_1h')  IS NOT NULL;"
require_true "H05 indicators_4h"  "SELECT to_regclass('market.indicators_4h')  IS NOT NULL;"
require_true "H05 indicators_1d"  "SELECT to_regclass('market.indicators_1d')  IS NOT NULL;"

# --- H06 Hypertables counts ---
require_min_int "H06 hypertables candles_*"    "SELECT count(*) FROM timescaledb_information.hypertables WHERE hypertable_schema='market' AND hypertable_name LIKE 'candles_%';" 6
require_min_int "H06 hypertables indicators_*" "SELECT count(*) FROM timescaledb_information.hypertables WHERE hypertable_schema='market' AND hypertable_name LIKE 'indicators_%';" 6
require_min_int "H06 hypertable raw_signals"   "SELECT count(*) FROM timescaledb_information.hypertables WHERE hypertable_schema='market' AND hypertable_name='raw_signals';" 1
require_min_int "H06 hypertable final_signals" "SELECT count(*) FROM timescaledb_information.hypertables WHERE hypertable_schema='trade'  AND hypertable_name='final_signals';" 1
require_min_int "H06 hypertable ml_train_examples" "SELECT count(*) FROM timescaledb_information.hypertables WHERE hypertable_schema='trade' AND hypertable_name='ml_train_examples';" 1
require_min_int "H06 hypertable position_events" "SELECT count(*) FROM timescaledb_information.hypertables WHERE hypertable_schema='trade' AND hypertable_name='position_events';" 1

# --- H07 Compression enabled (candles + indicators) ---
# Если compression_enabled хоть где-то false — считаем ошибкой
comp_bad="$(q "SELECT count(*) FROM timescaledb_information.hypertables
              WHERE hypertable_schema='market'
                AND (hypertable_name LIKE 'candles_%' OR hypertable_name LIKE 'indicators_%')
                AND compression_enabled = false;")"
[[ "$comp_bad" =~ ^[0-9]+$ ]] || fail "H07 compression check returned non-int: $comp_bad"
(( comp_bad == 0 )) || fail "H07 compression enabled: found $comp_bad hypertables without compression"
ok "H07 compression enabled for candles/indicators"

# --- H08 Retention policy jobs exist (soft check) ---
# Используем связь через hypertable_name напрямую из вьюхи jobs
ret_c1d="$(q "SELECT count(*) FROM timescaledb_information.jobs WHERE proc_name='policy_retention' AND hypertable_name = 'candles_1d' AND hypertable_schema = 'market';")"
ret_i1d="$(q "SELECT count(*) FROM timescaledb_information.jobs WHERE proc_name='policy_retention' AND hypertable_name = 'indicators_1d' AND hypertable_schema = 'market';")"

[[ "$ret_c1d" =~ ^[0-9]+$ ]] || fail "H08 retention candles_1d bad: $ret_c1d"
[[ "$ret_i1d" =~ ^[0-9]+$ ]] || fail "H08 retention indicators_1d bad: $ret_i1d"
(( ret_c1d >= 1 )) || fail "H08 retention candles_1d missing"
(( ret_i1d >= 1 )) || fail "H08 retention indicators_1d missing"
ok "H08 retention policies present (at least for 1d candles/indicators)"

# --- H09 Compression policy jobs exist (soft check) ---
comp_c1d="$(q "SELECT count(*) FROM timescaledb_information.jobs WHERE proc_name='policy_compression' AND hypertable_name = 'candles_1d' AND hypertable_schema = 'market';")"
comp_i1d="$(q "SELECT count(*) FROM timescaledb_information.jobs WHERE proc_name='policy_compression' AND hypertable_name = 'indicators_1d' AND hypertable_schema = 'market';")"

[[ "$comp_c1d" =~ ^[0-9]+$ ]] || fail "H09 comp candles_1d bad: $comp_c1d"
[[ "$comp_i1d" =~ ^[0-9]+$ ]] || fail "H09 comp indicators_1d bad: $comp_i1d"
(( comp_c1d >= 1 )) || fail "H09 compression policy candles_1d missing"
(( comp_i1d >= 1 )) || fail "H09 compression policy indicators_1d missing"
ok "H09 compression policies present (at least for 1d candles/indicators)"

ok "DB HEALTH OK ✅"
