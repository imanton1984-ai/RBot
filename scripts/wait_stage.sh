#!/usr/bin/env bash
set -euo pipefail

URL="${1:?url required}"
DESIRED="${2:?desired stage required}"
TIMEOUT="${3:-120}"

# порядок стадий (твоя строка из бинарника уже содержит эти имена)
STAGES=(STARTING LOADING_PAIRS PAIRS_READY LOADING_CANDLES BACKFILL_CANDLES_READY RUN)

idx_of() {
  local s="$1"
  local i=0
  for st in "${STAGES[@]}"; do
    if [[ "$st" == "$s" ]]; then echo "$i"; return 0; fi
    ((i++))
  done
  echo "-1"
}

desired_i="$(idx_of "$DESIRED")"
if [[ "$desired_i" == "-1" ]]; then
  echo "❌ wait_stage: unknown desired stage=$DESIRED"
  exit 2
fi

deadline=$(( $(date +%s) + TIMEOUT ))
last=""

while [[ $(date +%s) -lt $deadline ]]; do
  # ожидаем JSON вида {"stage":"..."}
  resp="$(curl -fsS "$URL" 2>/dev/null || true)"
  stage="$(printf '%s' "$resp" | sed -n 's/.*"stage"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
  if [[ -n "$stage" ]]; then
    last="$stage"
    cur_i="$(idx_of "$stage")"
    # если стадия неизвестна — не валимся, просто продолжаем
    if [[ "$cur_i" != "-1" && "$cur_i" -ge "$desired_i" ]]; then
      echo "✅ wait_stage ok. desired=$DESIRED current=$stage url=$URL"
      exit 0
    fi
  fi
  sleep 1
done

echo "❌ wait_stage timeout. desired=$DESIRED last=$last url=$URL"
exit 1

