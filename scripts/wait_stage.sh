#!/usr/bin/env bash
set -euo pipefail

URL="${1:-}"
DESIRED="${2:-}"
TIMEOUT_SEC="${3:-120}"

if [[ -z "$URL" || -z "$DESIRED" ]]; then
  echo "usage: wait_stage.sh <url> <DESIRED_STAGE> [timeout_sec]"
  exit 2
fi

deadline=$(( $(date +%s) + TIMEOUT_SEC ))
last=""

extract_stage() {
  python3 - <<'PY' 2>/dev/null || true
import sys, json
s = sys.stdin.read().strip()
if not s:
    print("")
    sys.exit(0)
if s.startswith("{"):
    try:
        j = json.loads(s)
        st = j.get("stage", j)
        if isinstance(st, dict):
            print(st.get("stage", ""))
        else:
            print(str(st))
    except Exception:
        print("")
else:
    print(s)
PY
}

while true; do
  now=$(date +%s)
  if (( now > deadline )); then
    echo "❌ wait_stage timeout. desired=$DESIRED last=$last url=$URL"
    exit 1
  fi

  # Берём и body и HTTP code
  resp="$(curl -sS -w $'\n%{http_code}' "$URL" 2>/dev/null || true)"
  body="$(printf "%s" "$resp" | sed '$d')"
  code="$(printf "%s" "$resp" | tail -n 1)"

  if [[ "$code" != "200" ]]; then
    sleep 1
    continue
  fi

  stage="$(printf "%s" "$body" | extract_stage)"
  if [[ -z "$stage" ]]; then
    sleep 1
    continue
  fi

  last="$stage"

  if [[ "$stage" == "$DESIRED" ]]; then
    echo "✅ stage reached: $stage"
    exit 0
  fi

  if [[ "$stage" == ERROR* ]]; then
    echo "❌ stage error: $stage"
    echo "$body"
    exit 1
  fi

  sleep 1
done
