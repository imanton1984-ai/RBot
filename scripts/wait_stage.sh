#!/usr/bin/env bash
set -euo pipefail

URL="${1:?stagez url}"
TARGET="${2:?target stage}"
TIMEOUT="${3:-600}"

start_ts="$(date +%s)"
while true; do
  stage="$(curl -fsS "$URL" | sed -n 's/.*"stage"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1 || true)"
  if [[ "$stage" == "$TARGET" ]]; then
    echo "✅ stage reached: $TARGET"
    exit 0
  fi
  now="$(date +%s)"
  if (( now - start_ts > TIMEOUT )); then
    echo "❌ stage timeout. expected=$TARGET got=${stage:-unknown}"
    exit 1
  fi
  sleep 2
done