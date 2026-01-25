#!/usr/bin/env bash
set -euo pipefail

WAIT=0
if [[ "${1:-}" == "--wait" ]]; then WAIT=1; fi

check_url () {
  local name="$1" url="$2"
  curl -fsS "$url" >/dev/null && echo "✅ $name" || (echo "❌ $name" && return 1)
}

check_pg () {
  docker compose -f infra/docker-compose.yml exec -T timescaledb pg_isready -U postgres >/dev/null
}

check_redpanda () {
  docker compose -f infra/docker-compose.yml exec -T redpanda rpk cluster health >/dev/null
}

check_gpu () {
  command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null
}

try_once () {
  echo "== system checks =="
  check_gpu && echo "✅ GPU" || (echo "❌ GPU"; return 1)
  check_pg && echo "✅ Postgres" || (echo "❌ Postgres"; return 1)
  check_redpanda && echo "✅ Redpanda" || (echo "❌ Redpanda"; return 1)

  echo "== service readiness =="
  check_url "api_gateway"     "http://localhost:8080/readyz"
  check_url "market_ingest"   "http://localhost:8081/readyz"
  check_url "compute_core"    "http://localhost:8082/readyz"
  check_url "data_writer"       "http://localhost:8083/readyz"
  check_url "order_engine"    "http://localhost:8084/readyz"
  check_url "position_tracker""http://localhost:8085/readyz"
}

if [[ "$WAIT" -eq 1 ]]; then
  for i in {1..60}; do
    if try_once; then exit 0; fi
    sleep 1
  done
  echo "❌ health gate timeout"
  exit 1
else
  try_once
fi