#!/usr/bin/env bash
#
# fullrun.sh — Запуск всего: GPU бот + стратегия + WebUI
#
# Использование:
#   ./fullrun.sh
#
# Автоматически определяет стратегию из config/order_manager.toml:
#   strategy_type = "ml_super_entry_nodir"  → Super Entry
#   strategy_type = "ml_pump_dump"          → Pump/Dump
#
# Запускает:
#   1. startGPU.sh с нужным флагом стратегии
#   2. scripts/runweb.sh (WebUI сервер)
#

set -uo pipefail
# NOTE: -e removed intentionally — startGPU.sh may have non-fatal errors
# (like CMake warnings) that should not abort the WebUI launch.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Auto-detect strategy from config/order_manager.toml
# Use sed for robustness (no -P flag dependency, handles whitespace)
STRATEGY_TYPE=$(sed -n 's/^strategy_type[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' config/order_manager.toml 2>/dev/null | head -1 | tr -d '[:space:]')
[ -z "$STRATEGY_TYPE" ] && STRATEGY_TYPE="ml_super_entry_nodir"

echo -e "\033[1;36m═══════════════════════════════════════════\033[0m"
echo -e "\033[1;36m  🚀 FULL RUN: GPU + Strategy + WebUI\033[0m"
echo -e "\033[1;36m  Strategy from config: [${STRATEGY_TYPE}]\033[0m"
echo -e "\033[1;36m═══════════════════════════════════════════\033[0m"

# 1. Start GPU bot with detected strategy
case "$STRATEGY_TYPE" in
    *pump_dump*)
        echo -e "\n\033[1;33m[1/2] Starting GPU bot with --pump-dump...\033[0m"
        ./startGPU.sh --pump-dump || true
        ;;
    *)
        echo -e "\n\033[1;33m[1/2] Starting GPU bot with --super-entry...\033[0m"
        ./startGPU.sh --super-entry || true
        ;;
esac

# 2. Start WebUI server
echo -e "\n\033[1;33m[2/2] Starting WebUI server...\033[0m"
./scripts/runweb.sh || true

echo -e "\n\033[1;32m═══════════════════════════════════════════\033[0m"
echo -e "\033[1;32m  ✅ FULL RUN COMPLETE (strategy: ${STRATEGY_TYPE})\033[0m"
echo -e "\033[1;32m═══════════════════════════════════════════\033[0m"
