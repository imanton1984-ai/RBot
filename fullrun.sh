#!/usr/bin/env bash
#
# fullrun.sh — Запуск всего: GPU бот + Super Entry + WebUI
#
# Использование:
#   ./fullrun.sh
#
# Запускает:
#   1. startGPU.sh --super-entry  (инфра, БД, сборка, сервисы)
#   2. scripts/runweb.sh           (WebUI сервер)
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo -e "\033[1;36m═══════════════════════════════════════════\033[0m"
echo -e "\033[1;36m  🚀 FULL RUN: GPU + Super Entry + WebUI\033[0m"
echo -e "\033[1;36m═══════════════════════════════════════════\033[0m"

# 1. Start GPU bot with super-entry strategy
echo -e "\n\033[1;33m[1/2] Starting GPU bot with --super-entry...\033[0m"
./startGPU.sh --super-entry

# 2. Start WebUI server
echo -e "\n\033[1;33m[2/2] Starting WebUI server...\033[0m"
./scripts/runweb.sh

echo -e "\n\033[1;32m═══════════════════════════════════════════\033[0m"
echo -e "\033[1;32m  ✅ FULL RUN COMPLETE\033[0m"
echo -e "\033[1;32m═══════════════════════════════════════════\033[0m"
