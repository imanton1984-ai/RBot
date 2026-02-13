#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/utils.sh"

section "CPU SYSTEM CHECK"

log "Running in CPU-only mode. GPU checks skipped."

if ! command -v rustc >/dev/null 2>&1; then
    die "rustc not found. Install Rust via rustup: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi

if ! command -v cargo >/dev/null 2>&1; then
    die "cargo not found. Install Rust via rustup: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi

ok "Rust toolchain found: $(rustc -V)"

# Check for required build tools
if ! command -v cmake &> /dev/null; then
    warn "cmake not found. This may be needed for XGBoost build."
fi

if ! command -v ninja &> /dev/null; then
    warn "ninja not found. This may be needed for XGBoost build."
fi

if ! command -v gcc &> /dev/null; then
    die "gcc not found. Please install build-essential: sudo apt-get install build-essential"
fi

if ! command -v g++ &> /dev/null; then
    die "g++ not found. Please install build-essential: sudo apt-get install build-essential"
fi

ok "CPU system check completed successfully."