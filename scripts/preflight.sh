#!/usr/bin/env bash
set -euo pipefail

echo "Checking system requirements..."

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

# Check if docker compose is installed
if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi

# Check if nvidia-smi is available (for GPU support)
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  NVIDIA drivers not detected - GPU acceleration will be disabled"
else
    echo "✅ NVIDIA drivers detected"
fi

# Check if rust is installed
if ! command -v rustc &> /dev/null; then
    echo "❌ Rust is not installed"
    exit 1
fi

# Check if cargo is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Cargo is not installed"
    exit 1
fi

# Check if nightly toolchain is installed
if ! rustc --version | grep -q "nightly"; then
    echo "⚠️  Nightly Rust toolchain not detected - installing..."
    rustup toolchain install nightly
    rustup default nightly
fi

# Check if required components are installed
if ! rustc --version | grep -q "rust-src"; then
    echo "Installing rust-src component..."
    rustup component add rust-src
fi

if ! rustc --version | grep -q "clippy"; then
    echo "Installing clippy component..."
    rustup component add clippy
fi

if ! rustc --version | grep -q "rustfmt"; then
    echo "Installing rustfmt component..."
    rustup component add rustfmt
fi

echo "✅ All prerequisites met"