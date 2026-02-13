#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/utils.sh"

section "GPU SYSTEM CHECK"

# 1. Check NVCC
SELECTED_CUDA="/usr/local/cuda"
if [ ! -x "$SELECTED_CUDA/bin/nvcc" ]; then
    if [ -x "/usr/local/cuda-13.1/bin/nvcc" ]; then
        SELECTED_CUDA="/usr/local/cuda-13.1"
    elif [ -x "/usr/local/cuda-12.9/bin/nvcc" ]; then
        SELECTED_CUDA="/usr/local/cuda-12.9"
    else
        # Look for any CUDA installation
        for cuda_dir in /usr/local/cuda-*; do
            if [ -d "$cuda_dir" ] && [ -x "$cuda_dir/bin/nvcc" ]; then
                SELECTED_CUDA="$cuda_dir"
                break
            fi
        done
        
        if [ ! -x "$SELECTED_CUDA/bin/nvcc" ]; then
            die "CUDA binaries not found. Please check /usr/local/cuda symlink or install CUDA."
        fi
    fi
fi

# Export for current session (callers should create wrappers or source this)
export CUDA_HOME="$SELECTED_CUDA"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_ROOT="$CUDA_HOME"

ok "Using CUDA from: $CUDA_HOME"
log "NVCC Version: $(nvcc --version | grep release)"

# 2. Check nvidia-smi to verify driver is working
if ! command -v nvidia-smi &> /dev/null; then
    warn "nvidia-smi not found. GPU may not be accessible."
else
    log "nvidia-smi available: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
fi

# 3. Check cuDNN (auto-install logic from original script)
if command -v apt-get &> /dev/null; then
    if ! ldconfig -p | grep -q libcudnn.so.9; then
        warn "libcudnn.so.9 not found. Attempting to install NVIDIA cuDNN 9..."
        if [ "$EUID" -ne 0 ]; then
            echo "Please enter your password for sudo to continue installation."
        fi
        
        # Simple installation attempt
        if ! command -v wget &> /dev/null; then
             sudo apt-get update && sudo apt-get install -y wget
        fi
        
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
        sudo dpkg -i /tmp/cuda-keyring.deb
        sudo apt-get update
        sudo apt-get install -y libcudnn9-cuda-12
        rm /tmp/cuda-keyring.deb
        ok "cuDNN 9 installation complete."
    else
        ok "Found libcudnn.so.9"
    fi
else
    warn "apt-get not found, skipping auto cuDNN check."
fi

# 4. Version Check for XGBoost compatibility
CUDA_VERSION=$(nvcc --version | grep "V[0-9]" | cut -d' ' -f6 | sed 's/V//')
REQUIRED_CUDA_VERSION="12.9"
log "Detected CUDA: $CUDA_VERSION (Req: >= $REQUIRED_CUDA_VERSION for latest XGBoost)"

# Simple version compare logic
if [ "$(printf '%s\n%s' "$REQUIRED_CUDA_VERSION" "$CUDA_VERSION" | sort -V | head -n1)" != "$REQUIRED_CUDA_VERSION" ]; then
    warn "CUDA version $CUDA_VERSION is old. You might need to use compatibility build script."
fi

# 5. Check for required build tools
if ! command -v cmake &> /dev/null; then
    die "cmake not found. Please install cmake."
fi

if ! command -v ninja &> /dev/null; then
    die "ninja not found. Please install ninja-build."
fi

ok "GPU system check completed successfully."