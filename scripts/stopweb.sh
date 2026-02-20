#!/bin/bash
# stopweb.sh - Stop all WebUI processes (backend + frontend dev servers)
# Usage: ./scripts/stopweb.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "  Stopping WebUI Processes"
echo "========================================"
echo ""

# Function to kill processes by port
kill_by_port() {
    local port=$1
    local name=$2
    
    PID=$(lsof -ti:$port 2>/dev/null || true)
    if [ ! -z "$PID" ]; then
        echo "[INFO] Killing $name (PID: $PID) on port $port"
        kill -15 $PID 2>/dev/null || kill -9 $PID 2>/dev/null || true
        sleep 1
        echo "[OK] $name stopped"
    else
        echo "[INFO] No process found on port $port"
    fi
}

# Function to kill processes by name
kill_by_name() {
    local pattern=$1
    local name=$2
    
    PIDS=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ ! -z "$PIDS" ]; then
        echo "[INFO] Killing $name processes (PIDs: $PIDS)"
        echo "$PIDS" | xargs kill -15 2>/dev/null || true
        sleep 1
        # Force kill if still running
        PIDS=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [ ! -z "$PIDS" ]; then
            echo "$PIDS" | xargs kill -9 2>/dev/null || true
        fi
        echo "[OK] $name stopped"
    else
        echo "[INFO] No $name processes found"
    fi
}

# Stop backend server (port 3000)
echo "[1/4] Stopping backend server..."
kill_by_port 3000 "WebUI backend"

# Stop frontend dev server (port 5173)
echo ""
echo "[2/4] Stopping frontend dev server..."
kill_by_port 5173 "Vite dev server"

# Stop any remaining webui_server processes
echo ""
echo "[3/4] Stopping any remaining webui_server processes..."
kill_by_name "webui_server" "webui_server"

# Stop any remaining vite processes
echo ""
echo "[4/4] Stopping any remaining vite/npm processes..."
kill_by_name "vite" "Vite"

echo ""
echo "========================================"
echo "  All WebUI processes stopped"
echo "========================================"

# Show remaining processes on our ports (if any)
REMAINING=$(lsof -ti:3000,5173 2>/dev/null || true)
if [ ! -z "$REMAINING" ]; then
    echo ""
    echo "[WARNING] Some processes still running:"
    echo "$REMAINING" | xargs ps -p 2>/dev/null || true
    echo ""
    echo "To force kill, run:"
    echo "  kill -9 $(lsof -ti:3000,5173 | tr '\n' ' ')"
fi
