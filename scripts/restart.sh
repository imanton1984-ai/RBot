#!/bin/bash

# Script to restart all services (stop + start)

echo "==========================================="
echo "Restarting All Services"
echo "==========================================="

# Set up logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
RESTART_LOG="$LOG_DIR/restart_$(date +%Y%m%d_%H%M%S).log"

echo "Log file: $RESTART_LOG"
echo "Restart time: $(date)" | tee -a "$RESTART_LOG"

# Run the stop script first
echo "Stopping services before restart..." | tee -a "$RESTART_LOG"
./scripts/stop.sh 2>&1 | tee -a "$RESTART_LOG"

# Wait a moment for services to fully stop
echo "Waiting 5 seconds for services to stop completely..." | tee -a "$RESTART_LOG"
sleep 5

# Run the start script
echo "Starting services..." | tee -a "$RESTART_LOG"
./scripts/start.sh 2>&1 | tee -a "$RESTART_LOG"

# Check the result of the start script
START_RESULT=${PIPESTATUS[0]}

echo "===========================================" | tee -a "$RESTART_LOG"
echo "Restart completed at: $(date)" | tee -a "$RESTART_LOG"
if [ $START_RESULT -eq 0 ]; then
    echo "✅ Restart completed successfully" | tee -a "$RESTART_LOG"
else
    echo "❌ Restart completed with errors (Exit code: $START_RESULT)" | tee -a "$RESTART_LOG"
fi
echo "==========================================="

echo "Restart process completed. Check logs in $LOG_DIR for details."
exit $START_RESULT