#!/bin/bash

# Script to run connection tests with detailed logs and performance metrics

echo "==========================================="
echo "Running Connection Tests with Performance Metrics"
echo "==========================================="

# Load environment variables
if [ -f "connections/.env" ]; then
    echo "Loading environment variables from connections/.env"
    export $(grep -v '^#' connections/.env | xargs)
elif [ -f ".env" ]; then
    echo "Loading environment variables from .env"
    export $(grep -v '^#' .env | xargs)
fi

# Set up logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TEST_LOG="$LOG_DIR/connection_test_$(date +%Y%m%d_%H%M%S).log"

echo "Log file: $TEST_LOG"
echo "-------------------------------------------" | tee -a "$TEST_LOG"

# Record start time
START_TIME=$(date +%s.%N)

# Change to connections directory and run tests
cd connections
echo "Running connection tests..." | tee -a "../$TEST_LOG"
echo "Test started at: $(date)" | tee -a "../$TEST_LOG"

# Run tests and capture exit code
if cargo test -- --nocapture 2>&1 | tee -a "../$TEST_LOG"; then
    TEST_RESULT=0
    echo "✅ Tests completed successfully" | tee -a "../$TEST_LOG"
else
    TEST_RESULT=$?
    echo "❌ Tests failed with exit code: $TEST_RESULT" | tee -a "../$TEST_LOG"
fi

# Go back to original directory
cd ..

# Record end time and calculate duration
END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc)

echo "-------------------------------------------" | tee -a "$TEST_LOG"
echo "Test completed at: $(date)" | tee -a "$TEST_LOG"
echo "Total execution time: ${DURATION}s" | tee -a "$TEST_LOG"

# Analyze test results
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ All tests PASSED" | tee -a "$TEST_LOG"
else
    echo "❌ Some tests FAILED (Exit code: $TEST_RESULT)" | tee -a "$TEST_LOG"
fi

# Extract performance metrics if available in logs
echo "-------------------------------------------" | tee -a "$TEST_LOG"
echo "Performance Summary:" | tee -a "$TEST_LOG"

# Look for ping/response time metrics in the log
PING_COUNT=$(grep -c "ms" "$TEST_LOG" || echo 0)
if [ $PING_COUNT -gt 0 ]; then
    echo "Found $PING_COUNT ping/response time measurements" | tee -a "$TEST_LOG"
    grep "ms" "$TEST_LOG" | tail -10 | tee -a "$TEST_LOG"
fi

echo "===========================================" | tee -a "$TEST_LOG"
echo "Connection test completed. Log saved to: $TEST_LOG"
echo "==========================================="

exit $TEST_RESULT