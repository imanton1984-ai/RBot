#!/bin/bash

# Script to run the Python training script in the virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRAINER_DIR="$PROJECT_ROOT/trainer"

echo "Running Python training script in virtual environment..."
echo "Project root: $PROJECT_ROOT"
echo "Trainer directory: $TRAINER_DIR"

# Activate the virtual environment and run the training script
cd "$TRAINER_DIR" && source .venv/bin/activate && python src/train_from_db.py

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Training script executed successfully!"
else
    echo "Training script failed!"
    exit 1
fi