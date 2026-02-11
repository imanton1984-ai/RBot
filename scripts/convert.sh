#!/bin/bash

# Script to convert ONNX models to Hummingbird-optimized format
# This enables GPU acceleration with tensor cores

set -e  # Exit on any error

echo "Starting ONNX to Hummingbird conversion..."

# Check if virtual environment exists in trainer folder, create if not
if [ ! -d "/home/anton/Desktop/Rust_trader/trainer/.venv" ]; then
    echo "Creating virtual environment in trainer folder..."
    python3 -m venv /home/anton/Desktop/Rust_trader/trainer/.venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source /home/anton/Desktop/Rust_trader/trainer/.venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages including the ones needed for the conversion script
echo "Installing required packages..."
pip install onnx hummingbird-ml xgboost lightgbm

# Change to the trainer/src directory where the Python script is located
cd /home/anton/Desktop/Rust_trader/trainer/src

# Run the conversion script
python3 convert_to_hb.py --input-dir ../../models --output-dir ../../models

echo "Conversion completed!"
echo "New Hummingbird-optimized models have been saved with '_gpu' suffix in the models directory."

echo "Virtual environment created at /home/anton/Desktop/Rust_trader/trainer/.venv"
echo "To activate it in the future, run: source /home/anton/Desktop/Rust_trader/trainer/.venv/bin/activate"