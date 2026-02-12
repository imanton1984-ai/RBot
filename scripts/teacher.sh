#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv_trainer"

echo "[teacher] root=${ROOT_DIR}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "[teacher] creating virtual environment..."
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# ТОЛЬКО обучение. Конвертеры не нужны.
echo "[teacher] installing dependencies..."
python -m pip install \
  numpy pandas sqlalchemy psycopg2-binary scikit-learn \
  xgboost

echo "[teacher] starting training from DB -> models/*.ubj + schema json"

# Set the correct database URL
export DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5433/timescaledb_binance}"

# Ensure models directory exists
mkdir -p "${ROOT_DIR}/models"

# Run the training
python "${ROOT_DIR}/trainer/src/train_from_db.py"

echo "[teacher] training completed. Models saved to models/"