#!/usr/bin/env bash
# setup.sh — Create the infant-cry-classifier project skeleton
# Usage: bash setup.sh

set -euo pipefail

PROJECT="infant-cry-classifier"

echo "Creating project structure: ${PROJECT}/"

# Top-level directories
mkdir -p "${PROJECT}/data/raw/hunger"
mkdir -p "${PROJECT}/data/raw/belly_pain"
mkdir -p "${PROJECT}/data/raw/burping"
mkdir -p "${PROJECT}/data/raw/discomfort"
mkdir -p "${PROJECT}/data/raw/tiredness"
mkdir -p "${PROJECT}/data/processed"
mkdir -p "${PROJECT}/features"
mkdir -p "${PROJECT}/notebooks"
mkdir -p "${PROJECT}/src"
mkdir -p "${PROJECT}/models"
mkdir -p "${PROJECT}/results/plots"
mkdir -p "${PROJECT}/results/metrics"
mkdir -p "${PROJECT}/report"

# Create __init__.py so src/ is importable
touch "${PROJECT}/src/__init__.py"

echo "Done. Directory tree:"
find "${PROJECT}" -type d | head -30
