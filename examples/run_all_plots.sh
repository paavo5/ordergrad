#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${1:-examples/data}"
OUT_DIR="${2:-examples/artifacts/compiled}"
PYTHONPATH=. python3 examples/plot_stored_data.py --data-dir "$DATA_DIR" --output-dir "$OUT_DIR"
