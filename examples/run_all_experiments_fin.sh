#!/usr/bin/env bash
set -euo pipefail

# Final orchestration wrapper:
# 1) Run full experiment generation
# 2) Rebuild compiled plots from stored data
# 3) Generate report for compiled folder with explicit data-dir mapping
TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
OVERWRITE_ARG="${2:-}"

if [[ "$OVERWRITE_ARG" == "--overwrite" ]]; then
  bash examples/run_all_experiments.sh "$TAG" --overwrite
else
  bash examples/run_all_experiments.sh "$TAG"
fi

DATA_DIR="examples/data/${TAG}"
COMPILED_DIR="examples/artifacts/${TAG}/compiled"

bash examples/run_all_plots.sh "$DATA_DIR" "$COMPILED_DIR"

# Ensure compiled report also gets explicit metadata directory.
PYTHONPATH=. python3 examples/write_experiment_report.py --art-dir "$COMPILED_DIR" --data-dir "$DATA_DIR" --output "$COMPILED_DIR/report.tex" || true

echo "Done(fin). data_dir=$DATA_DIR compiled_dir=$COMPILED_DIR"
