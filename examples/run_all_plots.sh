#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${1:-examples/data}"
OUT_DIR="${2:-examples/artifacts/compiled}"
mkdir -p "$OUT_DIR"

# Rebuild compiled/aggregated plots from stored NPZ/JSON and copy recorded figures.
PYTHONPATH=. python3 examples/plot_stored_data.py --data-dir "$DATA_DIR" --output-dir "$OUT_DIR"

# Rebuild combined sweep plots if matching source runs exist.
PYTHONPATH=. python3 examples/plot_dimensionality_snr.py --data-dir "$DATA_DIR" --tag-prefix "mc_snr_continuous__snr_cont_fixN_varydim_" --output "$OUT_DIR/snr_cont_fixN_varydim_combined.png" || true
PYTHONPATH=. python3 examples/plot_num_arms_snr.py --data-dir "$DATA_DIR" --tag-prefix "mc_snr_multiarm__snr_multiarm_fixN_varyarms_" --output "$OUT_DIR/snr_multiarm_fixN_varyarms_combined.png" || true

# Rebuild LaTeX report from compiled artifacts + metadata.
PYTHONPATH=. python3 examples/write_experiment_report.py --art-dir "$OUT_DIR" --data-dir "$DATA_DIR" --output "$OUT_DIR/report.tex" || true
