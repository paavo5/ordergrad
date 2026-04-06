#!/usr/bin/env bash
set -euo pipefail

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
OVERWRITE=0
if [[ "${2:-}" == "--overwrite" || "${1:-}" == "--overwrite" ]]; then
  OVERWRITE=1
  if [[ "${1:-}" == "--overwrite" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
  fi
fi
DATA_DIR="examples/data/${TS}"
ART_DIR="examples/artifacts/${TS}"
if [[ -e "$DATA_DIR" || -e "$ART_DIR" ]]; then
  if [[ "$OVERWRITE" -ne 1 ]]; then
    echo "Target data/artifact folder already exists for tag '$TS'. Re-run with --overwrite to reuse it." >&2
    exit 1
  fi
  rm -rf "$DATA_DIR" "$ART_DIR"
fi
mkdir -p "$DATA_DIR" "$ART_DIR"

# Weight plots: different ranks (N=100, k=10)
PYTHONPATH=. python3 examples/plot_order_weights.py --N 100 --k 10 --ranks 1,3,10 --output "$ART_DIR/weights_ranks_k10.png" || true
# Weight plots: different weighting schemes, different k
PYTHONPATH=. python3 examples/plot_order_weights.py --N 100 --k 10,20 --ranks 0 --a TopM:2,HarrellDavis:0.25 --output "$ART_DIR/weights_schemes_multi_k.png" || true

# MC gradient SNR: fix N vary k, multiarm reward modes
for RM in linear exp gaussian; do
  PYTHONPATH=. python3 examples/mc_snr_multiarm.py --N 64 --num-arms 8 --k-grid 1,2,3,4,5,6 --num-mc 1000 --reward-mode "$RM" --a ReMax --store-data --tag "snr_multiarm_fixN_varyk_${RM}" --data-dir "$DATA_DIR" --output "$ART_DIR/snr_multiarm_fixN_varyk_${RM}.png" || true
done
# Continuous fix N vary k, objective modes
for OBJ in quadratic quad_sin; do
  PYTHONPATH=. python3 examples/mc_snr_continuous.py --N 64 --dim 2 --k-grid 1,2,3,4,5,6 --num-mc 1000 --objective "$OBJ" --sin-freq 4.0 --a ReMax --store-data --tag "snr_cont_fixN_varyk_${OBJ}" --data-dir "$DATA_DIR" --output "$ART_DIR/snr_cont_fixN_varyk_${OBJ}.png" || true
done

# MC gradient SNR: fix N vary arms/dims
for ARMS in 4 8 16; do
  PYTHONPATH=. python3 examples/mc_snr_multiarm.py --N 64 --num-arms "$ARMS" --k-grid 4 --num-mc 1000 --reward-mode linear --a ReMax --store-data --tag "snr_multiarm_fixN_varyarms_${ARMS}" --data-dir "$DATA_DIR" --output "$ART_DIR/snr_multiarm_fixN_varyarms_${ARMS}.png" --no-plot || true
done
for DIM in 1 2 4 8; do
  PYTHONPATH=. python3 examples/mc_snr_continuous.py --N 64 --dim "$DIM" --k-grid 4 --num-mc 1000 --objective quad_sin --sin-freq 6.0 --a ReMax --store-data --tag "snr_cont_fixN_varydim_${DIM}" --data-dir "$DATA_DIR" --output "$ART_DIR/snr_cont_fixN_varydim_${DIM}.png" --no-plot || true
done

# Unbiasedness checks (gradient error down)
PYTHONPATH=. python3 examples/mc_gradients_multiarm.py --N 64 --k 6 --num-arms 8 --reward-mode gaussian --prob-mode random --t-grid 1,2,5,10,20,50,100,200 --store-data --tag grad_multiarm_unbias --data-dir "$DATA_DIR" --output "$ART_DIR/grad_multiarm_unbias.png" || true
PYTHONPATH=. python3 examples/mc_gradients_continuous.py --N 64 --k 6 --dim 2 --objective quad_sin --sin-freq 5.0 --t-grid 1,2,5,10,20,50,100,200 --store-data --tag grad_cont_unbias --data-dir "$DATA_DIR" --output "$ART_DIR/grad_cont_unbias.png" || true

# Benchmark runtime bar chart
PYTHONPATH=. python3 examples/benchmark_runtime_bar.py --N 300 --k 30 --repeats 100 --tag runtime_bar --data-dir "$DATA_DIR" --output "$ART_DIR/runtime_bar.png" || true

# MC accuracy debugging
PYTHONPATH=. python3 examples/monte_carlo_accuracy.py --backend np --N 64 --k 6 --num-arms 8 --arm-rank 1 --reward-mode gaussian --prob-mode random --plot-arm-details --store-data --tag mc_accuracy_rank1 --data-dir "$DATA_DIR" --output "$ART_DIR/mc_accuracy_rank1.png" || true
PYTHONPATH=. python3 examples/monte_carlo_accuracy.py --backend np --N 64 --k 6 --num-arms 8 --arm-rank 3 --reward-mode gaussian --prob-mode random --plot-arm-details --store-data --tag mc_accuracy_rank3 --data-dir "$DATA_DIR" --output "$ART_DIR/mc_accuracy_rank3.png" || true
PYTHONPATH=. python3 examples/monte_carlo_accuracy.py --backend np --N 64 --k 6 --num-arms 8 --arm-rank 6 --reward-mode gaussian --prob-mode random --plot-arm-details --store-data --tag mc_accuracy_rank_last --data-dir "$DATA_DIR" --output "$ART_DIR/mc_accuracy_rank_last.png" || true

# Reward CDF vs interpolated quantile-CDF curves
PYTHONPATH=. python3 examples/plot_reward_cdf_quantile.py --dist uniform --N 64 --k 10 --num-estimates 300 --estimator Quantile,QuantileWeibull,QuantileBlom --store-data --tag reward_cdf_uniform_k10 --data-dir "$DATA_DIR" --output "$ART_DIR/reward_cdf_uniform_k10.png" || true
PYTHONPATH=. python3 examples/plot_reward_cdf_quantile.py --dist gaussian_mixture --N 64 --k 10 --num-estimates 300 --estimator QuantileHazen,QuantileBlom --mix-centers 0.0,2.0 --mix-scales 1.0,0.7 --mix-weights 0.65,0.35 --store-data --tag reward_cdf_gmix_k10 --data-dir "$DATA_DIR" --output "$ART_DIR/reward_cdf_gmix_k10.png" || true

# Quantile vs HarrellDavis accuracy
PYTHONPATH=. python3 examples/quantile_estimator_accuracy.py --dist uniform --quantile 0.25 --N 64 --k-list 6,10 --store-data --tag quantile_uniform_q025 --data-dir "$DATA_DIR" --output "$ART_DIR/quantile_uniform_q025.png" || true
PYTHONPATH=. python3 examples/quantile_estimator_accuracy.py --dist gaussian --quantile 0.25 --N 64 --k-list 6,10 --store-data --tag quantile_gaussian_q025 --data-dir "$DATA_DIR" --output "$ART_DIR/quantile_gaussian_q025.png" || true

# Combined dimensionality dependence plot (single figure, x-axis = dimension)
PYTHONPATH=. python3 examples/plot_dimensionality_snr.py --data-dir "$DATA_DIR" --tag-prefix "mc_snr_continuous__snr_cont_fixN_varydim_" --output "$ART_DIR/snr_cont_fixN_varydim_combined.png" || true
PYTHONPATH=. python3 examples/plot_num_arms_snr.py --data-dir "$DATA_DIR" --tag-prefix "mc_snr_multiarm__snr_multiarm_fixN_varyarms_" --output "$ART_DIR/snr_multiarm_fixN_varyarms_combined.png" || true

# Build compiled plots + compiled report from stored metadata/artifacts
bash examples/run_all_plots.sh "$DATA_DIR" "$ART_DIR/compiled" || true

# Also keep a direct report for the raw artifact folder (best-effort)
PYTHONPATH=. python3 examples/write_experiment_report.py --art-dir "$ART_DIR" --data-dir "$DATA_DIR" --output "$ART_DIR/report.tex" || true

echo "Done. data_dir=$DATA_DIR artifacts_dir=$ART_DIR compiled_dir=$ART_DIR/compiled"
