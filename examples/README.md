# Examples playground

These scripts are intended as a hands-on playground for understanding and profiling `ordergrad` behavior.

## 1) Plot order weights

`plot_order_weights.py` plots order-statistic weight curves over sorted rank `m`.

### Features
- `--mode unconditional`: plot `W[m,j]`.
- `--mode conditional`: plot conditional inclusion weights for a fixed conditioned sorted rank `r`.
- `--ranks` supports explicit lists and inclusive ranges: e.g. `1,3..6,10`.
- Optional `--a` lets you define sparse L-stat coefficients on listed ranks:
  - one value: broadcast to all listed ranks,
  - or one value per listed rank,
  - or a preset string (e.g. `TopM:3`, `Median`, `TopBot:2`).
- In conditional mode, `--show-delta` overlays `W_cond - W`.
- In conditional mode, `--show-leave-one-out` overlays leave-one-out weights for excluding the same conditioned rank.

```bash
python examples/plot_order_weights.py --N 120 --k 20 --ranks 1,5,10,15,20
python examples/plot_order_weights.py --N 120 --k 20 --ranks 1,3..8,12 --a 0.25
python examples/plot_order_weights.py --N 120 --k 20 --ranks 1,3..8,12 --a TopM:3
python examples/plot_order_weights.py --mode conditional --conditioned-rank 40 --N 120 --k 20 --ranks 1,5,10 --show-delta
python examples/plot_order_weights.py --mode conditional --conditioned-rank 40 --N 120 --k 20 --ranks 1,5,10 --show-leave-one-out
```

## 2) Benchmark methods

`benchmark_methods.py` compares runtime tradeoffs for three backends (`np`, `jax`, `torch`) selected by `--backend`.

- Backends are imported lazily (JAX/Torch are imported only if requested).
- Reports precompute time for:
  - **no-dense**: does not build dense rank-space operators,
  - **dense**: builds dense matrices (`M_inc`, `M_loo`, `M_adv`) used by matmul mode.
- Reports per-call runtime for unconditional, inclusion (`inc`), advantage (`adv`), and L-advantage methods.
- Also explicitly compares **full order-stat computation + dot with `a`** versus **direct preweighted L-stat computation** (`with_lstat_weights(a)`), which behaves like computing one weighted statistic directly.
- `--efficient` runs only efficient/no-dense computations (skips dense precompute and matmul rows), including direct preweighted (`with_lstat_weights(a)`) efficient benchmarks.

```bash
python examples/benchmark_methods.py --backend np --N 500 --k 40 --repeats 100
python examples/benchmark_methods.py --backend jax --N 500 --k 40 --repeats 100
python examples/benchmark_methods.py --backend torch --N 500 --k 40 --repeats 100
python examples/benchmark_methods.py --backend np --N 500 --k 40 --repeats 100 --efficient
```

## 3) Monte Carlo estimator accuracy curve

`monte_carlo_accuracy.py` checks that repeated averages of the **batch estimator** converge to the exact known-`(r,p)` target.

- One estimator run = one batch of `N` sampled values with estimator parameter `k`.
- `--backend {np,jax,torch}` selects backend lazily (JAX/Torch imported only when requested).
- `--num-arms` controls the size of the known `(r,p)` model used for comparison.
- `--t-grid` is the number of independent repeated estimator runs to average.
- `--a` sets L-stat weights (single value broadcast, comma list of length `floor(k)`, or preset string such as `TopM:3`).
- Internally the script preweights using `with_lstat_weights(a)` so `L-advantage` uses the precomputed fast path even for default `a`.
- Plots both **absolute** and **relative** error versus `t` for:
  - order-statistics,
  - inclusion,
  - advantage,
  - L-advantage.
- Optional `--plot-arm-details` saves an extra figure comparing exact vs estimated per-arm
  inclusion/advantage (for rank selected by `--arm-rank`) and L-advantage at `t=max(t-grid)`.
- Uses a buffered sampler (`--sample-buffer-size`) that pre-draws many arm indices at once,
  then serves per-batch requests from that buffer before refilling.

```bash
python examples/monte_carlo_accuracy.py --backend np --N 64 --k 6 --num-arms 8 --t-grid 1,2,5,10,20,50,100,200
python examples/monte_carlo_accuracy.py --backend torch --N 64 --k 6 --num-arms 8 --plot-arm-details --arm-rank 1
python examples/monte_carlo_accuracy.py --backend jax --N 64 --k 6 --num-arms 8 --sample-buffer-size 500000
python examples/monte_carlo_accuracy.py --backend np --N 64 --k 6 --num-arms 8 --a TopM:3
python examples/monte_carlo_accuracy.py --backend np --N 64 --k 6 --num-arms 8 --a 0.2,0.1,0.3,0.15,0.1,0.15
```


## 4) Monte Carlo gradient check (multi-arm)

`mc_gradients_multiarm.py` (torch-only) compares an LR gradient estimator
(using the computed L-advantage baseline) against an exact known-(r,p) gradient
obtained by differentiating the exact objective with torch autograd.
It applies a `k` multiplier to the LR estimator (`k * mean(adv * score)`),
which is needed for unbiasedness under this formulation.

```bash
python examples/mc_gradients_multiarm.py --N 64 --k 6 --num-arms 8 --t-grid 1,2,5,10,20,50,100,200
```

## 5) Monte Carlo gradient check (continuous)

`mc_gradients_continuous.py` (torch/autograd-only) compares reparameterization
(pathwise/RP) and advantage-based LR gradient estimators for a continuous
Normal-location model with a quadratic reward transform.
This script also uses the `k` multiplier in the LR estimator (`k * mean(adv * score)`),
as required for unbiasedness in this setup.
The dimensionality is configurable via `--dim`: for `dim>1`, the script uses a
vector location parameter and reports vector-gradient mismatch statistics.

```bash
python examples/mc_gradients_continuous.py --N 64 --k 6 --mu 0.5 --center 1.0 --t-grid 1,2,5,10,20,50,100,200
python examples/mc_gradients_continuous.py --N 64 --k 6 --dim 4 --mu 0.5 --center 1.0 --t-grid 1,2,5,10,20,50,100,200
```

## 6) Gradient variance / SNR vs k (multi-arm)

`mc_snr_multiarm.py` estimates gradient variance and signal-to-noise ratio
for the multi-arm LR estimator across a list of `k` values.

- Uses Monte Carlo repeated gradient estimates (`--num-mc`) for each `k` in `--k-grid`.
- Reports and plots:
  - `V[g]`: sum of per-dimension variances,
  - `SNR = ||E[g]||^2 / V[g]`.
- Supports numeric or preset `--a` definitions (e.g. `TopM:3`).
- Optional `--store-data` writes arrays (`.npz`) and experiment setup metadata (`.json`) to `--data-dir`, keyed by `--tag`.

```bash
python examples/mc_snr_multiarm.py --N 64 --num-arms 8 --k-grid 1,2,3,4,5,6 --num-mc 2000 --a TopM:3
python examples/mc_snr_multiarm.py --N 64 --num-arms 8 --k-grid 1,2,3,4,5,6 --num-mc 2000 --a TopM:3 --store-data --tag topm3_baseline
```

## 7) Gradient variance / SNR vs k (continuous)

`mc_snr_continuous.py` compares RP and LR gradient estimator variance/SNR as
`k` changes in the continuous Normal-location setting.

- Computes RP and LR gradient samples for each `k` in `--k-grid`.
- Plots `V[g]` and `SNR = ||E[g]||^2 / V[g]` for both estimators.
- Supports multi-dimensional parameterization via `--dim` and numeric/preset `--a`.
- Optional `--store-data` writes arrays (`.npz`) and experiment setup metadata (`.json`) to `--data-dir`, keyed by `--tag`.

```bash
python examples/mc_snr_continuous.py --N 64 --dim 2 --k-grid 1,2,3,4,5,6 --num-mc 2000 --a TopBot:2
python examples/mc_snr_continuous.py --N 64 --dim 2 --k-grid 1,2,3,4,5,6 --num-mc 2000 --a TopBot:2 --store-data --tag topbot2_dim2
```

---

## Notes

- Plotting scripts require `matplotlib`.
- Outputs are saved under `examples/artifacts/` by default.
