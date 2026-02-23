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
  - or one value per listed rank.
- In conditional mode, `--show-delta` overlays `W_cond - W`.

```bash
python examples/plot_order_weights.py --N 120 --k 20 --ranks 1,5,10,15,20
python examples/plot_order_weights.py --N 120 --k 20 --ranks 1,3..8,12 --a 0.25
python examples/plot_order_weights.py --mode conditional --conditioned-rank 40 --N 120 --k 20 --ranks 1,5,10 --show-delta
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
- `--num-arms` controls the size of the known `(r,p)` model used for comparison.
- `--t-grid` is the number of independent repeated estimator runs to average.
- Plots both **absolute** and **relative** error versus `t` for:
  - order-statistics,
  - inclusion,
  - advantage,
  - L-advantage.
- Optional `--plot-arm-details` saves an extra figure comparing exact vs estimated per-arm
  inclusion/advantage (for rank selected by `--arm-rank`) and L-advantage at `t=max(t-grid)`.

```bash
python examples/monte_carlo_accuracy.py --N 64 --k 6 --num-arms 8 --t-grid 1,2,5,10,20,50,100,200
python examples/monte_carlo_accuracy.py --N 64 --k 6 --num-arms 8 --plot-arm-details --arm-rank 1
```

---

## Notes

- Plotting scripts require `matplotlib`.
- Outputs are saved under `examples/artifacts/` by default.
