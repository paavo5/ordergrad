# ordergrad

`ordergrad` implements fast order-statistics transforms used in the rank-note setting (`docs/ranknote.pdf`) with two complementary viewpoints:

1. **Known distribution regime (`r, p`)** (sampling with replacement): compute exact quantities for the population distribution.
2. **Batch regime (`N` observed samples)** (subset sampling without replacement inside a batch): compute exact batch-level subset expectations that act as **unbiased estimators** (in expectation over i.i.d. batches) of the known-`(r,p)` targets.

Implemented with a shared API in **NumPy**, **PyTorch**, and **JAX**.

> Reference derivations and motivation: `docs/ranknote.pdf`.

---

## What is computed

Given an order-statistic index `j` and subset/sample size parameter `k`:

- Draw `k` members (according to the regime: with replacement for known `(r,p)`, without replacement from a fixed batch for batch mode).
- Sort the resulting `k` values from smallest to largest.
- The **`j`-th order statistic** is the `j`-th ranked value in that sorted list.
- `v_j` is the expectation of that `j`-th ranked value.
- `q` is the inclusion-conditioned analogue (conditioning on a specific arm/index being included/observed first, depending on regime).
- `a = q - v` is the corresponding advantage-style difference.

For L-statistics with weights `alpha`:

\[
T = \sum_j \alpha_j X_{(j:k)},
\]

the same transforms are exposed in scalar/per-item form.

### Important interpretation

For the **batch APIs** (`expected_orderstats`, `expected_orderstats_inclusion`, `expected_orderstats_leave_one_out`, `...advantage`), the library computes these quantities **exactly for the realized batch**.  
When the batch itself is sampled i.i.d. from an underlying arm distribution, these batch quantities are the unbiased-estimation objects discussed in the note.

---

## Regimes

### A) Known `(r, p)` regime (with replacement)

- `r[b]`: fixed reward value of arm `b`
- `p[b]`: arm probability (nonnegative, sums to 1)
- draws are i.i.d. with replacement

Use:

- `expected_orderstats_known_rp(r, p)`
- `expected_orderstats_inclusion_known_rp(r, p)`
- `expected_orderstats_advantage_known_rp(r, p)`
- and L-stat counterparts.

These are exact (up to floating-point error) for the specified `(r,p)` model.

### B) Batch regime (`N` observed rewards)

- `x[0..N-1]` is a realized batch
- subset operations are without replacement within that batch

Use:

- `expected_orderstats(x)`
- `expected_orderstats_inclusion(x)`
- `expected_orderstats_leave_one_out(x)`
- `expected_orderstats_advantage(x)`
- and L-stat counterparts.

These are exact for the realized batch and are the estimator objects used in the note’s unbiasedness/equivalence results.



## Gradient-estimation motivation

A primary use-case is gradient estimation for order-statistic objectives:

- **Reparameterization-style gradient:** differentiating the computed batch quantity (e.g. `expected_lstat(...)`) gives an unbiased reparameterization-style estimator under the ranknote assumptions.
- **Likelihood-ratio-style gradient:** using the computed advantage term in a score-function estimator yields an unbiased estimator with typically lower variance.

By default, advantage outputs are detached from the computation graph (`detach_advantage=True`).
If needed for experimentation, this can be disabled by setting `detach_advantage=False`.
Non-advantage outputs are not detached.

---

## Installation

```bash
pip install -e .
```

Optional backends:

```bash
pip install -e ".[torch]"
pip install -e ".[jax]"
pip install -e ".[dev]"
```

`import ordergrad` requires only NumPy. Torch/JAX are imported lazily when those backends are requested.

---

## Quick start

### NumPy (batch regime)

```python
import numpy as np
from ordergrad import numpy_backend

N, k = 30, 8
os = numpy_backend.OrderStatTransform.precompute(N, k, dtype=np.float64)

x = np.random.randn(N)
a = np.ones(os.k) / os.k  # os.k is floor(k) when k is real

E = os.expected_orderstats(x)                      # (os.k,)
E_inc = os.expected_orderstats_inclusion(x)        # (N, os.k)
E_loo = os.expected_orderstats_leave_one_out(x)    # (N, os.k)
adv = os.expected_orderstats_advantage(x)          # (N, os.k)

l_adv = os.expected_lstat_advantage(x, a)          # (N,)

# Preset shorthands are also supported for a:
#   "TopM:m", "BotM:m", "TrimM:m", "WinsorizedM:m", "MidrangeM:m", "TopBot:m",
#   "ReMax", "ReMin", "Median", "Quantile:q", "UpperTailMean:q", "LowerTailMean:q", "HarrellDavis:q",
#   "GiniMeanDifference" (or "GMD"), "LMoment:r"
l_top2 = os.expected_lstat(x, "TopM:2")
```

Common LR-style usage with fixed `a` (preweighted transform):

```python
import numpy as np
from ordergrad import numpy_backend

N, k = 30, 8
a = np.ones(int(np.floor(k)), dtype=np.float64) / int(np.floor(k))

# Option 1: explicit two-step preweighting
os = numpy_backend.OrderStatTransform.precompute(N, k, dtype=np.float64)
os_l = os.with_lstat_weights(a)

# Option 2: one-shot convenience helper
# os_l = numpy_backend.OrderStatTransform.precompute_lstat(N, k, a, dtype=np.float64)

x = np.random.randn(N)
l_adv = os_l.expected_lstat_advantage(x)          # detached by default, shape (N,)
l_inc = os_l.expected_lstat_inclusion(x)          # shape (N,)
```

### NumPy (known `(r,p)` regime)

```python
import numpy as np
from ordergrad import numpy_backend

# N only configures precomputed batch matrices; known-(r,p) calls can still be used directly.
os = numpy_backend.OrderStatTransform.precompute(32, 5, dtype=np.float64)

r = np.array([-1.0, 0.2, 1.1, 2.4], dtype=np.float64)
p = np.array([0.1, 0.45, 0.3, 0.15], dtype=np.float64)
a = np.array([0.2, -0.1, 0.4, 0.3, 0.2], dtype=np.float64)

v = os.expected_orderstats_known_rp(r, p)                 # (os.k,)
q = os.expected_orderstats_inclusion_known_rp(r, p)        # (m, os.k)
adv = os.expected_orderstats_advantage_known_rp(r, p)      # (m, os.k)
l_adv = os.expected_lstat_advantage_known_rp(r, p, a)      # (m,)
```

Torch/JAX provide matching methods.

---


## Performance recommendations

For most workloads, prefer the **efficient** evaluation path and preweighting when `a` is fixed:

- Use `method="efficient"` for inclusion/leave-one-out/advantage calls; this is typically faster and more memory-efficient in the benchmark scripts.
- Use `with_lstat_weights(a)` (or `precompute_lstat(...)`) when the L-stat vector `a` is reused across many calls.
  This precomputes the `a`-contractions once and usually gives significantly faster repeated `expected_lstat*` evaluations than recomputing full order-stat tensors and multiplying by `a` each call.
- Enable `compute_dense_matrices=True` only when you explicitly want matmul-based dense operator paths (or are comparing methods).
  Dense precompute can be slower and uses more memory.

## API notes

Each backend exposes `OrderStatTransform` with:

- `precompute(N, k, ..., compute_dense_matrices=False)`
  - `k` may be integer or real
  - internal order-stat dimension is `floor(k)` and available as `transform.k`
- batch-regime order-stat methods:
  - `expected_orderstats(x)`
  - `expected_orderstats_inclusion(x, method="efficient"|"matmul"|"auto")`
  - `expected_orderstats_leave_one_out(x, method="efficient"|"matmul"|"auto")`
  - `expected_orderstats_advantage(x, method="efficient"|"matmul"|"auto", detach_advantage=True)`
- batch-regime L-stat methods:
  - `expected_lstat(x, a=None)`
  - `expected_lstat_inclusion(x, a=None, method="efficient"|"matmul"|"auto")`
  - `expected_lstat_leave_one_out(x, a=None, method="efficient"|"matmul"|"auto")`
  - `expected_lstat_advantage(x, a=None, method="efficient"|"matmul"|"auto", detach_advantage=True)`
  - `a` can be either a numeric vector of shape `(floor(k),)` or a preset string:
    - `"TopM:m"`: equal weight on top `m` ranks (`j=1` is top rank)
    - `"BotM:m"`: equal weight on bottom `m` ranks (largest `j` ranks)
    - `"TrimM:m"`: trimmed mean over middle ranks after removing top/bottom `m` (requires `2*m < floor(k)`)
    - `"WinsorizedM:m"`: winsorized mean (replace bottom/top `m` values by the next interior values)
    - `"MidrangeM:m"` / `"TopBot:m"`: average of bottom-`m` mean and top-`m` mean
    - `"ReMax"`: top-1 only (`j=1`)
    - `"ReMin"`: bottom-1 only (`j=floor(k)`)
    - `"Median"`: sample median (middle rank or average of two middle ranks)
    - `"Quantile:q"`: place all mass on the rank nearest quantile `q` (`q=0 -> j=1`, `q=1 -> j=floor(k)`)
    - `"UpperTailMean:q"`: mean over top `ceil(q * floor(k))` ranks (`0 < q <= 1`)
    - `"LowerTailMean:q"`: mean over bottom `ceil(q * floor(k))` ranks (`0 < q <= 1`)
    - `"HarrellDavis:q"` (alias `"HarrelDavis:q"`): Harrell–Davis quantile estimator at quantile `q in [0,1]`
    - `"GiniMeanDifference"` / `"GMD"`: sample Gini mean difference L-estimator
    - `"LMoment:r"`: sample L-moment of order `r` (`1 <= r <= floor(k)`)
- known-`(r,p)` order-stat methods:
  - `expected_orderstats_known_rp(r, p)`
  - `expected_orderstats_inclusion_known_rp(r, p)`
  - `expected_orderstats_advantage_known_rp(r, p, detach_advantage=True)`
- known-`(r,p)` L-stat methods:
  - `expected_lstat_known_rp(r, p, a)`
  - `expected_lstat_inclusion_known_rp(r, p, a)`
  - `expected_lstat_advantage_known_rp(r, p, a, detach_advantage=True)`

When `compute_dense_matrices=True`, inclusion/leave-one-out/advantage can use explicit dense matmul paths; otherwise efficient prefix/suffix implementations are used.

---

## Testing

```bash
pytest -q
```

Backend-specific examples:

```bash
pytest -m "not torch and not jax"
pytest -m torch
pytest -m jax
```

---

## Limitations / assumptions

- Known-`(r,p)` formulas assume i.i.d. sampling with replacement from the specified discrete distribution.
- Batch transforms assume uniform subset sampling without replacement within the realized batch.
- Gradients wrt values are piecewise-constant away from ties (stable sorting is used).

## License

MIT
