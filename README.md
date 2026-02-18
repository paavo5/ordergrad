# ordergrad

Exact, fast computation of **expected order statistics** under **uniform sampling without replacement**:

- You have a population of `N` values `x[0..N-1]`.
- You sample a subset `S` of size `k` uniformly at random (no replacement).
- Let `X_(1:k) <= ... <= X_(k:k)` be the **order statistics** of the selected values `{x[i] : i in S}`.

This library computes (exactly, up to floating-point precision):

1. **Unconditional** expectations: `E[X_(j:k)]` for all `j=1..k`
2. **Conditional inclusion** expectations: `E[X_(j:k) | i ∈ S]` for all items `i`
3. **Leave-one-out** expectations: `E[X_(j:k)]` when sampling from the population with item `i` removed

and wraps these into a convenient **reward transform** for any **L-statistic**

\[
T(S) = \sum_{j=1}^k a_j X_{(j:k)}.
\]

Implemented in **NumPy**, **PyTorch**, and **JAX** with the same API.

> Reference note / derivations: see `docs/ranknote.pdf`.


## Key identity (unconditional weights)

Let `x_(1) <= ... <= x_(N)` be the sorted population values.

\[
\mathbb{E}[X_{(j:k)}] = \sum_{m=1}^N x_{(m)} W_{m,j},
\qquad
W_{m,j} = \frac{\binom{m-1}{j-1}\binom{N-m}{k-j}}{\binom{N}{k}}.
\]

For fixed `(N, k)`, the matrix `W` depends only on combinatorics, so we precompute it once.
Then each evaluation is: **sort** `x` and do a single **matrix–vector multiply**.

The conditional-inclusion and leave-one-out computations are also vectorized and run in `O(Nk)`
(using only elementwise multiplies and prefix/suffix sums).


## Installation

This repo is a standard Python package.

```bash
pip install -e .
```

Optional backends:

```bash
pip install -e ".[torch]"   # PyTorch backend
pip install -e ".[jax]"     # JAX backend
pip install -e ".[dev]"     # pytest
```

### Important: optional imports

`ordergrad` is designed so that you can install **only what you use**:

- `import ordergrad` requires **NumPy only**
- PyTorch is imported only if you import `ordergrad.torch_backend` (or `get_backend("torch")`)
- JAX is imported only if you import `ordergrad.jax_backend` (or `get_backend("jax")`)


## Quick start

### NumPy

```python
import numpy as np
from ordergrad import numpy_backend

N, k = 30, 8
os = numpy_backend.OrderStatTransform.precompute(N, k, dtype=np.float64)

x = np.random.randn(N)

E = os.expected_orderstats(x)                # (k,)
E_inc = os.expected_orderstats_inclusion(x)  # (N,k)
E_loo = os.expected_orderstats_leave_one_out(x)  # (N,k)

# L-statistic: T(S) = sum_j a_j X_(j:k)
a = np.ones(k) / k

scalar = os.expected_lstat(x, a)                 # scalar
per_item_inc = os.expected_lstat_inclusion(x, a) # (N,)
per_item_loo = os.expected_lstat_leave_one_out(x, a) # (N,)

# A common “advantage-style” transform:
adv = os.expected_lstat_advantage(x, a)          # (N,) == per_item_inc - per_item_loo
```

### PyTorch

```python
import torch
from ordergrad import torch_backend

N, k = 30, 8
os = torch_backend.OrderStatTransform.precompute(N, k, dtype=torch.float64)

x = torch.randn(N, dtype=torch.float64, requires_grad=True)
a = torch.ones(k, dtype=torch.float64) / k

loss = os.expected_lstat(x, a)
loss.backward()

# Piecewise-constant “order-statistics gradient” (away from ties)
print(x.grad)
```

### JAX

```python
import jax
import jax.numpy as jnp
from ordergrad import jax_backend

jax.config.update("jax_enable_x64", True)

N, k = 30, 8
os = jax_backend.OrderStatTransform.precompute(N, k, dtype=jnp.float64)

x = jnp.arange(N, dtype=jnp.float64)
a = jnp.ones(k, dtype=jnp.float64) / k

g = jax.grad(lambda z: os.expected_lstat(z, a))(x)
print(g)
```


## API notes

Each backend exposes an `OrderStatTransform` class with:

- `precompute(N, k, ..., compute_dense_matrices=False[, kappa=None])` (NumPy supports optional real `kappa`)
- `expected_orderstats(x) -> (k,)`
- `expected_orderstats_inclusion(x, method="efficient"|"matmul"|"auto") -> (N,k)`
- `expected_orderstats_leave_one_out(x, method="efficient"|"matmul"|"auto") -> (N,k)` (requires `k <= N-1`)
- `expected_lstat(x, a) -> scalar`
- `expected_lstat_inclusion(x, a, method="efficient"|"matmul"|"auto") -> (N,)`
- `expected_lstat_leave_one_out(x, a, method="efficient"|"matmul"|"auto") -> (N,)`
- `expected_lstat_advantage(x, a, method="efficient"|"matmul"|"auto") -> (N,)`
- `expected_orderstats_advantage(x, method="efficient"|"matmul"|"auto") -> (N,k)`
- `expected_orderstats_known_rank_position(x, p) -> (N,k)` and `expected_lstat_known_rank_position(x, a, p) -> (N,)` for the known `(r,p)` conditioning variant
- `with_lstat_weights(a)` and `precompute_lstat(N, k, a, ...)` for preweighted L-stat transforms

When `compute_dense_matrices=True`, inclusion/leave-one-out/advantage can run in the explicit pipeline:
`sort -> matmul -> revert sort`. The original prefix/suffix efficient methods remain available via `method="efficient"`.


## Testing

```bash
pytest
```

Run backend-specific tests in separate environments:

```bash
# NumPy-only checks
pytest -m "not torch and not jax"

# PyTorch-only checks
pytest -m torch
# or: pytest tests/test_torch_backend.py

# JAX-only checks
pytest -m jax
# or: pytest tests/test_jax_backend.py
```


## Performance notes

- Precomputation cost is paid once per `(N, k)`.
- Memory is dominated by storing the weight matrices (roughly `O(Nk)`).
- NumPy precompute supports `chunk_size=` to reduce temporary peak memory.

## Limitations

- Assumes **uniform** subset sampling without replacement.
- Gradients w.r.t. `x` are only well-defined **away from ties** (stable sort is used for deterministic behavior at ties).

## License

MIT.
