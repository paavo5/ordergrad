# OrderGrad

`ordergrad` provides fast order-statistic and L-statistic reward transforms for optimizing objectives beyond the ordinary mean. It is designed to plug into familiar likelihood-ratio policy-gradient updates and reparameterized/pathwise updates while changing only the scalar learning signal.

![OrderGrad overview](docs/assets/ordergrad_overview.png)

OrderGrad sorts rewards inside a small comparison set, chooses which ranks matter, and uses those rank weights to target objectives such as best-of-`k`, Top-`M`@`K`, lower-tail / CVaR-style criteria, medians, trimmed means, quantiles, and other L-statistics.

Implemented backends:

- NumPy
- PyTorch
- JAX

---

## Installation

```bash
pip install -e .
```

Optional backend dependencies:

```bash
pip install -e ".[torch]"
pip install -e ".[jax]"
pip install -e ".[dev]"
```

`import ordergrad` only requires NumPy. Torch and JAX are imported lazily when their backends are requested.

---

## Quick start: standard LR / score-function policy gradient

The most common use is a REINFORCE-style or policy-gradient update. In the ordinary mean-reward objective, the score-function estimator uses

\[
\hat g_{\text{mean}} = \frac{1}{N}\sum_{i=1}^N (R_i - b_i)\nabla_\theta \log p_\theta(x_i).
\]

OrderGrad keeps the same update structure, but replaces the scalar reward-minus-baseline term with a rank-based advantage computed from the batch:

\[
\hat g_{\text{LR-OG}} = \frac{k}{N}\sum_{i=1}^N a_i^{(\alpha)}\nabla_\theta \log p_\theta(x_i).
\]

Here `k` is the order-statistic comparison size, and `alpha` or a preset such as `"TopM:2"` says which sorted ranks to optimize.

```python
import torch
from ordergrad import torch_backend

# N = optimization batch size; k = comparison size for the order objective.
N = 32
k = 4
objective = "TopM:2"  # average the top 2 rewards among a size-4 comparison set

og = torch_backend.OrderStatTransform.precompute_lstat(
    N, k, objective, dtype=torch.float32
)

# Minimal bandit-style example. In an RL/LLM setting, replace this with
# samples, log_probs, and rewards from your policy rollout code.
num_arms = 5
true_rewards = torch.tensor([-1.0, 0.0, 0.2, 1.0, 2.0])
logits = torch.zeros(num_arms, requires_grad=True)
optimizer = torch.optim.Adam([logits], lr=3e-3)

optimizer.zero_grad()
dist = torch.distributions.Categorical(logits=logits)
actions = dist.sample((N,))
log_probs = dist.log_prob(actions)
rewards = true_rewards[actions]

# OrderGrad LR transform: rewards -> rank advantages.
# detach_advantage=True is the default and is usually what you want for LR.
advantages = og.lstat_advantage(rewards, detach_advantage=True)

# Maximize the OrderGrad objective. PyTorch optimizers minimize losses,
# so use a minus sign. k * mean(...) equals (k / N) * sum(...).
loss = -k * (advantages * log_probs).mean()
loss.backward()
optimizer.step()
```

For prompt-conditioned LLM post-training or grouped RL rollouts, compute `lstat_advantage` inside each prompt/context group, then concatenate the per-sample advantages before applying the usual policy-gradient, PPO, or GRPO-style loss.

---

## Quick start: reparameterization / pathwise gradient

Use the reparameterization style when your samples are differentiable transformations of parameter-free noise, for example `x = T_theta(eps)`, and the reward is differentiable with respect to the sample. In this case, do not use `lstat_advantage`; differentiate the rank-weighted batch value directly.

```python
import torch
from ordergrad import torch_backend

N = 64
k = 8
objective = "LowerTailMean:0.25"  # optimize the lower 25% tail of the size-k sample

og = torch_backend.OrderStatTransform.precompute_lstat(
    N, k, objective, dtype=torch.float32
)

theta = torch.tensor(0.0, requires_grad=True)
optimizer = torch.optim.Adam([theta], lr=3e-3)

optimizer.zero_grad()
eps = torch.randn(N)
x = theta + eps                         # reparameterized samples
rewards = -(x - 2.0).pow(2)             # differentiable reward
value = og.lstat(rewards)               # rank-weighted value v^alpha
loss = -value                           # maximize value
loss.backward()
optimizer.step()
```

Away from ties, the sorting permutation is locally constant, so the gradient is the weighted sum of reward gradients at the sorted batch positions. With exact ties or atoms, use the library's stable-sort behavior, or use a differentiable sorting relaxation if your application needs smooth tie handling.

---

## Method in one page

### Objective

Given `k` sampled rewards, sort them from smallest to largest:

\[
R_{(1:k)} \le \cdots \le R_{(k:k)}.
\]

An OrderGrad objective is a finite-sample L-statistic:

\[
J_{k,\alpha}(\theta) = \mathbb{E}\left[\sum_{j=1}^k \alpha_j R_{(j:k)}\right].
\]

Changing only the rank weights `alpha` changes the learning objective:

- `"TopM:1"` or `"ReMax"`: best-of-`k` / max@`k`
- `"TopM:m"`: average the top `m` outcomes
- `"BotM:m"`: average the bottom `m` outcomes
- `"LowerTailMean:q"`: lower-tail / CVaR-style objective
- `"UpperTailMean:q"`: upper-tail mean
- `"Median"`: median objective
- `"TrimM:m"`: trimmed mean
- `"Quantile:q"`: quantile-style objective
- `"TopBot:m"`: place weight on both tails

In code, preset names count ranks in the intuitive top-rank convention: `Rank:1` is the maximum and `TopM:2` means the two largest ranks. Numeric vectors are also provided in top-rank order. Internally, the mathematical order statistics are sorted from smallest to largest.

### Batch value computation

For a realized reward batch `R_1, ..., R_N`, OrderGrad considers uniformly sampled size-`k` subsets from the batch. The expected `j`-th order statistic of such a subset is

\[
v_j = \mathbb{E}\left[(R_S)_{(j:k)} \mid R_{1:N}\right].
\]

After sorting the full batch as `R_(1:N) <= ... <= R_(N:N)`, this becomes a fixed weighted sum:

\[
v_j = \sum_{m=1}^N R_{(m:N)} W_{m,j},
\]

where `W_{m,j}` is the probability that sorted batch item `m` becomes rank `j` in a uniformly selected size-`k` subset. The L-statistic value is

\[
v^\alpha = \sum_{j=1}^k \alpha_j v_j.
\]

This is what `lstat(rewards, alpha)` computes.

### LR-style advantage computation

For likelihood-ratio gradients, OrderGrad needs a per-sample learning signal. For each sample `i` and rank `j`, it computes:

\[
q_{i,j} = \mathbb{E}\left[(R_S)_{(j:k)} \mid i \in S, R_{1:N}\right],
\]

which is the include-one value, and

\[
v^{(-i)}_j = \mathbb{E}\left[(R_S)_{(j:k)} \mid S \subset [N] \setminus \{i\}, R_{1:N}\right],
\]

which is the leave-one-out baseline. The rankwise advantage is

\[
a_{i,j} = q_{i,j} - v^{(-i)}_j,
\]

and the L-statistic advantage is

\[
a_i^{(\alpha)} = \sum_{j=1}^k \alpha_j a_{i,j}.
\]

This is what `lstat_advantage(rewards, alpha)` returns. The leave-one-out baseline excludes sample `i`, so it can be multiplied by `nabla log p_theta(x_i)` in a score-function estimator without introducing baseline bias. The LR advantage path requires `k < N`, because the leave-one-out computation must still be able to draw a size-`k` subset after removing one item.

### RP-style computation

For reparameterized samples, OrderGrad uses the differentiable batch value:

\[
\hat g_{\text{RP-OG}} = \nabla_\theta v^\alpha(\theta).
\]

In code this is just:

```python
value = og.lstat(rewards)
loss = -value
loss.backward()
```

Use this path when `rewards` carries the autograd path to the model parameters. Use the LR path when samples are discrete, non-differentiable, or only `log_prob` gradients are available.

---

## API overview

Each backend exposes an `OrderStatTransform` class.

```python
from ordergrad import numpy_backend
from ordergrad import torch_backend
from ordergrad import jax_backend
```

Precompute once for fixed `(N, k)`:

```python
og = torch_backend.OrderStatTransform.precompute(N, k)
```

Precompute and collapse a fixed L-statistic objective once:

```python
og = torch_backend.OrderStatTransform.precompute_lstat(N, k, "TopM:2")
# equivalent to: OrderStatTransform.precompute(N, k).with_lstat_weights("TopM:2")
```

Common batch-regime methods:

```python
og.orderstats(rewards)                    # shape (k,)
og.orderstats_inclusion(rewards)          # shape (N, k)
og.orderstats_leave_one_out(rewards)      # shape (N, k)
og.orderstats_advantage(rewards)          # shape (N, k)

og.lstat(rewards, "TopM:2")               # scalar value
og.lstat_inclusion(rewards, "TopM:2")     # shape (N,)
og.lstat_leave_one_out(rewards, "TopM:2") # shape (N,)
og.lstat_advantage(rewards, "TopM:2")     # shape (N,)
```

When you use `precompute_lstat`, the objective can be omitted:

```python
og = torch_backend.OrderStatTransform.precompute_lstat(N, k, "TopM:2")
value = og.lstat(rewards)
adv = og.lstat_advantage(rewards)
```

Known discrete distribution regime:

```python
v = og.orderstats_known_rp(r, p)
q = og.orderstats_inclusion_known_rp(r, p)
adv = og.orderstats_advantage_known_rp(r, p)

l_value = og.lstat_known_rp(r, p, "TopM:2")
l_adv = og.lstat_advantage_known_rp(r, p, "TopM:2")
```

Here `r[b]` is the reward for arm `b`, and `p[b]` is the sampling probability.

---

## L-statistic presets

Useful presets include:

| Preset | Meaning |
| --- | --- |
| `"ReMax"` | maximum / best-of-`k` objective |
| `"ReMin"` | minimum-of-`k` objective |
| `"Rank:r"` | one-based top-rank selector; `Rank:1` is max |
| `"TopM:m"` | average top `m` ranks |
| `"BotM:m"` | average bottom `m` ranks |
| `"TopBot:m"` | average top `m` and bottom `m` ranks |
| `"Median"` | sample median |
| `"TrimM:m"` | drop top and bottom `m`, average the middle |
| `"WinsorizedM:m"` | winsorized mean |
| `"Quantile:q"` | Hazen-style quantile with mass-below convention |
| `"LowerTailMean:q"` | average lower `ceil(q*k)` ranks |
| `"UpperTailMean:q"` | average upper `ceil(q*k)` ranks |
| `"RangeLowerTailMean:lo:hi"` | lower-tail band |
| `"RangeUpperTailMean:lo:hi"` | upper-tail band |
| `"TrimmedMeanFrac:lo:hi"` | central fractional trimmed mean |
| `"HarrellDavis:q"` | Harrell-Davis quantile estimator |
| `"GiniMeanDifference"` / `"GMD"` | signed spread statistic |
| `"LMoment:r"` | sample L-moment |

You can also pass a numeric vector of length `floor(k)`. Numeric vectors are interpreted in top-rank order. For example, with `k = 4`, `[1, 0, 0, 0]` selects the maximum, while `[0, 0, 0, 1]` selects the minimum.

---

## Performance tips

- Precompute the transform once for each fixed `(N, k)`.
- Use `precompute_lstat(...)` or `with_lstat_weights(...)` when the rank weights are fixed across many updates.
- Prefer `method="efficient"` unless you are explicitly benchmarking dense matrix paths.
- For fixed `N`, `k`, and `alpha`, the collapsed L-statistic path is dominated by sorting, so a minibatch update costs `O(N log N)` after precomputation.
- Build dense matrices only when you need them: `compute_dense_matrices=True` can be useful for comparisons but uses more memory.

---

## Testing

```bash
pytest -q
```

Backend-specific tests:

```bash
pytest -m "not torch and not jax"
pytest -m torch
pytest -m jax
```

---

## Assumptions and limitations

- The batch-regime value and advantage transforms assume uniform subset sampling without replacement from the realized batch.
- The LR advantage estimator uses `k < N`; value/RP computations can use `k <= N`.
- The known-`(r, p)` formulas assume i.i.d. sampling with replacement from the specified discrete distribution.
- Gradients through sorting are piecewise valid away from ties. Stable sorting is used for ties.
- In practical PPO/GRPO-style training, clipping, normalization, off-policy data, reward-model noise, and prompt dependence are application-level considerations layered on top of the exact transform.

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
