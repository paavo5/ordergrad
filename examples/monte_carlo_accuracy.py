#!/usr/bin/env python3
"""Monte Carlo estimator accuracy playground.

Each estimator run draws one batch of N i.i.d. samples from a known discrete
(r, p) distribution, then applies the batch estimator with parameter k.
The script averages across t independent runs and compares that average to the
exact known-(r,p) target, showing convergence as t grows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _load_backend(name: str):
    """Load requested backend lazily and return helpers."""
    if name == "np":
        from ordergrad.numpy_backend import OrderStatTransform

        def to_backend(arr: np.ndarray) -> np.ndarray:
            return arr

        def to_numpy(arr: Any) -> np.ndarray:
            return np.asarray(arr)

        return "NumPy", OrderStatTransform, to_backend, to_numpy, np.float64

    if name == "jax":
        import jax.numpy as jnp
        from ordergrad.jax_backend import OrderStatTransform

        def to_backend(arr: np.ndarray):
            return jnp.asarray(arr)

        def to_numpy(arr: Any) -> np.ndarray:
            return np.asarray(arr)

        return "JAX", OrderStatTransform, to_backend, to_numpy, jnp.float64

    if name == "torch":
        import torch
        from ordergrad.torch_backend import OrderStatTransform

        def to_backend(arr: np.ndarray):
            return torch.tensor(arr, dtype=torch.float64)

        def to_numpy(arr: Any) -> np.ndarray:
            return arr.detach().cpu().numpy()

        return "PyTorch", OrderStatTransform, to_backend, to_numpy, torch.float64

    raise ValueError(f"Unsupported backend: {name}")


class BufferedIndexSampler:
    """Sample arm indices via large buffered draws for lower overhead."""

    def __init__(self, rng: np.random.Generator, num_arms: int, probs: np.ndarray, *, buffer_size: int):
        self.rng = rng
        self.num_arms = int(num_arms)
        self.probs = np.asarray(probs, dtype=np.float64)
        self.buffer_size = int(buffer_size)
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self._buf = np.empty(0, dtype=np.int64)
        self._pos = 0

    def _refill(self) -> None:
        self._buf = self.rng.choice(self.num_arms, size=self.buffer_size, replace=True, p=self.probs)
        self._pos = 0

    def sample(self, n: int) -> np.ndarray:
        n = int(n)
        out = np.empty(n, dtype=np.int64)
        filled = 0
        while filled < n:
            if self._pos >= self._buf.size:
                self._refill()
            take = min(n - filled, self._buf.size - self._pos)
            out[filled : filled + take] = self._buf[self._pos : self._pos + take]
            self._pos += take
            filled += take
        return out


def _single_batch_estimates(
    os,
    os_l,
    idx_sampler: BufferedIndexSampler,
    *,
    r: np.ndarray,
    N: int,
    to_backend: Callable[[np.ndarray], Any],
    to_numpy: Callable[[Any], np.ndarray],
):
    idx = idx_sampler.sample(N)
    x = to_backend(r[idx])
    v = to_numpy(os.expected_orderstats(x))
    inc = to_numpy(os.expected_orderstats_inclusion(x))
    adv = to_numpy(os.expected_orderstats_advantage(x))
    l_adv = to_numpy(os_l.expected_lstat_advantage(x))
    return v, inc, adv, l_adv, idx


def _arm_means_from_items(values: np.ndarray, idx: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate item-indexed values to arm means.

    values: (N,k) or (N,)
    idx: (N,) arm ids
    Returns (means_by_arm, counts_by_arm).
    """
    counts = np.bincount(idx, minlength=m).astype(np.int64)
    if values.ndim == 1:
        sums = np.bincount(idx, weights=values, minlength=m).astype(np.float64)
        means = np.full(m, np.nan, dtype=np.float64)
        mask = counts > 0
        means[mask] = sums[mask] / counts[mask]
        return means, counts

    _, k = values.shape
    out = np.full((m, k), np.nan, dtype=np.float64)
    for j in range(k):
        sums_j = np.bincount(idx, weights=values[:, j], minlength=m).astype(np.float64)
        mask = counts > 0
        out[mask, j] = sums_j[mask] / counts[mask]
    return out, counts


def _parse_a(spec: str | None, k_ord: int) -> np.ndarray:
    if spec is None:
        return np.linspace(0.3, 1.0, k_ord, dtype=np.float64)
    vals = [float(x) for x in spec.split(",") if x.strip()]
    if len(vals) == 0:
        raise SystemExit("--a was provided but no values were parsed.")
    if len(vals) == 1:
        vals = vals * k_ord
    elif len(vals) != k_ord:
        raise SystemExit(f"--a must have either 1 value or exactly floor(k)={k_ord} values.")
    return np.asarray(vals, dtype=np.float64)


def _mean_abs_and_rel_error(est: np.ndarray, exact: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    abs_err = np.abs(est - exact)
    rel_err = abs_err / (np.abs(exact) + eps)
    return float(np.nanmean(abs_err)), float(np.nanmean(rel_err))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot estimator error vs number of repeated batch estimates (t).")
    ap.add_argument("--backend", type=str, default="np", choices=["np", "jax", "torch"])
    ap.add_argument("--N", type=int, default=64, help="Batch size per estimator evaluation.")
    ap.add_argument("--k", type=float, default=6.0, help="Estimator k parameter (can be real).")
    ap.add_argument("--num-arms", type=int, default=6, help="Number of arms in the known-(r,p) model.")
    ap.add_argument("--a", type=str, default=None, help="L-stat weights: single value (broadcast) or comma-separated floor(k)-vector.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-buffer-size", type=int, default=200_000, help="Number of arm indices to pre-sample per buffer refill.")
    ap.add_argument(
        "--t-grid",
        type=str,
        default="1,2,5,10,20,50,100,200,500",
        help="Comma-separated repetition counts t (number of independent estimator batches to average).",
    )
    ap.add_argument("--arm-rank", type=int, default=1, help="1-based rank index used in arm-detail plots.")
    ap.add_argument("--plot-arm-details", action="store_true", help="Also save arm-wise exact vs estimate plots at max(t-grid).")
    ap.add_argument("--output", type=str, default="examples/artifacts/mc_error_curve.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.") from e

    backend_name, OrderStatTransform, to_backend, to_numpy, dtype = _load_backend(args.backend)

    rng = np.random.default_rng(args.seed)
    if args.num_arms < 2:
        raise SystemExit("--num-arms must be >= 2")
    m = args.num_arms
    r = np.sort(rng.normal(loc=0.0, scale=1.0, size=m).astype(np.float64))
    p = rng.dirichlet(np.ones(m, dtype=np.float64)).astype(np.float64)

    idx_sampler = BufferedIndexSampler(rng, m, p, buffer_size=args.sample_buffer_size)

    k_ord = int(np.floor(args.k))
    if k_ord < 1:
        raise SystemExit("Need floor(k) >= 1")
    if not (1 <= args.arm_rank <= k_ord):
        raise SystemExit(f"--arm-rank must be in [1, {k_ord}]")
    a = _parse_a(args.a, k_ord)
    a_backend = to_backend(a)

    # Batch estimator (unknown-distribution regime).
    os_batch = OrderStatTransform.precompute(
        args.N,
        args.k,
        dtype=dtype,
        compute_conditional=True,
        compute_leave_one_out=True,
    )
    os_batch_l = os_batch.with_lstat_weights(a_backend)

    # Exact known-(r,p) target (known-distribution regime).
    os_exact = OrderStatTransform.precompute(max(args.N, 2), args.k, dtype=dtype, compute_conditional=True, compute_leave_one_out=True)
    os_exact_l = os_exact.with_lstat_weights(a_backend)
    v_exact = to_numpy(os_exact.expected_orderstats_known_rp(r, p))
    inc_exact_by_arm = to_numpy(os_exact.expected_orderstats_inclusion_known_rp(r, p))
    adv_exact_by_arm = to_numpy(os_exact.expected_orderstats_advantage_known_rp(r, p))
    l_adv_exact_by_arm = to_numpy(os_exact_l.expected_lstat_advantage_known_rp(r, p, a_backend))

    t_grid = [int(x) for x in args.t_grid.split(",") if x.strip()]
    if any(t <= 0 for t in t_grid):
        raise SystemExit("All t-grid entries must be positive.")

    err_v_abs, err_v_rel = [], []
    err_inc_abs, err_inc_rel = [], []
    err_adv_abs, err_adv_rel = [], []
    err_ladv_abs, err_ladv_rel = [], []

    last_inc_est_by_arm = None
    last_adv_est_by_arm = None
    last_ladv_est_by_arm = None

    for t in t_grid:
        vals = np.zeros((t, k_ord), dtype=np.float64)

        inc_sum = np.zeros((m, k_ord), dtype=np.float64)
        inc_cnt = np.zeros(m, dtype=np.int64)
        adv_sum = np.zeros((m, k_ord), dtype=np.float64)
        adv_cnt = np.zeros(m, dtype=np.int64)
        ladv_sum = np.zeros(m, dtype=np.float64)
        ladv_cnt = np.zeros(m, dtype=np.int64)

        for i in range(t):
            v_i, inc_i, adv_i, ladv_i, idx_i = _single_batch_estimates(
                os_batch,
                os_batch_l,
                idx_sampler,
                r=r,
                N=args.N,
                to_backend=to_backend,
                to_numpy=to_numpy,
            )
            vals[i] = v_i

            inc_mean_i, cnt_i = _arm_means_from_items(inc_i, idx_i, m)
            adv_mean_i, _ = _arm_means_from_items(adv_i, idx_i, m)
            ladv_mean_i, _ = _arm_means_from_items(ladv_i, idx_i, m)

            present = cnt_i > 0
            inc_sum[present] += inc_mean_i[present] * cnt_i[present, None]
            adv_sum[present] += adv_mean_i[present] * cnt_i[present, None]
            ladv_sum[present] += ladv_mean_i[present] * cnt_i[present]
            inc_cnt[present] += cnt_i[present]
            adv_cnt[present] += cnt_i[present]
            ladv_cnt[present] += cnt_i[present]

        v_mean = vals.mean(axis=0)

        inc_est_by_arm = np.full((m, k_ord), np.nan, dtype=np.float64)
        adv_est_by_arm = np.full((m, k_ord), np.nan, dtype=np.float64)
        ladv_est_by_arm = np.full(m, np.nan, dtype=np.float64)

        have_inc = inc_cnt > 0
        have_adv = adv_cnt > 0
        have_ladv = ladv_cnt > 0
        inc_est_by_arm[have_inc] = inc_sum[have_inc] / inc_cnt[have_inc, None]
        adv_est_by_arm[have_adv] = adv_sum[have_adv] / adv_cnt[have_adv, None]
        ladv_est_by_arm[have_ladv] = ladv_sum[have_ladv] / ladv_cnt[have_ladv]

        a_abs, a_rel = _mean_abs_and_rel_error(v_mean, v_exact)
        err_v_abs.append(a_abs)
        err_v_rel.append(a_rel)

        a_abs, a_rel = _mean_abs_and_rel_error(inc_est_by_arm, inc_exact_by_arm)
        err_inc_abs.append(a_abs)
        err_inc_rel.append(a_rel)

        a_abs, a_rel = _mean_abs_and_rel_error(adv_est_by_arm, adv_exact_by_arm)
        err_adv_abs.append(a_abs)
        err_adv_rel.append(a_rel)

        a_abs, a_rel = _mean_abs_and_rel_error(ladv_est_by_arm, l_adv_exact_by_arm)
        err_ladv_abs.append(a_abs)
        err_ladv_rel.append(a_rel)

        if t == t_grid[-1]:
            last_inc_est_by_arm = inc_est_by_arm
            last_adv_est_by_arm = adv_est_by_arm
            last_ladv_est_by_arm = ladv_est_by_arm

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))

    ax = axes[0]
    ax.plot(t_grid, err_v_abs, marker="o", label="order-stats")
    ax.plot(t_grid, err_inc_abs, marker="d", label="inclusion")
    ax.plot(t_grid, err_adv_abs, marker="s", label="advantage")
    ax.plot(t_grid, err_ladv_abs, marker="^", label="L-advantage")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of repeated batch estimates (t)")
    ax.set_ylabel("mean absolute error")
    ax.set_title("Absolute error vs known-(r,p) targets")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(t_grid, err_v_rel, marker="o", label="order-stats")
    ax.plot(t_grid, err_inc_rel, marker="d", label="inclusion")
    ax.plot(t_grid, err_adv_rel, marker="s", label="advantage")
    ax.plot(t_grid, err_ladv_rel, marker="^", label="L-advantage")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of repeated batch estimates (t)")
    ax.set_ylabel("mean relative error")
    ax.set_title("Relative error vs known-(r,p) targets")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(f"{backend_name} estimator convergence (N={args.N}, k={args.k}, floor(k)={k_ord}, arms={m})")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved error curves: {out}")

    if args.plot_arm_details and last_inc_est_by_arm is not None and last_adv_est_by_arm is not None and last_ladv_est_by_arm is not None:
        rank_idx = args.arm_rank - 1
        arms = np.arange(m)
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.8))

        axes2[0].plot(arms, inc_exact_by_arm[:, rank_idx], marker="o", label="exact")
        axes2[0].plot(arms, last_inc_est_by_arm[:, rank_idx], marker="x", label=f"estimate (t={t_grid[-1]})")
        axes2[0].set_title(f"Inclusion by arm (rank j={args.arm_rank})")
        axes2[0].set_xlabel("arm")
        axes2[0].grid(alpha=0.3)
        axes2[0].legend(fontsize=8)

        axes2[1].plot(arms, adv_exact_by_arm[:, rank_idx], marker="o", label="exact")
        axes2[1].plot(arms, last_adv_est_by_arm[:, rank_idx], marker="x", label=f"estimate (t={t_grid[-1]})")
        axes2[1].set_title(f"Advantage by arm (rank j={args.arm_rank})")
        axes2[1].set_xlabel("arm")
        axes2[1].grid(alpha=0.3)
        axes2[1].legend(fontsize=8)

        axes2[2].plot(arms, l_adv_exact_by_arm, marker="o", label="exact")
        axes2[2].plot(arms, last_ladv_est_by_arm, marker="x", label=f"estimate (t={t_grid[-1]})")
        axes2[2].set_title("L-advantage by arm")
        axes2[2].set_xlabel("arm")
        axes2[2].grid(alpha=0.3)
        axes2[2].legend(fontsize=8)

        details_out = out.with_name(out.stem + "_arms" + out.suffix)
        fig2.tight_layout()
        fig2.savefig(details_out, dpi=150)
        print(f"Saved arm details: {details_out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
