#!/usr/bin/env python3
"""Monte Carlo estimator accuracy playground.

Each estimator run draws one batch of N i.i.d. samples from a known discrete
(r, p) distribution, then applies the batch estimator with parameter k.
The script averages across t independent runs and compares that average to the
exact known-(r,p) target, showing convergence as t grows.
"""

from __future__ import annotations

import argparse
import json
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



def _make_rewards(m: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "gaussian":
        return np.sort(rng.normal(loc=0.0, scale=1.0, size=m).astype(np.float64))
    if mode in {"linear", "arange"}:
        return np.arange(m, dtype=np.float64)
    if mode in {"exp", "pow2", "2^m"}:
        return np.power(2.0, np.arange(m, dtype=np.float64), dtype=np.float64)
    raise SystemExit("--reward-mode must be one of: gaussian, linear, exp")


def _make_probs(m: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "random":
        return rng.dirichlet(np.ones(m, dtype=np.float64)).astype(np.float64)
    if mode in {"uniform", "constant", "equal"}:
        return np.full(m, 1.0 / float(m), dtype=np.float64)
    raise SystemExit("--prob-mode must be one of: random, uniform")

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
    l_inc = to_numpy(os_l.expected_lstat_inclusion(x))
    l_adv = to_numpy(os_l.expected_lstat_advantage(x))
    return v, inc, adv, l_inc, l_adv, idx


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


def _parse_a(spec: str | None, k_ord: int):
    if spec is None:
        return np.linspace(0.3, 1.0, k_ord, dtype=np.float64)
    text = spec.strip()
    if any(ch.isalpha() for ch in text):
        return text
    vals = [float(x) for x in text.split(",") if x.strip()]
    if len(vals) == 0:
        raise SystemExit("--a was provided but no values were parsed.")
    if len(vals) == 1:
        vals = vals * k_ord
    elif len(vals) != k_ord:
        raise SystemExit(f"--a must have either 1 value, exactly floor(k)={k_ord} values, or a preset string.")
    # Numeric vectors are interpreted in top-rank order (j=1 highest),
    # so reverse to internal ascending order.
    return np.asarray(vals, dtype=np.float64)[::-1].copy()


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
    ap.add_argument("--reward-mode", type=str, default="gaussian", choices=["gaussian", "linear", "exp"], help="How arm rewards are generated: gaussian (fixed random), linear (arange), exp (2**m).")
    ap.add_argument("--prob-mode", type=str, default="random", choices=["random", "uniform"], help="How action sampling probabilities are set: random Dirichlet draw or uniform over actions.")
    ap.add_argument("--a", type=str, default=None, help="L-stat weights: single value (broadcast), comma-separated floor(k)-vector in top-rank order (j=1 highest), or preset string (e.g. TopM:3).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-buffer-size", type=int, default=200_000, help="Number of arm indices to pre-sample per buffer refill.")
    ap.add_argument(
        "--t-grid",
        type=str,
        default="1,2,5,10,20,50,100,200,500",
        help="Comma-separated repetition counts t (number of independent estimator batches to average).",
    )
    ap.add_argument("--arm-rank", type=int, default=1, help="1-based estimator rank j from top used in arm-detail plots (j=1 is highest order-stat).")
    ap.add_argument("--plot-arm-details", action="store_true", help="Also save arm-wise exact vs estimate plots at max(t-grid).")
    ap.add_argument("--output", type=str, default="examples/artifacts/mc_error_curve.png")
    ap.add_argument("--store-data", action="store_true", help="Store experiment arrays and metadata to disk.")
    ap.add_argument("--tag", type=str, default="default", help="Tag used in stored data filename/metadata.")
    ap.add_argument("--data-dir", type=str, default="examples/data", help="Directory where experiment data is stored.")
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
    r = _make_rewards(m, args.reward_mode, rng)
    p = _make_probs(m, args.prob_mode, rng)

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
    l_inc_exact_by_arm = to_numpy(os_exact_l.expected_lstat_inclusion_known_rp(r, p, a_backend))
    l_adv_exact_by_arm = to_numpy(os_exact_l.expected_lstat_advantage_known_rp(r, p, a_backend))

    t_grid = [int(x) for x in args.t_grid.split(",") if x.strip()]
    if any(t <= 0 for t in t_grid):
        raise SystemExit("All t-grid entries must be positive.")

    err_v_abs, err_v_rel = [], []
    err_inc_abs, err_inc_rel = [], []
    err_adv_abs, err_adv_rel = [], []
    err_linc_abs, err_linc_rel = [], []
    err_ladv_abs, err_ladv_rel = [], []

    last_inc_est_by_arm = None
    last_adv_est_by_arm = None
    last_linc_est_by_arm = None
    last_ladv_est_by_arm = None

    for t in t_grid:
        vals = np.zeros((t, k_ord), dtype=np.float64)

        inc_sum = np.zeros((m, k_ord), dtype=np.float64)
        inc_cnt = np.zeros(m, dtype=np.int64)
        adv_sum = np.zeros((m, k_ord), dtype=np.float64)
        adv_cnt = np.zeros(m, dtype=np.int64)
        linc_sum = np.zeros(m, dtype=np.float64)
        linc_cnt = np.zeros(m, dtype=np.int64)
        ladv_sum = np.zeros(m, dtype=np.float64)
        ladv_cnt = np.zeros(m, dtype=np.int64)

        for i in range(t):
            v_i, inc_i, adv_i, linc_i, ladv_i, idx_i = _single_batch_estimates(
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
            linc_mean_i, _ = _arm_means_from_items(linc_i, idx_i, m)
            ladv_mean_i, _ = _arm_means_from_items(ladv_i, idx_i, m)

            present = cnt_i > 0
            inc_sum[present] += inc_mean_i[present] * cnt_i[present, None]
            adv_sum[present] += adv_mean_i[present] * cnt_i[present, None]
            linc_sum[present] += linc_mean_i[present] * cnt_i[present]
            ladv_sum[present] += ladv_mean_i[present] * cnt_i[present]
            inc_cnt[present] += cnt_i[present]
            adv_cnt[present] += cnt_i[present]
            linc_cnt[present] += cnt_i[present]
            ladv_cnt[present] += cnt_i[present]

        v_mean = vals.mean(axis=0)

        inc_est_by_arm = np.full((m, k_ord), np.nan, dtype=np.float64)
        adv_est_by_arm = np.full((m, k_ord), np.nan, dtype=np.float64)
        linc_est_by_arm = np.full(m, np.nan, dtype=np.float64)
        ladv_est_by_arm = np.full(m, np.nan, dtype=np.float64)

        have_inc = inc_cnt > 0
        have_adv = adv_cnt > 0
        have_linc = linc_cnt > 0
        have_ladv = ladv_cnt > 0
        inc_est_by_arm[have_inc] = inc_sum[have_inc] / inc_cnt[have_inc, None]
        adv_est_by_arm[have_adv] = adv_sum[have_adv] / adv_cnt[have_adv, None]
        linc_est_by_arm[have_linc] = linc_sum[have_linc] / linc_cnt[have_linc]
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

        a_abs, a_rel = _mean_abs_and_rel_error(linc_est_by_arm, l_inc_exact_by_arm)
        err_linc_abs.append(a_abs)
        err_linc_rel.append(a_rel)

        a_abs, a_rel = _mean_abs_and_rel_error(ladv_est_by_arm, l_adv_exact_by_arm)
        err_ladv_abs.append(a_abs)
        err_ladv_rel.append(a_rel)

        if t == t_grid[-1]:
            last_inc_est_by_arm = inc_est_by_arm
            last_adv_est_by_arm = adv_est_by_arm
            last_linc_est_by_arm = linc_est_by_arm
            last_ladv_est_by_arm = ladv_est_by_arm

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))

    ax = axes[0]
    ax.plot(t_grid, err_v_abs, marker="o", label="order-stats")
    ax.plot(t_grid, err_inc_abs, marker="d", label="inclusion")
    ax.plot(t_grid, err_adv_abs, marker="s", label="advantage")
    ax.plot(t_grid, err_linc_abs, marker="v", label="L-inclusion")
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
    ax.plot(t_grid, err_linc_rel, marker="v", label="L-inclusion")
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

    if args.store_data:
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        stem = f"monte_carlo_accuracy__{args.tag}"
        npz_path = data_dir / f"{stem}.npz"
        json_path = data_dir / f"{stem}.json"
        np.savez(
            npz_path,
            t=np.asarray(t_grid, dtype=np.int64),
            err_v_abs=np.asarray(err_v_abs, dtype=np.float64),
            err_inc_abs=np.asarray(err_inc_abs, dtype=np.float64),
            err_adv_abs=np.asarray(err_adv_abs, dtype=np.float64),
            err_linc_abs=np.asarray(err_linc_abs, dtype=np.float64),
            err_ladv_abs=np.asarray(err_ladv_abs, dtype=np.float64),
            err_v_rel=np.asarray(err_v_rel, dtype=np.float64),
            err_inc_rel=np.asarray(err_inc_rel, dtype=np.float64),
            err_adv_rel=np.asarray(err_adv_rel, dtype=np.float64),
            err_linc_rel=np.asarray(err_linc_rel, dtype=np.float64),
            err_ladv_rel=np.asarray(err_ladv_rel, dtype=np.float64),
        )
        metadata = {
            "experiment": "monte_carlo_accuracy",
            "tag": args.tag,
            "setup": {
                "backend": args.backend,
                "N": int(args.N),
                "k": float(args.k),
                "num_arms": int(args.num_arms),
                "reward_mode": args.reward_mode,
                "prob_mode": args.prob_mode,
                "a": args.a,
                "seed": int(args.seed),
                "t_grid": t_grid,
            },
            "artifacts": {"plot_png": str(out), "data_npz": str(npz_path)},
        }
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved: {npz_path}")
        print(f"Saved: {json_path}")

    if args.plot_arm_details and last_inc_est_by_arm is not None and last_adv_est_by_arm is not None and last_linc_est_by_arm is not None and last_ladv_est_by_arm is not None:
        rank_idx = k_ord - args.arm_rank
        arms = np.arange(m)
        fig2, axes2 = plt.subplots(1, 4, figsize=(19, 4.8))

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

        axes2[2].plot(arms, l_inc_exact_by_arm, marker="o", label="exact")
        axes2[2].plot(arms, last_linc_est_by_arm, marker="x", label=f"estimate (t={t_grid[-1]})")
        axes2[2].set_title("L-inclusion by arm")
        axes2[2].set_xlabel("arm")
        axes2[2].grid(alpha=0.3)
        axes2[2].legend(fontsize=8)

        axes2[3].plot(arms, l_adv_exact_by_arm, marker="o", label="exact")
        axes2[3].plot(arms, last_ladv_est_by_arm, marker="x", label=f"estimate (t={t_grid[-1]})")
        axes2[3].set_title("L-advantage by arm")
        axes2[3].set_xlabel("arm")
        axes2[3].grid(alpha=0.3)
        axes2[3].legend(fontsize=8)

        details_out = out.with_name(out.stem + "_arms" + out.suffix)
        fig2.tight_layout()
        fig2.savefig(details_out, dpi=150)
        print(f"Saved arm details: {details_out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
