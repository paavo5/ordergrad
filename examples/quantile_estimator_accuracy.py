#!/usr/bin/env python3
"""Monte Carlo accuracy comparison: Quantile:q vs HarrellDavis:q.

For each estimator and each repetition count t, the script averages t independent
batch estimates and compares that average to the exact population quantile.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist

import numpy as np

from ordergrad.numpy_backend import OrderStatTransform


def _parse_k_list(spec: str) -> list[float]:
    vals = [float(x) for x in spec.split(",") if x.strip()]
    if not vals:
        raise SystemExit("--k-list must contain at least one value.")
    if len(vals) == 1:
        return [vals[0], vals[0]]
    if len(vals) == 2:
        return vals
    raise SystemExit("--k-list must contain either one value (broadcast) or two values (Quantile,HarrellDavis).")


def _parse_t_grid(spec: str) -> list[int]:
    t_grid = [int(x) for x in spec.split(",") if x.strip()]
    if not t_grid:
        raise SystemExit("--t-grid must contain at least one value.")
    if any(t <= 0 for t in t_grid):
        raise SystemExit("All --t-grid values must be positive.")
    return t_grid


def _draw_batch(rng: np.random.Generator, n: int, dist: str) -> np.ndarray:
    if dist == "uniform":
        return rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    if dist == "gaussian":
        return rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float64)
    raise RuntimeError(f"Unsupported dist: {dist}")


def _exact_quantile(q: float, dist: str) -> float:
    if dist == "uniform":
        return float(q)
    if dist == "gaussian":
        return float(NormalDist(mu=0.0, sigma=1.0).inv_cdf(q))
    raise RuntimeError(f"Unsupported dist: {dist}")


def _safe_for_logplot(vals, eps: float = 1e-16):
    out = []
    for v in vals:
        fv = float(v)
        if not (fv > 0.0):
            fv = eps
        out.append(fv)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Quantile:q and HarrellDavis:q MC accuracy vs repetitions t.")
    ap.add_argument("--N", type=int, default=64, help="Batch size per estimator evaluation.")
    ap.add_argument("--k-list", type=str, default="6", help="Comma-separated k values for estimators in order Quantile,HarrellDavis (one value broadcasts).")
    ap.add_argument("--quantile", type=float, default=0.25, help="Target quantile q in [0,1].")
    ap.add_argument("--dist", type=str, default="uniform", choices=["uniform", "gaussian"], help="Sampling distribution for rewards.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-grid", type=str, default="1,2,5,10,20,50,100,200,500", help="Comma-separated repetition counts t.")
    ap.add_argument("--output", type=str, default="examples/artifacts/quantile_estimator_accuracy.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.") from e

    q = float(args.quantile)
    if not (0.0 <= q <= 1.0):
        raise SystemExit("--quantile must be in [0, 1].")

    k_quantile, k_hd = _parse_k_list(args.k_list)
    if int(np.floor(k_quantile)) < 1 or int(np.floor(k_hd)) < 1:
        raise SystemExit("Need floor(k) >= 1 for both estimators.")
    if k_quantile > args.N or k_hd > args.N:
        raise SystemExit("Require k <= N for both estimators.")

    t_grid = _parse_t_grid(args.t_grid)
    rng = np.random.default_rng(args.seed)

    os_quantile = OrderStatTransform.precompute(args.N, k_quantile, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    os_hd = OrderStatTransform.precompute(args.N, k_hd, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)

    spec_quantile = f"Quantile:{q}"
    spec_hd = f"HarrellDavis:{q}"

    exact = _exact_quantile(q, args.dist)

    q_abs_err, q_rel_err = [], []
    hd_abs_err, hd_rel_err = [], []

    for t in t_grid:
        q_vals = np.empty(t, dtype=np.float64)
        hd_vals = np.empty(t, dtype=np.float64)

        for i in range(t):
            x = _draw_batch(rng, args.N, args.dist)
            q_vals[i] = os_quantile.expected_lstat(x, spec_quantile)
            hd_vals[i] = os_hd.expected_lstat(x, spec_hd)

        q_mean = float(np.mean(q_vals))
        hd_mean = float(np.mean(hd_vals))

        q_abs = abs(q_mean - exact)
        hd_abs = abs(hd_mean - exact)

        denom = abs(exact) + 1e-12
        q_rel = q_abs / denom
        hd_rel = hd_abs / denom

        q_abs_err.append(q_abs)
        hd_abs_err.append(hd_abs)
        q_rel_err.append(q_rel)
        hd_rel_err.append(hd_rel)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))

    ax = axes[0]
    ax.plot(t_grid, _safe_for_logplot(q_abs_err), marker="o", label=f"Quantile:q (k={k_quantile:g})")
    ax.plot(t_grid, _safe_for_logplot(hd_abs_err), marker="s", label=f"HarrellDavis:q (k={k_hd:g})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of repeated batch estimates (t)")
    ax.set_ylabel("absolute error")
    ax.set_title("Absolute error vs exact quantile")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(t_grid, _safe_for_logplot(q_rel_err), marker="o", label=f"Quantile:q (k={k_quantile:g})")
    ax.plot(t_grid, _safe_for_logplot(hd_rel_err), marker="s", label=f"HarrellDavis:q (k={k_hd:g})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of repeated batch estimates (t)")
    ax.set_ylabel("relative error")
    ax.set_title("Relative error vs exact quantile")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Quantile estimator accuracy (dist={args.dist}, q={q}, N={args.N}, kQ={k_quantile:g}, kHD={k_hd:g})"
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
