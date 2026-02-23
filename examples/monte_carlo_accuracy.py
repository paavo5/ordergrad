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

import numpy as np

from ordergrad.numpy_backend import OrderStatTransform


def _single_batch_estimates(
    os: OrderStatTransform,
    rng: np.random.Generator,
    *,
    r: np.ndarray,
    p: np.ndarray,
    a: np.ndarray,
    N: int,
):
    idx = rng.choice(len(r), size=N, replace=True, p=p)
    x = r[idx]
    v = os.expected_orderstats(x)
    adv = os.expected_orderstats_advantage(x)
    l_adv = os.expected_lstat_advantage(x, a)
    return v, adv, l_adv


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot estimator error vs number of repeated batch estimates (t).")
    ap.add_argument("--N", type=int, default=64, help="Batch size per estimator evaluation.")
    ap.add_argument("--k", type=float, default=6.0, help="Estimator k parameter (can be real).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--t-grid",
        type=str,
        default="1,2,5,10,20,50,100,200,500",
        help="Comma-separated repetition counts t (number of independent estimator batches to average).",
    )
    ap.add_argument("--output", type=str, default="examples/artifacts/mc_error_curve.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.") from e

    rng = np.random.default_rng(args.seed)
    r = np.array([-1.5, -0.4, 0.2, 0.9, 1.6, 2.4], dtype=np.float64)
    p = np.array([0.10, 0.18, 0.25, 0.20, 0.17, 0.10], dtype=np.float64)
    p = p / p.sum()

    k_ord = int(np.floor(args.k))
    if k_ord < 1:
        raise SystemExit("Need floor(k) >= 1")
    a = np.linspace(0.3, 1.0, k_ord, dtype=np.float64)

    # Batch estimator (unknown-distribution regime).
    os_batch = OrderStatTransform.precompute(
        args.N,
        args.k,
        dtype=np.float64,
        compute_conditional=True,
        compute_leave_one_out=True,
    )

    # Exact known-(r,p) target (known-distribution regime).
    os_exact = OrderStatTransform.precompute(max(args.N, 2), args.k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    v_exact = os_exact.expected_orderstats_known_rp(r, p)
    adv_exact = os_exact.expected_orderstats_advantage_known_rp(r, p)
    l_adv_exact = os_exact.expected_lstat_advantage_known_rp(r, p, a)

    t_grid = [int(x) for x in args.t_grid.split(",") if x.strip()]
    if any(t <= 0 for t in t_grid):
        raise SystemExit("All t-grid entries must be positive.")

    err_v = []
    err_adv = []
    err_ladv = []

    for t in t_grid:
        vals = np.zeros((t, k_ord), dtype=np.float64)
        advs = np.zeros((t, args.N, k_ord), dtype=np.float64)
        ladvs = np.zeros((t, args.N), dtype=np.float64)
        for i in range(t):
            v_i, adv_i, ladv_i = _single_batch_estimates(os_batch, rng, r=r, p=p, a=a, N=args.N)
            vals[i] = v_i
            advs[i] = adv_i
            ladvs[i] = ladv_i

        v_mean = vals.mean(axis=0)
        adv_mean = advs.mean(axis=0)
        ladv_mean = ladvs.mean(axis=0)

        err_v.append(float(np.mean(np.abs(v_mean - v_exact))))
        err_adv.append(float(np.mean(np.abs(adv_mean - adv_exact))))
        err_ladv.append(float(np.mean(np.abs(ladv_mean - l_adv_exact))))

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.plot(t_grid, err_v, marker="o", label="order-stats mean abs error")
    ax.plot(t_grid, err_adv, marker="s", label="advantage mean abs error")
    ax.plot(t_grid, err_ladv, marker="^", label="L-advantage mean abs error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of repeated batch estimates (t)")
    ax.set_ylabel("mean absolute error vs known-(r,p) exact target")
    ax.set_title(f"Estimator convergence with repetitions t (N={args.N}, k={args.k}, floor(k)={k_ord})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
