#!/usr/bin/env python3
"""Monte Carlo accuracy playground.

Compares Monte Carlo estimates against exact known-(r,p) values and plots
absolute error vs number of MC samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ordergrad.numpy_backend import OrderStatTransform


def mc_unconditional(r: np.ndarray, p: np.ndarray, k_ord: int, T: int, rng: np.random.Generator) -> np.ndarray:
    keys = rng.choice(len(r), size=(T, k_ord), replace=True, p=p)
    vals = np.sort(r[keys], axis=1)
    return vals.mean(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot MC estimation error against sample count.")
    ap.add_argument("--k", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=20, help="Independent repeats per T for error bars.")
    ap.add_argument("--t-grid", type=str, default="100,300,1000,3000,10000", help="Comma-separated MC sample counts.")
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
    if not (1 <= k_ord):
        raise SystemExit("Need floor(k) >= 1")

    os = OrderStatTransform.precompute(64, args.k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    exact = os.expected_orderstats_known_rp(r, p)

    T_grid = [int(x) for x in args.t_grid.split(",") if x.strip()]
    mean_err = []
    std_err = []

    for T in T_grid:
        errs = []
        for _ in range(args.trials):
            est = mc_unconditional(r, p, k_ord, T, rng)
            errs.append(float(np.mean(np.abs(est - exact))))
        errs = np.asarray(errs)
        mean_err.append(float(errs.mean()))
        std_err.append(float(errs.std(ddof=1) if errs.size > 1 else 0.0))

    mean_err = np.asarray(mean_err)
    std_err = np.asarray(std_err)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(T_grid, mean_err, yerr=std_err, marker="o", capsize=3, label="MC mean abs error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of MC samples (T)")
    ax.set_ylabel("mean absolute error")
    ax.set_title(f"MC error vs samples (known (r,p), k={args.k}, floor(k)={k_ord})")
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
