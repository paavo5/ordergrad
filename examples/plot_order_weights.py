#!/usr/bin/env python3
"""Plot unconditional order-statistic weight curves W[m,j] for a batch size N.

This script visualizes how each order-stat index j (1..k) weights the sorted positions m (1..N)
in E[X_(j:k)] = sum_m x_(m) W[m,j].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ordergrad.numpy_backend import precompute_W_unconditional


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot unconditional order-statistic weights by sorted rank.")
    parser.add_argument("--N", type=int, default=100, help="Batch/population size.")
    parser.add_argument("--k", type=float, default=20.0, help="Subset/sample parameter k (can be real).")
    parser.add_argument(
        "--ranks",
        type=str,
        default="1,5,10,15,20",
        help="Comma-separated list of j ranks to plot (1-based).",
    )
    parser.add_argument("--output", type=str, default="examples/artifacts/order_weights.png", help="Output PNG path.")
    parser.add_argument("--show", action="store_true", help="Show interactive window in addition to saving.")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.") from e

    k_ord = int(np.floor(args.k))
    if not (1 <= k_ord <= args.N):
        raise SystemExit("Require 1 <= floor(k) <= N.")

    W = precompute_W_unconditional(args.N, args.k, dtype=np.float64)

    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    bad = [j for j in ranks if not (1 <= j <= k_ord)]
    if bad:
        raise SystemExit(f"Invalid ranks {bad}; require each in [1, {k_ord}].")

    m = np.arange(1, args.N + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for j in ranks:
        ax.plot(m, W[:, j - 1], label=f"j={j}")

    ax.set_title(f"Order-statistic weights W[m,j] (N={args.N}, k={args.k}, floor(k)={k_ord})")
    ax.set_xlabel("sorted index m")
    ax.set_ylabel("weight W[m,j]")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
