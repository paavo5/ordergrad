#!/usr/bin/env python3
"""Plot order-statistic weight curves for different modes.

Modes:
- unconditional: plots W[m,j]
- conditional: plots W_cond[r,m,j] for a fixed conditioned sorted rank r

For conditional mode, optional delta overlays show how conditioning changes each
curve relative to unconditional weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ordergrad.numpy_backend import (
    OrderStatTransform,
    precompute_ABC_conditional_including_rank,
    precompute_W_leave_one_out,
    precompute_W_unconditional,
)


def _parse_ranks(spec: str, *, k_ord: int) -> list[int]:
    """Parse rank spec with comma tokens and inclusive a..b ranges."""
    out: list[int] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if ".." in token:
            parts = token.split("..")
            if len(parts) != 2 or (not parts[0].strip()) or (not parts[1].strip()):
                raise SystemExit(f"Invalid range token '{token}'. Use a..b.")
            a = int(parts[0])
            b = int(parts[1])
            if a > b:
                raise SystemExit(f"Invalid descending range '{token}'. Use a <= b.")
            out.extend(range(a, b + 1))
        else:
            out.append(int(token))

    if not out:
        raise SystemExit("No ranks were provided.")

    bad = [j for j in out if not (1 <= j <= k_ord)]
    if bad:
        raise SystemExit(f"Invalid ranks {bad}; require each in [1, {k_ord}].")

    # De-duplicate while preserving order.
    dedup = list(dict.fromkeys(out))
    return dedup


def _parse_a(spec: str, *, ranks: list[int], k_ord: int) -> np.ndarray:
    text = spec.strip()
    if any(ch.isalpha() for ch in text):
        try:
            return OrderStatTransform._preset_lstat_weights(k_ord, text, dtype=np.float64)
        except Exception as e:
            raise SystemExit(f"Invalid preset for --a: {e}") from e

    vals = [float(x) for x in text.split(",") if x.strip()]
    if len(vals) == 0:
        raise SystemExit("--a was provided but no weights were parsed.")
    if len(vals) == 1:
        vals = vals * len(ranks)
    elif len(vals) != len(ranks):
        raise SystemExit("--a must have either one entry or the same number of entries as --ranks.")

    a = np.zeros(k_ord, dtype=np.float64)
    for j, w in zip(ranks, vals):
        a[j - 1] = w
    return a


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot order-statistic weights by sorted rank.")
    parser.add_argument("--N", type=int, default=100, help="Batch/population size.")
    parser.add_argument("--k", type=float, default=20.0, help="Subset/sample parameter k (can be real).")
    parser.add_argument(
        "--ranks",
        type=str,
        default="1,5,10,15,20",
        help="Comma-separated ranks and/or ranges (1-based), e.g. 1,3..6,10.",
    )
    parser.add_argument(
        "--a",
        type=str,
        default=None,
        help=(
            "Optional L-stat weights for listed ranks. "
            "Provide either one value (broadcast to all listed ranks), one value per listed rank, or a preset string (e.g. TopM:3, Median)."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="unconditional",
        choices=["unconditional", "conditional"],
        help="Weight mode to plot.",
    )
    parser.add_argument(
        "--conditioned-rank",
        type=int,
        default=1,
        help="For conditional mode: conditioned sorted rank r (1-based) of the included item.",
    )
    parser.add_argument(
        "--show-delta",
        action="store_true",
        help="In conditional mode, also plot the change vs unconditional: W_cond - W.",
    )
    parser.add_argument(
        "--show-leave-one-out",
        action="store_true",
        help=(
            "In conditional mode, also plot leave-one-out weights for excluding "
            "the same conditioned rank."
        ),
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

    ranks = _parse_ranks(args.ranks, k_ord=k_ord)
    W = precompute_W_unconditional(args.N, args.k, dtype=np.float64)

    a = _parse_a(args.a, ranks=ranks, k_ord=k_ord) if args.a is not None else None

    m = np.arange(1, args.N + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    if args.mode == "unconditional":
        W_plot = W
        title = f"Unconditional order-stat weights W[m,j] (N={args.N}, k={args.k}, floor(k)={k_ord})"
        for j in ranks:
            ax.plot(m, W_plot[:, j - 1], label=f"j={j}")
    else:
        r = int(args.conditioned_rank)
        if not (1 <= r <= args.N):
            raise SystemExit(f"--conditioned-rank must be in [1, {args.N}].")
        A, B, C = precompute_ABC_conditional_including_rank(args.N, args.k, dtype=np.float64)
        W_cond = np.empty_like(W)
        W_cond[: r - 1, :] = A[: r - 1, :]
        W_cond[r - 1, :] = B[r - 1, :]
        W_cond[r:, :] = C[r:, :]

        W_loo = None
        if args.show_leave_one_out:
            if args.k > args.N - 1:
                raise SystemExit("--show-leave-one-out requires k <= N-1.")
            Wm = precompute_W_leave_one_out(args.N, args.k, dtype=np.float64)
            W_loo = np.zeros_like(W)
            W_loo[: r - 1, :] = Wm[: r - 1, :]
            W_loo[r:, :] = Wm[r - 1 :, :]

        title = (
            "Conditional inclusion weights "
            f"W_cond[r,m,j] with conditioned sorted rank r={r} (N={args.N}, k={args.k}, floor(k)={k_ord})"
        )
        for j in ranks:
            ax.plot(m, W_cond[:, j - 1], label=f"cond j={j}")
            ax.plot(m, W[:, j - 1], linestyle=":", alpha=0.8, label=f"uncond j={j}")
            if W_loo is not None:
                ax.plot(m, W_loo[:, j - 1], linestyle="-.", alpha=0.9, label=f"loo-excl j={j}")
            if args.show_delta:
                ax.plot(m, W_cond[:, j - 1] - W[:, j - 1], linestyle="--", alpha=0.9, label=f"delta j={j}")

        if a is not None:
            w_rank_cond = W_cond @ a
            w_rank_uncond = W @ a
            ax.plot(
                m,
                w_rank_cond,
                color="black",
                linestyle="-",
                linewidth=1.8,
                label="combined conditional W_cond @ a",
            )
            ax.plot(
                m,
                w_rank_uncond,
                color="black",
                linestyle=":",
                linewidth=1.5,
                label="combined unconditional W @ a",
            )
            if W_loo is not None:
                w_rank_loo = W_loo @ a
                ax.plot(
                    m,
                    w_rank_loo,
                    color="black",
                    linestyle="-.",
                    linewidth=1.5,
                    label="combined leave-one-out excl @ a",
                )
            if args.show_delta:
                ax.plot(
                    m,
                    w_rank_cond - w_rank_uncond,
                    color="gray",
                    linestyle="--",
                    linewidth=1.3,
                    label="combined delta",
                )

    ax.set_title(title)
    ax.set_xlabel("sorted index m")
    ax.set_ylabel("weight")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="upper left")

    if a is not None and args.mode == "unconditional":
        w_rank = W @ a
        ax.plot(m, w_rank, color="black", linestyle="--", linewidth=1.8, label="combined W @ a")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
