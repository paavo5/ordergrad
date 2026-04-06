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

from examples.logging_utils import infer_data_dir_from_output, save_metadata_json

from ordergrad.numpy_backend import (
    OrderStatTransform,
    precompute_ABC_conditional_including_rank,
    precompute_W_leave_one_out,
    precompute_W_unconditional,
)


def _parse_k_list(spec: str) -> list[float]:
    vals = [tok.strip() for tok in str(spec).split(",") if tok.strip()]
    if not vals:
        raise SystemExit("No --k values were provided.")
    out = [float(v) for v in vals]
    if any(v <= 0.0 for v in out):
        raise SystemExit("All --k values must be > 0.")
    return out


def _parse_ranks(spec: str, *, k_ord: int) -> list[int]:
    """Parse rank spec with comma tokens and inclusive a..b ranges.

    Special case: ``0`` means "plot no individual rank curves".
    """
    if spec.strip() == "0":
        return []

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

    return list(dict.fromkeys(out))


def _parse_single_a(spec: str, *, ranks: list[int], k_ord: int) -> np.ndarray:
    text = spec.strip()
    if not text:
        raise SystemExit("--a contains an empty entry.")

    if any(ch.isalpha() for ch in text):
        try:
            return OrderStatTransform._preset_lstat_weights(k_ord, text, dtype=np.float64)
        except Exception as e:
            raise SystemExit(f"Invalid preset for --a entry '{text}': {e}") from e

    vals = [float(x) for x in text.split(",") if x.strip()]
    if len(vals) == 0:
        raise SystemExit("--a was provided but no weights were parsed.")

    if len(ranks) == 0:
        if len(vals) != k_ord:
            raise SystemExit(
                f"When --ranks=0, numeric --a must provide exactly floor(k)={k_ord} values or use a preset string."
            )
        return np.asarray(vals, dtype=np.float64)[::-1].copy()

    if len(vals) == 1:
        vals = vals * len(ranks)
    elif len(vals) != len(ranks):
        raise SystemExit("--a must have either one entry or the same number of entries as --ranks.")

    a = np.zeros(k_ord, dtype=np.float64)
    for j, w in zip(ranks, vals):
        a[k_ord - j] = w
    return a


def _parse_a_specs(spec: str) -> list[str]:
    text = spec.strip()
    if not text:
        raise SystemExit("--a was provided but no definitions were parsed.")
    if any(ch.isalpha() for ch in text):
        entries = [tok.strip() for tok in text.split(",") if tok.strip()]
        if not entries:
            raise SystemExit("--a was provided but no definitions were parsed.")
        return entries
    return [text]


def _broadcast_pair_lists(k_list: list[float], a_specs: list[str]) -> list[tuple[float, str]]:
    if len(k_list) == len(a_specs):
        return list(zip(k_list, a_specs))
    if len(k_list) == 1:
        return [(k_list[0], a_spec) for a_spec in a_specs]
    if len(a_specs) == 1:
        return [(k, a_specs[0]) for k in k_list]
    raise SystemExit(
        "When providing multiple --k and --a entries, either lengths must match, or one side must have length 1 (broadcast)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot order-statistic weights by sorted rank.")
    parser.add_argument("--N", type=int, default=100, help="Batch/population size.")
    parser.add_argument(
        "--k",
        type=str,
        default="20",
        help="Subset/sample parameter k (real). Can be a comma-separated list, e.g. 10,20,30.",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default="1,5,10,15,20",
        help="Comma-separated ranks/ranges (1-based from top, so j=1 is highest), e.g. 1,3..6,10. Use 0 to disable individual-rank curves.",
    )
    parser.add_argument(
        "--a",
        type=str,
        default=None,
        help=(
            "Optional L-stat definitions. Provide one numeric definition (broadcast/per-rank), one preset, or multiple presets"
            " as a comma-separated list (e.g. TopM:3,BotM:3,Median). When combined with multi-k, lengths must match or one side"
            " must be length 1 (broadcast)."
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
    parser.add_argument("--show-delta", action="store_true", help="In conditional mode, also plot W_cond - W.")
    parser.add_argument(
        "--show-leave-one-out",
        action="store_true",
        help="In conditional mode, also plot leave-one-out weights for excluding the same conditioned rank.",
    )
    parser.add_argument("--output", type=str, default="examples/artifacts/order_weights.png", help="Output PNG path.")
    parser.add_argument("--store-data", action=argparse.BooleanOptionalAction, default=True, help="Store run metadata JSON (default: on).")
    parser.add_argument("--data-dir", type=str, default=None, help="Metadata directory. If omitted, inferred from output path.")
    parser.add_argument("--tag", type=str, default=None, help="Metadata tag (defaults to output stem).")
    parser.add_argument("--show", action="store_true", help="Show interactive window in addition to saving.")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.") from e

    k_values = _parse_k_list(args.k)
    k_ord_values = [int(np.floor(k)) for k in k_values]
    if any((k_ord < 1 or k_ord > args.N) for k_ord in k_ord_values):
        raise SystemExit("Require 1 <= floor(k) <= N for all provided --k values.")

    k_ord_min = min(k_ord_values)
    ranks = _parse_ranks(args.ranks, k_ord=k_ord_min)

    # Precompute matrices for each unique k to avoid repeated setup.
    W_cache: dict[float, np.ndarray] = {}
    for k in sorted(set(k_values)):
        W_cache[k] = precompute_W_unconditional(args.N, k, dtype=np.float64)

    a_pairs: list[tuple[float, str, np.ndarray]] = []
    if args.a is not None:
        a_specs = _parse_a_specs(args.a)
        for k, a_spec in _broadcast_pair_lists(k_values, a_specs):
            k_ord = int(np.floor(k))
            a_vec = _parse_single_a(a_spec, ranks=ranks, k_ord=k_ord)
            a_pairs.append((k, a_spec, a_vec))

    m = np.arange(1, args.N + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    if args.mode == "unconditional":
        if ranks:
            for k in sorted(set(k_values)):
                W = W_cache[k]
                k_ord = int(np.floor(k))
                for j in ranks:
                    if j > k_ord:
                        raise SystemExit(f"Rank j={j} is invalid for k={k} (floor(k)={k_ord}).")
                    if len(set(k_values)) == 1:
                        label = f"j={j}"
                    else:
                        label = f"k={k:g}, j={j}"
                    col = k_ord - j
                    ax.plot(m, W[:, col], label=label)

        title = f"Unconditional order-stat weights (N={args.N})"
        if len(set(k_values)) == 1:
            k0 = k_values[0]
            title = f"Unconditional order-stat weights W[m,j] (N={args.N}, k={k0:g}, floor(k)={int(np.floor(k0))})"

        for k, spec_label, a_vec in a_pairs:
            W = W_cache[k]
            ax.plot(m, W @ a_vec, linestyle="--", linewidth=1.8, label=f"combined W @ a (k={k:g}, a={spec_label})")

    else:
        if len(set(k_values)) != 1:
            raise SystemExit("Conditional mode currently supports a single --k value.")

        k = k_values[0]
        k_ord = int(np.floor(k))
        r = int(args.conditioned_rank)
        if not (1 <= r <= args.N):
            raise SystemExit(f"--conditioned-rank must be in [1, {args.N}].")

        W = W_cache[k]
        A, B, C = precompute_ABC_conditional_including_rank(args.N, k, dtype=np.float64)
        W_cond = np.empty_like(W)
        W_cond[: r - 1, :] = A[: r - 1, :]
        W_cond[r - 1, :] = B[r - 1, :]
        W_cond[r:, :] = C[r:, :]

        W_loo = None
        if args.show_leave_one_out:
            if k > args.N - 1:
                raise SystemExit("--show-leave-one-out requires k <= N-1.")
            Wm = precompute_W_leave_one_out(args.N, k, dtype=np.float64)
            W_loo = np.zeros_like(W)
            W_loo[: r - 1, :] = Wm[: r - 1, :]
            W_loo[r:, :] = Wm[r - 1 :, :]

        title = (
            "Conditional inclusion weights "
            f"W_cond[r,m,j] with conditioned sorted rank r={r} (N={args.N}, k={k:g}, floor(k)={k_ord})"
        )
        for j in ranks:
            if j > k_ord:
                raise SystemExit(f"Rank j={j} is invalid for k={k} (floor(k)={k_ord}).")
            col = k_ord - j
            ax.plot(m, W_cond[:, col], label=f"cond j={j}")
            ax.plot(m, W[:, col], linestyle=":", alpha=0.8, label=f"uncond j={j}")
            if W_loo is not None:
                ax.plot(m, W_loo[:, col], linestyle="-.", alpha=0.9, label=f"loo-excl j={j}")
            if args.show_delta:
                ax.plot(m, W_cond[:, col] - W[:, col], linestyle="--", alpha=0.9, label=f"delta j={j}")

        for _, spec_label, a_vec in a_pairs:
            w_rank_cond = W_cond @ a_vec
            w_rank_uncond = W @ a_vec
            ax.plot(m, w_rank_cond, linestyle="-", linewidth=1.8, label=f"combined conditional (a={spec_label})")
            ax.plot(m, w_rank_uncond, linestyle=":", linewidth=1.5, label=f"combined unconditional (a={spec_label})")
            if W_loo is not None:
                w_rank_loo = W_loo @ a_vec
                ax.plot(m, w_rank_loo, linestyle="-.", linewidth=1.5, label=f"combined loo-excl (a={spec_label})")
            if args.show_delta:
                ax.plot(m, w_rank_cond - w_rank_uncond, linestyle="--", linewidth=1.3, label=f"combined delta (a={spec_label})")

    ax.set_title(title)
    ax.set_xlabel("sorted index m")
    ax.set_ylabel("weight")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="upper left")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    pdf_out = out.with_suffix(".pdf")
    fig.savefig(pdf_out)
    print(f"Saved: {out}")
    print(f"Saved: {pdf_out}")

    if args.store_data:
        data_dir = Path(args.data_dir) if args.data_dir else infer_data_dir_from_output(out)
        tag = str(args.tag) if args.tag else out.stem
        save_metadata_json(
            experiment="plot_order_weights",
            tag=tag,
            setup={
                "N": int(args.N),
                "k": [float(x) for x in k_values],
                "k_ord": [int(x) for x in k_ord_values],
                "ranks": [int(r) for r in ranks],
                "a": args.a,
                "mode": args.mode,
                "conditioned_rank": int(args.conditioned_rank),
                "show_delta": bool(args.show_delta),
                "show_leave_one_out": bool(args.show_leave_one_out),
            },
            artifacts={
                "plot_png": str(out),
                "plot_pdf": str(pdf_out),
            },
            data_dir=data_dir,
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
