#!/usr/bin/env python3
"""Monte Carlo accuracy comparison for quantile-style L-stat estimators.

For each estimator and each repetition count t, the script averages t independent
batch estimates and compares that average to the exact population quantile.

Quantile notation here follows the standard CDF convention: q is the
fraction of mass below the quantile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from ordergrad.numpy_backend import OrderStatTransform


def _parse_csv(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def _parse_methods(spec: str) -> list[str]:
    methods = _parse_csv(spec)
    if not methods:
        raise SystemExit("--a must contain at least one method.")
    allowed = {
        "quantile", "quantileweibull", "quantilehazen", "quantileblom",
        "topquantile", "topquantileweibull", "topquantilehazen", "topquantileblom",
        "harrelldavis", "harreldavis",
    }
    for m in methods:
        if m.lower() not in allowed:
            raise SystemExit(
                f"Unsupported method '{m}' in --a. Allowed: Quantile, QuantileWeibull, QuantileHazen, QuantileBlom, "
                "TopQuantile, TopQuantileWeibull, TopQuantileHazen, TopQuantileBlom, HarrellDavis."
            )
    return methods


def _parse_k_list(spec: str, n_methods: int) -> list[float]:
    vals = [float(x) for x in spec.split(",") if x.strip()]
    if not vals:
        raise SystemExit("--k-list must contain at least one value.")
    if len(vals) == 1:
        return vals * n_methods
    if len(vals) == n_methods:
        return vals
    raise SystemExit(f"--k-list must contain either one value (broadcast) or exactly len(--a)={n_methods} values.")


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
    ap = argparse.ArgumentParser(description="Compare quantile-style estimators (Quantile*/HarrellDavis) MC accuracy vs repetitions t.")
    ap.add_argument("--N", type=int, default=64, help="Batch size per estimator evaluation.")
    ap.add_argument("--a", type=str, default="Quantile,HarrellDavis", help="Comma-separated estimator methods. Allowed: Quantile, QuantileWeibull, QuantileHazen, QuantileBlom, TopQuantile, TopQuantileWeibull, TopQuantileHazen, TopQuantileBlom, HarrellDavis.")
    ap.add_argument("--k-list", type=str, default="6", help="Comma-separated k values aligned with --a (one value broadcasts).")
    ap.add_argument("--quantile", type=float, default=0.25, help="Target quantile q in [0,1] (q mass below the threshold).")
    ap.add_argument("--dist", type=str, default="uniform", choices=["uniform", "gaussian"], help="Sampling distribution for rewards.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-grid", type=str, default="1,2,5,10,20,50,100,200,500", help="Comma-separated repetition counts t.")
    ap.add_argument("--output", type=str, default="examples/artifacts/quantile_estimator_accuracy.png")
    ap.add_argument("--store-data", action="store_true", help="Store experiment arrays and metadata to disk.")
    ap.add_argument("--tag", type=str, default="default", help="Tag used in stored data filename/metadata.")
    ap.add_argument("--data-dir", type=str, default="examples/data", help="Directory where experiment data is stored.")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.") from e

    q = float(args.quantile)
    if not (0.0 <= q <= 1.0):
        raise SystemExit("--quantile must be in [0, 1].")

    methods = _parse_methods(args.a)
    k_list = _parse_k_list(args.k_list, len(methods))
    if any(int(np.floor(kv)) < 1 for kv in k_list):
        raise SystemExit("Need floor(k) >= 1 for all estimators.")
    if any(kv > args.N for kv in k_list):
        raise SystemExit("Require k <= N for all estimators.")

    t_grid = _parse_t_grid(args.t_grid)
    rng = np.random.default_rng(args.seed)

    estimators = []
    for method, k_val in zip(methods, k_list):
        spec = f"{method}:{q}"
        os = OrderStatTransform.precompute(args.N, k_val, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
        estimators.append((method, float(k_val), spec, os))

    exact = _exact_quantile(q, args.dist)

    print(
        f"Quantile convention: q is mass-below (q={q}), exact target={exact:.8g}, dist={args.dist}, "
        f"methods={[m for m, _, _, _ in estimators]}, k_list={[k for _, k, _, _ in estimators]}"
    )

    header_fmt = "{:<22} {:>10} {:>8} {:>14} {:>12} {:>12} {:>14} {:>14} {:>14}"
    row_fmt = "{:<22} {:>10.6g} {:>8d} {:>14.8g} {:>12.3e} {:>12.3e} {:>14.3e} {:>14.3e} {:>14.8g}"
    print(header_fmt.format("method", "k", "t_max", "mean", "abs_err", "rel_err", "rmse_single", "rmse_mean_t", "exact"))

    err_abs = {method: [] for method, _, _, _ in estimators}
    err_rel = {method: [] for method, _, _, _ in estimators}
    t_max = max(t_grid)
    final_stats = {}

    for t in t_grid:
        vals_by_method = {method: np.empty(t, dtype=np.float64) for method, _, _, _ in estimators}

        for i in range(t):
            x = _draw_batch(rng, args.N, args.dist)
            for method, _, spec, os in estimators:
                vals_by_method[method][i] = os.expected_lstat(x, spec)

        denom = abs(exact) + 1e-12
        for method, k_val, _, _ in estimators:
            vals = vals_by_method[method]
            mean_v = float(np.mean(vals))
            abs_e = abs(mean_v - exact)
            rel_e = abs_e / denom
            var_v = float(np.var(vals, ddof=1)) if t > 1 else 0.0
            rmse_single = float(np.sqrt(np.mean((vals - exact) ** 2)))
            rmse_mean_t = float(np.sqrt(var_v / float(t) + (mean_v - exact) ** 2))
            err_abs[method].append(abs_e)
            err_rel[method].append(rel_e)
            if t == t_max:
                final_stats[method] = {
                    "k": float(k_val),
                    "t_max": int(t),
                    "mean": mean_v,
                    "abs_err": abs_e,
                    "rel_err": rel_e,
                    "rmse_single": rmse_single,
                    "rmse_mean_t": rmse_mean_t,
                }

    for method, _, _, _ in estimators:
        st = final_stats[method]
        print(
            row_fmt.format(
                method,
                st["k"],
                st["t_max"],
                st["mean"],
                st["abs_err"],
                st["rel_err"],
                st["rmse_single"],
                st["rmse_mean_t"],
                exact,
            )
        )


    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    markers = ["o", "s", "^", "d", "x", "+", "v", "<", ">"]

    ax = axes[0]
    for j, (method, k_val, _, _) in enumerate(estimators):
        ax.plot(t_grid, _safe_for_logplot(err_abs[method]), marker=markers[j % len(markers)], label=f"{method}:q (k={k_val:g})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of repeated batch estimates (t)")
    ax.set_ylabel("absolute error")
    ax.set_title("Absolute error vs exact quantile")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for j, (method, k_val, _, _) in enumerate(estimators):
        ax.plot(t_grid, _safe_for_logplot(err_rel[method]), marker=markers[j % len(markers)], label=f"{method}:q (k={k_val:g})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of repeated batch estimates (t)")
    ax.set_ylabel("relative error")
    ax.set_title("Relative error vs exact quantile")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(f"Quantile estimator accuracy (dist={args.dist}, q={q}, N={args.N}, methods={','.join(methods)})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    pdf_out = out.with_suffix(".pdf")
    fig.savefig(pdf_out)
    print(f"Saved: {out}")
    print(f"Saved: {pdf_out}")

    if args.store_data:
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        stem = f"quantile_estimator_accuracy__{args.tag}"
        npz_path = data_dir / f"{stem}.npz"
        json_path = data_dir / f"{stem}.json"
        np.savez(
            npz_path,
            t=np.asarray(t_grid, dtype=np.int64),
            methods=np.asarray([m for m, _, _, _ in estimators]),
            k_list=np.asarray([k for _, k, _, _ in estimators], dtype=np.float64),
            abs_err=np.asarray([err_abs[m] for m, _, _, _ in estimators], dtype=np.float64),
            rel_err=np.asarray([err_rel[m] for m, _, _, _ in estimators], dtype=np.float64),
        )
        metadata = {
            "experiment": "quantile_estimator_accuracy",
            "tag": args.tag,
            "setup": {
                "N": int(args.N),
                "methods": [m for m, _, _, _ in estimators],
                "k_list": [k for _, k, _, _ in estimators],
                "quantile": float(q),
                "dist": args.dist,
                "seed": int(args.seed),
                "t_grid": t_grid,
            },
            "artifacts": {
                "plot_png": str(out),
                "plot_pdf": str(pdf_out),
                "data_npz": str(npz_path),
            },
        }
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved: {npz_path}")
        print(f"Saved: {json_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
