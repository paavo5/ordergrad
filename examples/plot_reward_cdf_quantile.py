#!/usr/bin/env python3
"""Compare true reward CDF with quantile curves inferred from full order-statistics.

For each estimator method, this script computes all expected order statistics for a
single k, maps ranks r=1..k to estimator-specific plotting positions p_r, and then
builds a linear interpolation in (x, p). This yields an estimated CDF curve that can
be compared directly to the true CDF of the chosen reward distribution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from ordergrad.numpy_backend import OrderStatTransform


def _parse_estimators(spec: str) -> list[str]:
    out = [s.strip() for s in spec.split(",") if s.strip()]
    if not out:
        raise SystemExit("--estimator must contain at least one estimator.")
    allowed = {"Quantile", "QuantileHazen", "QuantileWeibull", "QuantileBlom"}
    bad = [m for m in out if m not in allowed]
    if bad:
        raise SystemExit(f"Only {sorted(allowed)} are supported in this script. Invalid: {bad}")
    return out


def _parse_float_list(spec: str, name: str) -> list[float]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise SystemExit(f"--{name} must contain at least one comma-separated float.")
    return vals


def _validate_mixture(centers: list[float], scales: list[float], weights: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (len(centers) == len(scales) == len(weights)):
        raise SystemExit(
            "Gaussian mixture parameters must have the same length: "
            "--mix-centers, --mix-scales, --mix-weights"
        )
    c = np.asarray(centers, dtype=np.float64)
    s = np.asarray(scales, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if np.any(s <= 0.0):
        raise SystemExit("--mix-scales values must all be positive.")
    if np.any(w < 0.0):
        raise SystemExit("--mix-weights values must all be nonnegative.")
    sw = float(np.sum(w))
    if sw <= 0.0:
        raise SystemExit("--mix-weights must sum to a positive value.")
    return c, s, w / sw


def _sample_rewards(
    rng: np.random.Generator,
    n: int,
    dist: str,
    *,
    mix_centers: np.ndarray,
    mix_scales: np.ndarray,
    mix_weights: np.ndarray,
) -> np.ndarray:
    if dist == "uniform":
        return rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    if dist == "gaussian":
        return rng.normal(0.0, 1.0, size=n).astype(np.float64)
    if dist == "gaussian_mixture":
        comp = rng.choice(len(mix_weights), size=n, p=mix_weights)
        means = mix_centers[comp]
        scales = mix_scales[comp]
        return rng.normal(loc=means, scale=scales, size=n).astype(np.float64)
    raise RuntimeError(dist)


def _true_cdf(
    x: np.ndarray,
    dist: str,
    *,
    mix_centers: np.ndarray,
    mix_scales: np.ndarray,
    mix_weights: np.ndarray,
) -> np.ndarray:
    nd = NormalDist(0.0, 1.0)
    if dist == "uniform":
        return np.clip(x, 0.0, 1.0)
    if dist == "gaussian":
        return np.asarray([nd.cdf(float(v)) for v in x], dtype=np.float64)
    if dist == "gaussian_mixture":
        out = np.zeros_like(x, dtype=np.float64)
        for mu, sigma, w in zip(mix_centers, mix_scales, mix_weights):
            nd_i = NormalDist(float(mu), float(sigma))
            out += float(w) * np.asarray([nd_i.cdf(float(v)) for v in x], dtype=np.float64)
        return out
    raise RuntimeError(dist)


def _plotting_positions(method: str, k_ord: int) -> np.ndarray:
    # p_r = (r-a)/(k+1-2a), r = 1..k
    if method in {"Quantile", "QuantileHazen"}:
        a = 0.5
    elif method == "QuantileWeibull":
        a = 0.0
    elif method == "QuantileBlom":
        a = 3.0 / 8.0
    else:
        raise RuntimeError(method)

    r = np.arange(1, k_ord + 1, dtype=np.float64)
    p = (r - a) / (k_ord + 1.0 - 2.0 * a)
    return np.clip(p, 0.0, 1.0)


def _interp_quantile_from_orderstats(q: float, orderstats: np.ndarray, p_knots: np.ndarray) -> float:
    return float(np.interp(q, p_knots, orderstats, left=orderstats[0], right=orderstats[-1]))


def _cdf_from_orderstats(x: np.ndarray, orderstats: np.ndarray, p_knots: np.ndarray) -> np.ndarray:
    cdf = np.interp(x, orderstats, p_knots).astype(np.float64)
    mask = (x < float(orderstats[0])) | (x > float(orderstats[-1]))
    cdf[mask] = np.nan
    return cdf


def _assert_quantile_method_consistency(rng: np.random.Generator) -> None:
    """Safety check: interpolation from full orderstats must match backend quantile preset."""
    n = 32
    k = 10.0
    os = OrderStatTransform.precompute(n, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    k_ord = int(math.floor(k))
    q_grid = [0.05, 0.25, 0.5, 0.9]

    for _ in range(3):
        x = rng.normal(size=n).astype(np.float64)
        expected = os.expected_orderstats(x)
        for method in ["Quantile", "QuantileHazen", "QuantileWeibull", "QuantileBlom"]:
            p_knots = _plotting_positions(method, k_ord)
            for q in q_grid:
                interp_val = _interp_quantile_from_orderstats(q, expected, p_knots)
                api_val = float(os.expected_lstat(x, f"{method}:{q}"))
                if not np.isclose(interp_val, api_val, rtol=1e-10, atol=1e-12):
                    raise AssertionError(
                        f"Quantile interpolation mismatch for {method} q={q}: "
                        f"interp={interp_val}, api={api_val}"
                    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot true reward CDF and compare with CDF inferred from full expected order-statistics for quantile plotting-position methods."
    )
    ap.add_argument("--dist", choices=["uniform", "gaussian", "gaussian_mixture"], default="gaussian")
    ap.add_argument("--estimator", type=str, default="Quantile,QuantileWeibull,QuantileBlom")
    ap.add_argument("--k", type=float, default=10.0)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--num-estimates", type=int, default=500)
    ap.add_argument("--cdf-grid", type=int, default=600)
    ap.add_argument("--mix-centers", type=str, default="0.0,2.0", help="Comma-separated means for gaussian_mixture components.")
    ap.add_argument("--mix-scales", type=str, default="1.0,0.7", help="Comma-separated std-devs for gaussian_mixture components.")
    ap.add_argument("--mix-weights", type=str, default="0.65,0.35", help="Comma-separated nonnegative mixture weights (auto-normalized).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default="examples/artifacts/reward_cdf_quantile.png")
    ap.add_argument("--store-data", action="store_true")
    ap.add_argument("--data-dir", type=str, default="examples/data")
    ap.add_argument("--tag", type=str, default="default")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.") from e

    if args.k > args.N or math.floor(args.k) < 1:
        raise SystemExit("k must satisfy floor(k)>=1 and k<=N.")

    estimators = _parse_estimators(args.estimator)
    rng = np.random.default_rng(args.seed)
    _assert_quantile_method_consistency(np.random.default_rng(args.seed + 9991))

    mix_centers, mix_scales, mix_weights = _validate_mixture(
        _parse_float_list(args.mix_centers, "mix-centers"),
        _parse_float_list(args.mix_scales, "mix-scales"),
        _parse_float_list(args.mix_weights, "mix-weights"),
    )

    k_ord = int(math.floor(args.k))
    os = OrderStatTransform.precompute(args.N, args.k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)

    orderstats_mc = np.zeros(k_ord, dtype=np.float64)
    for _ in range(args.num_estimates):
        x = _sample_rewards(
            rng,
            args.N,
            args.dist,
            mix_centers=mix_centers,
            mix_scales=mix_scales,
            mix_weights=mix_weights,
        )
        orderstats_mc += os.expected_orderstats(x)
    orderstats_mc /= float(args.num_estimates)

    if args.dist == "uniform":
        xmin, xmax = -0.1, 1.1
    elif args.dist == "gaussian":
        xmin, xmax = -4.0, 4.0
    else:
        lo = float(np.min(mix_centers - 4.0 * mix_scales))
        hi = float(np.max(mix_centers + 4.0 * mix_scales))
        xmin, xmax = lo, hi
    xgrid = np.linspace(xmin, xmax, args.cdf_grid, dtype=np.float64)
    cdf_true = _true_cdf(
        xgrid,
        args.dist,
        mix_centers=mix_centers,
        mix_scales=mix_scales,
        mix_weights=mix_weights,
    )

    curves: dict[str, dict[str, np.ndarray | float]] = {}
    for method in estimators:
        p_knots = _plotting_positions(method, k_ord)
        cdf_est = _cdf_from_orderstats(xgrid, orderstats_mc, p_knots)
        rmse = float(np.sqrt(np.mean((cdf_est - cdf_true) ** 2)))
        curves[method] = {
            "p_knots": p_knots,
            "cdf_est": cdf_est,
            "rmse": rmse,
        }

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    ax.plot(xgrid, cdf_true, color="black", linewidth=2.2, label=f"True CDF ({args.dist})")

    for i, (method, stats) in enumerate(curves.items()):
        cdf_est = np.asarray(stats["cdf_est"], dtype=np.float64)
        rmse = float(stats["rmse"])
        ax.plot(xgrid, cdf_est, linewidth=1.8, color=f"C{i}", label=f"{method} (k={args.k:g}, RMSE={rmse:.3e})")

    ax.set_xlabel("reward value")
    ax.set_ylabel("CDF")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("True CDF vs CDF inferred from interpolated full order-statistics")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    pdf_out = out.with_suffix(".pdf")
    fig.savefig(pdf_out)
    print(f"Saved: {out}")
    print(f"Saved: {pdf_out}")

    print("method\tk\trmse_cdf")
    for method, stats in curves.items():
        print(f"{method}\t{args.k:.6g}\t{float(stats['rmse']):.6e}")

    if args.store_data:
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        stem = f"reward_cdf_quantile__{args.tag}"
        npz_path = data_dir / f"{stem}.npz"
        json_path = data_dir / f"{stem}.json"
        np.savez(
            npz_path,
            xgrid=xgrid,
            cdf_true=cdf_true,
            orderstats=orderstats_mc,
            methods=np.asarray(list(curves.keys())),
            p_knots=np.asarray([np.asarray(curves[m]["p_knots"], dtype=np.float64) for m in curves]),
            cdf_est=np.asarray([np.asarray(curves[m]["cdf_est"], dtype=np.float64) for m in curves]),
            rmse=np.asarray([float(curves[m]["rmse"]) for m in curves], dtype=np.float64),
            k=np.asarray([args.k], dtype=np.float64),
        )
        meta = {
            "experiment": "reward_cdf_quantile",
            "tag": args.tag,
            "setup": {
                "dist": args.dist,
                "N": int(args.N),
                "k": float(args.k),
                "k_ord": int(k_ord),
                "num_estimates": int(args.num_estimates),
                "methods": list(curves.keys()),
                "mix_centers": [float(x) for x in mix_centers],
                "mix_scales": [float(x) for x in mix_scales],
                "mix_weights": [float(x) for x in mix_weights],
                "seed": int(args.seed),
            },
            "artifacts": {
                "plot_png": str(out),
                "plot_pdf": str(pdf_out),
                "data_npz": str(npz_path),
            },
        }
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved: {npz_path}")
        print(f"Saved: {json_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
