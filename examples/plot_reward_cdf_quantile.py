#!/usr/bin/env python3
"""Plot true reward CDF and compare to quantile estimates from chosen L-stat estimator(s)."""

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
    return out


def _parse_k_list(spec: str, n: int) -> list[float]:
    vals = [float(x) for x in spec.split(",") if x.strip()]
    if not vals:
        raise SystemExit("--k-list must contain at least one value.")
    if len(vals) == 1:
        return vals * n
    if len(vals) == n:
        return vals
    raise SystemExit(f"--k-list must have 1 value or exactly len(--estimator)={n} values.")


def _sample_rewards(rng: np.random.Generator, n: int, dist: str, *, mix_weight: float, mix_mu: float, mix_sigma: float) -> np.ndarray:
    if dist == "uniform":
        return rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    if dist == "gaussian":
        return rng.normal(0.0, 1.0, size=n).astype(np.float64)
    if dist == "gaussian_mixture":
        z = rng.random(size=n)
        x0 = rng.normal(0.0, 1.0, size=n)
        x1 = rng.normal(mix_mu, mix_sigma, size=n)
        return np.where(z < mix_weight, x1, x0).astype(np.float64)
    raise RuntimeError(dist)


def _true_cdf(x: np.ndarray, dist: str, *, mix_weight: float, mix_mu: float, mix_sigma: float) -> np.ndarray:
    nd = NormalDist(0.0, 1.0)
    if dist == "uniform":
        return np.clip(x, 0.0, 1.0)
    if dist == "gaussian":
        return np.asarray([nd.cdf(float(v)) for v in x], dtype=np.float64)
    if dist == "gaussian_mixture":
        nd1 = NormalDist(mix_mu, mix_sigma)
        return (1.0 - mix_weight) * np.asarray([nd.cdf(float(v)) for v in x], dtype=np.float64) + mix_weight * np.asarray([nd1.cdf(float(v)) for v in x], dtype=np.float64)
    raise RuntimeError(dist)


def _true_quantile(q: float, dist: str, *, mix_weight: float, mix_mu: float, mix_sigma: float) -> float:
    if dist == "uniform":
        return q
    if dist == "gaussian":
        return NormalDist(0.0, 1.0).inv_cdf(q)
    # numeric inverse CDF for mixture
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        c = _true_cdf(np.asarray([mid]), dist, mix_weight=mix_weight, mix_mu=mix_mu, mix_sigma=mix_sigma)[0]
        if c < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot true CDF and quantile-estimator comparison for selected reward distribution.")
    ap.add_argument("--dist", choices=["uniform", "gaussian", "gaussian_mixture"], default="gaussian")
    ap.add_argument("--quantile", type=float, default=0.25)
    ap.add_argument("--estimator", type=str, default="Quantile,HarrellDavis", help="Comma-separated estimators (e.g. QuantileHazen,QuantileBlom,HarrellDavis).")
    ap.add_argument("--k-list", type=str, default="6")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--num-estimates", type=int, default=500)
    ap.add_argument("--cdf-grid", type=int, default=500)
    ap.add_argument("--mix-weight", type=float, default=0.35)
    ap.add_argument("--mix-mu", type=float, default=2.0)
    ap.add_argument("--mix-sigma", type=float, default=0.7)
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

    q = float(args.quantile)
    if not (0.0 <= q <= 1.0):
        raise SystemExit("--quantile must be in [0,1].")

    estimators = _parse_estimators(args.estimator)
    k_list = _parse_k_list(args.k_list, len(estimators))
    if any(k > args.N or math.floor(k) < 1 for k in k_list):
        raise SystemExit("All k must satisfy floor(k)>=1 and k<=N.")

    rng = np.random.default_rng(args.seed)

    exact_q = _true_quantile(q, args.dist, mix_weight=args.mix_weight, mix_mu=args.mix_mu, mix_sigma=args.mix_sigma)

    # define x-range for CDF view
    if args.dist == "uniform":
        xmin, xmax = -0.1, 1.1
    elif args.dist == "gaussian":
        xmin, xmax = -4.0, 4.0
    else:
        xmin, xmax = -4.0, max(6.0, args.mix_mu + 4 * args.mix_sigma)
    xgrid = np.linspace(xmin, xmax, args.cdf_grid, dtype=np.float64)
    cdf = _true_cdf(xgrid, args.dist, mix_weight=args.mix_weight, mix_mu=args.mix_mu, mix_sigma=args.mix_sigma)

    estimates = {}
    for method, k in zip(estimators, k_list):
        os = OrderStatTransform.precompute(args.N, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
        spec = f"{method}:{q}"
        vals = np.empty(args.num_estimates, dtype=np.float64)
        for i in range(args.num_estimates):
            x = _sample_rewards(rng, args.N, args.dist, mix_weight=args.mix_weight, mix_mu=args.mix_mu, mix_sigma=args.mix_sigma)
            vals[i] = os.expected_lstat(x, spec)
        estimates[method] = {
            "k": float(k),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if args.num_estimates > 1 else 0.0,
            "vals": vals,
        }

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(xgrid, cdf, color="black", linewidth=2.0, label=f"True CDF ({args.dist})")
    ax.axhline(q, color="gray", linestyle="--", linewidth=1.2, label=f"target q={q:g}")
    ax.axvline(exact_q, color="tab:green", linestyle="-", linewidth=1.8, label=f"True quantile={exact_q:.4g}")

    for j, (method, st) in enumerate(estimates.items()):
        color = f"C{j}"
        ax.axvline(st["mean"], color=color, linestyle="--", linewidth=1.6, label=f"{method}:q (k={st['k']:g}) mean={st['mean']:.4g}, sd={st['std']:.3g}")

    ax.set_xlabel("reward value")
    ax.set_ylabel("CDF")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("True reward CDF and quantile-estimator comparison")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    pdf_out = out.with_suffix(".pdf")
    fig.savefig(pdf_out)
    print(f"Saved: {out}")
    print(f"Saved: {pdf_out}")

    print("method\tk\tmean_est\tstd_est\ttrue_quantile\tabs_err")
    for method, st in estimates.items():
        ae = abs(st["mean"] - exact_q)
        print(f"{method}\t{st['k']:.6g}\t{st['mean']:.8g}\t{st['std']:.3e}\t{exact_q:.8g}\t{ae:.3e}")

    if args.store_data:
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        stem = f"reward_cdf_quantile__{args.tag}"
        npz_path = data_dir / f"{stem}.npz"
        json_path = data_dir / f"{stem}.json"
        np.savez(
            npz_path,
            xgrid=xgrid,
            cdf=cdf,
            methods=np.asarray(list(estimates.keys())),
            k_list=np.asarray([estimates[m]["k"] for m in estimates], dtype=np.float64),
            means=np.asarray([estimates[m]["mean"] for m in estimates], dtype=np.float64),
            stds=np.asarray([estimates[m]["std"] for m in estimates], dtype=np.float64),
            exact_q=np.asarray([exact_q], dtype=np.float64),
            target_q=np.asarray([q], dtype=np.float64),
        )
        meta = {
            "experiment": "reward_cdf_quantile",
            "tag": args.tag,
            "setup": {
                "dist": args.dist,
                "quantile": q,
                "N": int(args.N),
                "num_estimates": int(args.num_estimates),
                "methods": list(estimates.keys()),
                "k_list": [estimates[m]["k"] for m in estimates],
                "mix_weight": float(args.mix_weight),
                "mix_mu": float(args.mix_mu),
                "mix_sigma": float(args.mix_sigma),
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
