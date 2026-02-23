#!/usr/bin/env python3
"""Monte Carlo gradient check (continuous setting).

Compares reparameterization (RP/pathwise) and LR-advantage gradient estimators
for a Normal location model transformed through a quadratic reward function.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ordergrad.numpy_backend import OrderStatTransform


class BufferedNormalSampler:
    def __init__(self, rng: np.random.Generator, *, buffer_size: int = 200_000):
        self.rng = rng
        self.buffer_size = int(buffer_size)
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self._buf = np.empty(0, dtype=np.float64)
        self._pos = 0

    def _refill(self) -> None:
        self._buf = self.rng.normal(size=self.buffer_size).astype(np.float64)
        self._pos = 0

    def sample(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            if self._pos >= self._buf.size:
                self._refill()
            take = min(n - i, self._buf.size - self._pos)
            out[i : i + take] = self._buf[self._pos : self._pos + take]
            self._pos += take
            i += take
        return out


def _parse_a(spec: str | None, k_ord: int) -> np.ndarray:
    if spec is None:
        return np.linspace(0.3, 1.0, k_ord, dtype=np.float64)
    vals = [float(x) for x in spec.split(",") if x.strip()]
    if len(vals) == 0:
        raise SystemExit("--a was provided but no values were parsed.")
    if len(vals) == 1:
        vals = vals * k_ord
    elif len(vals) != k_ord:
        raise SystemExit(f"--a must have either 1 value or exactly floor(k)={k_ord} values.")
    return np.asarray(vals, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="MC gradient check in continuous setting (RP vs LR-adv).")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--k", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu", type=float, default=0.5, help="Normal location parameter.")
    ap.add_argument("--center", type=float, default=1.0, help="Quadratic reward center c for f(z)=-(z-c)^2.")
    ap.add_argument("--sample-buffer-size", type=int, default=200_000)
    ap.add_argument("--a", type=str, default=None)
    ap.add_argument("--t-grid", type=str, default="1,2,5,10,20,50,100,200,500")
    ap.add_argument("--output", type=str, default="examples/artifacts/mc_grad_continuous.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.") from e

    rng = np.random.default_rng(args.seed)
    k_ord = int(np.floor(args.k))
    if k_ord < 1:
        raise SystemExit("Need floor(k) >= 1")
    a = _parse_a(args.a, k_ord)

    os = OrderStatTransform.precompute(args.N, args.k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    os_l = os.with_lstat_weights(a)

    eps_sampler = BufferedNormalSampler(rng, buffer_size=args.sample_buffer_size)

    t_grid = [int(x) for x in args.t_grid.split(",") if x.strip()]
    if any(t <= 0 for t in t_grid):
        raise SystemExit("All t-grid entries must be positive.")

    abs_gap = []
    rel_gap = []
    rp_trace = []
    lr_trace = []

    for t in t_grid:
        rp_sum = 0.0
        lr_sum = 0.0
        for _ in range(t):
            e = eps_sampler.sample(args.N)
            z = args.mu + e
            x = -((z - args.center) ** 2)

            # RP (pathwise): d/dmu x_i = -2(z_i-center)
            dx_dmu = -2.0 * (z - args.center)
            w_item = os_l.lstat_weight_by_item(x)  # gradient wrt x (away from ties)
            g_rp = float(np.dot(w_item, dx_dmu))

            # LR with advantage baseline: grad log N(z|mu,1) wrt mu = z-mu
            score = z - args.mu
            l_adv = os_l.expected_lstat_advantage(x)
            g_lr = float(np.mean(l_adv * score))

            rp_sum += g_rp
            lr_sum += g_lr

        rp_mean = rp_sum / t
        lr_mean = lr_sum / t
        rp_trace.append(rp_mean)
        lr_trace.append(lr_mean)

        gap = abs(rp_mean - lr_mean)
        abs_gap.append(gap)
        rel_gap.append(gap / (abs(rp_mean) + 1e-12))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].plot(t_grid, abs_gap, marker="o")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Continuous gradient: |RP - LR|")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("absolute gap")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(t_grid, rel_gap, marker="s")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Continuous gradient: relative gap")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("relative gap")
    axes[1].grid(True, which="both", alpha=0.3)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

    comp_out = out.with_name(out.stem + "_traces" + out.suffix)
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t_grid, rp_trace, marker="o", label="RP mean gradient")
    ax.plot(t_grid, lr_trace, marker="x", label="LR-adv mean gradient")
    ax.set_xscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("gradient estimate")
    ax.set_title("RP vs LR gradient estimates")
    ax.grid(alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(comp_out, dpi=150)
    print(f"Saved: {comp_out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
