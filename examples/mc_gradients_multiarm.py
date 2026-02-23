#!/usr/bin/env python3
"""Monte Carlo gradient check (multi-arm discrete setting).

Compares LR gradient estimates (using ordergrad L-advantage baseline) against
an exact known-(r,p) gradient computed by finite differences on logits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ordergrad.numpy_backend import OrderStatTransform


class BufferedIndexSampler:
    def __init__(self, rng: np.random.Generator, m: int, p: np.ndarray, *, buffer_size: int = 200_000):
        self.rng = rng
        self.m = int(m)
        self.p = np.asarray(p, dtype=np.float64)
        self.buffer_size = int(buffer_size)
        self._buf = np.empty(0, dtype=np.int64)
        self._pos = 0

    def _refill(self) -> None:
        self._buf = self.rng.choice(self.m, size=self.buffer_size, replace=True, p=self.p)
        self._pos = 0

    def sample(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.int64)
        i = 0
        while i < n:
            if self._pos >= self._buf.size:
                self._refill()
            take = min(n - i, self._buf.size - self._pos)
            out[i : i + take] = self._buf[self._pos : self._pos + take]
            self._pos += take
            i += take
        return out


def _softmax(theta: np.ndarray) -> np.ndarray:
    t = theta - np.max(theta)
    e = np.exp(t)
    return e / e.sum()


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


def _exact_grad_fd(os_exact: OrderStatTransform, r: np.ndarray, theta: np.ndarray, a: np.ndarray, eps: float) -> np.ndarray:
    g = np.zeros_like(theta)
    for i in range(theta.size):
        tp = theta.copy()
        tm = theta.copy()
        tp[i] += eps
        tm[i] -= eps
        pp = _softmax(tp)
        pm = _softmax(tm)
        fp = float(os_exact.expected_lstat_known_rp(r, pp, a))
        fm = float(os_exact.expected_lstat_known_rp(r, pm, a))
        g[i] = (fp - fm) / (2.0 * eps)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description="MC gradient check in multi-arm setting (LR vs exact known-(r,p) gradient).")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--k", type=float, default=6.0)
    ap.add_argument("--num-arms", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fd-eps", type=float, default=1e-5)
    ap.add_argument("--sample-buffer-size", type=int, default=200_000)
    ap.add_argument("--a", type=str, default=None)
    ap.add_argument("--t-grid", type=str, default="1,2,5,10,20,50,100,200,500")
    ap.add_argument("--output", type=str, default="examples/artifacts/mc_grad_multiarm.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.") from e

    rng = np.random.default_rng(args.seed)
    m = args.num_arms
    if m < 2:
        raise SystemExit("--num-arms must be >= 2")

    k_ord = int(np.floor(args.k))
    if k_ord < 1:
        raise SystemExit("Need floor(k) >= 1")

    r = np.sort(rng.normal(size=m).astype(np.float64))
    theta = rng.normal(size=m).astype(np.float64)
    p = _softmax(theta)
    a = _parse_a(args.a, k_ord)

    os_batch = OrderStatTransform.precompute(args.N, args.k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    os_batch_l = os_batch.with_lstat_weights(a)
    os_exact = OrderStatTransform.precompute(max(args.N, 2), args.k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)

    g_exact = _exact_grad_fd(os_exact, r, theta, a, args.fd_eps)

    sampler = BufferedIndexSampler(rng, m, p, buffer_size=args.sample_buffer_size)
    t_grid = [int(x) for x in args.t_grid.split(",") if x.strip()]

    abs_err = []
    rel_err = []
    g_last = None

    for t in t_grid:
        g_sum = np.zeros(m, dtype=np.float64)
        for _ in range(t):
            idx = sampler.sample(args.N)
            x = r[idx]
            l_adv = os_batch_l.expected_lstat_advantage(x)  # (N,)
            score = np.eye(m, dtype=np.float64)[idx] - p[None, :]  # (N,m)
            g_batch = (l_adv[:, None] * score).mean(axis=0)
            g_sum += g_batch
        g_mc = g_sum / t
        g_last = g_mc

        ae = np.mean(np.abs(g_mc - g_exact))
        re = np.mean(np.abs(g_mc - g_exact) / (np.abs(g_exact) + 1e-12))
        abs_err.append(float(ae))
        rel_err.append(float(re))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].plot(t_grid, abs_err, marker="o")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Multi-arm LR gradient: absolute error")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("mean abs error")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(t_grid, rel_err, marker="s")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Multi-arm LR gradient: relative error")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("mean relative error")
    axes[1].grid(True, which="both", alpha=0.3)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

    if g_last is not None:
        comp_out = out.with_name(out.stem + "_components" + out.suffix)
        fig2, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(np.arange(m), g_exact, marker="o", label="exact (finite-diff)")
        ax.plot(np.arange(m), g_last, marker="x", label=f"MC LR estimate (t={t_grid[-1]})")
        ax.set_xlabel("logit index")
        ax.set_ylabel("gradient")
        ax.set_title("Gradient components")
        ax.grid(alpha=0.3)
        ax.legend()
        fig2.tight_layout()
        fig2.savefig(comp_out, dpi=150)
        print(f"Saved: {comp_out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
