#!/usr/bin/env python3
"""Monte Carlo gradient check (continuous, torch/autograd-only).

Compares reparameterization/pathwise (RP) and LR-advantage gradient estimators
for a Normal location model transformed through a quadratic reward function.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


class BufferedNormalSampler:
    def __init__(self, torch_mod: Any, *, buffer_size: int = 200_000, device=None, dtype=None):
        self.torch = torch_mod
        self.buffer_size = int(buffer_size)
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self.device = device
        self.dtype = dtype
        self._buf = self.torch.empty(0, dtype=self.dtype, device=self.device)
        self._pos = 0

    def _refill(self) -> None:
        self._buf = self.torch.randn(self.buffer_size, dtype=self.dtype, device=self.device)
        self._pos = 0

    def sample(self, n: int):
        out = self.torch.empty(n, dtype=self.dtype, device=self.device)
        i = 0
        while i < n:
            if self._pos >= self._buf.numel():
                self._refill()
            take = min(n - i, self._buf.numel() - self._pos)
            out[i : i + take] = self._buf[self._pos : self._pos + take]
            self._pos += take
            i += take
        return out


def _parse_a(spec: str | None, k_ord: int, torch_mod: Any, *, device, dtype):
    if spec is None:
        return torch_mod.linspace(0.3, 1.0, steps=k_ord, dtype=dtype, device=device)
    vals = [float(x) for x in spec.split(",") if x.strip()]
    if len(vals) == 0:
        raise SystemExit("--a was provided but no values were parsed.")
    if len(vals) == 1:
        vals = vals * k_ord
    elif len(vals) != k_ord:
        raise SystemExit(f"--a must have either 1 value or exactly floor(k)={k_ord} values.")
    return torch_mod.tensor(vals, dtype=dtype, device=device)


def _safe_for_logplot(vals, eps: float = 1e-16):
    out = []
    for v in vals:
        fv = float(v)
        if not (fv > 0.0):
            fv = eps
        out.append(fv)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="MC gradient check in continuous setting (torch RP vs torch LR-adv).")
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

    try:
        import torch
        from ordergrad.torch_backend import OrderStatTransform
    except Exception as e:  # pragma: no cover
        raise SystemExit("torch/ordergrad torch backend is required for this example. Install with `pip install torch`.") from e

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    dtype = torch.float64

    k_ord = int(torch.floor(torch.tensor(args.k)).item())
    if k_ord < 1:
        raise SystemExit("Need floor(k) >= 1")
    a = _parse_a(args.a, k_ord, torch, device=device, dtype=dtype)

    os = OrderStatTransform.precompute(args.N, args.k, dtype=dtype, compute_conditional=True, compute_leave_one_out=True)
    os_l = os.with_lstat_weights(a)

    eps_sampler = BufferedNormalSampler(torch, buffer_size=args.sample_buffer_size, device=device, dtype=dtype)

    t_grid = [int(x) for x in args.t_grid.split(",") if x.strip()]
    if any(t <= 0 for t in t_grid):
        raise SystemExit("All t-grid entries must be positive.")

    abs_gap = []
    rel_gap = []
    rp_trace = []
    lr_trace = []

    const = 0.5 * torch.log(torch.tensor(2.0 * 3.141592653589793, dtype=dtype, device=device))

    for t in t_grid:
        rp_sum = 0.0
        lr_sum = 0.0
        for _ in range(t):
            eps = eps_sampler.sample(args.N)

            # RP estimator via autograd through reparameterized objective.
            mu_rp = torch.tensor(args.mu, dtype=dtype, device=device, requires_grad=True)
            z_rp = mu_rp + eps
            x_rp = -((z_rp - args.center) ** 2)
            l_rp = os_l.expected_lstat(x_rp)
            g_rp = torch.autograd.grad(l_rp, mu_rp, retain_graph=False, create_graph=False)[0]

            # LR estimator via autograd score terms with k multiplier.
            mu_lr = torch.tensor(args.mu, dtype=dtype, device=device, requires_grad=True)
            z_lr = mu_lr + eps
            x_lr = -((z_lr - args.center) ** 2)
            l_adv = os_l.expected_lstat_advantage(x_lr).detach()

            # log N(z|mu,1) = -0.5*(z-mu)^2 - const
            logp = -0.5 * ((z_lr - mu_lr) ** 2) - const
            weighted_score = torch.zeros((), dtype=dtype, device=device)
            for n in range(args.N):
                g_n = torch.autograd.grad(logp[n], mu_lr, retain_graph=True, create_graph=False)[0]
                weighted_score = weighted_score + l_adv[n] * g_n
            g_lr = (float(args.k) * weighted_score) / float(args.N)

            rp_sum += float(g_rp.item())
            lr_sum += float(g_lr.item())

        rp_mean = rp_sum / t
        lr_mean = lr_sum / t
        rp_trace.append(rp_mean)
        lr_trace.append(lr_mean)

        gap = abs(rp_mean - lr_mean)
        abs_gap.append(gap)
        rel_gap.append(gap / (abs(rp_mean) + 1e-12))

    abs_gap_plot = _safe_for_logplot(abs_gap)
    rel_gap_plot = _safe_for_logplot(rel_gap)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].plot(t_grid, abs_gap_plot, marker="o")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Continuous gradient: |RP - LR|")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("absolute gap")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(t_grid, rel_gap_plot, marker="s")
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
