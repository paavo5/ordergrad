#!/usr/bin/env python3
"""Monte Carlo gradient check (continuous, torch/autograd-only).

Compares reparameterization/pathwise (RP) and LR-advantage gradient estimators
for a Normal location model transformed through a quadratic reward function.
Supports both 1D and multi-dimensional location parameters.
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
        self._buf = self.torch.empty((0, 1), dtype=self.dtype, device=self.device)
        self._pos = 0
        self._dim = 1

    def _refill(self, dim: int) -> None:
        self._dim = int(dim)
        self._buf = self.torch.randn((self.buffer_size, self._dim), dtype=self.dtype, device=self.device)
        self._pos = 0

    def sample(self, n: int, dim: int):
        dim = int(dim)
        if self._buf.shape[0] == 0 or self._pos >= self._buf.shape[0] or self._dim != dim:
            self._refill(dim)
        out = self.torch.empty((n, dim), dtype=self.dtype, device=self.device)
        i = 0
        while i < n:
            if self._pos >= self._buf.shape[0]:
                self._refill(dim)
            take = min(n - i, self._buf.shape[0] - self._pos)
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
    ap.add_argument("--dim", type=int, default=1, help="Dimensionality of location parameter mu.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu", type=float, default=0.5, help="Base Normal location parameter (broadcast to all dims).")
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

    if args.dim < 1:
        raise SystemExit("--dim must be >= 1")

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
        rp_sum = torch.zeros(args.dim, dtype=dtype, device=device)
        lr_sum = torch.zeros(args.dim, dtype=dtype, device=device)
        for _ in range(t):
            eps = eps_sampler.sample(args.N, args.dim)

            # RP estimator via autograd through reparameterized objective.
            mu_rp = torch.full((args.dim,), float(args.mu), dtype=dtype, device=device, requires_grad=True)
            z_rp = mu_rp[None, :] + eps
            x_rp = -torch.sum((z_rp - float(args.center)) ** 2, dim=1)
            l_rp = os_l.expected_lstat(x_rp)
            g_rp = torch.autograd.grad(l_rp, mu_rp, retain_graph=False, create_graph=False)[0]

            # LR estimator via autograd score terms with k multiplier.
            # Important: detach samples in the score term (treat sampled z as fixed)
            # so grad_mu log p(z; mu) is computed correctly.
            z_sample = (torch.full((args.dim,), float(args.mu), dtype=dtype, device=device)[None, :] + eps).detach()
            x_lr = -torch.sum((z_sample - float(args.center)) ** 2, dim=1)
            l_adv = os_l.expected_lstat_advantage(x_lr).detach()

            mu_lr = torch.full((args.dim,), float(args.mu), dtype=dtype, device=device, requires_grad=True)
            # log N(z|mu, I) = -0.5*||z-mu||^2 - d*const
            logp = -0.5 * torch.sum((z_sample - mu_lr[None, :]) ** 2, dim=1) - (args.dim * const)
            lr_objective = (float(args.k) / float(args.N)) * torch.sum(l_adv * logp)
            g_lr = torch.autograd.grad(lr_objective, mu_lr, retain_graph=False, create_graph=False)[0]

            rp_sum += g_rp
            lr_sum += g_lr

        rp_mean = rp_sum / float(t)
        lr_mean = lr_sum / float(t)
        rp_norm = float(torch.linalg.norm(rp_mean).item())
        lr_norm = float(torch.linalg.norm(lr_mean).item())
        rp_trace.append(rp_norm)
        lr_trace.append(lr_norm)

        gap = float(torch.mean(torch.abs(rp_mean - lr_mean)).item())
        scale = float(torch.mean(torch.abs(rp_mean)).item())
        abs_gap.append(gap)
        rel_gap.append(gap / (scale + 1e-12))

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
    ax.plot(t_grid, rp_trace, marker="o", label="RP ||mean gradient||")
    ax.plot(t_grid, lr_trace, marker="x", label="LR-adv ||mean gradient||")
    ax.set_xscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("gradient norm")
    ax.set_title("RP vs LR gradient norms")
    ax.grid(alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(comp_out, dpi=150)
    print(f"Saved: {comp_out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
