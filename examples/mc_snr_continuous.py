#!/usr/bin/env python3
"""MC variance/SNR study for continuous gradients (torch/autograd-only).

For each k in --k-grid, estimates variance and SNR for:
- RP/pathwise gradient estimator
- LR-advantage gradient estimator

SNR is computed as ||E[g]||^2 / V[g], where V[g] is sum of per-dimension
component variances.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class BufferedNormalSampler:
    def __init__(self, torch_mod: Any, *, buffer_size: int = 200_000, device=None, dtype=None):
        self.torch = torch_mod
        self.buffer_size = int(buffer_size)
        self.device = device
        self.dtype = dtype
        self._buf = self.torch.empty((0, 1), dtype=dtype, device=device)
        self._pos = 0
        self._dim = 1

    def _refill(self, dim: int) -> None:
        self._dim = dim
        self._buf = self.torch.randn((self.buffer_size, dim), dtype=self.dtype, device=self.device)
        self._pos = 0

    def sample(self, n: int, dim: int):
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
    text = spec.strip()
    if any(ch.isalpha() for ch in text):
        return text
    vals = [float(x) for x in text.split(",") if x.strip()]
    if len(vals) == 1:
        vals = vals * k_ord
    elif len(vals) != k_ord:
        raise SystemExit(f"--a must have 1 value, exactly floor(k)={k_ord} values, or a preset string.")
    return torch_mod.tensor(vals, dtype=dtype, device=device).flip(0)


def _parse_k_grid(spec: str) -> list[float]:
    ks = [float(x) for x in spec.split(",") if x.strip()]
    if not ks:
        raise SystemExit("--k-grid must contain at least one value")
    return ks


def _snr_from_samples(g):
    mean_g = g.mean(dim=0)
    var_g = g.var(dim=0, unbiased=True).sum()
    snr = (mean_g.pow(2).sum() / (var_g + 1e-18)).item()
    return mean_g, var_g.item(), snr


def main() -> None:
    ap = argparse.ArgumentParser(description="MC SNR vs k for continuous RP and LR estimators (torch).")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--dim", type=int, default=1)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--center", type=float, default=1.0)
    ap.add_argument("--objective", type=str, default="quadratic", choices=["quadratic", "quad_sin"], help="Objective shape for rewards in continuous case.")
    ap.add_argument("--sin-freq", type=float, default=4.0, help="Frequency used when --objective=quad_sin.")
    ap.add_argument("--k-grid", type=str, default="1,2,3,4,5,6")
    ap.add_argument("--num-mc", type=int, default=2000)
    ap.add_argument("--a", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-buffer-size", type=int, default=200_000)
    ap.add_argument("--output", type=str, default="examples/artifacts/mc_snr_continuous.png")
    ap.add_argument("--store-data", action="store_true", help="Store experiment data + setup metadata to disk.")
    ap.add_argument("--tag", type=str, default="default", help="Tag used in stored data filename/metadata.")
    ap.add_argument("--data-dir", type=str, default="examples/data", help="Directory where experiment data is stored.")
    ap.add_argument("--no-plot", action="store_true", help="Skip plot rendering/saving (useful for data-only sweeps).")
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
        raise SystemExit("torch/ordergrad torch backend is required.") from e

    torch.manual_seed(args.seed)
    dtype = torch.float64
    device = torch.device("cpu")
    ks = _parse_k_grid(args.k_grid)

    eps_sampler = BufferedNormalSampler(torch, buffer_size=args.sample_buffer_size, device=device, dtype=dtype)
    const = 0.5 * torch.log(torch.tensor(2.0 * 3.141592653589793, dtype=dtype, device=device))

    rp_var, lr_var, rp_snr, lr_snr = [], [], [], []

    for k in ks:
        k_ord = int(torch.floor(torch.tensor(k)).item())
        if k_ord < 1:
            raise SystemExit("All k in --k-grid must satisfy floor(k) >= 1")
        a = _parse_a(args.a, k_ord, torch, device=device, dtype=dtype)
        os = OrderStatTransform.precompute(args.N, k, dtype=dtype, compute_conditional=True, compute_leave_one_out=True)
        os_l = os.with_lstat_weights(a)

        g_rp_all = torch.empty((args.num_mc, args.dim), dtype=dtype, device=device)
        g_lr_all = torch.empty((args.num_mc, args.dim), dtype=dtype, device=device)

        for t in range(args.num_mc):
            eps = eps_sampler.sample(args.N, args.dim)

            mu_rp = torch.full((args.dim,), float(args.mu), dtype=dtype, device=device, requires_grad=True)
            z_rp = mu_rp[None, :] + eps
            if args.objective == "quadratic":
                x_rp = -torch.sum((z_rp - float(args.center)) ** 2, dim=1)
            else:
                x_rp = -torch.sum((z_rp - float(args.center)) ** 2, dim=1) + 0.2 * torch.sum(torch.sin(float(args.sin_freq) * z_rp), dim=1)
            l_rp = os_l.expected_lstat(x_rp)
            g_rp_all[t] = torch.autograd.grad(l_rp, mu_rp, retain_graph=False, create_graph=False)[0]

            z_sample = (torch.full((args.dim,), float(args.mu), dtype=dtype, device=device)[None, :] + eps).detach()
            if args.objective == "quadratic":
                x_lr = -torch.sum((z_sample - float(args.center)) ** 2, dim=1)
            else:
                x_lr = -torch.sum((z_sample - float(args.center)) ** 2, dim=1) + 0.2 * torch.sum(torch.sin(float(args.sin_freq) * z_sample), dim=1)
            l_adv = os_l.expected_lstat_advantage(x_lr).detach()
            mu_lr = torch.full((args.dim,), float(args.mu), dtype=dtype, device=device, requires_grad=True)
            logp = -0.5 * torch.sum((z_sample - mu_lr[None, :]) ** 2, dim=1) - (args.dim * const)
            obj = (float(k) / float(args.N)) * torch.sum(l_adv * logp)
            g_lr_all[t] = torch.autograd.grad(obj, mu_lr, retain_graph=False, create_graph=False)[0]

        _, vrp, srp = _snr_from_samples(g_rp_all)
        _, vlr, slr = _snr_from_samples(g_lr_all)
        rp_var.append(vrp)
        lr_var.append(vlr)
        rp_snr.append(srp)
        lr_snr.append(slr)

    out = Path(args.output)
    if not args.no_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        axes[0].plot(ks, rp_var, marker='o', label='RP Var[g]')
        axes[0].plot(ks, lr_var, marker='x', label='LR Var[g]')
        axes[0].set_xlabel('k')
        axes[0].set_title('Estimator variance vs k')
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(ks, rp_snr, marker='o', label='RP SNR')
        axes[1].plot(ks, lr_snr, marker='x', label='LR SNR')
        axes[1].set_xlabel('k')
        axes[1].set_title('SNR = ||E[g]||^2 / V[g]')
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        pdf_out = out.with_suffix('.pdf')
        fig.savefig(pdf_out)
        print(f"Saved: {out}")
        print(f"Saved: {pdf_out}")

    if args.store_data:
        import numpy as np

        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        stem = f"mc_snr_continuous__{args.tag}"
        npz_path = data_dir / f"{stem}.npz"
        json_path = data_dir / f"{stem}.json"

        np.savez(
            npz_path,
            k=np.asarray(ks, dtype=np.float64),
            rp_variance=np.asarray(rp_var, dtype=np.float64),
            lr_variance=np.asarray(lr_var, dtype=np.float64),
            rp_snr=np.asarray(rp_snr, dtype=np.float64),
            lr_snr=np.asarray(lr_snr, dtype=np.float64),
        )

        metadata = {
            "experiment": "mc_snr_continuous",
            "tag": args.tag,
            "setup": {
                "N": args.N,
                "dim": args.dim,
                "mu": args.mu,
                "center": args.center,
                "k_grid": ks,
                "num_mc": args.num_mc,
                "a": args.a,
                "seed": args.seed,
                "sample_buffer_size": args.sample_buffer_size,
            },
            "artifacts": {
                "plot": (str(out) if not args.no_plot else None),
                "data_npz": str(npz_path),
            },
        }
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved: {npz_path}")
        print(f"Saved: {json_path}")

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
