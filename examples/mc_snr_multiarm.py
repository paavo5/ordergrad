#!/usr/bin/env python3
"""MC variance/SNR study for multi-arm LR gradients (torch-only).

For each k in --k-grid, estimates:
- mean gradient E[g]
- scalar variance V[g] (sum of per-dimension variances)
- signal-to-noise ratio SNR = ||E[g]||^2 / V[g]

Gradient estimator uses ordergrad L-advantage with LR score-function scaling.
"""

from __future__ import annotations

import argparse
from pathlib import Path


class BufferedIndexSampler:
    def __init__(self, probs, *, buffer_size: int = 200_000):
        import torch

        self.probs = probs
        self.buffer_size = int(buffer_size)
        self._buf = torch.empty(0, dtype=torch.long, device=probs.device)
        self._pos = 0

    def _refill(self) -> None:
        import torch

        self._buf = torch.multinomial(self.probs, num_samples=self.buffer_size, replacement=True)
        self._pos = 0

    def sample(self, n: int):
        out = self.probs.new_empty((n,), dtype=self._buf.dtype)
        i = 0
        while i < n:
            if self._pos >= self._buf.numel():
                self._refill()
            take = min(n - i, self._buf.numel() - self._pos)
            out[i : i + take] = self._buf[self._pos : self._pos + take]
            self._pos += take
            i += take
        return out


def _parse_a(spec: str | None, k_ord: int, *, device, dtype):
    import torch

    if spec is None:
        return torch.linspace(0.3, 1.0, steps=k_ord, dtype=dtype, device=device)
    text = spec.strip()
    if any(ch.isalpha() for ch in text):
        return text
    vals = [float(x) for x in text.split(",") if x.strip()]
    if len(vals) == 1:
        vals = vals * k_ord
    elif len(vals) != k_ord:
        raise SystemExit(f"--a must have 1 value, exactly floor(k)={k_ord} values, or a preset string.")
    return torch.tensor(vals, dtype=dtype, device=device)


def _parse_k_grid(spec: str) -> list[float]:
    ks = [float(x) for x in spec.split(",") if x.strip()]
    if not ks:
        raise SystemExit("--k-grid must contain at least one value")
    return ks


def main() -> None:
    ap = argparse.ArgumentParser(description="MC SNR vs k for multi-arm LR gradient estimator (torch-only).")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--num-arms", type=int, default=6)
    ap.add_argument("--k-grid", type=str, default="1,2,3,4,5,6")
    ap.add_argument("--num-mc", type=int, default=2000, help="Number of independent gradient-estimator draws per k.")
    ap.add_argument("--a", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-buffer-size", type=int, default=200_000)
    ap.add_argument("--output", type=str, default="examples/artifacts/mc_snr_multiarm.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.") from e

    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise SystemExit("torch is required. Install with `pip install torch`.") from e

    from ordergrad.torch_backend import OrderStatTransform

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    dtype = torch.float64

    m = int(args.num_arms)
    if m < 2:
        raise SystemExit("--num-arms must be >= 2")

    ks = _parse_k_grid(args.k_grid)
    if any(k < 1 or k > args.N for k in ks):
        raise SystemExit("All k in --k-grid must satisfy 1 <= k <= N")

    r = torch.sort(torch.randn(m, dtype=dtype, device=device))[0]
    theta = torch.randn(m, dtype=dtype, device=device)
    p = torch.softmax(theta, dim=0)
    sampler = BufferedIndexSampler(p, buffer_size=args.sample_buffer_size)

    var_vals, snr_vals, mean_norm_vals = [], [], []

    for k in ks:
        k_ord = int(torch.floor(torch.tensor(k)).item())
        a = _parse_a(args.a, k_ord, device=device, dtype=dtype)
        os = OrderStatTransform.precompute(args.N, k, dtype=dtype, compute_conditional=True, compute_leave_one_out=True)
        os_l = os.with_lstat_weights(a)

        grads = torch.empty((args.num_mc, m), dtype=dtype, device=device)
        for t in range(args.num_mc):
            idx = sampler.sample(args.N)
            x = r[idx]
            l_adv = os_l.expected_lstat_advantage(x).detach()  # (N,)
            score = torch.nn.functional.one_hot(idx, num_classes=m).to(dtype) - p[None, :]  # (N,m)
            g = float(k) * (l_adv[:, None] * score).mean(dim=0)
            grads[t] = g

        mean_g = grads.mean(dim=0)
        var_g = grads.var(dim=0, unbiased=True).sum()
        snr = (mean_g.pow(2).sum() / (var_g + 1e-18)).item()

        mean_norm_vals.append(torch.linalg.norm(mean_g).item())
        var_vals.append(var_g.item())
        snr_vals.append(snr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].plot(ks, var_vals, marker="o", label="Var[g]")
    axes[0].plot(ks, mean_norm_vals, marker="x", label="||E[g]||")
    axes[0].set_xlabel("k")
    axes[0].set_title("Multi-arm LR gradient moments vs k")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ks, snr_vals, marker="s", color="tab:green")
    axes[1].set_xlabel("k")
    axes[1].set_title("SNR = ||E[g]||^2 / V[g]")
    axes[1].grid(alpha=0.3)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise SystemExit("torch is required. Install with `pip install torch`.") from e
