#!/usr/bin/env python3
"""Monte Carlo gradient check (multi-arm, torch-only).

Compares LR gradient estimates (using ordergrad L-advantage baseline) against an
exact known-(r,p) gradient computed with torch autograd.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typing import Any

import numpy as np
import torch


class BufferedIndexSampler:
    """Sample arm indices via buffered categorical draws."""

    def __init__(self, probs, *, buffer_size: int = 200_000, device=None):
        self.probs = probs
        self.buffer_size = int(buffer_size)
        self.device = device if device is not None else probs.device
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self._buf = torch.empty(0, dtype=torch.long, device=self.device)
        self._pos = 0

    def _refill(self) -> None:
        self._buf = torch.multinomial(self.probs, num_samples=self.buffer_size, replacement=True)
        self._pos = 0

    def sample(self, n: int) -> torch.Tensor:
        n = int(n)
        out = torch.empty(n, dtype=torch.long, device=self.device)
        filled = 0
        while filled < n:
            if self._pos >= self._buf.numel():
                self._refill()
            take = min(n - filled, self._buf.numel() - self._pos)
            out[filled : filled + take] = self._buf[self._pos : self._pos + take]
            self._pos += take
            filled += take
        return out


def _safe_for_logplot(vals, eps: float = 1e-16):
    """Ensure strictly positive values for log-scaled plots."""
    out = []
    for v in vals:
        fv = float(v)
        if not (fv > 0.0):
            fv = eps
        out.append(fv)
    return out



def _make_rewards(m: int, mode: str, *, device, dtype) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode == "gaussian":
        return torch.sort(torch.randn(m, dtype=dtype, device=device))[0]
    if mode in {"linear", "arange"}:
        return torch.arange(m, dtype=dtype, device=device)
    if mode in {"exp", "pow2", "2^m"}:
        return torch.pow(torch.tensor(2.0, dtype=dtype, device=device), torch.arange(m, dtype=dtype, device=device))
    raise SystemExit("--reward-mode must be one of: gaussian, linear, exp")


def _init_theta(m: int, mode: str, *, device, dtype) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode == "random":
        return torch.randn(m, dtype=dtype, device=device, requires_grad=True)
    if mode in {"uniform", "constant", "equal"}:
        return torch.zeros(m, dtype=dtype, device=device, requires_grad=True)
    raise SystemExit("--prob-mode must be one of: random, uniform")

def _parse_a(spec: str | None, k_ord: int, *, device, dtype):
    if spec is None:
        return torch.linspace(0.3, 1.0, steps=k_ord, dtype=dtype, device=device)
    text = spec.strip()
    if any(ch.isalpha() for ch in text):
        if "," in text:
            tokens = [tok.strip() for tok in text.split(",") if tok.strip()]
            raise SystemExit(
                "Invalid preset list for --a: this script expects a single preset string "
                f"or numeric weights. Received multiple preset-like entries: {tokens}. "
                "Pass one preset (e.g., TopM:3) or numeric comma-separated weights."
            )
        return text
    vals = [float(x) for x in text.split(",") if x.strip()]
    if len(vals) == 0:
        raise SystemExit("--a was provided but no values were parsed.")
    if len(vals) == 1:
        vals = vals * k_ord
    elif len(vals) != k_ord:
        raise SystemExit(f"--a must have either 1 value, exactly floor(k)={k_ord} values, or a preset string.")
    return torch.tensor(vals, dtype=dtype, device=device).flip(0)


def main() -> None:
    ap = argparse.ArgumentParser(description="MC gradient check in multi-arm setting (torch LR vs torch exact gradient).")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--k", type=float, default=6.0)
    ap.add_argument("--num-arms", type=int, default=6)
    ap.add_argument("--reward-mode", type=str, default="gaussian", choices=["gaussian", "linear", "exp"], help="How arm rewards are generated: gaussian (fixed random), linear (arange), exp (2**m).")
    ap.add_argument("--prob-mode", type=str, default="random", choices=["random", "uniform"], help="How action sampling probabilities are initialized: random softmax logits or uniform over actions.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-buffer-size", type=int, default=200_000)
    ap.add_argument("--a", type=str, default=None)
    ap.add_argument("--t-grid", type=str, default="1,2,5,10,20,50,100,200,500")
    ap.add_argument("--output", type=str, default="examples/artifacts/mc_grad_multiarm.png")
    ap.add_argument("--store-data", action="store_true", help="Store experiment arrays and metadata to disk.")
    ap.add_argument("--tag", type=str, default="default", help="Tag used in stored data filename/metadata.")
    ap.add_argument("--data-dir", type=str, default="examples/data", help="Directory where experiment data is stored.")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.") from e

    try:
        from ordergrad.torch_backend import OrderStatTransform
    except Exception as e:  # pragma: no cover
        raise SystemExit("torch/ordergrad torch backend is required for this example. Install with `pip install torch`.") from e

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    dtype = torch.float64

    m = int(args.num_arms)
    if m < 2:
        raise SystemExit("--num-arms must be >= 2")

    k_ord = int(torch.floor(torch.tensor(args.k)).item())
    if k_ord < 1:
        raise SystemExit("Need floor(k) >= 1")

    r = _make_rewards(m, args.reward_mode, device=device, dtype=dtype)
    theta = _init_theta(m, args.prob_mode, device=device, dtype=dtype)
    p = torch.softmax(theta, dim=0)
    a = _parse_a(args.a, k_ord, device=device, dtype=dtype)

    os_batch = OrderStatTransform.precompute(args.N, args.k, dtype=dtype, compute_conditional=True, compute_leave_one_out=True)
    os_batch_l = os_batch.with_lstat_weights(a)
    os_exact = OrderStatTransform.precompute(max(args.N, 2), args.k, dtype=dtype, compute_conditional=False, compute_leave_one_out=False)

    # Exact gradient via known-(r,p) objective + autograd.
    f_exact = os_exact.lstat_known_rp(r, p, a)
    g_exact = torch.autograd.grad(f_exact, theta, retain_graph=False, create_graph=False)[0].detach()

    sampler = BufferedIndexSampler(p.detach(), buffer_size=args.sample_buffer_size, device=device)
    t_grid = [int(x) for x in args.t_grid.split(",") if x.strip()]
    if any(t <= 0 for t in t_grid):
        raise SystemExit("All t-grid entries must be positive.")

    abs_err = []
    rel_err = []
    g_last = None


    for t in t_grid:
        g_sum = torch.zeros(m, dtype=dtype, device=device)
        for _ in range(t):
            idx = sampler.sample(args.N)
            x = r[idx]
            l_adv = os_batch_l.lstat_advantage(x)  # (N,)

            # score-function term from autograd: d/dtheta log p(idx_n)
            logp = torch.log_softmax(theta, dim=0)

            # Build mean_n [ l_adv_n * grad_theta log p(idx_n) ] using autograd
            weighted_score = torch.zeros_like(theta)
            for n in range(args.N):
                g_n = torch.autograd.grad(logp[idx[n]], theta, retain_graph=True, create_graph=False)[0]
                weighted_score = weighted_score + l_adv[n].detach() * g_n
            g_batch = (float(args.k) * weighted_score) / float(args.N)

            g_sum = g_sum + g_batch

        g_mc = g_sum / float(t)
        g_last = g_mc

        ae = torch.mean(torch.abs(g_mc - g_exact)).item()
        re = torch.mean(torch.abs(g_mc - g_exact) / (torch.abs(g_exact) + 1e-12)).item()
        abs_err.append(float(ae))
        rel_err.append(float(re))

    g_exact_np = g_exact.detach().cpu().numpy()
    g_last_np = g_last.detach().cpu().numpy() if g_last is not None else None

    abs_err_plot = _safe_for_logplot(abs_err)
    rel_err_plot = _safe_for_logplot(rel_err)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].plot(t_grid, abs_err_plot, marker="o")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Multi-arm LR gradient: absolute error")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("mean abs error")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(t_grid, rel_err_plot, marker="s")
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

    if g_last_np is not None:
        comp_out = out.with_name(out.stem + "_components" + out.suffix)
        fig2, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(range(m), g_exact_np, marker="o", label="exact (autograd)")
        ax.plot(range(m), g_last_np, marker="x", label=f"MC LR estimate (t={t_grid[-1]})")
        ax.set_xlabel("logit index")
        ax.set_ylabel("gradient")
        ax.set_title("Gradient components")
        ax.grid(alpha=0.3)
        ax.legend()
        fig2.tight_layout()
        fig2.savefig(comp_out, dpi=150)
        print(f"Saved: {comp_out}")

    pdf_out = out.with_suffix(".pdf")
    fig.savefig(pdf_out)
    print(f"Saved: {pdf_out}")
    if g_last_np is not None:
        pdf_comp_out = comp_out.with_suffix(".pdf")
        fig2.savefig(pdf_comp_out)
        print(f"Saved: {pdf_comp_out}")

    if args.store_data:
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        stem = f"mc_gradients_multiarm__{args.tag}"
        npz_path = data_dir / f"{stem}.npz"
        json_path = data_dir / f"{stem}.json"
        np.savez(
            npz_path,
            t=np.asarray(t_grid, dtype=np.int64),
            abs_err=np.asarray(abs_err, dtype=np.float64),
            rel_err=np.asarray(rel_err, dtype=np.float64),
            g_exact=g_exact_np,
            g_last=(g_last_np if g_last_np is not None else np.zeros_like(g_exact_np)),
        )
        metadata = {
            "experiment": "mc_gradients_multiarm",
            "tag": args.tag,
            "setup": {
                "N": int(args.N),
                "k": float(args.k),
                "num_arms": int(args.num_arms),
                "reward_mode": args.reward_mode,
                "prob_mode": args.prob_mode,
                "a": args.a,
                "seed": int(args.seed),
                "t_grid": t_grid,
            },
            "artifacts": {
                "plot_png": str(out),
                "plot_pdf": str(pdf_out),
                "components_png": str(comp_out) if g_last_np is not None else None,
                "components_pdf": str(comp_out.with_suffix('.pdf')) if g_last_np is not None else None,
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
