#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot variance/SNR against dimensionality from stored mc_snr_continuous npz files.")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--tag-prefix", type=str, default="mc_snr_continuous__snr_cont_fixN_varydim_")
    ap.add_argument("--output", type=str, default="examples/artifacts/snr_cont_varydim_combined.png")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.") from e

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob(f"{args.tag_prefix}*.npz"))
    if not files:
        raise SystemExit(f"No files found matching {args.tag_prefix}*.npz in {data_dir}")

    dims, rp_var, lr_var, rp_snr, lr_snr = [], [], [], [], []
    for f in files:
        dim = int(f.stem.split("_")[-1])
        d = np.load(f)
        dims.append(dim)
        rp_var.append(float(np.asarray(d["rp_variance"])[0]))
        lr_var.append(float(np.asarray(d["lr_variance"])[0]))
        rp_snr.append(float(np.asarray(d["rp_snr"])[0]))
        lr_snr.append(float(np.asarray(d["lr_snr"])[0]))

    order = np.argsort(np.asarray(dims))
    dims = np.asarray(dims)[order]
    rp_var = np.asarray(rp_var)[order]
    lr_var = np.asarray(lr_var)[order]
    rp_snr = np.asarray(rp_snr)[order]
    lr_snr = np.asarray(lr_snr)[order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].plot(dims, rp_var, marker="o", label="RP Var[g]")
    axes[0].plot(dims, lr_var, marker="x", label="LR Var[g]")
    axes[0].set_xlabel("dimensionality")
    axes[0].set_ylabel("variance")
    axes[0].set_title("Variance vs dimensionality")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(dims, rp_snr, marker="o", label="RP SNR")
    axes[1].plot(dims, lr_snr, marker="x", label="LR SNR")
    axes[1].set_xlabel("dimensionality")
    axes[1].set_ylabel("SNR")
    axes[1].set_title("SNR vs dimensionality")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix('.pdf'))
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
