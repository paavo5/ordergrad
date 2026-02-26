#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot variance/SNR against number of arms from stored mc_snr_multiarm npz files.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tag-prefix", default="mc_snr_multiarm__snr_multiarm_fixN_varyarms_")
    ap.add_argument("--output", default="examples/artifacts/snr_multiarm_fixN_varyarms_combined.png")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.") from e

    files = sorted(Path(args.data_dir).glob(f"{args.tag_prefix}*.npz"))
    if not files:
        raise SystemExit(f"No files found matching {args.tag_prefix}*.npz")

    arms, var, snr = [], [], []
    for f in files:
        n = int(f.stem.split("_")[-1])
        d = np.load(f)
        arms.append(n)
        var.append(float(np.asarray(d["variance"])[0]))
        snr.append(float(np.asarray(d["snr"])[0]))

    order = np.argsort(np.asarray(arms))
    arms = np.asarray(arms)[order]
    var = np.asarray(var)[order]
    snr = np.asarray(snr)[order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].plot(arms, var, marker="o")
    axes[0].set_xlabel("number of arms")
    axes[0].set_ylabel("variance")
    axes[0].set_title("Variance vs number of arms")
    axes[0].grid(alpha=0.3)

    axes[1].plot(arms, snr, marker="o")
    axes[1].set_xlabel("number of arms")
    axes[1].set_ylabel("SNR")
    axes[1].set_title("SNR vs number of arms")
    axes[1].grid(alpha=0.3)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
