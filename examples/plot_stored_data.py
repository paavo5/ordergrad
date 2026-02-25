#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_meta(data_dir: Path):
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(data_dir.glob('*.json'))]


def main() -> None:
    ap = argparse.ArgumentParser(description='Plot all stored experiment data into png/pdf outputs.')
    ap.add_argument('--data-dir', type=str, default='examples/data')
    ap.add_argument('--output-dir', type=str, default='examples/artifacts/compiled')
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit('matplotlib is required. Install with `pip install matplotlib`.') from e

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = _load_meta(data_dir)

    # Aggregate SNR by experiment type
    for exp in ['mc_snr_multiarm', 'mc_snr_continuous']:
        entries = [m for m in metas if m.get('experiment') == exp]
        if not entries:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in entries:
            npz = np.load(m['artifacts']['data_npz'])
            ks = npz['k']
            if exp == 'mc_snr_multiarm':
                y = npz['snr']
                label = m.get('tag', 'untagged')
            else:
                y = npz['lr_snr']
                label = f"{m.get('tag','untagged')} (LR)"
            ax.plot(ks, y, marker='o', label=label)
        ax.set_xlabel('k')
        ax.set_ylabel('SNR')
        ax.set_title(f'Stored {exp} SNR curves')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        out_png = out_dir / f'{exp}_compiled.png'
        out_pdf = out_dir / f'{exp}_compiled.pdf'
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        fig.savefig(out_pdf)
        print(f'Saved: {out_png}')
        print(f'Saved: {out_pdf}')

    # Quantile accuracy aggregate
    q_entries = [m for m in metas if m.get('experiment') == 'quantile_estimator_accuracy']
    if q_entries:
        fig, axes = plt.subplots(1,2,figsize=(12,5))
        for m in q_entries:
            npz = np.load(m['artifacts']['data_npz'])
            t = npz['t']
            axes[0].plot(t, npz['q_abs_err'], marker='o', label=f"{m.get('tag')} Quantile")
            axes[0].plot(t, npz['hd_abs_err'], marker='x', label=f"{m.get('tag')} HD")
            axes[1].plot(t, npz['q_rel_err'], marker='o', label=f"{m.get('tag')} Quantile")
            axes[1].plot(t, npz['hd_rel_err'], marker='x', label=f"{m.get('tag')} HD")
        for ax in axes:
            ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
            ax.set_xlabel('t')
        axes[0].set_title('Quantile absolute error')
        axes[1].set_title('Quantile relative error')
        out_png = out_dir / 'quantile_accuracy_compiled.png'
        out_pdf = out_dir / 'quantile_accuracy_compiled.pdf'
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        fig.savefig(out_pdf)
        print(f'Saved: {out_png}')
        print(f'Saved: {out_pdf}')


if __name__ == '__main__':
    main()
