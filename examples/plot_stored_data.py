#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def _load_meta(data_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(data_dir.glob('*.json')):
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _copy_recorded_artifacts(metas: list[dict[str, Any]], out_dir: Path) -> None:
    """Copy any plot artifacts already referenced by metadata to output dir."""
    copied: set[str] = set()
    for m in metas:
        arts = m.get('artifacts', {})
        if not isinstance(arts, dict):
            continue
        for v in arts.values():
            if not isinstance(v, str):
                continue
            src = Path(v)
            if src.suffix.lower() not in {'.png', '.pdf'}:
                continue
            if not src.exists():
                continue
            dst = out_dir / src.name
            if str(dst) in copied:
                continue
            shutil.copy2(src, dst)
            copied.add(str(dst))
            print(f'Saved: {dst}')


def _plot_snr_compiled(metas: list[dict[str, Any]], out_dir: Path, exp: str) -> None:
    entries = [m for m in metas if m.get('experiment') == exp]
    if not entries:
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in entries:
        npz = np.load(m['artifacts']['data_npz'])
        ks = npz['k']
        if exp == 'mc_snr_multiarm':
            y = npz['snr']
            label = m.get('tag', 'untagged')
        else:
            y = npz['lr_snr']
            label = f"{m.get('tag', 'untagged')} (LR)"
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


def _plot_quantile_accuracy_compiled(metas: list[dict[str, Any]], out_dir: Path) -> None:
    entries = [m for m in metas if m.get('experiment') == 'quantile_estimator_accuracy']
    if not entries:
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m in entries:
        npz = np.load(m['artifacts']['data_npz'])
        t = npz['t']

        # New schema: abs_err/rel_err arrays with leading method axis.
        if 'abs_err' in npz and 'rel_err' in npz:
            abs_err = np.asarray(npz['abs_err'])
            rel_err = np.asarray(npz['rel_err'])
            methods = np.asarray(npz['methods']).astype(str) if 'methods' in npz else np.array([f'method_{i}' for i in range(abs_err.shape[0])])
            for i, method in enumerate(methods):
                axes[0].plot(t, abs_err[i], marker='o', label=f"{m.get('tag')} {method}")
                axes[1].plot(t, rel_err[i], marker='x', label=f"{m.get('tag')} {method}")
        # Backward-compatible schema fallback.
        elif {'q_abs_err', 'hd_abs_err', 'q_rel_err', 'hd_rel_err'}.issubset(set(npz.files)):
            axes[0].plot(t, npz['q_abs_err'], marker='o', label=f"{m.get('tag')} Quantile")
            axes[0].plot(t, npz['hd_abs_err'], marker='x', label=f"{m.get('tag')} HD")
            axes[1].plot(t, npz['q_rel_err'], marker='o', label=f"{m.get('tag')} Quantile")
            axes[1].plot(t, npz['hd_rel_err'], marker='x', label=f"{m.get('tag')} HD")
        else:
            print(f"Warning: unsupported quantile npz schema for tag={m.get('tag')}")

    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
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


def main() -> None:
    ap = argparse.ArgumentParser(description='Plot all stored experiment data into png/pdf outputs.')
    ap.add_argument('--data-dir', type=str, default='examples/data')
    ap.add_argument('--output-dir', type=str, default='examples/artifacts/compiled')
    ap.add_argument('--copy-recorded-artifacts', action='store_true', default=True)
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit('matplotlib is required. Install with `pip install matplotlib`.') from e

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = _load_meta(data_dir)

    if args.copy_recorded_artifacts:
        _copy_recorded_artifacts(metas, out_dir)

    _plot_snr_compiled(metas, out_dir, 'mc_snr_multiarm')
    _plot_snr_compiled(metas, out_dir, 'mc_snr_continuous')
    _plot_quantile_accuracy_compiled(metas, out_dir)


if __name__ == '__main__':
    main()
