#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ordergrad.numpy_backend import OrderStatTransform


def _timeit(fn, repeats: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark runtime and save bar chart data.")
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--k", type=float, default=30)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--output", type=str, default="examples/artifacts/benchmark_runtime_bar.png")
    ap.add_argument("--tag", type=str, default="default")
    ap.add_argument("--data-dir", type=str, default="examples/data")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required. Install with `pip install matplotlib`.") from e

    rng = np.random.default_rng(0)
    x = rng.normal(size=args.N).astype(np.float64)
    k_ord = int(np.floor(args.k))
    a = np.linspace(0.2, 1.0, k_ord, dtype=np.float64)

    os = OrderStatTransform.precompute(args.N, args.k, dtype=np.float64, compute_dense_matrices=True)
    os_l = os.with_lstat_weights(a)

    tasks = [
        ("orderstats", lambda: os.expected_orderstats(x)),
        ("inclusion", lambda: os.expected_orderstats_inclusion(x, method="efficient")),
        ("advantage", lambda: os.expected_orderstats_advantage(x, method="efficient")),
        ("L-adv", lambda: os_l.expected_lstat_advantage(x, method="efficient")),
    ]
    names, ms = [], []
    for n, fn in tasks:
        names.append(n)
        ms.append(_timeit(fn, args.repeats) * 1e3)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(names, ms)
    ax.set_ylabel("avg runtime (ms)")
    ax.set_title(f"Runtime benchmark (N={args.N}, k={args.k})")
    ax.grid(axis="y", alpha=0.3)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    pdf_out = out.with_suffix('.pdf')
    fig.savefig(pdf_out)
    print(f"Saved: {out}")
    print(f"Saved: {pdf_out}")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    stem = f"benchmark_runtime_bar__{args.tag}"
    npz_path = data_dir / f"{stem}.npz"
    json_path = data_dir / f"{stem}.json"
    np.savez(npz_path, names=np.asarray(names), runtime_ms=np.asarray(ms, dtype=np.float64))
    json_path.write_text(json.dumps({"experiment": "benchmark_runtime_bar", "tag": args.tag, "setup": {"N": args.N, "k": args.k, "repeats": args.repeats}, "artifacts": {"plot_png": str(out), "plot_pdf": str(pdf_out), "data_npz": str(npz_path)}}, indent=2), encoding='utf-8')
    print(f"Saved: {npz_path}")
    print(f"Saved: {json_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
