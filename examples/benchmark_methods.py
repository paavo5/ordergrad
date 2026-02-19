#!/usr/bin/env python3
"""Benchmark efficient vs matmul paths (with/without dense precompute).

Covers order-stats and L-stat advantage calls for repeated x vectors.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ordergrad.numpy_backend import OrderStatTransform


def timeit(fn, repeats: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark numpy backend methods.")
    p.add_argument("--N", type=int, default=400)
    p.add_argument("--k", type=float, default=40.0)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    k_ord = int(np.floor(args.k))
    x = rng.normal(size=args.N).astype(np.float64) + 1e-9 * np.arange(args.N)
    a = rng.normal(size=k_ord).astype(np.float64)

    t_pre_no = time.perf_counter()
    os_no = OrderStatTransform.precompute(args.N, args.k, dtype=np.float64, compute_dense_matrices=False)
    t_pre_no = time.perf_counter() - t_pre_no

    t_pre_dense = time.perf_counter()
    os_dense = OrderStatTransform.precompute(args.N, args.k, dtype=np.float64, compute_dense_matrices=True)
    t_pre_dense = time.perf_counter() - t_pre_dense

    benchmarks = [
        ("inc efficient (no-dense)", lambda: os_no.expected_orderstats_inclusion(x, method="efficient")),
        ("inc matmul fallback (no-dense)", lambda: os_no.expected_orderstats_inclusion(x, method="matmul")),
        ("inc efficient (dense)", lambda: os_dense.expected_orderstats_inclusion(x, method="efficient")),
        ("inc matmul (dense)", lambda: os_dense.expected_orderstats_inclusion(x, method="matmul")),
        ("adv efficient (dense)", lambda: os_dense.expected_orderstats_advantage(x, method="efficient")),
        ("adv matmul (dense)", lambda: os_dense.expected_orderstats_advantage(x, method="matmul")),
        ("l-adv efficient (dense)", lambda: os_dense.expected_lstat_advantage(x, a, method="efficient")),
        ("l-adv matmul (dense)", lambda: os_dense.expected_lstat_advantage(x, a, method="matmul")),
    ]

    print(f"N={args.N}, k={args.k} (floor={k_ord}), repeats={args.repeats}")
    print(f"precompute no-dense: {t_pre_no*1e3:.2f} ms")
    print(f"precompute dense:    {t_pre_dense*1e3:.2f} ms")
    print("\nPer-call runtime:")
    for name, fn in benchmarks:
        dt = timeit(fn, args.repeats)
        print(f"  {name:30s} {dt*1e3:9.3f} ms")


if __name__ == "__main__":
    main()
