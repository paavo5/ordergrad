#!/usr/bin/env python3
"""Benchmark ordergrad evaluation modes.

Key terms used in this benchmark:
- efficient mode:
  Uses the prefix/suffix-style algorithms (no dense rank-space operator matmul).
- matmul mode:
  Uses dense precomputed rank-space operators (when available) and einsum/matmul.
- no-dense precompute:
  Does NOT build dense operators `M_inc`, `M_loo`, `M_adv`.
- dense precompute:
  Builds dense operators so matmul mode can be used directly.

So, "efficient" is a runtime evaluation method, while "dense/no-dense" is a
precompute/storage choice.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Callable

import numpy as np


def timeit(fn: Callable[[], Any], repeats: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def _load_backend(name: str):
    if name == "np":
        from ordergrad.numpy_backend import OrderStatTransform

        return "NumPy", OrderStatTransform, lambda x: x, np.float64
    if name == "jax":
        import jax.numpy as jnp
        from ordergrad.jax_backend import OrderStatTransform

        return "JAX", OrderStatTransform, lambda x: jnp.asarray(x), jnp.float64
    if name == "torch":
        import torch
        from ordergrad.torch_backend import OrderStatTransform

        return "PyTorch", OrderStatTransform, lambda x: torch.tensor(x, dtype=torch.float64), torch.float64
    raise ValueError(f"Unsupported backend: {name}")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark ordergrad methods for NumPy/JAX/PyTorch.")
    p.add_argument("--backend", type=str, default="np", choices=["np", "jax", "torch"])
    p.add_argument("--N", type=int, default=400)
    p.add_argument("--k", type=float, default=40.0)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--efficient",
        action="store_true",
        help=(
            "Run only efficient-mode computations with no dense precompute. "
            "Skips dense/matmul benchmarks and related precompute timings."
        ),
    )
    args = p.parse_args()

    backend_name, OrderStatTransform, to_backend, dtype = _load_backend(args.backend)

    rng = np.random.default_rng(args.seed)
    k_ord = int(np.floor(args.k))
    x_np = rng.normal(size=args.N).astype(np.float64) + 1e-9 * np.arange(args.N)
    a_np = rng.normal(size=k_ord).astype(np.float64)
    x = to_backend(x_np)
    a = to_backend(a_np)

    print("=" * 88)
    print(f"Backend: {backend_name}")
    print(f"Estimator parameters: N={args.N}, k={args.k} (floor(k)={k_ord})")
    print(f"Repeats per benchmark: {args.repeats}")
    print(f"Efficient-only mode: {args.efficient}")
    print("Definitions:")
    print("  - efficient = prefix/suffix computation path (does not require dense M_* tensors).")
    print("  - matmul    = dense operator path using precomputed M_inc/M_loo/M_adv tensors.")
    print("  - no-dense  = precompute does not build M_*; matmul requests fall back to efficient.")
    print("  - dense     = precompute builds M_*; matmul can run directly.")
    print("inc = inclusion-conditional order stats, adv = inclusion - leave-one-out.")
    print("L-inc/L-adv compare full-order-stat computation vs direct preweighted-a computation.")
    print("=" * 88)

    t_pre_no = time.perf_counter()
    os_no = OrderStatTransform.precompute(args.N, args.k, dtype=dtype, compute_dense_matrices=False)
    t_pre_no = time.perf_counter() - t_pre_no

    t_pre_lstat_no = time.perf_counter()
    os_lstat_no = os_no.with_lstat_weights(a)
    t_pre_lstat_no = time.perf_counter() - t_pre_lstat_no

    benchmarks: list[tuple[str, Callable[[], Any]]] = [
        ("orderstats unconditional (always W-based)", lambda: os_no.expected_orderstats(x)),
        ("inc efficient (no-dense)", lambda: os_no.expected_orderstats_inclusion(x, method="efficient")),
        ("adv efficient (no-dense)", lambda: os_no.expected_orderstats_advantage(x, method="efficient")),
        ("L-adv efficient (no-dense; a passed)", lambda: os_no.expected_lstat_advantage(x, a, method="efficient")),
        ("L-adv direct efficient (no-dense; prew)", lambda: os_lstat_no.expected_lstat_advantage(x, method="efficient")),
        ("L-inc full+dot efficient (no-dense)", lambda: os_no.expected_orderstats_inclusion(x, method="efficient") @ a),
        ("L-inc direct efficient (no-dense; prew)", lambda: os_lstat_no.expected_lstat_inclusion(x, method="efficient")),
        ("L-uncond full+dot (no-dense)", lambda: os_no.expected_orderstats(x) @ a),
        ("L-uncond direct preweighted (no-dense)", lambda: os_lstat_no.expected_lstat(x)),
    ]

    t_pre_dense = None
    t_pre_lstat = None
    if not args.efficient:
        t_pre_dense = time.perf_counter()
        os_dense = OrderStatTransform.precompute(args.N, args.k, dtype=dtype, compute_dense_matrices=True)
        t_pre_dense = time.perf_counter() - t_pre_dense

        t_pre_lstat = time.perf_counter()
        os_lstat_dense = os_dense.with_lstat_weights(a)
        t_pre_lstat = time.perf_counter() - t_pre_lstat

        # Full-order-stat vs direct-preweighted comparisons:
        # - "full+dot": compute full vectors/matrices then contract with a each call.
        # - "direct": use with_lstat_weights(a) so contractions are precomputed once.
        benchmarks.extend(
            [
                ("inc matmul request (no-dense fallback)", lambda: os_no.expected_orderstats_inclusion(x, method="matmul")),
                ("inc efficient (dense)", lambda: os_dense.expected_orderstats_inclusion(x, method="efficient")),
                ("inc matmul (dense)", lambda: os_dense.expected_orderstats_inclusion(x, method="matmul")),
                ("adv efficient (dense)", lambda: os_dense.expected_orderstats_advantage(x, method="efficient")),
                ("adv matmul (dense)", lambda: os_dense.expected_orderstats_advantage(x, method="matmul")),
                ("L-adv (dense; a passed each call)", lambda: os_dense.expected_lstat_advantage(x, a, method="efficient")),
                ("L-adv matmul (dense; a passed)", lambda: os_dense.expected_lstat_advantage(x, a, method="matmul")),
                ("L-adv direct (dense; preweighted)", lambda: os_lstat_dense.expected_lstat_advantage(x, method="efficient")),
                ("L-adv direct matmul (dense; prew)", lambda: os_lstat_dense.expected_lstat_advantage(x, method="matmul")),
                ("L-inc full+dot (dense)", lambda: os_dense.expected_orderstats_inclusion(x, method="efficient") @ a),
                ("L-inc direct (dense; preweighted)", lambda: os_lstat_dense.expected_lstat_inclusion(x, method="efficient")),
                ("L-inc full+dot matmul (dense)", lambda: os_dense.expected_orderstats_inclusion(x, method="matmul") @ a),
                ("L-inc direct matmul (dense; prew)", lambda: os_lstat_dense.expected_lstat_inclusion(x, method="matmul")),
                ("L-uncond full+dot", lambda: os_dense.expected_orderstats(x) @ a),
                ("L-uncond direct preweighted", lambda: os_lstat_dense.expected_lstat(x)),
            ]
        )

    print(f"Precompute time (no-dense):           {t_pre_no*1e3:10.3f} ms")
    print(f"Precompute time (no-dense+lstat a):   {t_pre_lstat_no*1e3:10.3f} ms")
    if t_pre_dense is not None and t_pre_lstat is not None:
        print(f"Precompute time (dense):              {t_pre_dense*1e3:10.3f} ms")
        print(f"Precompute time (dense+lstat a):      {t_pre_lstat*1e3:10.3f} ms")
    else:
        print("Precompute time (dense):              skipped (--efficient enabled)")
        print("Precompute time (dense+lstat a):      skipped (--efficient enabled)")

    print("\nAverage per-call runtime:")
    for name, fn in benchmarks:
        dt = timeit(fn, args.repeats)
        print(f"  - {name:40s}: {dt*1e3:10.3f} ms")


if __name__ == "__main__":
    main()
