import itertools

import numpy as np
import pytest

from ordergrad.numpy_backend import OrderStatTransform


def _exact_orderstat_means(x: np.ndarray, k: int) -> np.ndarray:
    """Exact E[X_(j:k)] by complete enumeration over all k-subsets."""
    N = x.shape[0]
    all_vals = []
    for subset in itertools.combinations(range(N), k):
        all_vals.append(np.sort(x[list(subset)]))
    return np.mean(np.asarray(all_vals, dtype=np.float64), axis=0)


def _exact_conditional_included_means(x: np.ndarray, i: int, k: int) -> np.ndarray:
    """Exact E[X_(j:k) | i included] by full enumeration."""
    N = x.shape[0]
    others = [j for j in range(N) if j != i]
    vals = []
    for subset_other in itertools.combinations(others, k - 1):
        subset = (i, *subset_other)
        vals.append(np.sort(x[list(subset)]))
    return np.mean(np.asarray(vals, dtype=np.float64), axis=0)


def _exact_leave_one_out_means(x: np.ndarray, i: int, k: int) -> np.ndarray:
    """Exact E[X_(j:k)] with i removed, by full enumeration."""
    N = x.shape[0]
    others = [j for j in range(N) if j != i]
    vals = []
    for subset in itertools.combinations(others, k):
        vals.append(np.sort(x[list(subset)]))
    return np.mean(np.asarray(vals, dtype=np.float64), axis=0)


def test_exact_matches_full_enumeration_on_small_problem():
    """Cross-check all APIs against exact combinatorial enumeration."""
    x = np.array([2.0, -1.5, 4.0, 0.25, 3.0], dtype=np.float64)
    N, k = x.size, 3

    os = OrderStatTransform.precompute(
        N,
        k,
        dtype=np.float64,
        compute_conditional=True,
        compute_leave_one_out=True,
    )

    np.testing.assert_allclose(os.expected_orderstats(x), _exact_orderstat_means(x, k), atol=1e-12, rtol=1e-12)

    E_inc = os.expected_orderstats_inclusion(x)
    E_loo = os.expected_orderstats_leave_one_out(x)

    for i in range(N):
        np.testing.assert_allclose(E_inc[i], _exact_conditional_included_means(x, i, k), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(E_loo[i], _exact_leave_one_out_means(x, i, k), atol=1e-12, rtol=1e-12)


def test_edge_case_k_equals_1():
    """If k=1, the only order statistic is the sampled element itself."""
    x = np.array([3.0, -2.0, 1.0, 7.0], dtype=np.float64)
    N, k = x.size, 1

    os = OrderStatTransform.precompute(
        N,
        k,
        dtype=np.float64,
        compute_conditional=True,
        compute_leave_one_out=True,
    )

    # Unconditional: sample one element uniformly -> mean(x)
    np.testing.assert_allclose(os.expected_orderstats(x), [x.mean()], atol=1e-12, rtol=1e-12)

    # Conditional inclusion: forced sample is {i} -> exactly x[i]
    E_inc = os.expected_orderstats_inclusion(x)
    np.testing.assert_allclose(E_inc[:, 0], x, atol=1e-12, rtol=1e-12)

    # Leave-one-out: mean of remaining N-1 elements
    E_loo = os.expected_orderstats_leave_one_out(x)
    expected = np.array([(np.sum(x) - x[i]) / (N - 1) for i in range(N)], dtype=np.float64)
    np.testing.assert_allclose(E_loo[:, 0], expected, atol=1e-12, rtol=1e-12)


def test_edge_case_k_equals_N_for_unconditional_and_conditional():
    """If k=N, the sampled set is always the full set, so order stats are deterministic."""
    x = np.array([5.0, -1.0, 2.5], dtype=np.float64)
    N, k = x.size, x.size

    os = OrderStatTransform.precompute(
        N,
        k,
        dtype=np.float64,
        compute_conditional=True,
        compute_leave_one_out=False,  # leave-one-out not defined when k=N
    )

    x_sorted = np.sort(x)
    np.testing.assert_allclose(os.expected_orderstats(x), x_sorted, atol=1e-12, rtol=1e-12)

    E_inc = os.expected_orderstats_inclusion(x)
    expected = np.tile(x_sorted, (N, 1))
    np.testing.assert_allclose(E_inc, expected, atol=1e-12, rtol=1e-12)


def test_lstat_identities():
    """L-statistic helpers should match explicit combinations of order-stat expectations."""
    rng = np.random.default_rng(0)
    N, k = 12, 5
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N, dtype=np.float64)
    a = rng.normal(size=k).astype(np.float64)

    os = OrderStatTransform.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)

    E = os.expected_orderstats(x)
    E_inc = os.expected_orderstats_inclusion(x)
    E_loo = os.expected_orderstats_leave_one_out(x)

    np.testing.assert_allclose(os.expected_lstat(x, a), float(E @ a), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(os.expected_lstat_inclusion(x, a), E_inc @ a, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(os.expected_lstat_leave_one_out(x, a), E_loo @ a, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(os.expected_lstat_advantage(x, a), (E_inc @ a) - (E_loo @ a), atol=1e-12, rtol=1e-12)


def test_preweighted_lstat_and_direct_advantage_match_baseline():
    rng = np.random.default_rng(123)
    N, k = 11, 4
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N)
    a = rng.normal(size=k).astype(np.float64)

    base = OrderStatTransform.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    weighted = base.with_lstat_weights(a)

    np.testing.assert_allclose(weighted.expected_lstat(x), base.expected_lstat(x, a), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(weighted.expected_lstat_inclusion(x), base.expected_lstat_inclusion(x, a), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(weighted.expected_lstat_leave_one_out(x), base.expected_lstat_leave_one_out(x, a), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(weighted.expected_lstat_advantage(x), base.expected_lstat_advantage(x, a), atol=1e-12, rtol=1e-12)

    adv_os = base.expected_orderstats_advantage(x)
    np.testing.assert_allclose(adv_os, base.expected_orderstats_inclusion(x) - base.expected_orderstats_leave_one_out(x), atol=1e-12, rtol=1e-12)


def test_dense_matmul_path_matches_efficient_path():
    rng = np.random.default_rng(456)
    N, k = 10, 4
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N, dtype=np.float64)
    a = rng.normal(size=k).astype(np.float64)

    dense = OrderStatTransform.precompute(
        N,
        k,
        dtype=np.float64,
        compute_conditional=True,
        compute_leave_one_out=True,
        compute_dense_matrices=True,
    )

    np.testing.assert_allclose(
        dense.expected_orderstats_inclusion(x, method="matmul"),
        dense.expected_orderstats_inclusion(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        dense.expected_orderstats_leave_one_out(x, method="matmul"),
        dense.expected_orderstats_leave_one_out(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        dense.expected_orderstats_advantage(x, method="matmul"),
        dense.expected_orderstats_advantage(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )

    weighted = dense.with_lstat_weights(a)
    np.testing.assert_allclose(
        weighted.expected_lstat_inclusion(x, method="matmul"),
        weighted.expected_lstat_inclusion(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        weighted.expected_lstat_leave_one_out(x, method="matmul"),
        weighted.expected_lstat_leave_one_out(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        weighted.expected_lstat_advantage(x, method="matmul"),
        weighted.expected_lstat_advantage(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )


def test_real_kappa_matches_integer_k_when_equal():
    rng = np.random.default_rng(7)
    N, k = 9, 4
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N, dtype=np.float64)

    os_int = OrderStatTransform.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    os_real = OrderStatTransform.precompute(N, k, kappa=float(k), dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)

    np.testing.assert_allclose(os_real.expected_orderstats(x), os_int.expected_orderstats(x), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(os_real.expected_orderstats_inclusion(x), os_int.expected_orderstats_inclusion(x), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(os_real.expected_orderstats_leave_one_out(x), os_int.expected_orderstats_leave_one_out(x), atol=1e-12, rtol=1e-12)


def test_real_kappa_fractional_is_supported_and_finite():
    rng = np.random.default_rng(8)
    N, k = 10, 4
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N, dtype=np.float64)
    a = rng.normal(size=k).astype(np.float64)

    os_frac = OrderStatTransform.precompute(
        N,
        k,
        kappa=4.3,
        dtype=np.float64,
        compute_conditional=True,
        compute_leave_one_out=True,
        compute_dense_matrices=True,
    ).with_lstat_weights(a)

    E = os_frac.expected_orderstats(x)
    E_inc = os_frac.expected_orderstats_inclusion(x, method="matmul")
    E_loo = os_frac.expected_orderstats_leave_one_out(x, method="matmul")
    Adv = os_frac.expected_orderstats_advantage(x, method="matmul")
    l_adv = os_frac.expected_lstat_advantage(x, method="matmul")

    assert np.isfinite(E).all()
    assert np.isfinite(E_inc).all()
    assert np.isfinite(E_loo).all()
    assert np.isfinite(Adv).all()
    assert np.isfinite(l_adv).all()


def test_matmul_method_fallback_and_auto_selection_behaviors():
    rng = np.random.default_rng(99)
    N, k = 9, 3
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N, dtype=np.float64)
    a = rng.normal(size=k).astype(np.float64)

    no_dense = OrderStatTransform.precompute(
        N,
        k,
        dtype=np.float64,
        compute_conditional=True,
        compute_leave_one_out=True,
        compute_dense_matrices=False,
    )
    dense = OrderStatTransform.precompute(
        N,
        k,
        dtype=np.float64,
        compute_conditional=True,
        compute_leave_one_out=True,
        compute_dense_matrices=True,
    )

    # matmul should gracefully fall back to efficient when dense matrices are not precomputed.
    np.testing.assert_allclose(
        no_dense.expected_orderstats_inclusion(x, method="matmul"),
        no_dense.expected_orderstats_inclusion(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        no_dense.expected_orderstats_leave_one_out(x, method="matmul"),
        no_dense.expected_orderstats_leave_one_out(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        no_dense.expected_orderstats_advantage(x, method="matmul"),
        no_dense.expected_orderstats_advantage(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )

    # auto should choose the available dense path when present.
    np.testing.assert_allclose(
        dense.expected_orderstats_inclusion(x, method="auto"),
        dense.expected_orderstats_inclusion(x, method="matmul"),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        dense.expected_orderstats_leave_one_out(x, method="auto"),
        dense.expected_orderstats_leave_one_out(x, method="matmul"),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        dense.expected_orderstats_advantage(x, method="auto"),
        dense.expected_orderstats_advantage(x, method="matmul"),
        atol=1e-12,
        rtol=1e-12,
    )

    # Explicit a with matmul should work and agree with efficient mode.
    np.testing.assert_allclose(
        dense.expected_lstat_inclusion(x, a, method="matmul"),
        dense.expected_lstat_inclusion(x, a, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        dense.expected_lstat_leave_one_out(x, a, method="matmul"),
        dense.expected_lstat_leave_one_out(x, a, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        dense.expected_lstat_advantage(x, a, method="matmul"),
        dense.expected_lstat_advantage(x, a, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )
