import itertools

import numpy as np
import pytest

from orderstat_reward.numpy_backend import OrderStatTransform


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
