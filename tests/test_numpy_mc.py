import numpy as np
import pytest

from ordergrad.numpy_backend import OrderStatTransform


def _uniform_k_subset_indices(N: int, k: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorized uniform k-subset sampling without replacement.

    Generate i.i.d. continuous keys per trial and take indices of the k smallest keys.
    This is equivalent to taking the first k elements of a random permutation.
    """
    keys = rng.random((T, N))
    idx = np.argpartition(keys, kth=k - 1, axis=1)[:, :k]
    return idx


def _mc_orderstats_unconditional(x: np.ndarray, k: int, T: int, rng: np.random.Generator):
    N = x.shape[0]
    idx = _uniform_k_subset_indices(N, k, T, rng)
    vals = np.sort(x[idx], axis=1)  # (T,k)
    return vals.mean(axis=0), vals.std(axis=0, ddof=1)


def _mc_orderstats_cond_include(x: np.ndarray, i: int, k: int, T: int, rng: np.random.Generator):
    N = x.shape[0]
    if k == 1:
        mean = np.array([x[i]], dtype=np.float64)
        std = np.array([0.0], dtype=np.float64)
        return mean, std

    others = np.delete(np.arange(N, dtype=np.int64), i)
    keys = rng.random((T, N - 1))
    idx_other = np.argpartition(keys, kth=k - 2, axis=1)[:, : k - 1]
    idx = others[idx_other]

    vals = np.concatenate([np.full((T, 1), x[i], dtype=np.float64), x[idx].astype(np.float64)], axis=1)
    vals = np.sort(vals, axis=1)
    return vals.mean(axis=0), vals.std(axis=0, ddof=1)


def _mc_orderstats_leave_one_out(x: np.ndarray, i: int, k: int, T: int, rng: np.random.Generator):
    N = x.shape[0]
    others = np.delete(np.arange(N, dtype=np.int64), i)
    keys = rng.random((T, N - 1))
    idx_other = np.argpartition(keys, kth=k - 1, axis=1)[:, :k]
    idx = others[idx_other]

    vals = np.sort(x[idx].astype(np.float64), axis=1)
    return vals.mean(axis=0), vals.std(axis=0, ddof=1)


def _assert_close_mc(analytic: np.ndarray, mc_mean: np.ndarray, mc_std: np.ndarray, T: int, nsig: float = 4.0):
    analytic = np.asarray(analytic, dtype=np.float64)
    mc_mean = np.asarray(mc_mean, dtype=np.float64)
    mc_std = np.asarray(mc_std, dtype=np.float64)

    stderr = mc_std / np.sqrt(T)
    tol = np.maximum(nsig * stderr, 2e-12)

    diff = np.abs(analytic - mc_mean)
    if not np.all(diff <= tol):
        j_bad = int(np.argmax(diff - tol))
        raise AssertionError(
            f"MC check failed at j={j_bad+1}: diff={diff[j_bad]:.6g}, tol={tol[j_bad]:.6g}. "
            f"(max diff={diff.max():.6g}, max tol={tol.max():.6g})"
        )


@pytest.mark.parametrize("N,k,T,seed", [(25, 7, 20000, 0)])
def test_unconditional_orderstats_matches_monte_carlo(N, k, T, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N, dtype=np.float64)

    os = OrderStatTransform.precompute(
        N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True
    )
    analytic = os.expected_orderstats(x)
    mc_mean, mc_std = _mc_orderstats_unconditional(x, k, T, rng)

    _assert_close_mc(analytic, mc_mean, mc_std, T)


def test_conditional_included_orderstats_matches_monte_carlo():
    N, k, T, seed = 30, 8, 15000, 123
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N, dtype=np.float64)

    os = OrderStatTransform.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=False)
    E_inc = os.expected_orderstats_inclusion(x)  # (N,k)

    for i in [0, N // 2, N - 1, 7]:
        mc_mean, mc_std = _mc_orderstats_cond_include(x, i, k, T, rng)
        _assert_close_mc(E_inc[i], mc_mean, mc_std, T)


def test_leave_one_out_orderstats_matches_monte_carlo():
    N, k, T, seed = 28, 6, 18000, 999
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N, dtype=np.float64)

    os = OrderStatTransform.precompute(N, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=True)
    E_loo = os.expected_orderstats_leave_one_out(x)

    for i in [0, N // 2, N - 1, 5]:
        mc_mean, mc_std = _mc_orderstats_leave_one_out(x, i, k, T, rng)
        _assert_close_mc(E_loo[i], mc_mean, mc_std, T)


def _sample_known_rp_batch(r: np.ndarray, p: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    arms = rng.choice(len(r), size=N, replace=True, p=p)
    return r[arms], arms


def test_known_rp_matches_monte_carlo_unconditional_and_conditional():
    rng = np.random.default_rng(777)
    r = np.array([-1.2, -0.2, 0.8, 2.3], dtype=np.float64)
    p = np.array([0.15, 0.35, 0.30, 0.20], dtype=np.float64)
    k = 4
    T = 30000

    os = OrderStatTransform.precompute(20, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    v = os.expected_orderstats_known_rp(r, p)
    q = os.expected_orderstats_inclusion_known_rp(r, p)

    keys = rng.choice(len(r), size=(T, k), replace=True, p=p)
    samples = np.sort(r[keys], axis=1)
    mc_v = samples.mean(axis=0)
    mc_v_std = samples.std(axis=0, ddof=1)
    _assert_close_mc(v, mc_v, mc_v_std, T, nsig=4.0)

    for b in range(len(r)):
        keys_b = rng.choice(len(r), size=(T, k - 1), replace=True, p=p)
        samp_b = np.sort(np.concatenate([np.full((T, 1), r[b]), r[keys_b]], axis=1), axis=1)
        mc_q = samp_b.mean(axis=0)
        mc_q_std = samp_b.std(axis=0, ddof=1)
        _assert_close_mc(q[b], mc_q, mc_q_std, T, nsig=4.0)


def test_batch_advantage_matches_known_rp_advantage_in_expectation():
    rng = np.random.default_rng(888)
    r = np.array([-1.0, 0.0, 0.7, 1.9], dtype=np.float64)
    p = np.array([0.20, 0.30, 0.35, 0.15], dtype=np.float64)
    N, k, B = 24, 5, 1200

    os_batch = OrderStatTransform.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    os_known = OrderStatTransform.precompute(N, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    adv_exact = os_known.expected_orderstats_advantage_known_rp(r, p)  # (m,k)

    sum_adv = np.zeros_like(adv_exact)
    sumsq_adv = np.zeros_like(adv_exact)
    cnt = np.zeros((len(r), 1), dtype=np.int64)

    for _ in range(B):
        x, arms = _sample_known_rp_batch(r, p, N, rng)
        adv_i = os_batch.expected_orderstats_advantage(x)  # (N,k)
        for b in range(len(r)):
            mask = arms == b
            if np.any(mask):
                vals = adv_i[mask]
                sum_adv[b] += vals.sum(axis=0)
                sumsq_adv[b] += (vals * vals).sum(axis=0)
                cnt[b, 0] += int(vals.shape[0])

    n = np.maximum(cnt, 1)
    est = sum_adv / n
    ex2 = sumsq_adv / n
    var = np.maximum(ex2 - est * est, 0.0)
    stderr = np.sqrt(var / n)
    tol = 4.0 * stderr

    diff = np.abs(est - adv_exact)
    if not np.all(diff <= tol):
        bad = np.unravel_index(np.argmax(diff - tol), diff.shape)
        raise AssertionError(
            f"Advantage MC check failed at arm={bad[0]}, j={bad[1]+1}: "
            f"diff={diff[bad]:.6g}, tol={tol[bad]:.6g}."
        )


def test_sampling_fractional_k_raises_with_suggestion():
    with pytest.raises(ValueError, match=r"Fractional k=4\.7"):
        OrderStatTransform.precompute(
            20,
            4.7,
            dtype=np.float64,
            compute_conditional=False,
            compute_leave_one_out=False,
        )


def test_known_rp_fractional_k_is_branch_aware_and_distinct_from_floor_k():
    r = np.array([-1.2, -0.2, 0.8, 2.3], dtype=np.float64)
    p = np.array([0.15, 0.35, 0.30, 0.20], dtype=np.float64)

    k_real = 4.7
    k_floor = int(np.floor(k_real))

    os_real = OrderStatTransform.for_known_rp(k_real, dtype=np.float64)
    os_floor = OrderStatTransform.for_known_rp(k_floor, dtype=np.float64)

    # Fractional full order-statistics are ambiguous without a branch.
    with pytest.raises(ValueError, match="branch"):
        os_real.expected_orderstats_known_rp(r, p)

    v_floor = os_floor.expected_orderstats_known_rp(r, p)

    v_bottom = os_real.expected_orderstats_known_rp(r, p, branch="bottom")
    v_top = os_real.expected_orderstats_known_rp(r, p, branch="top")

    assert v_bottom.shape == (k_floor,)
    assert v_top.shape == (k_floor,)

    assert np.all(np.isfinite(v_bottom))
    assert np.all(np.isfinite(v_top))

    # Fractional beta branches should not collapse to floor(k).
    assert not np.allclose(v_bottom, v_floor)
    assert not np.allclose(v_top, v_floor)

    mean = float(np.dot(r, p))

    remin = os_real.expected_lstat_known_rp(r, p, "ReMin")
    remax = os_real.expected_lstat_known_rp(r, p, "ReMax")

    # Direction sanity checks.
    assert remin < mean
    assert remax > mean

    # Preset branch inference should match the corresponding beta branch.
    np.testing.assert_allclose(remin, v_bottom[0], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(remax, v_top[-1], atol=1e-12, rtol=1e-12)
