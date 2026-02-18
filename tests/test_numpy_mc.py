import numpy as np
import pytest

from orderstat_reward.numpy_backend import OrderStatTransform


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


def _assert_close_mc(analytic: np.ndarray, mc_mean: np.ndarray, mc_std: np.ndarray, T: int, scale: float, nsig: float = 6.0):
    analytic = np.asarray(analytic, dtype=np.float64)
    mc_mean = np.asarray(mc_mean, dtype=np.float64)
    mc_std = np.asarray(mc_std, dtype=np.float64)

    stderr = mc_std / np.sqrt(T)
    tol = nsig * stderr + 1e-3 * scale

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

    scale = float(np.ptp(x) + 1.0)
    _assert_close_mc(analytic, mc_mean, mc_std, T, scale)


def test_conditional_included_orderstats_matches_monte_carlo():
    N, k, T, seed = 30, 8, 15000, 123
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N, dtype=np.float64)

    os = OrderStatTransform.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=False)
    E_inc = os.expected_orderstats_inclusion(x)  # (N,k)

    scale = float(np.ptp(x) + 1.0)
    for i in [0, N // 2, N - 1, 7]:
        mc_mean, mc_std = _mc_orderstats_cond_include(x, i, k, T, rng)
        _assert_close_mc(E_inc[i], mc_mean, mc_std, T, scale)


def test_leave_one_out_orderstats_matches_monte_carlo():
    N, k, T, seed = 28, 6, 18000, 999
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N, dtype=np.float64)

    os = OrderStatTransform.precompute(N, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=True)
    E_loo = os.expected_orderstats_leave_one_out(x)

    scale = float(np.ptp(x) + 1.0)
    for i in [0, N // 2, N - 1, 5]:
        mc_mean, mc_std = _mc_orderstats_leave_one_out(x, i, k, T, rng)
        _assert_close_mc(E_loo[i], mc_mean, mc_std, T, scale)
