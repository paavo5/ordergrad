import numpy as np
import pytest

from orderstat_reward.numpy_backend import OrderStatTransform as NP


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# Enable float64 for stable cross-backend comparisons.
jax.config.update("jax_enable_x64", True)

from orderstat_reward.jax_backend import OrderStatTransform as JX


def _rand_x_no_ties(rng: np.random.Generator, N: int) -> np.ndarray:
    x = rng.normal(size=N).astype(np.float64)
    return x + 1e-6 * np.arange(N, dtype=np.float64)


@pytest.mark.jax
@pytest.mark.parametrize("N,k", [(40, 9), (25, 5)])
def test_jax_matches_numpy(N, k):
    rng = np.random.default_rng(1)
    x_np = _rand_x_no_ties(rng, N)

    os_np = NP.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    os_jx = JX.precompute(N, k, dtype=jnp.float64, compute_conditional=True, compute_leave_one_out=True)

    x_jx = jnp.asarray(x_np, dtype=jnp.float64)

    E_np = os_np.expected_orderstats(x_np)
    E_jx = np.asarray(os_jx.expected_orderstats(x_jx))
    np.testing.assert_allclose(E_jx, E_np, rtol=1e-12, atol=1e-12)

    E_inc_np = os_np.expected_orderstats_inclusion(x_np)
    E_inc_jx = np.asarray(os_jx.expected_orderstats_inclusion(x_jx))
    np.testing.assert_allclose(E_inc_jx, E_inc_np, rtol=1e-12, atol=1e-12)

    E_loo_np = os_np.expected_orderstats_leave_one_out(x_np)
    E_loo_jx = np.asarray(os_jx.expected_orderstats_leave_one_out(x_jx))
    np.testing.assert_allclose(E_loo_jx, E_loo_np, rtol=1e-12, atol=1e-12)


@pytest.mark.jax
def test_jax_gradient_matches_rank_weights():
    rng = np.random.default_rng(3)
    N, k = 35, 7

    x_np = _rand_x_no_ties(rng, N)
    os_jx = JX.precompute(N, k, dtype=jnp.float64, compute_conditional=False, compute_leave_one_out=False)

    x = jnp.asarray(x_np, dtype=jnp.float64)
    a = jnp.asarray(rng.normal(size=k), dtype=jnp.float64)

    def f(z):
        return os_jx.expected_lstat(z, a)

    grad = np.asarray(jax.grad(f)(x))
    w = np.asarray(os_jx.lstat_weight_by_item(x, a))

    np.testing.assert_allclose(grad, w, rtol=1e-10, atol=1e-10)
