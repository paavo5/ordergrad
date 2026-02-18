import numpy as np
import pytest

from ordergrad.numpy_backend import OrderStatTransform as NP


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# Enable float64 for stable cross-backend comparisons.
jax.config.update("jax_enable_x64", True)

from ordergrad.jax_backend import OrderStatTransform as JX


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


@pytest.mark.jax
def test_jax_lstat_and_advantage_match_numpy_with_and_without_preweights():
    rng = np.random.default_rng(12)
    N, k = 30, 6
    x_np = _rand_x_no_ties(rng, N)
    a_np = rng.normal(size=k).astype(np.float64)

    os_np = NP.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    os_jx = JX.precompute(N, k, dtype=jnp.float64, compute_conditional=True, compute_leave_one_out=True)

    x_jx = jnp.asarray(x_np, dtype=jnp.float64)
    a_jx = jnp.asarray(a_np, dtype=jnp.float64)

    np.testing.assert_allclose(np.asarray(os_jx.expected_lstat(x_jx, a_jx)), os_np.expected_lstat(x_np, a_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(os_jx.expected_lstat_inclusion(x_jx, a_jx)), os_np.expected_lstat_inclusion(x_np, a_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(os_jx.expected_lstat_leave_one_out(x_jx, a_jx)), os_np.expected_lstat_leave_one_out(x_np, a_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(os_jx.expected_orderstats_advantage(x_jx)), os_np.expected_orderstats_advantage(x_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(os_jx.expected_lstat_advantage(x_jx, a_jx)), os_np.expected_lstat_advantage(x_np, a_np), rtol=1e-12, atol=1e-12)

    # Preweighted path should match explicit-a path.
    os_jx_w = os_jx.with_lstat_weights(a_jx)
    np.testing.assert_allclose(np.asarray(os_jx_w.expected_lstat(x_jx)), np.asarray(os_jx.expected_lstat(x_jx, a_jx)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(os_jx_w.expected_lstat_inclusion(x_jx)), np.asarray(os_jx.expected_lstat_inclusion(x_jx, a_jx)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(os_jx_w.expected_lstat_leave_one_out(x_jx)), np.asarray(os_jx.expected_lstat_leave_one_out(x_jx, a_jx)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(os_jx_w.expected_lstat_advantage(x_jx)), np.asarray(os_jx.expected_lstat_advantage(x_jx, a_jx)), rtol=1e-12, atol=1e-12)


@pytest.mark.jax
def test_jax_dense_matmul_variants_match_efficient_and_auto():
    rng = np.random.default_rng(14)
    N, k = 22, 5
    x_np = _rand_x_no_ties(rng, N)
    a_np = rng.normal(size=k).astype(np.float64)

    x_jx = jnp.asarray(x_np, dtype=jnp.float64)
    a_jx = jnp.asarray(a_np, dtype=jnp.float64)

    dense = JX.precompute(N, k, dtype=jnp.float64, compute_conditional=True, compute_leave_one_out=True, compute_dense_matrices=True)
    nodense = JX.precompute(N, k, dtype=jnp.float64, compute_conditional=True, compute_leave_one_out=True, compute_dense_matrices=False)

    np.testing.assert_allclose(
        np.asarray(dense.expected_orderstats_inclusion(x_jx, method="matmul")),
        np.asarray(dense.expected_orderstats_inclusion(x_jx, method="efficient")),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(dense.expected_orderstats_leave_one_out(x_jx, method="matmul")),
        np.asarray(dense.expected_orderstats_leave_one_out(x_jx, method="efficient")),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(dense.expected_orderstats_advantage(x_jx, method="matmul")),
        np.asarray(dense.expected_orderstats_advantage(x_jx, method="efficient")),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(dense.expected_orderstats_inclusion(x_jx, method="auto")),
        np.asarray(dense.expected_orderstats_inclusion(x_jx, method="matmul")),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(nodense.expected_orderstats_inclusion(x_jx, method="matmul")),
        np.asarray(nodense.expected_orderstats_inclusion(x_jx, method="efficient")),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(dense.expected_lstat_inclusion(x_jx, a_jx, method="matmul")),
        np.asarray(dense.expected_lstat_inclusion(x_jx, a_jx, method="efficient")),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(dense.expected_lstat_leave_one_out(x_jx, a_jx, method="matmul")),
        np.asarray(dense.expected_lstat_leave_one_out(x_jx, a_jx, method="efficient")),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(dense.expected_lstat_advantage(x_jx, a_jx, method="matmul")),
        np.asarray(dense.expected_lstat_advantage(x_jx, a_jx, method="efficient")),
        rtol=1e-12,
        atol=1e-12,
    )

    dense_w = dense.with_lstat_weights(a_jx)
    np.testing.assert_allclose(
        np.asarray(dense_w.expected_lstat_advantage(x_jx, method="matmul")),
        np.asarray(dense_w.expected_lstat_advantage(x_jx, method="efficient")),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.jax
def test_jax_known_rank_position_matches_numpy_and_recovers_inclusion_and_advantage():
    rng = np.random.default_rng(15)
    N, k = 18, 5
    x_np = _rand_x_no_ties(rng, N)
    a_np = rng.normal(size=k).astype(np.float64)

    os_np = NP.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    os_jx = JX.precompute(N, k, dtype=jnp.float64, compute_conditional=True, compute_leave_one_out=True)

    x_jx = jnp.asarray(x_np, dtype=jnp.float64)
    a_jx = jnp.asarray(a_np, dtype=jnp.float64)

    E_inc_np = os_np.expected_orderstats_inclusion(x_np)
    E_inc_jx = np.asarray(os_jx.expected_orderstats_inclusion(x_jx))
    E_loo_np = os_np.expected_orderstats_leave_one_out(x_np)

    perm = np.argsort(x_np, kind="mergesort")
    inv = np.empty(N, dtype=np.int64)
    inv[perm] = np.arange(N, dtype=np.int64)

    rec_inc_np = np.zeros_like(E_inc_np)
    rec_inc_jx = np.zeros_like(E_inc_np)
    rec_adv_np = np.zeros_like(E_inc_np)
    rec_adv_jx = np.zeros_like(E_inc_np)
    rec_l_inc_np = np.zeros(N, dtype=np.float64)
    rec_l_inc_jx = np.zeros(N, dtype=np.float64)

    for ppos in range(1, k + 1):
        rp_np = os_np.expected_orderstats_known_rank_position(x_np, ppos)
        rp_jx = np.asarray(os_jx.expected_orderstats_known_rank_position(x_jx, ppos))
        np.testing.assert_allclose(rp_jx, rp_np, rtol=1e-12, atol=1e-12)

        lp_np = os_np.expected_lstat_known_rank_position(x_np, a_np, ppos)
        lp_jx = np.asarray(os_jx.expected_lstat_known_rank_position(x_jx, a_jx, ppos))
        np.testing.assert_allclose(lp_jx, lp_np, rtol=1e-12, atol=1e-12)

        probs = os_np.B[:, ppos - 1][inv]
        rec_inc_np += probs[:, None] * rp_np
        rec_inc_jx += probs[:, None] * rp_jx
        rec_adv_np += probs[:, None] * (rp_np - E_loo_np)
        rec_adv_jx += probs[:, None] * (rp_jx - E_loo_np)
        rec_l_inc_np += probs * lp_np
        rec_l_inc_jx += probs * lp_jx

    np.testing.assert_allclose(rec_inc_np, E_inc_np, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rec_inc_jx, E_inc_jx, rtol=1e-12, atol=1e-12)

    np.testing.assert_allclose(rec_adv_np, os_np.expected_orderstats_advantage(x_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rec_adv_jx, np.asarray(os_jx.expected_orderstats_advantage(x_jx)), rtol=1e-12, atol=1e-12)

    np.testing.assert_allclose(rec_l_inc_np, os_np.expected_lstat_inclusion(x_np, a_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rec_l_inc_jx, np.asarray(os_jx.expected_lstat_inclusion(x_jx, a_jx)), rtol=1e-12, atol=1e-12)
