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
    E_jx = np.asarray(jax.jit(os_jx.expected_orderstats)(x_jx))
    np.testing.assert_allclose(E_jx, E_np, rtol=1e-12, atol=1e-12)

    E_inc_np = os_np.expected_orderstats_inclusion(x_np)
    E_inc_jx = np.asarray(jax.jit(os_jx.expected_orderstats_inclusion)(x_jx))
    np.testing.assert_allclose(E_inc_jx, E_inc_np, rtol=1e-12, atol=1e-12)

    E_loo_np = os_np.expected_orderstats_leave_one_out(x_np)
    E_loo_jx = np.asarray(jax.jit(os_jx.expected_orderstats_leave_one_out)(x_jx))
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

    grad = np.asarray(jax.jit(jax.grad(f))(x))
    w = np.asarray(jax.jit(os_jx.lstat_weight_by_item)(x, a))

    np.testing.assert_allclose(grad, w, rtol=1e-10, atol=1e-10)


@pytest.mark.jax
def test_jax_advantage_detach_flag_controls_gradient_flow():
    rng = np.random.default_rng(31)
    N, k = 20, 5
    x_np = _rand_x_no_ties(rng, N)
    a_np = rng.normal(size=k).astype(np.float64)

    os_jx = JX.precompute(N, k, dtype=jnp.float64, compute_conditional=True, compute_leave_one_out=True)
    x = jnp.asarray(x_np, dtype=jnp.float64)
    a = jnp.asarray(a_np, dtype=jnp.float64)
    c = jnp.asarray(rng.normal(size=N), dtype=jnp.float64)

    g_det = np.asarray(jax.jit(jax.grad(lambda z: jnp.dot(c, os_jx.expected_lstat_advantage(z, a, detach_advantage=True))))(x))
    g_att = np.asarray(jax.jit(jax.grad(lambda z: jnp.dot(c, os_jx.expected_lstat_advantage(z, a, detach_advantage=False))))(x))

    np.testing.assert_allclose(g_det, np.zeros_like(g_det), atol=1e-12, rtol=1e-12)
    assert not np.allclose(g_att, 0.0, atol=1e-10, rtol=1e-10)


@pytest.mark.jax
def test_jax_lstat_and_advantage_match_numpy_with_and_without_preweights():
    rng = np.random.default_rng(12)
    N, k = 30, 6
    x_np = _rand_x_no_ties(rng, N)
    a_np = rng.normal(size=int(np.floor(k))).astype(np.float64)

    os_np = NP.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    os_jx = JX.precompute(N, k, dtype=jnp.float64, compute_conditional=True, compute_leave_one_out=True)

    x_jx = jnp.asarray(x_np, dtype=jnp.float64)
    a_jx = jnp.asarray(a_np, dtype=jnp.float64)

    np.testing.assert_allclose(np.asarray(jax.jit(os_jx.expected_lstat)(x_jx, a_jx)), os_np.expected_lstat(x_np, a_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(os_jx.expected_lstat_inclusion)(x_jx, a_jx)), os_np.expected_lstat_inclusion(x_np, a_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(os_jx.expected_lstat_leave_one_out)(x_jx, a_jx)), os_np.expected_lstat_leave_one_out(x_np, a_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(os_jx.expected_orderstats_advantage)(x_jx)), os_np.expected_orderstats_advantage(x_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(os_jx.expected_lstat_advantage)(x_jx, a_jx)), os_np.expected_lstat_advantage(x_np, a_np), rtol=1e-12, atol=1e-12)

    # Preweighted path should match explicit-a path.
    os_jx_w = os_jx.with_lstat_weights(a_jx)
    np.testing.assert_allclose(np.asarray(jax.jit(os_jx_w.expected_lstat)(x_jx)), np.asarray(jax.jit(os_jx.expected_lstat)(x_jx, a_jx)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(os_jx_w.expected_lstat_inclusion)(x_jx)), np.asarray(jax.jit(os_jx.expected_lstat_inclusion)(x_jx, a_jx)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(os_jx_w.expected_lstat_leave_one_out)(x_jx)), np.asarray(jax.jit(os_jx.expected_lstat_leave_one_out)(x_jx, a_jx)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(os_jx_w.expected_lstat_advantage)(x_jx)), np.asarray(jax.jit(os_jx.expected_lstat_advantage)(x_jx, a_jx)), rtol=1e-12, atol=1e-12)


@pytest.mark.jax
def test_jax_dense_matmul_variants_match_efficient_and_auto():
    rng = np.random.default_rng(14)
    N, k = 22, 5
    x_np = _rand_x_no_ties(rng, N)
    a_np = rng.normal(size=int(np.floor(k))).astype(np.float64)

    x_jx = jnp.asarray(x_np, dtype=jnp.float64)
    a_jx = jnp.asarray(a_np, dtype=jnp.float64)

    dense = JX.precompute(N, k, dtype=jnp.float64, compute_conditional=True, compute_leave_one_out=True, compute_dense_matrices=True)
    nodense = JX.precompute(N, k, dtype=jnp.float64, compute_conditional=True, compute_leave_one_out=True, compute_dense_matrices=False)

    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda x: dense.expected_orderstats_inclusion(x, method="matmul"))(x_jx)),
        np.asarray(jax.jit(lambda x: dense.expected_orderstats_inclusion(x, method="efficient"))(x_jx)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda x: dense.expected_orderstats_leave_one_out(x, method="matmul"))(x_jx)),
        np.asarray(jax.jit(lambda x: dense.expected_orderstats_leave_one_out(x, method="efficient"))(x_jx)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda x: dense.expected_orderstats_advantage(x, method="matmul"))(x_jx)),
        np.asarray(jax.jit(lambda x: dense.expected_orderstats_advantage(x, method="efficient"))(x_jx)),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda x: dense.expected_orderstats_inclusion(x, method="auto"))(x_jx)),
        np.asarray(jax.jit(lambda x: dense.expected_orderstats_inclusion(x, method="matmul"))(x_jx)),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda x: nodense.expected_orderstats_inclusion(x, method="matmul"))(x_jx)),
        np.asarray(jax.jit(lambda x: nodense.expected_orderstats_inclusion(x, method="efficient"))(x_jx)),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda x, a: dense.expected_lstat_inclusion(x, a, method="matmul"))(x_jx, a_jx)),
        np.asarray(jax.jit(lambda x, a: dense.expected_lstat_inclusion(x, a, method="efficient"))(x_jx, a_jx)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda x, a: dense.expected_lstat_leave_one_out(x, a, method="matmul"))(x_jx, a_jx)),
        np.asarray(jax.jit(lambda x, a: dense.expected_lstat_leave_one_out(x, a, method="efficient"))(x_jx, a_jx)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda x, a: dense.expected_lstat_advantage(x, a, method="matmul"))(x_jx, a_jx)),
        np.asarray(jax.jit(lambda x, a: dense.expected_lstat_advantage(x, a, method="efficient"))(x_jx, a_jx)),
        rtol=1e-12,
        atol=1e-12,
    )

    dense_w = dense.with_lstat_weights(a_jx)
    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda x: dense_w.expected_lstat_advantage(x, method="matmul"))(x_jx)),
        np.asarray(jax.jit(lambda x: dense_w.expected_lstat_advantage(x, method="efficient"))(x_jx)),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.jax
@pytest.mark.jax


@pytest.mark.jax
def test_jax_known_rp_matches_numpy():
    r_np = np.array([-1.0, 0.2, 1.1, 2.4], dtype=np.float64)
    p_np = np.array([0.1, 0.45, 0.3, 0.15], dtype=np.float64)
    k = 4
    a_np = np.array([0.2, -0.1, 0.4, 0.3], dtype=np.float64)

    os_np = NP.precompute(12, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    os_jx = JX.precompute(12, k, dtype=jnp.float64, compute_conditional=False, compute_leave_one_out=False)

    r_jx = jnp.asarray(r_np, dtype=jnp.float64)
    p_jx = jnp.asarray(p_np, dtype=jnp.float64)
    a_jx = jnp.asarray(a_np, dtype=jnp.float64)

    np.testing.assert_allclose(np.asarray(jax.jit(os_jx.expected_orderstats_known_rp)(r_jx, p_jx)), os_np.expected_orderstats_known_rp(r_np, p_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(os_jx.expected_orderstats_inclusion_known_rp)(r_jx, p_jx)), os_np.expected_orderstats_inclusion_known_rp(r_np, p_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(os_jx.expected_orderstats_advantage_known_rp)(r_jx, p_jx)), os_np.expected_orderstats_advantage_known_rp(r_np, p_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(os_jx.expected_lstat_advantage_known_rp)(r_jx, p_jx, a_jx)), os_np.expected_lstat_advantage_known_rp(r_np, p_np, a_np), rtol=1e-12, atol=1e-12)


def test_jax_real_k_matches_integer_k_when_equal():
    rng = np.random.default_rng(17)
    N, k = 14, 5
    x_np = _rand_x_no_ties(rng, N)
    x_jx = jnp.asarray(x_np, dtype=jnp.float64)

    os_int = JX.precompute(N, k, dtype=jnp.float64, compute_conditional=True, compute_leave_one_out=True)
    os_real = JX.precompute(N, float(k), dtype=jnp.float64, compute_conditional=True, compute_leave_one_out=True)

    np.testing.assert_allclose(np.asarray(jax.jit(os_real.expected_orderstats)(x_jx)), np.asarray(jax.jit(os_int.expected_orderstats)(x_jx)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(os_real.expected_orderstats_inclusion)(x_jx)), np.asarray(jax.jit(os_int.expected_orderstats_inclusion)(x_jx)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(os_real.expected_orderstats_leave_one_out)(x_jx)), np.asarray(jax.jit(os_int.expected_orderstats_leave_one_out)(x_jx)), rtol=1e-12, atol=1e-12)


@pytest.mark.jax
def test_jax_real_k_fractional_is_supported():
    rng = np.random.default_rng(18)
    N, k = 15, 5.4
    x_np = _rand_x_no_ties(rng, N)
    a_np = rng.normal(size=int(np.floor(k))).astype(np.float64)

    x_jx = jnp.asarray(x_np, dtype=jnp.float64)
    a_jx = jnp.asarray(a_np, dtype=jnp.float64)

    os_frac = JX.precompute(N, k, dtype=jnp.float64, compute_conditional=True, compute_leave_one_out=True, compute_dense_matrices=True)

    assert np.isfinite(np.asarray(jax.jit(os_frac.expected_orderstats)(x_jx))).all()
    assert np.isfinite(np.asarray(jax.jit(lambda x: os_frac.expected_orderstats_inclusion(x, method="matmul"))(x_jx))).all()
    assert np.isfinite(np.asarray(jax.jit(lambda x: os_frac.expected_orderstats_leave_one_out(x, method="matmul"))(x_jx))).all()
    assert np.isfinite(np.asarray(jax.jit(lambda x: os_frac.expected_orderstats_advantage(x, method="matmul"))(x_jx))).all()
    assert np.isfinite(np.asarray(jax.jit(lambda x, a: os_frac.expected_lstat_advantage(x, a, method="matmul"))(x_jx, a_jx))).all()

    


@pytest.mark.jax
def test_jax_quantile_preset_matches_numpy_reference():
    k = 6
    os_np = NP.precompute(k, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    os_jx = JX.precompute(k, k, dtype=jnp.float64, compute_conditional=False, compute_leave_one_out=False)

    for spec in ["Quantile:0.1", "QuantileWeibull:0.1", "QuantileHazen:0.1", "QuantileBlom:0.1", "TopQuantileBlom:0.1"]:
        w_np = os_np._preset_lstat_weights(k, spec, dtype=np.float64)
        w_jx = os_jx._preset_lstat_weights(k, spec, dtype=jnp.float64)
        np.testing.assert_allclose(np.asarray(w_jx), w_np, rtol=1e-12, atol=1e-12)


@pytest.mark.jax
def test_jax_rank_preset_matches_numpy_reference():
    k = 6
    os_np = NP.precompute(k, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    os_jx = JX.precompute(k, k, dtype=jnp.float64, compute_conditional=False, compute_leave_one_out=False)

    w_np = os_np._preset_lstat_weights(k, "Rank:3", dtype=np.float64)
    w_jx = os_jx._preset_lstat_weights(k, "Rank:3", dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(w_jx), w_np, rtol=1e-12, atol=1e-12)


@pytest.mark.jax
def test_jax_tailmean_presets_match_numpy_reference():
    k = 6
    os_np = NP.precompute(k, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    os_jx = JX.precompute(k, k, dtype=jnp.float64, compute_conditional=False, compute_leave_one_out=False)

    for spec in ["UpperTailMean:0.25", "LowerTailMean:0.25"]:
        w_np = os_np._preset_lstat_weights(k, spec, dtype=np.float64)
        w_jx = os_jx._preset_lstat_weights(k, spec, dtype=jnp.float64)
        np.testing.assert_allclose(np.asarray(w_jx), w_np, rtol=1e-12, atol=1e-12)


@pytest.mark.jax
def test_jax_fractional_range_lstat_presets_match_numpy_reference():
    k = 10
    os_np = NP.precompute(k, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    os_jx = JX.precompute(k, k, dtype=jnp.float64, compute_conditional=False, compute_leave_one_out=False)

    for spec in [
        "RangeUpperTailMean:0.2:0.5",
        "RangeLowerTailMean:0.2:0.5",
        "TrimmedMeanFrac:0.2:0.8",
    ]:
        w_np = os_np._preset_lstat_weights(k, spec, dtype=np.float64)
        w_jx = os_jx._preset_lstat_weights(k, spec, dtype=jnp.float64)
        np.testing.assert_allclose(np.asarray(w_jx), w_np, rtol=1e-12, atol=1e-12)
