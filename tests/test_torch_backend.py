import numpy as np
import pytest

from ordergrad.numpy_backend import OrderStatTransform as NP


torch = pytest.importorskip("torch")

from ordergrad.torch_backend import OrderStatTransform as TH


def _rand_x_no_ties(rng: np.random.Generator, N: int) -> np.ndarray:
    x = rng.normal(size=N).astype(np.float64)
    return x + 1e-6 * np.arange(N, dtype=np.float64)


@pytest.mark.torch
@pytest.mark.parametrize("N,k", [(40, 9), (25, 5)])
def test_torch_matches_numpy(N, k):
    rng = np.random.default_rng(0)
    x_np = _rand_x_no_ties(rng, N)

    os_np = NP.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    os_th = TH.precompute(N, k, dtype=torch.float64, compute_conditional=True, compute_leave_one_out=True)

    x_th = torch.tensor(x_np, dtype=torch.float64)

    E_np = os_np.expected_orderstats(x_np)
    E_th = os_th.expected_orderstats(x_th).detach().cpu().numpy()
    np.testing.assert_allclose(E_th, E_np, rtol=1e-12, atol=1e-12)

    E_inc_np = os_np.expected_orderstats_inclusion(x_np)
    E_inc_th = os_th.expected_orderstats_inclusion(x_th).detach().cpu().numpy()
    np.testing.assert_allclose(E_inc_th, E_inc_np, rtol=1e-12, atol=1e-12)

    E_loo_np = os_np.expected_orderstats_leave_one_out(x_np)
    E_loo_th = os_th.expected_orderstats_leave_one_out(x_th).detach().cpu().numpy()
    np.testing.assert_allclose(E_loo_th, E_loo_np, rtol=1e-12, atol=1e-12)


@pytest.mark.torch
def test_torch_gradient_matches_rank_weights():
    rng = np.random.default_rng(2)
    N, k = 35, 7

    x_np = _rand_x_no_ties(rng, N)
    os_th = TH.precompute(N, k, dtype=torch.float64, compute_conditional=False, compute_leave_one_out=False)

    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    a = torch.tensor(rng.normal(size=k), dtype=torch.float64)

    y = os_th.expected_lstat(x, a)
    y.backward()

    grad = x.grad.detach().cpu().numpy()
    w = os_th.lstat_weight_by_item(x.detach(), a.detach()).cpu().numpy()

    np.testing.assert_allclose(grad, w, rtol=1e-10, atol=1e-10)


@pytest.mark.torch
def test_torch_advantage_detach_flag_controls_gradient_flow():
    rng = np.random.default_rng(32)
    N, k = 20, 5
    x_np = _rand_x_no_ties(rng, N)
    a_np = rng.normal(size=k).astype(np.float64)

    os_th = TH.precompute(N, k, dtype=torch.float64, compute_conditional=True, compute_leave_one_out=True)
    a_th = torch.tensor(a_np, dtype=torch.float64)
    c_th = torch.tensor(rng.normal(size=N), dtype=torch.float64)

    x_det = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    y_det = torch.dot(c_th, os_th.expected_lstat_advantage(x_det, a_th, detach_advantage=True))
    assert not y_det.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        y_det.backward()

    x_att = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    y_att = torch.dot(c_th, os_th.expected_lstat_advantage(x_att, a_th, detach_advantage=False))
    y_att.backward()
    g_att = x_att.grad.detach().cpu().numpy()

    assert not np.allclose(g_att, 0.0, atol=1e-10, rtol=1e-10)


@pytest.mark.torch
def test_torch_lstat_and_advantage_match_numpy_with_and_without_preweights():
    rng = np.random.default_rng(11)
    N, k = 30, 6
    x_np = _rand_x_no_ties(rng, N)
    a_np = rng.normal(size=int(np.floor(k))).astype(np.float64)

    os_np = NP.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    os_th = TH.precompute(N, k, dtype=torch.float64, compute_conditional=True, compute_leave_one_out=True)

    x_th = torch.tensor(x_np, dtype=torch.float64)
    a_th = torch.tensor(a_np, dtype=torch.float64)

    np.testing.assert_allclose(
        os_th.expected_lstat(x_th, a_th).detach().cpu().numpy(),
        os_np.expected_lstat(x_np, a_np),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        os_th.expected_lstat_inclusion(x_th, a_th).detach().cpu().numpy(),
        os_np.expected_lstat_inclusion(x_np, a_np),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        os_th.expected_lstat_leave_one_out(x_th, a_th).detach().cpu().numpy(),
        os_np.expected_lstat_leave_one_out(x_np, a_np),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        os_th.expected_orderstats_advantage(x_th).detach().cpu().numpy(),
        os_np.expected_orderstats_advantage(x_np),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        os_th.expected_lstat_advantage(x_th, a_th).detach().cpu().numpy(),
        os_np.expected_lstat_advantage(x_np, a_np),
        rtol=1e-12,
        atol=1e-12,
    )

    # Preweighted path should match explicit-a path.
    os_th_w = os_th.with_lstat_weights(a_th)
    np.testing.assert_allclose(
        os_th_w.expected_lstat(x_th).detach().cpu().numpy(),
        os_th.expected_lstat(x_th, a_th).detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        os_th_w.expected_lstat_inclusion(x_th).detach().cpu().numpy(),
        os_th.expected_lstat_inclusion(x_th, a_th).detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        os_th_w.expected_lstat_leave_one_out(x_th).detach().cpu().numpy(),
        os_th.expected_lstat_leave_one_out(x_th, a_th).detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        os_th_w.expected_lstat_advantage(x_th).detach().cpu().numpy(),
        os_th.expected_lstat_advantage(x_th, a_th).detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.torch
def test_torch_dense_matmul_variants_match_efficient_and_auto():
    rng = np.random.default_rng(13)
    N, k = 22, 5
    x_np = _rand_x_no_ties(rng, N)
    a_np = rng.normal(size=int(np.floor(k))).astype(np.float64)

    x_th = torch.tensor(x_np, dtype=torch.float64)
    a_th = torch.tensor(a_np, dtype=torch.float64)

    dense = TH.precompute(N, k, dtype=torch.float64, compute_conditional=True, compute_leave_one_out=True, compute_dense_matrices=True)
    nodense = TH.precompute(N, k, dtype=torch.float64, compute_conditional=True, compute_leave_one_out=True, compute_dense_matrices=False)

    # Dense matmul path parity.
    np.testing.assert_allclose(
        dense.expected_orderstats_inclusion(x_th, method="matmul").detach().cpu().numpy(),
        dense.expected_orderstats_inclusion(x_th, method="efficient").detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        dense.expected_orderstats_leave_one_out(x_th, method="matmul").detach().cpu().numpy(),
        dense.expected_orderstats_leave_one_out(x_th, method="efficient").detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        dense.expected_orderstats_advantage(x_th, method="matmul").detach().cpu().numpy(),
        dense.expected_orderstats_advantage(x_th, method="efficient").detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    # Auto selects dense path when available.
    np.testing.assert_allclose(
        dense.expected_orderstats_inclusion(x_th, method="auto").detach().cpu().numpy(),
        dense.expected_orderstats_inclusion(x_th, method="matmul").detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    # matmul falls back to efficient when dense matrices are absent.
    np.testing.assert_allclose(
        nodense.expected_orderstats_inclusion(x_th, method="matmul").detach().cpu().numpy(),
        nodense.expected_orderstats_inclusion(x_th, method="efficient").detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    # L-stat matmul variants (explicit a).
    np.testing.assert_allclose(
        dense.expected_lstat_inclusion(x_th, a_th, method="matmul").detach().cpu().numpy(),
        dense.expected_lstat_inclusion(x_th, a_th, method="efficient").detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        dense.expected_lstat_leave_one_out(x_th, a_th, method="matmul").detach().cpu().numpy(),
        dense.expected_lstat_leave_one_out(x_th, a_th, method="efficient").detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        dense.expected_lstat_advantage(x_th, a_th, method="matmul").detach().cpu().numpy(),
        dense.expected_lstat_advantage(x_th, a_th, method="efficient").detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    # L-stat preweighted no-a matmul path parity.
    dense_w = dense.with_lstat_weights(a_th)
    np.testing.assert_allclose(
        dense_w.expected_lstat_advantage(x_th, method="matmul").detach().cpu().numpy(),
        dense_w.expected_lstat_advantage(x_th, method="efficient").detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.torch
@pytest.mark.torch


@pytest.mark.torch
def test_torch_known_rp_matches_numpy():
    r_np = np.array([-1.0, 0.2, 1.1, 2.4], dtype=np.float64)
    p_np = np.array([0.1, 0.45, 0.3, 0.15], dtype=np.float64)
    k = 4
    a_np = np.array([0.2, -0.1, 0.4, 0.3], dtype=np.float64)

    os_np = NP.precompute(12, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    os_th = TH.precompute(12, k, dtype=torch.float64, compute_conditional=False, compute_leave_one_out=False)

    r_th = torch.tensor(r_np, dtype=torch.float64)
    p_th = torch.tensor(p_np, dtype=torch.float64)
    a_th = torch.tensor(a_np, dtype=torch.float64)

    np.testing.assert_allclose(os_th.expected_orderstats_known_rp(r_th, p_th).detach().cpu().numpy(), os_np.expected_orderstats_known_rp(r_np, p_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(os_th.expected_orderstats_inclusion_known_rp(r_th, p_th).detach().cpu().numpy(), os_np.expected_orderstats_inclusion_known_rp(r_np, p_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(os_th.expected_orderstats_advantage_known_rp(r_th, p_th).detach().cpu().numpy(), os_np.expected_orderstats_advantage_known_rp(r_np, p_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(os_th.expected_lstat_advantage_known_rp(r_th, p_th, a_th).detach().cpu().numpy(), os_np.expected_lstat_advantage_known_rp(r_np, p_np, a_np), rtol=1e-12, atol=1e-12)


def test_torch_real_k_matches_integer_k_when_equal():
    rng = np.random.default_rng(19)
    N, k = 14, 5
    x_np = _rand_x_no_ties(rng, N)
    x_th = torch.tensor(x_np, dtype=torch.float64)

    os_int = TH.precompute(N, k, dtype=torch.float64, compute_conditional=True, compute_leave_one_out=True)
    os_real = TH.precompute(N, float(k), dtype=torch.float64, compute_conditional=True, compute_leave_one_out=True)

    np.testing.assert_allclose(os_real.expected_orderstats(x_th).detach().cpu().numpy(), os_int.expected_orderstats(x_th).detach().cpu().numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(os_real.expected_orderstats_inclusion(x_th).detach().cpu().numpy(), os_int.expected_orderstats_inclusion(x_th).detach().cpu().numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(os_real.expected_orderstats_leave_one_out(x_th).detach().cpu().numpy(), os_int.expected_orderstats_leave_one_out(x_th).detach().cpu().numpy(), rtol=1e-12, atol=1e-12)


@pytest.mark.torch
def test_torch_real_k_fractional_is_supported():
    rng = np.random.default_rng(20)
    N, k = 15, 5.4
    x_np = _rand_x_no_ties(rng, N)
    a_np = rng.normal(size=int(np.floor(k))).astype(np.float64)

    x_th = torch.tensor(x_np, dtype=torch.float64)
    a_th = torch.tensor(a_np, dtype=torch.float64)

    os_frac = TH.precompute(N, k, dtype=torch.float64, compute_conditional=True, compute_leave_one_out=True, compute_dense_matrices=True)

    assert np.isfinite(os_frac.expected_orderstats(x_th).detach().cpu().numpy()).all()
    assert np.isfinite(os_frac.expected_orderstats_inclusion(x_th, method="matmul").detach().cpu().numpy()).all()
    assert np.isfinite(os_frac.expected_orderstats_leave_one_out(x_th, method="matmul").detach().cpu().numpy()).all()
    assert np.isfinite(os_frac.expected_orderstats_advantage(x_th, method="matmul").detach().cpu().numpy()).all()
    assert np.isfinite(os_frac.expected_lstat_advantage(x_th, a_th, method="matmul").detach().cpu().numpy()).all()


@pytest.mark.torch
def test_torch_harrell_davis_matches_numpy_reference_weights_and_alias():
    k = 9
    q = 0.25

    os_np = NP.precompute(k, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    os_th = TH.precompute(k, k, dtype=torch.float64, compute_conditional=False, compute_leave_one_out=False)

    w_np = os_np._preset_lstat_weights(k, f"HarrellDavis:{q}", dtype=np.float64)
    w_th = os_th._preset_lstat_weights(k, f"HarrellDavis:{q}", dtype=torch.float64, device=torch.device("cpu"))
    w_alias = os_th._preset_lstat_weights(k, f"HarrelDavis:{q}", dtype=torch.float64, device=torch.device("cpu"))

    np.testing.assert_allclose(w_th.detach().cpu().numpy(), w_np, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(w_alias.detach().cpu().numpy(), w_th.detach().cpu().numpy(), rtol=1e-12, atol=1e-12)

    


@pytest.mark.torch
def test_torch_quantile_preset_matches_numpy_reference():
    k = 6
    os_np = NP.precompute(k, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    os_th = TH.precompute(k, k, dtype=torch.float64, compute_conditional=False, compute_leave_one_out=False)

    for spec in ["Quantile:0.1", "QuantileWeibull:0.1", "QuantileHazen:0.1", "QuantileBlom:0.1", "TopQuantileBlom:0.1"]:
        w_np = os_np._preset_lstat_weights(k, spec, dtype=np.float64)
        w_th = os_th._preset_lstat_weights(k, spec, dtype=torch.float64, device=torch.device("cpu"))
        np.testing.assert_allclose(w_th.detach().cpu().numpy(), w_np, rtol=1e-12, atol=1e-12)


@pytest.mark.torch
def test_torch_rank_preset_matches_numpy_reference():
    k = 6
    os_np = NP.precompute(k, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    os_th = TH.precompute(k, k, dtype=torch.float64, compute_conditional=False, compute_leave_one_out=False)

    w_np = os_np._preset_lstat_weights(k, "Rank:3", dtype=np.float64)
    w_th = os_th._preset_lstat_weights(k, "Rank:3", dtype=torch.float64, device=torch.device("cpu"))
    np.testing.assert_allclose(w_th.detach().cpu().numpy(), w_np, rtol=1e-12, atol=1e-12)


@pytest.mark.torch
def test_torch_tailmean_presets_match_numpy_reference():
    k = 6
    os_np = NP.precompute(k, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    os_th = TH.precompute(k, k, dtype=torch.float64, compute_conditional=False, compute_leave_one_out=False)

    for spec in ["UpperTailMean:0.25", "LowerTailMean:0.25"]:
        w_np = os_np._preset_lstat_weights(k, spec, dtype=np.float64)
        w_th = os_th._preset_lstat_weights(k, spec, dtype=torch.float64, device=torch.device("cpu"))
        np.testing.assert_allclose(w_th.detach().cpu().numpy(), w_np, rtol=1e-12, atol=1e-12)
