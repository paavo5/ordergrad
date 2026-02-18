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
def test_torch_known_rank_position_matches_numpy_and_recovers_inclusion_and_advantage():
    rng = np.random.default_rng(16)
    N, k = 18, 5
    x_np = _rand_x_no_ties(rng, N)
    a_np = rng.normal(size=int(np.floor(k))).astype(np.float64)

    os_np = NP.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    os_th = TH.precompute(N, k, dtype=torch.float64, compute_conditional=True, compute_leave_one_out=True)

    x_th = torch.tensor(x_np, dtype=torch.float64)
    a_th = torch.tensor(a_np, dtype=torch.float64)

    E_inc_np = os_np.expected_orderstats_inclusion(x_np)
    E_inc_th = os_th.expected_orderstats_inclusion(x_th).detach().cpu().numpy()
    E_loo_np = os_np.expected_orderstats_leave_one_out(x_np)

    perm = np.argsort(x_np, kind="mergesort")
    inv = np.empty(N, dtype=np.int64)
    inv[perm] = np.arange(N, dtype=np.int64)

    rec_inc_np = np.zeros_like(E_inc_np)
    rec_inc_th = np.zeros_like(E_inc_np)
    rec_adv_np = np.zeros_like(E_inc_np)
    rec_adv_th = np.zeros_like(E_inc_np)
    rec_l_inc_np = np.zeros(N, dtype=np.float64)
    rec_l_inc_th = np.zeros(N, dtype=np.float64)

    for ppos in range(1, k + 1):
        rp_np = os_np.expected_orderstats_known_rank_position(x_np, ppos)
        rp_th = os_th.expected_orderstats_known_rank_position(x_th, ppos).detach().cpu().numpy()
        np.testing.assert_allclose(rp_th, rp_np, rtol=1e-12, atol=1e-12)

        lp_np = os_np.expected_lstat_known_rank_position(x_np, a_np, ppos)
        lp_th = os_th.expected_lstat_known_rank_position(x_th, a_th, ppos).detach().cpu().numpy()
        np.testing.assert_allclose(lp_th, lp_np, rtol=1e-12, atol=1e-12)

        probs = os_np.B[:, ppos - 1][inv]
        rec_inc_np += probs[:, None] * rp_np
        rec_inc_th += probs[:, None] * rp_th
        rec_adv_np += probs[:, None] * (rp_np - E_loo_np)
        rec_adv_th += probs[:, None] * (rp_th - E_loo_np)
        rec_l_inc_np += probs * lp_np
        rec_l_inc_th += probs * lp_th

    np.testing.assert_allclose(rec_inc_np, E_inc_np, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rec_inc_th, E_inc_th, rtol=1e-12, atol=1e-12)

    np.testing.assert_allclose(rec_adv_np, os_np.expected_orderstats_advantage(x_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rec_adv_th, os_th.expected_orderstats_advantage(x_th).detach().cpu().numpy(), rtol=1e-12, atol=1e-12)

    np.testing.assert_allclose(rec_l_inc_np, os_np.expected_lstat_inclusion(x_np, a_np), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rec_l_inc_th, os_th.expected_lstat_inclusion(x_th, a_th).detach().cpu().numpy(), rtol=1e-12, atol=1e-12)


@pytest.mark.torch
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
def test_torch_real_k_fractional_is_supported_and_known_rp_rejected():
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

    with pytest.raises(ValueError, match=r"known \(r,p\) variant"):
        os_frac.expected_orderstats_known_rank_position(x_th, 2)
