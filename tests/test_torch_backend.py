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
