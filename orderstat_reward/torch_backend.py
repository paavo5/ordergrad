from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import torch
except Exception as e:  # pragma: no cover
    raise ImportError(
        "orderstat_reward.torch_backend requires PyTorch. "
        "Install it with: `pip install orderstat-reward[torch]`"
    ) from e


def _log_choose(n: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Vectorized log binomial log C(n,r) with -inf for invalid."""
    # Broadcast and compute with masking. Work in float64 for stability.
    n = n.to(dtype=torch.float64)
    r = r.to(dtype=torch.float64)
    valid = (n >= 0) & (r >= 0) & (r <= n)

    n0 = torch.where(valid, n, torch.zeros_like(n))
    r0 = torch.where(valid, r, torch.zeros_like(r))
    nr0 = torch.where(valid, n0 - r0, torch.zeros_like(n0))

    out = torch.lgamma(n0 + 1.0) - torch.lgamma(r0 + 1.0) - torch.lgamma(nr0 + 1.0)
    out = torch.where(valid, out, torch.full_like(out, float("-inf")))
    return out


def _build_weight_matrix(
    N_rows: int,
    k: int,
    log_den: torch.Tensor,
    log_term_fn,  # (m_1based (M,1), j_1based (1,k)) -> (M,k)
    *,
    dtype: torch.dtype,
    device: torch.device,
    renormalize_cols: bool = False,
) -> torch.Tensor:
    j = torch.arange(1, k + 1, device=device, dtype=torch.int64)[None, :]
    m = torch.arange(1, N_rows + 1, device=device, dtype=torch.int64)[:, None]
    logw = log_term_fn(m, j) - log_den
    W = torch.exp(logw).to(dtype=dtype)
    if renormalize_cols:
        W = W / W.sum(dim=0, keepdim=True)
    return W


def precompute_W_unconditional(
    N: int, k: int, *, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None
) -> torch.Tensor:
    """W[m-1,j-1] = P(X_(j:k) == x_(m)) for uniform k-subset from N."""
    if not (1 <= k <= N):
        raise ValueError("Require 1 <= k <= N")
    device = device or torch.device("cpu")

    log_den = _log_choose(
        torch.tensor(N, device=device), torch.tensor(k, device=device)
    )

    def log_term(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m, k - j)

    return _build_weight_matrix(
        N,
        k,
        log_den,
        log_term,
        dtype=dtype,
        device=device,
        renormalize_cols=True,
    )


def precompute_ABC_conditional_including_rank(
    N: int, k: int, *, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute conditional matrices A,B,C (shape (N,k))."""
    if not (1 <= k <= N):
        raise ValueError("Require 1 <= k <= N")
    device = device or torch.device("cpu")

    log_den = _log_choose(
        torch.tensor(N - 1, device=device), torch.tensor(k - 1, device=device)
    )

    def logA(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m - 1, k - j - 1)

    def logB(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m, k - j)

    def logC(m, j):
        return _log_choose(m - 2, j - 2) + _log_choose(N - m, k - j)

    A = _build_weight_matrix(N, k, log_den, logA, dtype=dtype, device=device, renormalize_cols=False)
    B = _build_weight_matrix(N, k, log_den, logB, dtype=dtype, device=device, renormalize_cols=False)
    C = _build_weight_matrix(N, k, log_den, logC, dtype=dtype, device=device, renormalize_cols=False)
    return A, B, C


def precompute_W_leave_one_out(
    N: int, k: int, *, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Wm for reduced population size (N-1). Shape (N-1,k), columns sum to 1."""
    if not (1 <= k <= N - 1):
        raise ValueError("Require 1 <= k <= N-1 for leave-one-out")
    device = device or torch.device("cpu")

    log_den = _log_choose(
        torch.tensor(N - 1, device=device), torch.tensor(k, device=device)
    )

    def log_term(p, j):
        return _log_choose(p - 1, j - 1) + _log_choose((N - 1) - p, k - j)

    return _build_weight_matrix(
        N - 1,
        k,
        log_den,
        log_term,
        dtype=dtype,
        device=device,
        renormalize_cols=True,
    )


@dataclass(frozen=True)
class OrderStatTransform:
    """PyTorch implementation.

    Returned inclusion/leave-one-out matrices/vectors are in **original index order**.
    """

    N: int
    k: int
    W: torch.Tensor
    A: Optional[torch.Tensor]
    B: Optional[torch.Tensor]
    C: Optional[torch.Tensor]
    Wm: Optional[torch.Tensor]

    @classmethod
    def precompute(
        cls,
        N: int,
        k: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
        compute_conditional: bool = True,
        compute_leave_one_out: bool = True,
    ) -> "OrderStatTransform":
        device = device or torch.device("cpu")
        W = precompute_W_unconditional(N, k, dtype=dtype, device=device)

        A = B = C = None
        if compute_conditional:
            A, B, C = precompute_ABC_conditional_including_rank(N, k, dtype=dtype, device=device)

        Wm = None
        if compute_leave_one_out:
            if k > N - 1:
                raise ValueError("Leave-one-out requires k <= N-1")
            Wm = precompute_W_leave_one_out(N, k, dtype=dtype, device=device)

        return cls(N=N, k=k, W=W, A=A, B=B, C=C, Wm=Wm)

    def _sort_with_inverse_rank(self, x: torch.Tensor):
        if x.ndim != 1 or x.shape[0] != self.N:
            raise ValueError(f"x must be shape ({self.N},)")
        try:
            perm = torch.argsort(x, stable=True)
        except TypeError:
            perm = torch.argsort(x)
        x_sorted = x[perm]
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(self.N, device=x.device, dtype=perm.dtype)
        return x_sorted, inv

    # -------- order-statistics expectations --------

    def expected_orderstats(self, x: torch.Tensor) -> torch.Tensor:
        x_sorted, _ = self._sort_with_inverse_rank(x)
        return x_sorted @ self.W

    def expected_orderstats_inclusion(self, x: torch.Tensor) -> torch.Tensor:
        if self.A is None or self.B is None or self.C is None:
            raise ValueError("Conditional matrices A,B,C were not precomputed.")

        x_sorted, inv = self._sort_with_inverse_rank(x)

        XA = x_sorted[:, None] * self.A
        XC = x_sorted[:, None] * self.C

        prefA = torch.cumsum(XA, dim=0)
        prefA_excl = torch.cat([torch.zeros((1, self.k), dtype=XA.dtype, device=XA.device), prefA[:-1]], dim=0)

        prefC = torch.cumsum(XC, dim=0)
        totalC = prefC[-1:]
        suffC_excl = totalC - prefC

        diag = x_sorted[:, None] * self.B
        E_by_rank = prefA_excl + diag + suffC_excl
        return E_by_rank[inv, :]

    def expected_orderstats_leave_one_out(self, x: torch.Tensor) -> torch.Tensor:
        if self.Wm is None:
            raise ValueError("Leave-one-out matrix Wm was not precomputed.")
        if self.k > self.N - 1:
            raise ValueError("Leave-one-out requires k <= N-1")

        x_sorted, inv = self._sort_with_inverse_rank(x)
        u = x_sorted[:-1]
        v = x_sorted[1:]

        P1 = u[:, None] * self.Wm
        P2 = v[:, None] * self.Wm

        pref1 = torch.cumsum(P1, dim=0)
        pref1_excl = torch.cat([torch.zeros((1, self.k), dtype=P1.dtype, device=P1.device), pref1], dim=0)

        pref2 = torch.cumsum(P2, dim=0)
        total2 = pref2[-1:]
        pref2_before = torch.cat([torch.zeros((1, self.k), dtype=P2.dtype, device=P2.device), pref2], dim=0)
        suffix2 = total2 - pref2_before

        E_by_rank = pref1_excl + suffix2
        return E_by_rank[inv, :]

    # -------- L-statistics (reward transforms) --------

    def lstat_weight_by_rank(self, a: torch.Tensor) -> torch.Tensor:
        if a.shape != (self.k,):
            raise ValueError(f"a must be shape ({self.k},)")
        return self.W @ a

    def lstat_weight_by_item(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        _, inv = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return w_rank[inv]

    def expected_lstat(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x_sorted, _ = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return (x_sorted * w_rank).sum()

    def expected_lstat_inclusion(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        E_inc = self.expected_orderstats_inclusion(x)
        return E_inc @ a

    def expected_lstat_leave_one_out(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        E_loo = self.expected_orderstats_leave_one_out(x)
        return E_loo @ a


    def expected_lstat_advantage(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Convenience: per-item advantage-style transform.

        Defined as:
            adv[i] = E[T(S) | i included] - E[T(S)] on population with i removed.

        Shape: (N,)
        """
        return self.expected_lstat_inclusion(x, a) - self.expected_lstat_leave_one_out(x, a)

    # ---- Backwards-compatible aliases (rankpg-style naming) ----
    expected_all_j = expected_orderstats
    expected_all_j_conditional_included_all_i = expected_orderstats_inclusion
    expected_all_j_leave_one_out_all_i = expected_orderstats_leave_one_out


# Backwards-compatible class alias
OrderStatKofN = OrderStatTransform
