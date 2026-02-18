import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# -----------------------------
# Combinatorics helpers
# -----------------------------

def _log_factorials(n: int, dtype=np.float64) -> np.ndarray:
    """lf[t] = log(t!) for t=0..n."""
    lf = np.empty(n + 1, dtype=dtype)
    lf[0] = 0.0
    if n > 0:
        lf[1:] = np.cumsum(np.log(np.arange(1, n + 1, dtype=dtype)))
    return lf


def _log_choose(lf: np.ndarray, n, r):
    """Vectorized log binomial log C(n,r).

    Returns -inf where invalid (r<0 or r>n or n<0).
    n, r may be broadcastable integer arrays.
    """
    n = np.asarray(n)
    r = np.asarray(r)
    valid = (n >= 0) & (r >= 0) & (r <= n)

    n0 = np.where(valid, n, 0)
    r0 = np.where(valid, r, 0)
    nr0 = np.where(valid, n - r, 0)

    out = lf[n0] - lf[r0] - lf[nr0]
    return np.where(valid, out, -np.inf)


def _build_weight_matrix(
    k: int,
    log_den: float,
    log_term_fn,  # callable (m_1based: (B,1), j_1based: (1,k)) -> log weights (B,k)
    out_shape: Tuple[int, int],
    dtype=np.float64,
    chunk_size: Optional[int] = None,
    renormalize_cols: bool = False,
) -> np.ndarray:
    """Generic builder for weight-like matrices with optional chunking."""
    out = np.empty(out_shape, dtype=dtype)
    j = np.arange(1, k + 1, dtype=np.int64)[None, :]  # (1,k)

    if chunk_size is None:
        m = np.arange(1, out_shape[0] + 1, dtype=np.int64)[:, None]  # (M,1)
        logw = log_term_fn(m, j) - log_den
        out[:] = np.exp(logw).astype(dtype, copy=False)
    else:
        M = out_shape[0]
        for start in range(1, M + 1, chunk_size):
            stop = min(M + 1, start + chunk_size)
            m = np.arange(start, stop, dtype=np.int64)[:, None]  # (B,1)
            logw = log_term_fn(m, j) - log_den
            out[start - 1 : stop - 1, :] = np.exp(logw).astype(dtype, copy=False)

    if renormalize_cols:
        colsum = out.sum(axis=0, keepdims=True)
        out /= colsum

    return out


# -----------------------------
# Weight precomputation
# -----------------------------


def precompute_W_unconditional(
    N: int, k: int, *, dtype=np.float64, chunk_size: Optional[int] = None
) -> np.ndarray:
    """W[m-1,j-1] = P(X_(j:k) == x_(m)) for uniform k-subset from N.

    Shape: (N,k). Columns sum to 1.
    """
    if not (1 <= k <= N):
        raise ValueError("Require 1 <= k <= N")

    lf = _log_factorials(N, dtype=np.float64)
    log_den = lf[N] - lf[k] - lf[N - k]  # log C(N,k)

    def log_term(m, j):
        return _log_choose(lf, m - 1, j - 1) + _log_choose(lf, N - m, k - j)

    return _build_weight_matrix(
        k,
        log_den,
        log_term,
        out_shape=(N, k),
        dtype=dtype,
        chunk_size=chunk_size,
        renormalize_cols=True,
    )


def precompute_ABC_conditional_including_rank(
    N: int, k: int, *, dtype=np.float64, chunk_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute conditional matrices A,B,C (all shape (N,k)).

    These correspond to the three cases in the conditional pmf given inclusion of rank r.
    """
    if not (1 <= k <= N):
        raise ValueError("Require 1 <= k <= N")

    lf = _log_factorials(N, dtype=np.float64)
    log_den = lf[N - 1] - lf[k - 1] - lf[(N - 1) - (k - 1)]  # log C(N-1,k-1)

    def logA(m, j):
        # m<r case weights
        return _log_choose(lf, m - 1, j - 1) + _log_choose(lf, N - m - 1, k - j - 1)

    def logB(m, j):
        # m==r diagonal case
        return _log_choose(lf, m - 1, j - 1) + _log_choose(lf, N - m, k - j)

    def logC(m, j):
        # m>r case weights
        return _log_choose(lf, m - 2, j - 2) + _log_choose(lf, N - m, k - j)

    A = _build_weight_matrix(
        k,
        log_den,
        logA,
        out_shape=(N, k),
        dtype=dtype,
        chunk_size=chunk_size,
        renormalize_cols=False,
    )
    B = _build_weight_matrix(
        k,
        log_den,
        logB,
        out_shape=(N, k),
        dtype=dtype,
        chunk_size=chunk_size,
        renormalize_cols=False,
    )
    C = _build_weight_matrix(
        k,
        log_den,
        logC,
        out_shape=(N, k),
        dtype=dtype,
        chunk_size=chunk_size,
        renormalize_cols=False,
    )
    return A, B, C


def precompute_W_leave_one_out(
    N: int, k: int, *, dtype=np.float64, chunk_size: Optional[int] = None
) -> np.ndarray:
    """Wm[p-1,j-1] = P(order stat equals p-th smallest) for a population of size (N-1).

    Shape: (N-1,k). Columns sum to 1. Requires k <= N-1.
    """
    if not (1 <= k <= N - 1):
        raise ValueError("Require 1 <= k <= N-1 for leave-one-out")

    lf = _log_factorials(N, dtype=np.float64)
    log_den = lf[N - 1] - lf[k] - lf[(N - 1) - k]  # log C(N-1,k)

    def log_term(p, j):
        return _log_choose(lf, p - 1, j - 1) + _log_choose(lf, (N - 1) - p, k - j)

    return _build_weight_matrix(
        k,
        log_den,
        log_term,
        out_shape=(N - 1, k),
        dtype=dtype,
        chunk_size=chunk_size,
        renormalize_cols=True,
    )


# -----------------------------
# Main API
# -----------------------------


@dataclass(frozen=True)
class OrderStatTransform:
    """NumPy implementation.

    Precompute weight matrices once for fixed (N,k), then evaluate for many x.

    All `*_inclusion` / `*_leave_one_out` results are returned in **original index order**.
    """

    N: int
    k: int
    W: np.ndarray  # (N,k)
    A: Optional[np.ndarray]  # (N,k)
    B: Optional[np.ndarray]  # (N,k)
    C: Optional[np.ndarray]  # (N,k)
    Wm: Optional[np.ndarray]  # (N-1,k)

    @classmethod
    def precompute(
        cls,
        N: int,
        k: int,
        *,
        dtype=np.float64,
        chunk_size: Optional[int] = None,
        compute_conditional: bool = True,
        compute_leave_one_out: bool = True,
    ):
        W = precompute_W_unconditional(N, k, dtype=dtype, chunk_size=chunk_size)

        A = B = C = None
        if compute_conditional:
            A, B, C = precompute_ABC_conditional_including_rank(
                N, k, dtype=dtype, chunk_size=chunk_size
            )

        Wm = None
        if compute_leave_one_out:
            if k > N - 1:
                raise ValueError("Leave-one-out requires k <= N-1")
            Wm = precompute_W_leave_one_out(N, k, dtype=dtype, chunk_size=chunk_size)

        return cls(N=N, k=k, W=W, A=A, B=B, C=C, Wm=Wm)

    def _sort_with_inverse_rank(self, x: np.ndarray):
        x = np.asarray(x)
        if x.ndim != 1 or x.shape[0] != self.N:
            raise ValueError(f"x must be shape ({self.N},)")
        perm = np.argsort(x, kind="mergesort")
        x_sorted = x[perm]
        inv = np.empty_like(perm)
        inv[perm] = np.arange(self.N, dtype=perm.dtype)
        return x_sorted, inv

    # -------- order-statistics expectations --------

    def expected_orderstats(self, x: np.ndarray) -> np.ndarray:
        """Return E[X_(j:k)] for j=1..k. Shape (k,)."""
        x_sorted, _ = self._sort_with_inverse_rank(x)
        return x_sorted @ self.W

    def expected_orderstats_inclusion(self, x: np.ndarray) -> np.ndarray:
        """Return E[X_(j:k) | i included] for all i. Shape (N,k)."""
        if self.A is None or self.B is None or self.C is None:
            raise ValueError("Conditional matrices A,B,C were not precomputed.")

        x_sorted, inv = self._sort_with_inverse_rank(x)

        XA = x_sorted[:, None] * self.A
        XC = x_sorted[:, None] * self.C

        prefA = np.cumsum(XA, axis=0)
        prefA_excl = np.vstack([np.zeros((1, self.k), dtype=XA.dtype), prefA[:-1]])

        prefC = np.cumsum(XC, axis=0)
        totalC = prefC[-1:]
        suffC_excl = totalC - prefC

        diag = x_sorted[:, None] * self.B

        E_by_rank = prefA_excl + diag + suffC_excl
        return E_by_rank[inv, :]

    def expected_orderstats_leave_one_out(self, x: np.ndarray) -> np.ndarray:
        """Return E[X_(j:k)] on population with i removed. Shape (N,k)."""
        if self.Wm is None:
            raise ValueError("Leave-one-out matrix Wm was not precomputed.")
        if self.k > self.N - 1:
            raise ValueError("Leave-one-out requires k <= N-1")

        x_sorted, inv = self._sort_with_inverse_rank(x)

        u = x_sorted[:-1]
        v = x_sorted[1:]

        P1 = u[:, None] * self.Wm
        P2 = v[:, None] * self.Wm

        pref1 = np.cumsum(P1, axis=0)
        pref1_excl = np.vstack([np.zeros((1, self.k), dtype=P1.dtype), pref1])

        pref2 = np.cumsum(P2, axis=0)
        total2 = pref2[-1:]
        pref2_before = np.vstack([np.zeros((1, self.k), dtype=P2.dtype), pref2])
        suffix2 = total2 - pref2_before

        E_by_rank = pref1_excl + suffix2
        return E_by_rank[inv, :]

    # -------- L-statistics (reward transforms) --------

    def lstat_weight_by_rank(self, a: np.ndarray) -> np.ndarray:
        """Return rank-weight vector w of shape (N,) such that E[T(S)] = x_sorted @ w."""
        a = np.asarray(a)
        if a.shape != (self.k,):
            raise ValueError(f"a must be shape ({self.k},)")
        return self.W @ a

    def lstat_weight_by_item(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Return item-weight vector in original index order (gradient away from ties)."""
        _, inv = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return w_rank[inv]

    def expected_lstat(self, x: np.ndarray, a: np.ndarray) -> float:
        """Return E[T(S)] where T(S)=sum_j a_j X_(j:k)."""
        x_sorted, _ = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return float(x_sorted @ w_rank)

    def expected_lstat_inclusion(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Return E[T(S) | i included] for all i. Shape (N,)."""
        E_inc = self.expected_orderstats_inclusion(x)
        a = np.asarray(a)
        if a.shape != (self.k,):
            raise ValueError(f"a must be shape ({self.k},)")
        return E_inc @ a

    def expected_lstat_leave_one_out(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Return E[T(S)] on population with i removed, for all i. Shape (N,)."""
        E_loo = self.expected_orderstats_leave_one_out(x)
        a = np.asarray(a)
        if a.shape != (self.k,):
            raise ValueError(f"a must be shape ({self.k},)")
        return E_loo @ a


    def expected_lstat_advantage(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
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
