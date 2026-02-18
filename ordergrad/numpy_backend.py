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
    Wa: Optional[np.ndarray] = None  # (N,)
    Aa: Optional[np.ndarray] = None  # (N,)
    Ba: Optional[np.ndarray] = None  # (N,)
    Ca: Optional[np.ndarray] = None  # (N,)
    Wma: Optional[np.ndarray] = None  # (N-1,)
    M_inc: Optional[np.ndarray] = None  # (N,N,k)
    M_loo: Optional[np.ndarray] = None  # (N,N,k)
    M_adv: Optional[np.ndarray] = None  # (N,N,k)
    M_inc_a: Optional[np.ndarray] = None  # (N,N)
    M_loo_a: Optional[np.ndarray] = None  # (N,N)
    M_adv_a: Optional[np.ndarray] = None  # (N,N)

    @staticmethod
    def _validate_a(a: Optional[np.ndarray], k: int) -> np.ndarray:
        if a is None:
            raise ValueError("a is required unless using a transform precomputed with l-statistic weights.")
        a = np.asarray(a)
        if a.shape != (k,):
            raise ValueError(f"a must be shape ({k},)")
        return a

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
        compute_dense_matrices: bool = False,
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

        M_inc = M_loo = M_adv = None
        if compute_dense_matrices:
            M_inc = cls._build_dense_inclusion_matrix(A, B, C) if (A is not None and B is not None and C is not None) else None
            M_loo = cls._build_dense_leave_one_out_matrix(Wm, N, k) if Wm is not None else None
            if M_inc is not None and M_loo is not None:
                M_adv = M_inc - M_loo

        return cls(N=N, k=k, W=W, A=A, B=B, C=C, Wm=Wm, M_inc=M_inc, M_loo=M_loo, M_adv=M_adv)

    def with_lstat_weights(self, a: np.ndarray) -> "OrderStatTransform":
        """Return a new transform with preweighted L-statistic coefficients."""
        a = self._validate_a(a, self.k)
        Wa = self.W @ a
        Aa = Ba = Ca = Wma = None
        if self.A is not None and self.B is not None and self.C is not None:
            Aa = self.A @ a
            Ba = self.B @ a
            Ca = self.C @ a
        if self.Wm is not None:
            Wma = self.Wm @ a
        return OrderStatTransform(
            N=self.N,
            k=self.k,
            W=self.W,
            A=self.A,
            B=self.B,
            C=self.C,
            Wm=self.Wm,
            Wa=Wa,
            Aa=Aa,
            Ba=Ba,
            Ca=Ca,
            Wma=Wma,
            M_inc=self.M_inc,
            M_loo=self.M_loo,
            M_adv=self.M_adv,
            M_inc_a=(np.tensordot(self.M_inc, a, axes=([2], [0])) if self.M_inc is not None else None),
            M_loo_a=(np.tensordot(self.M_loo, a, axes=([2], [0])) if self.M_loo is not None else None),
            M_adv_a=(np.tensordot(self.M_adv, a, axes=([2], [0])) if self.M_adv is not None else None),
        )

    @classmethod
    def precompute_lstat(cls, N: int, k: int, a: np.ndarray, **kwargs) -> "OrderStatTransform":
        """Precompute matrices and pre-apply L-statistic weights `a`."""
        return cls.precompute(N, k, **kwargs).with_lstat_weights(a)

    @staticmethod
    def _build_dense_inclusion_matrix(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Build dense inclusion map M_inc[r,m,j] for rank-space matmul eval."""
        N, k = A.shape
        r = np.arange(N)[:, None]
        m = np.arange(N)[None, :]
        lt = (m < r)[:, :, None]
        eq = (m == r)[:, :, None]
        gt = (m > r)[:, :, None]
        return lt * A[None, :, :] + eq * B[None, :, :] + gt * C[None, :, :]

    @staticmethod
    def _build_dense_leave_one_out_matrix(Wm: np.ndarray, N: int, k: int) -> np.ndarray:
        """Build dense leave-one-out map M_loo[r,m,j] for rank-space matmul eval."""
        M = np.zeros((N, N, k), dtype=Wm.dtype)
        for r in range(N):
            if r > 0:
                M[r, :r, :] = Wm[:r, :]
            if r < N - 1:
                M[r, r + 1 :, :] = Wm[r:, :]
        return M

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

    def expected_orderstats_inclusion(self, x: np.ndarray, *, method: str = "efficient") -> np.ndarray:
        """Return E[X_(j:k) | i included] for all i. Shape (N,k)."""
        if method not in {"efficient", "matmul", "auto"}:
            raise ValueError("method must be one of {'efficient','matmul','auto'}")

        x_sorted, inv = self._sort_with_inverse_rank(x)
        if method in {"matmul", "auto"} and self.M_inc is not None:
            E_by_rank = np.einsum("rmj,m->rj", self.M_inc, x_sorted)
            return E_by_rank[inv, :]

        if self.A is None or self.B is None or self.C is None:
            raise ValueError("Conditional matrices A,B,C were not precomputed.")

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

    def expected_orderstats_leave_one_out(self, x: np.ndarray, *, method: str = "efficient") -> np.ndarray:
        """Return E[X_(j:k)] on population with i removed. Shape (N,k)."""
        if method not in {"efficient", "matmul", "auto"}:
            raise ValueError("method must be one of {'efficient','matmul','auto'}")

        x_sorted, inv = self._sort_with_inverse_rank(x)
        if method in {"matmul", "auto"} and self.M_loo is not None:
            E_by_rank = np.einsum("rmj,m->rj", self.M_loo, x_sorted)
            return E_by_rank[inv, :]

        if self.Wm is None:
            raise ValueError("Leave-one-out matrix Wm was not precomputed.")
        if self.k > self.N - 1:
            raise ValueError("Leave-one-out requires k <= N-1")

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

    def lstat_weight_by_rank(self, a: Optional[np.ndarray] = None) -> np.ndarray:
        """Return rank-weight vector w of shape (N,) such that E[T(S)] = x_sorted @ w."""
        if a is None:
            if self.Wa is None:
                raise ValueError("No preweighted l-statistic vector is available. Pass a or use with_lstat_weights().")
            return self.Wa
        a = self._validate_a(a, self.k)
        return self.W @ a

    def lstat_weight_by_item(self, x: np.ndarray, a: Optional[np.ndarray] = None) -> np.ndarray:
        """Return item-weight vector in original index order (gradient away from ties)."""
        _, inv = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return w_rank[inv]

    def expected_lstat(self, x: np.ndarray, a: Optional[np.ndarray] = None) -> float:
        """Return E[T(S)] where T(S)=sum_j a_j X_(j:k)."""
        x_sorted, _ = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return float(x_sorted @ w_rank)

    def expected_lstat_inclusion(self, x: np.ndarray, a: Optional[np.ndarray] = None, *, method: str = "efficient") -> np.ndarray:
        """Return E[T(S) | i included] for all i. Shape (N,)."""
        if a is None and self.Aa is not None and self.Ba is not None and self.Ca is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return self._expected_lstat_inclusion_by_rank(x_sorted)[inv]
        if method in {"matmul", "auto"} and self.M_inc_a is not None and a is None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return (self.M_inc_a @ x_sorted)[inv]
        E_inc = self.expected_orderstats_inclusion(x, method=method)
        return E_inc @ self._validate_a(a, self.k)

    def expected_lstat_leave_one_out(self, x: np.ndarray, a: Optional[np.ndarray] = None, *, method: str = "efficient") -> np.ndarray:
        """Return E[T(S)] on population with i removed, for all i. Shape (N,)."""
        if a is None and self.Wma is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return self._expected_lstat_leave_one_out_by_rank(x_sorted)[inv]
        if method in {"matmul", "auto"} and self.M_loo_a is not None and a is None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return (self.M_loo_a @ x_sorted)[inv]
        E_loo = self.expected_orderstats_leave_one_out(x, method=method)
        return E_loo @ self._validate_a(a, self.k)

    def _expected_lstat_inclusion_by_rank(self, x_sorted: np.ndarray) -> np.ndarray:
        XA = x_sorted * self.Aa
        prefA = np.cumsum(XA)
        prefA_excl = np.concatenate([np.zeros(1, dtype=XA.dtype), prefA[:-1]])
        XC = x_sorted * self.Ca
        prefC = np.cumsum(XC)
        suffC_excl = prefC[-1] - prefC
        diag = x_sorted * self.Ba
        return prefA_excl + diag + suffC_excl

    def _expected_lstat_leave_one_out_by_rank(self, x_sorted: np.ndarray) -> np.ndarray:
        u = x_sorted[:-1]
        v = x_sorted[1:]
        p1 = u * self.Wma
        p2 = v * self.Wma
        pref1 = np.cumsum(p1)
        pref1_excl = np.concatenate([np.zeros(1, dtype=p1.dtype), pref1])
        pref2 = np.cumsum(p2)
        pref2_before = np.concatenate([np.zeros(1, dtype=p2.dtype), pref2])
        suffix2 = pref2[-1] - pref2_before
        return pref1_excl + suffix2

    def expected_orderstats_advantage(self, x: np.ndarray, *, method: str = "efficient") -> np.ndarray:
        """Return E[X_(j:k)|i included] - E[X_(j:k)] on population with i removed. Shape (N,k)."""
        if method not in {"efficient", "matmul", "auto"}:
            raise ValueError("method must be one of {'efficient','matmul','auto'}")
        if method in {"matmul", "auto"} and self.M_adv is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return np.einsum("rmj,m->rj", self.M_adv, x_sorted)[inv, :]
        return self.expected_orderstats_inclusion(x, method=method) - self.expected_orderstats_leave_one_out(x, method=method)


    def expected_lstat_advantage(self, x: np.ndarray, a: Optional[np.ndarray] = None, *, method: str = "efficient") -> np.ndarray:
        """Convenience: per-item advantage-style transform.

        Defined as:
            adv[i] = E[T(S) | i included] - E[T(S)] on population with i removed.

        Shape: (N,)
        """
        if a is None and self.Aa is not None and self.Ba is not None and self.Ca is not None and self.Wma is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            adv_by_rank = self._expected_lstat_inclusion_by_rank(x_sorted) - self._expected_lstat_leave_one_out_by_rank(x_sorted)
            return adv_by_rank[inv]
        if method in {"matmul", "auto"} and self.M_adv_a is not None and a is None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return (self.M_adv_a @ x_sorted)[inv]
        return self.expected_lstat_inclusion(x, a, method=method) - self.expected_lstat_leave_one_out(x, a, method=method)
