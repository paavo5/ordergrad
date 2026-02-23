import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Tuple


# -----------------------------
# Combinatorics helpers
# -----------------------------

import math


def _log_gamma_np(x):
    """Vectorized log-gamma using Python's math.lgamma."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        return np.array(math.lgamma(float(x)), dtype=np.float64)
    return np.vectorize(math.lgamma, otypes=[np.float64])(x)


def _log_choose(n, r):
    """Vectorized log binomial log C(n,r) via log-gamma for real-valued n,r.

    Returns -inf where invalid (r<0, r>n, n<0).
    """
    n = np.asarray(n, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    valid = (n >= 0.0) & (r >= 0.0) & (r <= n)

    n0 = np.where(valid, n, 1.0)
    r0 = np.where(valid, r, 0.0)
    nr0 = np.where(valid, n - r, 1.0)

    out = _log_gamma_np(n0 + 1.0) - _log_gamma_np(r0 + 1.0) - _log_gamma_np(nr0 + 1.0)
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
        np.divide(out, colsum, out=out, where=colsum > 0)

    return out


# -----------------------------
# Weight precomputation
# -----------------------------


def precompute_W_unconditional(
    N: int,
    k: float,
    *,
    dtype=np.float64,
    chunk_size: Optional[int] = None
) -> np.ndarray:
    """W[m-1,j-1] = P(X_(j:k) == x_(m)) for uniform k-subset from N.

    Shape: (N,k). Columns sum to 1.
    """
    if not (1 <= k <= N):
        raise ValueError("Require real k with 1 <= k <= N")
    k_eff = float(k)
    k_ord = int(np.floor(k_eff))

    log_den = _log_choose(float(N), k_eff)  # log C(N,k)

    def log_term(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m, k_eff - j)

    return _build_weight_matrix(
        k_ord,
        log_den,
        log_term,
        out_shape=(N, k_ord),
        dtype=dtype,
        chunk_size=chunk_size,
        renormalize_cols=True,
    )


def precompute_ABC_conditional_including_rank(
    N: int,
    k: float,
    *,
    dtype=np.float64,
    chunk_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute conditional matrices A,B,C (all shape (N,k)).

    These correspond to the three cases in the conditional pmf given inclusion of rank r.
    """
    if not (1 <= k <= N):
        raise ValueError("Require real k with 1 <= k <= N")
    k_eff = float(k)
    k_ord = int(np.floor(k_eff))

    log_den = _log_choose(float(N - 1), k_eff - 1.0)  # log C(N-1,k-1)

    def logA(m, j):
        # m<r case weights
        return _log_choose(m - 1, j - 1) + _log_choose(N - m - 1, k_eff - j - 1.0)

    def logB(m, j):
        # m==r diagonal case
        return _log_choose(m - 1, j - 1) + _log_choose(N - m, k_eff - j)

    def logC(m, j):
        # m>r case weights
        return _log_choose(m - 2, j - 2) + _log_choose(N - m, k_eff - j)

    A = _build_weight_matrix(
        k_ord,
        log_den,
        logA,
        out_shape=(N, k_ord),
        dtype=dtype,
        chunk_size=chunk_size,
        renormalize_cols=False,
    )
    B = _build_weight_matrix(
        k_ord,
        log_den,
        logB,
        out_shape=(N, k_ord),
        dtype=dtype,
        chunk_size=chunk_size,
        renormalize_cols=False,
    )
    C = _build_weight_matrix(
        k_ord,
        log_den,
        logC,
        out_shape=(N, k_ord),
        dtype=dtype,
        chunk_size=chunk_size,
        renormalize_cols=False,
    )
    return A, B, C


def precompute_W_leave_one_out(
    N: int,
    k: float,
    *,
    dtype=np.float64,
    chunk_size: Optional[int] = None
) -> np.ndarray:
    """Wm[p-1,j-1] = P(order stat equals p-th smallest) for a population of size (N-1).

    Shape: (N-1,k). Columns sum to 1. Requires k <= N-1.
    """
    if not (1 <= k <= N - 1):
        raise ValueError("Require real k with 1 <= k <= N-1 for leave-one-out")
    k_eff = float(k)
    k_ord = int(np.floor(k_eff))

    log_den = _log_choose(float(N - 1), k_eff)  # log C(N-1,k)

    def log_term(p, j):
        return _log_choose(p - 1, j - 1) + _log_choose((N - 1) - p, k_eff - j)

    return _build_weight_matrix(
        k_ord,
        log_den,
        log_term,
        out_shape=(N - 1, k_ord),
        dtype=dtype,
        chunk_size=chunk_size,
        renormalize_cols=True,
    )


def _binom_tail_table(k: float, F: np.ndarray, k_ord: int, *, dtype=np.float64) -> np.ndarray:
    """T[t,j-1] = P(Bin(k,F_t) >= j), with t=0..m and j=1..k_ord."""
    F = np.asarray(F, dtype=np.float64)
    s = np.arange(0, k_ord + 1, dtype=np.float64)[:, None]  # (k_ord+1,1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logF = np.where(F[None, :] > 0, np.log(F[None, :]), 0.0)
        log1mF = np.where(F[None, :] < 1, np.log1p(-F[None, :]), 0.0)
        term1 = np.where(s > 0, np.where(F[None, :] > 0, s * logF, -np.inf), 0.0)
        term2 = np.where((float(k) - s) > 0, np.where(F[None, :] < 1, (float(k) - s) * log1mF, -np.inf), 0.0)
        logpmf = _log_choose(float(k), s) + term1 + term2
    pmf = np.exp(logpmf)  # (k_ord+1, m+1)
    cols = [pmf[j:, :].sum(axis=0) for j in range(1, k_ord + 1)]
    return np.stack(cols, axis=1).astype(dtype, copy=False)


def known_rp_orderstats(
    r: np.ndarray,
    p: np.ndarray,
    k: float,
    *,
    return_sorted: bool = False,
    dtype=np.float64,
):
    """Exact known-(r,p) with-replacement order-statistics quantities.

    Returns tuple (v, q, adv):
      - v: (k_ord,) unconditional E[X_(j:k)]
      - q: (m,k_ord) conditional E[X_(j:k) | A1=b]
      - adv: (m,k_ord) = q - v
    where k_ord = floor(k).
    """
    r = np.asarray(r, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if r.ndim != 1 or p.ndim != 1 or r.shape[0] != p.shape[0]:
        raise ValueError("r and p must be 1D arrays of equal length")
    if np.any(p < 0):
        raise ValueError("p must be nonnegative")
    ps = p.sum()
    if not np.isfinite(ps) or ps <= 0:
        raise ValueError("sum(p) must be positive and finite")
    p = p / ps

    m = r.shape[0]
    k_eff = float(k)
    k_ord = int(np.floor(k_eff))
    if not (1 <= k_eff):
        raise ValueError("Require real k >= 1")
    if k_ord < 1:
        raise ValueError("floor(k) must be >= 1")

    perm = np.argsort(r, kind="mergesort")
    inv = np.empty_like(perm)
    inv[perm] = np.arange(m, dtype=perm.dtype)
    rs = r[perm]
    ps = p[perm]

    F = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(ps)])  # (m+1,)
    T_k = _binom_tail_table(k_eff, F, k_ord, dtype=dtype)  # (m+1,k_ord)
    W = T_k[1:, :] - T_k[:-1, :]
    v = rs @ W

    T_km1 = _binom_tail_table(k_eff - 1.0, F, k_ord, dtype=dtype)

    q_sorted = np.empty((m, k_ord), dtype=dtype)
    for b in range(m):
        delta = np.concatenate([np.array([0], dtype=np.int64), (rs[b] <= rs).astype(np.int64)])
        Q = np.zeros((m + 1, k_ord), dtype=np.float64)
        rows = np.arange(m + 1)
        for j in range(1, k_ord + 1):
            need = j - delta
            idx = np.clip(need - 1, 0, k_ord - 1)
            take = T_km1[rows, idx]
            col = np.where(need <= 0, 1.0, np.where(need <= k_ord, take, 0.0))
            Q[:, j - 1] = col
        Wq = Q[1:, :] - Q[:-1, :]
        q_sorted[b, :] = rs @ Wq

    adv_sorted = q_sorted - v[None, :]
    if return_sorted:
        return v.astype(dtype, copy=False), q_sorted.astype(dtype, copy=False), adv_sorted.astype(dtype, copy=False), perm, inv
    return (
        v.astype(dtype, copy=False),
        q_sorted[inv, :].astype(dtype, copy=False),
        adv_sorted[inv, :].astype(dtype, copy=False),
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
    k_eff: float
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
    def _preset_lstat_weights(k: int, spec: str, *, dtype) -> np.ndarray:
        text = str(spec).strip()
        if not text:
            raise ValueError("l-stat preset cannot be empty")
        name, _, m_txt = text.partition(":")
        key = name.strip().lower()

        if key in {"remax", "remin"}:
            if m_txt.strip():
                raise ValueError(f"{name} does not take an m value")
            out = np.zeros((k,), dtype=dtype)
            out[k - 1 if key == "remax" else 0] = 1.0
            return out

        if not m_txt.strip():
            raise ValueError(f"Preset '{name}' requires ':m' (e.g. {name}:3)")
        m = int(m_txt)
        if not (1 <= m <= k):
            raise ValueError(f"m must satisfy 1 <= m <= {k}")

        out = np.zeros((k,), dtype=dtype)
        if key == "topm":
            out[k - m :] = 1.0 / m
        elif key == "botm":
            out[:m] = 1.0 / m
        elif key in {"winsorizedm", "windosrizedm"}:
            if 2 * m >= k:
                raise ValueError(f"WinsorizedM requires 2*m < k (got m={m}, k={k})")
            out[m : k - m] = 1.0 / (k - 2 * m)
        else:
            raise ValueError(
                "Unknown l-stat preset. Supported: TopM:m, BotM:m, WinsorizedM:m, ReMax, ReMin"
            )
        return out

    @classmethod
    def _validate_a(cls, a: Any, k: int, *, dtype=np.float64) -> np.ndarray:
        if a is None:
            raise ValueError("a is required unless using a transform precomputed with l-statistic weights.")
        if isinstance(a, str):
            return cls._preset_lstat_weights(k, a, dtype=dtype)
        a = np.asarray(a, dtype=dtype)
        if a.shape != (k,):
            raise ValueError(f"a must be shape ({k},)")
        return a

    @classmethod
    def precompute(
        cls,
        N: int,
        k: float,
        *,
        dtype=np.float64,
        chunk_size: Optional[int] = None,
        compute_conditional: bool = True,
        compute_leave_one_out: bool = True,
        compute_dense_matrices: bool = False,
    ):
        k_eff = float(k)
        k_ord = int(np.floor(k_eff))
        if not (1 <= k_eff <= N):
            raise ValueError("Require real k with 1 <= k <= N")
        if not (1 <= k_ord <= N):
            raise ValueError("floor(k) must satisfy 1 <= floor(k) <= N")
        W = precompute_W_unconditional(N, k_eff, dtype=dtype, chunk_size=chunk_size)

        A = B = C = None
        if compute_conditional:
            A, B, C = precompute_ABC_conditional_including_rank(
                N, k_eff, dtype=dtype, chunk_size=chunk_size
            )

        Wm = None
        if compute_leave_one_out:
            if k_eff > N - 1:
                raise ValueError("Leave-one-out requires real k <= N-1")
            Wm = precompute_W_leave_one_out(N, k_eff, dtype=dtype, chunk_size=chunk_size)

        M_inc = M_loo = M_adv = None
        if compute_dense_matrices:
            M_inc = cls._build_dense_inclusion_matrix(A, B, C) if (A is not None and B is not None and C is not None) else None
            M_loo = cls._build_dense_leave_one_out_matrix(Wm, N, k_ord) if Wm is not None else None
            if M_inc is not None and M_loo is not None:
                M_adv = M_inc - M_loo

        return cls(N=N, k=k_ord, k_eff=k_eff, W=W, A=A, B=B, C=C, Wm=Wm, M_inc=M_inc, M_loo=M_loo, M_adv=M_adv)

    def with_lstat_weights(self, a: np.ndarray) -> "OrderStatTransform":
        """Return a new transform with preweighted L-statistic coefficients."""
        a = self._validate_a(a, self.k, dtype=self.W.dtype)
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
            k_eff=self.k_eff,
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
        a = self._validate_a(a, self.k, dtype=self.W.dtype)
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
        return E_inc @ self._validate_a(a, self.k, dtype=self.W.dtype)

    def expected_lstat_leave_one_out(self, x: np.ndarray, a: Optional[np.ndarray] = None, *, method: str = "efficient") -> np.ndarray:
        """Return E[T(S)] on population with i removed, for all i. Shape (N,)."""
        if a is None and self.Wma is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return self._expected_lstat_leave_one_out_by_rank(x_sorted)[inv]
        if method in {"matmul", "auto"} and self.M_loo_a is not None and a is None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return (self.M_loo_a @ x_sorted)[inv]
        E_loo = self.expected_orderstats_leave_one_out(x, method=method)
        return E_loo @ self._validate_a(a, self.k, dtype=self.W.dtype)

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

    def expected_orderstats_advantage(self, x: np.ndarray, *, method: str = "efficient", detach_advantage: bool = True) -> np.ndarray:
        """Return E[X_(j:k)|i included] - E[X_(j:k)] on population with i removed. Shape (N,k)."""
        if method not in {"efficient", "matmul", "auto"}:
            raise ValueError("method must be one of {'efficient','matmul','auto'}")
        if method in {"matmul", "auto"} and self.M_adv is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return np.einsum("rmj,m->rj", self.M_adv, x_sorted)[inv, :]
        return self.expected_orderstats_inclusion(x, method=method) - self.expected_orderstats_leave_one_out(x, method=method)

    def expected_orderstats_known_rp(self, r: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Known-(r,p) exact unconditional order-statistics expectation. Shape (k,)."""
        v, _, _ = known_rp_orderstats(r, p, self.k_eff, dtype=self.W.dtype)
        return v

    def expected_orderstats_inclusion_known_rp(self, r: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Known-(r,p) exact conditional E[X_(j:k) | A1=b]. Shape (m,k)."""
        _, q, _ = known_rp_orderstats(r, p, self.k_eff, dtype=self.W.dtype)
        return q

    def expected_orderstats_advantage_known_rp(self, r: np.ndarray, p: np.ndarray, *, detach_advantage: bool = True) -> np.ndarray:
        """Known-(r,p) exact advantage q-v. Shape (m,k)."""
        _, _, adv = known_rp_orderstats(r, p, self.k_eff, dtype=self.W.dtype)
        return adv

    def expected_lstat_known_rp(self, r: np.ndarray, p: np.ndarray, a: np.ndarray) -> float:
        a = self._validate_a(a, self.k, dtype=self.W.dtype)
        return float(self.expected_orderstats_known_rp(r, p) @ a)

    def expected_lstat_inclusion_known_rp(self, r: np.ndarray, p: np.ndarray, a: np.ndarray) -> np.ndarray:
        a = self._validate_a(a, self.k, dtype=self.W.dtype)
        return self.expected_orderstats_inclusion_known_rp(r, p) @ a

    def expected_lstat_advantage_known_rp(self, r: np.ndarray, p: np.ndarray, a: np.ndarray, *, detach_advantage: bool = True) -> np.ndarray:
        a = self._validate_a(a, self.k, dtype=self.W.dtype)
        return self.expected_orderstats_advantage_known_rp(r, p) @ a


    def expected_lstat_advantage(self, x: np.ndarray, a: Optional[np.ndarray] = None, *, method: str = "efficient", detach_advantage: bool = True) -> np.ndarray:
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
