from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln
except Exception as e:  # pragma: no cover
    raise ImportError(
        "ordergrad.jax_backend requires JAX + jaxlib. "
        "Install them with: `pip install ordergrad[jax]`"
    ) from e


def _log_choose(n: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    """Vectorized log binomial log C(n,r) with -inf for invalid."""
    n = jnp.asarray(n)
    r = jnp.asarray(r)
    valid = (n >= 0) & (r >= 0) & (r <= n)

    n0 = jnp.where(valid, n, 0)
    r0 = jnp.where(valid, r, 0)
    nr0 = jnp.where(valid, n0 - r0, 0)

    out = gammaln(n0 + 1.0) - gammaln(r0 + 1.0) - gammaln(nr0 + 1.0)
    out = jnp.where(valid, out, -jnp.inf)
    return out


def _build_weight_matrix(
    N_rows: int,
    k: int,
    log_den: jnp.ndarray,
    log_term_fn,
    *,
    dtype,
    renormalize_cols: bool = False,
) -> jnp.ndarray:
    j = jnp.arange(1, k + 1, dtype=jnp.int32)[None, :]
    m = jnp.arange(1, N_rows + 1, dtype=jnp.int32)[:, None]
    logw = log_term_fn(m, j) - log_den
    W = jnp.exp(logw).astype(dtype)
    if renormalize_cols:
        W = W / jnp.sum(W, axis=0, keepdims=True)
    return W


def precompute_W_unconditional(N: int, k: int, *, dtype=jnp.float64) -> jnp.ndarray:
    """W[m-1,j-1] = P(X_(j:k) == x_(m)) for uniform k-subset from N."""
    if not (1 <= k <= N):
        raise ValueError("Require 1 <= k <= N")
    log_den = _log_choose(N, k)

    def log_term(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m, k - j)

    return _build_weight_matrix(N, k, log_den, log_term, dtype=dtype, renormalize_cols=True)


def precompute_ABC_conditional_including_rank(N: int, k: int, *, dtype=jnp.float64) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute conditional matrices A,B,C (shape (N,k))."""
    if not (1 <= k <= N):
        raise ValueError("Require 1 <= k <= N")
    log_den = _log_choose(N - 1, k - 1)

    def logA(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m - 1, k - j - 1)

    def logB(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m, k - j)

    def logC(m, j):
        return _log_choose(m - 2, j - 2) + _log_choose(N - m, k - j)

    A = _build_weight_matrix(N, k, log_den, logA, dtype=dtype, renormalize_cols=False)
    B = _build_weight_matrix(N, k, log_den, logB, dtype=dtype, renormalize_cols=False)
    C = _build_weight_matrix(N, k, log_den, logC, dtype=dtype, renormalize_cols=False)
    return A, B, C


def precompute_W_leave_one_out(N: int, k: int, *, dtype=jnp.float64) -> jnp.ndarray:
    """Wm for reduced population size (N-1). Shape (N-1,k), columns sum to 1."""
    if not (1 <= k <= N - 1):
        raise ValueError("Require 1 <= k <= N-1 for leave-one-out")
    log_den = _log_choose(N - 1, k)

    def log_term(p, j):
        return _log_choose(p - 1, j - 1) + _log_choose((N - 1) - p, k - j)

    return _build_weight_matrix(N - 1, k, log_den, log_term, dtype=dtype, renormalize_cols=True)


def precompute_W_known_rank_position(N: int, k: int, p: int, *, dtype=jnp.float64) -> jnp.ndarray:
    """Weights for E[X_(j:k) | rank r included and has sample position p]."""
    if not (1 <= k <= N):
        raise ValueError("Require 1 <= k <= N")
    if not (1 <= p <= k):
        raise ValueError("Require 1 <= p <= k")

    r = jnp.arange(1, N + 1, dtype=jnp.int32)[:, None, None]
    m = jnp.arange(1, N + 1, dtype=jnp.int32)[None, :, None]
    j = jnp.arange(1, k + 1, dtype=jnp.int32)[None, None, :]

    K = jnp.zeros((N, N, k), dtype=dtype)

    valid_low = (m < r) & (j < p) & ((r - 1) >= (p - 1))
    log_low = _log_choose(m - 1, j - 1) + _log_choose(r - m - 1, (p - 1) - j) - _log_choose(r - 1, p - 1)
    log_low = jnp.where(valid_low, log_low, -jnp.inf)
    K = jnp.where(valid_low, jnp.exp(log_low).astype(dtype), K)

    K = K.at[:, :, p - 1].set((m[:, :, 0] == r[:, :, 0]).astype(dtype))

    valid_high = (m > r) & (j > p) & ((N - r) >= (k - p))
    q = j - p
    log_high = _log_choose(m - r - 1, q - 1) + _log_choose(N - m, (k - p) - q) - _log_choose(N - r, k - p)
    log_high = jnp.where(valid_high, log_high, -jnp.inf)
    K = jnp.where(valid_high, jnp.exp(log_high).astype(dtype), K)
    return K


@dataclass(frozen=True)
class OrderStatTransform:
    """JAX implementation.

    Returned inclusion/leave-one-out matrices/vectors are in **original index order**.
    """

    N: int
    k: int
    W: jnp.ndarray
    A: Optional[jnp.ndarray]
    B: Optional[jnp.ndarray]
    C: Optional[jnp.ndarray]
    Wm: Optional[jnp.ndarray]
    M_inc: Optional[jnp.ndarray] = None
    M_loo: Optional[jnp.ndarray] = None
    M_adv: Optional[jnp.ndarray] = None
    M_inc_a: Optional[jnp.ndarray] = None
    M_loo_a: Optional[jnp.ndarray] = None
    M_adv_a: Optional[jnp.ndarray] = None

    @classmethod
    def precompute(
        cls,
        N: int,
        k: int,
        *,
        dtype=jnp.float64,
        compute_conditional: bool = True,
        compute_leave_one_out: bool = True,
        compute_dense_matrices: bool = False,
    ) -> "OrderStatTransform":
        W = precompute_W_unconditional(N, k, dtype=dtype)

        A = B = C = None
        if compute_conditional:
            A, B, C = precompute_ABC_conditional_including_rank(N, k, dtype=dtype)

        Wm = None
        if compute_leave_one_out:
            if k > N - 1:
                raise ValueError("Leave-one-out requires k <= N-1")
            Wm = precompute_W_leave_one_out(N, k, dtype=dtype)

        M_inc = M_loo = M_adv = None
        if compute_dense_matrices:
            M_inc = cls._build_dense_inclusion_matrix(A, B, C) if (A is not None and B is not None and C is not None) else None
            M_loo = cls._build_dense_leave_one_out_matrix(Wm, N, k) if Wm is not None else None
            if M_inc is not None and M_loo is not None:
                M_adv = M_inc - M_loo

        return cls(N=N, k=k, W=W, A=A, B=B, C=C, Wm=Wm, M_inc=M_inc, M_loo=M_loo, M_adv=M_adv)

    @staticmethod
    def _build_dense_inclusion_matrix(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
        N, _ = A.shape
        r = jnp.arange(N)[:, None]
        m = jnp.arange(N)[None, :]
        lt = (m < r)[:, :, None]
        eq = (m == r)[:, :, None]
        gt = (m > r)[:, :, None]
        return lt * A[None, :, :] + eq * B[None, :, :] + gt * C[None, :, :]

    @staticmethod
    def _build_dense_leave_one_out_matrix(Wm: jnp.ndarray, N: int, k: int) -> jnp.ndarray:
        M = jnp.zeros((N, N, k), dtype=Wm.dtype)
        for r in range(N):
            if r > 0:
                M = M.at[r, :r, :].set(Wm[:r, :])
            if r < N - 1:
                M = M.at[r, r + 1 :, :].set(Wm[r:, :])
        return M

    def _sort_with_inverse_rank(self, x: jnp.ndarray):
        x = jnp.asarray(x)
        if x.ndim != 1 or x.shape[0] != self.N:
            raise ValueError(f"x must be shape ({self.N},)")
        # Newer JAX releases intentionally do **not** support NumPy's `kind=`
        # argument. Instead, JAX uses `stable=` to control stability.
        #
        # Use a small compatibility shim so this backend works across a wider
        # range of JAX versions.
        try:
            perm = jnp.argsort(x, stable=True)
        except TypeError:  # pragma: no cover
            perm = jnp.argsort(x, kind="stable")
        x_sorted = x[perm]
        inv = jnp.empty_like(perm)
        inv = inv.at[perm].set(jnp.arange(self.N, dtype=perm.dtype))
        return x_sorted, inv

    # -------- order-statistics expectations --------

    def expected_orderstats(self, x: jnp.ndarray) -> jnp.ndarray:
        x_sorted, _ = self._sort_with_inverse_rank(x)
        return x_sorted @ self.W

    def expected_orderstats_inclusion(self, x: jnp.ndarray, *, method: str = "efficient") -> jnp.ndarray:
        if method not in {"efficient", "matmul", "auto"}:
            raise ValueError("method must be one of {'efficient','matmul','auto'}")

        x_sorted, inv = self._sort_with_inverse_rank(x)
        if method in {"matmul", "auto"} and self.M_inc is not None:
            return jnp.einsum("rmj,m->rj", self.M_inc, x_sorted)[inv, :]

        if self.A is None or self.B is None or self.C is None:
            raise ValueError("Conditional matrices A,B,C were not precomputed.")

        XA = x_sorted[:, None] * self.A
        XC = x_sorted[:, None] * self.C

        prefA = jnp.cumsum(XA, axis=0)
        prefA_excl = jnp.concatenate([jnp.zeros((1, self.k), dtype=XA.dtype), prefA[:-1]], axis=0)

        prefC = jnp.cumsum(XC, axis=0)
        totalC = prefC[-1:]
        suffC_excl = totalC - prefC

        diag = x_sorted[:, None] * self.B
        E_by_rank = prefA_excl + diag + suffC_excl
        return E_by_rank[inv, :]

    def expected_orderstats_leave_one_out(self, x: jnp.ndarray, *, method: str = "efficient") -> jnp.ndarray:
        if method not in {"efficient", "matmul", "auto"}:
            raise ValueError("method must be one of {'efficient','matmul','auto'}")

        x_sorted, inv = self._sort_with_inverse_rank(x)
        if method in {"matmul", "auto"} and self.M_loo is not None:
            return jnp.einsum("rmj,m->rj", self.M_loo, x_sorted)[inv, :]

        if self.Wm is None:
            raise ValueError("Leave-one-out matrix Wm was not precomputed.")
        if self.k > self.N - 1:
            raise ValueError("Leave-one-out requires k <= N-1")
        u = x_sorted[:-1]
        v = x_sorted[1:]

        P1 = u[:, None] * self.Wm
        P2 = v[:, None] * self.Wm

        pref1 = jnp.cumsum(P1, axis=0)
        pref1_excl = jnp.concatenate([jnp.zeros((1, self.k), dtype=P1.dtype), pref1], axis=0)

        pref2 = jnp.cumsum(P2, axis=0)
        total2 = pref2[-1:]
        pref2_before = jnp.concatenate([jnp.zeros((1, self.k), dtype=P2.dtype), pref2], axis=0)
        suffix2 = total2 - pref2_before

        E_by_rank = pref1_excl + suffix2
        return E_by_rank[inv, :]

    # -------- L-statistics (reward transforms) --------

    def lstat_weight_by_rank(self, a: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if a is None:
            if not hasattr(self, "Wa") or self.Wa is None:
                raise ValueError("No preweighted l-statistic vector is available. Pass a or use with_lstat_weights().")
            return self.Wa
        a = jnp.asarray(a)
        if a.shape != (self.k,):
            raise ValueError(f"a must be shape ({self.k},)")
        return self.W @ a

    def lstat_weight_by_item(self, x: jnp.ndarray, a: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        _, inv = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return w_rank[inv]

    def expected_lstat(self, x: jnp.ndarray, a: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        x_sorted, _ = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return jnp.sum(x_sorted * w_rank)

    def with_lstat_weights(self, a: jnp.ndarray) -> "OrderStatTransform":
        a = jnp.asarray(a)
        if a.shape != (self.k,):
            raise ValueError(f"a must be shape ({self.k},)")
        out = self.__class__(
            N=self.N,
            k=self.k,
            W=self.W,
            A=self.A,
            B=self.B,
            C=self.C,
            Wm=self.Wm,
            M_inc=self.M_inc,
            M_loo=self.M_loo,
            M_adv=self.M_adv,
        )
        object.__setattr__(out, "Wa", self.W @ a)
        if self.A is not None and self.B is not None and self.C is not None:
            object.__setattr__(out, "Aa", self.A @ a)
            object.__setattr__(out, "Ba", self.B @ a)
            object.__setattr__(out, "Ca", self.C @ a)
        else:
            object.__setattr__(out, "Aa", None)
            object.__setattr__(out, "Ba", None)
            object.__setattr__(out, "Ca", None)
        object.__setattr__(out, "Wma", self.Wm @ a if self.Wm is not None else None)
        object.__setattr__(out, "M_inc_a", jnp.tensordot(self.M_inc, a, axes=([2], [0])) if self.M_inc is not None else None)
        object.__setattr__(out, "M_loo_a", jnp.tensordot(self.M_loo, a, axes=([2], [0])) if self.M_loo is not None else None)
        object.__setattr__(out, "M_adv_a", jnp.tensordot(self.M_adv, a, axes=([2], [0])) if self.M_adv is not None else None)
        return out

    @classmethod
    def precompute_lstat(cls, N: int, k: int, a: jnp.ndarray, **kwargs) -> "OrderStatTransform":
        return cls.precompute(N, k, **kwargs).with_lstat_weights(a)

    def expected_lstat_inclusion(self, x: jnp.ndarray, a: Optional[jnp.ndarray] = None, *, method: str = "efficient") -> jnp.ndarray:
        if a is None and hasattr(self, "Aa") and self.Aa is not None and self.Ba is not None and self.Ca is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            xa = x_sorted * self.Aa
            pref_a = jnp.cumsum(xa, axis=0)
            pref_a_excl = jnp.concatenate([jnp.zeros((1,), dtype=xa.dtype), pref_a[:-1]], axis=0)
            xc = x_sorted * self.Ca
            pref_c = jnp.cumsum(xc, axis=0)
            inc = pref_a_excl + (x_sorted * self.Ba) + (pref_c[-1] - pref_c)
            return inc[inv]
        if method in {"matmul", "auto"} and self.M_inc_a is not None and a is None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return (self.M_inc_a @ x_sorted)[inv]
        E_inc = self.expected_orderstats_inclusion(x, method=method)
        if a is None:
            raise ValueError(f"a must be shape ({self.k},)")
        a = jnp.asarray(a)
        if a.shape != (self.k,):
            raise ValueError(f"a must be shape ({self.k},)")
        return E_inc @ a

    def expected_lstat_leave_one_out(self, x: jnp.ndarray, a: Optional[jnp.ndarray] = None, *, method: str = "efficient") -> jnp.ndarray:
        if a is None and hasattr(self, "Wma") and self.Wma is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            p1 = x_sorted[:-1] * self.Wma
            p2 = x_sorted[1:] * self.Wma
            pref1 = jnp.cumsum(p1, axis=0)
            left = jnp.concatenate([jnp.zeros((1,), dtype=p1.dtype), pref1], axis=0)
            pref2 = jnp.cumsum(p2, axis=0)
            right = pref2[-1] - jnp.concatenate([jnp.zeros((1,), dtype=p2.dtype), pref2], axis=0)
            return (left + right)[inv]
        if method in {"matmul", "auto"} and self.M_loo_a is not None and a is None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return (self.M_loo_a @ x_sorted)[inv]
        E_loo = self.expected_orderstats_leave_one_out(x, method=method)
        if a is None:
            raise ValueError(f"a must be shape ({self.k},)")
        a = jnp.asarray(a)
        if a.shape != (self.k,):
            raise ValueError(f"a must be shape ({self.k},)")
        return E_loo @ a

    def expected_orderstats_advantage(self, x: jnp.ndarray, *, method: str = "efficient") -> jnp.ndarray:
        if method not in {"efficient", "matmul", "auto"}:
            raise ValueError("method must be one of {'efficient','matmul','auto'}")
        if method in {"matmul", "auto"} and self.M_adv is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return jnp.einsum("rmj,m->rj", self.M_adv, x_sorted)[inv, :]
        return self.expected_orderstats_inclusion(x, method=method) - self.expected_orderstats_leave_one_out(x, method=method)

    def expected_orderstats_known_rank_position(self, x: jnp.ndarray, p: int) -> jnp.ndarray:
        """Return E[X_(j:k) | i included and has sample position p] for all i."""
        K = precompute_W_known_rank_position(self.N, self.k, p, dtype=self.W.dtype)
        x_sorted, inv = self._sort_with_inverse_rank(x)
        return jnp.einsum("rmj,m->rj", K, x_sorted)[inv, :]

    def expected_lstat_known_rank_position(self, x: jnp.ndarray, a: jnp.ndarray, p: int) -> jnp.ndarray:
        """Return E[T(S) | i included and has sample position p] for all i."""
        a = jnp.asarray(a)
        if a.shape != (self.k,):
            raise ValueError(f"a must be shape ({self.k},)")
        return self.expected_orderstats_known_rank_position(x, p) @ a

    def expected_lstat_advantage(self, x: jnp.ndarray, a: Optional[jnp.ndarray] = None, *, method: str = "efficient") -> jnp.ndarray:
        if method in {"matmul", "auto"} and self.M_adv_a is not None and a is None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return (self.M_adv_a @ x_sorted)[inv]
        return self.expected_lstat_inclusion(x, a, method=method) - self.expected_lstat_leave_one_out(x, a, method=method)
