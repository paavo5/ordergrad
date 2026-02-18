from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln
except Exception as e:  # pragma: no cover
    raise ImportError(
        "orderstat_reward.jax_backend requires JAX + jaxlib. "
        "Install them with: `pip install orderstat-reward[jax]`"
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

    @classmethod
    def precompute(
        cls,
        N: int,
        k: int,
        *,
        dtype=jnp.float64,
        compute_conditional: bool = True,
        compute_leave_one_out: bool = True,
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

        return cls(N=N, k=k, W=W, A=A, B=B, C=C, Wm=Wm)

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

    def expected_orderstats_inclusion(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.A is None or self.B is None or self.C is None:
            raise ValueError("Conditional matrices A,B,C were not precomputed.")

        x_sorted, inv = self._sort_with_inverse_rank(x)

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

    def expected_orderstats_leave_one_out(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.Wm is None:
            raise ValueError("Leave-one-out matrix Wm was not precomputed.")
        if self.k > self.N - 1:
            raise ValueError("Leave-one-out requires k <= N-1")

        x_sorted, inv = self._sort_with_inverse_rank(x)
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

    def lstat_weight_by_rank(self, a: jnp.ndarray) -> jnp.ndarray:
        a = jnp.asarray(a)
        if a.shape != (self.k,):
            raise ValueError(f"a must be shape ({self.k},)")
        return self.W @ a

    def lstat_weight_by_item(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        _, inv = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return w_rank[inv]

    def expected_lstat(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        x_sorted, _ = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return jnp.sum(x_sorted * w_rank)

    def expected_lstat_inclusion(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        E_inc = self.expected_orderstats_inclusion(x)
        return E_inc @ a

    def expected_lstat_leave_one_out(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        E_loo = self.expected_orderstats_leave_one_out(x)
        return E_loo @ a


    def expected_lstat_advantage(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
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
