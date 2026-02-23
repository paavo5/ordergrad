from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

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


def precompute_W_unconditional(N: int, k: float, *, dtype=jnp.float64) -> jnp.ndarray:
    """W[m-1,j-1] = P(X_(j:k) == x_(m)) for uniform k-subset from N."""
    if not (1 <= k <= N):
        raise ValueError("Require 1 <= k <= N")
    k_eff = float(k)
    log_den = _log_choose(float(N), k_eff)

    def log_term(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m, k_eff - j)

    return _build_weight_matrix(N, int(float(k)//1), log_den, log_term, dtype=dtype, renormalize_cols=True)


def precompute_ABC_conditional_including_rank(N: int, k: float, *, dtype=jnp.float64) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute conditional matrices A,B,C (shape (N,k))."""
    if not (1 <= k <= N):
        raise ValueError("Require 1 <= k <= N")
    k_eff = float(k)
    log_den = _log_choose(float(N - 1), k_eff - 1.0)

    def logA(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m - 1, k_eff - j - 1.0)

    def logB(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m, k_eff - j)

    def logC(m, j):
        return _log_choose(m - 2, j - 2) + _log_choose(N - m, k_eff - j)

    A = _build_weight_matrix(N, int(float(k)//1), log_den, logA, dtype=dtype, renormalize_cols=False)
    B = _build_weight_matrix(N, int(float(k)//1), log_den, logB, dtype=dtype, renormalize_cols=False)
    C = _build_weight_matrix(N, int(float(k)//1), log_den, logC, dtype=dtype, renormalize_cols=False)
    return A, B, C


def precompute_W_leave_one_out(N: int, k: float, *, dtype=jnp.float64) -> jnp.ndarray:
    """Wm for reduced population size (N-1). Shape (N-1,k), columns sum to 1."""
    if not (1 <= k <= N - 1):
        raise ValueError("Require 1 <= k <= N-1 for leave-one-out")
    k_eff = float(k)
    log_den = _log_choose(float(N - 1), k_eff)

    def log_term(p, j):
        return _log_choose(p - 1, j - 1) + _log_choose((N - 1) - p, k_eff - j)

    return _build_weight_matrix(N - 1, int(float(k)//1), log_den, log_term, dtype=dtype, renormalize_cols=True)


def _binom_tail_table(k: float, F: jnp.ndarray, k_ord: int, *, dtype=jnp.float64) -> jnp.ndarray:
    """T[t,j-1] = P(Bin(k,F_t) >= j), with t=0..m and j=1..k_ord."""
    F = jnp.asarray(F, dtype=jnp.float64)
    s = jnp.arange(0, k_ord + 1, dtype=jnp.float64)[:, None]
    logF = jnp.where(F[None, :] > 0, jnp.log(F[None, :]), 0.0)
    log1mF = jnp.where(F[None, :] < 1, jnp.log1p(-F[None, :]), 0.0)
    term1 = jnp.where(s > 0, jnp.where(F[None, :] > 0, s * logF, -jnp.inf), 0.0)
    term2 = jnp.where((float(k) - s) > 0, jnp.where(F[None, :] < 1, (float(k) - s) * log1mF, -jnp.inf), 0.0)
    logpmf = _log_choose(float(k), s) + term1 + term2
    pmf = jnp.exp(logpmf)
    cols = [jnp.sum(pmf[j:, :], axis=0) for j in range(1, k_ord + 1)]
    return jnp.stack(cols, axis=1).astype(dtype)


def known_rp_orderstats(r: jnp.ndarray, p: jnp.ndarray, k: float, *, dtype=jnp.float64):
    """Exact known-(r,p) with-replacement order-statistics quantities."""
    r = jnp.asarray(r, dtype=jnp.float64)
    p = jnp.asarray(p, dtype=jnp.float64)
    if r.ndim != 1 or p.ndim != 1 or r.shape[0] != p.shape[0]:
        raise ValueError("r and p must be 1D arrays of equal length")
    if jnp.any(p < 0):
        raise ValueError("p must be nonnegative")
    p = p / jnp.sum(p)
    m = int(r.shape[0])
    k_eff = float(k)
    k_ord = int(jnp.floor(k_eff))
    if not (k_eff >= 1):
        raise ValueError("Require real k >= 1")
    if k_ord < 1:
        raise ValueError("floor(k) must be >= 1")

    perm = jnp.argsort(r, stable=True)
    inv = jnp.empty_like(perm)
    inv = inv.at[perm].set(jnp.arange(m, dtype=perm.dtype))
    rs = r[perm]
    ps = p[perm]

    F = jnp.concatenate([jnp.array([0.0], dtype=jnp.float64), jnp.cumsum(ps)])
    T_k = _binom_tail_table(k_eff, F, k_ord, dtype=dtype)
    W = T_k[1:, :] - T_k[:-1, :]
    v = rs @ W

    T_km1 = _binom_tail_table(k_eff - 1.0, F, k_ord, dtype=dtype)
    q_sorted = []
    for b in range(m):
        delta = jnp.concatenate([jnp.array([0], dtype=jnp.int32), (rs[b] <= rs).astype(jnp.int32)])
        cols = []
        rows = jnp.arange(m + 1)
        for j in range(1, k_ord + 1):
            need = j - delta
            idx = jnp.clip(need - 1, 0, k_ord - 1)
            take = T_km1[rows, idx]
            col = jnp.where(need <= 0, 1.0, jnp.where(need <= k_ord, take, 0.0))
            cols.append(col)
        Q = jnp.stack(cols, axis=1)
        Wq = Q[1:, :] - Q[:-1, :]
        q_sorted.append(rs @ Wq)
    q_sorted = jnp.stack(q_sorted, axis=0)
    adv_sorted = q_sorted - v[None, :]
    return v.astype(dtype), q_sorted[inv, :].astype(dtype), adv_sorted[inv, :].astype(dtype)


@dataclass(frozen=True)
class OrderStatTransform:
    """JAX implementation.

    Returned inclusion/leave-one-out matrices/vectors are in **original index order**.
    """

    N: int
    k: int
    k_eff: float
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
        k: float,
        *,
        dtype=jnp.float64,
        compute_conditional: bool = True,
        compute_leave_one_out: bool = True,
        compute_dense_matrices: bool = False,
    ) -> "OrderStatTransform":
        k_eff = float(k)
        k_ord = int(k_eff // 1)
        if not (1 <= k_eff <= N):
            raise ValueError("Require real k with 1 <= k <= N")
        if not (1 <= k_ord <= N):
            raise ValueError("floor(k) must satisfy 1 <= floor(k) <= N")
        W = precompute_W_unconditional(N, k_eff, dtype=dtype)

        A = B = C = None
        if compute_conditional:
            A, B, C = precompute_ABC_conditional_including_rank(N, k_eff, dtype=dtype)

        Wm = None
        if compute_leave_one_out:
            if k_eff > N - 1:
                raise ValueError("Leave-one-out requires real k <= N-1")
            Wm = precompute_W_leave_one_out(N, k_eff, dtype=dtype)

        M_inc = M_loo = M_adv = None
        if compute_dense_matrices:
            M_inc = cls._build_dense_inclusion_matrix(A, B, C) if (A is not None and B is not None and C is not None) else None
            M_loo = cls._build_dense_leave_one_out_matrix(Wm, N, k_ord) if Wm is not None else None
            if M_inc is not None and M_loo is not None:
                M_adv = M_inc - M_loo

        return cls(N=N, k=k_ord, k_eff=k_eff, W=W, A=A, B=B, C=C, Wm=Wm, M_inc=M_inc, M_loo=M_loo, M_adv=M_adv)

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

    @staticmethod
    def _preset_lstat_weights(k: int, spec: str, *, dtype) -> jnp.ndarray:
        text = str(spec).strip()
        if not text:
            raise ValueError("l-stat preset cannot be empty")
        name, _, m_txt = text.partition(":")
        key = name.strip().lower()

        if key in {"remax", "remin"}:
            if m_txt.strip():
                raise ValueError(f"{name} does not take an m value")
            out = jnp.zeros((k,), dtype=dtype)
            idx = k - 1 if key == "remax" else 0
            return out.at[idx].set(1.0)

        if not m_txt.strip():
            raise ValueError(f"Preset '{name}' requires ':m' (e.g. {name}:3)")
        m = int(m_txt)
        if not (1 <= m <= k):
            raise ValueError(f"m must satisfy 1 <= m <= {k}")

        out = jnp.zeros((k,), dtype=dtype)
        if key == "topm":
            out = out.at[k - m :].set(1.0 / m)
        elif key == "botm":
            out = out.at[:m].set(1.0 / m)
        elif key in {"winsorizedm", "windosrizedm"}:
            if 2 * m >= k:
                raise ValueError(f"WinsorizedM requires 2*m < k (got m={m}, k={k})")
            out = out.at[m : k - m].set(1.0 / (k - 2 * m))
        else:
            raise ValueError(
                "Unknown l-stat preset. Supported: TopM:m, BotM:m, WinsorizedM:m, ReMax, ReMin"
            )
        return out

    def _coerce_a(self, a: Any) -> jnp.ndarray:
        if isinstance(a, str):
            return self._preset_lstat_weights(self.k, a, dtype=self.W.dtype)
        a = jnp.asarray(a, dtype=self.W.dtype)
        if a.shape != (self.k,):
            raise ValueError(f"a must be shape ({self.k},)")
        return a

    def lstat_weight_by_rank(self, a: Optional[Any] = None) -> jnp.ndarray:
        if a is None:
            if not hasattr(self, "Wa") or self.Wa is None:
                raise ValueError("No preweighted l-statistic vector is available. Pass a or use with_lstat_weights().")
            return self.Wa
        a = self._coerce_a(a)
        return self.W @ a

    def lstat_weight_by_item(self, x: jnp.ndarray, a: Optional[Any] = None) -> jnp.ndarray:
        _, inv = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return w_rank[inv]

    def expected_lstat(self, x: jnp.ndarray, a: Optional[Any] = None) -> jnp.ndarray:
        x_sorted, _ = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return jnp.sum(x_sorted * w_rank)

    def with_lstat_weights(self, a: Any) -> "OrderStatTransform":
        a = self._coerce_a(a)
        out = self.__class__(
            N=self.N,
            k=self.k,
            k_eff=self.k_eff,
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
    def precompute_lstat(cls, N: int, k: int, a: Any, **kwargs) -> "OrderStatTransform":
        return cls.precompute(N, k, **kwargs).with_lstat_weights(a)

    def expected_lstat_inclusion(self, x: jnp.ndarray, a: Optional[Any] = None, *, method: str = "efficient") -> jnp.ndarray:
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
        a = self._coerce_a(a)
        return E_inc @ a

    def expected_lstat_leave_one_out(self, x: jnp.ndarray, a: Optional[Any] = None, *, method: str = "efficient") -> jnp.ndarray:
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
        a = self._coerce_a(a)
        return E_loo @ a

    def expected_orderstats_advantage(self, x: jnp.ndarray, *, method: str = "efficient", detach_advantage: bool = True) -> jnp.ndarray:
        if method not in {"efficient", "matmul", "auto"}:
            raise ValueError("method must be one of {'efficient','matmul','auto'}")
        if method in {"matmul", "auto"} and self.M_adv is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            out = jnp.einsum("rmj,m->rj", self.M_adv, x_sorted)[inv, :]
            return jax.lax.stop_gradient(out) if detach_advantage else out
        out = self.expected_orderstats_inclusion(x, method=method) - self.expected_orderstats_leave_one_out(x, method=method)
        return jax.lax.stop_gradient(out) if detach_advantage else out

    def expected_orderstats_known_rp(self, r: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        v, _, _ = known_rp_orderstats(r, p, self.k_eff, dtype=self.W.dtype)
        return v

    def expected_orderstats_inclusion_known_rp(self, r: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        _, q, _ = known_rp_orderstats(r, p, self.k_eff, dtype=self.W.dtype)
        return q

    def expected_orderstats_advantage_known_rp(self, r: jnp.ndarray, p: jnp.ndarray, *, detach_advantage: bool = True) -> jnp.ndarray:
        _, _, adv = known_rp_orderstats(r, p, self.k_eff, dtype=self.W.dtype)
        return jax.lax.stop_gradient(adv) if detach_advantage else adv

    def expected_lstat_known_rp(self, r: jnp.ndarray, p: jnp.ndarray, a: Any) -> jnp.ndarray:
        a = self._coerce_a(a)
        return self.expected_orderstats_known_rp(r, p) @ a

    def expected_lstat_inclusion_known_rp(self, r: jnp.ndarray, p: jnp.ndarray, a: Any) -> jnp.ndarray:
        a = self._coerce_a(a)
        return self.expected_orderstats_inclusion_known_rp(r, p) @ a

    def expected_lstat_advantage_known_rp(self, r: jnp.ndarray, p: jnp.ndarray, a: Any, *, detach_advantage: bool = True) -> jnp.ndarray:
        a = self._coerce_a(a)
        out = self.expected_orderstats_advantage_known_rp(r, p, detach_advantage=detach_advantage) @ a
        return jax.lax.stop_gradient(out) if detach_advantage else out

    def expected_lstat_advantage(self, x: jnp.ndarray, a: Optional[Any] = None, *, method: str = "efficient", detach_advantage: bool = True) -> jnp.ndarray:
        if method in {"matmul", "auto"} and self.M_adv_a is not None and a is None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            out = (self.M_adv_a @ x_sorted)[inv]
            return jax.lax.stop_gradient(out) if detach_advantage else out
        out = self.expected_lstat_inclusion(x, a, method=method) - self.expected_lstat_leave_one_out(x, a, method=method)
        return jax.lax.stop_gradient(out) if detach_advantage else out
