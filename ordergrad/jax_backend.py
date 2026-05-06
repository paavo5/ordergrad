from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, Tuple

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import betainc, gammaln
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

_K_INTEGER_TOL = 1e-12


def _is_integer_k(k_eff: float) -> bool:
    return math.isfinite(float(k_eff)) and abs(float(k_eff) - round(float(k_eff))) <= _K_INTEGER_TOL


def _fractional_sampling_k_message(context: str, k_eff: float) -> str:
    lo = math.floor(k_eff)
    hi = math.ceil(k_eff)
    frac = k_eff - lo
    return (
        f"Fractional k={k_eff:g} is not supported for sampling-based order statistics in {context}. "
        "Sampling/subset order statistics require an integer sample size. "
        f"A reasonable smooth proxy is to compute the two nearest integer transforms "
        f"k={lo} and k={hi} and interpolate their outputs with weight frac={frac:g}. "
        "Alternatively, choose an integer L-statistic that matches your intent, such as "
        "TopM:m at integer K for optimistic/ReMax-style transforms or BotM:m at integer K "
        "for pessimistic/ReMin-style transforms. For known (r,p), use the beta "
        "continuation instead: known_rp_orderstats(..., branch='top' or 'bottom') "
        "or known_rp_lstat(..., a='ReMax'/'ReMin'/'TopM:m'/'BotM:m')."
    )


def _require_integer_sampling_k(k: float, *, max_k: int, context: str) -> tuple[int, float]:
    k_eff = float(k)
    if not math.isfinite(k_eff):
        raise ValueError("k must be finite")
    if not (1 <= k_eff <= max_k):
        raise ValueError(f"Require integer k with 1 <= k <= {max_k}")
    if not _is_integer_k(k_eff):
        raise ValueError(_fractional_sampling_k_message(context, k_eff))
    k_int = int(round(k_eff))
    return k_int, float(k_int)


def _normalize_known_rp_branch(branch: Optional[str]) -> Optional[str]:
    if branch is None:
        return None
    key = str(branch).strip().lower()
    if key in {"bottom", "lower", "remin", "min"}:
        return "bottom"
    if key in {"top", "upper", "remax", "max"}:
        return "top"
    raise ValueError("branch must be one of {'bottom', 'top', 'ReMin', 'ReMax'}")


def _infer_known_rp_branch_from_lstat(a: Any) -> Optional[str]:
    if not isinstance(a, str):
        return None
    name, _, _ = str(a).strip().partition(":")
    key = name.strip().lower()
    if key in {"remax", "topm", "uppertailmean", "rangeuppertailmean", "rank"} or key.startswith("topquantile"):
        return "top"
    if key in {"remin", "botm", "lowertailmean", "rangelowertailmean", "rangecvar", "trimmedcvar"} or key.startswith("quantile"):
        return "bottom"
    return None


def _resolve_known_rp_lstat_branch(a: Any, branch: Optional[str], k_eff: float) -> Optional[str]:
    branch = _normalize_known_rp_branch(branch)
    if branch is not None:
        return branch
    inferred = _infer_known_rp_branch_from_lstat(a)
    if inferred is not None:
        return inferred
    if _is_integer_k(k_eff):
        return None
    raise ValueError(
        f"Fractional known-(r,p) L-statistics are branch-ambiguous for k={k_eff:g}. "
        "Pass branch='top' for upper/ReMax-style ranks or branch='bottom' for "
        "lower/ReMin-style ranks. Logical presets infer this automatically for "
        "ReMax, ReMin, TopM:m, BotM:m, UpperTailMean:q, LowerTailMean:q, "
        "TopQuantile:q, Quantile:q, and Rank:r."
    )


def _beta_cdf_table(
    F: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    dtype=jnp.float64,
) -> jnp.ndarray:
    """Regularized-beta CDF table with conditional-boundary conventions."""
    F = jnp.clip(jnp.asarray(F, dtype=jnp.float64), 0.0, 1.0)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    beta = jnp.asarray(beta, dtype=jnp.float64)

    valid = (alpha > 0.0) & (beta > 0.0)
    alpha_safe = jnp.where(valid, alpha, 1.0)
    beta_safe = jnp.where(valid, beta, 1.0)

    T = betainc(alpha_safe[None, :], beta_safe[None, :], F[:, None])
    T = jnp.where(
        alpha[None, :] <= 0.0,
        1.0,
        jnp.where(beta[None, :] <= 0.0, 0.0, T),
    )
    return T.astype(dtype)


def _known_rp_beta_order_params(
    k_eff: float,
    k_ord: int,
    branch: Optional[str],
    *,
    dtype=jnp.float64,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    branch = _normalize_known_rp_branch(branch)
    if branch is None:
        if not _is_integer_k(k_eff):
            raise ValueError(
                f"Fractional known-(r,p) order statistics are branch-ambiguous for k={k_eff:g}. "
                "Use branch='top' for ReMax/top-aligned ranks or branch='bottom' "
                "for ReMin/bottom-aligned ranks. For L-statistics, known_rp_lstat "
                "and expected_lstat_known_rp infer the branch from logical presets "
                "such as ReMax, ReMin, TopM:m, and BotM:m."
            )
        branch = "bottom"

    if branch == "bottom":
        alpha = jnp.arange(1, k_ord + 1, dtype=dtype)
        beta = k_eff + 1.0 - alpha
        return alpha, beta

    # Top-aligned branches are returned in ascending value order.
    # Example k=3.6: [third-best, second-best, best]
    ell = jnp.arange(k_ord, 0, -1, dtype=dtype)
    alpha = k_eff + 1.0 - ell
    beta = ell
    return alpha, beta



def precompute_W_unconditional(N: int, k: float, *, dtype=jnp.float64) -> jnp.ndarray:
    """W[m-1,j-1] = P(X_(j:k) == x_(m)) for an integer uniform k-subset from N."""
    k_int, k_eff = _require_integer_sampling_k(k, max_k=N, context="precompute_W_unconditional")
    log_den = _log_choose(float(N), k_eff)

    def log_term(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m, k_eff - j)

    return _build_weight_matrix(N, k_int, log_den, log_term, dtype=dtype, renormalize_cols=True)


def precompute_ABC_conditional_including_rank(N: int, k: float, *, dtype=jnp.float64) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute conditional matrices A,B,C for an integer sample size k."""
    k_int, k_eff = _require_integer_sampling_k(k, max_k=N, context="precompute_ABC_conditional_including_rank")
    log_den = _log_choose(float(N - 1), k_eff - 1.0)

    def logA(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m - 1, k_eff - j - 1.0)

    def logB(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m, k_eff - j)

    def logC(m, j):
        return _log_choose(m - 2, j - 2) + _log_choose(N - m, k_eff - j)

    A = _build_weight_matrix(N, k_int, log_den, logA, dtype=dtype, renormalize_cols=False)
    B = _build_weight_matrix(N, k_int, log_den, logB, dtype=dtype, renormalize_cols=False)
    C = _build_weight_matrix(N, k_int, log_den, logC, dtype=dtype, renormalize_cols=False)
    return A, B, C


def precompute_W_leave_one_out(N: int, k: float, *, dtype=jnp.float64) -> jnp.ndarray:
    """Wm for reduced population size (N-1) and an integer sample size k."""
    k_int, k_eff = _require_integer_sampling_k(k, max_k=N - 1, context="precompute_W_leave_one_out")
    log_den = _log_choose(float(N - 1), k_eff)

    def log_term(p, j):
        return _log_choose(p - 1, j - 1) + _log_choose((N - 1) - p, k_eff - j)

    return _build_weight_matrix(N - 1, k_int, log_den, log_term, dtype=dtype, renormalize_cols=True)


def known_rp_orderstats(
    r: jnp.ndarray,
    p: jnp.ndarray,
    k: float,
    *,
    branch: Optional[str] = None,
    dtype=jnp.float64,
):
    """Exact known-(r,p) with-replacement beta-continuation order statistics.

    For integer k, the default branch returns the ordinary ascending order-statistic
    vector [min@k, ..., max@k]. For fractional k, pass branch='bottom' for
    bottom-aligned/ReMin ranks or branch='top' for top-aligned/ReMax ranks.
    """
    r = jnp.asarray(r, dtype=jnp.float64)
    p = jnp.asarray(p, dtype=jnp.float64)
    if r.ndim != 1 or p.ndim != 1 or r.shape[0] != p.shape[0]:
        raise ValueError("r and p must be 1D arrays of equal length")
    if jnp.any(p < 0):
        raise ValueError("p must be nonnegative")
    p = p / jnp.sum(p)
    m = int(r.shape[0])
    k_eff = float(k)
    k_ord = int(math.floor(k_eff))
    if not (k_eff >= 1):
        raise ValueError("Require real k >= 1")
    if k_ord < 1:
        raise ValueError("floor(k) must be >= 1")

    alpha, beta = _known_rp_beta_order_params(k_eff, k_ord, branch, dtype=jnp.float64)

    perm = jnp.argsort(r, stable=True)
    inv = jnp.empty_like(perm)
    inv = inv.at[perm].set(jnp.arange(m, dtype=perm.dtype))
    rs = r[perm]
    ps = p[perm]

    F = jnp.concatenate([jnp.array([0.0], dtype=jnp.float64), jnp.cumsum(ps)])
    T = _beta_cdf_table(F, alpha, beta, dtype=dtype)
    W = T[1:, :] - T[:-1, :]
    v = rs @ W

    # Conditional CDF table with one draw fixed. If the fixed draw is <= the
    # threshold, it contributes one lower-side count: alpha -> alpha - 1.
    # Otherwise it contributes one upper-side count: beta -> beta - 1.
    T_le = _beta_cdf_table(F, alpha - 1.0, beta, dtype=dtype)
    T_gt = _beta_cdf_table(F, alpha, beta - 1.0, dtype=dtype)
    q_sorted = []
    for b in range(m):
        forced_le = jnp.concatenate([jnp.array([False]), rs[b] <= rs])
        G = jnp.where(forced_le[:, None], T_le, T_gt)
        Wq = G[1:, :] - G[:-1, :]
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
    sampling_valid: bool = True

    def _require_sampling_valid(self, context: str) -> None:
        if not self.sampling_valid:
            raise ValueError(
                f"{context} is a sampling-based method, but this transform was created "
                "for fractional/known-(r,p)-only use. Sampling-based order statistics "
                "require integer k. Use OrderStatTransform.precompute(N, floor(k)) and "
                "OrderStatTransform.precompute(N, ceil(k)) and interpolate their outputs, "
                "or choose an integer L-statistic such as TopM:m or BotM:m. For known "
                "(r,p), call the *_known_rp methods or known_rp_orderstats/known_rp_lstat."
            )

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
        k_ord, k_eff = _require_integer_sampling_k(k, max_k=N, context="OrderStatTransform.precompute")
        W = precompute_W_unconditional(N, k_eff, dtype=dtype)

        A = B = C = None
        if compute_conditional:
            A, B, C = precompute_ABC_conditional_including_rank(N, k_eff, dtype=dtype)

        Wm = None
        if compute_leave_one_out:
            if k_eff > N - 1:
                raise ValueError("Leave-one-out requires integer k <= N-1")
            Wm = precompute_W_leave_one_out(N, k_eff, dtype=dtype)

        M_inc = M_loo = M_adv = None
        if compute_dense_matrices:
            M_inc = cls._build_dense_inclusion_matrix(A, B, C) if (A is not None and B is not None and C is not None) else None
            M_loo = cls._build_dense_leave_one_out_matrix(Wm, N, k_ord) if Wm is not None else None
            if M_inc is not None and M_loo is not None:
                M_adv = M_inc - M_loo

        return cls(N=N, k=k_ord, k_eff=k_eff, W=W, A=A, B=B, C=C, Wm=Wm, M_inc=M_inc, M_loo=M_loo, M_adv=M_adv)

    @classmethod
    def precompute_known_rp(cls, k: float, *, dtype=jnp.float64) -> "OrderStatTransform":
        """Create a lightweight transform for known-(r,p) methods with real k.

        Sampling-based methods on the returned object intentionally raise, because
        fractional sample sizes do not define a literal sampled subset.
        """
        k_eff = float(k)
        if not (k_eff >= 1.0):
            raise ValueError("Require real k >= 1")
        k_ord = int(math.floor(k_eff))
        if k_ord < 1:
            raise ValueError("floor(k) must be >= 1")
        W = jnp.zeros((0, k_ord), dtype=dtype)
        return cls(N=0, k=k_ord, k_eff=k_eff, W=W, A=None, B=None, C=None, Wm=None, sampling_valid=False)

    @classmethod
    def for_known_rp(cls, k: float, *, dtype=jnp.float64) -> "OrderStatTransform":
        """Construct a known-(r,p)-only transform for real k.

        Sampling/batch methods are intentionally disabled on the returned object;
        use the *_known_rp methods with an explicit branch for ambiguous
        fractional-k order statistics.
        """
        return cls.precompute_known_rp(k, dtype=dtype)

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
        self._require_sampling_valid("expected_orderstats")
        x_sorted, _ = self._sort_with_inverse_rank(x)
        return x_sorted @ self.W

    def expected_orderstats_inclusion(self, x: jnp.ndarray, *, method: str = "efficient") -> jnp.ndarray:
        self._require_sampling_valid("expected_orderstats_inclusion")
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
        self._require_sampling_valid("expected_orderstats_leave_one_out")
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

        if key in {"remax", "remin", "median", "ginimeandifference", "gmd"}:
            if m_txt.strip():
                raise ValueError(f"{name} does not take an m value")
            out = jnp.zeros((k,), dtype=dtype)
            if key == "remax":
                return out.at[k - 1].set(1.0)
            if key == "remin":
                return out.at[0].set(1.0)
            if key == "median":
                if k % 2 == 1:
                    return out.at[k // 2].set(1.0)
                return out.at[(k // 2) - 1].set(0.5).at[k // 2].set(0.5)
            j = jnp.arange(1, k + 1, dtype=dtype)
            return (2.0 * (2.0 * j - (k + 1.0))) / (k * (k - 1.0))

        if key in {"harrelldavis", "harreldavis"}:
            if not m_txt.strip():
                raise ValueError("Preset 'HarrellDavis' requires ':q' (e.g. HarrellDavis:0.75)")
            q = float(m_txt)
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"HarrellDavis:q requires 0 <= q <= 1 (got q={q})")
            a = (k + 1) * q
            b = (k + 1) * (1.0 - q)
            u_hi = jnp.arange(1, k + 1, dtype=dtype) / float(k)
            u_lo = jnp.arange(0, k, dtype=dtype) / float(k)
            return betainc(a, b, u_hi) - betainc(a, b, u_lo)

        if key in {
            "quantile", "topquantile",
            "quantileweibull", "quantilehazen", "quantileblom",
            "topquantileweibull", "topquantilehazen", "topquantileblom",
        }:
            if not m_txt.strip():
                raise ValueError(f"Preset '{name}' requires ':q' (e.g. {name}:0.25)")
            q = float(m_txt)
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"{name}:q requires 0 <= q <= 1 (got q={q})")

            is_top = key.startswith("topquantile")
            variant = key[len("topquantile"):] if is_top else key[len("quantile"):]
            if variant == "":
                a_pp = 0.5
            elif variant == "weibull":
                a_pp = 0.0
            elif variant == "hazen":
                a_pp = 0.5
            elif variant == "blom":
                a_pp = 3.0 / 8.0
            else:
                raise ValueError(f"Unknown quantile preset variant in '{name}'")

            q_eff = (1.0 - q) if is_top else q
            out = jnp.zeros((k,), dtype=dtype)
            denom = (k + 1.0 - 2.0 * a_pp)
            pos = q_eff * denom + a_pp
            if pos <= 1.0:
                return out.at[0].set(1.0)
            if pos >= float(k):
                return out.at[k - 1].set(1.0)
            i_left = int(math.floor(pos))
            frac = pos - float(i_left)
            return out.at[i_left - 1].set(1.0 - frac).at[i_left].set(frac)

        if key == "rank":
            if not m_txt.strip():
                raise ValueError("Preset 'Rank' requires ':r' (e.g. Rank:3)")
            r = int(m_txt)
            if not (1 <= r <= k):
                raise ValueError(f"Rank:r requires integer r with 1 <= r <= {k} (got r={r})")
            out = jnp.zeros((k,), dtype=dtype)
            return out.at[k - r].set(1.0)

        if key in {"uppertailmean", "lowertailmean"}:
            if not m_txt.strip():
                raise ValueError(f"Preset '{name}' requires ':q' (e.g. {name}:0.25)")
            q = float(m_txt)
            if not (0.0 < q <= 1.0):
                raise ValueError(f"{name}:q requires 0 < q <= 1 (got q={q})")
            m = max(1, int(math.ceil(q * k)))
            out = jnp.zeros((k,), dtype=dtype)
            if key == "uppertailmean":
                return out.at[k - m :].set(1.0 / m)
            return out.at[:m].set(1.0 / m)


        if key in {"rangelowertailmean", "rangeuppertailmean", "rangemean", "trimmedmeanfrac", "rangecvar", "trimmedcvar"}:
            parts = [p.strip() for p in m_txt.split(":") if p.strip()]
            if len(parts) != 2:
                raise ValueError(f"Preset '{name}' requires ':lo:hi' (e.g. {name}:0.02:0.20)")
            lo, hi = float(parts[0]), float(parts[1])
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(f"{name}:lo:hi requires 0 <= lo < hi <= 1 (got lo={lo}, hi={hi})")
            def _lower_bounds(lo_frac, hi_frac):
                start = int(math.floor(lo_frac * k)); stop = int(math.ceil(hi_frac * k))
                start = max(0, min(k - 1, start)); stop = max(start + 1, min(k, stop))
                return start, stop
            out = jnp.zeros((k,), dtype=dtype)
            if key in {"rangelowertailmean", "rangecvar", "trimmedcvar", "rangemean", "trimmedmeanfrac"}:
                start, stop = _lower_bounds(lo, hi)
            else:
                start = k - int(math.ceil(hi * k)); stop = k - int(math.floor(lo * k))
                start = max(0, min(k - 1, start)); stop = max(start + 1, min(k, stop))
            return out.at[start:stop].set(1.0 / float(stop - start))

        if key == "lmoment":
            if not m_txt.strip():
                raise ValueError("Preset 'LMoment' requires ':r' (e.g. LMoment:2)")
            r = int(m_txt)
            if not (1 <= r <= k):
                raise ValueError(f"LMoment:r requires integer r with 1 <= r <= k (got r={r}, k={k})")
            out = jnp.zeros((k,), dtype=dtype)
            for m in range(r):
                sign = -1.0 if (m % 2) else 1.0
                coeff = sign * math.comb(r - 1, m) / math.comb(r - 1 + m, m)
                t = r - m
                den = math.comb(k - 1, t - 1)
                b_w = jnp.zeros((k,), dtype=dtype)
                for j in range(t, k + 1):
                    b_w = b_w.at[j - 1].set(math.comb(j - 1, t - 1) / den)
                out = out + coeff * (b_w / float(k))
            return out / float(r)

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
        elif key in {"midrangem", "topbot"}:
            out = out.at[:m].set(0.5 / m)
            out = out.at[k - m :].add(0.5 / m)
        elif key in {"winsorizedm", "windosrizedm"}:
            if 2 * m >= k:
                raise ValueError(f"WinsorizedM requires 2*m < k (got m={m}, k={k})")
            out = out.at[m : k - m].set(1.0 / k)
            out = out.at[m].add(m / k)
            out = out.at[k - m - 1].add(m / k)
        elif key in {"trimm", "trimmedm", "trimmeanm"}:
            if 2 * m >= k:
                raise ValueError(f"TrimM requires 2*m < k (got m={m}, k={k})")
            out = out.at[m : k - m].set(1.0 / (k - 2 * m))
        else:
            raise ValueError(
                "Unknown l-stat preset. Supported: TopM:m, BotM:m, TrimM:m, WinsorizedM:m, MidrangeM:m, TopBot:m, ReMax, ReMin, Median, Rank:r, Quantile:q (Hazen default), QuantileWeibull:q, QuantileHazen:q, QuantileBlom:q, TopQuantile:q (Hazen default), TopQuantileWeibull:q, TopQuantileHazen:q, TopQuantileBlom:q, UpperTailMean:q, LowerTailMean:q, RangeLowerTailMean:lo:hi, RangeUpperTailMean:lo:hi, RangeMean:lo:hi, TrimmedMeanFrac:lo:hi, RangeCVaR:lo:hi, TrimmedCVaR:lo:hi, HarrellDavis:q, GiniMeanDifference, LMoment:r"
            )
        return out

    def _coerce_a(self, a: Any) -> jnp.ndarray:
        if isinstance(a, str):
            return self._preset_lstat_weights(self.k, a, dtype=self.W.dtype)
        a = jnp.asarray(a, dtype=self.W.dtype)
        if a.shape != (self.k,):
            raise ValueError(f"a must be shape ({self.k},)")
        # Numeric vectors are provided in top-rank order (j=1 highest).
        return a[::-1]

    def lstat_weight_by_rank(self, a: Optional[Any] = None) -> jnp.ndarray:
        self._require_sampling_valid("lstat_weight_by_rank")
        if a is None:
            if not hasattr(self, "Wa") or self.Wa is None:
                raise ValueError("No preweighted l-statistic vector is available. Pass a or use with_lstat_weights().")
            return self.Wa
        a = self._coerce_a(a)
        return self.W @ a

    def lstat_weight_by_item(self, x: jnp.ndarray, a: Optional[Any] = None) -> jnp.ndarray:
        self._require_sampling_valid("lstat_weight_by_item")
        _, inv = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return w_rank[inv]

    def expected_lstat(self, x: jnp.ndarray, a: Optional[Any] = None) -> jnp.ndarray:
        self._require_sampling_valid("expected_lstat")
        x_sorted, _ = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return jnp.sum(x_sorted * w_rank)

    def with_lstat_weights(self, a: Any) -> "OrderStatTransform":
        self._require_sampling_valid("with_lstat_weights")
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
            sampling_valid=self.sampling_valid,
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
        self._require_sampling_valid("expected_lstat_inclusion")
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
        self._require_sampling_valid("expected_lstat_leave_one_out")
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
        self._require_sampling_valid("expected_orderstats_advantage")
        if method not in {"efficient", "matmul", "auto"}:
            raise ValueError("method must be one of {'efficient','matmul','auto'}")
        if method in {"matmul", "auto"} and self.M_adv is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            out = jnp.einsum("rmj,m->rj", self.M_adv, x_sorted)[inv, :]
            return jax.lax.stop_gradient(out) if detach_advantage else out
        out = self.expected_orderstats_inclusion(x, method=method) - self.expected_orderstats_leave_one_out(x, method=method)
        return jax.lax.stop_gradient(out) if detach_advantage else out

    def expected_orderstats_known_rp(self, r: jnp.ndarray, p: jnp.ndarray, *, branch: Optional[str] = None) -> jnp.ndarray:
        v, _, _ = known_rp_orderstats(r, p, self.k_eff, branch=branch, dtype=self.W.dtype)
        return v

    def expected_orderstats_inclusion_known_rp(self, r: jnp.ndarray, p: jnp.ndarray, *, branch: Optional[str] = None) -> jnp.ndarray:
        _, q, _ = known_rp_orderstats(r, p, self.k_eff, branch=branch, dtype=self.W.dtype)
        return q

    def expected_orderstats_advantage_known_rp(self, r: jnp.ndarray, p: jnp.ndarray, *, branch: Optional[str] = None, detach_advantage: bool = True) -> jnp.ndarray:
        _, _, adv = known_rp_orderstats(r, p, self.k_eff, branch=branch, dtype=self.W.dtype)
        return jax.lax.stop_gradient(adv) if detach_advantage else adv

    def expected_lstat_known_rp(self, r: jnp.ndarray, p: jnp.ndarray, a: Any, *, branch: Optional[str] = None) -> jnp.ndarray:
        branch = _resolve_known_rp_lstat_branch(a, branch, self.k_eff)
        a = self._coerce_a(a)
        return self.expected_orderstats_known_rp(r, p, branch=branch) @ a

    def expected_lstat_inclusion_known_rp(self, r: jnp.ndarray, p: jnp.ndarray, a: Any, *, branch: Optional[str] = None) -> jnp.ndarray:
        branch = _resolve_known_rp_lstat_branch(a, branch, self.k_eff)
        a = self._coerce_a(a)
        return self.expected_orderstats_inclusion_known_rp(r, p, branch=branch) @ a

    def expected_lstat_advantage_known_rp(self, r: jnp.ndarray, p: jnp.ndarray, a: Any, *, branch: Optional[str] = None, detach_advantage: bool = True) -> jnp.ndarray:
        branch = _resolve_known_rp_lstat_branch(a, branch, self.k_eff)
        a = self._coerce_a(a)
        out = self.expected_orderstats_advantage_known_rp(r, p, branch=branch, detach_advantage=detach_advantage) @ a
        return jax.lax.stop_gradient(out) if detach_advantage else out

    def expected_lstat_advantage(self, x: jnp.ndarray, a: Optional[Any] = None, *, method: str = "efficient", detach_advantage: bool = True) -> jnp.ndarray:
        self._require_sampling_valid("expected_lstat_advantage")
        if method in {"matmul", "auto"} and self.M_adv_a is not None and a is None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            out = (self.M_adv_a @ x_sorted)[inv]
            return jax.lax.stop_gradient(out) if detach_advantage else out
        out = self.expected_lstat_inclusion(x, a, method=method) - self.expected_lstat_leave_one_out(x, a, method=method)
        return jax.lax.stop_gradient(out) if detach_advantage else out


def known_rp_lstat(
    r: jnp.ndarray,
    p: jnp.ndarray,
    k: float,
    a: Any,
    *,
    branch: Optional[str] = None,
    dtype=jnp.float64,
):
    """Known-(r,p) L-statistic under the beta order-statistic continuation.

    Returns (v, q, adv), where v is scalar, q has shape (num_actions,), and
    adv=q-v. For fractional k, branch is inferred for logical presets such as
    ReMax/ReMin/TopM/BotM; otherwise pass branch='top' or branch='bottom'.
    """
    k_eff = float(k)
    if not (k_eff >= 1.0):
        raise ValueError("Require real k >= 1")
    k_ord = int(math.floor(k_eff))
    if k_ord < 1:
        raise ValueError("floor(k) must be >= 1")

    branch = _resolve_known_rp_lstat_branch(a, branch, k_eff)
    if isinstance(a, str):
        weights = OrderStatTransform._preset_lstat_weights(k_ord, a, dtype=dtype)
    else:
        weights = jnp.asarray(a, dtype=dtype)
        if weights.shape != (k_ord,):
            raise ValueError(f"a must be shape ({k_ord},)")
        weights = weights[::-1]

    v, q, adv = known_rp_orderstats(r, p, k_eff, branch=branch, dtype=dtype)
    return v @ weights, q @ weights, adv @ weights
