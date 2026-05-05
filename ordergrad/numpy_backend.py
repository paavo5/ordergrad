import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Tuple


# -----------------------------
# Combinatorics helpers
# -----------------------------

import math


def _betainc_regularized(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    def _betacf(aa: float, bb: float, xx: float) -> float:
        qab = aa + bb
        qap = aa + 1.0
        qam = aa - 1.0
        c = 1.0
        d = 1.0 - qab * xx / qap
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        h = d
        for m in range(1, 201):
            m2 = 2 * m
            num = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + num * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + num / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            h *= d * c

            num = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1.0 + num * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + num / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 3e-14:
                break
        return h

    ln_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _harrell_davis_weights(k: int, q: float, *, dtype=np.float64) -> np.ndarray:
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"HarrellDavis:q requires 0 <= q <= 1 (got q={q})")
    a = (k + 1) * q
    b = (k + 1) * (1.0 - q)
    u_hi = np.arange(1, k + 1, dtype=np.float64) / float(k)
    u_lo = np.arange(0, k, dtype=np.float64) / float(k)
    w = np.array([_betainc_regularized(a, b, hi) - _betainc_regularized(a, b, lo) for lo, hi in zip(u_lo, u_hi)], dtype=np.float64)
    return np.asarray(w, dtype=dtype)


def _l_moment_weights(k: int, r: int, *, dtype=np.float64) -> np.ndarray:
    if not (1 <= r <= k):
        raise ValueError(f"LMoment:r requires integer r with 1 <= r <= k (got r={r}, k={k})")

    def _b_weights(t: int) -> np.ndarray:
        w = np.zeros((k,), dtype=np.float64)
        den = math.comb(k - 1, t - 1)
        for j in range(t, k + 1):
            w[j - 1] = math.comb(j - 1, t - 1) / den
        return w / float(k)

    out = np.zeros((k,), dtype=np.float64)
    for m in range(r):
        sign = -1.0 if (m % 2) else 1.0
        coeff = sign * math.comb(r - 1, m) / math.comb(r - 1 + m, m)
        out += coeff * _b_weights(r - m)
    return np.asarray(out / float(r), dtype=dtype)


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
    if key in {"remax", "topm", "uppertailmean", "rank"} or key.startswith("topquantile"):
        return "top"
    if key in {"remin", "botm", "lowertailmean"} or key.startswith("quantile"):
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


def _beta_cdf_table(F: np.ndarray, alpha: np.ndarray, beta: np.ndarray, *, dtype=np.float64) -> np.ndarray:
    """Regularized-beta CDF table with conditional-boundary conventions."""
    F = np.clip(np.asarray(F, dtype=np.float64), 0.0, 1.0)
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    out = np.empty((F.shape[0], alpha.shape[0]), dtype=np.float64)
    for c, (aa, bb) in enumerate(zip(alpha, beta)):
        if aa <= 0.0:
            out[:, c] = 1.0
        elif bb <= 0.0:
            out[:, c] = 0.0
        else:
            out[:, c] = [_betainc_regularized(float(aa), float(bb), float(x)) for x in F]
    return out.astype(dtype, copy=False)


def _known_rp_beta_order_params(k_eff: float, k_ord: int, branch: Optional[str], *, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
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
        alpha = np.arange(1, k_ord + 1, dtype=dtype)
        beta = k_eff + 1.0 - alpha
        return alpha, beta
    ell = np.arange(k_ord, 0, -1, dtype=dtype)
    alpha = k_eff + 1.0 - ell
    beta = ell
    return alpha, beta



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
    """W[m-1,j-1] = P(X_(j:k) == x_(m)) for an integer uniform k-subset from N.

    Shape: (N,k). Columns sum to 1.
    """
    k_int, k_eff = _require_integer_sampling_k(k, max_k=N, context="precompute_W_unconditional")

    log_den = _log_choose(float(N), k_eff)

    def log_term(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m, k_eff - j)

    return _build_weight_matrix(
        k_int,
        log_den,
        log_term,
        out_shape=(N, k_int),
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
    """Precompute conditional matrices A,B,C for an integer sample size k."""
    k_int, k_eff = _require_integer_sampling_k(k, max_k=N, context="precompute_ABC_conditional_including_rank")

    log_den = _log_choose(float(N - 1), k_eff - 1.0)

    def logA(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m - 1, k_eff - j - 1.0)

    def logB(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m, k_eff - j)

    def logC(m, j):
        return _log_choose(m - 2, j - 2) + _log_choose(N - m, k_eff - j)

    A = _build_weight_matrix(k_int, log_den, logA, out_shape=(N, k_int), dtype=dtype, chunk_size=chunk_size, renormalize_cols=False)
    B = _build_weight_matrix(k_int, log_den, logB, out_shape=(N, k_int), dtype=dtype, chunk_size=chunk_size, renormalize_cols=False)
    C = _build_weight_matrix(k_int, log_den, logC, out_shape=(N, k_int), dtype=dtype, chunk_size=chunk_size, renormalize_cols=False)
    return A, B, C


def precompute_W_leave_one_out(
    N: int,
    k: float,
    *,
    dtype=np.float64,
    chunk_size: Optional[int] = None
) -> np.ndarray:
    """Wm[p-1,j-1] for population size (N-1) and integer sample size k."""
    k_int, k_eff = _require_integer_sampling_k(k, max_k=N - 1, context="precompute_W_leave_one_out")

    log_den = _log_choose(float(N - 1), k_eff)

    def log_term(p, j):
        return _log_choose(p - 1, j - 1) + _log_choose((N - 1) - p, k_eff - j)

    return _build_weight_matrix(
        k_int,
        log_den,
        log_term,
        out_shape=(N - 1, k_int),
        dtype=dtype,
        chunk_size=chunk_size,
        renormalize_cols=True,
    )


def known_rp_orderstats(
    r: np.ndarray,
    p: np.ndarray,
    k: float,
    *,
    return_sorted: bool = False,
    branch: Optional[str] = None,
    dtype=np.float64,
):
    """Known-(r,p) with-replacement beta-continuation order statistics.

    For integer k, the default branch returns the ordinary ascending order-statistic
    vector [min@k, ..., max@k]. For fractional k, pass branch='bottom' for
    bottom-aligned/ReMin ranks or branch='top' for top-aligned/ReMax ranks.
    """
    r = np.asarray(r, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if r.ndim != 1 or p.ndim != 1 or r.shape[0] != p.shape[0]:
        raise ValueError("r and p must be 1D arrays of equal length")
    if np.any(p < 0):
        raise ValueError("p must be nonnegative")
    psum = p.sum()
    if not np.isfinite(psum) or psum <= 0:
        raise ValueError("sum(p) must be positive and finite")
    p = p / psum

    m = r.shape[0]
    k_eff = float(k)
    k_ord = int(math.floor(k_eff))
    if not (1 <= k_eff):
        raise ValueError("Require real k >= 1")
    if k_ord < 1:
        raise ValueError("floor(k) must be >= 1")

    alpha, beta = _known_rp_beta_order_params(k_eff, k_ord, branch, dtype=np.float64)

    perm = np.argsort(r, kind="mergesort")
    inv = np.empty_like(perm)
    inv[perm] = np.arange(m, dtype=perm.dtype)
    rs = r[perm]
    ps = p[perm]

    F = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(ps)])
    T = _beta_cdf_table(F, alpha, beta, dtype=dtype)
    W = T[1:, :] - T[:-1, :]
    v = rs @ W

    T_le = _beta_cdf_table(F, alpha - 1.0, beta, dtype=dtype)
    T_gt = _beta_cdf_table(F, alpha, beta - 1.0, dtype=dtype)

    q_sorted = np.empty((m, k_ord), dtype=dtype)
    for b in range(m):
        forced_le = np.concatenate([np.array([False]), rs[b] <= rs])
        G = np.where(forced_le[:, None], T_le, T_gt)
        Wq = G[1:, :] - G[:-1, :]
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

    @staticmethod
    def _preset_lstat_weights(k: int, spec: str, *, dtype) -> np.ndarray:
        text = str(spec).strip()
        if not text:
            raise ValueError("l-stat preset cannot be empty")
        name, _, m_txt = text.partition(":")
        key = name.strip().lower()

        if key in {"remax", "remin", "median", "ginimeandifference", "gmd"}:
            if m_txt.strip():
                raise ValueError(f"{name} does not take an m value")
            out = np.zeros((k,), dtype=dtype)
            if key == "remax":
                out[k - 1] = 1.0
            elif key == "remin":
                out[0] = 1.0
            elif key == "median":
                if k % 2 == 1:
                    out[k // 2] = 1.0
                else:
                    out[(k // 2) - 1] = 0.5
                    out[k // 2] = 0.5
            else:  # Gini mean difference
                j = np.arange(1, k + 1, dtype=dtype)
                out = (2.0 * (2.0 * j - (k + 1.0))) / (k * (k - 1.0))
            return out

        if key in {"harrelldavis", "harreldavis"}:
            if not m_txt.strip():
                raise ValueError("Preset 'HarrellDavis' requires ':q' (e.g. HarrellDavis:0.75)")
            q = float(m_txt)
            return _harrell_davis_weights(k, q, dtype=dtype)

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
            # Default Quantile/TopQuantile to Hazen.
            # p_r(a) = (r-a)/(k+1-2a), with common choices:
            # Weibull(a=0), Hazen(a=0.5), Blom(a=3/8).
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
            out = np.zeros((k,), dtype=dtype)
            denom = (k + 1.0 - 2.0 * a_pp)
            pos = q_eff * denom + a_pp
            if pos <= 1.0:
                out[0] = 1.0
            elif pos >= float(k):
                out[k - 1] = 1.0
            else:
                i_left = int(math.floor(pos))
                frac = pos - float(i_left)
                out[i_left - 1] = 1.0 - frac
                out[i_left] = frac
            return out

        if key == "rank":
            if not m_txt.strip():
                raise ValueError("Preset 'Rank' requires ':r' (e.g. Rank:3)")
            r = int(m_txt)
            if not (1 <= r <= k):
                raise ValueError(f"Rank:r requires integer r with 1 <= r <= {k} (got r={r})")
            out = np.zeros((k,), dtype=dtype)
            out[k - r] = 1.0
            return out

        if key in {"uppertailmean", "lowertailmean"}:
            if not m_txt.strip():
                raise ValueError(f"Preset '{name}' requires ':q' (e.g. {name}:0.25)")
            q = float(m_txt)
            if not (0.0 < q <= 1.0):
                raise ValueError(f"{name}:q requires 0 < q <= 1 (got q={q})")
            m = max(1, int(math.ceil(q * k)))
            out = np.zeros((k,), dtype=dtype)
            if key == "uppertailmean":
                out[k - m :] = 1.0 / m
            else:
                out[:m] = 1.0 / m
            return out

        if key == "lmoment":
            if not m_txt.strip():
                raise ValueError("Preset 'LMoment' requires ':r' (e.g. LMoment:2)")
            r = int(m_txt)
            return _l_moment_weights(k, r, dtype=dtype)

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
        elif key in {"midrangem", "topbot"}:
            out[:m] = 0.5 / m
            out[k - m :] += 0.5 / m
        elif key in {"winsorizedm", "windosrizedm"}:
            if 2 * m >= k:
                raise ValueError(f"WinsorizedM requires 2*m < k (got m={m}, k={k})")
            out[m : k - m] = 1.0 / k
            out[m] += m / k
            out[k - m - 1] += m / k
        elif key in {"trimm", "trimmedm", "trimmeanm"}:
            if 2 * m >= k:
                raise ValueError(f"TrimM requires 2*m < k (got m={m}, k={k})")
            out[m : k - m] = 1.0 / (k - 2 * m)
        else:
            raise ValueError(
                "Unknown l-stat preset. Supported: TopM:m, BotM:m, TrimM:m, WinsorizedM:m, MidrangeM:m, TopBot:m, ReMax, ReMin, Median, Rank:r, Quantile:q (Hazen default), QuantileWeibull:q, QuantileHazen:q, QuantileBlom:q, TopQuantile:q (Hazen default), TopQuantileWeibull:q, TopQuantileHazen:q, TopQuantileBlom:q, UpperTailMean:q, LowerTailMean:q, HarrellDavis:q, GiniMeanDifference, LMoment:r"
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
        # Numeric vectors are provided in top-rank order (j=1 highest).
        return a[::-1].copy()

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
        k_ord, k_eff = _require_integer_sampling_k(k, max_k=N, context="OrderStatTransform.precompute")
        W = precompute_W_unconditional(N, k_eff, dtype=dtype, chunk_size=chunk_size)

        A = B = C = None
        if compute_conditional:
            A, B, C = precompute_ABC_conditional_including_rank(
                N, k_eff, dtype=dtype, chunk_size=chunk_size
            )

        Wm = None
        if compute_leave_one_out:
            if k_eff > N - 1:
                raise ValueError("Leave-one-out requires integer k <= N-1")
            Wm = precompute_W_leave_one_out(N, k_eff, dtype=dtype, chunk_size=chunk_size)

        M_inc = M_loo = M_adv = None
        if compute_dense_matrices:
            M_inc = cls._build_dense_inclusion_matrix(A, B, C) if (A is not None and B is not None and C is not None) else None
            M_loo = cls._build_dense_leave_one_out_matrix(Wm, N, k_ord) if Wm is not None else None
            if M_inc is not None and M_loo is not None:
                M_adv = M_inc - M_loo

        return cls(N=N, k=k_ord, k_eff=k_eff, W=W, A=A, B=B, C=C, Wm=Wm, M_inc=M_inc, M_loo=M_loo, M_adv=M_adv)

    @classmethod
    def precompute_known_rp(cls, k: float, *, dtype=np.float64):
        """Create a lightweight transform for known-(r,p) methods with real k."""
        k_eff = float(k)
        if not (k_eff >= 1.0):
            raise ValueError("Require real k >= 1")
        k_ord = int(math.floor(k_eff))
        if k_ord < 1:
            raise ValueError("floor(k) must be >= 1")
        W = np.zeros((0, k_ord), dtype=dtype)
        return cls(N=0, k=k_ord, k_eff=k_eff, W=W, A=None, B=None, C=None, Wm=None, sampling_valid=False)

    def with_lstat_weights(self, a: np.ndarray) -> "OrderStatTransform":
        """Return a new transform with preweighted L-statistic coefficients."""
        self._require_sampling_valid("with_lstat_weights")
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
            sampling_valid=self.sampling_valid,
            M_inc_a=(np.tensordot(self.M_inc, a, axes=([2], [0])) if self.M_inc is not None else None),
            M_loo_a=(np.tensordot(self.M_loo, a, axes=([2], [0])) if self.M_loo is not None else None),
            M_adv_a=(np.tensordot(self.M_adv, a, axes=([2], [0])) if self.M_adv is not None else None),
        )

    @classmethod
    def precompute_lstat(cls, N: int, k: int, a: np.ndarray, **kwargs) -> "OrderStatTransform":
        """Precompute matrices and pre-apply L-statistic weights `a`."""
        return cls.precompute(N, k, **kwargs).with_lstat_weights(a)

    @classmethod
    def for_known_rp(cls, k: float, *, dtype=np.float64) -> "OrderStatTransform":
        """Construct a known-(r,p)-only transform for real k.

        Sampling/batch methods are intentionally disabled on the returned object;
        use the *_known_rp methods with an explicit branch for ambiguous
        fractional-k order statistics.
        """
        return cls.precompute_known_rp(k, dtype=dtype)

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
        self._require_sampling_valid("expected_orderstats")
        x_sorted, _ = self._sort_with_inverse_rank(x)
        return x_sorted @ self.W

    def expected_orderstats_inclusion(self, x: np.ndarray, *, method: str = "efficient") -> np.ndarray:
        """Return E[X_(j:k) | i included] for all i. Shape (N,k)."""
        self._require_sampling_valid("expected_orderstats_inclusion")
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
        self._require_sampling_valid("expected_orderstats_leave_one_out")
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
        self._require_sampling_valid("lstat_weight_by_rank")
        if a is None:
            if self.Wa is None:
                raise ValueError("No preweighted l-statistic vector is available. Pass a or use with_lstat_weights().")
            return self.Wa
        a = self._validate_a(a, self.k, dtype=self.W.dtype)
        return self.W @ a

    def lstat_weight_by_item(self, x: np.ndarray, a: Optional[np.ndarray] = None) -> np.ndarray:
        """Return item-weight vector in original index order (gradient away from ties)."""
        self._require_sampling_valid("lstat_weight_by_item")
        _, inv = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return w_rank[inv]

    def expected_lstat(self, x: np.ndarray, a: Optional[np.ndarray] = None) -> float:
        """Return E[T(S)] where T(S)=sum_j a_j X_(j:k)."""
        self._require_sampling_valid("expected_lstat")
        x_sorted, _ = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return float(x_sorted @ w_rank)

    def expected_lstat_inclusion(self, x: np.ndarray, a: Optional[np.ndarray] = None, *, method: str = "efficient") -> np.ndarray:
        """Return E[T(S) | i included] for all i. Shape (N,)."""
        self._require_sampling_valid("expected_lstat_inclusion")
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
        self._require_sampling_valid("expected_lstat_leave_one_out")
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
        self._require_sampling_valid("expected_orderstats_advantage")
        if method not in {"efficient", "matmul", "auto"}:
            raise ValueError("method must be one of {'efficient','matmul','auto'}")
        if method in {"matmul", "auto"} and self.M_adv is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return np.einsum("rmj,m->rj", self.M_adv, x_sorted)[inv, :]
        return self.expected_orderstats_inclusion(x, method=method) - self.expected_orderstats_leave_one_out(x, method=method)

    def expected_orderstats_known_rp(self, r: np.ndarray, p: np.ndarray, *, branch: Optional[str] = None) -> np.ndarray:
        v, _, _ = known_rp_orderstats(r, p, self.k_eff, dtype=self.W.dtype, branch=branch)
        return v

    def expected_orderstats_inclusion_known_rp(self, r: np.ndarray, p: np.ndarray, *, branch: Optional[str] = None) -> np.ndarray:
        _, q, _ = known_rp_orderstats(r, p, self.k_eff, dtype=self.W.dtype, branch=branch)
        return q

    def expected_orderstats_advantage_known_rp(self, r: np.ndarray, p: np.ndarray, *, branch: Optional[str] = None, detach_advantage: bool = True) -> np.ndarray:
        _, _, adv = known_rp_orderstats(r, p, self.k_eff, dtype=self.W.dtype, branch=branch)
        return adv.copy() if detach_advantage else adv

    def expected_lstat_known_rp(self, r: np.ndarray, p: np.ndarray, a: np.ndarray, *, branch: Optional[str] = None) -> float:
        branch = _resolve_known_rp_lstat_branch(a, branch, self.k_eff)
        a = self._validate_a(a, self.k, dtype=self.W.dtype)
        return float(self.expected_orderstats_known_rp(r, p, branch=branch) @ a)

    def expected_lstat_inclusion_known_rp(self, r: np.ndarray, p: np.ndarray, a: np.ndarray, *, branch: Optional[str] = None) -> np.ndarray:
        branch = _resolve_known_rp_lstat_branch(a, branch, self.k_eff)
        a = self._validate_a(a, self.k, dtype=self.W.dtype)
        return self.expected_orderstats_inclusion_known_rp(r, p, branch=branch) @ a

    def expected_lstat_advantage_known_rp(self, r: np.ndarray, p: np.ndarray, a: np.ndarray, *, branch: Optional[str] = None, detach_advantage: bool = True) -> np.ndarray:
        branch = _resolve_known_rp_lstat_branch(a, branch, self.k_eff)
        a = self._validate_a(a, self.k, dtype=self.W.dtype)
        out = self.expected_orderstats_advantage_known_rp(r, p, branch=branch, detach_advantage=detach_advantage) @ a
        return out.copy() if detach_advantage else out

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
        self._require_sampling_valid("expected_lstat_advantage")
        if method in {"matmul", "auto"} and self.M_adv_a is not None and a is None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return (self.M_adv_a @ x_sorted)[inv]
        return self.expected_lstat_inclusion(x, a, method=method) - self.expected_lstat_leave_one_out(x, a, method=method)


def known_rp_lstat(
    r: np.ndarray,
    p: np.ndarray,
    k: float,
    a: Any,
    *,
    branch: Optional[str] = None,
    dtype=np.float64,
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
    weights = OrderStatTransform._validate_a(a, k_ord, dtype=dtype)
    v, q, adv = known_rp_orderstats(r, p, k_eff, branch=branch, dtype=dtype)
    return v @ weights, q @ weights, adv @ weights
