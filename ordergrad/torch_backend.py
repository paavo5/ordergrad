from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, Tuple

try:
    import torch
except Exception as e:  # pragma: no cover
    raise ImportError(
        "ordergrad.torch_backend requires PyTorch. "
        "Install it with: `pip install ordergrad[torch]`"
    ) from e


def _log_choose(n: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Vectorized log binomial log C(n,r) with -inf for invalid."""
    # Accept tensor or scalar inputs; work in float64 for stability.
    n = torch.as_tensor(n, dtype=torch.float64)
    r = torch.as_tensor(r, dtype=torch.float64, device=n.device)
    valid = (n >= 0) & (r >= 0) & (r <= n)

    n0 = torch.where(valid, n, torch.zeros_like(n))
    r0 = torch.where(valid, r, torch.zeros_like(r))
    nr0 = torch.where(valid, n0 - r0, torch.zeros_like(n0))

    out = torch.lgamma(n0 + 1.0) - torch.lgamma(r0 + 1.0) - torch.lgamma(nr0 + 1.0)
    out = torch.where(valid, out, torch.full_like(out, float("-inf")))
    return out


def _betainc_regularized(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b) via continued fractions."""
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
    N: int, k: float, *, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None
) -> torch.Tensor:
    """W[m-1,j-1] = P(X_(j:k) == x_(m)) for uniform k-subset from N."""
    if not (1 <= k <= N):
        raise ValueError("Require 1 <= k <= N")
    device = device or torch.device("cpu")

    k_eff = float(k)
    log_den = _log_choose(
        torch.tensor(float(N), device=device), torch.tensor(k_eff, device=device)
    )

    def log_term(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m, k_eff - j)

    return _build_weight_matrix(
        N,
        int(float(k)//1),
        log_den,
        log_term,
        dtype=dtype,
        device=device,
        renormalize_cols=True,
    )


def precompute_ABC_conditional_including_rank(
    N: int, k: float, *, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute conditional matrices A,B,C (shape (N,k))."""
    if not (1 <= k <= N):
        raise ValueError("Require 1 <= k <= N")
    device = device or torch.device("cpu")

    k_eff = float(k)
    log_den = _log_choose(
        torch.tensor(float(N - 1), device=device), torch.tensor(k_eff - 1.0, device=device)
    )

    def logA(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m - 1, k_eff - j - 1.0)

    def logB(m, j):
        return _log_choose(m - 1, j - 1) + _log_choose(N - m, k_eff - j)

    def logC(m, j):
        return _log_choose(m - 2, j - 2) + _log_choose(N - m, k_eff - j)

    A = _build_weight_matrix(N, int(float(k)//1), log_den, logA, dtype=dtype, device=device, renormalize_cols=False)
    B = _build_weight_matrix(N, int(float(k)//1), log_den, logB, dtype=dtype, device=device, renormalize_cols=False)
    C = _build_weight_matrix(N, int(float(k)//1), log_den, logC, dtype=dtype, device=device, renormalize_cols=False)
    return A, B, C


def precompute_W_leave_one_out(
    N: int, k: float, *, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Wm for reduced population size (N-1). Shape (N-1,k), columns sum to 1."""
    if not (1 <= k <= N - 1):
        raise ValueError("Require 1 <= k <= N-1 for leave-one-out")
    device = device or torch.device("cpu")

    k_eff = float(k)
    log_den = _log_choose(
        torch.tensor(float(N - 1), device=device), torch.tensor(k_eff, device=device)
    )

    def log_term(p, j):
        return _log_choose(p - 1, j - 1) + _log_choose((N - 1) - p, k_eff - j)

    return _build_weight_matrix(
        N - 1,
        int(float(k)//1),
        log_den,
        log_term,
        dtype=dtype,
        device=device,
        renormalize_cols=True,
    )


def _binom_tail_table(k: float, F: torch.Tensor, k_ord: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """T[t,j-1] = P(Bin(k,F_t) >= j), with t=0..m and j=1..k_ord."""
    F = F.to(dtype=torch.float64, device=device)
    s = torch.arange(0, k_ord + 1, dtype=torch.float64, device=device)[:, None]
    logF = torch.where(F[None, :] > 0, torch.log(F[None, :]), torch.zeros_like(F[None, :]))
    log1mF = torch.where(F[None, :] < 1, torch.log1p(-F[None, :]), torch.zeros_like(F[None, :]))
    term1 = torch.where(s > 0, torch.where(F[None, :] > 0, s * logF, torch.full_like(s * logF, float("-inf"))), torch.zeros_like(s * logF))
    term2 = torch.where((float(k) - s) > 0, torch.where(F[None, :] < 1, (float(k) - s) * log1mF, torch.full_like(s * logF, float("-inf"))), torch.zeros_like(s * logF))
    logpmf = _log_choose(float(k), s) + term1 + term2
    pmf = torch.exp(logpmf)
    cols = [pmf[j:, :].sum(dim=0) for j in range(1, k_ord + 1)]
    return torch.stack(cols, dim=1).to(dtype=dtype)


def known_rp_orderstats(
    r: torch.Tensor,
    p: torch.Tensor,
    k: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
):
    """Exact known-(r,p) with-replacement order-statistics quantities."""
    device = device or (r.device if isinstance(r, torch.Tensor) else torch.device("cpu"))
    r = torch.as_tensor(r, dtype=torch.float64, device=device)
    p = torch.as_tensor(p, dtype=torch.float64, device=device)
    if r.ndim != 1 or p.ndim != 1 or r.shape[0] != p.shape[0]:
        raise ValueError("r and p must be 1D arrays of equal length")
    if torch.any(p < 0):
        raise ValueError("p must be nonnegative")
    p = p / p.sum()
    m = int(r.shape[0])
    k_eff = float(k)
    k_ord = int(k_eff // 1)
    if not (k_eff >= 1):
        raise ValueError("Require real k >= 1")
    if k_ord < 1:
        raise ValueError("floor(k) must be >= 1")

    perm = torch.argsort(r, stable=True)
    rs = r[perm]
    ps = p[perm]
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(m, device=device, dtype=perm.dtype)

    F = torch.cat([torch.tensor([0.0], dtype=torch.float64, device=device), torch.cumsum(ps, dim=0)], dim=0)
    T_k = _binom_tail_table(k_eff, F, k_ord, dtype=dtype, device=device)
    W = T_k[1:, :] - T_k[:-1, :]
    v = rs @ W

    T_km1 = _binom_tail_table(k_eff - 1.0, F, k_ord, dtype=dtype, device=device)
    q_sorted = []
    ar = torch.arange(m + 1, device=device)
    for b in range(m):
        delta = torch.cat([torch.tensor([0], dtype=torch.int64, device=device), (rs[b] <= rs).to(torch.int64)], dim=0)
        cols = []
        for jv in range(1, k_ord + 1):
            need = jv - delta
            idx = torch.clamp(need - 1, 0, k_ord - 1)
            take = T_km1[ar, idx]
            col = torch.where(need <= 0, torch.ones_like(need, dtype=torch.float64), torch.where(need <= k_ord, take, torch.zeros_like(need, dtype=torch.float64)))
            cols.append(col)
        Q = torch.stack(cols, dim=1)
        Wq = Q[1:, :] - Q[:-1, :]
        q_sorted.append(rs @ Wq)
    q_sorted = torch.stack(q_sorted, dim=0).to(dtype=dtype)
    adv_sorted = q_sorted - v.unsqueeze(0)
    return v.to(dtype=dtype), q_sorted[inv, :], adv_sorted[inv, :]


@dataclass(frozen=True)
class OrderStatTransform:
    """PyTorch implementation.

    Returned inclusion/leave-one-out matrices/vectors are in **original index order**.
    """

    N: int
    k: int
    k_eff: float
    W: torch.Tensor
    A: Optional[torch.Tensor]
    B: Optional[torch.Tensor]
    C: Optional[torch.Tensor]
    Wm: Optional[torch.Tensor]
    M_inc: Optional[torch.Tensor] = None
    M_loo: Optional[torch.Tensor] = None
    M_adv: Optional[torch.Tensor] = None
    M_inc_a: Optional[torch.Tensor] = None
    M_loo_a: Optional[torch.Tensor] = None
    M_adv_a: Optional[torch.Tensor] = None

    @classmethod
    def precompute(
        cls,
        N: int,
        k: float,
        *,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
        compute_conditional: bool = True,
        compute_leave_one_out: bool = True,
        compute_dense_matrices: bool = False,
    ) -> "OrderStatTransform":
        device = device or torch.device("cpu")
        k_eff = float(k)
        k_ord = int(k_eff // 1)
        if not (1 <= k_eff <= N):
            raise ValueError("Require real k with 1 <= k <= N")
        if not (1 <= k_ord <= N):
            raise ValueError("floor(k) must satisfy 1 <= floor(k) <= N")
        W = precompute_W_unconditional(N, k_eff, dtype=dtype, device=device)

        A = B = C = None
        if compute_conditional:
            A, B, C = precompute_ABC_conditional_including_rank(N, k_eff, dtype=dtype, device=device)

        Wm = None
        if compute_leave_one_out:
            if k_eff > N - 1:
                raise ValueError("Leave-one-out requires real k <= N-1")
            Wm = precompute_W_leave_one_out(N, k_eff, dtype=dtype, device=device)

        M_inc = M_loo = M_adv = None
        if compute_dense_matrices:
            M_inc = cls._build_dense_inclusion_matrix(A, B, C) if (A is not None and B is not None and C is not None) else None
            M_loo = cls._build_dense_leave_one_out_matrix(Wm, N, k_ord) if Wm is not None else None
            if M_inc is not None and M_loo is not None:
                M_adv = M_inc - M_loo

        return cls(N=N, k=k_ord, k_eff=k_eff, W=W, A=A, B=B, C=C, Wm=Wm, M_inc=M_inc, M_loo=M_loo, M_adv=M_adv)

    @staticmethod
    def _build_dense_inclusion_matrix(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        N, _ = A.shape
        r = torch.arange(N, device=A.device)[:, None]
        m = torch.arange(N, device=A.device)[None, :]
        lt = (m < r).unsqueeze(-1)
        eq = (m == r).unsqueeze(-1)
        gt = (m > r).unsqueeze(-1)
        return lt * A.unsqueeze(0) + eq * B.unsqueeze(0) + gt * C.unsqueeze(0)

    @staticmethod
    def _build_dense_leave_one_out_matrix(Wm: torch.Tensor, N: int, k: int) -> torch.Tensor:
        M = torch.zeros((N, N, k), dtype=Wm.dtype, device=Wm.device)
        for r in range(N):
            if r > 0:
                M[r, :r, :] = Wm[:r, :]
            if r < N - 1:
                M[r, r + 1 :, :] = Wm[r:, :]
        return M

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

    def expected_orderstats_inclusion(self, x: torch.Tensor, *, method: str = "efficient") -> torch.Tensor:
        if method not in {"efficient", "matmul", "auto"}:
            raise ValueError("method must be one of {'efficient','matmul','auto'}")

        x_sorted, inv = self._sort_with_inverse_rank(x)
        if method in {"matmul", "auto"} and self.M_inc is not None:
            return torch.einsum("rmj,m->rj", self.M_inc, x_sorted)[inv, :]

        if self.A is None or self.B is None or self.C is None:
            raise ValueError("Conditional matrices A,B,C were not precomputed.")

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

    def expected_orderstats_leave_one_out(self, x: torch.Tensor, *, method: str = "efficient") -> torch.Tensor:
        if method not in {"efficient", "matmul", "auto"}:
            raise ValueError("method must be one of {'efficient','matmul','auto'}")

        x_sorted, inv = self._sort_with_inverse_rank(x)
        if method in {"matmul", "auto"} and self.M_loo is not None:
            return torch.einsum("rmj,m->rj", self.M_loo, x_sorted)[inv, :]

        if self.Wm is None:
            raise ValueError("Leave-one-out matrix Wm was not precomputed.")
        if self.k > self.N - 1:
            raise ValueError("Leave-one-out requires k <= N-1")
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

    @staticmethod
    def _preset_lstat_weights(k: int, spec: str, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        text = str(spec).strip()
        if not text:
            raise ValueError("l-stat preset cannot be empty")
        name, _, m_txt = text.partition(":")
        key = name.strip().lower()

        if key in {"remax", "remin", "median", "ginimeandifference", "gmd"}:
            if m_txt.strip():
                raise ValueError(f"{name} does not take an m value")
            out = torch.zeros((k,), dtype=dtype, device=device)
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
            else:
                j = torch.arange(1, k + 1, dtype=dtype, device=device)
                out = (2.0 * (2.0 * j - (k + 1.0))) / (k * (k - 1.0))
            return out

        if key in {"harrelldavis", "harreldavis"}:
            if not m_txt.strip():
                raise ValueError("Preset 'HarrellDavis' requires ':q' (e.g. HarrellDavis:0.75)")
            q = float(m_txt)
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"HarrellDavis:q requires 0 <= q <= 1 (got q={q})")
            aa = (k + 1) * q
            bb = (k + 1) * (1.0 - q)
            u_hi = torch.arange(1, k + 1, dtype=torch.float64, device=device) / float(k)
            u_lo = torch.arange(0, k, dtype=torch.float64, device=device) / float(k)
            w = [
                _betainc_regularized(aa, bb, float(hi.item()))
                - _betainc_regularized(aa, bb, float(lo.item()))
                for lo, hi in zip(u_lo, u_hi)
            ]
            return torch.as_tensor(w, dtype=dtype, device=device)

        if key in {"quantile", "topquantile"}:
            if not m_txt.strip():
                raise ValueError(f"Preset '{name}' requires ':q' (e.g. {name}:0.25)")
            q = float(m_txt)
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"{name}:q requires 0 <= q <= 1 (got q={q})")
            # Quantile:q uses standard CDF convention (q mass below threshold).
            # TopQuantile:q uses top-tail convention (q mass above threshold).
            q_eff = q if key == "quantile" else (1.0 - q)
            # Interpolate between adjacent rank bins using rank centers
            # c_j = (j - 0.5) / k so boundaries split mass across neighbors.
            s_pos = q_eff * k + 0.5
            out = torch.zeros((k,), dtype=dtype, device=device)
            if s_pos <= 1.0:
                out[0] = 1.0
            elif s_pos >= float(k):
                out[k - 1] = 1.0
            else:
                left = int(math.floor(s_pos))
                frac = s_pos - float(left)
                out[left - 1] = 1.0 - frac
                out[left] = frac
            return out

        if key == "rank":
            if not m_txt.strip():
                raise ValueError("Preset 'Rank' requires ':r' (e.g. Rank:3)")
            r = int(m_txt)
            if not (1 <= r <= k):
                raise ValueError(f"Rank:r requires integer r with 1 <= r <= {k} (got r={r})")
            out = torch.zeros((k,), dtype=dtype, device=device)
            out[k - r] = 1.0
            return out

        if key in {"uppertailmean", "lowertailmean"}:
            if not m_txt.strip():
                raise ValueError(f"Preset '{name}' requires ':q' (e.g. {name}:0.25)")
            q = float(m_txt)
            if not (0.0 < q <= 1.0):
                raise ValueError(f"{name}:q requires 0 < q <= 1 (got q={q})")
            m = max(1, int(math.ceil(q * k)))
            out = torch.zeros((k,), dtype=dtype, device=device)
            if key == "uppertailmean":
                out[k - m :] = 1.0 / m
            else:
                out[:m] = 1.0 / m
            return out

        if key == "lmoment":
            if not m_txt.strip():
                raise ValueError("Preset 'LMoment' requires ':r' (e.g. LMoment:2)")
            r = int(m_txt)
            if not (1 <= r <= k):
                raise ValueError(f"LMoment:r requires integer r with 1 <= r <= k (got r={r}, k={k})")
            out = torch.zeros((k,), dtype=dtype, device=device)
            for m in range(r):
                sign = -1.0 if (m % 2) else 1.0
                coeff = sign * math.comb(r - 1, m) / math.comb(r - 1 + m, m)
                t = r - m
                den = math.comb(k - 1, t - 1)
                b_w = torch.zeros((k,), dtype=dtype, device=device)
                for j in range(t, k + 1):
                    b_w[j - 1] = math.comb(j - 1, t - 1) / den
                out = out + coeff * (b_w / float(k))
            return out / float(r)

        if not m_txt.strip():
            raise ValueError(f"Preset '{name}' requires ':m' (e.g. {name}:3)")
        m = int(m_txt)
        if not (1 <= m <= k):
            raise ValueError(f"m must satisfy 1 <= m <= {k}")

        out = torch.zeros((k,), dtype=dtype, device=device)
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
                "Unknown l-stat preset. Supported: TopM:m, BotM:m, TrimM:m, WinsorizedM:m, MidrangeM:m, TopBot:m, ReMax, ReMin, Median, Rank:r, Quantile:q, TopQuantile:q, UpperTailMean:q, LowerTailMean:q, HarrellDavis:q, GiniMeanDifference, LMoment:r"
            )
        return out

    def _coerce_a(self, a: Any) -> torch.Tensor:
        if isinstance(a, str):
            return self._preset_lstat_weights(self.k, a, dtype=self.W.dtype, device=self.W.device)
        a = torch.as_tensor(a, dtype=self.W.dtype, device=self.W.device)
        if a.shape != (self.k,):
            raise ValueError(f"a must be shape ({self.k},)")
        # Numeric vectors are provided in top-rank order (j=1 highest).
        return torch.flip(a, dims=(0,))

    def lstat_weight_by_rank(self, a: Optional[Any] = None) -> torch.Tensor:
        if a is None:
            if not hasattr(self, "Wa") or self.Wa is None:
                raise ValueError("No preweighted l-statistic vector is available. Pass a or use with_lstat_weights().")
            return self.Wa
        a = self._coerce_a(a)
        return self.W @ a

    def lstat_weight_by_item(self, x: torch.Tensor, a: Optional[Any] = None) -> torch.Tensor:
        _, inv = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return w_rank[inv]

    def expected_lstat(self, x: torch.Tensor, a: Optional[Any] = None) -> torch.Tensor:
        x_sorted, _ = self._sort_with_inverse_rank(x)
        w_rank = self.lstat_weight_by_rank(a)
        return (x_sorted * w_rank).sum()

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
        object.__setattr__(out, "M_inc_a", torch.tensordot(self.M_inc, a, dims=([2], [0])) if self.M_inc is not None else None)
        object.__setattr__(out, "M_loo_a", torch.tensordot(self.M_loo, a, dims=([2], [0])) if self.M_loo is not None else None)
        object.__setattr__(out, "M_adv_a", torch.tensordot(self.M_adv, a, dims=([2], [0])) if self.M_adv is not None else None)
        return out

    @classmethod
    def precompute_lstat(cls, N: int, k: int, a: Any, **kwargs) -> "OrderStatTransform":
        return cls.precompute(N, k, **kwargs).with_lstat_weights(a)

    def expected_lstat_inclusion(self, x: torch.Tensor, a: Optional[Any] = None, *, method: str = "efficient") -> torch.Tensor:
        if a is None and hasattr(self, "Aa") and self.Aa is not None and self.Ba is not None and self.Ca is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            xa = x_sorted * self.Aa
            pref_a = torch.cumsum(xa, dim=0)
            pref_a_excl = torch.cat([torch.zeros((1,), dtype=xa.dtype, device=xa.device), pref_a[:-1]], dim=0)
            xc = x_sorted * self.Ca
            pref_c = torch.cumsum(xc, dim=0)
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

    def expected_lstat_leave_one_out(self, x: torch.Tensor, a: Optional[Any] = None, *, method: str = "efficient") -> torch.Tensor:
        if a is None and hasattr(self, "Wma") and self.Wma is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            p1 = x_sorted[:-1] * self.Wma
            p2 = x_sorted[1:] * self.Wma
            pref1 = torch.cumsum(p1, dim=0)
            left = torch.cat([torch.zeros((1,), dtype=p1.dtype, device=p1.device), pref1], dim=0)
            pref2 = torch.cumsum(p2, dim=0)
            right = pref2[-1] - torch.cat([torch.zeros((1,), dtype=p2.dtype, device=p2.device), pref2], dim=0)
            return (left + right)[inv]
        if method in {"matmul", "auto"} and self.M_loo_a is not None and a is None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            return (self.M_loo_a @ x_sorted)[inv]
        E_loo = self.expected_orderstats_leave_one_out(x, method=method)
        if a is None:
            raise ValueError(f"a must be shape ({self.k},)")
        a = self._coerce_a(a)
        return E_loo @ a

    def expected_orderstats_advantage(self, x: torch.Tensor, *, method: str = "efficient", detach_advantage: bool = True) -> torch.Tensor:
        if method not in {"efficient", "matmul", "auto"}:
            raise ValueError("method must be one of {'efficient','matmul','auto'}")
        if method in {"matmul", "auto"} and self.M_adv is not None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            out = torch.einsum("rmj,m->rj", self.M_adv, x_sorted)[inv, :]
            return out.detach() if detach_advantage else out
        out = self.expected_orderstats_inclusion(x, method=method) - self.expected_orderstats_leave_one_out(x, method=method)
        return out.detach() if detach_advantage else out


    def expected_orderstats_known_rp(self, r: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        v, _, _ = known_rp_orderstats(r, p, self.k_eff, dtype=self.W.dtype, device=self.W.device)
        return v

    def expected_orderstats_inclusion_known_rp(self, r: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        _, q, _ = known_rp_orderstats(r, p, self.k_eff, dtype=self.W.dtype, device=self.W.device)
        return q

    def expected_orderstats_advantage_known_rp(self, r: torch.Tensor, p: torch.Tensor, *, detach_advantage: bool = True) -> torch.Tensor:
        _, _, adv = known_rp_orderstats(r, p, self.k_eff, dtype=self.W.dtype, device=self.W.device)
        return adv.detach() if detach_advantage else adv

    def expected_lstat_known_rp(self, r: torch.Tensor, p: torch.Tensor, a: Any) -> torch.Tensor:
        a = self._coerce_a(a)
        return self.expected_orderstats_known_rp(r, p) @ a

    def expected_lstat_inclusion_known_rp(self, r: torch.Tensor, p: torch.Tensor, a: Any) -> torch.Tensor:
        a = self._coerce_a(a)
        return self.expected_orderstats_inclusion_known_rp(r, p) @ a

    def expected_lstat_advantage_known_rp(self, r: torch.Tensor, p: torch.Tensor, a: Any, *, detach_advantage: bool = True) -> torch.Tensor:
        a = self._coerce_a(a)
        out = self.expected_orderstats_advantage_known_rp(r, p, detach_advantage=detach_advantage) @ a
        return out.detach() if detach_advantage else out

    def expected_lstat_advantage(self, x: torch.Tensor, a: Optional[Any] = None, *, method: str = "efficient", detach_advantage: bool = True) -> torch.Tensor:
        if method in {"matmul", "auto"} and self.M_adv_a is not None and a is None:
            x_sorted, inv = self._sort_with_inverse_rank(x)
            out = (self.M_adv_a @ x_sorted)[inv]
            return out.detach() if detach_advantage else out
        out = self.expected_lstat_inclusion(x, a, method=method) - self.expected_lstat_leave_one_out(x, a, method=method)
        return out.detach() if detach_advantage else out
