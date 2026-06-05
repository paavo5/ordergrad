import itertools

import numpy as np
import pytest

from ordergrad.numpy_backend import OrderStatTransform


def _exact_orderstat_means(x: np.ndarray, k: int) -> np.ndarray:
    """Exact E[X_(j:k)] by complete enumeration over all k-subsets."""
    N = x.shape[0]
    all_vals = []
    for subset in itertools.combinations(range(N), k):
        all_vals.append(np.sort(x[list(subset)]))
    return np.mean(np.asarray(all_vals, dtype=np.float64), axis=0)


def _exact_conditional_included_means(x: np.ndarray, i: int, k: int) -> np.ndarray:
    """Exact E[X_(j:k) | i included] by full enumeration."""
    N = x.shape[0]
    others = [j for j in range(N) if j != i]
    vals = []
    for subset_other in itertools.combinations(others, k - 1):
        subset = (i, *subset_other)
        vals.append(np.sort(x[list(subset)]))
    return np.mean(np.asarray(vals, dtype=np.float64), axis=0)


def _exact_leave_one_out_means(x: np.ndarray, i: int, k: int) -> np.ndarray:
    """Exact E[X_(j:k)] with i removed, by full enumeration."""
    N = x.shape[0]
    others = [j for j in range(N) if j != i]
    vals = []
    for subset in itertools.combinations(others, k):
        vals.append(np.sort(x[list(subset)]))
    return np.mean(np.asarray(vals, dtype=np.float64), axis=0)


def test_exact_matches_full_enumeration_on_small_problem():
    """Cross-check all APIs against exact combinatorial enumeration."""
    x = np.array([2.0, -1.5, 4.0, 0.25, 3.0], dtype=np.float64)
    N, k = x.size, 3

    os = OrderStatTransform.precompute(
        N,
        k,
        dtype=np.float64,
        compute_conditional=True,
        compute_leave_one_out=True,
    )

    np.testing.assert_allclose(os.orderstats(x), _exact_orderstat_means(x, k), atol=1e-12, rtol=1e-12)

    E_inc = os.orderstats_inclusion(x)
    E_loo = os.orderstats_leave_one_out(x)

    for i in range(N):
        np.testing.assert_allclose(E_inc[i], _exact_conditional_included_means(x, i, k), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(E_loo[i], _exact_leave_one_out_means(x, i, k), atol=1e-12, rtol=1e-12)


def test_edge_case_k_equals_1():
    """If k=1, the only order statistic is the sampled element itself."""
    x = np.array([3.0, -2.0, 1.0, 7.0], dtype=np.float64)
    N, k = x.size, 1

    os = OrderStatTransform.precompute(
        N,
        k,
        dtype=np.float64,
        compute_conditional=True,
        compute_leave_one_out=True,
    )

    # Unconditional: sample one element uniformly -> mean(x)
    np.testing.assert_allclose(os.orderstats(x), [x.mean()], atol=1e-12, rtol=1e-12)

    # Conditional inclusion: forced sample is {i} -> exactly x[i]
    E_inc = os.orderstats_inclusion(x)
    np.testing.assert_allclose(E_inc[:, 0], x, atol=1e-12, rtol=1e-12)

    # Leave-one-out: mean of remaining N-1 elements
    E_loo = os.orderstats_leave_one_out(x)
    expected = np.array([(np.sum(x) - x[i]) / (N - 1) for i in range(N)], dtype=np.float64)
    np.testing.assert_allclose(E_loo[:, 0], expected, atol=1e-12, rtol=1e-12)


def test_edge_case_k_equals_N_for_unconditional_and_conditional():
    """If k=N, the sampled set is always the full set, so order stats are deterministic."""
    x = np.array([5.0, -1.0, 2.5], dtype=np.float64)
    N, k = x.size, x.size

    os = OrderStatTransform.precompute(
        N,
        k,
        dtype=np.float64,
        compute_conditional=True,
        compute_leave_one_out=False,  # leave-one-out not defined when k=N
    )

    x_sorted = np.sort(x)
    np.testing.assert_allclose(os.orderstats(x), x_sorted, atol=1e-12, rtol=1e-12)

    E_inc = os.orderstats_inclusion(x)
    expected = np.tile(x_sorted, (N, 1))
    np.testing.assert_allclose(E_inc, expected, atol=1e-12, rtol=1e-12)


def test_lstat_identities():
    """L-statistic helpers should match explicit combinations of order-stat expectations."""
    rng = np.random.default_rng(0)
    N, k = 12, 5
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N, dtype=np.float64)
    a = rng.normal(size=k).astype(np.float64)

    os = OrderStatTransform.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)

    E = os.orderstats(x)
    E_inc = os.orderstats_inclusion(x)
    E_loo = os.orderstats_leave_one_out(x)

    a_internal = a[::-1]
    np.testing.assert_allclose(os.lstat(x, a), float(E @ a_internal), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(os.lstat_inclusion(x, a), E_inc @ a_internal, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(os.lstat_leave_one_out(x, a), E_loo @ a_internal, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(os.lstat_advantage(x, a), (E_inc @ a_internal) - (E_loo @ a_internal), atol=1e-12, rtol=1e-12)




def test_lstat_presets_match_manual_vectors():
    N, k = 12, 6
    x = np.linspace(-1.0, 1.0, N, dtype=np.float64)
    os = OrderStatTransform.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)

    presets = {
        "TopM:2": np.array([0.5, 0.5, 0, 0, 0, 0], dtype=np.float64),
        "BotM:3": np.array([0, 0, 0, 1 / 3, 1 / 3, 1 / 3], dtype=np.float64),
        "TrimM:1": np.array([0, 0.25, 0.25, 0.25, 0.25, 0], dtype=np.float64),
        "WinsorizedM:1": np.array([0, 1/3, 1/6, 1/6, 1/3, 0], dtype=np.float64),
        "WindosrizedM:1": np.array([0, 1/3, 1/6, 1/6, 1/3, 0], dtype=np.float64),
        "MidrangeM:2": np.array([0.25, 0.25, 0, 0, 0.25, 0.25], dtype=np.float64),
        "TopBot:2": np.array([0.25, 0.25, 0, 0, 0.25, 0.25], dtype=np.float64),
        "ReMax": np.array([1, 0, 0, 0, 0, 0], dtype=np.float64),
        "ReMin": np.array([0, 0, 0, 0, 0, 1], dtype=np.float64),
        "Median": np.array([0, 0, 0.5, 0.5, 0, 0], dtype=np.float64),
        "Rank:3": np.array([0, 0, 1, 0, 0, 0], dtype=np.float64),
        "Quantile:0": np.array([0, 0, 0, 0, 0, 1], dtype=np.float64),
        "TopQuantile:0": np.array([1, 0, 0, 0, 0, 0], dtype=np.float64),
        "Quantile:1": np.array([1, 0, 0, 0, 0, 0], dtype=np.float64),
        "TopQuantile:1": np.array([0, 0, 0, 0, 0, 1], dtype=np.float64),
        "Quantile:0.25": np.array([0, 0, 0, 0, 1.0, 0], dtype=np.float64),
        "TopQuantile:0.25": np.array([0, 1.0, 0, 0, 0, 0], dtype=np.float64),
        "Quantile:0.1": np.array([0, 0, 0, 0, 0.1, 0.9], dtype=np.float64),
        "TopQuantile:0.1": np.array([0.9, 0.1, 0, 0, 0, 0], dtype=np.float64),
        "QuantileHazen:0.25": np.array([0, 0, 0, 0, 1.0, 0], dtype=np.float64),
        "QuantileWeibull:0.25": np.array([0, 0, 0, 0, 0.75, 0.25], dtype=np.float64),
        "QuantileBlom:0.25": np.array([0, 0, 0, 0, 0.9375, 0.0625], dtype=np.float64),
        "TopQuantileHazen:0.25": np.array([0, 1.0, 0, 0, 0, 0], dtype=np.float64),
        "TopQuantileWeibull:0.25": np.array([0.25, 0.75, 0, 0, 0, 0], dtype=np.float64),
        "TopQuantileBlom:0.25": np.array([0.0625, 0.9375, 0, 0, 0, 0], dtype=np.float64),
        "UpperTailMean:0.25": np.array([0.5, 0.5, 0, 0, 0, 0], dtype=np.float64),
        "LowerTailMean:0.25": np.array([0, 0, 0, 0, 0.5, 0.5], dtype=np.float64),
        "GiniMeanDifference": np.array([1 / 3, 1 / 5, 1 / 15, -1 / 15, -1 / 5, -1 / 3], dtype=np.float64),
        "LMoment:1": np.ones((k,), dtype=np.float64) / k,
    }

    for spec, a in presets.items():
        np.testing.assert_allclose(os.lstat(x, spec), os.lstat(x, a), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(os.with_lstat_weights(spec).lstat_advantage(x), os.lstat_advantage(x, a), atol=1e-12, rtol=1e-12)


def test_winsorized_differs_from_trimmed_and_matches_definition():
    k = 8
    x = np.linspace(-2.0, 3.0, k, dtype=np.float64)
    os = OrderStatTransform.precompute(k, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=False)
    m = 2
    trim = os.lstat(x, f"TrimM:{m}")
    wins = os.lstat(x, f"WinsorizedM:{m}")

    xw = x.copy()
    xw[:m] = x[m]
    xw[-m:] = x[-m - 1]
    expected_wins = float(np.mean(xw))

    assert wins != trim
    np.testing.assert_allclose(wins, expected_wins, atol=1e-12, rtol=1e-12)




def test_harrel_davis_alias_matches_harrell_davis():
    k = 9
    x = np.linspace(-1.0, 1.0, k, dtype=np.float64)
    os = OrderStatTransform.precompute(k, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=False)
    np.testing.assert_allclose(
        os.lstat(x, "HarrelDavis:0.25"),
        os.lstat(x, "HarrellDavis:0.25"),
        atol=1e-12,
        rtol=1e-12,
    )

def test_harrell_davis_weights_form_valid_quantile_l_estimator():
    k = 9
    x = np.linspace(-1.0, 1.0, k, dtype=np.float64)
    os = OrderStatTransform.precompute(k, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=False)
    q = 0.5
    hd = os.with_lstat_weights(f"HarrellDavis:{q}")

    w_rank = hd.lstat_weight_by_rank()
    assert np.all(w_rank >= -1e-14)
    np.testing.assert_allclose(np.sum(w_rank), 1.0, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(hd.lstat(x), os.lstat(x, f"HarrellDavis:{q}"), atol=1e-12, rtol=1e-12)


def test_lstat_preset_validation_errors():
    os = OrderStatTransform.precompute(10, 5, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)

    with pytest.raises(ValueError, match="requires ':m'"):
        os.lstat(np.arange(10, dtype=np.float64), "TopM")
    with pytest.raises(ValueError, match="Unknown l-stat preset"):
        os.lstat(np.arange(10, dtype=np.float64), "Foo:2")
    with pytest.raises(ValueError, match=r"TrimM requires 2\*m < k"):
        os.lstat(np.arange(10, dtype=np.float64), "TrimM:3")
    with pytest.raises(ValueError, match=r"WinsorizedM requires 2\*m < k"):
        os.lstat(np.arange(10, dtype=np.float64), "WinsorizedM:3")
    with pytest.raises(ValueError, match="requires ':q'"):
        os.lstat(np.arange(10, dtype=np.float64), "HarrellDavis")
    with pytest.raises(ValueError, match="requires ':q'"):
        os.lstat(np.arange(10, dtype=np.float64), "Quantile")
    with pytest.raises(ValueError, match="requires ':r'"):
        os.lstat(np.arange(10, dtype=np.float64), "Rank")
    with pytest.raises(ValueError, match="requires ':q'"):
        os.lstat(np.arange(10, dtype=np.float64), "UpperTailMean")
    with pytest.raises(ValueError, match="requires ':q'"):
        os.lstat(np.arange(10, dtype=np.float64), "LowerTailMean")
    with pytest.raises(ValueError, match="requires ':r'"):
        os.lstat(np.arange(10, dtype=np.float64), "LMoment")

def test_preweighted_lstat_and_direct_advantage_match_baseline():
    rng = np.random.default_rng(123)
    N, k = 11, 4
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N)
    a = rng.normal(size=k).astype(np.float64)

    base = OrderStatTransform.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    weighted = base.with_lstat_weights(a)

    np.testing.assert_allclose(weighted.lstat(x), base.lstat(x, a), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(weighted.lstat_inclusion(x), base.lstat_inclusion(x, a), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(weighted.lstat_leave_one_out(x), base.lstat_leave_one_out(x, a), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(weighted.lstat_advantage(x), base.lstat_advantage(x, a), atol=1e-12, rtol=1e-12)

    adv_os = base.orderstats_advantage(x)
    np.testing.assert_allclose(adv_os, base.orderstats_inclusion(x) - base.orderstats_leave_one_out(x), atol=1e-12, rtol=1e-12)


def test_dense_matmul_path_matches_efficient_path():
    rng = np.random.default_rng(456)
    N, k = 10, 4
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N, dtype=np.float64)
    a = rng.normal(size=k).astype(np.float64)

    dense = OrderStatTransform.precompute(
        N,
        k,
        dtype=np.float64,
        compute_conditional=True,
        compute_leave_one_out=True,
        compute_dense_matrices=True,
    )

    np.testing.assert_allclose(
        dense.orderstats_inclusion(x, method="matmul"),
        dense.orderstats_inclusion(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        dense.orderstats_leave_one_out(x, method="matmul"),
        dense.orderstats_leave_one_out(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        dense.orderstats_advantage(x, method="matmul"),
        dense.orderstats_advantage(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )

    weighted = dense.with_lstat_weights(a)
    np.testing.assert_allclose(
        weighted.lstat_inclusion(x, method="matmul"),
        weighted.lstat_inclusion(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        weighted.lstat_leave_one_out(x, method="matmul"),
        weighted.lstat_leave_one_out(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        weighted.lstat_advantage(x, method="matmul"),
        weighted.lstat_advantage(x, method="efficient"),
        atol=1e-12,
        rtol=1e-12,
    )


def test_real_k_matches_integer_k_when_equal():
    rng = np.random.default_rng(7)
    N, k = 9, 4
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N, dtype=np.float64)

    os_int = OrderStatTransform.precompute(N, k, dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)
    os_real = OrderStatTransform.precompute(N, float(k), dtype=np.float64, compute_conditional=True, compute_leave_one_out=True)

    np.testing.assert_allclose(os_real.orderstats(x), os_int.orderstats(x), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(os_real.orderstats_inclusion(x), os_int.orderstats_inclusion(x), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(os_real.orderstats_leave_one_out(x), os_int.orderstats_leave_one_out(x), atol=1e-12, rtol=1e-12)


def test_real_k_fractional_is_supported_and_finite():
    rng = np.random.default_rng(8)
    N, k = 10, 4.3
    x = rng.normal(size=N).astype(np.float64) + 1e-6 * np.arange(N, dtype=np.float64)
    a = rng.normal(size=int(np.floor(k))).astype(np.float64)

    os_frac = OrderStatTransform.precompute(
        N,
        k,
        dtype=np.float64,
        compute_conditional=True,
        compute_leave_one_out=True,
        compute_dense_matrices=True,
    ).with_lstat_weights(a)

    E = os_frac.orderstats(x)
    E_inc = os_frac.orderstats_inclusion(x, method="matmul")
    E_loo = os_frac.orderstats_leave_one_out(x, method="matmul")
    Adv = os_frac.orderstats_advantage(x, method="matmul")
    l_adv = os_frac.lstat_advantage(x, method="matmul")

    assert np.isfinite(E).all()
    assert np.isfinite(E_inc).all()
    assert np.isfinite(E_loo).all()
    assert np.isfinite(Adv).all()
    assert np.isfinite(l_adv).all()


def test_known_rp_exact_matches_bruteforce_small_case():
    r = np.array([-1.0, 0.5, 2.0], dtype=np.float64)
    p = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    k = 3

    os = OrderStatTransform.precompute(8, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    v, q, adv = os.orderstats_known_rp(r, p), os.orderstats_inclusion_known_rp(r, p), os.orderstats_advantage_known_rp(r, p)

    import itertools
    vals = []
    probs = []
    for seq in itertools.product(range(len(r)), repeat=k):
        x = np.sort(r[list(seq)])
        pr = np.prod(p[list(seq)])
        vals.append(x)
        probs.append(pr)
    vals = np.asarray(vals)
    probs = np.asarray(probs)
    v_ref = (probs[:, None] * vals).sum(axis=0)
    np.testing.assert_allclose(v, v_ref, atol=1e-12, rtol=1e-12)

    for b in range(len(r)):
        vals_b = []
        probs_b = []
        for tail in itertools.product(range(len(r)), repeat=k - 1):
            seq = (b,) + tail
            x = np.sort(r[list(seq)])
            pr = np.prod(p[list(tail)])
            vals_b.append(x)
            probs_b.append(pr)
        vals_b = np.asarray(vals_b)
        probs_b = np.asarray(probs_b)
        q_ref = (probs_b[:, None] * vals_b).sum(axis=0)
        np.testing.assert_allclose(q[b], q_ref, atol=1e-12, rtol=1e-12)

    np.testing.assert_allclose(adv, q - v[None, :], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose((p[:, None] * adv).sum(axis=0), np.zeros_like(v), atol=1e-12, rtol=1e-12)


def test_fractional_range_lstat_presets_match_manual_vectors_and_equivalences():
    k = 10
    os = OrderStatTransform.precompute(k, k, dtype=np.float64, compute_conditional=False, compute_leave_one_out=False)
    x = np.linspace(-1.0, 1.0, k, dtype=np.float64)

    expected = {
        "RangeUpperTailMean:0.2:0.5": np.array([0, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0, 0], dtype=np.float64),
        "RangeLowerTailMean:0.2:0.5": np.array([0, 0, 0, 0, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0], dtype=np.float64),
        "TrimmedMeanFrac:0.2:0.8": np.array([0, 0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 0, 0], dtype=np.float64),
    }

    for spec, a in expected.items():
        np.testing.assert_allclose(os._preset_lstat_weights(k, spec, dtype=np.float64)[::-1], a, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(os.lstat(x, spec), os.lstat(x, a), atol=1e-12, rtol=1e-12)

    np.testing.assert_allclose(
        os._preset_lstat_weights(k, "RangeUpperTailMean:0.0:0.3", dtype=np.float64),
        os._preset_lstat_weights(k, "UpperTailMean:0.3", dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        os._preset_lstat_weights(k, "RangeLowerTailMean:0.0:0.3", dtype=np.float64),
        os._preset_lstat_weights(k, "LowerTailMean:0.3", dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        os._preset_lstat_weights(k, "TrimmedMeanFrac:0.2:0.8", dtype=np.float64),
        os._preset_lstat_weights(k, "TrimM:2", dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "spec",
    [
        "RangeLowerTailMean:0.2:0.2",
        "RangeLowerTailMean:0.3:0.2",
        "RangeLowerTailMean:-0.1:0.2",
        "RangeLowerTailMean:0.1:1.2",
        "TrimmedMeanFrac:0.6:0.4",
    ],
)
def test_fractional_range_lstat_presets_reject_invalid_ranges(spec):
    with pytest.raises(ValueError, match="lo|hi"):
        OrderStatTransform._preset_lstat_weights(10, spec, dtype=np.float64)
