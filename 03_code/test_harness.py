"""Unit tests for the CYP challenge evaluation harness.

Run with:  pytest -q test_harness.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cyp_metrics as cm
import cyp_tdi as ct
import validate_submission as vs

# ==================================================================================
# soft_threshold_ae — the CONFIRMED numerator
# ==================================================================================


def test_st_ae_zero_inside_interval():
    e = cm.soft_threshold_ae([5.0, 5.5, 6.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0])
    assert np.allclose(e, 0.0)  # bounds themselves are inside


def test_st_ae_distance_to_nearest_bound():
    e = cm.soft_threshold_ae([4.0, 7.0], [5.0, 5.0], [6.0, 6.0])
    assert np.allclose(e, [1.0, 1.0])


def test_st_ae_zero_width_interval_is_plain_ae():
    e = cm.soft_threshold_ae([4.0, 6.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    assert np.allclose(e, [1.0, 1.0, 0.0])


def test_st_ae_handles_swapped_bounds():
    a = cm.soft_threshold_ae([4.0], [6.0], [5.0])
    b = cm.soft_threshold_ae([4.0], [5.0], [6.0])
    assert np.allclose(a, b)


def test_st_ae_propagates_nan():
    e = cm.soft_threshold_ae([np.nan, 4.0, 4.0], [5.0, np.nan, 5.0], [6.0, 6.0, np.nan])
    assert np.all(np.isnan(e))


def test_st_ae_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        cm.soft_threshold_ae([1.0, 2.0], [1.0], [2.0])


def test_st_ae_never_negative():
    rng = np.random.default_rng(1)
    lo = rng.normal(5, 1, 500)
    hi = lo + rng.gamma(2, 0.3, 500)
    e = cm.soft_threshold_ae(rng.normal(5, 2, 500), lo, hi)
    assert (e >= 0).all()


# ==================================================================================
# st_rae
# ==================================================================================


def test_st_rae_perfect_prediction_is_zero():
    y = np.array([4.0, 5.0, 6.0, 7.0])
    v = cm.st_rae(y, y - 0.2, y + 0.2, y)
    assert v == pytest.approx(0.0)


def test_st_rae_denominator_none_is_mean_st_ae():
    yp = np.array([3.0, 8.0, 5.5])
    lo, hi = np.array([5.0, 5.0, 5.0]), np.array([6.0, 6.0, 6.0])
    # errors 2.0, 2.0, 0.0 -> mean 4/3
    assert cm.st_rae(yp, lo, hi, denominator="none") == pytest.approx(4 / 3)


def test_st_rae_none_needs_no_y_true():
    assert np.isfinite(cm.st_rae([3.0], [5.0], [6.0], denominator="none"))


def test_st_rae_other_policies_need_y_true():
    with pytest.raises(ValueError, match="needs y_true"):
        cm.st_rae([3.0], [5.0], [6.0], denominator="st_mean")


def test_st_rae_unknown_policy_raises():
    with pytest.raises(ValueError, match="unknown denominator policy"):
        cm.st_rae([3.0], [5.0], [6.0], [5.5], denominator="banana")


def test_st_rae_mae_mean_matches_hand_computation():
    yt = np.array([4.0, 6.0])
    yp = np.array([2.0, 8.0])
    lo, hi = yt.copy(), yt.copy()          # zero-width -> numerator is plain AE = 4
    # denominator sum|y - mean(y)| = |4-5| + |6-5| = 2
    assert cm.st_rae(yp, lo, hi, yt, denominator="mae_mean") == pytest.approx(2.0)


def test_st_rae_all_policies_finite_on_realistic_data():
    rng = np.random.default_rng(7)
    yt = rng.uniform(3.5, 8.0, 300)
    lo, hi = yt - 0.25, yt + 0.25
    yp = yt + rng.normal(0, 0.6, 300)
    vals = {
        p: cm.st_rae(yp, lo, hi, yt, denominator=p) for p in cm.DENOMINATOR_POLICIES
    }
    assert all(np.isfinite(v) for v in vals.values()), vals
    assert all(v >= 0 for v in vals.values()), vals


def test_st_rae_excludes_nan_ground_truth():
    yt = np.array([5.0, np.nan, 6.0])
    d = cm.st_rae(
        [5.0, 99.0, 6.0], yt - 0.1, yt + 0.1, yt, return_detail=True
    )
    assert d.n_scored == 2 and d.n_excluded == 1
    assert d.value == pytest.approx(0.0)


def test_st_rae_excludes_infinite_prediction():
    yt = np.array([5.0, 6.0])
    d = cm.st_rae([np.inf, 6.0], yt - 0.1, yt + 0.1, yt, return_detail=True)
    assert d.n_scored == 1 and d.n_excluded == 1


def test_st_rae_all_nan_returns_nan():
    d = cm.st_rae([np.nan] * 3, [1.0] * 3, [2.0] * 3, [1.5] * 3, return_detail=True)
    assert np.isnan(d.value) and d.n_scored == 0


def test_st_rae_degenerate_denominator_flagged_not_inf():
    # Intervals so wide that the mean predictor sits inside every one of them.
    yt = np.array([4.0, 5.0, 6.0])
    lo, hi = np.full(3, -50.0), np.full(3, 50.0)
    d = cm.st_rae([0.0, 0.0, 0.0], lo, hi, yt, denominator="st_mean", return_detail=True)
    assert d.degenerate_denominator is True
    assert np.isnan(d.value)


def test_st_rae_detail_counts_inside_interval():
    yt = np.array([5.0, 5.0, 5.0])
    d = cm.st_rae(
        [5.0, 5.0, 99.0], yt - 0.5, yt + 0.5, yt, denominator="none", return_detail=True
    )
    assert d.n_inside_interval == 2


def test_st_rae_monotone_in_prediction_error():
    yt = np.linspace(4, 8, 50)
    lo, hi = yt - 0.2, yt + 0.2
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1, 50)
    vals = [
        cm.st_rae(yt + k * noise, lo, hi, yt, denominator="none") for k in (0, 0.5, 1, 2)
    ]
    assert vals == sorted(vals), vals


def test_st_rae_wide_ci_neutralises_error():
    """The core design claim: wide credible intervals absorb prediction error."""
    yt = np.full(100, 3.5)  # low-activity compounds
    yp = np.full(100, 5.0)  # badly over-confident
    tight = cm.st_rae(yp, yt - 0.1, yt + 0.1, yt, denominator="none")
    wide = cm.st_rae(yp, yt - 2.0, yt + 2.0, yt, denominator="none")
    assert tight == pytest.approx(1.4)
    assert wide == pytest.approx(0.0)


# ==================================================================================
# ma_st_rae
# ==================================================================================


def _toy_frames(n=60, seed=0):
    rng = np.random.default_rng(seed)
    yt = pd.DataFrame(
        {ep: rng.uniform(3.5, 8.0, n) for ep in cm.REGRESSION_ENDPOINTS}
    )
    lo = yt - 0.25
    hi = yt + 0.25
    return yt, lo, hi, rng


def test_ma_st_rae_is_unweighted_mean_of_endpoints():
    yt, lo, hi, rng = _toy_frames()
    yp = yt + rng.normal(0, 0.7, yt.shape)
    macro, per = cm.ma_st_rae(yp, lo, hi, yt, return_per_endpoint=True)
    assert macro == pytest.approx(float(np.mean(list(per.values()))))
    assert set(per) == set(cm.REGRESSION_ENDPOINTS)


def test_ma_st_rae_perfect_is_zero():
    yt, lo, hi, _ = _toy_frames()
    assert cm.ma_st_rae(yt, lo, hi, yt) == pytest.approx(0.0)


def test_ma_st_rae_skips_nan_endpoint_by_default():
    yt, lo, hi, rng = _toy_frames()
    yp = yt + rng.normal(0, 0.5, yt.shape)
    dead = cm.REGRESSION_ENDPOINTS[0]
    yp = yp.copy()
    yp[dead] = np.nan  # that endpoint becomes unscorable
    skipped = cm.ma_st_rae(yp, lo, hi, yt, skip_nan_endpoints=True)
    poisoned = cm.ma_st_rae(yp, lo, hi, yt, skip_nan_endpoints=False)
    assert np.isfinite(skipped)
    assert np.isnan(poisoned)


# ==================================================================================
# secondary battery
# ==================================================================================


def test_mae_and_rae_hand_values():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([2.0, 2.0, 2.0])
    assert cm.mae(yt, yp) == pytest.approx(2 / 3)
    # sum|y-yhat| = 2 ; sum|y-mean| = 1+0+1 = 2
    assert cm.rae(yt, yp) == pytest.approx(1.0)


def test_r2_perfect_and_mean_predictor():
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    assert cm.r2(yt, yt) == pytest.approx(1.0)
    assert cm.r2(yt, np.full(4, yt.mean())) == pytest.approx(0.0)


def test_r2_is_nan_for_constant_truth():
    assert np.isnan(cm.r2(np.ones(5), np.arange(5.0)))


def test_rank_correlations_perfect_and_reversed():
    yt = np.arange(20.0)
    assert cm.spearman_r(yt, yt) == pytest.approx(1.0)
    assert cm.kendall_tau(yt, yt) == pytest.approx(1.0)
    assert cm.spearman_r(yt, -yt) == pytest.approx(-1.0)


def test_rank_correlations_nan_for_constant_prediction():
    yt = np.arange(20.0)
    assert np.isnan(cm.spearman_r(yt, np.ones(20)))
    assert np.isnan(cm.kendall_tau(yt, np.ones(20)))


def test_secondary_metrics_exclude_nan_pairs():
    yt = np.array([1.0, np.nan, 3.0])
    yp = np.array([1.0, 5.0, 3.0])
    assert cm.mae(yt, yp) == pytest.approx(0.0)


def test_endpoint_metrics_has_exact_leaderboard_keys():
    yt, lo, hi, rng = _toy_frames(40)
    ep = cm.REGRESSION_ENDPOINTS[0]
    row = cm.endpoint_metrics(yt[ep], yt[ep] + 0.1, lo[ep], hi[ep])
    assert tuple(row) == cm.REGRESSION_METRICS


# ==================================================================================
# bootstrap
# ==================================================================================


def test_bootstrap_indices_shape_and_determinism():
    a = cm.make_bootstrap_indices(50, 100, seed=42)
    b = cm.make_bootstrap_indices(50, 100, seed=42)
    assert a.shape == (100, 50)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, cm.make_bootstrap_indices(50, 100, seed=43))
    assert a.min() >= 0 and a.max() < 50


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(5)
    x = rng.normal(3.0, 1.0, 400)
    res = cm.bootstrap_ci(lambda sel: float(np.mean(x[sel])), 400, metric="mean", n=500)
    assert res.ci_low < res.point < res.ci_high
    assert res.n_bootstrap == 500 and res.n_valid == 500
    assert res.mean == pytest.approx(res.point, abs=0.1)


def test_bootstrap_ci_default_is_1000_resamples():
    res = cm.bootstrap_ci(lambda sel: float(len(sel)), 10, metric="n")
    assert res.n_bootstrap == cm.N_BOOTSTRAP == 1000


def test_bootstrap_ci_all_nan_metric():
    res = cm.bootstrap_ci(lambda sel: np.nan, 10, metric="dead", n=20)
    assert res.n_valid == 0 and np.isnan(res.mean)


def test_bootstrap_ci_rejects_wrong_index_shape():
    with pytest.raises(ValueError, match="indices must have shape"):
        cm.bootstrap_ci(lambda sel: 1.0, 10, indices=np.zeros((5, 9), dtype=int))


def test_bootstrap_result_as_row_keys():
    res = cm.bootstrap_ci(lambda sel: 1.0, 5, metric="RAE", n=10)
    assert set(res.as_row()) == {
        "RAE_mean",
        "RAE_std",
        "RAE_point",
        "RAE_ci_low",
        "RAE_ci_high",
    }


def test_bootstrap_results_frame_layout_matches_space():
    """One row per (Sample, Endpoint), including the synthetic 'MA' pseudo-endpoint."""
    yt, lo, hi, rng = _toy_frames(40)
    yp = yt + rng.normal(0, 0.5, yt.shape)
    boot = cm.bootstrap_results_frame(yt, yp, lo, hi, n=25, seed=1)
    assert list(boot.columns) == ["Sample", "Endpoint", *cm.REGRESSION_METRICS]
    assert len(boot) == 25 * (len(cm.REGRESSION_ENDPOINTS) + 1)
    assert set(boot["Endpoint"]) == set(cm.REGRESSION_ENDPOINTS) | {"MA"}
    assert boot["Sample"].nunique() == 25


def test_ma_row_is_mean_of_endpoint_rows_within_a_sample():
    yt, lo, hi, rng = _toy_frames(40)
    yp = yt + rng.normal(0, 0.5, yt.shape)
    boot = cm.bootstrap_results_frame(yt, yp, lo, hi, n=5, seed=2)
    for s in range(5):
        g = boot[boot["Sample"] == s]
        eps = g[g["Endpoint"] != "MA"]["RAE"].to_numpy()
        ma = float(g[g["Endpoint"] == "MA"]["RAE"].iloc[0])
        assert ma == pytest.approx(float(np.nanmean(eps)))


def test_averaged_results_frame_has_mean_std_schema():
    yt, lo, hi, rng = _toy_frames(40)
    yp = yt + rng.normal(0, 0.5, yt.shape)
    boot = cm.bootstrap_results_frame(yt, yp, lo, hi, n=30, seed=3)
    avg = cm.averaged_results_frame(boot)
    for m in cm.REGRESSION_METRICS:
        assert f"{m}_mean" in avg.columns and f"{m}_std" in avg.columns
    assert len(avg) == len(cm.REGRESSION_ENDPOINTS) + 1


def test_paired_delta_shared_indices_ranks_better_predictor():
    yt, lo, hi, rng = _toy_frames(80, seed=9)
    good = yt + rng.normal(0, 0.2, yt.shape)
    bad = yt + rng.normal(0, 2.0, yt.shape)
    idx = cm.make_bootstrap_indices(len(yt), 200, seed=11)
    ba = cm.bootstrap_results_frame(yt, good, lo, hi, indices=idx)
    bb = cm.bootstrap_results_frame(yt, bad, lo, hi, indices=idx)
    d = cm.paired_delta(ba, bb, metric="RAE")
    assert d["n"] == 200
    assert d["a_wins"] > d["b_wins"]
    assert d["mean_delta"] < 0  # lower RAE is better, so A - B < 0
    assert d["p_two_sided"] < 0.05


# ==================================================================================
# TDI labeler — every boundary
# ==================================================================================


def test_tdi_positive_clear_case():
    cats, labels = ct.label_tdi([6.0], [6.5])
    assert cats[0] == "positive" and labels[0] == 1.0


def test_tdi_negative_clear_case():
    cats, labels = ct.label_tdi([6.0], [6.1])
    assert cats[0] == "negative" and labels[0] == 0.0


def test_tdi_shift_exactly_threshold_is_negative():
    """shift == 0.301 exactly: the rule is 'exceeds', so this is NEGATIVE."""
    cats, labels = ct.label_tdi([6.0], [6.301])
    assert cats[0] == "negative" and labels[0] == 0.0


def test_tdi_shift_just_above_threshold_is_positive():
    cats, _ = ct.label_tdi([6.0], [6.302])
    assert cats[0] == "positive"


def test_tdi_negative_shift_is_negative():
    cats, _ = ct.label_tdi([6.0], [5.0])
    assert cats[0] == "negative"


def test_tdi_inferred_positive():
    cats, labels = ct.label_tdi([3.5], [5.0])
    assert cats[0] == "inferred_positive" and labels[0] == 1.0


def test_tdi_tdi_exactly_4_301_is_band_negative():
    """tdi == 4.301 exactly fails 'exceeds 4.301' and also fails 'tdi < 4'.

    It therefore lands in the previously undefined band, which the released
    labels resolve as negative.
    """
    cats, labels = ct.label_tdi([3.5], [4.301])
    assert cats[0] == "band_negative" and labels[0] == 0.0
    cats_s, labels_s = ct.label_tdi([3.5], [4.301], organizer_rule=False)
    assert cats_s[0] == "unassignable" and np.isnan(labels_s[0])


def test_tdi_just_above_4_301_is_inferred_positive():
    cats, _ = ct.label_tdi([3.5], [4.3011])
    assert cats[0] == "inferred_positive"


def test_tdi_assigned_negative():
    cats, labels = ct.label_tdi([3.0], [3.2])
    assert cats[0] == "assigned_negative" and labels[0] == 0.0


def test_tdi_tdi_exactly_4_is_band_negative():
    """Confirmed against the released labels: the band is negative, not excluded."""
    cats, labels = ct.label_tdi([3.5], [4.0])
    assert cats[0] == "band_negative" and labels[0] == 0.0
    # the strict published reading still leaves it unassignable
    cats_s, labels_s = ct.label_tdi([3.5], [4.0], organizer_rule=False)
    assert cats_s[0] == "unassignable" and np.isnan(labels_s[0])


def test_tdi_band_is_negative_in_released_labels():
    """direct < 4 with 4 <= tdi <= 4.301 was undefined in the FAQ.

    The 2026-08-17 training release resolves it: all 18 CYP2D6 and 159 CYP3A4
    rows in this band carry ``is_TDI = False``.
    """
    cats, labels = ct.label_tdi([3.5, 3.5, 3.5], [4.0, 4.15, 4.301])
    assert list(cats) == ["band_negative"] * 3
    assert np.all(labels == 0.0)
    cats_s, labels_s = ct.label_tdi(
        [3.5, 3.5, 3.5], [4.0, 4.15, 4.301], organizer_rule=False
    )
    assert list(cats_s) == ["unassignable"] * 3
    assert np.all(np.isnan(labels_s))


def test_tdi_direct_exactly_4_is_unassignable_strict():
    """Every published rule uses a strict inequality against 4, so direct == 4 falls
    through both the 'above 4' and 'below 4' branches."""
    cats, labels = ct.label_tdi([4.0, 4.0], [4.5, 3.0])
    assert list(cats) == ["unassignable", "unassignable"]
    assert np.all(np.isnan(labels))


def test_tdi_non_strict_mode_is_exhaustive():
    """With strict_boundaries=False no compound is left unassignable."""
    rng = np.random.default_rng(4)
    d = np.concatenate([rng.uniform(2, 9, 400), [4.0, 4.0, 3.5, 3.5]])
    t = np.concatenate([rng.uniform(2, 9, 400), [4.5, 3.0, 4.0, 4.301]])
    cats, labels = ct.label_tdi(d, t, strict_boundaries=False)
    assert "unassignable" not in set(cats)
    assert not np.isnan(labels).any()


def test_tdi_non_strict_moves_the_gap_to_assigned_negative():
    cats, labels = ct.label_tdi([3.5, 3.5], [4.0, 4.2], strict_boundaries=False)
    assert list(cats) == ["assigned_negative", "assigned_negative"]
    assert list(labels) == [0.0, 0.0]


def test_tdi_non_strict_direct_exactly_4_uses_shift_rule():
    cats, _ = ct.label_tdi([4.0, 4.0], [4.5, 4.1], strict_boundaries=False)
    assert list(cats) == ["positive", "negative"]


def test_tdi_boundary_atol_defeats_binary_rounding():
    """6.301 - 6.0 == 0.30100000000000016 in binary float, which would spuriously
    'exceed' 0.301 and flip a negative to a positive without BOUNDARY_ATOL."""
    assert (6.301 - 6.0) > ct.TDI_SHIFT_THRESHOLD  # the raw hazard
    cats, _ = ct.label_tdi([6.0], [6.301])
    assert cats[0] == "negative"  # tolerance restores the intended call


def test_tdi_atol_does_not_swallow_real_differences():
    """A tolerance of 1e-9 must not change any call a DRC fit could plausibly produce."""
    cats, _ = ct.label_tdi([6.0, 6.0], [6.3011, 6.3009])
    assert list(cats) == ["positive", "negative"]


def test_tdi_atol_is_configurable():
    cats, _ = ct.label_tdi([6.0], [6.31], atol=0.05)
    assert cats[0] == "negative"  # a deliberately huge tolerance absorbs the shift


def test_tdi_missing_direct_arm_is_negative_missing_tdi_arm_is_unassignable():
    """The two NaN cases are NOT symmetric in the organizers' convention.

    A missing direct arm is labelled negative regardless of the TDI-arm value
    (1,249/1,249 CYP3A4 and 4/4 CYP2D6 released rows are ``False``, including
    1,055 whose TDI-arm pIC50 exceeds 4.301). A missing TDI arm stays excluded.
    """
    cats, labels = ct.label_tdi([np.nan, 6.0], [6.5, np.nan])
    assert cats[0] == "no_direct_arm_negative" and labels[0] == 0.0
    assert cats[1] == "unassignable" and np.isnan(labels[1])
    # a missing direct arm is negative even with strongly measurable TDI
    cats_hi, labels_hi = ct.label_tdi([np.nan], [7.5])
    assert cats_hi[0] == "no_direct_arm_negative" and labels_hi[0] == 0.0
    # strict published reading excludes both
    cats_s, labels_s = ct.label_tdi([np.nan, 6.0], [6.5, np.nan], organizer_rule=False)
    assert list(cats_s) == ["unassignable", "unassignable"]
    assert np.all(np.isnan(labels_s))


def test_tdi_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        ct.label_tdi([1.0, 2.0], [1.0])


def test_tdi_positive_class_is_positives_plus_inferred():
    d = np.array([6.0, 3.5, 6.0, 3.0])
    t = np.array([7.0, 5.0, 6.05, 3.1])
    cats, labels = ct.label_tdi(d, t)
    assert list(cats) == ["positive", "inferred_positive", "negative", "assigned_negative"]
    assert list(labels) == [1.0, 1.0, 0.0, 0.0]


def test_label_tdi_frame_adds_expected_columns():
    df = pd.DataFrame(
        {
            "CYP3A4_pIC50_direct_inhibition": [6.0, 3.5],
            "CYP3A4_pIC50_tdi": [7.0, 5.0],
        }
    )
    out = ct.label_tdi_frame(df, "CYP3A4")
    assert list(out["CYP3A4_tdi_category"]) == ["positive", "inferred_positive"]
    assert list(out["CYP3A4_is_TDI_label"]) == [1.0, 1.0]
    assert out["CYP3A4_tdi_shift"].iloc[0] == pytest.approx(1.0)


def test_label_tdi_frame_missing_column_raises():
    with pytest.raises(KeyError):
        ct.label_tdi_frame(pd.DataFrame({"a": [1]}), "CYP3A4")


# ==================================================================================
# classification metrics
# ==================================================================================


def test_confusion_counts_hand_values():
    c = ct.confusion_counts([1, 1, 0, 0], [1, 0, 1, 0])
    assert c == {"TP": 1, "TN": 1, "FP": 1, "FN": 1, "n": 4}


def test_mcc_perfect_is_one():
    assert ct.mcc([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_mcc_inverted_is_minus_one():
    assert ct.mcc([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)


def test_mcc_balanced_random_is_zero():
    assert ct.mcc([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_mcc_single_class_prediction_is_zero_by_convention():
    """MCC's denominator is 0 here. We define the result as 0.0 (sklearn convention)."""
    assert ct.mcc([1, 1, 0, 0], [1, 1, 1, 1]) == 0.0
    assert ct.mcc([1, 1, 0, 0], [0, 0, 0, 0]) == 0.0


def test_mcc_single_class_truth_is_zero_by_convention():
    assert ct.mcc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.0


def test_mcc_undefined_override_gives_nan():
    assert np.isnan(ct.mcc([1, 1, 0, 0], [1, 1, 1, 1], undefined=float("nan")))


def test_mcc_all_nan_labels_returns_nan():
    assert np.isnan(ct.mcc([np.nan, np.nan], [1, 0]))


def test_classification_metrics_exclude_unassignable_labels():
    """NaN ground truth (unassignable) drops out; the remaining 2 are perfect."""
    yt = [1.0, np.nan, 0.0]
    yp = [True, True, False]
    m = ct.classification_metrics(yt, yp)
    assert m["MCC"] == pytest.approx(1.0)
    assert m["Accuracy"] == pytest.approx(1.0)
    assert ct.confusion_counts(yt, yp)["n"] == 2


def test_precision_recall_f1_hand_values():
    yt = [1, 1, 1, 0, 0]
    yp = [1, 1, 0, 1, 0]
    assert ct.precision(yt, yp) == pytest.approx(2 / 3)
    assert ct.recall(yt, yp) == pytest.approx(2 / 3)
    assert ct.f1(yt, yp) == pytest.approx(2 / 3)
    assert ct.accuracy(yt, yp) == pytest.approx(3 / 5)


def test_precision_undefined_when_nothing_predicted_positive():
    assert ct.precision([1, 0], [0, 0]) == 0.0
    assert np.isnan(ct.precision([1, 0], [0, 0], undefined=float("nan")))


def test_recall_undefined_when_no_true_positive():
    assert ct.recall([0, 0], [1, 0]) == 0.0


def test_f1_zero_when_precision_and_recall_zero():
    assert ct.f1([1, 1], [0, 0]) == 0.0


def test_classification_accepts_bools_and_ints():
    a = ct.classification_metrics([True, False, True], [1, 0, 1])
    assert a["Accuracy"] == pytest.approx(1.0)


def test_classification_rejects_non_binary_numbers():
    with pytest.raises(ValueError, match="non-binary value"):
        ct.confusion_counts([1, 0], [0.5, 0])


def test_classification_rejects_strings():
    with pytest.raises(ValueError, match="non-boolean value"):
        ct.confusion_counts([1, 0], ["yes", "no"])


def test_classification_metrics_keys_match_leaderboard():
    m = ct.classification_metrics([1, 0], [1, 0])
    assert tuple(m) == ct.CLASSIFICATION_METRICS


def test_bootstrap_classification_all_metrics_and_1000_default():
    rng = np.random.default_rng(6)
    yt = rng.integers(0, 2, 200).astype(float)
    yp = np.where(rng.random(200) < 0.85, yt, 1 - yt)
    res = ct.bootstrap_classification(yt, yp, n=200, seed=1)
    assert set(res) == set(ct.CLASSIFICATION_METRICS)
    assert res["MCC"].n_bootstrap == 200
    assert res["MCC"].ci_low < res["MCC"].point < res["MCC"].ci_high


def test_macro_classification_over_two_endpoints():
    yt = pd.DataFrame({"CYP2D6_is_TDI": [1, 0, 1, 0], "CYP3A4_is_TDI": [1, 1, 0, 0]})
    yp = pd.DataFrame({"CYP2D6_is_TDI": [1, 0, 1, 0], "CYP3A4_is_TDI": [0, 0, 1, 1]})
    macro, per = ct.macro_classification(yt, yp)
    assert per["CYP2D6_is_TDI"]["MCC"] == pytest.approx(1.0)
    assert per["CYP3A4_is_TDI"]["MCC"] == pytest.approx(-1.0)
    assert macro["MCC"] == pytest.approx(0.0)
    assert tuple(macro) == ct.CLASSIFICATION_METRICS


# ==================================================================================
# validator
# ==================================================================================


def _reg_df(n=750):
    return pd.DataFrame(
        {
            "SMILES": ["CCO"] * n,
            "Molecule_Name": [f"OADMET-{i:05d}" for i in range(n)],
            **{ep: np.full(n, 5.5) for ep in vs.REGRESSION_ENDPOINTS},
        }
    )


def _cls_df(n=750):
    return pd.DataFrame(
        {
            "SMILES": ["CCO"] * n,
            "Molecule_Name": [f"OADMET-{i:05d}" for i in range(n)],
            **{
                ep: np.tile([True, False], n // 2)
                for ep in vs.CLASSIFICATION_ENDPOINTS
            },
        }
    )


def test_validator_accepts_good_regression_parquet(tmp_path):
    p = tmp_path / "r.parquet"
    _reg_df().to_parquet(p, index=False)
    assert vs.validate(p, "regression").ok


def test_validator_accepts_good_regression_csv(tmp_path):
    p = tmp_path / "r.csv"
    _reg_df().to_csv(p, index=False)
    assert vs.validate(p, "regression").ok


def test_validator_accepts_good_classification(tmp_path):
    p = tmp_path / "c.parquet"
    _cls_df().to_parquet(p, index=False)
    assert vs.validate(p, "classification").ok


def test_validator_rejects_wrong_row_count(tmp_path):
    p = tmp_path / "r.parquet"
    _reg_df(749).to_parquet(p, index=False)
    rep = vs.validate(p, "regression")
    assert not rep.ok and any("Expected 750 rows, got 749" in e for e in rep.errors)


def test_validator_rejects_missing_column(tmp_path):
    p = tmp_path / "r.parquet"
    _reg_df().drop(columns=[vs.REGRESSION_ENDPOINTS[0]]).to_parquet(p, index=False)
    rep = vs.validate(p, "regression")
    assert not rep.ok and any("Missing required columns" in e for e in rep.errors)


def test_validator_is_case_sensitive_and_says_so(tmp_path):
    p = tmp_path / "r.parquet"
    df = _reg_df().rename(columns={"SMILES": "smiles"})
    df.to_parquet(p, index=False)
    rep = vs.validate(p, "regression")
    assert not rep.ok
    assert any("case mismatch" in e for e in rep.errors)


def test_validator_tolerates_extra_columns_with_a_warning(tmp_path):
    p = tmp_path / "r.parquet"
    df = _reg_df()
    df["my_uncertainty"] = 0.3
    df.to_parquet(p, index=False)
    rep = vs.validate(p, "regression")
    assert rep.ok
    assert any("beyond the required set" in w for w in rep.warnings)


def test_validator_rejects_nan_pic50(tmp_path):
    p = tmp_path / "r.parquet"
    df = _reg_df()
    df.loc[0, vs.REGRESSION_ENDPOINTS[0]] = np.nan
    df.to_parquet(p, index=False)
    rep = vs.validate(p, "regression")
    assert not rep.ok and any("contains NaN values" in e for e in rep.errors)


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_validator_rejects_infinite_pic50(tmp_path, bad):
    p = tmp_path / "r.parquet"
    df = _reg_df()
    df.loc[0, vs.REGRESSION_ENDPOINTS[1]] = bad
    df.to_parquet(p, index=False)
    rep = vs.validate(p, "regression")
    assert not rep.ok and any("infinite values" in e for e in rep.errors)


def test_validator_rejects_string_pic50_column(tmp_path):
    p = tmp_path / "r.parquet"
    df = _reg_df()
    df[vs.REGRESSION_ENDPOINTS[2]] = "not a number"
    df.to_parquet(p, index=False)
    rep = vs.validate(p, "regression")
    assert not rep.ok and any("rather than a float dtype" in e for e in rep.errors)


def test_validator_warns_on_implausible_pic50_range(tmp_path):
    p = tmp_path / "r.parquet"
    df = _reg_df()
    df[vs.REGRESSION_ENDPOINTS[0]] = -9.0e-8  # looks like a raw molar IC50
    df.to_parquet(p, index=False)
    rep = vs.validate(p, "regression")
    assert rep.ok  # advisory only
    assert any("plausible pIC50 range" in w for w in rep.warnings)


def test_validator_rejects_non_binary_tdi(tmp_path):
    p = tmp_path / "c.parquet"
    df = _cls_df()
    df[vs.CLASSIFICATION_ENDPOINTS[0]] = 0.5
    df.to_parquet(p, index=False)
    rep = vs.validate(p, "classification")
    assert not rep.ok and any("non-binary values" in e for e in rep.errors)


def test_validator_accepts_0_1_ints_for_tdi(tmp_path):
    p = tmp_path / "c.parquet"
    df = _cls_df()
    for ep in vs.CLASSIFICATION_ENDPOINTS:
        df[ep] = np.tile([1, 0], 375)
    df.to_parquet(p, index=False)
    assert vs.validate(p, "classification").ok


def test_validator_strict_rejects_tdi_nan_but_lenient_only_warns(tmp_path):
    p = tmp_path / "c.parquet"
    df = _cls_df()
    df[vs.CLASSIFICATION_ENDPOINTS[0]] = df[vs.CLASSIFICATION_ENDPOINTS[0]].astype(
        "object"
    )
    df.loc[0, vs.CLASSIFICATION_ENDPOINTS[0]] = None
    df.to_parquet(p, index=False)
    strict = vs.validate(p, "classification", strict=True)
    lenient = vs.validate(p, "classification", strict=False)
    assert not strict.ok
    assert lenient.ok  # this is what the Space itself would do
    assert any("null value" in w for w in lenient.warnings)


def test_validator_warns_on_single_class_tdi(tmp_path):
    p = tmp_path / "c.parquet"
    df = _cls_df()
    for ep in vs.CLASSIFICATION_ENDPOINTS:
        df[ep] = False
    df.to_parquet(p, index=False)
    rep = vs.validate(p, "classification")
    assert rep.ok
    assert any("single class" in w for w in rep.warnings)


def test_validator_warns_on_duplicate_molecule_names(tmp_path):
    p = tmp_path / "r.parquet"
    df = _reg_df()
    df.loc[1, "Molecule_Name"] = df.loc[0, "Molecule_Name"]
    df.to_parquet(p, index=False)
    rep = vs.validate(p, "regression")
    assert rep.ok and any("duplicate" in w for w in rep.warnings)


def test_validator_rejects_bad_extension(tmp_path):
    p = tmp_path / "r.txt"
    p.write_text("nope")
    rep = vs.validate(p, "regression")
    assert not rep.ok and any(".parquet or .csv" in e for e in rep.errors)


def test_validator_rejects_missing_file(tmp_path):
    rep = vs.validate(tmp_path / "nope.parquet", "regression")
    assert not rep.ok and any("does not exist" in e for e in rep.errors)


def test_validator_rejects_unreadable_parquet(tmp_path):
    p = tmp_path / "r.parquet"
    p.write_bytes(b"not parquet at all")
    rep = vs.validate(p, "regression")
    assert not rep.ok and any("Could not read parquet" in e for e in rep.errors)


def test_validator_unknown_track_raises(tmp_path):
    with pytest.raises(ValueError, match="track must be one of"):
        vs.validate(tmp_path / "x.parquet", "activity")


def test_structure_validator_counts_184(tmp_path):
    import zipfile as zf

    good = tmp_path / "good.zip"
    with zf.ZipFile(good, "w") as z:
        for i in range(184):
            z.writestr(f"x{i:05d}-1.pdb", "ATOM\n")
    assert vs.validate(good, "structure").ok

    bad = tmp_path / "bad.zip"
    with zf.ZipFile(bad, "w") as z:
        for i in range(183):
            z.writestr(f"x{i:05d}-1.pdb", "ATOM\n")
    rep = vs.validate(bad, "structure")
    assert not rep.ok and any("Expected 184 files" in e for e in rep.errors)


def test_structure_validator_flags_directory_entry(tmp_path):
    import zipfile as zf

    p = tmp_path / "folder.zip"
    with zf.ZipFile(p, "w") as z:
        z.writestr("poses/", "")
        for i in range(184):
            z.writestr(f"poses/x{i:05d}-1.pdb", "ATOM\n")
    rep = vs.validate(p, "structure")
    assert not rep.ok
    assert any("directory entry" in w for w in rep.warnings)


def test_structure_validator_rejects_non_zip(tmp_path):
    p = tmp_path / "s.parquet"
    p.write_bytes(b"x")
    rep = vs.validate(p, "structure")
    assert not rep.ok and any("must be a .zip" in e for e in rep.errors)


def test_cli_exit_codes(tmp_path):
    good = tmp_path / "g.parquet"
    _reg_df().to_parquet(good, index=False)
    assert vs.main(["--track", "regression", "--file", str(good), "--quiet"]) == 0
    bad = tmp_path / "b.parquet"
    _reg_df(10).to_parquet(bad, index=False)
    assert vs.main(["--track", "regression", "--file", str(bad), "--quiet"]) == 1


# ==================================================================================
# integration: dummy submissions must pass, and metrics must rank predictors sanely
# ==================================================================================


def test_metric_ranks_predictors_sensibly():
    rng = np.random.default_rng(21)
    n = 300
    yt = pd.DataFrame(
        {ep: rng.uniform(3.5, 8.0, n) for ep in cm.REGRESSION_ENDPOINTS}
    )
    lo, hi = yt - 0.3, yt + 0.3
    preds = {
        "perfect": yt,
        "good": yt + rng.normal(0, 0.4, yt.shape),
        "mediocre": yt + rng.normal(0, 1.0, yt.shape),
        "mean_only": pd.DataFrame(
            {ep: np.full(n, yt[ep].mean()) for ep in yt.columns}
        ),
        "shuffled": yt.sample(frac=1.0, random_state=1).reset_index(drop=True),
    }
    scores = {
        k: cm.ma_st_rae(v, lo, hi, yt, denominator="st_mean") for k, v in preds.items()
    }
    assert scores["perfect"] < scores["good"] < scores["mediocre"]
    assert scores["mediocre"] < scores["shuffled"]
    assert scores["perfect"] == pytest.approx(0.0)
