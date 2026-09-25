"""Regression metrics for the OpenADMET CYP Inhibition Blind Challenge.

Implements Soft-Threshold Relative Absolute Error (ST-RAE), its macro-average over
the four direct-inhibition endpoints (MA-ST-RAE), the secondary metric battery
(RAE, MAE, R2, Spearman rho, Kendall tau) and a paired compound-level bootstrap.

Provenance of every definition
------------------------------
CONFIRMED (announcement blog + in-Space FAQ text in ``config.py``):
    "Error is measured as the distance between your prediction and the nearest
    credible interval bound of the fitted dose-response curve. Predictions falling
    anywhere inside the credible interval incur zero error penalty."
    => the NUMERATOR of ST-RAE is pinned down exactly. See ``soft_threshold_ae``.

CONFIRMED (``config.ACTIVITY_METRIC_DISPLAY_NAMES``, which the source annotates as
"Must mirror backend.config.ACTIVITY_METRICS"): the metric keys as they appear in the
leaderboard CSV are exactly ``RAE``, ``MAE``, ``R2``, ``Spearman_R``, ``Kendall_Tau``,
with ``RAE`` the primary/sort metric and lower being better
(``head_to_head.LOWER_IS_BETTER = ["RAE"]``). Note there is no ``ST_RAE`` identifier
anywhere in the Space source -- the code calls the primary metric ``RAE``.

CONFIRMED (``config.MACRO_ENDPOINT_LABEL`` and ``head_to_head._load_bootstrap``):
``bootstrap-results.parquet`` holds one row per ``(Sample, Endpoint)`` pair, with the
synthetic endpoint ``"MA"`` holding "the multi-endpoint macro-averaged scores for that
bootstrap sample". => the macro-average is formed WITHIN each bootstrap sample.

CONFIRMED (``config.FAQ_MD``): "All metrics are bootstrapped over 1,000 resamples."

CONFIRMED (``leaderboards._prepare_activity_df`` consumes ``{metric}_mean`` and
``{metric}_std``): the leaderboard reports the bootstrap mean and SD, not percentile
intervals. We return both so either can be quoted.

INFERRED, and deliberately swappable:
    The RAE *denominator* -- the "trivial predictor" the absolute error is normalised
    against -- is not published. ``DENOMINATOR_POLICIES`` enumerates five choices;
    the default is ``"st_mean"`` (the soft-thresholded error of the mean-of-y_true
    predictor). Change ``denominator=`` in one place to re-fit the whole harness when
    the organizers publish theirs.

INFERRED: bootstrap resample indices are fixed and shared across submissions -- the
Space merges two participants' bootstrap frames on ``Sample`` and counts wins, which
is only valid for paired resamples. ``make_bootstrap_indices`` produces such a matrix
from a fixed seed so predictors can be compared pairwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "REGRESSION_ENDPOINTS",
    "MACRO_ENDPOINT_LABEL",
    "N_BOOTSTRAP",
    "DENOMINATOR_POLICIES",
    "soft_threshold_ae",
    "st_rae",
    "ma_st_rae",
    "rae",
    "mae",
    "r2",
    "spearman_r",
    "kendall_tau",
    "REGRESSION_METRICS",
    "secondary_metrics",
    "endpoint_metrics",
    "make_bootstrap_indices",
    "bootstrap_ci",
    "bootstrap_results_frame",
    "averaged_results_frame",
    "BootstrapResult",
]

# --- CONFIRMED constants, mirrored verbatim from the Space's config.py -------------

#: ``config.REGRESSION_ENDPOINTS`` -- also the exact submission column names.
REGRESSION_ENDPOINTS: tuple[str, ...] = (
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
)

#: ``config.MACRO_ENDPOINT_LABEL`` -- the synthetic macro-average pseudo-endpoint.
MACRO_ENDPOINT_LABEL: str = "MA"

#: ``config.FAQ_MD``: "All metrics are bootstrapped over 1,000 resamples."
N_BOOTSTRAP: int = 1000

#: The assay's reliable lower limit, in pIC50 units (blog post + FAQ).
PIC50_FLOOR: float = 4.0

DenominatorPolicy = Literal["st_mean", "st_median", "mae_mean", "mae_median", "none"]

#: Documented, swappable choices for the RAE normalisation denominator (INFERRED).
DENOMINATOR_POLICIES: dict[str, str] = {
    "st_mean": "sum of soft-threshold AE of the constant mean(y_true) predictor "
    "(default; soft-thresholding applied to the baseline too)",
    "st_median": "sum of soft-threshold AE of the constant median(y_true) predictor",
    "mae_mean": "sum |y_true - mean(y_true)| (classical RAE denominator, not "
    "soft-thresholded)",
    "mae_median": "sum |y_true - median(y_true)| (robust classical RAE denominator)",
    "none": "denominator fixed at 1.0, i.e. ST-RAE degenerates to mean soft-threshold "
    "absolute error",
}


# --- numerator: the one part of ST-RAE that is unambiguous ------------------------


def soft_threshold_ae(
    y_pred: np.ndarray | Sequence[float],
    ci_lower: np.ndarray | Sequence[float],
    ci_upper: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Per-compound soft-threshold absolute error.

    Zero if the prediction lies inside the ground-truth credible interval, otherwise
    the distance to the nearest bound::

        st_ae = max(ci_lower - y_pred, 0, y_pred - ci_upper)

    This is the definition given verbatim in the announcement and the in-Space FAQ.
    Bounds are normalised so that ``ci_lower <= ci_upper`` even if supplied swapped.

    Args:
        y_pred: Predicted pIC50 values.
        ci_lower: Lower credible-interval bound of the ground-truth DRC fit.
        ci_upper: Upper credible-interval bound.

    Returns:
        Array of non-negative errors, same shape as ``y_pred``. ``NaN`` propagates.
    """
    yp = np.asarray(y_pred, dtype=float)
    lo = np.asarray(ci_lower, dtype=float)
    hi = np.asarray(ci_upper, dtype=float)
    if not (yp.shape == lo.shape == hi.shape):
        raise ValueError(
            f"shape mismatch: y_pred {yp.shape}, ci_lower {lo.shape}, "
            f"ci_upper {hi.shape}"
        )
    real_lo = np.minimum(lo, hi)
    real_hi = np.maximum(lo, hi)
    below = real_lo - yp
    above = yp - real_hi
    out = np.maximum(np.maximum(below, above), 0.0)
    # keep NaN where any input was NaN (np.maximum already propagates, but be explicit)
    out = np.where(np.isnan(yp) | np.isnan(real_lo) | np.isnan(real_hi), np.nan, out)
    return out


def _clean_mask(*arrays: np.ndarray) -> np.ndarray:
    """Boolean mask of positions finite in every supplied array."""
    mask = np.ones(np.asarray(arrays[0]).shape, dtype=bool)
    for a in arrays:
        arr = np.asarray(a, dtype=float)
        mask &= np.isfinite(arr)
    return mask


def _denominator(
    y_true: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    policy: DenominatorPolicy,
) -> float:
    """Sum-of-errors of the trivial predictor, per the chosen (INFERRED) policy."""
    if policy == "none":
        return float(len(y_true))  # so that ratio == mean st_ae
    if policy in ("st_mean", "mae_mean"):
        baseline = float(np.mean(y_true))
    elif policy in ("st_median", "mae_median"):
        baseline = float(np.median(y_true))
    else:
        raise ValueError(
            f"unknown denominator policy {policy!r}; "
            f"choose from {sorted(DENOMINATOR_POLICIES)}"
        )
    const = np.full_like(y_true, baseline, dtype=float)
    if policy.startswith("st_"):
        return float(np.sum(soft_threshold_ae(const, ci_lower, ci_upper)))
    return float(np.sum(np.abs(y_true - const)))


@dataclass(frozen=True)
class STRAEDetail:
    """Diagnostics for one ST-RAE evaluation."""

    value: float
    numerator: float
    denominator: float
    n_scored: int
    n_excluded: int
    n_inside_interval: int
    denominator_policy: str
    degenerate_denominator: bool


def st_rae(
    y_pred,
    ci_lower,
    ci_upper,
    y_true=None,
    *,
    denominator: DenominatorPolicy = "st_mean",
    return_detail: bool = False,
    denominator_floor: float = 1e-12,
):
    """Soft-Threshold Relative Absolute Error for one endpoint.

    ``ST-RAE = sum(st_ae(y_pred)) / sum(error of a trivial predictor)``, lower is
    better, 0 means every prediction landed inside its credible interval.

    The numerator is CONFIRMED by the challenge text. The denominator is INFERRED --
    see :data:`DENOMINATOR_POLICIES`. With ``denominator="none"`` the function returns
    the plain mean soft-threshold absolute error, which needs no assumptions at all.

    Compounds are excluded when any of ``y_pred``, ``ci_lower``, ``ci_upper`` (and
    ``y_true`` when the policy needs it) is NaN or infinite -- i.e. NaN ground truth is
    handled by exclusion, per endpoint.

    Args:
        y_pred: Predicted pIC50 values.
        ci_lower: Ground-truth credible-interval lower bounds.
        ci_upper: Ground-truth credible-interval upper bounds.
        y_true: Ground-truth pIC50 point estimates. Required for every denominator
            policy except ``"none"``; ignored by ``"none"``.
        denominator: Which trivial-predictor normalisation to use.
        return_detail: If True return an :class:`STRAEDetail` instead of a bare float.
        denominator_floor: Denominators at or below this are treated as degenerate;
            the result is then ``nan`` (flagged in the detail) rather than ``inf``,
            because a near-zero denominator means the trivial predictor already sat
            inside almost every credible interval and the ratio carries no signal.

    Returns:
        float ST-RAE, or :class:`STRAEDetail` when ``return_detail=True``. Returns
        ``nan`` when no compound survives filtering.
    """
    if denominator not in DENOMINATOR_POLICIES:
        raise ValueError(
            f"unknown denominator policy {denominator!r}; "
            f"choose from {sorted(DENOMINATOR_POLICIES)}"
        )
    yp = np.asarray(y_pred, dtype=float)
    lo = np.asarray(ci_lower, dtype=float)
    hi = np.asarray(ci_upper, dtype=float)
    needs_true = denominator != "none"
    if needs_true:
        if y_true is None:
            raise ValueError(
                f"denominator={denominator!r} needs y_true; pass y_true or use "
                "denominator='none'"
            )
        yt = np.asarray(y_true, dtype=float)
        mask = _clean_mask(yp, lo, hi, yt)
    else:
        yt = np.full_like(yp, np.nan)
        mask = _clean_mask(yp, lo, hi)

    n_total = int(mask.size)
    n_scored = int(mask.sum())
    if n_scored == 0:
        detail = STRAEDetail(
            np.nan, np.nan, np.nan, 0, n_total, 0, denominator, False
        )
        return detail if return_detail else np.nan

    errs = soft_threshold_ae(yp[mask], lo[mask], hi[mask])
    num = float(np.sum(errs))
    den = _denominator(yt[mask], lo[mask], hi[mask], denominator)
    degenerate = den <= denominator_floor
    value = np.nan if degenerate else num / den
    if not return_detail:
        return value
    return STRAEDetail(
        value=value,
        numerator=num,
        denominator=den,
        n_scored=n_scored,
        n_excluded=n_total - n_scored,
        n_inside_interval=int(np.sum(errs == 0.0)),
        denominator_policy=denominator,
        degenerate_denominator=degenerate,
    )


def ma_st_rae(
    y_pred: Mapping[str, Sequence[float]] | pd.DataFrame,
    ci_lower: Mapping[str, Sequence[float]] | pd.DataFrame,
    ci_upper: Mapping[str, Sequence[float]] | pd.DataFrame,
    y_true: Mapping[str, Sequence[float]] | pd.DataFrame | None = None,
    *,
    endpoints: Sequence[str] = REGRESSION_ENDPOINTS,
    denominator: DenominatorPolicy = "st_mean",
    return_per_endpoint: bool = False,
    skip_nan_endpoints: bool = True,
):
    """Macro-averaged ST-RAE: the unweighted mean of per-endpoint ST-RAE.

    The Space stores this as the synthetic ``"MA"`` pseudo-endpoint of
    ``bootstrap-results.parquet``, one value per bootstrap sample -- so the macro-
    average is taken across endpoints FIRST, within a sample, and only then averaged
    over samples. :func:`bootstrap_results_frame` reproduces that ordering.

    Args:
        y_pred: Per-endpoint predictions (dict or DataFrame keyed by endpoint name).
        ci_lower: Per-endpoint ground-truth CI lower bounds.
        ci_upper: Per-endpoint ground-truth CI upper bounds.
        y_true: Per-endpoint ground-truth point estimates.
        endpoints: Endpoints to macro-average over. Defaults to the four confirmed
            regression endpoints.
        denominator: Passed through to :func:`st_rae`.
        return_per_endpoint: If True, also return the per-endpoint dict.
        skip_nan_endpoints: If True, endpoints whose ST-RAE is NaN (no scorable
            compound, or a degenerate denominator) are dropped from the mean rather
            than poisoning it. If False the macro-average is NaN whenever any
            endpoint is NaN.

    Returns:
        float, or ``(float, dict[str, float])`` when ``return_per_endpoint=True``.
    """
    per: dict[str, float] = {}
    for ep in endpoints:
        yt = None if y_true is None else y_true[ep]
        per[ep] = st_rae(
            y_pred[ep], ci_lower[ep], ci_upper[ep], yt, denominator=denominator
        )
    vals = np.array(list(per.values()), dtype=float)
    if skip_nan_endpoints:
        good = vals[np.isfinite(vals)]
        macro = float(np.mean(good)) if good.size else np.nan
    else:
        macro = float(np.mean(vals)) if np.all(np.isfinite(vals)) else np.nan
    return (macro, per) if return_per_endpoint else macro


# --- secondary battery: the leaderboard's MAE / R2 / Spearman_R / Kendall_Tau -----
# INFERRED: these are computed against the DRC point estimate, not the interval --
# an interval-aware R2 is not a standard object and the Space gives no hint of one.


def _pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape}, y_pred {yp.shape}")
    m = _clean_mask(yt, yp)
    return yt[m], yp[m]


def mae(y_true, y_pred) -> float:
    """Mean absolute error, NaN/inf pairs excluded."""
    yt, yp = _pair(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp))) if yt.size else np.nan


def rae(y_true, y_pred, *, baseline: Literal["mean", "median"] = "mean") -> float:
    """Classical relative absolute error: ``sum|y-yhat| / sum|y-baseline(y)|``."""
    yt, yp = _pair(y_true, y_pred)
    if yt.size == 0:
        return np.nan
    b = float(np.mean(yt)) if baseline == "mean" else float(np.median(yt))
    den = float(np.sum(np.abs(yt - b)))
    return np.nan if den <= 1e-12 else float(np.sum(np.abs(yt - yp))) / den


def r2(y_true, y_pred) -> float:
    """Coefficient of determination ``1 - SS_res/SS_tot`` (not squared correlation)."""
    yt, yp = _pair(y_true, y_pred)
    if yt.size < 2:
        return np.nan
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 1e-12:
        return np.nan
    return 1.0 - float(np.sum((yt - yp) ** 2)) / ss_tot


def spearman_r(y_true, y_pred) -> float:
    """Spearman rank correlation. NaN if either side is constant."""
    yt, yp = _pair(y_true, y_pred)
    if yt.size < 3 or np.ptp(yt) == 0 or np.ptp(yp) == 0:
        return np.nan
    return float(stats.spearmanr(yt, yp).statistic)


def kendall_tau(y_true, y_pred) -> float:
    """Kendall's tau-b rank correlation. NaN if either side is constant."""
    yt, yp = _pair(y_true, y_pred)
    if yt.size < 3 or np.ptp(yt) == 0 or np.ptp(yp) == 0:
        return np.nan
    return float(stats.kendalltau(yt, yp).statistic)


#: Leaderboard metric keys -> callables. Keys and order mirror
#: ``config.ACTIVITY_METRIC_DISPLAY_NAMES``; ``RAE`` is the primary/sort metric and is
#: computed as ST-RAE here (see metric_spec_reverse_engineered.md open question 1).
REGRESSION_METRICS: tuple[str, ...] = ("RAE", "MAE", "R2", "Spearman_R", "Kendall_Tau")

#: True where a larger value is better. ``head_to_head.LOWER_IS_BETTER = ["RAE"]``.
HIGHER_IS_BETTER: dict[str, bool] = {
    "RAE": False,
    "MAE": False,
    "R2": True,
    "Spearman_R": True,
    "Kendall_Tau": True,
}


def secondary_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, R2, Spearman_R, Kendall_Tau for one endpoint (point-estimate based)."""
    return {
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "Spearman_R": spearman_r(y_true, y_pred),
        "Kendall_Tau": kendall_tau(y_true, y_pred),
    }


def endpoint_metrics(
    y_true,
    y_pred,
    ci_lower,
    ci_upper,
    *,
    denominator: DenominatorPolicy = "st_mean",
) -> dict[str, float]:
    """Full leaderboard metric row for one endpoint: RAE(=ST-RAE) plus secondaries."""
    row = {
        "RAE": st_rae(
            y_pred, ci_lower, ci_upper, y_true, denominator=denominator
        )
    }
    row.update(secondary_metrics(y_true, y_pred))
    return row


# --- bootstrap ---------------------------------------------------------------------


def make_bootstrap_indices(
    n_compounds: int, n_bootstrap: int = N_BOOTSTRAP, seed: int = 0
) -> np.ndarray:
    """Fixed compound-resample index matrix, shape ``(n_bootstrap, n_compounds)``.

    Resampling is over COMPOUNDS with replacement, never over endpoints -- the "MA"
    pseudo-endpoint proves endpoints are aggregated deterministically within a sample.

    Generating the matrix once and reusing it for every predictor reproduces the
    Space's paired comparison, which merges two participants' bootstrap frames on the
    ``Sample`` column (``head_to_head.build_delta_plot``); that is only valid if
    ``Sample = s`` means the same compound draw for both.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_compounds, size=(n_bootstrap, n_compounds))


@dataclass
class BootstrapResult:
    """Bootstrap summary for one metric."""

    metric: str
    point: float
    mean: float
    std: float
    ci_low: float
    ci_high: float
    alpha: float
    n_bootstrap: int
    n_valid: int
    samples: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def as_row(self) -> dict[str, float]:
        """Leaderboard-shaped ``{metric}_mean`` / ``{metric}_std`` pair plus CIs."""
        return {
            f"{self.metric}_mean": self.mean,
            f"{self.metric}_std": self.std,
            f"{self.metric}_point": self.point,
            f"{self.metric}_ci_low": self.ci_low,
            f"{self.metric}_ci_high": self.ci_high,
        }


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray], float],
    n_compounds: int,
    *,
    metric: str = "metric",
    n: int = N_BOOTSTRAP,
    seed: int = 0,
    alpha: float = 0.05,
    indices: np.ndarray | None = None,
) -> BootstrapResult:
    """Percentile bootstrap over compounds for an arbitrary metric.

    Args:
        metric_fn: Callable taking an integer index array (the resampled compound
            positions) and returning a scalar. Called once with ``arange(n_compounds)``
            for the point estimate, then once per resample.
        n_compounds: Number of compounds in the evaluation set.
        metric: Name used in the returned :class:`BootstrapResult`.
        n: Number of resamples. Defaults to the challenge's 1000.
        seed: Seed for :func:`make_bootstrap_indices`; ignored when ``indices`` given.
        alpha: Two-sided level; the CI is the ``[alpha/2, 1-alpha/2]`` percentile pair.
        indices: Pre-built ``(n, n_compounds)`` resample matrix, for exact pairing
            across predictors.

    Returns:
        :class:`BootstrapResult` carrying the full-data point estimate, the bootstrap
        mean and SD (what the leaderboard displays) and the percentile CI (what we use
        for our own model selection). NaN resample values are dropped from all
        summaries and counted in ``n_valid``.
    """
    idx = make_bootstrap_indices(n_compounds, n, seed) if indices is None else np.asarray(indices)
    if idx.ndim != 2 or idx.shape[1] != n_compounds:
        raise ValueError(
            f"indices must have shape (n_bootstrap, {n_compounds}), got {idx.shape}"
        )
    point = float(metric_fn(np.arange(n_compounds)))
    samples = np.array([metric_fn(row) for row in idx], dtype=float)
    good = samples[np.isfinite(samples)]
    if good.size == 0:
        return BootstrapResult(
            metric, point, np.nan, np.nan, np.nan, np.nan, alpha, idx.shape[0], 0, samples
        )
    return BootstrapResult(
        metric=metric,
        point=point,
        mean=float(np.mean(good)),
        std=float(np.std(good, ddof=1)) if good.size > 1 else 0.0,
        ci_low=float(np.percentile(good, 100 * alpha / 2)),
        ci_high=float(np.percentile(good, 100 * (1 - alpha / 2))),
        alpha=alpha,
        n_bootstrap=idx.shape[0],
        n_valid=int(good.size),
        samples=samples,
    )


def bootstrap_results_frame(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    ci_lower: pd.DataFrame,
    ci_upper: pd.DataFrame,
    *,
    endpoints: Sequence[str] = REGRESSION_ENDPOINTS,
    n: int = N_BOOTSTRAP,
    seed: int = 0,
    denominator: DenominatorPolicy = "st_mean",
    indices: np.ndarray | None = None,
) -> pd.DataFrame:
    """Reproduce the backend's ``bootstrap-results.parquet`` layout.

    One row per ``(Sample, Endpoint)`` pair, where ``Endpoint`` takes each of
    ``endpoints`` plus the synthetic ``"MA"`` macro-average -- exactly as documented in
    ``config.MACRO_ENDPOINT_LABEL`` and ``head_to_head._load_bootstrap``. The ``MA``
    row for a sample is the unweighted mean over that sample's endpoint rows, so the
    macro-average is formed INSIDE the sample.

    Columns: ``Sample``, ``Endpoint``, then ``RAE``, ``MAE``, ``R2``, ``Spearman_R``,
    ``Kendall_Tau``.

    Returns:
        DataFrame with ``n * (len(endpoints) + 1)`` rows.
    """
    n_compounds = len(y_true)
    idx = make_bootstrap_indices(n_compounds, n, seed) if indices is None else np.asarray(indices)
    arrays = {
        ep: (
            np.asarray(y_true[ep], dtype=float),
            np.asarray(y_pred[ep], dtype=float),
            np.asarray(ci_lower[ep], dtype=float),
            np.asarray(ci_upper[ep], dtype=float),
        )
        for ep in endpoints
    }
    rows: list[dict] = []
    for s, sel in enumerate(idx):
        per_ep: list[dict[str, float]] = []
        for ep in endpoints:
            yt, yp, lo, hi = arrays[ep]
            m = endpoint_metrics(
                yt[sel], yp[sel], lo[sel], hi[sel], denominator=denominator
            )
            per_ep.append(m)
            rows.append({"Sample": s, "Endpoint": ep, **m})
        # np.nanmean over an all-NaN slice warns and returns NaN; guard explicitly so a
        # metric that is undefined for every endpoint (e.g. Spearman for a constant
        # predictor) yields a clean NaN rather than a RuntimeWarning.
        macro = {}
        for k in REGRESSION_METRICS:
            vals = np.array([d[k] for d in per_ep], dtype=float)
            good = vals[np.isfinite(vals)]
            macro[k] = float(np.mean(good)) if good.size else np.nan
        rows.append({"Sample": s, "Endpoint": MACRO_ENDPOINT_LABEL, **macro})
    return pd.DataFrame(rows, columns=["Sample", "Endpoint", *REGRESSION_METRICS])


def averaged_results_frame(
    bootstrap_df: pd.DataFrame, *, alpha: float = 0.05
) -> pd.DataFrame:
    """Collapse a bootstrap frame to the leaderboard's ``{metric}_mean/_std`` schema.

    Mirrors ``leaderboards._prepare_activity_df``, which reads ``{metric}_mean`` and
    ``{metric}_std`` from the leaderboard CSV. Percentile CI columns are added for our
    own use -- the Space does not display them.
    """
    out = []
    for ep, grp in bootstrap_df.groupby("Endpoint", sort=False):
        row: dict[str, object] = {"Endpoint": ep, "n_bootstrap": len(grp)}
        for metric in REGRESSION_METRICS:
            vals = grp[metric].to_numpy(dtype=float)
            good = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(np.mean(good)) if good.size else np.nan
            row[f"{metric}_std"] = (
                float(np.std(good, ddof=1)) if good.size > 1 else np.nan
            )
            row[f"{metric}_ci_low"] = (
                float(np.percentile(good, 100 * alpha / 2)) if good.size else np.nan
            )
            row[f"{metric}_ci_high"] = (
                float(np.percentile(good, 100 * (1 - alpha / 2)))
                if good.size
                else np.nan
            )
        out.append(row)
    return pd.DataFrame(out)


def paired_delta(
    boot_a: pd.DataFrame,
    boot_b: pd.DataFrame,
    *,
    metric: str = "RAE",
    endpoint: str = MACRO_ENDPOINT_LABEL,
) -> dict[str, float]:
    """Paired bootstrap comparison of two predictors, as the Space's head-to-head does.

    Merges on ``Sample`` (so both must come from the same resample matrix), narrows to
    ``endpoint``, and reports the win fraction and a two-sided bootstrap p-value.

    Returns:
        dict with ``n``, ``mean_delta``, ``a_wins``, ``b_wins``, ``p_two_sided``.
    """
    a = boot_a[boot_a["Endpoint"] == endpoint][["Sample", metric]]
    b = boot_b[boot_b["Endpoint"] == endpoint][["Sample", metric]]
    merged = a.merge(b, on="Sample", suffixes=("_a", "_b"))
    delta = (merged[f"{metric}_a"] - merged[f"{metric}_b"]).to_numpy(dtype=float)
    delta = delta[np.isfinite(delta)]
    n = int(delta.size)
    if n == 0:
        return {
            "n": 0,
            "mean_delta": np.nan,
            "a_wins": 0,
            "b_wins": 0,
            "p_two_sided": np.nan,
        }
    lower_better = not HIGHER_IS_BETTER.get(metric, True)
    a_wins = int((delta < 0).sum() if lower_better else (delta > 0).sum())
    b_wins = int((delta > 0).sum() if lower_better else (delta < 0).sum())
    # two-sided bootstrap p: fraction of resamples on the wrong side of 0, doubled
    frac = min((delta <= 0).mean(), (delta >= 0).mean())
    return {
        "n": n,
        "mean_delta": float(np.mean(delta)),
        "a_wins": a_wins,
        "b_wins": b_wins,
        "p_two_sided": float(min(1.0, 2 * frac)),
    }
