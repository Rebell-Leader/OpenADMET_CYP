"""TDI ground-truth labeling and classification scoring for the CYP challenge.

The four-way labeling rules are quoted verbatim from the in-Space FAQ
(``config.FAQ_MD``, "How exactly are TDI positive/negative labels assigned?"):

    * **Positive** -- direct-inhibition pIC50 above 4, and the shift to the TDI arm
      exceeds 2-fold (log10(2-fold) = 0.301).
    * **Negative** -- direct-inhibition pIC50 above 4, and the shift does not exceed
      2-fold.
    * **Inferred positive** -- direct-inhibition pIC50 below 4 (too low to reliably
      measure a shift), but the TDI-arm pIC50 exceeds 4.301.
    * **Assigned negative** -- direct-inhibition pIC50 below 4 *and* TDI-arm pIC50
      below 4.

    The scored **positive class = positives + inferred positives**; the scored
    **negative class = negatives + assigned negatives**.

A GAP IN THE PUBLISHED SPEC
---------------------------
The four cases are not exhaustive. Two regions are uncovered:

1. ``direct < 4`` and ``4 <= tdi <= 4.301`` -- the TDI arm has measurable signal but
   not enough to guarantee a >2-fold shift given only ``direct < 4``.
2. ``direct == 4`` exactly -- every rule uses a strict inequality against 4.

We label both ``"unassignable"`` and EXCLUDE them from scoring, which matches the FAQ's
own statement that "only compounds whose label can be assigned with confidence
contribute to the score". This is a documented inference, not a confirmed rule, and is
open question 7 in ``metric_spec_reverse_engineered.md``. ``strict_boundaries=False``
switches every comparison to its non-strict form so the alternative reading can be
scored without editing code.

Classification metric keys mirror ``config.CLASSIFICATION_METRIC_DISPLAY_NAMES``
("Must mirror backend.config.CLASSIFICATION_METRICS"): ``MCC`` (primary), ``Accuracy``,
``Precision``, ``Recall``, ``F1``.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd

from cyp_metrics import N_BOOTSTRAP, BootstrapResult, bootstrap_ci, make_bootstrap_indices

__all__ = [
    "CLASSIFICATION_ENDPOINTS",
    "PIC50_FLOOR",
    "TDI_SHIFT_THRESHOLD",
    "TDI_INFERRED_THRESHOLD",
    "TDI_CATEGORIES",
    "CLASSIFICATION_METRICS",
    "label_tdi",
    "label_tdi_frame",
    "mcc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_counts",
    "classification_metrics",
    "bootstrap_classification",
    "macro_classification",
]

#: ``config.CLASSIFICATION_ENDPOINTS`` -- exact submission column names.
CLASSIFICATION_ENDPOINTS: tuple[str, ...] = ("CYP2D6_is_TDI", "CYP3A4_is_TDI")

#: Assay's reliable lower limit in pIC50 units.
PIC50_FLOOR: float = 4.0

#: log10(2) -- the 2-fold IC50 shift threshold, given as 0.301 in the FAQ.
TDI_SHIFT_THRESHOLD: float = 0.301

#: PIC50_FLOOR + TDI_SHIFT_THRESHOLD, given as 4.301 in the FAQ.
TDI_INFERRED_THRESHOLD: float = 4.301

#: Absolute tolerance applied to every threshold comparison.
#:
#: Necessary, not cosmetic. The shift is computed as ``tdi - direct`` in binary floating
#: point, so a compound whose arms are exactly 6.301 and 6.000 yields a shift of
#: 0.30100000000000016, which would spuriously "exceed" the 0.301 threshold and flip the
#: label from negative to positive. Real pIC50 values from a Bayesian DRC fit carry at
#: most a few decimal places, so a 1e-9 tolerance can never change a genuine call while
#: it does make every on-threshold case land deterministically on the side the published
#: wording implies ("exceeds" => a shift of exactly 0.301 is NEGATIVE).
BOUNDARY_ATOL: float = 1e-9

TDI_CATEGORIES: tuple[str, ...] = (
    "positive",
    "negative",
    "inferred_positive",
    "assigned_negative",
    "unassignable",
    # Added 2026-08-17 after the real training labels were released. See
    # ``organizer_rule`` in :func:`label_tdi`.
    "band_negative",
    "no_direct_arm_negative",
)

#: Which categories collapse to which boolean. ``unassignable`` -> excluded (NaN).
_CATEGORY_TO_BOOL: dict[str, float] = {
    "positive": 1.0,
    "inferred_positive": 1.0,
    "negative": 0.0,
    "assigned_negative": 0.0,
    "unassignable": np.nan,
    "band_negative": 0.0,
    "no_direct_arm_negative": 0.0,
}

#: Leaderboard metric keys, primary first (``config.CLASSIFICATION_METRIC_DISPLAY_NAMES``).
CLASSIFICATION_METRICS: tuple[str, ...] = (
    "MCC",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
)


def label_tdi(
    direct_pic50,
    tdi_pic50,
    *,
    strict_boundaries: bool = True,
    organizer_rule: bool = True,
    floor: float = PIC50_FLOOR,
    shift_threshold: float = TDI_SHIFT_THRESHOLD,
    inferred_threshold: float = TDI_INFERRED_THRESHOLD,
    atol: float = BOUNDARY_ATOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign the TDI category and the collapsed boolean label.

    Implements the FAQ rules exactly::

        positive           : direct >  4  and (tdi - direct) >  0.301
        negative           : direct >  4  and (tdi - direct) <= 0.301
        inferred_positive  : direct <  4  and  tdi           >  4.301
        assigned_negative  : direct <  4  and  tdi           <  4
        unassignable       : everything else (see module docstring)

    **Validated against the real labels (2026-08-17).** With
    ``organizer_rule=True`` (the default) this function reproduces the
    organizers' released ``CYP2D6_is_TDI`` / ``CYP3A4_is_TDI`` training labels
    with **100 % agreement on all 5,081 labelled rows** (1,497 CYP2D6 + 3,584
    CYP3A4, zero disagreements). Two behaviours the published FAQ left undefined
    had to be pinned down empirically, and both resolve toward *negative*:

    * The ``direct < 4`` with ``4 <= tdi <= 4.301`` band — the gap in the
      published rules — is labelled **negative**, not excluded (18/18 CYP2D6 and
      159/159 CYP3A4 such rows are ``False``). Category ``band_negative``.
    * A **missing direct arm** is labelled **negative** regardless of the TDI-arm
      value (1,249/1,249 CYP3A4 and 4/4 CYP2D6 rows are ``False``, including
      1,055 CYP3A4 rows whose TDI-arm pIC50 exceeds 4.301, up to 7.54).
      Category ``no_direct_arm_negative``.

    The second point is consequential for training and worth reading twice: 34.8 %
    of released CYP3A4 labels have no direct-inhibition DRC, and the organizers
    call those negative. Treating them instead as inferred positives would move
    the CYP3A4 positive rate from 21.3 % to 50.8 %. Since the leaderboard scores
    against *their* convention, train against it — but consider down-weighting or
    excluding those rows rather than teaching a model that measurable TDI with an
    absent direct arm means "no TDI".

    Set ``organizer_rule=False`` to recover the strict published reading, where
    both cases are ``unassignable`` and excluded from scoring.

    Note the shift is computed in pIC50 (log10) space, so a 2-fold *potency increase*
    on preincubation is a ``+0.301`` shift in pIC50 -- the blog's "leftward shift" in
    IC50 is a rightward/upward shift in pIC50. Compounds with a NaN direct or TDI value
    are ``unassignable``.

    Args:
        direct_pic50: Direct-inhibition-arm pIC50 (no-NADPH preincubation arm).
        tdi_pic50: TDI-arm pIC50 (+NADPH preincubation arm).
        strict_boundaries: If True (default) use the FAQ's strict inequalities, which
            leaves ``direct == 4`` and the ``4 <= tdi <= 4.301`` band unassignable. If
            False, comparisons against ``floor`` and the thresholds become non-strict
            (``>=`` / ``<=``) so that ``direct == 4`` is treated as measurable and the
            band collapses into ``assigned_negative``; this is the alternative reading
            to score if the organizers confirm it.
        organizer_rule: If True (default) apply the two empirically confirmed
            conventions described above — the undefined band and a missing direct
            arm both become negative — reproducing the released labels exactly. If
            False, both stay ``unassignable`` and are excluded from scoring.
        floor: pIC50 reliability floor (4.0).
        shift_threshold: Minimum pIC50 shift for a positive call (0.301).
        inferred_threshold: TDI-arm pIC50 above which a sub-floor direct value implies
            a positive (4.301).
        atol: Absolute tolerance on every threshold comparison. See
            :data:`BOUNDARY_ATOL` for why this is required rather than cosmetic.

    Returns:
        ``(categories, labels)`` where ``categories`` is a string array drawn from
        :data:`TDI_CATEGORIES` and ``labels`` is a float array of 1.0 / 0.0 / NaN
        (NaN = excluded from scoring).
    """
    d = np.asarray(direct_pic50, dtype=float)
    t = np.asarray(tdi_pic50, dtype=float)
    if d.shape != t.shape:
        raise ValueError(f"shape mismatch: direct {d.shape}, tdi {t.shape}")

    cats = np.full(d.shape, "unassignable", dtype=object)
    valid = np.isfinite(d) & np.isfinite(t)
    shift = np.where(valid, t - d, np.nan)

    # Every comparison is tolerance-aware so that on-threshold values land on the side
    # the published wording implies rather than on the side binary rounding happens to
    # put them. "> x" becomes "> x + atol"; "< x" becomes "< x - atol".
    if strict_boundaries:
        above_floor = valid & (d > floor + atol)
        below_floor = valid & (d < floor - atol)
        big_shift = shift > shift_threshold + atol
        small_shift = shift <= shift_threshold + atol
        tdi_clears = t > inferred_threshold + atol
        tdi_dead = t < floor - atol
    else:
        above_floor = valid & (d >= floor - atol)
        below_floor = valid & (d < floor - atol)
        big_shift = shift >= shift_threshold - atol
        small_shift = shift < shift_threshold - atol
        tdi_clears = t >= inferred_threshold - atol
        tdi_dead = t < inferred_threshold - atol

    cats[above_floor & big_shift] = "positive"
    cats[above_floor & small_shift] = "negative"
    cats[below_floor & tdi_clears] = "inferred_positive"
    cats[below_floor & tdi_dead & ~tdi_clears] = "assigned_negative"

    if organizer_rule:
        # The undefined band (direct < 4, floor <= tdi <= inferred_threshold) is
        # negative in the released labels, not excluded.
        cats[below_floor & ~tdi_clears & ~tdi_dead] = "band_negative"
        # A missing direct arm is negative regardless of the TDI-arm value. Note
        # this fires on a NaN/absent direct measurement, NOT on a sub-floor one.
        cats[~np.isfinite(d)] = "no_direct_arm_negative"

    labels = np.array([_CATEGORY_TO_BOOL[c] for c in cats.ravel()], dtype=float).reshape(
        d.shape
    )
    return cats.astype(str), labels


def label_tdi_frame(
    df: pd.DataFrame,
    isoform: str,
    *,
    direct_col: str | None = None,
    tdi_col: str | None = None,
    strict_boundaries: bool = True,
) -> pd.DataFrame:
    """Label one isoform's TDI ground truth from a two-arm DataFrame.

    Args:
        df: Frame carrying both arms' pIC50 for ``isoform``.
        isoform: e.g. ``"CYP3A4"``. Used to build the default column names
            ``{isoform}_pIC50_direct_inhibition`` (CONFIRMED name) and
            ``{isoform}_pIC50_tdi`` (GUESSED -- the TDI-arm ground-truth column name has
            not been published; pass ``tdi_col`` explicitly once it is known).
        direct_col: Override for the direct-arm column.
        tdi_col: Override for the TDI-arm column.
        strict_boundaries: See :func:`label_tdi`.

    Returns:
        Copy of ``df`` with added columns ``{isoform}_tdi_category``,
        ``{isoform}_is_TDI_label`` (float 1/0/NaN) and ``{isoform}_tdi_shift``.
    """
    dcol = direct_col or f"{isoform}_pIC50_direct_inhibition"
    tcol = tdi_col or f"{isoform}_pIC50_tdi"
    for c in (dcol, tcol):
        if c not in df.columns:
            raise KeyError(f"column {c!r} not in frame; columns={list(df.columns)[:12]}")
    cats, labels = label_tdi(
        df[dcol].to_numpy(), df[tcol].to_numpy(), strict_boundaries=strict_boundaries
    )
    out = df.copy()
    out[f"{isoform}_tdi_category"] = cats
    out[f"{isoform}_is_TDI_label"] = labels
    out[f"{isoform}_tdi_shift"] = df[tcol].to_numpy() - df[dcol].to_numpy()
    return out


# --- classification metrics -------------------------------------------------------


def _binarise(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Align and clean a (label, prediction) pair.

    Excludes any position where the ground-truth label is NaN (unassignable compounds)
    or the prediction is NaN. Accepts bools, 0/1 ints/floats and pandas nullable
    booleans; anything else raises.
    """
    yt = np.asarray(pd.Series(y_true).astype("object").to_numpy(), dtype=object)
    yp = np.asarray(pd.Series(y_pred).astype("object").to_numpy(), dtype=object)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape}, y_pred {yp.shape}")

    def coerce(a: np.ndarray) -> np.ndarray:
        out = np.full(a.shape, np.nan, dtype=float)
        for i, v in enumerate(a):
            if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
                continue
            if isinstance(v, (bool, np.bool_)):
                out[i] = float(bool(v))
            elif isinstance(v, (int, float, np.integer, np.floating)):
                fv = float(v)
                if fv not in (0.0, 1.0):
                    raise ValueError(f"non-binary value {v!r} at position {i}")
                out[i] = fv
            else:
                raise ValueError(f"non-boolean value {v!r} at position {i}")
        return out

    ct, cp = coerce(yt), coerce(yp)
    m = np.isfinite(ct) & np.isfinite(cp)
    return ct[m].astype(int), cp[m].astype(int)


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """TP/TN/FP/FN plus the number of scored compounds."""
    yt, yp = _binarise(y_true, y_pred)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "n": int(yt.size)}


def mcc(y_true, y_pred, *, undefined: float = 0.0) -> float:
    """Matthews correlation coefficient.

    The denominator ``sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))`` is zero whenever the
    predictions or the labels are all one class. scikit-learn's convention -- and the
    convention we adopt -- is to return 0.0 there, i.e. "no better than chance".

    That behaviour is OUR CHOICE, not a confirmed backend rule (open question 8): a
    single-class submission could equally be scored NaN or rejected. Pass
    ``undefined=float("nan")`` to see the degenerate cases explicitly.

    Returns ``nan`` when no compound is scorable at all.
    """
    yt, yp = _binarise(y_true, y_pred)
    if yt.size == 0:
        return np.nan
    c = confusion_counts(yt, yp)
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
    num = tp * tn - fp * fn
    den = np.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    if den <= 0.0:
        return undefined
    return float(num / den)


def accuracy(y_true, y_pred) -> float:
    """(TP+TN)/n. NaN if nothing is scorable."""
    yt, yp = _binarise(y_true, y_pred)
    return float(np.mean(yt == yp)) if yt.size else np.nan


def precision(y_true, y_pred, *, undefined: float = 0.0) -> float:
    """TP/(TP+FP). ``undefined`` (default 0.0) when nothing is predicted positive."""
    c = confusion_counts(y_true, y_pred)
    if c["n"] == 0:
        return np.nan
    den = c["TP"] + c["FP"]
    return undefined if den == 0 else float(c["TP"] / den)


def recall(y_true, y_pred, *, undefined: float = 0.0) -> float:
    """TP/(TP+FN). ``undefined`` (default 0.0) when there is no true positive."""
    c = confusion_counts(y_true, y_pred)
    if c["n"] == 0:
        return np.nan
    den = c["TP"] + c["FN"]
    return undefined if den == 0 else float(c["TP"] / den)


def f1(y_true, y_pred, *, undefined: float = 0.0) -> float:
    """Harmonic mean of precision and recall; ``undefined`` when both are 0."""
    c = confusion_counts(y_true, y_pred)
    if c["n"] == 0:
        return np.nan
    p = precision(y_true, y_pred, undefined=0.0)
    r = recall(y_true, y_pred, undefined=0.0)
    if p + r <= 0:
        return undefined
    return float(2 * p * r / (p + r))


def classification_metrics(
    y_true, y_pred, *, undefined: float = 0.0
) -> dict[str, float]:
    """Full leaderboard metric row for one TDI endpoint (MCC first)."""
    return {
        "MCC": mcc(y_true, y_pred, undefined=undefined),
        "Accuracy": accuracy(y_true, y_pred),
        "Precision": precision(y_true, y_pred, undefined=undefined),
        "Recall": recall(y_true, y_pred, undefined=undefined),
        "F1": f1(y_true, y_pred, undefined=undefined),
    }


def bootstrap_classification(
    y_true,
    y_pred,
    *,
    n: int = N_BOOTSTRAP,
    seed: int = 0,
    alpha: float = 0.05,
    indices: np.ndarray | None = None,
    undefined: float = 0.0,
) -> dict[str, BootstrapResult]:
    """1000-sample compound-level bootstrap for every classification metric.

    Resamples COMPOUNDS with replacement over the full supplied vector (including
    unassignable compounds, which are then dropped inside each metric) so that the
    resample matrix can be shared with the regression track.
    """
    yt = pd.Series(y_true).to_numpy()
    yp = pd.Series(y_pred).to_numpy()
    n_compounds = len(yt)
    idx = (
        make_bootstrap_indices(n_compounds, n, seed) if indices is None else np.asarray(indices)
    )
    out: dict[str, BootstrapResult] = {}
    for name, fn in (
        ("MCC", lambda a, b: mcc(a, b, undefined=undefined)),
        ("Accuracy", accuracy),
        ("Precision", lambda a, b: precision(a, b, undefined=undefined)),
        ("Recall", lambda a, b: recall(a, b, undefined=undefined)),
        ("F1", lambda a, b: f1(a, b, undefined=undefined)),
    ):
        out[name] = bootstrap_ci(
            lambda sel, _fn=fn: _fn(yt[sel], yp[sel]),
            n_compounds,
            metric=name,
            n=n,
            seed=seed,
            alpha=alpha,
            indices=idx,
        )
    return out


def macro_classification(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    *,
    endpoints: Sequence[str] = CLASSIFICATION_ENDPOINTS,
    undefined: float = 0.0,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Per-endpoint and macro-averaged ("MA") classification metrics.

    The classification track has 2 endpoints, so ``len(endpoints) > 1`` and the Space
    renders an "Overall" tab with ``MA-`` prefixed columns
    (``leaderboards._add_activity_track_tabs``).

    Returns:
        ``(macro_row, per_endpoint_rows)``.
    """
    per = {
        ep: classification_metrics(y_true[ep], y_pred[ep], undefined=undefined)
        for ep in endpoints
    }
    macro = {
        k: float(np.nanmean([per[ep][k] for ep in endpoints]))
        for k in CLASSIFICATION_METRICS
    }
    return macro, per
