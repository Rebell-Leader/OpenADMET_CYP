"""Analog-series-aware cross-validation splits for the OpenADMET CYP challenge.

Why this module exists
----------------------
The challenge test set is not a random sample of chemical space. It was built by
taking the 25 most potent hits per isoform for CYP1A2 / CYP2C9 / CYP3A4 (75
parent compounds) and purchasing the **top 10 ECFP4-Tanimoto analogs** of each
from the Enamine US in-stock catalog, giving 750 compounds assayed densely
against all four isoforms. The live/blind leaderboard split is then made **by
chemisimilar series**, so every analog of a given parent lands on the same side.

Two consequences drive every function here:

1. A random K-fold split leaks analogs of a test compound into training and
   reports a score the leaderboard will not reproduce. Measured on the curated
   public corpus, random splitting is optimistic by **0.039–0.051 RAE** relative
   to a scaffold-grouped split, consistently across all four isoforms (see
   ``split_regime_benchmark.csv``).
2. The challenge condition is *not* "hold out a whole novel series". The parent
   hits come from the training DRC set, so at test time the model has seen a
   potent member of each series and is asked to interpolate within it. That is a
   materially different — and easier — task than extrapolating to an unseen
   series, and it is reproduced here by :func:`parent_in_train_folds`.

Use :func:`parent_in_train_folds` as the primary model-selection protocol and
:func:`series_held_out_folds` as the pessimistic bound.

All functions take precomputed ECFP4 bit vectors as a dense ``float32`` matrix
so the Tanimoto work can be done in a few chunked matrix products; nothing here
depends on the challenge data being present.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np


# ECFP4 Tanimoto radius used to define an analog series. 0.55 is deliberately
# looser than the ~0.7 a medicinal chemist would call "same series": the goal is
# to catch every compound whose presence in training would leak information
# about a test compound, so over-grouping is the safe error.
DEFAULT_SERIES_THRESHOLD = 0.55


def tanimoto_matrix(fp_a: np.ndarray, fp_b: np.ndarray) -> np.ndarray:
    """Dense Tanimoto similarity between two sets of binary fingerprints.

    Args:
        fp_a: ``(n, n_bits)`` binary matrix (float32, entries in {0, 1}).
        fp_b: ``(m, n_bits)`` binary matrix.

    Returns:
        ``(n, m)`` float array of Tanimoto coefficients.
    """
    inter = fp_a @ fp_b.T
    pc_a = fp_a.sum(1)[:, None]
    pc_b = fp_b.sum(1)[None, :]
    denom = pc_a + pc_b - inter
    return np.where(denom > 0, inter / np.maximum(denom, 1e-9), 0.0)


def leader_cluster(
    fps: np.ndarray,
    threshold: float = DEFAULT_SERIES_THRESHOLD,
) -> np.ndarray:
    """Sphere-exclusion (Taylor-Butina "leader") clustering into analog series.

    Compounds are visited in order of decreasing fingerprint density, each
    unassigned compound becomes a new series leader, and every unassigned
    compound within ``threshold`` Tanimoto of it joins that series. The result is
    deterministic and order-stable, which matters because the fold assignment
    derived from it must be reproducible across runs.

    Args:
        fps: ``(n, n_bits)`` binary fingerprint matrix.
        threshold: Tanimoto cutoff for joining a series.

    Returns:
        ``(n,)`` int array of series ids in ``[0, n_series)``.
    """
    pc = fps.sum(1)
    order = np.argsort(-pc)
    assigned = -np.ones(len(fps), dtype=np.int32)
    n_series = 0
    for i in order:
        if assigned[i] != -1:
            continue
        cid = n_series
        n_series += 1
        assigned[i] = cid
        inter = fps @ fps[i]
        denom = pc + pc[i] - inter
        sim = np.where(denom > 0, inter / np.maximum(denom, 1e-9), 0.0)
        joins = np.where((sim >= threshold) & (assigned == -1))[0]
        assigned[joins] = cid
    return assigned


def series_held_out_folds(
    series_id: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    min_series_size: int = 4,
    seed: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Leave-whole-series-out folds: the pessimistic bound.

    Every member of a held-out series is removed from training, so the model must
    extrapolate to a chemotype it has never seen. This is *harder* than the
    challenge, because the challenge's parent hits are in the training DRC set.
    Use it to bound downside risk, not to select a model.

    Only series with at least ``min_series_size`` members are eligible to be held
    out; singletons and tiny series stay permanently in training, which keeps the
    training set size comparable across folds.

    Args:
        series_id: ``(n,)`` series assignment from :func:`leader_cluster`.
        y: ``(n,)`` target values; non-finite entries are dropped.
        n_folds: Number of folds.
        min_series_size: Minimum members for a series to be held out.
        seed: Shuffling seed for series-to-fold assignment.

    Yields:
        ``(train_idx, test_idx)`` index arrays into the **original** arrays.
    """
    idx = np.where(np.isfinite(y))[0]
    cl = series_id[idx]
    uniq, counts = np.unique(cl, return_counts=True)
    eligible = np.array(uniq[counts >= min_series_size], dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    for fold in np.array_split(eligible, n_folds):
        in_test = np.isin(cl, fold)
        yield idx[~in_test], idx[in_test]


def parent_in_train_folds(
    series_id: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    min_series_size: int = 4,
    seed: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Challenge-matched folds: the most potent series member stays in training.

    This reproduces the actual test-set design. For each held-out series the
    single most potent member is kept in the training set — standing in for the
    challenge's parent hit, which comes from the training DRC data — and the
    remaining members become test compounds. The model is therefore scored on
    exactly the task the leaderboard poses: interpolating activity across analogs
    of a known potent hit, where ~9-13 % of similar pairs are activity cliffs.

    Args:
        series_id: ``(n,)`` series assignment from :func:`leader_cluster`.
        y: ``(n,)`` target values; non-finite entries are dropped.
        n_folds: Number of folds.
        min_series_size: Minimum members for a series to be held out. Must be
            at least 2 so that removing the parent leaves a non-empty test set.
        seed: Shuffling seed for series-to-fold assignment.

    Yields:
        ``(train_idx, test_idx)`` index arrays into the **original** arrays.
    """
    if min_series_size < 2:
        raise ValueError("min_series_size must be >= 2 to leave a test compound")
    idx = np.where(np.isfinite(y))[0]
    cl = series_id[idx]
    uniq, counts = np.unique(cl, return_counts=True)
    eligible = np.array(uniq[counts >= min_series_size], dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    for fold in np.array_split(eligible, n_folds):
        in_test = np.isin(cl, fold)
        test_pool = idx[in_test]
        parents = np.array(
            [
                members[np.argmax(y[members])]
                for members in (idx[cl == s] for s in fold)
            ]
        )
        test_idx = np.setdiff1d(test_pool, parents)
        train_idx = np.union1d(idx[~in_test], parents)
        yield train_idx, test_idx


def emulate_challenge_testset(
    fps: np.ndarray,
    potency: np.ndarray,
    n_parents_per_isoform: int = 25,
    n_analogs: int = 10,
    isoform_potency: Sequence[np.ndarray] | None = None,
) -> dict[int, list[int]]:
    """Build a challenge-shaped hold-out: potent parents plus their top analogs.

    Mirrors the published construction — top ``n_parents_per_isoform`` hits per
    isoform, then the ``n_analogs`` nearest ECFP4 neighbours of each — so a
    candidate model can be scored on a set with the same series structure and the
    same activity-cliff density as the real test set.

    Args:
        fps: ``(n, n_bits)`` binary fingerprint matrix.
        potency: ``(n,)`` potency used to rank parents when ``isoform_potency``
            is not given.
        n_parents_per_isoform: Parents drawn per isoform.
        n_analogs: Analogs retrieved per parent.
        isoform_potency: Optional per-isoform potency arrays; parents are drawn
            from each in turn, matching the challenge's per-CYP hit selection.

    Returns:
        Mapping of parent index to its list of analog indices.
    """
    pools = list(isoform_potency) if isoform_potency is not None else [potency]
    parents: list[int] = []
    for pool in pools:
        ranked = np.argsort(-np.where(np.isfinite(pool), pool, -np.inf))
        taken = 0
        for i in ranked:
            if i in parents:
                continue
            parents.append(int(i))
            taken += 1
            if taken >= n_parents_per_isoform:
                break
    series: dict[int, list[int]] = {}
    sim = tanimoto_matrix(fps[np.array(parents)], fps)
    for row, p in enumerate(parents):
        v = sim[row].copy()
        v[p] = -1.0
        series[p] = np.argsort(-v)[:n_analogs].tolist()
    return series
