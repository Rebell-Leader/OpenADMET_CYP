"""Score the P1 base learners and stack them into a submission.

Scoring instrument: the analog-covered subset of the compound-level analog folds
(`p1_common`), with the all-compound score reported alongside as the pessimistic
reference.

Stacking: per endpoint, a convex combination (weights >= 0, sum to 1) of the base
learners' out-of-fold predictions. Non-negativity is not cosmetic — an unconstrained
ridge/elastic-net combiner on correlated members produces large cancelling
coefficients that fit held-out folds and transfer badly. The reported stack score is
*nested*: weights are fitted on four folds and applied to the fifth, so the number is
not the in-sample optimum of the combiner.

    python 03_code/run_p1_stack.py            # score + stack, write benchmark CSVs
    python 03_code/run_p1_stack.py --submit    # also write submission_regression_v5.*
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_common import (  # noqa: E402
    ISO, N_FOLDS, P1, ROOT, TARGETS, available_learners, build_folds, features,
    labels, load_challenge, load_learner, macro, score_oof,
)


def convex_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weights >= 0 summing to 1 minimising squared error."""
    L = P.shape[1]
    w0 = np.full(L, 1 / L)
    res = minimize(lambda w: float(np.mean((P @ w - y) ** 2)), w0, method="SLSQP",
                   bounds=[(0.0, 1.0)] * L,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def main(submit: bool = False) -> None:
    inh, test = load_challenge()
    y, lo, hi = labels(inh)
    *_, bits_tr, _ = features(inh.SMILES.tolist(), test.SMILES.tolist())
    folds = build_folds(bits_tr, y)
    fold_of, scored = folds["fold_of"], folds["scored"]
    names = [n for n in available_learners() if n != "folds"]
    if not names:
        raise SystemExit("no learner npz files in 04_experiments/p1/")
    oofs = {n: load_learner(n)[0] for n in names}
    tests = {n: load_learner(n)[1] for n in names}

    # ---- per-learner scores
    per_learner = pd.concat(
        [score_oof(oofs[n], y, lo, hi, scored, n) for n in names] +
        [score_oof(oofs[n], y, lo, hi, np.ones(len(y), bool), n + " [all]") for n in names]
    )
    per_learner.to_csv(ROOT / "04_experiments" / "p1_learner_scores.csv", index=False)

    # ---- stack, per endpoint
    stack_oof = np.full((len(y), 4), np.nan)
    rows, weight_rows = [], []
    for j, target in enumerate(TARGETS):
        lab = np.isfinite(y[:, j]) & np.all([np.isfinite(oofs[n][:, j]) for n in names], axis=0)
        P = np.column_stack([oofs[n][lab, j] for n in names])
        yy = y[lab, j]
        f_lab = fold_of[lab]
        nested = np.zeros(len(yy))
        for f in range(N_FOLDS):
            tr, va = f_lab != f, f_lab == f
            if va.sum() == 0:
                continue
            nested[va] = P[va] @ convex_weights(P[tr], yy[tr])
        stack_oof[np.where(lab)[0], j] = nested
        w_full = convex_weights(P, yy)
        weight_rows.append(dict(target=target, **{n: round(float(w), 4)
                                                  for n, w in zip(names, w_full)}))
        rows.append(dict(target=target, n_labelled=int(lab.sum())))
    weights = pd.DataFrame(weight_rows)
    weights.to_csv(ROOT / "04_experiments" / "p1_stack_weights.csv", index=False)

    stack_scores = pd.concat([
        score_oof(stack_oof, y, lo, hi, scored, "stack (nested)"),
        score_oof(stack_oof, y, lo, hi, np.ones(len(y), bool), "stack (nested) [all]"),
    ])
    all_scores = pd.concat([per_learner, stack_scores])
    all_scores.to_csv(ROOT / "04_experiments" / "p1_benchmark.csv", index=False)

    summary = (all_scores.groupby("learner")[["st_rae", "mae", "r2", "spearman"]]
               .mean().sort_values("st_rae").round(4))
    print("macro over the four endpoints (analog-covered subset unless marked [all]):")
    print(summary.to_string())
    print("\nper-endpoint ST-RAE, analog-covered subset:")
    print(all_scores[~all_scores.learner.str.contains(r"\[all\]")]
          .pivot_table(index="target", columns="learner", values="st_rae").round(4).to_string())
    print("\nstack weights:")
    print(weights.to_string(index=False))

    if submit:
        pred = np.zeros((len(test), 4))
        for j, target in enumerate(TARGETS):
            w = weights.loc[j, names].to_numpy(float)
            pred[:, j] = np.column_stack([tests[n][:, j] for n in names]) @ w
            floor, ceil = np.nanmin(y[:, j]), np.nanmax(y[:, j])
            pred[:, j] = np.clip(pred[:, j], floor, ceil)
            assert pred[:, j].std() > 0.01, f"{target} fails MIN_PREDICTION_STD"
        sub = pd.DataFrame({"SMILES": test.SMILES, "Molecule_Name": test.Molecule_Name})
        for j, target in enumerate(TARGETS):
            sub[target] = pred[:, j]
        out = ROOT / "05_submissions"
        sub.to_parquet(out / "submission_regression_v5.parquet", index=False)
        sub.to_csv(out / "submission_regression_v5.csv", index=False)
        print("\nwrote submission_regression_v5.{parquet,csv}")
        print(sub[TARGETS].describe().round(3).to_string())


if __name__ == "__main__":
    main(submit="--submit" in sys.argv)
