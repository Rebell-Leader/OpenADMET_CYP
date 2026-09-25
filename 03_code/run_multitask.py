"""Benchmark the multitask variants against the single-task TabICL baseline.

Run from the work-dir root: python 03_code/run_multitask.py

Variants, all under identical 5-fold scaffold-grouped splits:

  st_mlp      four independent MLPs, same trunk width as the multitask net. The control
              that separates "multitask helped" from "a neural net helped".
  mt_3        shared trunk over CYP1A2/2C9/3A4, CYP2D6 fitted separately (Memo 3 §3.3
              default, following the orthogonal-SAR finding in Memo 2 §2).
  mt_all4     shared trunk over all four. Tests the CYP2D6 exclusion rather than
              assuming it.
  mt_3_aux    mt_3 plus auxiliary log2fc heads (aux_weight 0.3) on the dense screen.
  mt_all4_aux mt_all4 plus the same auxiliary heads.

Early stopping uses an inner scaffold-grouped split of the training fold, so no
held-out row influences the stopping epoch. Target standardisation is fitted on the
inner-train rows only.

Writes 04_experiments/multitask_benchmark.csv and per-variant OOF predictions to
04_experiments/cache/mt_oof_{variant}.npy for later ensembling.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "03_code"))
import cyp_metrics as CM  # noqa: E402
from multitask import MTConfig, fit_multitask, predict_multitask  # noqa: E402

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
CACHE = os.path.join(ROOT, "04_experiments", "cache")
DATA = os.path.join(ROOT, "02_data", "challenge_raw")
OUT = os.path.join(ROOT, "04_experiments", "multitask_benchmark.csv")
SEEDS = (0, 1, 2)


def load():
    tr = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_inhibition.csv"))
    sc = pd.read_csv(os.path.join(DATA, "cyp-challenge-single-concentration-TRAIN.csv"))
    piv = sc.pivot_table(
        index="Molecule_Name", columns="enzyme", values="log2fc_estimate", aggfunc="first"
    )
    piv.columns = [f"log2fc_{c}" for c in piv.columns]
    tr = tr.merge(piv, left_on="Molecule_Name", right_index=True, how="left")
    Y = np.column_stack([tr[f"{i}_pIC50_direct_inhibition"].values for i in ISO])
    A = np.column_stack([tr[f"log2fc_{i}"].values for i in ISO])
    lo = np.column_stack([tr[f"{i}_pIC50_direct_inhibition_conf_low"].values for i in ISO])
    hi = np.column_stack([tr[f"{i}_pIC50_direct_inhibition_conf_high"].values for i in ISO])
    return Y, A, lo, hi


def inner_split(groups: np.ndarray, rows: np.ndarray, seed: int = 0):
    """Scaffold-grouped 80/20 split of a training fold, for early stopping."""
    g = np.unique(groups[rows])
    rng = np.random.default_rng(seed)
    rng.shuffle(g)
    cut = max(1, int(0.2 * len(g)))
    va_g = set(g[:cut].tolist())
    is_va = np.array([x in va_g for x in groups[rows]])
    return rows[~is_va], rows[is_va]


def run_variant(variant, X, Y, A, groups, seed):
    """Return an OOF prediction matrix [n, 4] with NaN where the task is unobserved."""
    n = len(X)
    oof = np.full((n, 4), np.nan)
    share = [0, 1, 2, 3] if "all4" in variant else [0, 1, 3]
    solo = [] if "all4" in variant else [2]
    use_aux = variant.endswith("_aux")

    def fit_block(cols, tr_rows, te_rows):
        itr, iva = inner_split(groups, tr_rows, seed)
        cfg = MTConfig(
            aux_weight=0.3 if use_aux else 0.0, seed=seed,
            epochs=300, patience=40,
        )
        net, _ = fit_multitask(
            X[itr], Y[np.ix_(itr, cols)], X[iva], Y[np.ix_(iva, cols)], cfg,
            A[np.ix_(itr, cols)] if use_aux else None,
            A[np.ix_(iva, cols)] if use_aux else None,
        )
        oof[np.ix_(te_rows, cols)] = predict_multitask(net, X[te_rows])

    any_lab = np.isfinite(Y).any(1)
    idx = np.where(any_lab)[0]
    for a, b in GroupKFold(5).split(idx, groups=groups[idx]):
        tr_rows, te_rows = idx[a], idx[b]
        if variant == "st_mlp":
            for j in range(4):
                sub = tr_rows[np.isfinite(Y[tr_rows, j])]
                fit_block([j], sub, te_rows)
        else:
            fit_block(share, tr_rows, te_rows)
            for j in solo:
                sub = tr_rows[np.isfinite(Y[tr_rows, j])]
                fit_block([j], sub, te_rows)
    return oof


def main() -> None:
    Y, A, lo, hi = load()
    X = np.load(f"{CACHE}/Btr.npy")
    groups = np.load(f"{CACHE}/scaf_tr.npy")
    assert len(X) == len(Y) == len(groups), (X.shape, Y.shape, groups.shape)
    print(f"X {X.shape} | labels {int(np.isfinite(Y).sum())} | aux {int(np.isfinite(A).sum())}", flush=True)

    rows = []
    t0 = time.time()
    for variant in ["st_mlp", "mt_3", "mt_all4", "mt_3_aux", "mt_all4_aux"]:
        preds = [run_variant(variant, X, Y, A, groups, s) for s in SEEDS]
        oof = np.nanmean(np.stack(preds), axis=0)  # seed-average
        np.save(f"{CACHE}/mt_oof_{variant}.npy", oof)
        for j, iso in enumerate(ISO):
            k = np.isfinite(Y[:, j]) & np.isfinite(oof[:, j])
            sec = CM.secondary_metrics(Y[k, j], oof[k, j])
            rows.append(
                {
                    "variant": variant, "isoform": iso, "n": int(k.sum()),
                    "ST_RAE": CM.st_rae(
                        oof[k, j], lo[k, j], hi[k, j], y_true=Y[k, j], denominator="st_mean"
                    ),
                    "MAE": sec["MAE"], "R2": sec["R2"], "Spearman": sec["Spearman_R"],
                }
            )
        pd.DataFrame(rows).to_csv(OUT, index=False)
        sub = [r for r in rows if r["variant"] == variant]
        print(
            f"{variant:12s} MA-ST-RAE {np.mean([r['ST_RAE'] for r in sub]):.4f}  "
            + " ".join(f"{r['isoform'][3:]}={r['ST_RAE']:.3f}" for r in sub)
            + f"  [{time.time()-t0:.0f}s]",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
