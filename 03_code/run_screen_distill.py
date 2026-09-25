"""Distil the dense single-concentration screen into features for the sparse pIC50 task.

Run from the work-dir root: python 03_code/run_screen_distill.py

WHY
---
The single-concentration screen (`cyp-challenge-single-concentration-TRAIN.csv`) is the
largest untapped in-domain signal: 4,376 compounds x 4 enzymes at 4.95e-5 M, DENSE where
the dose-response file is sparse (CYP1A2 1,412 / CYP2C9 1,285 / CYP2D6 1,493 / CYP3A4
2,335 fitted curves). Observed log2 fold-change correlates with same-isoform DRC pIC50 at
Spearman -0.86 / -0.90 / -0.83 / -0.94 — stronger than any feature we have.

It cannot be used directly: ZERO of the 750 blinded test compounds appear in the screen
(checked by both Molecule_Name and SMILES). So we distil it — learn structure -> log2fc on
the dense labels, then feed the PREDICTED log2fc to the pIC50 model. This is the same
pattern as the v2 cross-isoform stacking, but the donor models see 2-3x more labels per
isoform, which is where the gain should come from.

LEAKAGE
-------
More dangerous than in v2. There, donors were other isoforms measured on largely
non-overlapping compounds, and the measured leak was 0.0008. Here the donor label and the
target label live on the SAME compound (4,375 of 4,376 screen compounds are in the DRC
file), so a donor fit on all rows would hand the target model a function of the held-out
compound's own measured behaviour. Every number below is therefore fully nested: inside
each outer fold we delete the held-out scaffold groups, refit all four donors on what
remains, and only then predict the held-out rows.

The `naive` variant is computed too, purely to measure how large the leak would have been.

Writes 04_experiments/screen_distill_benchmark.csv and, for the winning configuration,
04_experiments/cache/screen_feats_{train,test}.npy.
"""

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

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
CACHE = os.path.join(ROOT, "04_experiments", "cache")
DATA = os.path.join(ROOT, "02_data", "challenge_raw")
OUT = os.path.join(ROOT, "04_experiments", "screen_distill_benchmark.csv")


def load():
    tr = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_inhibition.csv"))
    sc = pd.read_csv(os.path.join(DATA, "cyp-challenge-single-concentration-TRAIN.csv"))
    piv = sc.pivot_table(
        index="Molecule_Name", columns="enzyme", values="log2fc_estimate", aggfunc="first"
    )
    piv.columns = [f"log2fc_{c}" for c in piv.columns]
    tr = tr.merge(piv, left_on="Molecule_Name", right_index=True, how="left")
    return tr


def main() -> None:
    import lightgbm as lgb
    from tabicl import TabICLRegressor

    tr = load()
    scaf = np.load(f"{CACHE}/scaf_tr.npy")
    Btr = np.load(f"{CACHE}/Btr.npy")
    Bte = np.load(f"{CACHE}/Bte.npy")
    assert len(tr) == len(scaf) == len(Btr), (len(tr), len(scaf), len(Btr))

    LGB = dict(
        n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=10,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=1.0,
        n_jobs=8, verbosity=-1, random_state=0,
    )
    log2 = {i: tr[f"log2fc_{i}"].values for i in ISO}
    print(
        "log2fc coverage: " + ", ".join(f"{i} {int(np.isfinite(log2[i]).sum())}" for i in ISO),
        flush=True,
    )

    def fit_donors(row_mask):
        """Fit structure -> log2fc for all four isoforms on the given rows only."""
        out = {}
        for i in ISO:
            k = row_mask & np.isfinite(log2[i])
            out[i] = lgb.LGBMRegressor(**LGB).fit(Btr[k], log2[i][k])
        return out

    rows = []
    t0 = time.time()

    # Naive (leaky) donor predictions, for leak measurement only.
    donors_all = fit_donors(np.ones(len(tr), bool))
    naive_feat = np.column_stack([donors_all[i].predict(Btr) for i in ISO]).astype(np.float32)

    for iso in ISO:
        pt = f"{iso}_pIC50_direct_inhibition"
        y = tr[pt].values
        lo = tr[pt + "_conf_low"].values
        hi = tr[pt + "_conf_high"].values
        m = np.isfinite(y)
        idx = np.where(m)[0]
        gi = scaf[m]

        oof = {"nested": np.full(len(idx), np.nan), "naive": np.full(len(idx), np.nan)}
        for a, b in GroupKFold(5).split(idx, y[idx], gi):
            tr_rows, te_rows = idx[a], idx[b]
            # nested: donors never see the held-out scaffold groups
            keep = np.zeros(len(tr), bool)
            keep[tr_rows] = True
            donors = fit_donors(keep)
            f_tr = np.column_stack([donors[i].predict(Btr[tr_rows]) for i in ISO])
            f_te = np.column_stack([donors[i].predict(Btr[te_rows]) for i in ISO])
            mdl = TabICLRegressor(n_estimators=32, device="cuda", random_state=0, n_jobs=1)
            mdl.fit(np.hstack([Btr[tr_rows], f_tr]).astype(np.float32), y[tr_rows])
            oof["nested"][b] = mdl.predict(
                np.hstack([Btr[te_rows], f_te]).astype(np.float32)
            )
            # naive: donors fit on everything (leaky by construction)
            mdl2 = TabICLRegressor(n_estimators=32, device="cuda", random_state=0, n_jobs=1)
            mdl2.fit(
                np.hstack([Btr[tr_rows], naive_feat[tr_rows]]).astype(np.float32), y[tr_rows]
            )
            oof["naive"][b] = mdl2.predict(
                np.hstack([Btr[te_rows], naive_feat[te_rows]]).astype(np.float32)
            )
            print(f"    {iso} fold done [{time.time()-t0:.0f}s]", flush=True)

        for variant, p in oof.items():
            k = np.isfinite(p)
            sec = CM.secondary_metrics(y[idx][k], p[k])
            rows.append(
                {
                    "isoform": iso,
                    "variant": variant,
                    "n": int(k.sum()),
                    "ST_RAE": CM.st_rae(
                        p[k], lo[idx][k], hi[idx][k], y_true=y[idx][k], denominator="st_mean"
                    ),
                    "MAE": sec["MAE"],
                    "R2": sec["R2"],
                    "Spearman": sec["Spearman_R"],
                }
            )
        pd.DataFrame(rows).to_csv(OUT, index=False)
        nn = [r for r in rows if r["isoform"] == iso]
        print(
            f"  {iso}: "
            + " | ".join(f"{r['variant']} ST-RAE {r['ST_RAE']:.4f} R2 {r['R2']:.3f}" for r in nn)
            + f" [{time.time()-t0:.0f}s]",
            flush=True,
        )

    d = pd.DataFrame(rows)
    for v in ["nested", "naive"]:
        print(f"MA-ST-RAE ({v}): {d[d.variant==v].ST_RAE.mean():.4f}", flush=True)

    # Donor features for the real test set, fit on all training rows (no test label exists,
    # so this cannot leak) -- saved for building a v3 submission.
    np.save(
        f"{CACHE}/screen_feats_test.npy",
        np.column_stack([donors_all[i].predict(Bte) for i in ISO]).astype(np.float32),
    )
    np.save(f"{CACHE}/screen_feats_train.npy", naive_feat)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
