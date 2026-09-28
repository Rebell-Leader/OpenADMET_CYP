"""Analog-expansion cross-validation: a validation instrument matched to the blind set.

The organizers built the 750-compound blind set by expanding potent training hits into
close analogues. Measured against the training library, blind compounds sit at median
nearest-neighbour Tanimoto 0.587 (ECFP4). Both fold schemes we have used so far —
random 5-fold and Murcko-scaffold GroupKFold — put validation compounds at ~0.40.
Model selection has therefore been run ~0.19 Tanimoto units away from the regime the
leaderboard scores, which under-rewards every model that exploits close analogues.

This builds folds that reproduce the deployment regime: a validation compound is only
admitted if at least one close analogue of it remains in the training part of the fold,
and admission is biased toward potent compounds the way the blind set was.

Outputs: 04_experiments/analog_cv_benchmark.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator as rfg
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "02_data" / "challenge_raw"
sys.path.insert(0, str(ROOT / "99_scratch" / "tutorial"))
from evaluation.custom_scoring_functions import (  # noqa: E402
    rae_soft_threshold_absolute_error as st_rae,
)

ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
TARGETS = [f"{i}_pIC50_direct_inhibition" for i in ISOFORMS]
SEED = 0
N_FOLDS = 5
SIM_FLOOR = 0.55          # a validation compound needs a neighbour at least this close
POTENCY_BIAS = 0.7        # fraction of val slots reserved for the potent half


def featurize(smiles):
    cgen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
    desc_names = [n for n, _ in Descriptors._descList]
    counts, descs, bits = [], [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        counts.append(cgen.GetCountFingerprintAsNumPy(mol).astype(np.float32))
        bits.append(cgen.GetFingerprint(mol))
        d = Descriptors.CalcMolDescriptors(mol)
        descs.append([d.get(n, np.nan) for n in desc_names])
    X_desc = np.nan_to_num(np.asarray(descs, float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.vstack(counts), X_desc, bits


def similarity_matrix(bits):
    n = len(bits)
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        S[i] = DataStructs.BulkTanimotoSimilarity(bits[i], bits)
    np.fill_diagonal(S, 0.0)
    return S


def analog_folds(S, y, n_folds=N_FOLDS, rng=None):
    """Assign each compound to a fold, admitting it to validation only when a close
    analogue stays behind in training. Potent compounds are admitted preferentially,
    mirroring how the blind set was expanded from potent hits."""
    rng = rng or np.random.default_rng(SEED)
    n = len(y)
    eligible = (S >= SIM_FLOOR).sum(axis=1) >= 1
    potent = y >= np.median(y)
    fold_of = np.full(n, -1)
    target_size = int(np.ceil(n / n_folds))
    for f in range(n_folds):
        pool = np.where((fold_of < 0) & eligible)[0]
        if len(pool) == 0:
            break
        pot_pool = pool[potent[pool]]
        weak_pool = pool[~potent[pool]]
        n_pot = min(len(pot_pool), int(target_size * POTENCY_BIAS))
        n_weak = min(len(weak_pool), target_size - n_pot)
        chosen = np.concatenate([
            rng.choice(pot_pool, n_pot, replace=False),
            rng.choice(weak_pool, n_weak, replace=False),
        ])
        # keep each chosen compound's closest analogue out of the same fold
        keep = []
        blocked = set()
        for i in chosen:
            if i in blocked:
                continue
            keep.append(i)
            nbrs = np.where(S[i] >= SIM_FLOOR)[0]
            if len(nbrs):
                blocked.update(nbrs[np.argsort(S[i][nbrs])[::-1][:1]].tolist())
        fold_of[np.array(keep, dtype=int)] = f
    return fold_of


def lgbm(X_tr, y_tr, X_te, **kw):
    params = dict(n_estimators=600, learning_rate=0.05, num_leaves=31,
                  colsample_bytree=0.6, subsample=0.8, subsample_freq=1,
                  min_child_samples=10, reg_lambda=1.0, random_state=SEED, verbose=-1)
    params.update(kw)
    return lgb.LGBMRegressor(**params).fit(X_tr, y_tr).predict(X_te)


def knn_tanimoto(S_te_tr, y_tr, k=5, power=3.0):
    preds = np.empty(S_te_tr.shape[0])
    for i in range(S_te_tr.shape[0]):
        idx = np.argsort(S_te_tr[i])[::-1][:k]
        w = np.maximum(S_te_tr[i][idx], 1e-6) ** power
        preds[i] = float(np.sum(w * y_tr[idx]) / np.sum(w))
    return preds


def run():
    inh = pd.read_csv(RAW / "cyp-challenge-TRAIN_inhibition.csv")
    X_fp, X_desc, bits = featurize(inh.SMILES.tolist())
    S_all = similarity_matrix(bits)

    rows = []
    for target in TARGETS:
        mask = inh[target].notna().to_numpy()
        idx = np.where(mask)[0]
        y = inh.loc[mask, target].to_numpy()
        lo = np.nan_to_num(inh.loc[mask, target + "_conf_low"].to_numpy(), nan=0.0)
        hi = np.nan_to_num(inh.loc[mask, target + "_conf_high"].to_numpy(), nan=0.0)
        lo = np.where(lo == 0, y, lo)
        hi = np.where(hi == 0, y, hi)
        S = S_all[np.ix_(idx, idx)]
        fp, desc = X_fp[mask], X_desc[mask]
        folds = analog_folds(S, y)

        models = ["lgbm_raw", "lgbm_svd256", "knn_tanimoto", "blend_lgbm_knn"]
        oof = {m: np.full(len(y), np.nan) for m in models}
        nn_sim = np.full(len(y), np.nan)
        for f in range(N_FOLDS):
            te = np.where(folds == f)[0]
            tr = np.where(folds != f)[0]          # includes unassigned compounds
            if len(te) == 0:
                continue
            nn_sim[te] = S[np.ix_(te, tr)].max(axis=1)
            sc = StandardScaler().fit(desc[tr])
            d_tr, d_te = sc.transform(desc[tr]), sc.transform(desc[te])
            oof["lgbm_raw"][te] = lgbm(np.hstack([fp[tr], d_tr]), y[tr],
                                       np.hstack([fp[te], d_te]))
            svd = TruncatedSVD(n_components=256, random_state=SEED).fit(fp[tr])
            oof["lgbm_svd256"][te] = lgbm(np.hstack([svd.transform(fp[tr]), d_tr]), y[tr],
                                          np.hstack([svd.transform(fp[te]), d_te]))
            oof["knn_tanimoto"][te] = knn_tanimoto(S[np.ix_(te, tr)], y[tr])
            oof["blend_lgbm_knn"][te] = 0.5 * oof["lgbm_raw"][te] + 0.5 * oof["knn_tanimoto"][te]

        scored = ~np.isnan(oof["lgbm_raw"])
        for m in models:
            p = oof[m][scored]
            yy, ll, hh = y[scored], lo[scored], hi[scored]
            rows.append(dict(
                target=target, model=m, n_scored=int(scored.sum()),
                st_rae=st_rae(yy, p, y_true_upper=hh, y_true_lower=ll),
                mae=float(np.mean(np.abs(yy - p))),
                r2=1 - np.sum((yy - p) ** 2) / np.sum((yy - yy.mean()) ** 2),
                spearman=pd.Series(yy).corr(pd.Series(p), method="spearman"),
                pred_std_over_true_std=p.std() / yy.std(),
                median_val_nn_tanimoto=float(np.nanmedian(nn_sim[scored])),
            ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    df.to_csv(ROOT / "04_experiments" / "analog_cv_benchmark.csv", index=False)
    print(df.groupby("model")[["st_rae", "mae", "r2", "spearman",
                               "pred_std_over_true_std", "median_val_nn_tanimoto"]]
          .mean().round(4).to_string())
    print()
    print(df.pivot_table(index="target", columns="model", values="st_rae").round(4).to_string())
