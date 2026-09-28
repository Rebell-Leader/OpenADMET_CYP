"""Split-regime diagnostic: is our local CV measuring the right thing?

The blinded test set is an *analog expansion* of the training library, so test
compounds sit closer to training chemistry than training compounds sit to each
other. Model selection under scaffold-grouped folds therefore optimises for an
extrapolation regime the leaderboard never asks about.

This script re-runs three model families under two fold regimes (scaffold-grouped
vs random) with the organizers' own ST-RAE, and reports the prediction-variance
ratio that drives the R2 collapse seen on the board.

Outputs: 04_experiments/split_regime_diagnostic.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "02_data" / "challenge_raw"
TUTORIAL = ROOT / "99_scratch" / "tutorial"
sys.path.insert(0, str(TUTORIAL))
from evaluation.custom_scoring_functions import (  # noqa: E402
    rae_soft_threshold_absolute_error as st_rae,
)

ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
TARGETS = [f"{i}_pIC50_direct_inhibition" for i in ISOFORMS]
SEED = 0


def featurize(smiles: list[str]) -> tuple[np.ndarray, np.ndarray, list]:
    """ECFP4 count fingerprints, RDKit 2D descriptors, and bit fingerprints."""
    cgen = rfg.GetMorganGenerator(radius=2, fpSize=2048, countSimulation=False)
    counts, descs, bits = [], [], []
    desc_names = [n for n, _ in Descriptors._descList]
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        counts.append(cgen.GetCountFingerprintAsNumPy(mol).astype(np.float32))
        bits.append(cgen.GetFingerprint(mol))
        d = Descriptors.CalcMolDescriptors(mol)
        descs.append([d.get(n, np.nan) for n in desc_names])
    X_fp = np.vstack(counts)
    X_desc = np.asarray(descs, dtype=np.float64)
    X_desc = np.nan_to_num(X_desc, nan=0.0, posinf=0.0, neginf=0.0)
    return X_fp, X_desc, bits


def scaffold_groups(smiles: list[str]) -> np.ndarray:
    scaffs = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        core = MurckoScaffold.GetScaffoldForMol(mol)
        scaffs.append(Chem.MolToSmiles(core) if core.GetNumAtoms() else smi)
    codes = {s: i for i, s in enumerate(sorted(set(scaffs)))}
    return np.array([codes[s] for s in scaffs])


def knn_tanimoto(bits_tr, y_tr, bits_te, k=10, power=3.0):
    preds = np.empty(len(bits_te))
    for i, fp in enumerate(bits_te):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, bits_tr))
        idx = np.argsort(sims)[::-1][:k]
        w = np.maximum(sims[idx], 1e-6) ** power
        preds[i] = float(np.sum(w * y_tr[idx]) / np.sum(w))
    return preds


def lgbm_fit_predict(X_tr, y_tr, X_te):
    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=31,
        colsample_bytree=0.6, subsample=0.8, subsample_freq=1,
        min_child_samples=10, reg_lambda=1.0, random_state=SEED, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def run() -> pd.DataFrame:
    inh = pd.read_csv(RAW / "cyp-challenge-TRAIN_inhibition.csv")
    X_fp, X_desc, bits = featurize(inh.SMILES.tolist())
    groups = scaffold_groups(inh.SMILES.tolist())

    rows = []
    for regime in ["scaffold", "random"]:
        for target in TARGETS:
            mask = inh[target].notna().to_numpy()
            idx_all = np.where(mask)[0]
            y = inh.loc[mask, target].to_numpy()
            lo = inh.loc[mask, target + "_conf_low"].to_numpy()
            hi = inh.loc[mask, target + "_conf_high"].to_numpy()
            lo = np.where(np.isnan(lo), y, lo)
            hi = np.where(np.isnan(hi), y, hi)
            fp_t, desc_t = X_fp[mask], X_desc[mask]
            bits_t = [bits[i] for i in idx_all]
            g = groups[mask]

            if regime == "scaffold":
                splitter = GroupKFold(n_splits=5).split(fp_t, y, groups=g)
            else:
                splitter = KFold(n_splits=5, shuffle=True, random_state=SEED).split(fp_t)

            oof = {m: np.zeros(len(y)) for m in ["lgbm_raw", "lgbm_svd256", "knn_tanimoto"]}
            nn_sims = np.zeros(len(y))
            for tr, te in splitter:
                # how close is each validation compound to its own fold's training set?
                for j, i_te in enumerate(te):
                    sims = DataStructs.BulkTanimotoSimilarity(
                        bits_t[i_te], [bits_t[i] for i in tr]
                    )
                    nn_sims[i_te] = max(sims)

                scaler = StandardScaler().fit(desc_t[tr])
                d_tr, d_te = scaler.transform(desc_t[tr]), scaler.transform(desc_t[te])
                raw_tr = np.hstack([fp_t[tr], d_tr])
                raw_te = np.hstack([fp_t[te], d_te])
                oof["lgbm_raw"][te] = lgbm_fit_predict(raw_tr, y[tr], raw_te)

                svd = TruncatedSVD(n_components=256, random_state=SEED).fit(fp_t[tr])
                s_tr = np.hstack([svd.transform(fp_t[tr]), d_tr])
                s_te = np.hstack([svd.transform(fp_t[te]), d_te])
                oof["lgbm_svd256"][te] = lgbm_fit_predict(s_tr, y[tr], s_te)

                oof["knn_tanimoto"][te] = knn_tanimoto(
                    [bits_t[i] for i in tr], y[tr], [bits_t[i] for i in te]
                )

            for model, pred in oof.items():
                score = st_rae(y, pred, y_true_upper=hi, y_true_lower=lo)
                # variance-matched affine recalibration (mean/std of y preserved)
                rescaled = (pred - pred.mean()) / pred.std() * y.std() + y.mean()
                score_rescaled = st_rae(y, rescaled, y_true_upper=hi, y_true_lower=lo)
                rows.append(dict(
                    regime=regime, target=target, model=model, n=len(y),
                    st_rae=score, st_rae_variance_matched=score_rescaled,
                    pred_std_over_true_std=pred.std() / y.std(),
                    r2=1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2),
                    spearman=pd.Series(y).corr(pd.Series(pred), method="spearman"),
                    median_val_nn_tanimoto=float(np.median(nn_sims)),
                ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    out = ROOT / "04_experiments" / "split_regime_diagnostic.csv"
    df.to_csv(out, index=False)
    macro = df.groupby(["regime", "model"])[
        ["st_rae", "st_rae_variance_matched", "pred_std_over_true_std", "r2",
         "spearman", "median_val_nn_tanimoto"]
    ].mean().round(4)
    print(macro.to_string())
