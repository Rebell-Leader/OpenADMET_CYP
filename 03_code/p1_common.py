"""Shared plumbing for the P1 rebuild: folds, features, metric, OOF persistence.

Design decisions that differ from the v1-v4 pipeline (see MEMO 05):

* **Folds are compound-level and analog-splitting.** The blind set is an analog
  expansion of potent training hits, sitting at median nearest-neighbour Tanimoto
  0.587 to the training library. Single-linkage clusters at 0.55 are therefore
  distributed *round-robin* across folds rather than held disjoint, so a scored
  compound always has analogues in the training part — the deployment regime.
  Assignment is per compound (not per endpoint) because the multitask learners see
  one compound's four labels at once; a per-endpoint split would leak.
* **Only analog-covered compounds are scored.** A compound with no neighbour at
  0.55 is still trained on, but it is not representative of the blind set, so it
  does not enter the reported score.
* **The headline score is computed on the analog-covered subset** — compounds whose
  nearest neighbour in the training part of their split is at least 0.50 (median
  0.567, n = 1427), the closest match to the blind set's 0.587 the library supports.
  The all-compound score is reported alongside it as the pessimistic reference.
* **The metric is the organizers' own function**, imported from the tutorial repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator as rfg

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "02_data" / "challenge_raw"
P1 = ROOT / "04_experiments" / "p1"
CACHE = ROOT / "04_experiments" / "cache"
P1.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

import sys

sys.path.insert(0, str(ROOT / "99_scratch" / "tutorial"))
from evaluation.custom_scoring_functions import (  # noqa: E402
    rae_soft_threshold_absolute_error as _st_rae,
)

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
TARGETS = [f"{i}_pIC50_direct_inhibition" for i in ISO]
N_FOLDS = 5
SIM_FLOOR = 0.55      # single-linkage floor for analog clusters
SCORE_FLOOR = 0.50    # analog-coverage floor for a compound to enter the score
SEED = 0


# --------------------------------------------------------------------------- data
def load_challenge() -> tuple[pd.DataFrame, pd.DataFrame]:
    inh = pd.read_csv(RAW / "cyp-challenge-TRAIN_inhibition.csv")
    test = pd.read_csv(RAW / "cyp-challenge-TEST-BLINDED.csv")
    return inh, test


def labels(inh: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, lo, hi) arrays of shape (n, 4); NaN where unlabelled. Missing credible
    bounds collapse to the point estimate, which makes soft-thresholding a no-op
    for that row — the organizers' documented behaviour."""
    y = inh[TARGETS].to_numpy(float)
    lo = inh[[t + "_conf_low" for t in TARGETS]].to_numpy(float)
    hi = inh[[t + "_conf_high" for t in TARGETS]].to_numpy(float)
    lo = np.where(np.isnan(lo), y, lo)
    hi = np.where(np.isnan(hi), y, hi)
    return y, lo, hi


# ----------------------------------------------------------------------- features
def _fps_and_desc(smiles: list[str]):
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


def features(smiles_train: list[str], smiles_test: list[str]):
    """ECFP4 counts + RDKit 2D descriptors for train and test, plus bit fingerprints
    (needed for the similarity-based fold construction). Cached on disk."""
    cache = CACHE / "p1_features.npz"
    if cache.exists():
        z = np.load(cache)
        bits = _bits_from_smiles(smiles_train + smiles_test)
        n = len(smiles_train)
        return (z["fp_tr"], z["desc_tr"], z["fp_te"], z["desc_te"], bits[:n], bits[n:])
    fp_tr, desc_tr, bits_tr = _fps_and_desc(smiles_train)
    fp_te, desc_te, bits_te = _fps_and_desc(smiles_test)
    np.savez_compressed(cache, fp_tr=fp_tr, desc_tr=desc_tr, fp_te=fp_te, desc_te=desc_te)
    return fp_tr, desc_tr, fp_te, desc_te, bits_tr, bits_te


def _bits_from_smiles(smiles: list[str]):
    cgen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
    return [cgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


# -------------------------------------------------------------------------- folds
def similarity_matrix(bits) -> np.ndarray:
    n = len(bits)
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        S[i] = DataStructs.BulkTanimotoSimilarity(bits[i], bits)
    np.fill_diagonal(S, 0.0)
    return S


def _single_linkage(S: np.ndarray, floor: float) -> np.ndarray:
    """Union-find over edges above `floor`."""
    n = S.shape[0]
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = np.where(np.triu(S >= floor, 1))
    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def build_folds(bits, y: np.ndarray, rebuild: bool = False) -> dict:
    """Compound-level analog-splitting folds. Returns fold_of, scored, headline mask."""
    cache = P1 / "folds.npz"
    if cache.exists() and not rebuild:
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    S = similarity_matrix(bits)
    clusters = _single_linkage(S, SIM_FLOOR)
    rng = np.random.default_rng(SEED)
    fold_of = np.full(len(bits), -1)
    for cid in np.unique(clusters):
        members = np.where(clusters == cid)[0]
        rng.shuffle(members)
        for k, idx in enumerate(members):
            fold_of[idx] = (k + int(cid)) % N_FOLDS

    # Analog coverage in the deployment sense: how close is a compound to the
    # *training part* of its own fold split. The blind set sits at median 0.587;
    # SCORE_FLOOR is calibrated so the scored subset matches that as closely as the
    # library allows while keeping usable n (0.50 -> median 0.567, n = 1427).
    nn_other = np.array([S[i][fold_of != fold_of[i]].max() for i in range(len(bits))])
    scored = nn_other >= SCORE_FLOOR
    out = dict(fold_of=fold_of, scored=scored, headline=scored,
               clusters=clusters, nn_sim=S.max(axis=1), nn_other=nn_other)
    np.savez_compressed(cache, **out)
    return out


# ------------------------------------------------------------------------- metric
def score_oof(oof: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
              mask: np.ndarray, label: str = "") -> pd.DataFrame:
    """Per-endpoint ST-RAE / MAE / R2 / Spearman over the masked compounds."""
    rows = []
    for j, target in enumerate(TARGETS):
        k = mask & np.isfinite(y[:, j]) & np.isfinite(oof[:, j])
        if k.sum() < 25:
            continue
        yy, pp = y[k, j], oof[k, j]
        rows.append(dict(
            learner=label, target=target, n=int(k.sum()),
            st_rae=_st_rae(yy, pp, y_true_upper=hi[k, j], y_true_lower=lo[k, j]),
            mae=float(np.mean(np.abs(yy - pp))),
            r2=1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2),
            spearman=float(pd.Series(yy).corr(pd.Series(pp), method="spearman")),
            pred_std_over_true_std=float(pp.std() / yy.std()),
        ))
    return pd.DataFrame(rows)


def macro(df: pd.DataFrame, col: str = "st_rae") -> float:
    return float(df.groupby("target")[col].mean().mean())


# ---------------------------------------------------------------------- artefacts
def save_learner(name: str, oof: np.ndarray, test_pred: np.ndarray, meta: dict | None = None):
    np.savez_compressed(P1 / f"{name}.npz", oof=oof, test_pred=test_pred,
                        meta=np.array([str(meta or {})]))
    print(f"saved {P1 / (name + '.npz')}  oof={oof.shape} test={test_pred.shape}")


def load_learner(name: str):
    z = np.load(P1 / f"{name}.npz", allow_pickle=True)
    return z["oof"], z["test_pred"]


def available_learners() -> list[str]:
    return sorted(p.stem for p in P1.glob("*.npz") if p.stem != "folds")
