"""Rebuild the feature caches that every experiment script loads.

Run from the repo root:  python 03_code/build_features.py

Produces, in 04_experiments/cache/:

  Xtr.npy, Xte.npy   full-width features, 2,265 dims = 2,048-bit ECFP4 + 217 RDKit 2D
                     descriptors. Used by the LightGBM baselines and by the
                     full-width ablation (which was worse -- see Memo 1 §6.2).
  Btr.npy, Bte.npy   the representation the shipped models use, 473 dims =
                     TruncatedSVD-256 of the ECFP block, concatenated with the 217
                     descriptors, then standardised. The SVD is fitted on TRAIN only
                     and applied to test; likewise the scaler.
  scaf_tr.npy        Bemis-Murcko scaffold group index for the training set, used by
                     every GroupKFold split in this repo.

These are gitignored (61 MB) and deterministic given the challenge CSVs, so the repo
ships the recipe rather than the arrays. Runtime ~3 min on 16 cores.

Note on the SVD: compressing the fingerprint block is not a convenience, it is load
bearing. TabICL on the full 2,265 dims scores 0.7611 against 0.7239 on the SVD-256
representation, and triggers CUDA OOM on a 12 GB card (Memo 1 §6.2).

TRANSDUCTIVE PREPROCESSING -- read before reusing this code
-----------------------------------------------------------
``--fit-on all`` (the default) fits the SVD basis and the standardiser on the
training AND blinded-test feature matrices stacked together. That is what every
number in this repository was computed with, so it is the default here for exact
reproducibility.

It is *unsupervised* use of the test set -- no test label exists to leak, and the
blinded test SMILES are public -- and transductive preprocessing of this kind is
common practice. It is nonetheless a deviation from a strictly inductive protocol,
and a reader is entitled to know. ``--fit-on train`` gives the strict version; the
resulting arrays differ from the cached ones (SVD bases are not comparable across
fits), so scores must be recomputed if you use it. The measured effect on the
primary metric is reported in the repository README and in Memo 1.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "02_data", "challenge_raw")
CACHE = os.path.join(ROOT, "04_experiments", "cache")

N_BITS = 2048
SVD_DIM = 256
SEED = 0

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)
_DESC_NAMES = [d[0] for d in Descriptors._descList]
_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)


def featurize(smiles: list[str]) -> np.ndarray:
    """ECFP4 bits concatenated with RDKit 2D descriptors, ``[n, 2048 + n_desc]``.

    Unparsable SMILES yield an all-zero row rather than being dropped, so the output
    stays row-aligned with the input table. Descriptors are clipped to +/-1e6 and
    non-finite values zeroed -- a handful of RDKit descriptors overflow on large or
    unusual structures and would otherwise poison the scaler.
    """
    fp = np.zeros((len(smiles), N_BITS), dtype=np.float32)
    desc = np.zeros((len(smiles), len(_DESC_NAMES)), dtype=np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if mol is None:
            continue
        fp[i, list(_GEN.GetFingerprint(mol).GetOnBits())] = 1.0
        try:
            desc[i] = np.array(_CALC.CalcDescriptors(mol), dtype=np.float32)
        except Exception:
            desc[i] = np.nan
    desc = np.nan_to_num(np.clip(desc, -1e6, 1e6), nan=0.0, posinf=0.0, neginf=0.0)
    return np.hstack([fp, desc]).astype(np.float32)


def safe_murcko(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES, tolerant of bond-stereo precondition failures.

    ``MurckoScaffoldSmiles`` raises on a subset of structures whose stereo annotation
    survives scaffold extraction; the fallback strips bond stereo and re-canonicalises.
    Acyclic molecules legitimately return an empty scaffold and are grouped together.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            Chem.SanitizeMol(core)
            for bond in core.GetBonds():
                if bond.GetStereo() != Chem.BondStereo.STEREONONE:
                    bond.SetStereo(Chem.BondStereo.STEREONONE)
            return Chem.MolToSmiles(core)
        except Exception:
            return ""


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fit-on", choices=["all", "train"], default="all",
        help="fit SVD/scaler on train+test stacked (default, reproduces the cached "
             "arrays and every published number) or on train only (strictly inductive)",
    )
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    tr = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_inhibition.csv"))
    te = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))

    Xtr = featurize(tr.SMILES.tolist())
    Xte = featurize(te.SMILES.tolist())
    np.save(f"{CACHE}/Xtr.npy", Xtr)
    np.save(f"{CACHE}/Xte.npy", Xte)

    fit_fp = np.vstack([Xtr[:, :N_BITS], Xte[:, :N_BITS]]) if args.fit_on == "all" else Xtr[:, :N_BITS]
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=SEED).fit(fit_fp)
    Btr_raw = np.hstack([svd.transform(Xtr[:, :N_BITS]), Xtr[:, N_BITS:]])
    Bte_raw = np.hstack([svd.transform(Xte[:, :N_BITS]), Xte[:, N_BITS:]])
    fit_b = np.vstack([Btr_raw, Bte_raw]) if args.fit_on == "all" else Btr_raw
    scaler = StandardScaler().fit(fit_b)
    Btr = np.nan_to_num(scaler.transform(Btr_raw), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Bte = np.nan_to_num(scaler.transform(Bte_raw), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.save(f"{CACHE}/Btr.npy", Btr)
    np.save(f"{CACHE}/Bte.npy", Bte)

    scaf = pd.factorize(pd.Series([safe_murcko(s) for s in tr.SMILES]))[0]
    np.save(f"{CACHE}/scaf_tr.npy", scaf)
    tr.to_parquet(f"{CACHE}/tr_cache.parquet", index=False)

    print(
        f"Xtr {Xtr.shape} Xte {Xte.shape} | Btr {Btr.shape} Bte {Bte.shape} | "
        f"scaffold groups {len(np.unique(scaf))} | "
        f"ECFP variance retained {svd.explained_variance_ratio_.sum():.3f}"
    )


if __name__ == "__main__":
    main()
