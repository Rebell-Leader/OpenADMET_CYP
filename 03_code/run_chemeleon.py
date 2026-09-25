"""Extract CheMeleon MPNN embeddings and benchmark them inside TabICL and LightGBM.

Motivation, from decomposing the organizers' four baselines on the same blinded test
set: the learned-representation effect (CheMeleon 0.8335 vs LGBM 0.8929 = -0.059) is
roughly a quarter of the architecture effect (TabICL 0.6755 = -0.217). Neither
ingredient alone reaches 0.6755, so the question is whether they compose.

CheMeleon is a chemprop message-passing encoder pretrained on Mordred descriptors;
`chemeleon_mp.pt` holds the MPNN state dict. We use it frozen as a featurizer (no
fine-tuning), producing a 2048-dim graph embedding per molecule, then score three
representations under the identical 5-fold scaffold-grouped protocol used everywhere
else in this project.

Writes chemeleon_benchmark.csv and chemeleon_embeddings.npz.
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, "mods")
import cyp_metrics as CM  # noqa: E402

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
CKPT = "chemeleon/chemeleon_mp.pt"


def build_encoder():
    """Load the CheMeleon MPNN as a frozen featurizer."""
    from chemprop.nn import BondMessagePassing

    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    hp = state["hyper_parameters"]
    mp = BondMessagePassing(**hp)
    mp.load_state_dict(state["state_dict"])
    mp.eval()
    return mp, hp


def embed(smiles, mp, batch_size=256):
    """Mean-aggregate CheMeleon node embeddings into one vector per molecule."""
    from chemprop.data import BatchMolGraph, MoleculeDatapoint, MoleculeDataset
    from chemprop.nn import MeanAggregation

    agg = MeanAggregation()
    ds = MoleculeDataset([MoleculeDatapoint.from_smi(s) for s in smiles])
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            graphs = [ds[j].mg for j in range(i, min(i + batch_size, len(ds)))]
            bmg = BatchMolGraph(graphs)
            H = mp(bmg)
            out.append(agg(H, bmg.batch).cpu().numpy())
    return np.vstack(out).astype(np.float32)


def cv_score(F, iso, tr, scaf, make_model):
    pt = f"{iso}_pIC50_direct_inhibition"
    y = tr[pt].values
    lo = tr[pt + "_conf_low"].values
    hi = tr[pt + "_conf_high"].values
    m = np.isfinite(y)
    Fi, yi, gi, loi, hii = F[m], y[m], scaf[m], lo[m], hi[m]
    oof = np.full(len(yi), np.nan)
    for trn, tst in GroupKFold(5).split(Fi, yi, gi):
        mdl = make_model()
        mdl.fit(Fi[trn], yi[trn])
        oof[tst] = mdl.predict(Fi[tst])
    k = np.isfinite(oof) & np.isfinite(loi) & np.isfinite(hii)
    sec = CM.secondary_metrics(yi[k], oof[k])
    return {
        "isoform": iso,
        "n": int(k.sum()),
        "ST_RAE": CM.st_rae(
            oof[k], loi[k], hii[k], y_true=yi[k], denominator="st_mean"
        ),
        "MAE": sec["MAE"],
        "R2": sec["R2"],
        "Spearman": sec["Spearman_R"],
    }


def main() -> None:
    import lightgbm as lgb
    from tabicl import TabICLRegressor

    tr = pd.read_parquet("tr_cache.parquet")
    te = pd.read_csv("challenge_data/cyp-challenge-TEST-BLINDED.csv")
    scaf = np.load("scaf_tr.npy")
    Btr = np.load("Btr.npy")
    Bte = np.load("Bte.npy")

    mp, hp = build_encoder()
    print("CheMeleon hyper_parameters:", hp, flush=True)
    t0 = time.time()
    Etr = embed(tr.SMILES.tolist(), mp)
    Ete = embed(te.SMILES.tolist(), mp)
    print(f"embeddings: {Etr.shape} {Ete.shape} [{time.time() - t0:.0f}s]", flush=True)
    np.savez_compressed("chemeleon_embeddings.npz", Etr=Etr, Ete=Ete)

    sc = StandardScaler().fit(np.vstack([Etr, Ete]))
    Ztr = np.nan_to_num(sc.transform(Etr), nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )
    Zte = np.nan_to_num(sc.transform(Ete), nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )

    # CheMeleon is 2048-dim, far above TabICL's pretraining feature regime, so it is
    # compressed the same way the fingerprint block was.
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=256, random_state=0).fit(np.vstack([Ztr, Zte]))
    Str = svd.transform(Ztr).astype(np.float32)
    Ste = svd.transform(Zte).astype(np.float32)
    print(
        "chemeleon SVD-256 explained var:",
        round(float(svd.explained_variance_ratio_.sum()), 3),
        flush=True,
    )

    reps = {
        "chemeleon_lgbm_full2048": (Ztr, "lgbm"),
        "chemeleon_tabicl_svd256": (Str, "tabicl"),
        "chemeleon_svd256_plus_B": (
            np.hstack([Str, Btr]).astype(np.float32),
            "tabicl",
        ),
    }
    LGB = dict(
        n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=10,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=1.0,
        n_jobs=16, verbosity=-1, random_state=0,
    )

    rows = []
    for name, (F, kind) in reps.items():
        mk = (
            (lambda: lgb.LGBMRegressor(**LGB))
            if kind == "lgbm"
            else (lambda: TabICLRegressor(device="cuda", random_state=0, n_jobs=1))
        )
        for iso in ISO:
            r = cv_score(F, iso, tr, scaf, mk)
            r["representation"] = name
            rows.append(r)
            print(
                f"  {name} {iso}: ST-RAE {r['ST_RAE']:.4f} [{time.time() - t0:.0f}s]",
                flush=True,
            )
        sub = [r for r in rows if r["representation"] == name]
        print(
            f"-> {name} MA-ST-RAE {np.mean([r['ST_RAE'] for r in sub]):.4f}", flush=True
        )

    pd.DataFrame(rows).to_csv("chemeleon_benchmark.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
