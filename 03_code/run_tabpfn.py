"""E3: TabPFN as a second in-context learner, tested against TabICL.

Run from the work-dir root: python 03_code/run_tabpfn.py

Rationale: TabICL beat gradient boosting on two disjoint feature families
(ECFP4+descriptors -0.039, CheMeleon embeddings -0.034). If in-context learning is
what carries that gain rather than TabICL specifically, its closest architectural
sibling should also beat GBM — and two different ICL models would then be worth
ensembling. If TabPFN instead lands near GBM, the gain is TabICL-specific and the
architecture story needs narrowing.

Same 5-fold scaffold-grouped folds and the same SVD-256+descriptor representation as
the v2 submission, so the comparison is like-for-like.
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
OUT = os.path.join(ROOT, "04_experiments", "tabpfn_benchmark.csv")

tr = pd.read_parquet(f"{CACHE}/tr_cache.parquet")
scaf = np.load(f"{CACHE}/scaf_tr.npy")
Btr = np.load(f"{CACHE}/Btr.npy")


def main() -> None:
    from tabpfn import TabPFNRegressor

    rows = []
    t0 = time.time()
    for iso in ISO:
        pt = f"{iso}_pIC50_direct_inhibition"
        y = tr[pt].values
        lo = tr[pt + "_conf_low"].values
        hi = tr[pt + "_conf_high"].values
        m = np.isfinite(y)
        Fi, yi, gi, loi, hii = Btr[m], y[m], scaf[m], lo[m], hi[m]
        oof = np.full(len(yi), np.nan)
        for a, b in GroupKFold(5).split(Fi, yi, gi):
            mdl = TabPFNRegressor(device="cuda", random_state=0)
            mdl.fit(Fi[a], yi[a])
            oof[b] = mdl.predict(Fi[b])
        k = np.isfinite(oof)
        sec = CM.secondary_metrics(yi[k], oof[k])
        rows.append(
            {
                "isoform": iso,
                "n": int(k.sum()),
                "ST_RAE": CM.st_rae(
                    oof[k], loi[k], hii[k], y_true=yi[k], denominator="st_mean"
                ),
                "MAE": sec["MAE"],
                "R2": sec["R2"],
                "Spearman": sec["Spearman_R"],
                "experiment": "E3_tabpfn_svd256",
            }
        )
        pd.DataFrame(rows).to_csv(OUT, index=False)
        np.save(os.path.join(CACHE, f"tabpfn_oof_{iso}.npy"), oof)
        print(
            f"  E3 {iso}: ST-RAE {rows[-1]['ST_RAE']:.4f} R2 {sec['R2']:.3f} "
            f"[{time.time()-t0:.0f}s]",
            flush=True,
        )
    print(f"-> E3 MA-ST-RAE {np.mean([r['ST_RAE'] for r in rows]):.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
