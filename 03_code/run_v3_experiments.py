"""v3 experiments: the accuracy levers left untested after the v2 leaderboard result.

Run from the CYP_challenge work-dir root:  python 03_code/run_v3_experiments.py

Board evidence (2026-08-22, 45 entries): v2 scored 0.8284 (rank 20) against rank 1's
0.5209. A controlled experiment on training data (identical |error| from truth, sign
flipped toward vs away from the credible-interval centre) shows prediction placement
is worth ~0.056 MA-ST-RAE at fixed accuracy — so at most ~18% of the 0.307 gap is
metric exploitation and >=82% is raw predictive accuracy. Rank 1's MA-R2 of 0.507
against our 0.116 says the same thing.

This script therefore tests accuracy levers only, each in isolation, under the same
5-fold scaffold-grouped protocol used for v1/v2. Local CV runs 0.113-0.127 optimistic
relative to the board (calibrated on two submissions), so add ~0.12 to forecast.

  E1  TabICL n_estimators sweep (1 / 8 / 32) — ensembling was never tried.
  E2  Full-width fingerprints (2,265 feat, no SVD) — SVD-128 vs SVD-256 differed by
      only 0.0004, suggesting compression is not binding, but this settles it.
  E3  TabPFN, the closest architectural sibling to TabICL. If the in-context-learning
      family is what matters (TabICL beat GBM on two disjoint feature families), a
      second ICL member should also beat GBM.

Writes 04_experiments/v3_benchmark.csv incrementally, so partial results survive an
interrupted run.
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
# Resolve paths against the repo root so the script runs from any cwd.
import os  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "03_code"))
import cyp_metrics as CM  # noqa: E402

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
CACHE = os.path.join(ROOT, "04_experiments", "cache")
OUT = os.path.join(ROOT, "04_experiments", "v3_benchmark.csv")

tr = pd.read_parquet(f"{CACHE}/tr_cache.parquet")
scaf = np.load(f"{CACHE}/scaf_tr.npy")
Btr = np.load(f"{CACHE}/Btr.npy")
Xtr = np.load(f"{CACHE}/Xtr.npy")


def cv(F, iso, make_model, folds=5):
    pt = f"{iso}_pIC50_direct_inhibition"
    y = tr[pt].values
    lo = tr[pt + "_conf_low"].values
    hi = tr[pt + "_conf_high"].values
    m = np.isfinite(y)
    Fi, yi, gi, loi, hii = F[m], y[m], scaf[m], lo[m], hi[m]
    oof = np.full(len(yi), np.nan)
    for a, b in GroupKFold(folds).split(Fi, yi, gi):
        mdl = make_model()
        mdl.fit(Fi[a], yi[a])
        oof[b] = mdl.predict(Fi[b])
    k = np.isfinite(oof) & np.isfinite(loi) & np.isfinite(hii)
    sec = CM.secondary_metrics(yi[k], oof[k])
    return {
        "isoform": iso,
        "n": int(k.sum()),
        "ST_RAE": CM.st_rae(oof[k], loi[k], hii[k], y_true=yi[k], denominator="st_mean"),
        "MAE": sec["MAE"],
        "R2": sec["R2"],
        "Spearman": sec["Spearman_R"],
    }


def flush(rows):
    pd.DataFrame(rows).to_csv(OUT, index=False)


def main() -> None:
    from tabicl import TabICLRegressor

    rows = []
    t0 = time.time()

    # --- E1: ensembling sweep on the v2 representation -------------------------
    for n_est in [1, 8, 32]:
        tag = f"E1_tabicl_n_estimators_{n_est}"
        for iso in ISO:
            r = cv(
                Btr,
                iso,
                lambda n=n_est: TabICLRegressor(
                    n_estimators=n, device="cuda", random_state=0, n_jobs=1
                ),
            )
            r["experiment"] = tag
            rows.append(r)
            flush(rows)
            print(f"  {tag} {iso}: ST-RAE {r['ST_RAE']:.4f} [{time.time()-t0:.0f}s]", flush=True)
        sub = [x["ST_RAE"] for x in rows if x["experiment"] == tag]
        print(f"-> {tag} MA-ST-RAE {np.mean(sub):.4f}", flush=True)

    # --- E2: full-width fingerprints, no SVD ----------------------------------
    for iso in ISO:
        try:
            r = cv(Xtr, iso, lambda: TabICLRegressor(device="cuda", random_state=0, n_jobs=1))
            r["experiment"] = "E2_tabicl_full_2265"
            rows.append(r)
            flush(rows)
            print(f"  E2 {iso}: ST-RAE {r['ST_RAE']:.4f} [{time.time()-t0:.0f}s]", flush=True)
        except Exception as exc:
            print(f"  E2 {iso}: FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            break

    # --- E3: TabPFN as a second in-context learner ----------------------------
    try:
        from tabpfn import TabPFNRegressor

        for iso in ISO:
            r = cv(Btr, iso, lambda: TabPFNRegressor(device="cuda", random_state=0))
            r["experiment"] = "E3_tabpfn_svd256"
            rows.append(r)
            flush(rows)
            print(f"  E3 {iso}: ST-RAE {r['ST_RAE']:.4f} [{time.time()-t0:.0f}s]", flush=True)
        sub = [x["ST_RAE"] for x in rows if x["experiment"] == "E3_tabpfn_svd256"]
        print(f"-> E3 MA-ST-RAE {np.mean(sub):.4f}", flush=True)
    except Exception as exc:
        print(f"  E3 FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
