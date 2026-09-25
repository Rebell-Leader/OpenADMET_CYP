"""Cross-validate TabICL against the LightGBM baseline on the released CYP training data.

TabICL leads the challenge regression leaderboard by 0.096 MA-ST-RAE over rank 2 (z=2.69),
the only statistically separated gap in the top 10 — so the question worth answering is
whether the architecture, not the feature set, is doing the work. Same features, same
scaffold-grouped folds, same metric as the LightGBM run, so the delta is attributable.
"""

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, "mods")
import cyp_metrics as CM  # noqa: E402

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

Rtr = np.load("Rtr.npy")
Rte = np.load("Rte.npy")
scaf_tr = np.load("scaf_tr.npy")
tr = pd.read_parquet("tr_cache.parquet")


def cv_one(make_model, iso, n_folds=5):
    pt = f"{iso}_pIC50_direct_inhibition"
    y = tr[pt].values
    lo = tr[pt + "_conf_low"].values
    hi = tr[pt + "_conf_high"].values
    m = np.isfinite(y)
    Ri, yi, gi, loi, hii = Rtr[m], y[m], scaf_tr[m], lo[m], hi[m]
    oof = np.full(len(yi), np.nan)
    for trn, tst in GroupKFold(n_folds).split(Ri, yi, gi):
        mdl = make_model()
        mdl.fit(Ri[trn], yi[trn])
        oof[tst] = mdl.predict(Ri[tst])
    k = np.isfinite(oof) & np.isfinite(loi) & np.isfinite(hii)
    sec = CM.secondary_metrics(yi[k], oof[k])
    return {
        "isoform": iso,
        "n": int(k.sum()),
        "ST_RAE": CM.st_rae(oof[k], loi[k], hii[k], y_true=yi[k], denominator="st_mean"),
        "MAE": sec["MAE"],
        "R2": sec["R2"],
        "Spearman": sec["Spearman_R"],
    }, oof


def main() -> None:
    from tabicl import TabICLRegressor

    rows = []
    preds = {}
    t0 = time.time()
    for iso in ISO:
        r, oof = cv_one(
            lambda: TabICLRegressor(device="cuda", random_state=0, n_jobs=1), iso
        )
        r["model"] = "TabICL"
        rows.append(r)
        preds[iso] = oof
        print(
            f"{iso}: ST-RAE={r['ST_RAE']:.4f} R2={r['R2']:.3f} rho={r['Spearman']:.3f} "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )
    df = pd.DataFrame(rows)
    df.to_csv("tabicl_benchmark.csv", index=False)
    print("MA-ST-RAE:", round(float(df.ST_RAE.mean()), 4), flush=True)

    # Refit on all labelled rows per isoform and predict the blinded test set.
    test_pred = {}
    for iso in ISO:
        pt = f"{iso}_pIC50_direct_inhibition"
        y = tr[pt].values
        m = np.isfinite(y)
        mdl = TabICLRegressor(device="cuda", random_state=0, n_jobs=1)
        mdl.fit(Rtr[m], y[m])
        test_pred[iso] = mdl.predict(Rte).tolist()
        print(f"refit {iso} done [{time.time() - t0:.0f}s]", flush=True)
    json.dump(test_pred, open("tabicl_test_predictions.json", "w"))
    json.dump(
        {k: v.tolist() for k, v in preds.items()}, open("tabicl_oof.json", "w")
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
