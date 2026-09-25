"""Leak-free (nested) cross-isoform stacking benchmark.

The quick stacking run filled cross-isoform features from models trained on ALL rows
labelled for the donor isoform. Those donor rows can be scaffold-mates of a row held
out in the target isoform's fold, so donor information crossed the CV boundary and the
resulting 0.7142 MA-ST-RAE is optimistic by an unknown amount.

This script rebuilds the cross-features *inside* each outer fold: for outer fold F of
target isoform j, every donor isoform i is refit on donor rows whose scaffold group is
absent from F, then used to predict both the training and held-out portions of F. No
information from a held-out scaffold group can reach the features of that group.

Outputs nested_stack_benchmark.csv with the honest per-isoform ST-RAE.
"""

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
N_FOLDS = 5

Btr = np.load("Btr.npy")
scaf_tr = np.load("scaf_tr.npy")
tr = pd.read_parquet("tr_cache.parquet")


def main() -> None:
    from tabicl import TabICLRegressor

    def new_model():
        return TabICLRegressor(device="cuda", random_state=0, n_jobs=1)

    labels = {i: tr[f"{i}_pIC50_direct_inhibition"].values for i in ISO}
    finite = {i: np.isfinite(labels[i]) for i in ISO}

    rows = []
    t0 = time.time()
    for target in ISO:
        donors = [i for i in ISO if i != target]
        y = labels[target]
        lo = tr[f"{target}_pIC50_direct_inhibition_conf_low"].values
        hi = tr[f"{target}_pIC50_direct_inhibition_conf_high"].values
        m = finite[target]
        idx = np.where(m)[0]
        oof = np.full(len(idx), np.nan)

        for trn_pos, tst_pos in GroupKFold(N_FOLDS).split(idx, y[idx], scaf_tr[idx]):
            trn_rows, tst_rows = idx[trn_pos], idx[tst_pos]
            held_groups = set(scaf_tr[tst_rows].tolist())

            # Donor features, refit without any row from the held-out scaffold groups.
            donor_cols_trn, donor_cols_tst = [], []
            for d in donors:
                d_ok = finite[d] & ~np.isin(scaf_tr, list(held_groups))
                dm = new_model()
                dm.fit(Btr[d_ok], labels[d][d_ok])
                donor_cols_trn.append(dm.predict(Btr[trn_rows]))
                donor_cols_tst.append(dm.predict(Btr[tst_rows]))

            F_trn = np.hstack([Btr[trn_rows], np.column_stack(donor_cols_trn)]).astype(
                np.float32
            )
            F_tst = np.hstack([Btr[tst_rows], np.column_stack(donor_cols_tst)]).astype(
                np.float32
            )
            mdl = new_model()
            mdl.fit(F_trn, y[trn_rows])
            oof[tst_pos] = mdl.predict(F_tst)
            print(f"  {target} fold done [{time.time() - t0:.0f}s]", flush=True)

        k = np.isfinite(oof) & np.isfinite(lo[idx]) & np.isfinite(hi[idx])
        sec = CM.secondary_metrics(y[idx][k], oof[k])
        rows.append(
            {
                "isoform": target,
                "n": int(k.sum()),
                "ST_RAE_nested": CM.st_rae(
                    oof[k], lo[idx][k], hi[idx][k], y_true=y[idx][k],
                    denominator="st_mean",
                ),
                "MAE": sec["MAE"],
                "R2": sec["R2"],
                "Spearman": sec["Spearman_R"],
            }
        )
        print(
            f"{target}: nested ST-RAE={rows[-1]['ST_RAE_nested']:.4f} "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv("nested_stack_benchmark.csv", index=False)
    print("MA-ST-RAE (nested, leak-free):", round(float(df.ST_RAE_nested.mean()), 4))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
