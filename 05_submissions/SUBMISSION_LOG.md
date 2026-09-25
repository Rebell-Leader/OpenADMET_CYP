# Submission log

One submission per team; one upload per 4 hours; **latest valid submission counts**.
Validate every candidate against both `03_code/validate_submission.py` and the
organizers' `03_code/organizers_tutorial/{activity,tdi}_validation.py` before upload.

| # | file | uploaded (UTC) | track | model | local MA-ST-RAE | board | rank at time |
|---|---|---|---|---|---|---|---|
| v1 | `submission_regression.parquet` | 2026-08-18 19:01 | regression | LightGBM, ECFP4+RDKit-2D (2,265 feat), per isoform | 0.7632 | **0.8898** ± 0.0295 | 8 / 16 |
| v1 | `submission_classification.parquet` | 2026-08-18 | classification | LightGBM + MCC-tuned thresholds (0.18 CYP3A4, 0.48 CYP2D6) | MCC 0.408 / 0.098 | **MA-MCC 0.3306** ± 0.0553 | 1 / 6 |
| v2 | `submission_regression_v2.parquet` | 2026-08-19 07:09 | regression | TabICL on ECFP4-SVD256+desc + cross-isoform stacking | 0.7150 (leak-free) | **0.8284** ± 0.0290 | 20 / 45 (on 08-22) |

Board snapshots at each point are in `../07_leaderboards/`.

## Notes per submission

**v1 regression.** First defensible entry, shipped within 24 h of the data drop.
Statistically indistinguishable from the organizers' own LGBM baseline (z = 0.08) and
XGB baseline (z = 0.19) — i.e. we reproduced the stock baselines rather than beating
them.

**v1 classification.** Rank 1, but all six entries were statistically tied (max
z = 0.76 against rank 6), so the rank is a coin flip and not evidence of an edge.
Most of the score comes from MCC threshold tuning (+0.096 on CYP3A4). Note the local
MCCs are optimistically biased — thresholds were selected on the same out-of-fold
predictions used to report them; the board's 0.3306 is the unbiased figure.

**v2 regression.** Two changes from v1, each measured in isolation: TabICL instead
of LightGBM (−0.039 local) and cross-isoform stacking (−0.009 local, leak-free).
Observed board gain −0.0614, slightly better than the −0.048 predicted. Every
secondary metric improved. Rank fell anyway because the field advanced faster —
this same score would have ranked 4/16 on the 08-18 board.

## Final package checklist (for 2026-11-03)

- [ ] Regression predictions: exactly 750 rows, columns `SMILES`, `Molecule_Name`,
      `CYP{1A2,2C9,2D6,3A4}_pIC50_direct_inhibition`, all finite floats
- [ ] Classification predictions: exactly 750 rows, columns `SMILES`,
      `Molecule_Name`, `CYP{2D6,3A4}_is_TDI`, real booleans
- [ ] Both files pass our validator (exit 0) and the organizers' validators
- [ ] Model report / method description written (the board has a "Model Report Link"
      column we have left as "Not submitted"; open-code checkbox is worth setting,
      and the Innovation in ML award is explicitly decoupled from leaderboard rank)
- [ ] Proprietary-data disclosure: **none used** — all training data is the
      organizers' release plus public sources
- [ ] Freeze with ≥ 24 h margin; the 4-hour rate limit means a failed validation
      close to the deadline can cost the slot
| 3 | `submission_regression_v3.parquet` | not uploaded | regression | TabICL n=32, hybrid: floor-clipped screen pseudo-labels (CYP2C9/CYP3A4) + cross-isoform stacking (CYP1A2/CYP2D6) | 0.7111 | — | validated both formats; projected board ~0.83 |
| 4 | `submission_regression_v4.parquet` | not uploaded | regression | blend: multitask (shared trunk, 4 heads + aux log2fc heads, 3-seed avg) x TabICL n=32 (+floor-clipped pseudo-labels on CYP2C9/CYP3A4), nested per-isoform weights | **0.6868** | — | best local; projected board ~0.807 |
