# MEMO 06 — P1: the rebuilt regression stack

**Date:** 2026-09-28 · **Instrument:** analog-covered compound-level folds (`03_code/p1_common.py`)
**Result:** nested convex stack **0.6408** macro ST-RAE, against **0.7434** for the fingerprint
floor and **0.790** for the entry that was on the board this morning.
**Submission written:** `05_submissions/submission_regression_v5.{parquet,csv}` — passes the
organizers' `activity_validation.py` with zero errors.

---

## 1. The instrument, restated

MEMO 05 §3 established that both our old fold schemes validated at nearest-neighbour
Tanimoto ≈ 0.40 while the blind set sits at 0.587. P1 therefore runs on new folds:

* **Compound-level**, not per-endpoint — the multitask learners see all four labels of a
  compound at once, so a per-endpoint split would leak a compound between its own tasks.
* **Analog-splitting**: single-linkage clusters at 0.55 are distributed round-robin across
  the five folds, so a scored compound has its analogues in the training part.
* **Scored subset** = compounds whose nearest neighbour in the training part is ≥ 0.50
  (n = 1427, median 0.567 — the closest match to the blind set's 0.587 the library allows).
  Everything is also reported on all compounds (`[all]`, median 0.44) as the pessimistic
  reference. Per-endpoint n on the scored subset is 307 / 387 / 273 / 925, so differences
  below ~0.02 macro should not be read as real.

## 2. Base learners

Eight members, chosen for *representation* diversity rather than capacity. Macro ST-RAE on
the analog-covered subset (all-compound score in brackets):

| member | what it is | macro ST-RAE |
|---|---|---|
| `lgbm_prot` | **LightGBM on the QUACPAC protomer** + charge bookkeeping | **0.7103** (0.7357) |
| `chemprop_pre` | Chemprop D-MPNN, **pretrained on the 37k public corpus**, fine-tuned | 0.7170 (0.7435) |
| `tabicl_svd` | TabICL over ECFP-SVD256 + descriptors | 0.7172 (0.7231) |
| `lgbm_fp` | LightGBM on ECFP4 counts + RDKit descriptors (the floor) | 0.7434 (0.7568) |
| `chemprop_prot` | Chemprop on the protomer graph | 0.7600 (0.7938) |
| `chemprop_mt` | Chemprop D-MPNN, masked multitask, no transfer | 0.7626 (0.7925) |
| `chemberta` | frozen ChemBERTa-77M embeddings + LightGBM | 0.7933 (0.7965) |
| `public_lgbm` | trained **only** on public data, never on challenge labels | 1.0596 (1.1100) |
| **convex stack (nested)** | weights ≥ 0, sum to 1, fitted per endpoint | **0.6408** (0.6620) |

Two findings worth keeping:

**Protonation state is worth 0.033 macro on its own.** `lgbm_prot` differs from `lgbm_fp`
only in that the molecule is featurised as its predominant protomer at physiological pH
(OpenEye QUACPAC; 1,026 compounds carry +1, 209 carry −1) with net charge, protonated-N
count and basic-N count appended. It improves three endpoints of four
(CYP1A2 0.778 → 0.723, CYP2C9 0.769 → 0.715, CYP2D6 0.956 → 0.928; CYP3A4 flat) and is the
single best member. This was the MEMO 05 §P2 hypothesis for the CYP2D6 hole and it is
confirmed in direction, if not in magnitude.

**Public data transfers as pretraining, not as rows.** Same architecture, same folds:
`chemprop_mt` 0.7626 → `chemprop_pre` 0.7170 with a public-corpus pretraining stage,
a −0.046 gain. MEMO 01 recorded the same corpus *hurting* by +0.057 when mixed in as extra
labelled rows. The corpus was never the problem; the mechanism was. `public_lgbm` — trained
on public data alone — is a poor predictor of the challenge assay (1.06), and yet the stack
still gives it 6–15 % weight on CYP1A2 and CYP2D6: it carries information the
challenge-trained members do not.

## 3. Stacking

Per endpoint, weights ≥ 0 summing to 1, fitted by SLSQP on the out-of-fold matrix. The
reported stack score is **nested** — weights fitted on four folds and applied to the fifth —
so it is not the in-sample optimum of the combiner. Non-negativity is deliberate: the rank-28
public write-up reports an unconstrained ridge combiner producing cancelling ±1.9/−0.6
coefficients that transferred badly, and our own v2→v3 board regression came from trusting a
combiner that looked better in CV.

| endpoint | stack | best single member | v3 on the board |
|---|---|---|---|
| CYP1A2 | 0.6597 | 0.7119 | 0.7303 |
| CYP2C9 | 0.6456 | 0.7016 | 0.5728 |
| CYP2D6 | **0.8495** | 0.8865 | 1.2774 |
| CYP3A4 | 0.4082 | 0.4572 | 0.5807 |
| **macro** | **0.6408** | 0.7103 | 0.7903 |

Weights (`04_experiments/p1_stack_weights.csv`) are spread over four to six members on every
endpoint; `chemberta` is the only member zeroed everywhere. The largest single weight is
carried by `lgbm_prot` on CYP1A2 (0.421) and CYP2D6 (0.287), by `chemprop_pre` on CYP2C9
(0.428), and by `tabicl_svd` on CYP3A4 (0.264) — no member dominates the set.

CYP2C9 is the one endpoint where the stack is *worse* than the v3 board number (0.646 vs
0.573). The v3 pipeline is not recoverable as a member (it was not built on these folds), so
this is a comparison across instruments, not a like-for-like regression — but it is the reason
to prefer the stack on the macro and to keep watching CYP2C9 on the board.

## 4. A leak found and fixed

The first `chemprop_pre` run scored 0.362 / 0.406 / 0.598 / 0.295 — better than the field
leader — and the stack collapsed onto it with weight 1.0. That is a leak signature, not a
result. Cause: the pretrained model object was passed to each fold's fine-tune and
`_chemprop_fit` mutates the message-passing trunk **in place**, so fold *k*'s model inherited
weights already fine-tuned on fold *k*'s validation compounds through the previous folds'
training sets. Fixed by deep-copying the trunk per fold (`run_p1.py`); the honest number is
0.7170. Nothing tainted was submitted.

The general lesson for the remaining weeks: any member that beats the field leader on our own
instrument is a bug until proven otherwise.

## 5. What P1 does not fix

* **CYP2D6 remains the hole** — 0.8495 against a field best of 0.4568. Protonation helped;
  it is not enough. Next: explicit basic-centre/aromatic-stacking pharmacophore distances,
  and a CYP2D6-only error analysis of which compounds drive the residual.
* **Shrinkage persists** — stack prediction std / label std is 0.55 / 0.60 / 0.50 / 0.78.
  MEMO 05 §3d showed rescaling makes ST-RAE worse, so this stays as it is.
* **P4 (blind-neighbourhood retrieval) is untouched** — still the one ingredient of the
  rank-28 recipe we have never tried.
* **No 3D member yet.** OMEGA + ROCS shape/electrostatic similarity is the obvious next
  member with a genuinely different inductive bias, and the analog regime is where it should
  pay. Not run here because conformer generation for 5,655 compounds is a job of its own.

## 6. Reproduction

```
python 03_code/build_protomer_features.py        # cyp-oe env, OE_LICENSE set
python 03_code/run_p1.py <member>                # lgbm_fp lgbm_prot tabicl_svd chemberta
                                                 # chemprop_mt chemprop_prot chemprop_pre public_lgbm
python 03_code/run_p1_stack.py --submit
```

Artefacts: `04_experiments/p1/*.npz` (per-member OOF + blind predictions),
`04_experiments/p1_benchmark.csv`, `04_experiments/p1_stack_weights.csv`,
`06_figures/p1_stack_benchmark.png`.
