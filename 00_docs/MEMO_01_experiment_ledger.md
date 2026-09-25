# Memo 1 — Experiment ledger: every approach tried, and what it scored

Team `avaliev`. Last updated **2026-08-22**. Challenge closes 2026-11-03;
intermediate deadline 2026-09-24.

This is the complete record of what we have tested, including the failures — the
negative results are the more useful half, because each one closes a direction and
several were things we expected to work.

**Reading the numbers.** Two scales appear and they are not interchangeable:

* **Local** = MA-ST-RAE under 5-fold Murcko-scaffold-grouped `GroupKFold` on the
  4,905-compound training set.
* **Board** = MA-ST-RAE on the 750-compound blinded test set, as scored by the
  organizers.

We now have two submissions scored on both scales, so the conversion is measured
rather than guessed: **local CV runs 0.113–0.127 optimistic.** Use ≈ +0.12 to
forecast a board score from a local one.

---

## 1. Scoreboard of everything tested

| # | approach | local MA-ST-RAE | board | verdict |
|---|---|---|---|---|
| A1 | LightGBM, ECFP4+RDKit-2D (2,265 feat), per isoform — **v1 submission** | 0.7632 | **0.8898** | shipped 08-18 |
| A2 | TabICL, ECFP4-SVD128 + descriptors (345 feat) | 0.7243 | — | −0.039 vs A1 |
| A3 | TabICL, ECFP4-SVD256 + descriptors (473 feat) | 0.7239 | — | best single model |
| A4 | TabICL, RDKit descriptors only (217 feat) | 0.7418 | — | worse than A3 |
| A5 | **A3 + cross-isoform stacking** — **v2 submission** | 0.7150¹ | **0.8284** | shipped 08-19, **best** |
| A6 | LightGBM + 33,008 public compounds (weight 0.3) | 0.8202 | — | **rejected**, hurts |
| A7 | Variance-expansion recalibration of A1 | 1.0041 | — | **rejected**, hurts badly |
| A8 | CheMeleon frozen embeddings (2,048) + LightGBM | 0.7956 | — | **rejected** |
| A9 | CheMeleon frozen, SVD-256 + TabICL | 0.7617 | — | **rejected** |
| A10 | CheMeleon-SVD256 ⊕ ECFP4-SVD256 ⊕ desc + TabICL | 0.7324 | — | **rejected** |
| A11 | 7 hand-built CYP2D6 pharmacophore features, LightGBM | RAE 1.088² | — | **rejected**, sub-no-skill |
| A12a | TabICL `n_estimators` = 1 | 0.7388 | — | ensembling matters |
| A12b | TabICL `n_estimators` = 8 (our default) | 0.7239 | — | −0.015 vs n=1 |
| A12c | TabICL `n_estimators` = 32 | **0.7226** | — | only −0.0013 more; **saturated** |
| A13 | TabICL on full-width 2,265 features (no SVD) | 0.7611 | — | **rejected**, worse |
| A14 | TabPFN as a second in-context learner | — | — | **blocked**: licence gate |

¹ leak-free nested estimate; the naive estimate was 0.7142 and the leak measured 0.0008.
² plain RAE on CYP2D6 alone, not MA-ST-RAE — the comparison point is A1's 0.981 on that isoform.

**Classification** (separate track, one submission): LightGBM on the released
`is_TDI` labels with MCC-tuned thresholds → board **MA-MCC 0.3306, rank 1 of 6**,
though all six entries were statistically tied.

---

## 2. What worked, and by how much

### 2.1 TabICL instead of gradient boosting (−0.039 local)

The single largest gain we found. Chosen from leaderboard evidence rather than
preference: on the 2026-08-18 board, `TabICL-baseline` (0.6755) was the **only**
entry in its own statistical tier, with rank 2 sitting 2.69 σ behind — the only
significant gap anywhere in the top ten.

Wins on **all four isoforms** against LightGBM with identical features and folds
(−0.042, −0.029, −0.051, −0.033), in 82 s of GPU time.

The effect **reproduces on a completely different feature family**: CheMeleon
embeddings go 0.7956 under LightGBM → 0.7617 under TabICL (−0.034). Confirmed
twice on disjoint representations, which makes it the most transferable finding we
have.

### 2.2 Cross-isoform stacking (−0.009 local)

Motivated by a structural asymmetry in the data: the training matrix is sparse —
3,596 compounds carry exactly one isoform's label and only **41** carry all four —
while the test set is dense across all four. So at test time every compound has
all four isoform predictions available; in training almost none do. Giving each
isoform's model the *other three* isoforms' predictions as features exploits that.

Helps three of four isoforms (CYP2C9 −0.018, CYP2D6 −0.017, CYP3A4 −0.009; CYP1A2
+0.005).

**We audited this for leakage and it survived.** The donor cross-features came from
models trained on all rows carrying the donor label, and those rows can be
scaffold-mates of a row held out in the target isoform's fold — 52–74 % of
cross-feature values came from that path. The nested version
(`run_nested_stack.py`) refits every donor with the held-out scaffold groups
deleted first. Leak: **0.0008** (0.7142 → 0.7150). Negligible, because donor
isoforms are labelled on largely non-overlapping compound sets.

### 2.3 MCC threshold tuning (+0.096 MCC on CYP3A4)

Where most of the classification result comes from. Optimal thresholds are far
below 0.5 (0.18 for CYP3A4, 0.48 for CYP2D6) because MCC on an imbalanced problem
rewards recall more than the default split does. Caveat: thresholds were selected
on the same out-of-fold predictions used to report MCC, so the local 0.408/0.098
are optimistically biased; the board's 0.331 is the unbiased figure.

### 2.4 Recovering the organizers' TDI labelling rule

Not a model improvement, but it is why the classification submission was correct on
day one. Our pre-launch labeler reproduced the released labels on **5,080 of 5,081
rows** — but only after empirically pinning down two cases the published FAQ leaves
undefined, both of which resolve toward *negative*:

* the band `direct < 4` with `4 ≤ tdi ≤ 4.301` is negative, not excluded (18/18
  CYP2D6, 159/159 CYP3A4);
* a **missing direct arm** is negative regardless of the TDI-arm value — 1,249/1,249
  CYP3A4 rows, including 1,055 whose TDI-arm pIC50 exceeds 4.301, up to 7.54.

The second matters for training: 34.8 % of CYP3A4 labels have no direct-arm DRC and
are called negative. Treating them as inferred positives would move the positive
rate from 21.3 % to 50.8 %. We train against the organizers' convention because
that is what scores, but a model learning "measurable TDI with an absent direct arm
⇒ no TDI" is learning an artifact of data availability. Still an open lever (Memo 3).

---

## 3. What failed, and why it is worth knowing

### 3.1 Public data does not transfer (A6)

We assembled a 37,006-compound public corpus (ChEMBL / PubChem / Octant) before
launch. Two independent tests, both negative:

* **Pre-launch:** a public-only LightGBM scored R² 0.117, ρ 0.472, ST-RAE 0.870 on
  the 1,084-compound Octant CYP3A4 set with real credible intervals — barely better
  than the 1.012 mean-baseline. Recalibration did not rescue it (oracle-linear
  0.871; oracle rank-mapping made it *worse* at 1.108).
* **Post-launch:** adding 33,008 leakage-free public compounds (InChIKey14-matched
  against both challenge splits; 5 test and 297 train collisions removed) at sample
  weight 0.3 *degrades* local MA-ST-RAE from 0.7632 to 0.8202, hurting three of
  four isoforms — CYP2C9 worst at +0.156.

One contributing cause is identifiable: the best public dataset (Octant) uses assay
protocol C3A4IAP with 30-min active-enzyme pre-incubation, so its pIC50 maps to the
challenge's **+NADPH/TDI arm**, not the direct arm that Track 1 scores. Public CYP
data needs domain adaptation or arm-aware handling, not concatenation.

### 3.2 Variance-expansion recalibration (A7)

v1 predictions carried only 0.51–0.91 of the label spread, which looked like
fixable miscalibration. Testing `pred = mean + (pred − mean)·α` over α ∈ [1.0, 2.0]:
the optimal α is **1.00 on every isoform**, and the no-oracle variance-matching
choice (α = 1.40–2.16) degrades MA-ST-RAE from 0.7632 to 1.0041.

The reason is structural: under an L1-type metric, shrinkage toward the conditional
median is *correct*. Prediction-spread compression is not a bug here. **Do not
revisit this.**

### 3.3 Frozen CheMeleon embeddings (A8–A10)

We downloaded the encoder (`chemeleon_mp.pt`, md5 verified), ran it frozen with
mean aggregation to give 2,048-dim graph embeddings, and scored three
representations under the identical protocol:

| representation | head | local MA-ST-RAE |
|---|---|---|
| ECFP4 + descriptors | LightGBM | 0.7632 |
| CheMeleon (2,048) | LightGBM | 0.7956 |
| **ECFP4-SVD256 + descriptors** | **TabICL** | **0.7239** |
| CheMeleon-SVD256 | TabICL | 0.7617 |
| both concatenated | TabICL | 0.7324 |

Frozen CheMeleon loses under both heads, and concatenating both loses to
ECFP4+descriptors alone. The **one** winning cell in the whole matrix is CYP2D6
under LightGBM (0.9796 → 0.9472).

**Limitation, stated plainly:** we used CheMeleon *frozen*. The organizers
fine-tune it end-to-end (`from_chemeleon: true`, 3 FFN layers of 1,024, 100 epochs
with early stopping). This rules out frozen embeddings as drop-in features, not
CheMeleon fine-tuned. Their own fine-tuned CheMeleon baseline scored 0.8335 against
their TabICL baseline's 0.6755, so the direction of the evidence is consistent.

### 3.4 Hand-built CYP2D6 pharmacophore features (A11)

Seven explicit descriptors — basic-N and quaternary-N counts, bond-path distances
from basic N to aromatic systems, logP, aromatic rings — score RAE **1.088** on
CYP2D6 alone, *worse than predicting the mean*, against the full feature set's
0.981. The basicity signal is real (see Memo 2) but weak and already captured.

### 3.5 No metric exploit at the assay floor

Checked pre-launch and re-checked on real intervals: blanket mid-range guessing on
sub-floor compounds scores MA-ST-RAE 0.74 against 0.21 for honest prediction.
Predicting inactives properly is correct. The 23× credible-interval widening below
pIC50 4 is real but not exploitable.

### 3.6 A split protocol we built and could not use

`cyp_splits.py` was written before launch to reproduce the test geometry —
`parent_in_train_folds`, holding out analogs of a potent parent that stays in
training. On the real data it **returns empty folds**: only 16 % of training
compounds fall in a multi-member analog series at T = 0.55. Training is a diversity
library (internal NN Tanimoto median 0.450, 17 % ≥ 0.55) while test is analog-dense
(median 0.656, 85 % ≥ 0.55).

This was a genuine pre-launch modelling error — we assumed both splits shared the
analog-series structure the *test* set was described as having. The scaffold-grouped
fallback is defensible and probably pessimistic (each test compound has a training
analog at median T = 0.587), but it is a surrogate.

---

## 4. Two diagnoses that were wrong, and their corrections

Recorded because both looked convincing and each would have sent us down a dead end.

**"The RAE denominator shrank."** First explanation for the v1 local→board gap
(0.7632 → 0.8898). Backwards. MAE rose 73 % (0.5777 → 0.9987) while ST-RAE rose
only 0.1265, because the test label distribution is wider than the training one
(training macro mean absolute deviation **0.718**, computed exactly; test **≈ 1.064**,
from the `MA-MAE / MA-ST-RAE` proxy of 1.122 de-biased by the 1.055× upward bias that
ratio carries — the proxy's denominator is soft-threshold error, which is zero inside
the credible interval and so ≤ raw absolute error). The *larger* test denominator
cushioned the score — about 48 % wider, not the 56 % the raw proxy suggests. The
test set is simply harder and wider-spread — which its construction (75 potency hits
plus analogs) implies directly.

**"We rank well but predict magnitudes poorly."** Suggested by percentile ranks
(56th on Spearman, 50th on ST-RAE) and it would have justified a rank-then-calibrate
two-stage design. Tested properly against the field median: z = +0.64 on Spearman,
z = −0.04 on ST-RAE. Both within noise. No real asymmetry.

---

## 5. v1 → v2: what the improvement actually bought

| metric | v1 | v2 | delta |
|---|---|---|---|
| MA-ST-RAE | 0.8898 ± 0.0295 | **0.8284 ± 0.0290** | **−0.0614** |
| MA-MAE | 0.9987 | 0.9669 | −0.0318 |
| MA-R² | 0.0582 | 0.1160 | +0.0579 |
| MA-Spearman ρ | 0.6132 | 0.6358 | +0.0225 |
| MA-Kendall τ | 0.4443 | 0.4685 | +0.0242 |

Every metric moved the right way, and the observed −0.061 slightly **exceeded** the
−0.048 our leak-free local CV predicted.

**On significance:** the unpaired z is 1.48 (p = 0.14), but that is the wrong test
and it is conservative. Both entries were scored on the same bootstrap resamples of
the same test set, and our v1/v2 predictions correlate 0.79–0.93, so the true
paired-difference standard deviation is far below `sqrt(s1²+s2²)`. We cannot compute
the paired delta without the organizers' per-resample scores. The corroboration is
that the sign and magnitude match the prior local prediction.

**Rank moved the wrong way anyway.** 8/16 → 20/45, because the field advanced faster
than we did: board best 0.6755 → 0.5209, board median 0.8909 → 0.8402, and 38 of 45
entries postdate our v2 submission. Our v2 score would have ranked **4/16** on the
08-18 board. The improvement was real; it was outpaced.

---

## 5b. Auxiliary-data experiments — 2026-08-22

Prompted by Memo 3 §3.4 (dense auxiliary labels as the top-ranked remaining lever).
Both mechanisms were tried; they behave completely differently, and the difference is
the most transferable thing learned this round.

| # | approach | nested MA-ST-RAE | vs TabICL n=32 (0.7226) | verdict |
|---|---|---|---|---|
| A15 | Screen **distillation** (predicted log2fc as a feature) | 0.7584 | **+0.036** | rejected |
| A16 | Screen **pseudo-labels**, all isoforms, linear map | see §5b.2 | catastrophic on 2 of 4 | rejected |
| A17 | Screen pseudo-labels, **floor-clipped**, a-priori isoform rule | **0.7144** | **−0.008** | **adopted** |
| A18 | A17 + cross-isoform stacking together | 0.7199 | −0.003 | rejected (competes) |
| A19 | **Hybrid**: pseudo where the rule fires, stacking elsewhere | **0.7111** | **−0.012** | **shipped as v3** |

### 5b.1 Distillation fails, and the leaky version looked like a breakthrough

The single-concentration screen is dense (4,376 compounds × 4 enzymes) where the
dose-response file is sparse, and its observed log2 fold-change correlates with
same-isoform DRC pIC50 at Spearman **−0.862 / −0.896 / −0.828 / −0.936**. But **zero**
of the 750 blinded test compounds appear in the screen (checked by both
`Molecule_Name` and `SMILES`), so the observed value is unavailable at test time and
the only route is to predict it.

Predicting it destroys the value. Structure → log2fc is learnable at scaffold-grouped
R² 0.30 / 0.49 / 0.31 / 0.50 — *better* than structure → pIC50 (mean R² 0.400 vs
0.351), which is why the idea looked promising. But the resulting feature is a
deterministic function of the same `Btr` the pIC50 model already receives, so no new
information enters; only a lossy re-encoding that costs capacity. Nested result:
**+0.036 worse on all four isoforms**.

**The near-miss is the important part.** The naive version — donors fit on all rows,
so each donor had seen the held-out compound's own measured log2fc — scored ST-RAE
**0.2771** with R² **0.834** on CYP1A2, against a nested 0.8342. That is a **0.557**
leak masquerading as a 0.53 improvement, and it would have looked like winning the
challenge outright. The measured log2fc is a near-perfect surrogate for the same
compound's pIC50 (|ρ| 0.86), so a donor that has seen it is effectively handing over
the answer.

Contrast v2 cross-isoform stacking, where the leak was 0.0008 and the gain survived:
there the donors were trained on **different compounds** (other isoforms' labels on
largely non-overlapping sets) and so genuinely added information.

> **Generalisable lesson.** A dense auxiliary label measured on the *same compounds*
> as the target adds nothing once it must be predicted from structure. Auxiliary data
> helps when it covers compounds the target labels do **not**.

### 5b.2 Pseudo-labelling: the lesson applied, and a censoring bug found

That lesson points at the other mechanism. For any isoform, ~2,900 compounds have a
screen reading but **no fitted DRC** (CYP1A2 2,963 / CYP2C9 3,090 / CYP2D6 2,882 /
CYP3A4 2,570) — exactly the "compounds the target does not cover" case. Convert their
log2fc to a pseudo-pIC50 via a within-fold linear calibration and add them as extra
**training rows**, not features.

First attempt was catastrophic on two isoforms (ST-RAE 1.42 CYP1A2, 2.11 CYP2D6
against baselines of 0.86 and 0.96). Diagnosis: the linear map **extrapolates below
the assay floor**. Compounds without a fitted curve are inactive, so their true pIC50
is *censored* at 4.0, not a point value near 2.9. Median pseudo-label shifts were
−1.80 (CYP1A2), −0.34 (CYP2C9), −1.83 (CYP2D6), −0.06 (CYP3A4), and CYP2D6's
pseudo-labels were **100 %** sub-4. The shift ranks the damage perfectly (Spearman
−1.00 against the per-isoform delta).

Clipping pseudo-labels at the floor — matching how the challenge censors its own
data — turns two of them positive: CYP2C9 −0.083, CYP3A4 −0.064 (LightGBM head).

**The isoform-selection rule is a priori, not fold-peeked.** Apply pseudo-labels only
when `|median(clipped pseudo) − median(real)| < 0.5`, computable from training labels
alone with no CV scores. It selects {CYP2C9, CYP3A4} — independently reproducing the
CV-optimal split. On the shipping TabICL head this gives **0.7144** (CYP2C9
0.6550→0.6364, CYP3A4 0.5023→0.4879; the other two untouched by construction).

### 5b.3 The two gains compete rather than compose

Measured, not assumed:

| configuration | MA-ST-RAE | gain vs baseline |
|---|---|---|
| TabICL n=32 | 0.7226 | — |
| + stacking only | 0.7150 | −0.0076 |
| + pseudo only | 0.7144 | −0.0082 |
| + **both** | 0.7199 | −0.0027 |
| naive sum of the two | — | (−0.0158) |

Combining them recovers barely a third of either gain alone. Both inject the same
*kind* of information — an estimate of the compound's behaviour in a related assay
readout — so the second is largely redundant and costs model capacity. Had we assumed
additivity we would have shipped the worst of the three.

The **hybrid** (pseudo where the rule fires, stacking on the other two isoforms)
reaches **0.7111**. Its partition is a priori; its fallback rests on v2's prior
measurement that stacking helped 3 of 4 isoforms. It is still the most
selection-influenced number in this table — treat the honest band as **0.711–0.723**,
and pseudo-only (0.7144) as the safest single-policy alternative.

### 5b.4 v3 submission built and validated

`05_submissions/submission_regression_v3.parquet` (+ `.csv`), 750 rows, hybrid policy,
all four columns finite. Passes our validator and — as CSV — the organizers' own
`activity_validation.validate_activity_submission` with all 750 IDs matched.

**Trap found:** the organizers' validator hardcodes `pd.read_csv` (line 31 of
`activity_validation.py`), so it rejects a parquet file with a UTF-8 decode error even
though the Space itself scored our parquet v2 without complaint. Always validate the
**CSV** copy locally, and keep both formats in `05_submissions/`.

Projected board score, using the calibrated +0.12 local→board offset: **≈ 0.83**.
That is barely distinguishable from v2's actual 0.8284, so v3 is a marginal
improvement at best — the structural gap to the 0.521 frontier is untouched.

---

## 5c. Multitask network and the two-family blend — 2026-08-22 (Memo 3 §3.3)

**This is the largest gain of the project: local MA-ST-RAE 0.7111 → 0.6868 (−0.024).**

| # | approach | MA-ST-RAE | vs TabICL n=32 | verdict |
|---|---|---|---|---|
| A20 | single-task MLP ×4 (architecture control) | 0.7422 | +0.020 | control only |
| A21 | multitask, CYP2D6 fitted apart | 0.7266 | +0.004 | superseded |
| A22 | multitask, all four shared | 0.7322 | +0.010 | superseded |
| A23 | A21 + auxiliary log2fc heads | 0.7199 | −0.003 | superseded |
| A24 | **A22 + auxiliary log2fc heads** | **0.7175** | **−0.005** | best single MT |
| A25 | **blend(A24, TabICL+pseudo), nested weights** | **0.6868** | **−0.036** | **shipped as v4** |

Code: `03_code/multitask.py` (module), `03_code/run_multitask.py` (benchmark).

### 5c.1 Why sharing was worth trying after pseudo-labelling returned only −0.008

The label matrix is 33.3 % filled: of 4,905 compounds, **3,596 carry exactly one**
pIC50 label and **only 41 carry all four**. Four independent fits each see
1,285–2,335 rows; a masked-loss trunk sees all 4,905 and lets every row shape the
shared representation. Meanwhile the screen fills **89.2 %** of the same matrix, so it
can supervise extra heads.

That auxiliary route is a **third mechanism**, distinct from both tested in §5b:

| mechanism | result | why |
|---|---|---|
| screen as a **feature** (distillation) | +0.036 | a structure-derived surrogate re-encodes input the model already has |
| screen as extra **rows** (pseudo-labels) | −0.008 | genuinely enlarges the labelled compound set |
| screen as an auxiliary **task** (here) | −0.007 to −0.015 | shapes the trunk during training, never enters at inference |

The auxiliary task cannot fail the way distillation did, because it never appears as
an input at prediction time — it only supplies gradient.

### 5c.2 Three ablations, each a controlled comparison

1. **Multitask beats single-task at equal architecture.** `st_mlp` 0.7422 → `mt_3`
   0.7266 (**−0.0156**). Same trunk width, same folds — so this isolates *sharing*,
   not "a neural net helped".
2. **The auxiliary task helps consistently**: −0.0068 on the 3-way trunk, −0.0147 on
   the 4-way trunk.
3. **The CYP2D6 exclusion prediction holds — then reverses.** Memo 2 §2 predicted
   CYP2D6 should be kept out of the shared trunk (orthogonal SAR, mean cross-isoform
   ρ = −0.002). Without the auxiliary task that is right: `mt_3` beats `mt_all4` by
   0.0056. **With** the auxiliary task it flips: `mt_all4_aux` beats `mt_3_aux` by
   0.0023. Interpretation: the dense log2fc supervision gives the trunk enough
   isoform-specific signal that CYP2D6 no longer has to fight the other three for
   representation. This is a genuine update to the Memo 2 recommendation — the
   orthogonality finding stands, but its architectural consequence does not survive
   the addition of dense auxiliary supervision.

### 5c.3 The blend, and what actually drives it

Per isoform the two families are strongly complementary: multitask wins CYP2C9 by
**0.051** and CYP3A4 by 0.011, TabICL wins CYP1A2 by 0.022 and CYP2D6 by 0.020.

Blend weight is chosen by **nested inner CV** — for each outer fold, `w` is swept on
the other folds only and then applied to the held-out rows — so the reported number is
not fold-peeked. Selected weights (multitask share): CYP1A2 0.39, CYP2C9 0.62,
CYP2D6 0.41, CYP3A4 0.47. All interior; no degenerate collapse to either parent.

**The blend beats BOTH parents on EVERY isoform**, including the two where multitask
alone was worse:

| isoform | TabICL(+pseudo) | multitask | blend | vs better parent |
|---|---|---|---|---|
| CYP1A2 | 0.8099 | 0.8317 | **0.7939** | −0.016 |
| CYP2C9 | 0.6550 | 0.6044 | **0.5904** | −0.014 |
| CYP2D6 | 0.9232 | 0.9427 | **0.9012** | −0.022 |
| CYP3A4 | 0.5023 | 0.4914 | **0.4617** | −0.030 |

**Correction to a claim made mid-analysis.** I first attributed the blend gain to
decorrelated errors, calling the error correlations "well below 1". They are
**0.888–0.938 — high**. The gain is not decorrelation; it is ordinary variance
reduction from averaging two different function classes with a tuned weight. Beating
both parents everywhere is the signature of that, not of one model dominating.

This also does **not** contradict §5b.3's "gains compete" finding. That concerned two
routes injecting the *same* information (screen-derived estimates) into one model.
This is two *different function classes* fit to the same data — a different situation,
and the pseudo-label policy still adds on top of the blend (0.6895 → 0.6868).

### 5c.4 v4 submission

`05_submissions/submission_regression_v4.parquet` (+ `.csv`). Multitask component is
seed-averaged over 3 seeds, refit on all labelled rows; TabICL component carries
floor-clipped pseudo-labels on CYP2C9/CYP3A4; blended at the nested weights above.
750 rows, all finite, passes our validator and the organizers' CSV validator with all
750 IDs matched.

Projected board score at the calibrated +0.12 offset: **≈ 0.807**, against v2's actual
0.8284 and the current frontier of 0.521.

**Caveat.** The weights are nested, but the *decision* to blend these two particular
families was made after seeing both benchmarks. The honest reading is that 0.6868 is
the best-supported local number we have, carrying the same ±0.01 calibration
uncertainty as every other local→board projection here.

---

## 6. v3 experiments — completed 2026-08-22

Full per-isoform results in `04_experiments/v3_benchmark.csv`.

### 6.1 Ensembling saturates at n_estimators = 8 (A12)

| isoform | n=1 | n=8 | n=32 |
|---|---|---|---|
| CYP1A2 | 0.8199 | 0.8081 | 0.8099 |
| CYP2C9 | 0.6827 | 0.6585 | 0.6550 |
| CYP2D6 | 0.9316 | 0.9244 | 0.9232 |
| CYP3A4 | 0.5213 | 0.5046 | 0.5023 |
| **macro** | **0.7388** | **0.7239** | **0.7226** |

Ensembling is worth −0.0149 going from 1 to 8, and only a further **−0.0013** going
from 8 to 32 — at 4× the runtime, and CYP1A2 actually gets slightly worse. The
default we already used is at the knee of the curve. **This lever is spent.**

### 6.2 Full-width features are worse, not better (A13)

TabICL on the full 2,265-dimensional representation (2,048-bit ECFP4 + 217
descriptors, no SVD) scores **0.7611** against 0.7239 for SVD-256 — worse on three
of four isoforms, CYP3A4 worst (0.5836 vs 0.5046). The run also triggered repeated
CUDA OOM warnings on the 12 GB GPU (attempting 900 MB allocations against ~600 MB
free), so it is both slower and less accurate.

This settles a question left open after v2: **SVD compression is helping, not
costing us.** It is consistent with TabICL's pretraining regime being
modest-dimensional tabular data. The earlier observation that SVD-128 and SVD-256
differ by only 0.0004 was not "compression is irrelevant" — it was "we are already
in the right regime, and leaving it hurts."

### 6.3 TabPFN is licence-blocked (A14)

TabPFN 3.x will not run offline: `fit()` aborts before touching the data, requiring
registration at `ux.priorlabs.ai`, licence acceptance, and a `TABPFN_TOKEN`
environment variable. We have no such credential and it is not an anonymous
download.

This matters because TabPFN was the cheapest way to test whether the gain is
*in-context learning as a family* or *TabICL specifically* — the two disjoint
feature families where TabICL beat GBM (−0.039 on ECFP4, −0.034 on CheMeleon) cannot
distinguish those. If a token is obtainable the test costs minutes; otherwise the
question stays open, and the architecture finding should be stated as "TabICL beats
GBM here", not "in-context learners beat GBM here".

### 6.4 Consequence for the plan

The three cheapest accuracy levers are now closed (ensembling saturated, full-width
worse, TabPFN blocked). Best local configuration remains **v2 + n_estimators=32 at
0.7226 − 0.009 stacking ≈ 0.714**, essentially identical to the 0.7150 already
submitted. There is no incremental-tuning path to the leaderboard frontier from here:
reaching rank 1's 0.521 needs a local ≈ 0.40 (Memo 3 §6), and we are at 0.715.

Priority therefore shifts entirely to the two structural levers in Memo 3 — the
dense auxiliary labels (§3.4) and a multitask architecture to exploit them (§3.3) —
rather than further tuning of the current pipeline.
