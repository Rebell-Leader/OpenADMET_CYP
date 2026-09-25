# Strategy, v1 post-mortem, v2 direction, and further work

Team `avaliev`. Written 2026-08-18, one day into a challenge that closes
**2026-11-03**, with an intermediate deadline of **2026-09-24**. Companion to
`METHODS.md`, which carries the reproducible detail.

---

## 1. Where we stand

| track | our score | rank | statistically tied with us | the real story |
|---|---|---|---|---|
| Regression | MA-ST-RAE 0.8898 ± 0.0295 | **8 / 16** | LGBM-baseline (z=0.08), XGB-baseline (z=0.19), Asidsal11, JustTheBasics, NorthCoil | we reproduced the organizers' stock GBM baselines, we did not beat them |
| Classification | MA-MCC 0.3306 ± 0.0553 | **1 / 6** | **all five others** (max z=0.76) | rank 1 in a six-way tie — a coin flip, not an edge |

Both ranks are misleading in opposite directions, and it matters for where effort
goes. Reading rank 1 in classification as a win would lead us to defend a position
that does not exist; reading rank 8 in regression as a mid-field result understates
that we are statistically indistinguishable from a stock LightGBM.

The one genuinely informative feature of the leaderboard is at the top:
**`TabICL-baseline` (0.6755) is alone in its statistical tier**, with rank 2 sitting
2.69 σ behind. That is the only significant gap anywhere in the top ten, and it is
the single strongest piece of evidence available about what works on this task.

## 2. v1 post-mortem — what went wrong, and what merely looked wrong

### 2.1 The headline gap

Local scaffold-grouped CV said MA-ST-RAE 0.7632; the leaderboard said 0.8898.

**Diagnosis (verified):** the test set is absolutely harder and wider-spread than
the training set. MAE rose 73 % (0.5777 → 0.9987) while ST-RAE rose only 0.1265,
because the test label distribution is wider than the training one (training macro
mean absolute deviation 0.718 exactly; test ≈ 1.064, estimated from
`MA-MAE / MA-ST-RAE` = 1.122 de-biased by the 1.055× inflation that ratio carries —
see METHODS.md §5.2). The larger RAE denominator *cushioned* the score. This follows
directly from the published construction — the test set is 75 potency hits plus
their analogs, so it is enriched in potent compounds and spans more pIC50 range than
a diversity library.

**Correction to an earlier claim of ours:** we first attributed the gap to a
*shrinking* denominator. That was backwards, and the arithmetic above is the
refutation. Recorded here because the wrong version would have sent us hunting a
non-existent normalization problem.

### 2.2 The split protocol we built could not be used

`cyp_splits.py` was written before launch to reproduce the test geometry —
`parent_in_train_folds`, holding out analogs of a potent parent that stays in
training. On the real data it **returns empty folds**: only 16 % of training
compounds fall in a multi-member analog series at T = 0.55. The training set is a
diversity library (internal NN Tanimoto median 0.450) while the test set is
analog-dense (median 0.656, 85 % with a neighbour ≥ 0.55).

This was a genuine pre-launch modelling error on our part — we assumed both splits
would share the analog-series structure the *test* set was described as having. The
fallback (5-fold scaffold-grouped `GroupKFold`) is defensible and probably
pessimistic, since each test compound has a training analog at median T = 0.587,
but it is a surrogate and the local→leaderboard gap is partly its cost.

### 2.3 Two rejected hypotheses that cost real time

**Variance-expansion recalibration.** v1 predictions carry 0.51–0.91 of the label
spread; expanding them to match looked like free accuracy. Optimal α is **1.00 on
every isoform**, and variance-matching (α = 1.40–2.16) degrades MA-ST-RAE 0.7632 →
1.0041. Under an L1-type metric, shrinkage toward the conditional median is correct.
**Consequence for strategy: stop treating prediction-spread compression as a bug.**

**"Ranks well, predicts magnitudes poorly."** Suggested by percentile ranks (56th on
Spearman, 50th on ST-RAE) but z = +0.64 and z = −0.04 against the field median. Not
a real asymmetry. **Consequence: no rank-then-calibrate two-stage design.**

### 2.4 What v1 got right

* The TDI labeler reproduced the organizers' released labels on 5,080/5,081 rows,
  after empirically resolving two cases the FAQ leaves undefined (both → negative).
  That work is why the classification submission was well-formed on day one.
* Threshold tuning for MCC is worth +0.096 on CYP3A4 — most of the classification
  result.
* Pre-launch we predicted a public-data-only model would transfer poorly. Confirmed
  twice: near no-skill on Octant before launch, and adding 33,008 public compounds
  to the real training set *degrades* MA-ST-RAE 0.7632 → 0.8202.
* No sub-floor exploit exists (blanket mid-range guessing scores 0.74 vs 0.21).
  Checked so we would not waste the intermediate deadline on it.

## 3. v2 direction and why

Two changes, each measured in isolation, total honest gain **−0.0482** MA-ST-RAE
(0.7632 → 0.7150 leak-free).

**Change 1 — TabICL replaces LightGBM (−0.0389).** Chosen from leaderboard evidence,
not preference: TabICL is the only entry in its own tier. It wins on all four
isoforms with identical features and folds, in 82 s of GPU time. A feature-width
sweep shows capacity is saturated — SVD-128 (0.7243) and SVD-256 (0.7239) differ by
0.0004 — so fingerprint compression is not the constraint.

**Change 2 — cross-isoform stacking (−0.0093).** Exploits the structural asymmetry:
in training only 41 compounds carry all four labels, but the test set is dense, so
every test compound has all four isoform predictions available. Each isoform's model
receives the other three as features. We flagged a leakage path (donor models
trained on rows that could be scaffold-mates of held-out rows), built the nested
leak-free version, and measured the leak at **0.0008** — negligible. The gain is
real.

## 4. Further work, in priority order

### 4.1 Close the representation-plus-architecture gap (highest value)

Our best local number (0.7150) is not comparable to the leaderboard's 0.6755 — one
is scaffold-grouped CV on training data, the other is the blinded test set. Given
our own local→leaderboard optimism of +0.13, v2 plausibly lands near 0.84, still
behind TabICL-baseline.

An important reordering of what we said last turn. Decomposing the organizers' four
baselines, all scored on the same test set:

| baseline | MA-ST-RAE | vs LGBM |
|---|---|---|
| TabICL | 0.6755 | **−0.2174** |
| CheMeleon | 0.8335 | −0.0593 |
| LGBM | 0.8929 | — |
| XGB | 0.8975 | +0.0046 |

**The learned molecular representation is worth −0.059; the in-context tabular
architecture is worth −0.217.** So "swap in CheMeleon embeddings" is *not* the big
lever — we had implied it was. The architecture is doing the work.

**Update — the CheMeleon experiment has now been run, and it is negative.** We
downloaded the encoder (`chemeleon_mp.pt`, md5 verified), used it frozen with mean
aggregation to produce 2,048-dim graph embeddings, and scored three representations
under the identical protocol:

| representation | head | MA-ST-RAE |
|---|---|---|
| ECFP4 + descriptors | LightGBM | 0.7632 |
| CheMeleon (2,048) | LightGBM | 0.7956 |
| ECFP4-SVD256 + descriptors | TabICL | **0.7243** |
| CheMeleon-SVD256 | TabICL | 0.7617 |
| CheMeleon-SVD256 + ECFP4-SVD256 + descriptors | TabICL | 0.7324 |

Frozen CheMeleon **loses to ECFP4 + descriptors under both heads** (+0.032 with
LightGBM, +0.037 with TabICL), and concatenating both representations also loses to
ECFP4 + descriptors alone. Two things are worth extracting:

* The architecture effect **reproduces on a different feature family**: CheMeleon
  features go 0.7956 (LightGBM) → 0.7617 (TabICL), −0.034. That is now confirmed
  twice and is the most transferable finding we have.
* The **one** winning cell in the whole matrix is CYP2D6 under LightGBM
  (0.9796 → 0.9472, −0.032) — the isoform whose SAR §4.2 shows to be orthogonal.

**Limitation, stated plainly:** we used CheMeleon *frozen*. The organizers' baseline
fine-tunes it end-to-end (`from_chemeleon: true`, 3 FFN layers of 1,024, 100 epochs
with early stopping). So this rules out frozen embeddings as drop-in features, not
CheMeleon fine-tuned. That said, their own fine-tuned CheMeleon baseline scored
0.8335 against their TabICL baseline's 0.6755, so the direction of the evidence is
consistent.

What remains untested for closing the gap: TabICL ensembling / `n_estimators`,
full-width fingerprints without the SVD compression TabICL's feature regime forced
on us, and CheMeleon fine-tuned rather than frozen (worthwhile mainly for CYP2D6).

### 4.2 CYP2D6 — the one isoform that resists everything

ST-RAE 0.91–0.98 (near no-skill) across LightGBM, TabICL and stacking; TDI MCC
0.098 against 0.408 for CYP3A4. It is also the only isoform read out by a different
assay technology — label-free Echo-MS dextromethorphan depletion, versus fluorogenic
probe displacement for CYP1A2/2C9/3A4. **This has now been investigated.** Two
hypotheses were separated because they have opposite implications: assay noise says
stop spending effort, harder SAR says spend it on features.

**H1 — the Echo-MS readout is noisy: REFUTED.** CYP2D6 has the **narrowest**
credible intervals of all four isoforms (median width 0.272 overall, 0.261 above the
assay floor; median `_std` 0.069), against CYP2C9's 0.526/0.432. Its
noise-to-signal ratio (median CI width above floor ÷ mean absolute deviation) is
0.414 — mid-pack, and better than CYP2C9's 0.729. The Echo-MS arm is, if anything,
the *cleanest* of the four. Nothing about the 0.98 is explained by measurement
uncertainty.

**H2 — CYP2D6's SAR is orthogonal: SUPPORTED, strongly.** Three independent lines:

1. **Cross-isoform correlation.** Mean off-diagonal Spearman ρ against the other
   three isoforms, on co-measured compounds only: CYP2D6 **−0.002**, versus +0.288
   (CYP1A2), +0.330 (CYP2C9), +0.375 (CYP3A4). CYP2C9↔CYP3A4 reach +0.688. CYP2D6
   potency is essentially uncorrelated with everything else in the panel.
2. **The physicochemical drivers are inverted.** Basic-nitrogen count is the only
   simple feature positively correlated with CYP2D6 (ρ = +0.302) and it is
   *negative* for all three others (−0.098, −0.190, −0.119). Conversely
   lipophilicity, which drives CYP3A4 (+0.602) and CYP2C9 (+0.485), is inert for
   CYP2D6 (+0.034) — as is aromatic-atom count (−0.001 vs +0.393).
3. **A clean dose-response on basicity.** Fraction potent (pIC50 ≥ 5) rises
   monotonically with basic-nitrogen count: 27.3 % (n=935) → 51.0 % (n=522) →
   69.4 % (n=36); median pIC50 4.62 → 5.01 → 5.51.

This is textbook CYP2D6 pharmacology — the active site engages a protonated basic
amine at an anionic residue — recovered from the challenge data alone.

Two further findings sharpen what to do about it. First, **the blinded test set is
depleted in basic nitrogen**: 11.6 % of test compounds carry at least one, versus
18.9 % of training compounds. So the one strong CYP2D6 handle is scarcer at test
time than in training. Second, **hand-built pharmacophore features do not fix it**:
seven explicit descriptors (basic-N and quaternary-N counts, bond-path distances
from basic N to aromatic systems, logP, aromatic rings) score RAE 1.088 on their
own — *worse than predicting the mean* — against the full feature set's 0.981. The
basicity signal is real but weak, and already captured.

**Conclusion:** CYP2D6's ceiling is a genuinely orthogonal, basicity-driven SAR that
the shared ECFP4-plus-descriptor family represents poorly. It is neither assay noise
nor a missing descriptor we can hand-write. Three things follow: stop expecting
shared-representation gains on this isoform; note that CheMeleon's *only* win in the
entire matrix above was CYP2D6, which makes a learned graph representation
(fine-tuned) the most promising remaining route; and prefer pKa- or
protonation-state-derived features over substructure counts if we try features
again.

### 4.3 TDI improvements

* **Shift regression instead of classification.** Predict `pIC50(TDI) − pIC50(direct)`
  and threshold at 0.301, which uses the magnitude information the boolean discards.
* **Use the Emax file.** It ships `is_TDI` for **all four** isoforms (not just the
  two scored), giving CYP1A2/CYP2C9 labels as auxiliary multitask signal — unused so
  far.
* **Reconsider the 1,249 no-direct-arm CYP3A4 negatives.** They are 34.8 % of the
  labels and 1,055 of them show measurable TDI. Training against the organizers'
  convention is correct for scoring, but down-weighting them may produce a model
  that generalizes better to test compounds that *do* have both arms.

### 4.4 Deliberately not doing

* **Public-data augmentation** — measured to hurt, twice.
* **Sub-floor gaming** — no exploit exists.
* **Variance recalibration** — measured to hurt.
* **Structure/pose track** — `STRUCTURE_TRACK_LIVE = False`; the organizers turned
  it off, and one complex cost > 36 GPU-min locally.

## 5. Submission plan

One submission per team, latest valid counts, one per 4 hours.

1. **Now:** replace v1 with v2 (validated against both our validator and the
   organizers' official one; honest local gain −0.048). Low risk, no unverified
   hypothesis behind it.
2. **Before 2026-09-24:** the CheMeleon×TabICL experiment and the CYP2D6
   investigation. Freeze with ≥ 24 h margin — the 4-hour rate limit means a failed
   validation near the deadline can cost the slot.
3. **After the interim reveal (2026-09-25):** that is the one-time full-test-set
   unblinding. Read the paired deltas and tier structure, not absolute ranks.
4. **Before 2026-11-03:** final freeze.

On classification, the six-way tie means our rank 1 will likely be displaced by
noise as more teams submit. The defensible way to hold it is §4.3, not threshold
re-tuning against a leaderboard we cannot see.
