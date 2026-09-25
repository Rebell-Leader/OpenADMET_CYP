# What transfers and what leaks: an ablation study on the OpenADMET CYP inhibition blind challenge

**Draft — team `avaliev`. Last revised 2026-09-21.**
Status: working draft. Every number is traceable to a file in this repository;
figures are in `../06_figures/`. Sections marked *[pending]* await the final
leaderboard.

---

## Abstract

Blind prediction challenges reward a single leaderboard number, but the transferable
product of an entry is usually the set of things that did *not* work and the reason
why. We report a controlled ablation study on the OpenADMET CYP inhibition blind
challenge, in which structure-only models predict direct-inhibition pIC<sub>50</sub>
against four cytochrome P450 isoforms and time-dependent inhibition class against two.

Across eight approaches benchmarked under identical scaffold-grouped
cross-validation, we find that (i) an in-context tabular learner outperforms gradient
boosting by 0.039 relative-absolute-error units, reproduced on two disjoint feature
families; (ii) a shared-trunk multitask network exploits a 33 %-filled label matrix
for a further 0.016, and auxiliary supervision from a dense single-concentration
screen adds 0.015 more; and (iii) blending two model families yields 0.031 — larger
than any single-model change and beating both parents on every isoform despite error
correlations of 0.89–0.94.

Three negative results carry more transferable content than the positive ones. A
dense auxiliary assay measured on the *same* compounds as the target degrades
performance when distilled into a feature (+0.036), while the identical data helps as
additional training rows or as an auxiliary task — the distinction being whether it
adds compounds or merely re-encodes existing inputs. Naive augmentation with 33 000
public ChEMBL/PubChem measurements hurts three of four isoforms (+0.057), consistent
with an assay-condition mismatch. And frozen embeddings from a molecular foundation
encoder underperform ECFP4 plus physicochemical descriptors.

We also document a methodological near-miss with a magnitude worth publicising: the
non-nested version of the distillation experiment scored 0.277 against 0.834 nested — a
0.557 artefact that would have read as a decisive win. We argue that surrogate
features built on a shared compound set require nested validation as a default, not as
a robustness check.

---

## 1. Introduction

Cytochrome P450 inhibition is a standard early-ADMET liability: a compound that
inhibits CYP3A4 or CYP2D6 carries drug–drug-interaction risk regardless of its
on-target potency. Predicting it from structure is a long-standing QSAR task, and a
blind challenge with a freshly measured, undisclosed test set is the cleanest
available test of whether reported gains generalise.

The OpenADMET challenge has a design that makes it unusually informative about
generalisation, and unusually easy to over-fit locally:

* **The training set is a diversity library; the test set is analog-dense.** 85 % of
  test compounds have a near neighbour at Tanimoto ≥ 0.55, against 17 % within
  training. The test set was built by taking the most potent hits per isoform and
  purchasing their nearest catalogue analogs.
* **The label matrix is sparse where the test matrix is dense.** Of 4 905 training
  compounds, 3 596 carry exactly one pIC<sub>50</sub> label and only 41 carry all
  four; all 750 test compounds are assayed against all four.
* **The primary metric is interval-aware.** Soft-threshold relative absolute error
  (ST-RAE) charges zero error for a prediction landing inside the measured credible
  interval, and those intervals are far wider below the assay floor than above it.

This combination means a random-split cross-validation is badly optimistic, and that
the metric is not a monotone function of accuracy. We therefore built the evaluation
apparatus before any model (§2), and report every experiment against it.

## 2. Methods

### 2.1 Evaluation apparatus

We reimplemented the challenge metrics from the published definition
(`03_code/cyp_metrics.py`, 109 unit tests) and the TDI labelling rules
(`03_code/cyp_tdi.py`). The labelling rules are underdetermined by the published FAQ
in two cases; both were resolved empirically against the released training labels, and
the reimplementation reproduces all 5 080 of them exactly. Notably, a missing
direct-arm measurement is labelled negative regardless of the pre-incubation-arm
signal — 1 055 CYP3A4 compounds show measurable time-dependent inhibition yet are
labelled negative under the organizers' convention.

### 2.2 Splitting

All cross-validation is 5-fold grouped by Bemis–Murcko scaffold. We also built an
analog-series-aware protocol mimicking the test construction
(`03_code/cyp_splits.py`), but the training set contains too few analog series to
populate it — the mimicry is only possible on the test side. Random splitting is
optimistic by 0.039–0.051 RAE relative to scaffold grouping, consistently across all
four isoforms.

Local scaffold-grouped CV is **+0.113 to +0.127** optimistic relative to the
leaderboard, measured against two scored submissions. We use +0.12 as a conversion
constant throughout.

### 2.3 Representation

ECFP4 (2 048 bits, radius 2) concatenated with 217 RDKit 2D descriptors. For the
in-context and neural models the fingerprint block is compressed to 256 dimensions by
truncated SVD and the result standardised (473 dimensions total). The compression is
load-bearing rather than cosmetic: the full-width representation scores 0.7611 against
0.7239 compressed.

The SVD basis and standardiser are fitted on training and blinded-test features
stacked together. This is unsupervised use of public test structures, with no label
available to leak, but it is not a strictly inductive protocol, so we quantified it.
Refitting both transforms on training rows only and re-running identical folds
*improves* the macro score slightly, from 0.7818 to 0.7740 (−0.0079; per isoform
−0.020, −0.001, −0.011, 0.000; LightGBM instrument). The transductive fit therefore
conferred no advantage and marginally cost us; no result reported here depends on
access to the test structures. Full table in
`04_experiments/transductive_check.csv`.

### 2.4 Models

Gradient boosting (LightGBM), an in-context tabular learner (TabICL), and a
shared-trunk multitask network with per-isoform heads and masked mean-squared-error
loss (`03_code/multitask.py`). The multitask model optionally carries auxiliary heads
supervised on single-concentration screen log<sub>2</sub> fold-change, which fills
89 % of the same compound × isoform matrix that the pIC<sub>50</sub> labels fill 33 %
of.

## 3. Results

### 3.1 Architecture dominates representation

Decomposing the organizers' four published baselines on the same blinded test set
separates the two effects cleanly: the learned-representation baseline improves on
gradient boosting by 0.059, while the in-context tabular baseline improves by 0.217.
We reproduced the architecture effect locally (−0.039) and, importantly, reproduced it
again on a completely different feature family (−0.034 on frozen CheMeleon
embeddings). An effect that survives a change of representation is the most
transferable result in this study.

The representation axis, by contrast, is exhausted. Frozen foundation-encoder
embeddings were *worse* than ECFP4 plus descriptors under both model heads;
concatenating them was worse still; and widening or narrowing the fingerprint
compression changed almost nothing (0.7239 vs 0.7243 at half the width). The single
exception is CYP2D6, the one isoform where learned embeddings beat substructure
counts — consistent with §3.4.

**Figure 1** (`../06_figures/v3_and_field_motion.png`) — field motion between two
leaderboard snapshots; ensembling saturation and the full-width penalty; and the
decomposition showing ≥ 82 % of our gap to the leader is raw accuracy rather than
metric handling.

### 3.2 The multitask trunk, and an auxiliary task that works

A shared trunk over all four isoforms beats four independent fits of the same
architecture by **0.016** — same folds, same width, so the comparison isolates
sharing. Auxiliary log<sub>2</sub> fold-change heads add a further **0.015**.

Our own prior analysis of this dataset (§3.4) predicted that CYP2D6 should be excluded
from any shared trunk, its structure–activity relationship being orthogonal to the
other three. The ablation confirms this *without* auxiliary supervision (exclusion
worth 0.0056) and **reverses it with** (sharing worth 0.0023). The dense auxiliary
signal appears to supply enough isoform-specific capacity that CYP2D6 no longer
competes for representation. We report this as an update rather than a refinement: the
orthogonality finding is unchanged, but the architectural recommendation derived from
it does not survive the addition of auxiliary supervision.

### 3.3 Blending beats both parents everywhere

The in-context and multitask models are complementary per isoform — the multitask
network wins CYP2C9 by 0.051, the in-context model wins CYP1A2 by 0.022. A blend with
weights selected by nested inner cross-validation reaches **0.6868**, beating *both*
parents on *every* isoform:

| isoform | TabICL | multitask | blend | vs. better parent |
|---|---|---|---|---|
| CYP1A2 | 0.8099 | 0.8317 | **0.7939** | −0.016 |
| CYP2C9 | 0.6550 | 0.6044 | **0.5904** | −0.014 |
| CYP2D6 | 0.9232 | 0.9427 | **0.9012** | −0.022 |
| CYP3A4 | 0.5023 | 0.4914 | **0.4617** | −0.030 |

Error correlations between the two families are **0.888–0.938** — high. The gain is
therefore not decorrelation but variance reduction from averaging two function classes
with a tuned weight; beating both parents everywhere is the signature of the latter.

**Figure 2** (`../06_figures/multitask_and_blend.png`) — multitask ablation ladder,
per-isoform blend comparison, and the four-submission trajectory against the frontier.

### 3.4 CYP2D6 is a different problem

CYP2D6 resists every model (ST-RAE 0.90–0.96) and every feature family. Two
hypotheses were separated. Assay noise is refuted: CYP2D6 has the *narrowest* credible
intervals of the four despite a different readout technology. Orthogonal
structure–activity is supported on three independent lines — near-zero mean
correlation with the other three isoforms' potencies on co-measured compounds;
inverted physicochemical drivers, where basic-nitrogen count is the only simple
descriptor positively correlated with CYP2D6 while being negative for all others and
lipophilicity is inert where it dominates elsewhere; and a monotone potency response
to basic-nitrogen count. This recovers textbook CYP2D6 pharmacology — a protonated
basic amine engaging the active site — from the challenge data alone.

Two consequences sharpen the outlook: the blinded test set is *depleted* in basic
nitrogen relative to training (11.6 % vs 18.9 %), and hand-built pharmacophore
descriptors score worse than the mean, so the signal is real, weak, and already
captured by the full feature set.

**Figure 3** (`../06_figures/chemeleon_and_cyp2d6.png`).

### 3.5 Negative results

**Public data augmentation hurts** (+0.057 macro; three of four isoforms, CYP1A2 the
exception). The best public source is measured in the pre-incubation arm rather than
the direct arm, and a model trained only on public data reaches R² 0.117 against real
assay values — near no-skill.

**Screen distillation hurts** (+0.036, all four isoforms). See §4.

**Ensembling saturates** at 8 estimators; a fourfold increase buys 0.001.

**Figure 4** (`../06_figures/v3_pseudolabel_results.png`) — the leak, the floor-shift
mechanism, and the competition between same-information gains.

## 4. A 0.557 leak that looked like a breakthrough

The single-concentration screen is dense and correlates with same-isoform
pIC<sub>50</sub> at |ρ| = 0.83–0.94, but no test compound appears in it, so using it
requires predicting it. Fitting the donor models on all rows — so each donor had seen
the held-out compound's own measured screen value — produced ST-RAE **0.2771** with
R² **0.834** on CYP1A2. The identical experiment with donors refitted inside each
outer fold scored **0.8342**, slightly worse than not doing it at all.

The leak is large because the donor target is a near-perfect surrogate for the
quantity being predicted on the same molecule. Any pipeline that builds surrogate
features on a compound set shared with the target labels has this failure mode, and
its magnitude here — 0.557, enough to fabricate a decisive leaderboard win — argues
for nesting as a default rather than a robustness check.

The correct use of the same data follows from the same analysis. Roughly 2 900
compounds per isoform carry a screen reading but no fitted dose–response curve; these
are precisely the compounds the target labels do not cover. Adding them as
pseudo-labelled training rows helps (−0.008), but only after clipping at the assay
floor: compounds without a fitted curve are *censored* at pIC<sub>50</sub> 4, not
point-valued near 2.9, and linear extrapolation below the floor was catastrophic
(ST-RAE 1.42 and 2.11 on two isoforms). The median pseudo-label shift predicts the
damage with rank correlation −1.00 across isoforms, giving an a-priori selection rule
that requires no cross-validation scores and independently reproduces the CV-optimal
isoform subset.

A third observation completes the picture: pseudo-labels (−0.008) and cross-isoform
stacking (−0.008) applied together give only −0.003. Both inject screen-derived or
isoform-derived estimates of the same underlying quantity, so the second is largely
redundant. Gains from the same information compete; gains from different function
classes (§3.3) compose.

**Generalisable statement.** A dense auxiliary label measured on the same compounds as
the target adds nothing once it must be predicted from structure; it helps when it
covers compounds the target labels do not, or when it supervises an auxiliary task
whose gradient shapes a shared representation without entering at inference.

## 5. Limitations

* **Transductive preprocessing.** §2.3. Measured and found to confer no advantage
  (strictly inductive refit is 0.008 *better*), so this is a documented protocol
  deviation rather than an unquantified threat to validity.
* **Selection influence on the best number.** The blend weights are nested, but the
  decision to blend those two families followed from seeing both benchmarks. 0.6868 is
  the best-supported local estimate, not an unbiased one.
* **The architecture claim is model-specific.** TabPFN, the obvious second in-context
  learner, requires registration and an API token we do not have, so we cannot
  distinguish "in-context learning beats gradient boosting" from "TabICL does".
* **Frozen, not fine-tuned.** The foundation-encoder result rules out drop-in
  embeddings, not end-to-end fine-tuning, which is what the organizers' own baseline
  does.
* **Single held-out test set.** All leaderboard comparisons rest on 750 compounds, and
  the bootstrap standard deviations on the public board (~0.03) exceed most of the gaps
  between neighbouring entries.
* **Assumption diagnostics were not run.** No residual normality, homoscedasticity, or
  calibration testing beyond what the reported metrics imply.

## 6. Conclusion *[pending final leaderboard]*

On this task the ordering of returns was: model family ≫ blending > multitask
sharing > auxiliary data > tuning ≫ representation, with public-data augmentation and
same-compound surrogate distillation both actively harmful. Five successive
single-model levers returned ≤ 0.012 each while blending two families returned 0.031,
which suggests the marginal value of a third diverse family exceeds that of further
refining any one of them.

## Data and code availability

Everything in this repository: <https://github.com/Rebell-Leader/OpenADMET_CYP>.
Challenge data is redistributed from the organizers' Hugging Face release under its
original terms. No proprietary data was used.
