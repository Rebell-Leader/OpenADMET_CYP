# CYP Public Training Corpus — Data Dictionary

**Corpus version:** 1.0.0 · **Build date:** 2026-08-06
**Purpose:** public/external training data for the OpenADMET CYP Inhibition Blind Challenge
(direct-inhibition pIC50 regression for CYP1A2/2C9/2D6/3A4; TDI classification for CYP3A4/2D6).
No challenge train/test data exists yet (data drop 2026-08-17); everything here is external data.

---

## ⚠️ THE CENTRAL CAVEAT: direct-inhibition arm vs. TDI arm

The challenge asks for **direct-inhibition** pIC50 (`CYP{1A2,2C9,2D6,3A4}_pIC50_direct_inhibition`),
measured *without* a pre-incubation that lets time-dependent inactivation develop.

**The single highest-quality public CYP resource — the Octant/OpenADMET CYP3A4 dose-response
release — was NOT measured in that arm.** Its protocol
(`OpenADMET/Octant_CYP_blog_post/protocols/cyp_inhibition_assay.md`, assay code **C3A4IAP**,
"CYP3A4 Inhibition, Active Preincubation") specifies a **30-minute ACTIVE-ENZYME pre-incubation**:
compound is pre-incubated with CYP3A4, NADP+ and the regenerating system before substrate addition.
The dataset README states the consequence directly: IC50 values reflect reversible inhibition
**plus** any time-dependent effects developing during the 30-minute pre-incubation, and this
"differs from standard reversible-only IC50 assays."

**Therefore:** `octant_cyp3a4_drc.parquet` and every corpus row with
`CYP3A4_arms == "TDI_plus_NADPH_preincubation"` maps to the challenge's **+NADPH / TDI arm**,
not the direct-inhibition arm. Using it as a direct-inhibition label will bias predictions
upward (more potent) for time-dependent inhibitors specifically — i.e. exactly the compounds
the TDI track is about.

The protocol file also names a sibling assay, **C3A4IIP** ("inactive preincubation", enzyme not
active during pre-incubation), which *would* correspond to the direct arm. **That data is not in
the public release.**

**Why the Octant table is still the most valuable public resource here:** it is the only public
CYP data carrying real per-compound **pIC50 credible-interval bounds**
(`CYP3A4_pIC50_ci_lower` / `_ci_upper`, from 12-point Bayesian DRC fits), which is precisely what
the challenge's MA-ST-RAE metric consumes. Use it to exercise the metric realistically
(→ shared with the eval-harness track), and as a TDI-arm signal — not as a direct-arm label.

---

## Files

| File | Rows | Description |
|---|---|---|
| `cyp_public_corpus.parquet` | 37,006 | Main corpus, one row per unique standardized compound (InChIKey14), per-isoform pIC50 columns |
| `octant_cyp3a4_drc.parquet` | 1,340 | Octant CYP3A4 DRC table with credible intervals, cleaned + standardized (TDI arm) |
| `tdi_auxiliary_labels.parquet` | 11,148 | TDI labels with explicit confidence tiers |
| `corpus_coverage_by_isoform.csv` | 5 | Per-isoform / per-source coverage summary |
| `mbi_alert_definitions.csv` | 22 | MBI structural-alert SMARTS with positive+negative controls |
| `mbi_alert_enrichment.csv` | 22 | Each alert's measured enrichment on the 621-compound labelled TDI set |
| `corpus_summary.png` | — | Coverage, pIC50 distributions by source, replicate agreement |
| `cyp_corpus_long_provenance.parquet` | 255,440 | Full un-aggregated long table: one row per measurement with all provenance columns (audit trail) |
| `cyp_standardize.py` | — | The standardization pipeline itself (importable) |

![Corpus summary]({{artifact:a9ecaf67-5941-4776-9a2c-4fa0e589e5db}})

---

## 1. `cyp_public_corpus.parquet`

One row per unique compound, keyed by 14-character InChIKey skeleton.

### Identity / structure

| Column | Type | Units | Description |
|---|---|---|---|
| `corpus_version` | str | — | `1.0.0` |
| `build_date` | str | ISO date | `2026-08-06` |
| `inchikey14` | str | — | **Primary key.** First 14 chars of InChIKey = connectivity skeleton (stereo- and salt-insensitive). Deduplication key. |
| `inchikey` | str | — | Full standard InChIKey (27 chars) of the standardized structure |
| `std_smiles` | str | — | Canonical SMILES after the standardization pipeline (§4) |
| `mw` | float | g/mol | Molecular weight of the standardized (desalted, neutral) structure |
| `n_heavy` | int | count | Heavy-atom count |
| `n_isoforms_with_pIC50` | int | 0–4 | How many of the four challenge isoforms have a pIC50 for this compound |

### Per-isoform activity — repeated for `CYP1A2`, `CYP2C9`, `CYP2D6`, `CYP3A4`, `CYP2C19`

`CYP2C19` is included as an **auxiliary multitask head only** — it is not a challenge endpoint.

| Column | Type | Units | Description |
|---|---|---|---|
| `{iso}_pIC50` | float | −log10(M) | **Aggregated potency label.** Median across all retained measurements. Computed from uncensored measurements when any exist; otherwise from censored ones (see `{iso}_all_censored`). Pools IC50/AC50/Ki/Potency/EC50/Kd — see `{iso}_value_types`. |
| `{iso}_pIC50_spread` | float | log units | max − min across measurements. 0.0 when n=1. |
| `{iso}_n` | Int64 | count | Number of measurements aggregated |
| `{iso}_n_censored` | Int64 | count | How many were censored (`>`/`<`/`>=`/`<=` relation) |
| `{iso}_all_censored` | bool | — | **True ⇒ the pIC50 is a bound, not a point estimate.** Treat as censored in the loss or drop. |
| `{iso}_discordant` | bool | — | `{iso}_pIC50_spread > 1.0` log unit. **Discordant duplicates — inspect or downweight.** |
| `{iso}_sources` | str | — | `\|`-joined contributing sources: `octant_openadmet`, `chembl_35plus`, `pubchem_qhts` |
| `{iso}_arms` | str | — | `\|`-joined assay arms (§3). `direct`, `TDI_or_preincubation`, `TDI_plus_NADPH_preincubation`, `unknown` |
| `{iso}_value_types` | str | — | `\|`-joined measurement types pooled into the median |
| `{iso}_pubchem_call` | str | — | Qualitative PubChem panel outcome (`Active`/`Inactive`/`Inconclusive`), modal value. **Present for many compounds with no pIC50** — use for auxiliary classification heads / pretraining. |

### TDI and structural alerts

| Column | Type | Description |
|---|---|---|
| `mbi_alerts` | str | `\|`-joined alert IDs matched (see `mbi_alert_definitions.csv`) |
| `n_mbi_alerts` | int | Count of matched alerts |
| `CYP3A4_is_TDI_tier1` / `CYP2D6_is_TDI_tier1` | Int64 | 1/0 experimental TDI label where a tier-1 source exists; null otherwise |
| `CYP3A4_TDI_tier` / `CYP2D6_TDI_tier` | str | Which tier-1 tier supplied the label |

---

## 2. `octant_cyp3a4_drc.parquet` — the credible-interval table

**Arm: TDI / +NADPH (30-min active-enzyme pre-incubation). NOT direct inhibition.**
Source: HF `openadmet/Octant_CYP_inhibition_reactivity_blog_release`, subset `inhibition`
(revision `96dc1cceaa545a22041d1e16a9c2524a658403f8`, updated 2026-08-06), CC-BY-4.0.
Assay: recombinant CYP3A4 supersomes (5 nM), DBOMF fluorogenic probe (2 µM), 100 mM KPO4 pH 8,
12-point DRC ~0.1 nM–50 µM (~3-fold dilutions), 6 µL total volume, 1536-well.

| Column | Type | Units | Description |
|---|---|---|---|
| `ocnt_batch` | str | — | Octant compound/batch identifier (original key) |
| `source_smiles` | str | — | SMILES as released by Octant (their `standardized_smiles`) |
| `std_smiles`, `inchikey`, `inchikey14`, `mw`, `n_heavy`, `std_status` | — | — | Our standardization output (§4) |
| `isoform` | str | — | Always `CYP3A4` |
| `CYP3A4_pIC50` | float | −log10(M) | Fitted pIC50 from the 12-point Bayesian DRC. Null for 256 rows that produced no fit. |
| `CYP3A4_pIC50_se` | float | log units | Standard error on pIC50 |
| `CYP3A4_pIC50_ci_lower` / `_ci_upper` | float | −log10(M) | **95% credible-interval bounds — the ST-RAE-relevant quantity.** Median CI width 0.094 log units (IQR 0.069–0.213; max 2.28). |
| `slope_log2` | float | — | Hill slope of the fitted curve |
| `emax_log2fc` | float | log2 FC | Maximum effect (log2 fold-change in fluorescence) |
| `activity_status` | str | — | `YES` (1,186) / `NO` (154) — detectable inhibition |
| `rollover_status` | str | — | Hook-effect / rollover artifact flag (`YES` 17) |
| `saturation_status` | str | — | Whether the curve reached saturation |
| `direction` | str | — | `DOWN` (inhibition) / `FLAT` |
| `drc_qc_status` / `drc_qc_flag` | str | — | Curve QC. 1,139 `PASS`; failures are mostly `FAIL INCOMPLETE` (181) |
| `qc_flag_primary` | str | — | Primary-screen QC flag |
| `plate_qc_status` | str | — | Plate-level QC (`FAIL` 96) |
| `qc_usable` | bool | — | **Recommended filter:** `drc_qc_status==PASS & plate_qc_status==PASS & pIC50 notnull` → **1,084 rows** |
| `below_lowest_dose` | bool | — | `pIC50 < 4` — below the lowest tested dose, unreliable (83 rows). The challenge's ST-RAE downweights these. |
| `arm` | str | — | Always `TDI_plus_NADPH_preincubation` |
| `preincubation_min` | int | min | Always 30 |
| `censored` | bool | — | Always False |

---

## 3. `tdi_auxiliary_labels.parquet` — TDI signal with confidence tiers

`is_TDI`: 1 = time-dependent/mechanism-based inhibitor, 0 = not, null = unlabelled (tier 2 only).

### Tiers

| Tier | Rows | Labelled | Positives | Meaning |
|---|---|---|---|---|
| `tier1_regulatory` | 6 | 6 | 6 | FDA DDI table, in-vitro inhibitor list, footnote (a) = "Time-dependent inhibitors." |
| `tier1_paired_experimental` | 3 | 3 | 1 | Octant paired ±active-enzyme-preincubation DRCs (real IC50-shift measurement) |
| `tier1_literature_curated` | 616 | 616 | 302 | Experimental CYP3A4 TDI calls, per-compound primary citation |
| `tier2_alert_match_only` | 10,523 | 0 | — | **Structural-alert match only — NOT an experimental label.** 5,310 unique compounds × {CYP3A4, CYP2D6} |

**Tier-1 total: 625 compound×isoform rows, 309 positive.** Every tier-1 entry carries a real,
retrievable source in `source_citation`. No compound lists were invented.

#### Tier-1 sources

1. **FDA** — *Drug Development and Drug Interactions: Table of Substrates, Inhibitors and Inducers*,
   Table 1-2 "Examples of in vitro inhibitors for CYP-mediated metabolism", footnote (a)
   ("Time-dependent inhibitors."), retrieved 2026-08-06. Yields: furafylline (CYP1A2),
   tienilic acid (CYP2C9), paroxetine (CYP2D6), azamulin / troleandomycin / verapamil (CYP3A4).
   Structures resolved from PubChem by name (formulas + InChIKeys verified), not asserted from memory.
2. **Octant paired ±preincubation** — `OpenADMET/Octant_CYP_blog_post`,
   `data/tdi_pic50_shift.tsv` + `data/tdi_drc_params.tsv`. Six drugs with pIC50 measured in both
   arms and a posterior delta. TDI called at `delta_mean > 0.3` (≈2-fold IC50 shift) with the 95%
   credible interval excluding 0. Reproduces known pharmacology: troleandomycin Δ=1.25,
   azamulin Δ=0.99, verapamil Δ=0.84, diltiazem Δ=0.35 → TDI; ketoconazole Δ=0.03,
   clotrimazole Δ=0.17 → not TDI.
3. **Faramarzi et al. 2024** — *Front Pharmacol* 15:1451164, doi:`10.3389/fphar.2024.1451164`
   (open access, CC-BY), Supplementary Table S1 (`Supplementary_TableS1_Full_Database.sdf`,
   10,129 chemicals). Its `CYP3A4 TDI` field carries 623 experimental calls (306 positive /
   317 negative) harvested from FDA drug approval packages and published literature, each with a
   DOI / PMID / named primary reference. 616 survived structure standardization + deduplication.

### Columns

| Column | Type | Description |
|---|---|---|
| `compound_name` | str | Name as given by the source (null for tier 2) |
| `isoform` | str | `CYP3A4`, `CYP2D6`, `CYP1A2`, `CYP2C9` |
| `is_TDI` | Int64 | 1 / 0 / null (tier 2 = null) |
| `tier` | str | See table above |
| `evidence` | str | What the label is based on, including the numeric delta for paired data |
| `source_citation` | str | Full retrievable citation (URL / DOI / repo path + per-compound primary ref) |
| `source_smiles` | str | SMILES as obtained from the source (or PubChem-resolved for named drugs) |
| `std_smiles`, `inchikey`, `inchikey14`, `mw`, `n_heavy` | — | Standardization output (§4) |
| `matched_alerts` | str | `\|`-joined MBI alerts matched by this structure |
| `max_alert_ppv` | float | Highest measured PPV among the *enriched* alerts this compound matches (see §5) |
| `octant_pIC50_no_preinc` | float | pIC50 without pre-incubation (6 drugs only) |
| `octant_pIC50_with_preinc` | float | pIC50 with active-enzyme pre-incubation (6 drugs only) |
| `octant_delta_mean` | float | Posterior mean ΔpIC50 (with − without) |
| `octant_delta_q2.5` / `_q97.5` | float | 95% credible interval on the delta |
| `has_paired_preincubation_data` | bool | True for the 7 rows with genuine paired ±preincubation measurements |
| `corpus_CYP3A4_pIC50` / `corpus_CYP2D6_pIC50` | float | Corpus potency for context (may be TDI-arm for CYP3A4 — see the central caveat) |
| `label_conflict` | bool | True if sources disagreed on `is_TDI` for this compound×isoform (0 occurrences) |

**Coverage gap to state plainly:** public tier-1 TDI labels are essentially **CYP3A4-only**.
The challenge also scores **CYP2D6_is_TDI**, for which exactly **one** tier-1 public label was
found (paroxetine, FDA). CYP2D6 TDI will have to be carried by transfer from CYP3A4 TDI, by the
alert profile, and by the challenge's own training data once it drops.

---

## 4. Standardization pipeline

One pipeline, applied identically to every source (`cyp_standardize.py`, RDKit 2026.03.5):

1. `rdMolStandardize.Cleanup` — sanitize, disconnect metals, normalize functional groups, reionize
2. `LargestFragmentChooser` — strip salts / solvents / counterions
3. `Uncharger` — neutralize
4. `TautomerEnumerator.Canonicalize` (max 200 tautomers / 200 transforms) — canonical tautomer
5. Canonical SMILES + full InChIKey + 14-char skeleton InChIKey

56,840 unique input SMILES → 56,839 standardized (`std_status == "ok"`); 1 failure
(`error:AtomValenceException`). `std_status` values: `ok`, `ok_no_tautomer`, `parse_fail`,
`empty`, `no_heavy_atoms`, `error:<Type>`.

Validated on: aspirin (unchanged), nicotine tartrate → nicotine (salt stripped), sodium benzoate →
benzoic acid (neutralized), cyclohexanone/cyclohex-1-en-1-ol → single key (tautomers collapsed).

### Deduplication and aggregation

Grouped on `(inchikey14, isoform)`. Aggregate = **median** of uncensored measurements
(falling back to censored if none uncensored), plus `min`/`max`/`spread`/`std`/`n`.
Discordance flag at spread > 1 log unit: **829 of 14,854** multi-measurement compound×isoform
pairs across the four challenge isoforms.

**Cross-source double-counting was explicitly removed:** 51,872 ChEMBL rows have `src_id == 7`
(*PubChem BioAssays*) and are re-deposits of the same NCGC AIDs harvested natively here. They are
tagged `redeposit_of_pubchem` in the long table and **excluded** from the aggregation pool.
AID 1851 similarly re-appears inside ChEMBL and is counted once, from the PubChem side.
AID 885 (CYP3A4 **activators**, not inhibitors) is excluded from the quantitative pool.

---

## 5. Provenance of each source

| Source tag | Records fetched | In long table | In quant. aggregation pool | Endpoint | Arm |
|---|---|---|---|---|---|
| `octant_openadmet` | 1,340 | 1,084 | 1,084 | 12-pt Bayesian DRC pIC50 + 95% CrI | **TDI / +NADPH (30-min active pre-incubation)** |
| `chembl_35plus` | 155,803 activities | 83,909 potency-typed | 32,037 (after removing 51,872 PubChem re-deposits) | IC50 / AC50 / Ki / Potency / EC50 / Kd | mostly `unknown`; 4,123 inferred TDI/pre-incubation, 534 inferred direct |
| `pubchem_qhts` | 173,765 rows / 14 AIDs | 170,447 | 74,617 quantitative AC50 (all 173,765 carry a qualitative call) | AC50 (from `Fit_LogAC50`, log10 M) + Active/Inactive/Inconclusive | `unknown` |

Total quantitative aggregation pool: **107,738** measurements → **83,880** compound×isoform
aggregates. The long table (255,440 rows) retains everything, including qualitative-only rows and
the tagged re-deposits, for auditability.

### ChEMBL targets (looked up and verified, not guessed)

Resolved via `target_components__accession` on the human UniProt accession; all four confirmed
`target_type = SINGLE PROTEIN`, `organism = Homo sapiens`, `tax_id = 9606`:

| Isoform | UniProt | ChEMBL target | Activities |
|---|---|---|---|
| CYP1A2 | P05177 | `CHEMBL3356` | 28,001 |
| CYP2C9 | P11712 | `CHEMBL3397` | 33,914 |
| CYP2D6 | P10635 | `CHEMBL289` | 36,650 |
| CYP3A4 | P08684 | `CHEMBL340` | 57,238 |

Protein-family targets (`CHEMBL4523986` CYP, `CHEMBL2111472` 3A4/3A5, `CHEMBL2364675` 3A,
`CHEMBL3544905` 1A, `CHEMBL6066546` 1A2/2C18/2C19) were **deliberately not used** — they are
multi-component and would confound isoform assignment.

**pIC50 conversion:** `pActivity = 9 − log10(value_nM)`. Units mapped explicitly
(pM/nM/µM/mM/M); rows with non-concentration units (%, s⁻¹, /min, hr, pmol/min …) or a missing
structure are excluded from the potency pool — of 129,318 potency-typed rows, 45,409 were dropped
(no valid concentration or no SMILES), leaving 83,909.

**Censoring is flagged, never silently dropped.** On a potency, relation `>` means *weaker than*
the bound, so the pActivity is an **upper bound** (`censoring = upper_bound_on_pActivity`,
15,508 rows); `<` gives a lower bound (698 rows); `=`/`~` are exact (67,703 rows).

### PubChem AIDs (each target verified before use)

| AID | Isoform | Assay | CIDs | Notes |
|---|---|---|---|---|
| 1851 | 1A2, 2C9, 2C19, 2D6, 3A4 | NCGC "Cytochrome panel assay with activity outcomes" | 16,560 | 5 panel members split by `Panel Name`; 17,143 SIDs × 5 |
| 410 | CYP1A2 | NCGC qHTS `p450-cyp1a2` | 8,354 | qualitative only (no AC50 column) |
| 883 | CYP2C9 | qHTS inhibitors & substrates | 9,385 | |
| 884 | CYP3A4 | qHTS inhibitors & substrates | 13,076 | |
| 885 | CYP3A4 | qHTS **activators** | 13,076 | **excluded from potency pool** (activation ≠ inhibition) |
| 891 | CYP2D6 | qHTS inhibitors & substrates | 9,385 | |
| 899 | CYP2C19 | qHTS inhibitors & substrates | 9,385 | auxiliary isoform |
| 1645840 / 1645841 / 1645842 | 2D6 / 3A4 / 2C9 | luciferase **cell-based** qHTS antagonists | 5,095 | 5 replicates; `assay_kind = cell_luciferase` — different matrix, do not pool naively with biochemical |
| 1919971 / 1919972 / 1919973 / 1919976 | 3A4 / 2D6 / 2C9 / 1A2 | qHTS vs NCATS DSHEA & TCM libraries | 209 | natural products |
| 1963596 | CYP3A7 | — | — | **not used** (wrong isoform) |

Fetched from `ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/` (PUG-REST returns
`PUGREST.BadRequest` on these record counts). `pAC50 = −Fit_LogAC50` (already log10 molar);
falls back to `−log10(Potency_µM × 1e-6)`. Replicate AIDs use the median across replicates.

### MBI structural alerts — measured, not assumed

22 alerts encoded as SMARTS in `mbi_alert_definitions.csv`, from Faramarzi et al. 2024 Table 4
(the alerts qualified there as statistically significant and predictive) and the reviews cited
therein (Kalgutkar 2007 `10.2174/138920007780866807`; Orr et al. 2012 `10.1021/jm300065h`;
Kalgutkar & Soglia 2005; Correia 2005; Fontana 2005; Bolleddula 2014; Yu 2015).
**Every SMARTS is verified against a named positive control and a near-miss negative control**
(e.g. catechol matches the ortho-hydroquinone alert, resorcinol does not) — all 22 pass both.

**Provenance precision.** Exactly **10** of the 22 encoded alerts correspond to Table 4 categories
(`in_faramarzi2024_table4 == True`): benzodioxole, furan, terminal alkyne, primary aliphatic amine,
para-hydroquinone, ortho-hydroquinone/catechol, epoxide, tertiary piperazine, alkylphenol, and
alkylaromatic ether. The remaining 12 come from the cited reviews only. In particular
**MBI04 `internal_alkyne` is NOT a Table 4 alert** — Table 4 qualifies only *terminal*
(omega / omega-1) alkynes; MBI04 is attributed to Kalgutkar 2007 / Correia 2005 and was not
enriched on the labelled set, so it does not contribute to tier 2.

Two Table 4 categories are **deliberately not encoded** and are recorded as `NOT_ENCODED_*` rows
in `mbi_alert_definitions.csv` with the reason: *reactive arenes* and *arenes (miscellaneous,
benzopyran analogues)*. Both are defined in the paper by drawn general structures with positional
constraints ("C9: no heteroatom attachment", "C7: saturated carbon", variable heteroaromatic bond
orders) that cannot be reconstructed unambiguously from text; guessing a SMARTS for them would
mis-state the published alert.

`mbi_alert_enrichment.csv` reports each alert's **measured** performance on the 621-compound
labelled CYP3A4 TDI set (306 pos / 315 neg, base rate 0.493), one-sided Fisher exact:

| Alert | n matched | PPV | OR | p | Verdict |
|---|---|---|---|---|---|
| benzodioxole (methylenedioxyphenyl) | 23 | 0.87 | 7.27 | 1.5e-4 | enriched |
| cyclopropylamine | 78 | 0.71 | 2.78 | 4.2e-5 | enriched |
| primary aliphatic amine | 33 | 0.76 | 3.41 | 1.4e-3 | enriched |
| furan | 22 | 0.73 | 2.84 | 0.021 | enriched |
| catechol (ortho-hydroquinone) | 48 | 0.65 | 1.98 | 0.019 | enriched |

The other 17 alerts were **not** significantly enriched on this set (thiophene PPV 0.40,
nitroaromatic 0.25, para-hydroquinone 0.40 — at or below the base rate). Only the 5 enriched
alerts are used to populate tier 2. This is an honest ceiling: alert-match alone is weak evidence,
which is exactly why tier 2 carries `is_TDI = null` rather than a fabricated 1.

---

## 6. Recommended usage

- **Direct-inhibition regression:** train on `{iso}_pIC50` where `{iso}_arms` does **not** contain
  `TDI_plus_NADPH_preincubation`; treat `{iso}_all_censored` rows as censored; downweight or drop
  `{iso}_discordant`; remember `pIC50 < 4` is below the assay floor.
- **Pretraining / auxiliary heads:** `{iso}_pubchem_call` gives 16.7k–23.4k qualitative labels per
  isoform (including CYP2C19), far more compounds than have pIC50 — well suited to a multitask
  classification head or a pretraining objective, not to primary pIC50 supervision.
- **TDI classification:** tier-1 rows only for supervision (625 rows, CYP3A4-dominated);
  `mbi_alerts` / `max_alert_ppv` as features, not labels.
- **Metric development:** `octant_cyp3a4_drc.parquet` is the realistic ST-RAE test bed — real
  pIC50 point estimates with real credible intervals, plus QC flags and sub-floor compounds.

## 7. Licences

- Octant / OpenADMET releases: CC-BY-4.0 (HF dataset card).
- ChEMBL: CC-BY-SA-3.0. PubChem BioAssay: public domain.
- Faramarzi et al. 2024 supplementary: CC-BY (Frontiers open access).
- FDA table: US Government work.

## 8. Not retrieved (stated explicitly rather than substituted)

- **Obach et al. 2007**, *Drug Metab Dispos* `10.1124/dmd.106.012633` — the canonical compilation of
  MBI kinact/K_I values. Closed access; Unpaywall/Semantic Scholar/PMC all returned no OA copy.
  No values from it are used or quoted anywhere in this corpus.
- **C3A4IIP** (Octant inactive-preincubation / direct-arm CYP3A4 assay) — named in the protocol
  file but not present in the public release.
- `zenodo.org` is unreachable from this sandbox.
