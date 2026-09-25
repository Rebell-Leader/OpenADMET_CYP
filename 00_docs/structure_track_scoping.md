# Structure Prediction track — scoping and feasibility

The CYP challenge blog post announces two tracks. The HuggingFace Space source
defines **three**. This document scopes the third one, which is live in code but
unannounced in prose.

Prepared 2026-08-06. Every claim below is traceable to a file I downloaded and
read, or to a command I ran locally; open questions are marked as such rather
than guessed at.

## 1. What the Space source actually says

From `config.py` in the Space snapshot:

```
STRUCTURE_TRACK        = "structure"
STRUCTURE_TRACK_LIVE   = True
STRUCTURE_DATASET_SIZE = 184          # "TODO: Update when final dataset is ready"
STRUCTURE_NAME         = "Structure Prediction (Pose)"
STRUCTURE_ENDPOINTS    = ["structure"]     # exactly one leaderboard
```

### Submission format — fully specified
From `submission.py`:

> Submit a **`.zip`** archive containing exactly **184 `.pdb` files**, one per
> compound, named after the compound identifier (e.g. `x00011-1.pdb`). Each file
> must be a full protein–ligand complex with the ligand residue named **`LIG`**.

Validation is shallow and worth knowing precisely: the Space checks only that the
upload is a `.zip` and that `len(zf.namelist()) == 184`. It does not check
filenames, PDB validity, chain content, or that the ligand is named `LIG` — those
are enforced downstream at scoring time, where a violation presumably scores zero
rather than erroring. The canonical stored filename is `structures.zip`, under
`structure/{username}/{submission_id}/`. Submissions are rate-limited to one per
4 hours, same as the activity tracks.

### Scoring metrics — recovered from the leaderboard code
`leaderboards.py` defines the structure leaderboard columns and the sort key:

| leaderboard column | source field | role |
|---|---|---|
| **LDDT-PLI** | `LDDT-PLI_mean` | **primary — the table is sorted by this, descending** |
| BiSyRMSD | `BiSyRMSD_mean` | secondary |
| LDDT-LP | `LDDT-LP_mean` | secondary |
| Ligand RMSD | `Ligand_RMSD_mean` | secondary (in `rename_map`, absent from the empty-frame columns) |
| Coverage | `coverage_mean` | secondary — presumably the fraction of the 184 successfully scored |

Every metric also has a `_std` counterpart, collapsed into `mean±std` in the
downloadable CSV.

**This is the CASP15/OpenStructure ligand-pose metric set**, not a bespoke one.
LDDT-PLI (local distance difference test over protein–ligand interactions) and
BiSyRMSD (binding-site-superposed symmetry-corrected RMSD) are the standard
`ost compare-ligand-structures` outputs. Two consequences:

* **Higher LDDT-PLI is better** (0–1); lower BiSyRMSD is better. The sort
  direction in the code confirms LDDT-PLI ascending=False.
* Scoring is almost certainly done with OpenStructure. That means **the pose is
  judged on its protein–ligand contact pattern, not on global fold accuracy** —
  getting the binding-site geometry and the ligand's placement within it right is
  what scores, and a perfect fold with a misplaced ligand scores near zero.

## 2. The PXR challenge is a working format template

`openadmet/pxr-challenge-train-test` contains a `structure` config with
**exactly 184 rows** and a `structure_ground_truth/` directory with **exactly 184
`.pdb` files**. The size match with `STRUCTURE_DATASET_SIZE = 184` is unlikely to
be coincidence — either the CYP figure is a placeholder copied from PXR (note the
"TODO: Update when final dataset is ready" comment), or the design is deliberately
parallel. **Treat 184 as provisional.**

`pxr-challenge_structure_TEST_BLINDED.csv` — 184 rows:

| column | example |
|---|---|
| `structure` | `x00011-1` |
| `smiles` | `NS(=O)(=O)C=1C=CC=2CCCC2C1` |
| `Molecule Name` | `OADMET-0002116` (often blank) |
| `OCNT_ID` | `OCNT-1871032` (often blank) |

`pxr-challenge_structure_TEST_identifiers.parquet` — 184 rows × 2 columns:
`Molecule Name` (str, = the `structure` id, 184 unique) and `phase` (int64, 2
distinct values). **So the structure track is split into two phases by compound**,
consistent with the CYP challenge's "starts halfway through" and its intermediate
(2026-09-24) / final (2026-11-03) deadlines.

### What a ground-truth file looks like (`x00011-1.pdb`)
Phenix-refined X-ray coordinates, 1.79 Å. 4,764 ATOM records in **two protein
chains A and B, 291 residues each** (the PXR ligand-binding-domain dimer), 54
HETATM records: `LIG A 701` (24 atoms — the actual ligand) plus DMS cryoprotectant.
Full `REMARK 3` refinement block retained; `CRYST1`/`SCALE` present; no `CONECT`
records for LIG.

Across the three files I inspected, HETATM residues are **only `LIG` and `DMS`** —
**no cofactor anywhere**. This is the single most important difference from the
CYP case, and it is why the CYP structure track is a harder problem than PXR was:

> **PXR has no cofactor. Every cytochrome P450 has an obligatory heme.**

## 3. What we know about the CYP structural target space

`cyp_pdb_inventory.csv` — every experimental PDB entry whose polymer entities map
to the four challenge isoforms (RCSB search by UniProt accession, via the
structures connector), with ligand chemistry resolved per entry.

| isoform | UniProt | PDB entries | with heme | with a drug-like ligand | best res. | median res. |
|---|---|---|---|---|---|---|
| CYP3A4 | P08684 | **122** | 122 | 116 | 1.70 Å (5VCC) | 2.55 Å |
| CYP2C9 | P11712 | 15 | 15 | 14 | 2.00 Å (1R9O) | 2.45 Å |
| CYP2D6 | P10635 | 14 | 14 | 13 | 2.10 Å (3TBG) | 2.55 Å |
| CYP1A2 | P05177 | **1** | 1 | 1 | 1.95 Å (2HI4) | 1.95 Å |
| **total unique** | | **152** | **152** | 144 | 1.70 Å | 2.55 Å |

All 152 are X-ray, all homomeric protein, released 2003-07-17 → 2025-12-03.
8 entries are heme-only (apo with respect to a drug-like ligand).

**Every single one of the 152 structures contains a heme** (150 `HEM`, 2 the
`HEC`/variant forms). There is no such thing as an apo-heme CYP structure in the
PDB. Any pose prediction that omits the heme is not a physical model of the
system.

Structural-template richness is wildly uneven, and this is strategically
important:

* **CYP3A4 is template-rich** (122 entries, best 1.70 Å) — a template-based or
  docking-into-crystal-structure approach is very well supported.
* **CYP1A2 has exactly one structure** (2HI4, 1.95 Å, with the inhibitor
  α-naphthoflavone). I cross-checked this: a UniProt-accession search returns
  `total_count = 1`, and a free-text "cytochrome P450 1A2" search returns 861 hits
  that are full-text noise, not 1A2 structures. So if the structure track includes
  CYP1A2, there is a single template for it, and co-folding has almost no
  isoform-specific structural signal to memorize.

**Which isoform(s) the track covers is not published** — `STRUCTURE_ENDPOINTS`
has one entry, `"structure"`, with no isoform in the slug. Given that Octant's
own released inhibition DRC data is CYP3A4-only and CYP3A4 is by far the most
structurally characterized, **CYP3A4 is the most likely target**, but this is
inference, not fact. See open questions.

## 4. Tool assessment: Boltz-2 vs Chai-1 for heme co-folding

The crux question was whether either tool can accept HEM as a cofactor alongside
the SMILES ligand of interest. **They differ decisively.**

### Boltz-2 — can do it, verified locally

Boltz-2's YAML schema accepts a ligand chain specified **either** by `smiles:`
**or** by `ccd:` (a PDB Chemical Component Dictionary code). Its parser resolves
`ccd:` codes against a local component library, `mols.tar`, fetched from
`huggingface.co/boltz-community/boltz-2`.

I downloaded that library (1,855,662,080 B) and checked directly:

```
mols/HEM.pkl  present   43 atoms, 50 bonds, C34H32FeN4O4, Fe present, 10 conformers
mols/HEC.pkl  present   43 atoms, 50 bonds, C34H34FeN4O4, Fe present, 10 conformers
45,227 components total
```

**HEM is a fully-formed CCD component in Boltz-2, iron included, with reference
conformers.** So the multi-ligand input the CYP problem requires is expressible:

```yaml
version: 1
sequences:
  - protein: {id: A, sequence: <503-aa CYP3A4, UniProt P08684>}
  - ligand:  {id: H, ccd: HEM}          # obligatory cofactor
  - ligand:  {id: L, smiles: '<test compound>'}
properties:
  - affinity: {binder: L}               # optional: pIC50-adjacent signal
```

Installed and running: `boltz 2.2.1`, `torch 2.13.0+cu130`, CUDA available on the
RTX 4080 Laptop (12,282 MiB). Both checkpoints (`boltz2_conf.ckpt`,
`boltz2_aff.ckpt`) downloaded. CYP3A4 is 503 residues — small for a co-folder,
and the model loaded onto the GPU using ~2.8 GB during the run, so **12 GB VRAM
is not a constraint for this target**.

An additional, non-obvious benefit: Boltz-2's affinity head predicts
log10(IC50 in μM) for a designated ligand chain. That is the *same physical
quantity* as the regression track's endpoint. It cannot be pointed at the
`ccd: HEM` chain (one affinity ligand per input, and it must be a `ligand`
chain — HEM qualifies structurally, so the binder must be named explicitly as
`L`). Worth evaluating as an orthogonal feature for the regression track, though
Boltz v2.2.x caps affinity ligands at 128 atoms.

### Chai-1 — cannot do it cleanly

Chai-1's input is a multi-entity FASTA with headers `>{entity_type}|name={id}`
and `entity_type ∈ {protein, rna, dna, ligand}`. **A `ligand` record's body is a
SMILES string. There is no CCD-code entry point.** Heme would have to be supplied
as SMILES — losing the CCD's idealized geometry and reference conformers, and
requiring the SMILES to encode the iron coordination, which RDKit-parseable heme
SMILES do inconsistently (the RCSB-served HEM SMILES writes explicit
`[Fe]` ring bonds that many parsers mangle).

Chai-1 is therefore the **second-choice** tool here, useful for consensus on the
ligand pose but not for a physically complete model. This reverses the usual
default: for a *cofactor-containing* pocket, CCD support is the deciding feature,
not sampler quality.

### Verdict on tooling

**Boltz-2 is the right tool for this track**, on the strength of CCD `HEM`
support alone. Chai-1 is viable only as a SMILES-heme approximation or as a
ligand-pose cross-check. A third route deserves equal weight: with 122 CYP3A4
crystal structures available (best 1.70 Å), **template-based placement — dock
into a real holo-CYP3A4 structure that already contains its heme — sidesteps the
cofactor problem entirely** and is likely competitive against co-folding on
LDDT-PLI, because LDDT-PLI rewards correct protein–ligand contacts and a crystal
receptor gives those for free. DiffDock-L (blind, no box required) is available
locally as a skill and is the natural pairing.

## 5. Feasibility test performed

A real end-to-end run, not a paper exercise: CYP3A4 (P08684, 503 aa) +
`ccd: HEM` + ketoconazole (a canonical CYP3A4 inhibitor) as a SMILES ligand,
with the affinity head requested.

Confirmed working:
* YAML with a CCD ligand and a SMILES ligand in the same input parses and passes
  Boltz's schema validation.
* MSA retrieval via the ColabFold MMseqs2 server succeeds (7.4 MB MSA for the
  single chain) once `api.colabfold.com` is reachable.
* Preprocessing produces the full manifest, structures, constraints and MSA
  tensors; the model then loads onto the GPU and enters diffusion at ~2.8 GB VRAM.

Two hard environment prerequisites were discovered and are recorded in
`env_and_versions.json`:

1. **`api.colabfold.com` must be allowlisted.** Without it Boltz aborts *before*
   the GPU stage with `Too many failed attempts for the MSA generation request`.
   `msa: empty` would work around it but is a documented accuracy sacrifice, not a
   resource optimization — the MSA search is CPU-side and costs no VRAM.
2. **`--no_kernels` is needed** unless `cuequivariance_ops_torch` is installed
   (~2× slower, numerically identical).

### Measured runtime — the binding constraint

This is the decision-relevant number, and it is worse than expected.

**One** CYP3A4 + HEM + ketoconazole complex, at deliberately *minimal* settings
(`--diffusion_samples 1 --recycling_steps 1 --sampling_steps 100 --no_kernels`),
ran **> 36 minutes of GPU wall time on the RTX 4080 Laptop (12 GB) without
completing its diffusion trajectory.** I stopped it at that point: the
measurement was already sufficient to make the decision, and letting it finish
would not have changed it. Confirmed on-GPU and progressing throughout (2,796 MiB
allocated, 16–55% utilization), with MSA and preprocessing complete well before
the diffusion stage began — so this is diffusion cost, not setup cost.

Extrapolation to a full submission, using >36 min as a **lower bound** per
complex:

| scenario | per complex | 184 complexes |
|---|---|---|
| measured floor, minimal settings, this GPU | > 36 min | **> 110 GPU-hours** (~4.6 days continuous) |
| paper-faithful (`--recycling_steps 3 --diffusion_samples 5`) | several× higher | plausibly **500+ GPU-hours** |

The MSA is computed once for the shared receptor and cached, so MSA cost does not
scale with 184 — but that is a small part of the total. `--no_kernels` costs
roughly 2×, so installing `cuequivariance_ops_torch` would help materially but
not change the order of magnitude.

**Conclusion: co-folding all 184 complexes is not viable on this laptop GPU.**
Three routes remain, in order of practicality:

1. **Template-based placement into a holo CYP3A4 crystal structure** (122
   available, best 1.70 Å, all with heme in place). Docking is seconds-to-minutes
   per ligand rather than tens of minutes, sidesteps the cofactor problem
   entirely, and scores well on a contact-based metric like LDDT-PLI. **This
   should be the primary route.**
2. Rent A100/H100 time for a Boltz-2 run, if co-folding is wanted as a
   differentiator. At datacentre-GPU speed with fast kernels this becomes a
   single-digit-GPU-day job rather than a laptop-infeasible one.
3. A hybrid: template-dock everything, then co-fold only the subset where the
   ligand is too large or too unusual for the crystal pocket.

## 6. Verdict

**Technically feasible via template-based docking; NOT feasible by local
co-folding. Lowest-priority of the three tracks — enter only if the activity
tracks are in good shape.**

The two decision-relevant facts, both established by direct measurement:

1. **Heme is not a blocker.** Boltz-2 accepts `ccd: HEM` alongside a `smiles:`
   ligand in the same input — verified: the YAML parses and `mols/HEM.pkl` is a
   complete 43-atom, iron-containing component with 10 reference conformers, from
   Boltz's 45,227-component CCD library. Chai-1 cannot do this (SMILES-only
   ligands), which reverses the usual tool preference for cofactor-containing
   pockets.
2. **Local co-folding runtime rules it out.** > 36 GPU-minutes for one complex at
   minimal settings ⇒ **> 110 GPU-hours for 184**, on this machine. Template
   docking into one of the 122 holo CYP3A4 crystal structures is the practical
   route, and is likely competitive anyway because LDDT-PLI scores
   protein–ligand contacts, which a crystal receptor supplies for free.

Supporting points:

* The submission format is fully known (zip of exactly 184 `LIG`-named complex
  PDBs) and the scoring metrics are recovered (LDDT-PLI primary, sorted
  descending). There is no format risk.
* VRAM is not the constraint — a 503-residue single-chain target used ~2.8 GB of
  the 12 GB card. Time is the constraint.
* The field will likely be thin: unannounced in the blog post, a different
  toolchain from the activity tracks, and real compute cost.

**Recommended priority: after both activity tracks.** The track is unannounced,
starts mid-challenge, and carries a compute bill the activity tracks do not.
Revisit once the isoform and the true dataset size are published.

## 7. Specific blocking questions

Ordered by how much they change the plan. Items 1–3 should be resolved at or soon
after the 2026-08-17 data drop.

1. **Which isoform(s) does the structure track cover?** Not published. Decides
   everything: CYP3A4 means 122 templates and a tractable problem; CYP1A2 means a
   single template (2HI4) and near-total reliance on co-folding.
2. **Is the count really 184?** The constant carries a
   "TODO: Update when final dataset is ready" comment and exactly matches the PXR
   track's size. The validator hard-rejects any zip that is not exactly
   `STRUCTURE_DATASET_SIZE` files, so a stale value is a submission-blocking risk.
3. **Will a reference receptor be provided, or must we produce the protein too?**
   The format demands "a full protein–ligand complex", which implies we supply
   coordinates for both. If a canonical receptor is published, template-based
   docking becomes strictly better than co-folding.
4. **Must the heme be present in the submitted PDB, and under what residue name?**
   The spec names only `LIG`. If the scorer's protein-model parse expects HEM as a
   HETATM cofactor, omitting it may perturb LDDT-LP; if the scorer ignores
   non-`LIG` HETATMs, it is cosmetic. Physically it should be included either way.
5. **Exact LDDT-PLI configuration.** OpenStructure's
   `compare-ligand-structures` has consequential options (inclusion radius,
   `--substructure-match`, symmetry handling, whether the cofactor counts as part
   of the binding site). These change scores materially and are not published.
6. **How is `Coverage` defined and does a missing/unscorable file cost more than
   zero?** Determines whether it is better to submit a low-confidence pose or an
   empty placeholder for a hard compound.
7. **Which compounds fall in phase 1 vs phase 2?** The PXR analogue shipped a
   `phase` column in the identifiers parquet; expect the same and plan the two
   deadlines around it.

## 8. Files

| file | contents |
|---|---|
| `cyp_pdb_inventory.csv` | 152 experimental PDB entries for the 4 isoforms: resolution, method, heme presence and comp-id, drug-like ligand ids/names/SMILES, all ligand comp-ids, citation/DOI/PubMed |
| `pxr_template/` | PXR structure-track format template: blinded CSV, identifiers parquet, 3 ground-truth PDBs |
| `boltz_test/cyp3a4_hem_ktz.yaml` | working Boltz-2 input demonstrating `ccd: HEM` + SMILES ligand + affinity head |
| `boltz_cache/mols/HEM.pkl` | the verified heme CCD component (from Boltz's 45,227-component library) |
| `env_and_versions.json` | environment versions, GPU/CUDA state, external asset checksums, known sandbox issues |
