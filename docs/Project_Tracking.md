<a id="session-handoff"></a>
## Session Handoff — 2026-07-09 (updated)

**(a) What was just finished:** Phase 5 (EDA) is fully complete for all 4
datasets (as before). In addition, this session resolved both Phase 6
pre-conditions that were previously open — see decisions (1)-(3) below.
`data/processed/ISIC_Archive_2/feature_whitelist.md` updated accordingly.

**(b) Immediate next action on resume:** Both blocking decisions are now
resolved, and Phase 6 Stage 1 scope (architecture, imbalance handling,
script structure, seed count, val/test discipline) has been proposed and
approved — see decisions (1)-(4) below. Stage 1 code is being written now
(`src/models/`, `src/evaluation/`).

**(c) Decisions made this session:**

### 1. ISIC Archive 2 sparse-field handling

**Resolved: exclude all 6 as source-leak risk, not impute-and-flag.**

Before choosing impute-vs-drop, ran a crosstab of `is_null(field)` vs.
`attribution` (the already-excluded source-identity column; 3 values:
Hospital Clínic de Barcelona 12,302 rows, ViDIR Group/Vienna 9,873 rows,
Anonymous 2,901 rows) for each of the 6 sparse fields, across all 25,076
rows (train+val+test combined):

| field | overall missing % | % present, Hospital Clínic | % present, ViDIR | % present, Anonymous |
|---|---|---|---|---|
| `anatom_site_3` | 87.56% | 0.00% | 23.45% | 27.75% |
| `anatom_site_4` | 99.05% | 0.00% | 0.00% | 8.20% |
| `anatom_site_5` | 99.84% | 0.00% | 0.00% | 1.34% |
| `family_hx_mm` | 97.83% | 0.00% | 0.00% | 18.75% |
| `personal_hx_mm` | 97.79% | 0.00% | 0.00% | 19.10% |
| `clin_size_long_diam_mm` | 97.79% | 0.00% | 0.00% | 19.13% |

**Result:** strong, near-deterministic correlation. Five of six fields are
*never* populated for Hospital Clínic or ViDIR rows — they only ever
appear (and only partially) for Anonymous rows. `anatom_site_3` is
slightly less extreme but still perfectly excludes Hospital Clínic
(0.00% present). Presence of any of these fields is a near-perfect proxy
for source institution — the same leak category as `attribution` itself
(already excluded, see `feature_whitelist.md`), not ordinary sparsity.

**Decision:** all 6 fields excluded entirely from Phase 6 metadata/fusion
models, not imputed-and-flagged. They remain in
`data/processed/ISIC_Archive_2/feature_whitelist.md` as documented,
excluded columns with the crosstab evidence recorded per-field.

### 2. Final ISIC Archive 2 Phase 6 Stage 1 baseline metadata feature list

`age_approx` (1.91% missing), `sex` (1.53% missing), `anatom_site_1`
(6.43% missing), `anatom_site_general` (10.17% missing). These are
reliably populated and showed no attribution-correlation pattern.

(`clin_size_long_diam_mm` was initially proposed for this list but
corrected out — it is one of the 6 fields excluded per decision 1 above,
97.79% missing, not reliably populated.)

`anatom_site_2`, `anatom_site_special`, and `dermoscopic_type` remain
allowed-in-principle in the whitelist but are deferred out of the active
Stage 1 set pending their own missingness/attribution check (not yet run).

### 3. Phase 6 input resolution

**224×224, aspect-ratio-preserving resize+pad, applied at data-load time
via the training pipeline's transform — not a stored resized copy.**

Rationale:

- 224×224 matches standard pretrained ImageNet backbone input expectations
  (ResNet/EfficientNet/ViT family), avoiding architecture-specific
  preprocessing mismatches.
- Applying the resize as a load-time transform (not a stored resized
  copy) is consistent with the project's existing no-image-copying
  policy (`PROJECT_PLAN.md` confirmed design decisions) — `data/raw/`
  stays untouched and no duplicate resized image set is written to disk.
- Aspect-ratio-preserving resize+pad (vs. naive stretch-to-square) is
  required given the documented image-dimension heterogeneity across and
  within datasets (HAM10000 fixed 600×450; PAD-UFES-20 167px–3,120px;
  ISIC Archive 2 bimodal ~600×450/1024×1024 — see `reports/eda/eda_summary.md`).
  Naive stretch would distort lesion shape/aspect ratio non-uniformly
  depending on source dataset and resolution cluster, which is itself a
  potential confound (e.g. correlated with the ISIC Archive 2 attribution
  clusters); resize+pad preserves true proportions at the cost of some
  border padding.

### 4. Phase 6 Stage 1 (Baseline Model Development, PAD-UFES-20) — scope

**Approved 2026-07-09.**

- **Architecture:** EfficientNet-B0 (ImageNet-pretrained, fine-tuned) for
  the image-only branch — chosen over ResNet-50 given PAD-UFES-20's small
  size (~2,298 images): EfficientNet-B0 has ~5.3M params vs ResNet-50's
  ~25.6M, materially lower overfitting risk on a dataset this size (with
  15:1+ class imbalance already documented), smaller memory/compute
  footprint suited to free-tier GPU (Kaggle/Colab), and comparable
  ImageNet top-1 accuracy (~77.1% vs ~76.1%) — no real accuracy tradeoff
  for the efficiency gain. Metadata-only branch: simple MLP (2-3 hidden
  layers) on PAD-UFES-20's whitelisted tabular features.
- **Class-imbalance handling:** class-weighted cross-entropy loss,
  weights = inverse class frequency computed from the train split only.
  Primary metrics: macro-F1 (headline) + per-class F1 + confusion matrix,
  per `PROJECT_PLAN.md`'s metrics rule; plain accuracy reported only as a
  secondary reference number.
- **Val/test discipline** (explicit, to prevent test-set leakage into
  tuning decisions): the `val` split is used for model selection, early
  stopping, and checkpoint picking during training. The `test` split is
  **not touched for any purpose** — no tuning decision, no early-stopping
  signal, no architecture/hyperparameter choice — until the single final
  evaluation run at the end of Stage 1, after all training/tuning is
  complete. This is enforced explicitly in `train.py`'s design (it never
  loads the test split) and `evaluate.py` is the only script that reads
  it.
- **Script structure:** `src/models/image_model.py` (EfficientNet-B0
  wrapper + 224x224 resize+pad transform), `src/models/metadata_model.py`
  (MLP + preprocessing fit on train split only), `src/models/dataset.py`
  (shared PyTorch `Dataset`, resolves `image_path` into `data/raw/`, no
  copying), `src/models/train.py` (`--branch {image,metadata} --seed`,
  trains one model per invocation using train+val only, saves checkpoint
  + per-epoch metrics to `logs/PAD_UFES20/`), `src/evaluation/evaluate.py`
  (loads a checkpoint, runs on val or test split, writes macro-F1,
  per-class F1, confusion matrix to `reports/PAD_UFES20/`).
- **Seeds:** 3 seeds per branch (image and metadata separately) = 6 runs
  total for Stage 1, results reported as mean +/- std, per
  `PROJECT_PLAN.md`'s "3-5 per experiment" reproducibility rule.

---

<a id="quick-summary"></a>
## Quick Summary

*Orientation aid — added during the docs readability pass. All detailed
sections below are unchanged and remain the source of truth.*

- **Current phase:** Phase 5 (EDA) complete (2026-07-09); Phase 4 (Dataset
  Preparation) formally closed 2026-07-08. Phase 6 (Baseline Model
  Development) Stage 1 scope approved 2026-07-09, code now being written.
- **Both Phase 6 blocking decisions resolved 2026-07-09:** ISIC Archive 2
  sparse-field handling (6 fields excluded as source-leak risk) and input
  resolution (224×224 resize-and-pad at load time).
- **Cross-dataset image overlap** between HAM10000 and both ISIC archives
  was found and fixed via exclusion lists (fix b), not a global re-split.
- **`biopsed` (PAD-UFES-20)** and several ISIC Archive 2 fields
  (`diagnosis_confirm_type`, `concomitant_biopsy`, `melanocytic`,
  `attribution`, etc.) were formally verified as leakage/label-source
  features and excluded from every dataset's `feature_whitelist.md`.
- **Stage 1 baseline:** EfficientNet-B0 (image) + MLP (metadata) on
  PAD-UFES-20, class-weighted loss, macro-F1 headline metric, 3 seeds per
  branch.
- **Documentation cleanup** (2026-07-08) archived 2 stale docs to
  `_archive/` and corrected `Dataset_Strategy.md` in place.
- Two items remain **outstanding, not yet actioned** (see "Documentation
  Cleanup" below): a `src/common/paths.py` config-duplication refactor,
  and archiving superseded re-run logs.

---

## Table of Contents

1. [Session Handoff — 2026-07-09](#session-handoff)
2. [Quick Summary](#quick-summary)
3. [Project Overview](#project-overview)
4. [Project Roadmap](#project-roadmap)
5. [Progress Tracker](#progress-tracker)
6. [Current Milestone](#current-milestone)
7. [Cross-Dataset Verification (2026-07-08)](#cross-dataset-verification)
8. [Phase Closure: Dataset Preparation — CLOSED (2026-07-08)](#phase-closure)
9. [Phase 5 (EDA) — Kickoff / Completed](#phase-5-eda)
10. [PROJECT_PLAN.md Current Phase — Status (2026-07-08)](#project-plan-status)
11. [Next Milestone](#next-milestone)
12. [`biopsed` Leakage/Shortcut Audit (2026-07-08)](#biopsed-audit)
13. [Label-Leakage Decision — Full Feature Exclusion List](#label-leakage-decision)
14. [Documentation Cleanup (2026-07-08)](#documentation-cleanup)
15. [Important Research Decisions](#important-research-decisions)
16. [Future Improvements](#future-improvements)
17. [Phase 6 Stage 1 — Code Status (2026-07-09)](#phase-6-stage-1-code-status)
18. [Docs Readability Pass (2026-07-09)](#docs-readability-pass)
19. [Phase 6 Stage 1 — COMPLETE, PAD-UFES-20 (2026-07-13)](#phase-6-stage-1-complete)
20. [Sequencing Decision — HAM10000 Baseline Before Phase 7 (2026-07-13)](#sequencing-decision-ham10000-vs-phase-7)
21. [HAM10000 Phase 6 Stage 1 — Scope Approved (2026-07-13)](#ham10000-stage-1-same-recipe-decision)
22. [HAM10000 Stage 1 — COMPLETE (2026-07-16)](#ham10000-stage-1-complete)
23. [Phase 7 Stage 1 — Late Fusion Scope Approved, PAD-UFES-20 Only (2026-07-16)](#phase-7-stage-1-scope-approved)

---

<a id="project-overview"></a>
## Project Overview

- **Research Title:** Multimodal Skin Disease Classification System Using Skin Images and Clinical Metadata
- **Research Goal:** Develop a multimodal deep learning framework combining skin lesion images and clinical metadata that improves classification performance and reliability over image-only approaches.
- **Publication Goal:** Master's thesis + conference/journal publication + scholarship portfolio.
- **Canonical plan:** `docs/PROJECT_PLAN.md` — this file (`Project_Tracking.md`) is the living status/decision log that tracks progress against it. The phase list and statuses below are kept phase-wise aligned to `PROJECT_PLAN.md`.
- **Current Phase:** Phase 5 — Exploratory Data Analysis (EDA), completed 2026-07-09. Dataset Preparation (Phase 4) formally closed 2026-07-08. Both Phase 6 pre-conditions (ISIC Archive 2 sparse-field handling, input resolution choice) resolved 2026-07-09 — see Session Handoff above. Phase 6 (Baseline Model Development) Stage 1 scope to be proposed and reviewed before model code starts.

---

<a id="project-roadmap"></a>
## Project Roadmap

*(Phase-wise, per `docs/PROJECT_PLAN.md` — Phase 4's sub-steps are this project's Dataset Preparation work; Phases 5-10 are `PROJECT_PLAN.md`'s "Future phases" list verbatim.)*

1. Planning
2. Literature Review
3. Dataset Collection
4. Dataset Preparation — Audit → Cleaning → Standardization → Splitting → cross-dataset leakage fix → label-leakage feature whitelist → docs cleanup
5. Exploratory Data Analysis (EDA) — class/demographic/Fitzpatrick distributions, missing-value visualization, sample image grids, image dimension distribution
6. Baseline Model Development (PAD-UFES-20) — image-only, metadata-only, using each dataset's `feature_whitelist.md`, class-weighted loss
7. Multimodal Model Development — late fusion, then cross-attention fusion
8. Experiments & Evaluation — PAD-UFES-20↔HAM10000 generalization (headline result), HAM10000→ISIC external validation (with exclusions applied), Fitzpatrick fairness analysis, bootstrap significance testing
9. Thesis Writing Support
10. arXiv preprint → workshop/journal submission

---

<a id="progress-tracker"></a>
## Progress Tracker

| Phase | Status | Completion Date | Description |
|---|---|---|---|
| 1. Planning | ✅ Completed | 2026-06-29 | README, Research_Plan, Dataset_Strategy, AI_Assistant_Instructions authored (Research_Plan.md since archived to `_archive/`, superseded by this file) |
| 2. Literature Review | 🟡 In Progress | — | 3 core papers reviewed (Mridha & Islam 2026; Suresh et al. 2026 TG-CAVNet; Watson et al. 2026); comparative analysis complete; ongoing as new papers are added |
| 3. Dataset Collection | ✅ Completed | — | PAD-UFES-20, HAM10000, ISIC Archive 1 & 2 acquired into `data/raw/` |
| 4. Dataset Preparation — PAD-UFES-20 | ✅ Completed | 2026-07-07 | Full audit (12 modules) + cleaning (schema standardization, label mapping, patient-wise stratified split, documentation) |
| 4. Dataset Preparation — HAM10000 | ✅ Completed | 2026-07-07 | Audit (11 modules) + cleaning (schema standardization, label mapping shared with PAD-UFES-20 where diseases overlap, lesion-wise stratified split, documentation) |
| 4. Dataset Preparation — ISIC Archive 1 | ✅ Completed | 2026-07-07 | Audit (folder-derived, no metadata.csv) found 155 images filed under conflicting class labels — excluded during cleaning. Kept archive's own Test split, carved val out of Train. |
| 4. Dataset Preparation — ISIC Archive 2 | ✅ Completed | 2026-07-07 | Audit (8 modules) + cleaning (diagnosis_3-based label mapping, group-wise split keyed on lesion_id/image_id fallback) |
| 4. Dataset Preparation — Cross-dataset leakage fix | ✅ Completed | 2026-07-08 | Fix (b) chosen; `external_validation_exclusions.csv` generated for ISIC Archive 1 & 2 |
| 4. Dataset Preparation — Label-leakage decision + `feature_whitelist.md` ×4 | ✅ Completed | 2026-07-08 | All leakage/label-source columns formally verified with real numbers and excluded; reviewed and approved by you |
| 4. Dataset Preparation — Docs cleanup | ✅ Completed | 2026-07-08 | `_archive/` created, 2 stale docs archived, `Dataset_Strategy.md` corrected in place, `README.md` updated |
| **4. Dataset Preparation — Phase closure** | **✅ CLOSED** | **2026-07-08** | All 4 `PROJECT_PLAN.md` Current Phase items done and reviewed |
| **5. EDA** | **✅ Completed** | **2026-07-09** | Scope approved 2026-07-08 (incl. approved image-dimension addition). `src/eda/` code executed for all 4 datasets + cross-dataset comparison; all outputs verified non-empty in `reports/eda/`; 5 notebooks (`notebooks/01-05_eda_*.ipynb`) and `reports/eda/eda_summary.md` written. Key finding: ISIC Archive 2's bimodal image resolution (600x450 / 1024x1024 clusters) visually confirms its documented image-source overlap with HAM10000. |
| 6. Baseline Model Development | ✅ Completed — both datasets' Stage 1 baselines done | 2026-07-16 | Pre-conditions resolved 2026-07-09 (sparse-field exclusion, 224×224 resize+pad). PAD-UFES-20: all 6 Stage 1 runs completed and verified 2026-07-13 (image mean macro-F1 0.5703+/-0.0130, metadata mean macro-F1 0.5762+/-0.0072) — see "Phase 6 Stage 1 — COMPLETE" entry. HAM10000: all 6 Stage 1 runs completed and verified 2026-07-16, same recipe (image mean macro-F1 0.6940+/-0.0041, metadata mean macro-F1 0.2521+/-0.0104) — see "HAM10000 Stage 1 — COMPLETE" entry. Sequencing decision (full HAM10000 baseline before Phase 7) fulfilled. Next: Phase 7 late fusion design. |
| 7. Multimodal Model Development | 🔄 In progress — Stage 1 (late fusion) code written, PAD-UFES-20 only | 2026-07-16 | Scope approved and implemented — see "Phase 7 Stage 1 — Late Fusion Scope Approved" entry. Smoke-tested locally on a real subset; Kaggle notebook (`notebooks/pad_ufes20_fusion_kaggle_notebook.md`) ready for the 3 seed runs, pending Stage 1 checkpoint upload as a new Kaggle dataset. Stage 2 (cross-attention fusion) not started. |
| 8. Experiments & Evaluation | ⏳ Pending | — | Includes external validation (HAM10000/ISIC, with exclusions applied) and bootstrap significance testing |
| 9. Thesis Writing Support | ⏳ Pending | — | |
| 10. arXiv preprint / submission | ⏳ Pending | — | |

---

<a id="current-milestone"></a>
## Current Milestone

All four datasets have completed dataset preparation (audit → cleaning), each leakage-verified:

- **PAD-UFES-20**: 12-module audit; cleaning with schema standardization (corrected a discovered `UNK`-as-missing bug in symptom columns), 6-code label mapping, patient-wise stratified 70/15/15 split, full documentation.
- **HAM10000**: 11-module audit; cleaning with schema standardization (corrected a discovered `unknown`-as-missing bug in `sex`/`localization`), 7-code label mapping (3 names shared with PAD-UFES-20: Basal Cell Carcinoma, Melanoma, Nevus), **lesion-wise** split (no patient identifier exists in this dataset).
- **ISIC Archive 1**: folder-derived audit (no metadata.csv — class label comes from folder name) found a genuine data-quality defect: 155 images filed under two conflicting class labels, systematically (78x melanoma↔seborrheic keratosis, 77x actinic keratosis↔nevus) — these were excluded during cleaning rather than arbitrarily assigned a label. Cleaning kept the archive's own Test split untouched and carved a validation set out of Train.
- **ISIC Archive 2**: 8-module audit (flat `images/` + rich 27-column metadata.csv); cleaning uses `diagnosis_3` (the finest populated diagnosis level) as the class label via a taxonomy shared with ISIC Archive 1 (`src/data_cleaning/common_label_mapping.py`), and a group-wise split keyed on `lesion_id` (93.4% coverage) falling back to `image_id` as a singleton group otherwise (`patient_id` covers <2% of rows, so true patient-wise splitting isn't possible here).
- Literature review of 3 directly relevant multimodal skin-lesion papers completed, with a comparative analysis identifying concrete gaps (multi-class fusion, leakage auditing, combined fusion sophistication) our thesis can address.

---

<a id="cross-dataset-verification"></a>
## Cross-Dataset Verification (2026-07-08)

A final, independent cross-dataset verification (`docs/Dataset_Preparation_Final_Report.md`)
was run after all four datasets' individual audit → cleaning pipelines were
complete. It found each dataset internally consistent, reproducible, and
leakage-free on its own split — **except for one critical, previously
undetected issue**:

- **HAM10000, ISIC Archive 1, and ISIC Archive 2 substantially share the same
  underlying images** (HAM10000 ↔ ISIC Archive 2: 98.6% image overlap; ISIC
  Archive 1 ↔ ISIC Archive 2: 81.7%; HAM10000 ↔ ISIC Archive 1: 66.5%).
  PAD-UFES-20 has zero overlap with any other source and is unaffected.
- Because each dataset's split was computed independently, 40–46% of the
  overlapping images land in *different* splits across datasets (e.g.
  HAM10000-train / ISIC Archive 2-test) — meaning **the planned use of the
  ISIC archives as an external validation set against a HAM10000-trained
  model is currently invalid** for the overlapping images.
- **This blocks declaring dataset preparation fully COMPLETE** for the
  external-validation use case, even though PAD-UFES-20 alone is unaffected
  and ready for EDA/Baseline Model Development now.
- Decision needed: either (a) build a global cross-dataset dedup step so
  every physical image gets one split assignment across all three
  overlapping datasets, or (b) filter external-validation evaluation to only
  the non-overlapping subset and document that filtering. See report §9 for
  detail.

### Decision Made (2026-07-08): Fix (b) — Restrict External-Validation Claims

**Chosen: fix (b), not fix (a).** Rationale:

- Fix (a) (global re-split) would require re-deriving train/val/test for
  HAM10000, ISIC Archive 1, and ISIC Archive 2 jointly, discarding the
  already-completed, individually-verified, reproducible splits for all
  three datasets — a large amount of rework for a problem that only affects
  one specific use case (cross-dataset external validation), not each
  dataset's own internal training/evaluation.
- Fix (b) (filter at evaluation time) preserves every dataset's existing,
  already-audited internal split untouched, and only restricts the specific
  claim that was actually invalid — "this ISIC archive is *external*
  validation for a HAM10000-trained model." It only removes the images that
  are genuinely non-independent for that specific comparison.
- Fix (b) is also more transparent for the thesis write-up: the exact
  exclusion counts are documented per archive (below) and reproducible via
  `src/data_cleaning/cross_dataset_leakage_filter.py`, rather than being
  folded invisibly into a new global split.

**Implementation:** `src/data_cleaning/cross_dataset_leakage_filter.py`
reads the existing `metadata_{train,val,test}.csv` for HAM10000, ISIC
Archive 1, and ISIC Archive 2 (read-only — no existing split file is
modified) and writes one new reference file per ISIC archive:
`data/processed/ISIC_Archive_1/external_validation_exclusions.csv` and
`data/processed/ISIC_Archive_2/external_validation_exclusions.csv`, each
listing the `image_id`s to drop from that archive before it is used as
external validation against a HAM10000-trained model. Full detail and row
counts are in each archive's own `dataset_description.md`.

| Dataset | Total images | Overlapping with HAM10000 (excluded for HAM10000-external-validation use) | Remaining, valid for HAM10000-external-validation |
|---|---|---|---|
| ISIC Archive 1 | 2,047 | 1,362 | 685 |
| ISIC Archive 2 | 25,076 | 9,873 | 15,203 |

This exclusion applies **only** when an ISIC archive is used as external
validation against a **HAM10000-trained** model. It does not apply to:

- each archive's own internal train/val/test evaluation (unaffected, unchanged), or
- external validation against a model trained only on **PAD-UFES-20** (zero
  overlap with any other dataset, so no exclusion is needed there).

**Also recorded (documented caveat, no action taken):** of the 1,673 images
shared between ISIC Archive 1 and ISIC Archive 2, 3 have disagreeing
`disease_label` values between the two archives — `ISIC_0028619` (Nevus in
Archive 1 vs. Actinic Keratosis in Archive 2) and `ISIC_0011126` /
`ISIC_0011118` (Seborrheic Keratosis in Archive 1 vs. Melanoma in Archive
2). This is a pre-existing annotation disagreement inherited from the
source archives, negligible in volume (0.18% of the shared images), not
introduced by this project's cleaning, and not fixed — only flagged so it
is not mistaken for a bug if noticed later.

---

<a id="phase-closure"></a>
## Phase Closure: Dataset Preparation — CLOSED (2026-07-08)

All 4 `PROJECT_PLAN.md` Current Phase items confirmed done and reviewed by
you, including the `feature_whitelist.md` review — `attribution` in
particular confirmed as a genuine dataset-source leak (not just a low-stakes
judgment call), given its exact match with the documented HAM10000↔ISIC
Archive 2 overlap count. Dataset Preparation is formally complete for all 4
datasets.

---

<a id="phase-5-eda"></a>
## Phase 5 (EDA) — Kickoff (2026-07-08)

Scope approved: class/demographic/Fitzpatrick distributions, missing-value
visualization, sample image grids, plus an approved addition — image
width/height/aspect-ratio distribution (sampled 250 images/dataset) to
inform Phase 6 input-resolution choice — and an approved optional
cross-dataset shared-label comparison. EDA restricted to each dataset's
`feature_whitelist.md` columns + `disease_label` (+ `fitspatrick` for
PAD-UFES-20 only) and to `metadata_train.csv` (val/test touched only for
the split-balance check, reusing existing split counts). Code in `src/eda/`,
outputs in `reports/eda/<Dataset>/`, notebooks in `notebooks/`.

### Phase 5 — Completed (2026-07-09)

All 5 EDA scripts (`eda_pad_ufes20.py`, `eda_ham10000.py`,
`eda_isic_archive_1.py`, `eda_isic_archive_2.py`, `eda_cross_dataset.py`) ran
successfully; every output file verified non-empty. 5 thin notebooks
(`notebooks/01-05_eda_*.ipynb`) import and call each script, then display
the generated figures inline with findings commentary. Full narrative
summary in `reports/eda/eda_summary.md`. Highlights:

- Every dataset needs class-weighted loss (imbalance 15:1 to ~393:1 in-train).
- ISIC Archive 1 has 0 usable metadata columns — image-only by necessity.
- ISIC Archive 2's image resolution is bimodal (600x450 / 1024x1024ish),
  matching its two `attribution` sources exactly — an independent visual
  confirmation of the documented HAM10000 image overlap, found via a
  completely different signal (pixel dimensions vs. `image_id` matching).
- Two items flagged, not decided: ISIC Archive 2 sparse-field handling
  (impute-and-flag vs. drop) and the Phase 6 input-resolution choice given
  the heterogeneity found across all 4 datasets.

---

<a id="project-plan-status"></a>
## PROJECT_PLAN.md Current Phase — Status (2026-07-08): ALL 4 ITEMS DONE

1. Label-leakage decision recorded — DONE (see "Label-Leakage Decision" above).
2. `feature_whitelist.md` per dataset — DONE (all 4 `data/processed/<Dataset>/feature_whitelist.md` created).
3. Docs cleanup pass — DONE (see "Documentation Cleanup" above).
4. Full verification printout — DONE (re-run complete with feature whitelist included, printed in full for all 4 datasets).

Per PROJECT_PLAN.md's ground rules, EDA/model code/Phase 5+ work still
awaits your review of this closure before starting.

---

<a id="next-milestone"></a>
## Next Milestone

1. ~~Resolve the cross-dataset image overlap~~ — **Resolved 2026-07-08** via
   fix (b), see "Decision Made" above. Any future protocol using ISIC
   Archive 1/2 as external validation against a HAM10000-trained model must
   apply the corresponding `external_validation_exclusions.csv` filter first.
2. ~~Add a leakage/shortcut audit step for PAD-UFES-20 metadata (particularly `biopsed`)~~ — **Resolved 2026-07-08**, see "`biopsed` Leakage/Shortcut Audit" below.
3. Begin Baseline Model Development: image-only classifier on PAD-UFES-20 (Experiment 1) — unaffected by item 1, can start immediately.
4. EDA across all four now-processed datasets, informed by the disease-taxonomy overlap established in cleaning (PAD-UFES-20/HAM10000/ISIC Archive 1/ISIC Archive 2 share several `disease_label` values verbatim) and by the cross-dataset overlap finding above.

---

<a id="biopsed-audit"></a>
## `biopsed` Leakage/Shortcut Audit (2026-07-08)

Per Watson et al. (2026)'s finding that clinically-derived-after-diagnosis
fields can let a model shortcut classification without genuine image or
symptom signal, PAD-UFES-20's `biopsed` field ("diagnosis confirmed via
biopsy vs. clinical judgment only") was checked against `disease_label`
across all 2,298 images.

**Finding:** every malignant-labeled image (Basal Cell Carcinoma, Melanoma,
Squamous Cell Carcinoma — 1,089 images total) has `biopsed=True`, with zero
exceptions, while non-malignant images (Actinic Keratosis, Nevus, Seborrheic
Keratosis) are biopsied only 21% of the time overall (ranging 6.4%–24.6% by
disease). This holds independently in each of train/val/test. Phi
coefficient = 0.80, chi-square = 1474.5 (n=2,298) — a very strong,
near-deterministic association for the malignant class. Full contingency
table in `data/processed/PAD_UFES20/dataset_description.md`.

**Decision: exclude `biopsed` from all model input features.** It reflects
real clinical practice (biopsy is near-mandatory to confirm cancer, optional
for benign lesions), not a data error, but using it as a training feature
would let a model trivially separate malignant/non-malignant via this one
flag instead of the image or genuine pre-diagnosis clinical signal —
defeating the purpose of the multimodal classification research. `biopsed`
remains in the metadata CSVs as a retained column for dataset
documentation/analysis only (e.g. sanity-checking that a trained model isn't
implicitly recovering it), never as a model input.

**Related, not yet audited:** the same quick check applied informally to
HAM10000's `diagnosis_confirm_type` field shows an analogous pattern (all
malignant — Basal Cell Carcinoma/Melanoma — images are confirmed via `histo`
with 0 exceptions, while non-malignant images are split across `histo`,
`consensus`, `follow_up`, `confocal`). This was not in scope for this audit
(scoped to PAD-UFES-20's `biopsed` per this milestone item) and needs its
own dedicated check before HAM10000 metadata is used in any model — flagged
here so it isn't missed.

---

<a id="label-leakage-decision"></a>
## Label-Leakage Decision — Full Feature Exclusion List (2026-07-08, extended per PROJECT_PLAN.md)

Following `docs/PROJECT_PLAN.md`'s expanded list of label-leakage/label-source
fields (beyond the original `biopsed`-only audit), every named field was
formally verified with real numbers before exclusion. Full detail and
per-dataset column-by-column reasoning is in each dataset's new
`feature_whitelist.md` (Current Phase item 2). Summary:

| Dataset | Column | Type | Evidence |
|---|---|---|---|
| PAD-UFES-20 | `biopsed` | leakage | phi=0.80, chi2=1474.5, n=2,298 — 100% of malignant vs. 21% of non-malignant |
| PAD-UFES-20 | `diagnostic_code` | label-source | verified 1:1 with `disease_label` (6/6 codes unambiguous) |
| HAM10000 | `diagnosis_confirm_type` | leakage | **formally verified** (was only informal before): phi=0.41, chi2=1700.67, n=10,015 — 100% of malignant (1,627/1,627) confirmed via `histo` with zero exceptions; non-malignant split across all 4 confirm types |
| HAM10000 | `diagnostic_code` | label-source | verified 1:1 with `disease_label` (7/7 dx codes unambiguous) |
| ISIC Archive 1 | `class_label` | label-source | verified 1:1 with `disease_label` (9/9 classes unambiguous — `disease_label` is a direct rename) |
| ISIC Archive 2 | `diagnosis_confirm_type` | leakage | **formally verified, not assumed** — phi=0.36, chi2=3171.74, n=25,076. Weaker/different shape than HAM10000: malignant cases are never confirmed via "serial imaging" or "single image consensus" (0/8,473), but histopathology is used for both malignant (90.7%) and non-malignant (55.4%) cases — **not** as clean a separator as PROJECT_PLAN.md's phrasing implied. |
| ISIC Archive 2 | `diagnosis_1`, `diagnosis_2`, `diagnosis_3` | label-source | `diagnosis_3` verified 1:1 with `disease_label`; `diagnosis_1`/`diagnosis_2` are fully-populated coarser levels of the same hierarchy |

### Extensions beyond PROJECT_PLAN.md's literal list (flagged explicitly, not applied silently)

- **`diagnosis_4`, `diagnosis_5` (ISIC Archive 2)** — per your instruction, excluded as label-source on the same reasoning as `diagnosis_1`-`diagnosis_3` (finer levels of the same diagnostic hierarchy the label was built from; 96.11%/98.35% missing when not populated, but label-derived when present).
- **`concomitant_biopsy` (ISIC Archive 2)** — discovered during this audit, not named in PROJECT_PLAN.md. Produces the *identical* contingency table to `diagnosis_confirm_type == "histopathology"` (same phi=0.36) — appears to be a duplicate encoding of the same confirmation-method signal. Excluded under the same leakage reasoning as `diagnosis_confirm_type`.
- **`melanocytic` (ISIC Archive 2)** — discovered during this audit, not named in PROJECT_PLAN.md or anticipated in the original request. **Perfect deterministic split** of `disease_label` (Melanoma/Nevus = 100% True, all other 7 classes = 100% False, zero exceptions) — a coarser restatement of the label, more severe than any previously-found leakage feature. Flagged to you directly before exclusion; you confirmed exclude.
- **`attribution`, `copyright_license`, `image_type` (ISIC Archive 2)** — my own judgment call, a 5th exclusion category ("non-clinical/administrative, not a genuine feature") not in PROJECT_PLAN.md's identifier/path/label-source/leakage list. `attribution`'s "ViDIR Group... Vienna" value = exactly 9,873 rows, matching the already-documented HAM10000 overlap count — a source identifier, not a clinical feature. `copyright_license` is redundant with `attribution`. `image_type` is a constant. Flagged in `ISIC_Archive_2/feature_whitelist.md` for your review — lower-stakes than `melanocytic`, did not stop to ask, but noting it here per the "never silently resolve" rule.

---

<a id="documentation-cleanup"></a>
## Documentation Cleanup (2026-07-08) — DONE

Executed per `docs/PROJECT_PLAN.md` Current Phase item 3.

### Archived to `_archive/` (never hard-deleted)

- **`docs/Research_Plan.md`** → `_archive/Research_Plan.md`. Reason: not in
  PROJECT_PLAN.md's canonical `docs/` list; its content (research
  background/objectives/methodology/timeline) is a rougher, less-complete
  version of what's already in this file's "Project Overview" section above.
  Nothing in it was more complete than what's already here — no merge
  needed beyond what already existed.
- **`docs/Project_Cleanup_Review.md`** → `_archive/Project_Cleanup_Review.md`.
  Reason: not in PROJECT_PLAN.md's canonical `docs/` list; superseded by
  this file. Its two still-valid, not-yet-actioned recommendations were
  salvaged here first (see "Outstanding Items" below) before archiving.
  Its other findings (README duplication/chat-fragment, Office `~$...xlsx`
  lock file, `__pycache__` directories) were already resolved by the time of
  this audit and needed no salvage.

### Fixed in place (not archived — PROJECT_PLAN.md explicitly says "kept, corrected of stale sections")

- **`docs/Dataset_Strategy.md`**: corrected to describe ISIC Archive 1 & 2
  separately (was: a single pre-split "ISIC Archive"), and corrected the
  final processed-structure naming to `metadata_{train,val,test}.csv` (was:
  `train.csv`/`val.csv`/`test.csv`, which never matched actual output).

### Outstanding items (salvaged from `Project_Cleanup_Review.md` before archiving it — still not actioned)

- **`src/data_audit/config.py` vs. `src/data_cleaning/config.py`** duplicate
  ~15 lines of path constants (raw-dir paths, filenames, image subdirs for
  all 4 datasets). Risk: a raw path change would need editing in two places.
  Proposed fix (not yet done): extract shared constants into
  `src/common/paths.py`, imported by both. Behavior-preserving only, no
  pipeline logic change.
- **Superseded re-run logs** not yet archived: `logs/ISIC_Archive_1/` (2
  audit + 2 cleaning logs — 2 runs each) and `logs/PAD_UFES20/` (3 cleaning
  logs) contain multiple timestamped dev-iteration runs of the same script.
  Proposed fix (not yet done): move all but the latest successful run per
  dataset/script into `logs/archive/`, per the "keep complete logs, never
  delete" rule — archive, don't remove.

### Confirmed, no action needed

- **`docs/Multimodal Skin Lesion Classification Using Image and Clinical
  Metadata.xlsx`**: confirmed a 2-sheet literature-review working
  spreadsheet (Dataset Comparison Table: 7 candidate datasets by size/
  classes/metadata richness; Paper summary: reviewed papers with task/
  methodology/accuracy/limitations columns), not a data file. Not part of
  the docs cleanup sweep — kept in place as-is.
- **`README.md`**: updated to drop its reference to the now-archived
  `docs/Project_Cleanup_Review.md` (see Documentation section).

---

<a id="important-research-decisions"></a>
## Important Research Decisions

- **`data/raw/` is permanently read-only.** No script ever writes into it; enforced in code via a runtime guard, not just convention.
- **Patient-wise splitting** (not image-wise) is mandatory for every dataset to prevent leakage — validated as standard practice by all three literature-review papers.
- **Label harmonization**: dataset-specific diagnostic codes are mapped to a single canonical disease taxonomy with a documented, reasoned mapping table (`label_mapping.csv`), never silently renamed.
- **No invented data**: missing values are preserved as missing (never imputed or guessed) at the dataset-preparation stage; explicit "unknown" markers in raw data (e.g., PAD-UFES-20's `UNK` strings) are mapped to standard missing-value representation, which is a standardization decision, not data invention.
- **Documentation policy**: every processing phase produces a dedicated report/log (`reports/<dataset>/`, `logs/<dataset>/`) and a human-readable `dataset_description.md`; documentation is corrected via the generator scripts (for reproducibility) rather than hand-edited output files.
- **Multi-class target**: the project targets the full diagnostic taxonomy (not a binary benign/malignant collapse used by all three reviewed papers), a deliberate differentiation point.
- **Shared cross-dataset label taxonomy**: where PAD-UFES-20, HAM10000, and the two ISIC archives contain the same disease, the standardized `disease_label` string is made identical across datasets (e.g. "Melanoma", "Basal Cell Carcinoma", "Nevus") rather than kept as separate per-dataset strings, so cross-dataset experiments (training on one, validating on another) don't require a second harmonization pass. See `src/data_cleaning/common_label_mapping.py` (used by both ISIC archives) and each dataset's own `label_mapping.csv`.
- **Ambiguous-label exclusion over guessing**: when a source dataset makes a label ambiguous or unusable (ISIC Archive 1's 155 images filed under two conflicting classes; ISIC Archive 2's 255 rows with no `diagnosis_3` value), those rows are excluded from the cleaned/split dataset rather than assigned a guessed label — consistent with the "no invented data" principle above.
- **Cross-dataset external-validation exclusion over global re-split (2026-07-08)**: where a dataset overlaps substantially with another (HAM10000 vs. ISIC Archive 1/2), external-validation claims are restricted via a documented exclusion list rather than re-splitting all affected datasets globally, to preserve each dataset's already-verified internal split. See "Decision Made" under Cross-Dataset Verification above.
- **`biopsed` excluded from model input (2026-07-08)**: PAD-UFES-20's `biopsed` field is a near-perfect proxy for malignant/non-malignant (100% biopsy rate for all malignant diagnoses vs. 21% for non-malignant, phi=0.80) and is retained only as a documentation/analysis column, never as a training feature. See "`biopsed` Leakage/Shortcut Audit" above.

---

<a id="future-improvements"></a>
## Future Improvements

*Not part of current implementation — ideas to revisit later:*

- Bootstrap-based statistical significance testing between fusion strategies (informed by Mridha & Islam, 2026).
- LLM-based bias/leading-language auditing for any future free-text clinical fields, if such data becomes available (informed by Watson et al., 2026).
- Dual-mechanism fusion (channel-wise gating + cross-attention) as an advanced fusion candidate for Experiment 4, informed by Suresh et al. (2026)'s TG-CAVNet design.
- Fairness/bias evaluation across Fitzpatrick skin type and age subgroups, once a working classifier exists.
- External validation protocol design for ISIC Archive once internal PAD-UFES-20 models are established.

---

<a id="phase-6-stage-1-code-status"></a>
## Phase 6 Stage 1 — Code Written, GPU-Absent Confirmed, Kaggle Move Decided; NOT Yet Verified End-to-End (2026-07-09)

Verified by direct inspection of `src/models/`, `src/evaluation/`,
`logs/PAD_UFES20/`, and `reports/PAD_UFES20/` on disk, plus a fresh run of
the GPU check — not inferred from file existence alone.

**Confirmed done:**

- Both Phase 6 Stage 1 branches for PAD-UFES-20 are fully written:
  `src/models/config.py`, `src/models/dataset.py`,
  `src/models/image_model.py` (EfficientNet-B0), `src/models/metadata_model.py`
  (MLP), `src/models/train.py`, and `src/evaluation/evaluate.py`. Code
  correctly implements the approved Stage 1 scope: class-weighted loss,
  224×224 resize-and-pad transform, 3-seed design, and val/test discipline
  (`evaluate.py` refuses to evaluate on `--split test` without an explicit
  `--confirm-final` flag).
- **GPU check confirmed real, not assumed:** `torch.cuda.is_available()`
  was executed directly on this machine and returns `False` (no local
  GPU).
- **Kaggle-move decision is real and already reflected in code**, not just
  discussed: `src/models/config.py`'s module docstring states *"moving
  Stage 1 training to Kaggle since this machine has no GPU"* (dated
  2026-07-09), and the file already contains Kaggle-environment detection
  (`IS_KAGGLE`) and environment-aware path resolution. However,
  `KAGGLE_DATASET_SLUGS` and `KAGGLE_PROCESSED_SLUG` are still literal
  `"REPLACE_WITH_ACTUAL_KAGGLE_SLUG"` placeholders — the Kaggle-side
  dataset upload/slug configuration has not been done yet.

**NOT confirmed — no end-to-end verification on real data:**

- `logs/PAD_UFES20/` contains no `train_*.csv` per-epoch metrics file and
  no `train_*_summary.json`; no `.pt` checkpoint file exists anywhere in
  the repository; `reports/PAD_UFES20/` has no `baseline/` output
  directory. `train.py` writes its metrics CSV header before the first
  epoch runs, so the complete absence of any such file means
  `train_one_run()` was never actually run to completion (or possibly
  never invoked at all) for either branch.
- The only evidence of any real code execution is a cached EfficientNet-B0
  pretrained-weights file (`efficientnet_b0_rwightman-7f5810bc.pth`, ~20MB)
  in the local torch hub cache, timestamped 2026-07-09 11:52 — proof the
  image model was instantiated at least once, but not proof of a
  completed epoch, a full dataset pass, or a finished training run.

**Bottom line:** Stage 1 code for both branches is written and appears
correct by inspection; the "no local GPU → move to Kaggle" decision is
independently verified true; but **no training run has been observed to
complete on real PAD-UFES-20 data**, and Kaggle is not yet configured to
run it. Do not describe this as "verified end-to-end" until an actual
training/evaluation run produces a checkpoint, metrics CSV, and evaluation
report on disk.

---

<a id="docs-readability-pass"></a>
## Docs Readability Pass (2026-07-09)

Docs readability pass — `PROJECT_OWNERSHIP.md`, `PROJECT_PLAN.md`,
`Dataset_Strategy.md`, and `AI_Assistant_Instructions.md` reformatted for
readability (TOC, quick-summary boxes, consistent formatting); no
factual/decision content changed.

---

<a id="phase-6-stage-1-complete"></a>
## Phase 6 Stage 1 — COMPLETE (2026-07-13)

All 6 Stage 1 runs (2 branches x 3 seeds) for PAD-UFES-20 ran to
completion on Kaggle and were verified on disk in this session — not
trusted from printed numbers alone. Verification method: confirmed all 6
checkpoint files (`logs/PAD_UFES20/checkpoints/{branch}_seed{N}_best.pt`)
and all 6 summary files (`logs/PAD_UFES20/train_{branch}_seed{N}_summary.json`)
exist, checkpoint file sizes are sane (image ~16MB each, matching
EfficientNet-B0; metadata ~90KB each, matching a small MLP), and each
summary's `best_val_macro_f1` was read directly and matches the reported
number exactly. Per-epoch CSVs (`train_{branch}_seed{N}.csv`) also present
and inspected to confirm the overfitting pattern below.

**Final results (best val macro-F1 per run, mean +/- std across 3 seeds):**

| branch | seed0 | seed1 | seed2 | mean | std |
|---|---|---|---|---|---|
| image (EfficientNet-B0) | 0.5529 | 0.5741 | 0.5840 | 0.5703 | 0.0130 |
| metadata (MLP) | 0.5861 | 0.5694 | 0.5732 | 0.5762 | 0.0072 |

**Findings:**

- **Image branch shows higher variance and heavy overfitting.** Per-epoch
  logs confirm train macro-F1 climbs to ~0.93-0.96 by the later epochs
  while val macro-F1 plateaus/oscillates around 0.49-0.58 (e.g. seed0:
  epoch 15 train F1 0.9576 vs. val F1 0.5145). Consistent with fine-tuning
  a large ImageNet-pretrained backbone (~5.3M params) on a small training
  set (1,606 training images for PAD-UFES-20) — the model memorizes train
  faster than it generalizes.
- **Metadata branch is slightly higher-performing and more stable**
  (higher mean, lower std: 0.5762+/-0.0072 vs. 0.5703+/-0.0130). Per-epoch
  logs show a much smaller train/val gap (e.g. seed0: epoch 11 train F1
  0.7126 vs. val F1 0.5382) — consistent with PAD-UFES-20's rich
  21-feature clinical metadata carrying real, non-overfit-prone signal for
  a simple MLP.
- **No leakage canary triggered.** Neither branch scores implausibly high
  for a 6-class, ~16:1-imbalanced problem (a leakage feature slipping
  through would typically show as a near-1.0 macro-F1 outlier, as seen
  historically with `biopsed`/`melanocytic`-style features during dataset
  prep — see "Label-Leakage Decision" above). Both branches land in a
  plausible 0.55-0.59 range.

**Process note — Kaggle session-loss incident:** the first attempt at
these 6 runs was lost when an interactive Kaggle session disconnected,
wiping `/kaggle/working/` before the checkpoints/summaries could be
downloaded (the session had no local GPU, hence the earlier move to
Kaggle — see "Phase 6 Stage 1 — Code Written..." above). All 6 runs were
redone in a second session. **Going forward: any Kaggle run set expected
to take longer than ~30 minutes must use "Save & Run All (Commit)"**
(which persists output independent of the interactive session) instead of
an interactive session alone, to avoid repeating this loss.

**Status:** Phase 6 Stage 1 (PAD-UFES-20 baseline) is now complete per
`PROJECT_PLAN.md`'s Current Phase 6 scope. Next: either replicate this
baseline pattern on HAM10000, or proceed to Phase 7 fusion design — open
decision, not yet made (see discussion below, to be logged once decided).

---

<a id="sequencing-decision-ham10000-vs-phase-7"></a>
## Sequencing Decision — HAM10000 Baseline Before Phase 7 (2026-07-13)

Two options were weighed after Phase 6 Stage 1 (PAD-UFES-20) completed:
(a) do a full HAM10000 Stage 1 baseline before Phase 7, matching
`PROJECT_PLAN.md`'s literal Phase 6 -> 7 sequencing; or (b) move to Phase 7
fusion design now, treating HAM10000 baselines as parallel/later work. My
initial lean (offered, not decided) was toward (b), on the reasoning that
fusion is the thesis's actual contribution and PAD-UFES-20 alone already
gives both single-modality numbers needed to know if fusion helps.

**Decision: full option (a).** You reconsidered and chose to complete a
full HAM10000 baseline (not an abbreviated leakage-check-only version)
before starting Phase 7. Reasoning, in your words: this project's priority
is thoroughness and publication-readiness over speed; a second full
baseline dataset gives a more rigorous foundation for fusion architecture
decisions; and it matches `PROJECT_PLAN.md`'s original literal sequencing
without needing to justify a deviation. This supersedes my earlier (b)
lean — noted here so the reversal and its reasoning are on record, not
just the final choice.

**Immediate consequence:** HAM10000's `diagnosis_confirm_type`/
`diagnostic_code` leakage audit — flagged as "not yet audited" back in the
`biopsed` Leakage/Shortcut Audit section (2026-07-08) — was formally
re-verified this session with real, independently recomputed statistics
(not reused from memory): `diagnosis_confirm_type` phi=0.4121,
chi2=1700.67, n=10,015, 1,627/1,627 malignant cases (100%, zero
exceptions) confirmed via `histo` in every split (train/val/test
independently checked); `diagnostic_code` reconfirmed 1:1 with
`disease_label` (all 7 codes unambiguous); `dataset_source` reconfirmed
constant. These numbers exactly reproduce the existing
`feature_whitelist.md`/"Label-Leakage Decision" entries — no changes to
the whitelist. HAM10000 Phase 6 Stage 1 scope (reusing `src/models/`,
same EfficientNet-B0 + MLP recipe, 3 allowed metadata features `age`/
`sex`/`anatomical_site`, 3 seeds x 2 branches) proposed and pending your
approval before Stage 2 (code) starts, per the same two-stage process used
for PAD-UFES-20.

---

<a id="ham10000-stage-1-same-recipe-decision"></a>
## HAM10000 Phase 6 Stage 1 — Scope Approved, Same-Recipe Decision (2026-07-13)

**Approved: no architecture or hyperparameter tweaks for HAM10000.**
Explicit instruction: keep EfficientNet-B0 + MLP, identical
hyperparameters (learning rates, weight decay, batch size, epochs, early
stopping patience), identical class-weighted loss mechanism (inverse
class frequency computed from HAM10000's own train split — the mechanism
is unchanged, only the resulting weight values naturally differ since
they're a function of HAM10000's own class distribution), same 3 seeds
per branch (0/1/2).

**Reasoning (your words):** any architecture change would confound later
cross-dataset comparisons in Phase 8 — if PAD-UFES-20 and HAM10000 are
trained with different recipes, a performance difference between them
could come from the recipe change rather than genuine dataset
characteristics (image quality, metadata richness, class balance, dataset
size). Keeping the recipe fixed isolates the dataset as the only varying
factor, which is required for Phase 8's cross-dataset generalization
claims to be valid.

**Stage 2 implementation status:** `src/models/config.py`,
`src/models/dataset.py`, `src/models/train.py`, and
`src/evaluation/evaluate.py` were refactored to take a `--dataset
{PAD_UFES20,HAM10000}` argument, replacing PAD-UFES-20-hardcoded imports
with a per-dataset `DatasetConfig` (paths, class list, metadata feature
lists) looked up via `get_dataset(name)`. This is a structural
generalization only — every hyperparameter, transform, and the
class-weighted-loss computation itself are byte-for-byte unchanged from
the completed PAD-UFES-20 run; verified locally by re-instantiating both
the PAD-UFES-20 metadata preprocessor (confirmed `input_dim=89`,
identical to the original run) and both datasets' image/metadata models
end-to-end (correct 6-class / 7-class output shapes) before any Kaggle
run. `image_model.py`/`metadata_model.py` now take `num_classes`
explicitly (was a config-level default) since two datasets have different
class counts (6 vs. 7). HAM10000's `CLASS_NAMES` (7, alphabetical) and
metadata feature lists (`age` numeric; `sex`, `anatomical_site`
categorical) were added to `config.py`, matching
`feature_whitelist.md`'s 3 allowed columns exactly.

**Not yet done, blocking an actual Kaggle run:** `KAGGLE_DATASET_SLUGS["HAM10000"]`
(raw image mirror) and `KAGGLE_PROCESSED_SLUGS["HAM10000"]` (this
project's `data/processed/HAM10000/` uploaded as a private Kaggle
dataset) are still `"REPLACE_WITH_..."` placeholders in `config.py` — same
one-time setup PAD-UFES-20 needed before its first real run. Per the
process note logged after the PAD-UFES-20 session-loss incident, the
HAM10000 Kaggle notebook will use "Save & Run All (Commit)" from the
start, not an interactive session, since 6 runs is expected to exceed the
~30-minute threshold.

---

<a id="ham10000-stage-1-complete"></a>
## HAM10000 Stage 1 — COMPLETE (2026-07-16)

All 6 Stage 1 runs (2 branches x 3 seeds) for HAM10000 ran to completion
via a single Kaggle "Save & Run All (Commit)" job — no session loss this
time, per the process note adopted after the PAD-UFES-20 incident.
Verified on disk this session, not trusted from printed numbers alone:
all 6 checkpoint files (`logs/HAM10000/checkpoints/{branch}_seed{N}_best.pt`)
and all 6 summary files (`logs/HAM10000/train_{branch}_seed{N}_summary.json`)
exist; checkpoint sizes are sane (image 16,367,949 bytes each, matching
EfficientNet-B0; metadata 54,523 bytes each, matching the small MLP); each
summary's `best_val_macro_f1` was read directly and matches the reported
number exactly.

**Final results (best val macro-F1 per run, mean +/- std across 3 seeds):**

| branch | seed0 | seed1 | seed2 | mean | std |
|---|---|---|---|---|---|
| image (EfficientNet-B0) | 0.6961 | 0.6882 | 0.6977 | 0.6940 | 0.0041 |
| metadata (MLP) | 0.2503 | 0.2657 | 0.2403 | 0.2521 | 0.0104 |

**Analysis — large image-vs-metadata gap, and why it is not a leakage signal:**

Unlike PAD-UFES-20, where image and metadata branches landed close
together (0.5703/0.5762), HAM10000 shows a large gap (0.6940 image vs.
0.2521 metadata). This is explained by metadata richness, not by a
leakage artifact:

- HAM10000's `feature_whitelist.md` allows only 3 metadata columns
  (`age`, `sex`, `anatomical_site`) — the dataset's `metadata_train.csv`
  has 10 total columns, 7 of which are excluded (identifiers,
  label-source, or the previously-audited `diagnosis_confirm_type`
  leakage feature; see "Label-Leakage Decision" above). PAD-UFES-20's
  whitelist allows 21 columns out of 29, including several
  clinically-informative fields (itch/grew/hurt/changed/bleed/elevation,
  skin/cancer history, environmental exposures) that HAM10000 simply does
  not collect.
- With only age/sex/anatomical-site to work with, the metadata MLP has
  far less genuine signal available for a 7-class problem — the gap
  reflects a real difference in *input information content* between the
  two datasets' metadata, not a difference in how the image and metadata
  branches were trained (identical recipe, per the same-recipe decision
  above).
- **Not a leakage signal:** a leakage feature slipping through typically
  manifests as an implausibly *high* score (as seen historically with
  `biopsed`/`melanocytic`-style features during dataset prep). Here the
  metadata score is low, not high. Both branches also land clearly above
  the 7-class random-chance baseline (1/7 = 0.143): image at 0.694 (4.9x
  baseline) and metadata at 0.252 (1.8x baseline) — consistent with
  weak-but-real signal in the 3-column metadata, not noise and not a
  shortcut.
- **Supporting finding for the thesis's multimodal-richness narrative:**
  this cross-dataset contrast (HAM10000's sparse 3-feature metadata vs.
  PAD-UFES-20's rich 21-feature metadata) gives a concrete, quantified
  illustration of how metadata richness affects standalone metadata-model
  performance — directly relevant to motivating why fusion architectures
  should be evaluated across datasets of differing metadata richness, and
  why a fusion gain on PAD-UFES-20 should not be assumed to transfer
  identically to HAM10000.

**Status:** HAM10000 Phase 6 Stage 1 baseline is now complete per
`PROJECT_PLAN.md`'s Current Phase 6 scope, run with the identical
architecture/hyperparameter recipe as PAD-UFES-20 (per the same-recipe
decision above), preserving cross-dataset comparability for Phase 8.

This fulfills the "Sequencing Decision — HAM10000 Baseline Before
Phase 7" (2026-07-13, above): option (a) — a full HAM10000 Stage 1
baseline before starting Phase 7 — is now complete. Both single-modality
baselines (PAD-UFES-20, HAM10000) needed to evaluate whether fusion helps
are now in hand.

**Proposed next step:** begin Phase 7 (Multimodal Model Development) —
late fusion design first, per `PROJECT_PLAN.md`'s Phase 7 scope
("late fusion, then cross-attention fusion"), pending your review/approval
of the late-fusion Stage 1 scope before code starts (same two-stage
scope-then-code process used for both baseline phases).

---

<a id="phase-7-stage-1-scope-approved"></a>
## Phase 7 Stage 1 — Late Fusion Scope Approved, PAD-UFES-20 Only (2026-07-16)

**Scope, approved as proposed:**

- **Fusion point:** concatenate the 1280-d penultimate embedding from the
  image branch (EfficientNet-B0's `classifier[-1]` input) with the 64-d
  penultimate embedding from the metadata branch (`MetadataMLP`'s
  pre-final-layer output) into a 1344-d joint vector.
- **Joint classifier head:** one hidden layer deep —
  `Linear(1344, 128) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(128, num_classes)`
  — rather than a single `Linear(1344, num_classes)`, so the head has
  room to learn a real weighting between modalities instead of the
  1280-d image branch mechanically dominating a bare linear combination.
  This is the one genuinely new architectural choice in Stage 1 (not
  copied from an existing branch).
- **Initialization:** warm-start both branches from their Stage 1
  checkpoints (seed-matched — fusion seed *s* loads image/metadata seed
  *s*), then fine-tune end-to-end, nothing frozen, at a lower LR
  (`LEARNING_RATE_FUSION = 1e-5`) than either branch's own Stage 1 LR.
  Chosen over from-scratch joint training because PAD-UFES-20's size
  (~2,298 images) makes training a 5.3M-param CNN jointly with a fusion
  head from scratch a materially higher overfitting risk, and because
  warm-start directly tests the actual thesis question — does fusion
  improve on the *already-optimized* unimodal branches — rather than
  conflating that with a weaker from-scratch joint baseline.
- **Loss/metric/seed/split discipline:** identical to Stage 1 — class-
  weighted `CrossEntropyLoss` (train-split-only inverse frequency),
  macro-F1 for model selection/early stopping/reporting, `SEEDS = [0, 1,
  2]`, same `metadata_{train,val,test}.csv` splits, test split untouched
  by training code. No exceptions, so the fusion-vs-unimodal comparison
  stays a controlled ablation.
- **Script structure:** `src/models/fusion_model.py` (`ImageEmbedder`,
  `MetadataEmbedder`, `FusionModel`), `src/models/train_fusion.py`
  (mirrors `train.py`'s structure, reuses its `set_seed`/
  `compute_class_weights`/CLI conventions rather than duplicating them),
  and a `FusionDataset` variant added to `src/models/dataset.py`
  returning `(image_tensor, metadata_tensor, label_idx)`.
- **Dataset scope: PAD-UFES-20 only, not HAM10000.** `PROJECT_PLAN.md`
  designates PAD-UFES-20 as "primary, multimodal" and HAM10000 as
  "benchmark" — HAM10000's metadata whitelist has only 3 columns (age,
  sex, anatomical_site) vs. PAD-UFES-20's 21, too thin to expect a
  meaningful, unambiguous fusion signal (a weak result there would be
  confounded — architecture problem vs. genuinely low metadata signal —
  rather than informative). Phase 8 (PAD-UFES-20<->HAM10000
  generalization) already covers HAM10000's role in this project; a
  second fusion model on it now would be scope creep. Enforced at the
  code level too — `train_fusion.py --dataset` only accepts
  `PAD_UFES20`.

**Expected limitation, logged now so it is not mistaken for an oversight
later:** the 1280:64 dimensionality imbalance between the image and
metadata embeddings means the image branch will likely dominate this
late-fusion representation numerically, even with the deeper joint head
described above. This is acceptable *because* late fusion is
deliberately the simple baseline step in Phase 7 — if it underperforms
the image-only Stage 1 baseline, or shows metadata's contribution being
diluted relative to the metadata-only Stage 1 baseline, that is an
expected outcome of concatenation-based fusion on unequal-dimension
embeddings, not a bug to chase. It becomes part of the motivation for
Phase 7 Stage 2 (cross-attention fusion), which addresses this
structurally by having metadata guide attention over image features
rather than relying on raw concatenation.

**Implementation status (2026-07-16):** `src/models/fusion_model.py`,
`src/models/train_fusion.py`, and the `FusionDataset` addition to
`src/models/dataset.py` are written. `src/models/config.py` extended
with `LEARNING_RATE_FUSION`, `stage1_checkpoints_dir` (a warm-start
checkpoint source distinct from `checkpoints_dir`, since the latter is
where fusion training *writes* its own checkpoints), and
`KAGGLE_STAGE1_CHECKPOINT_SLUGS` (Kaggle input-dataset resolution for
the Stage 1 checkpoints, mirroring `KAGGLE_PROCESSED_SLUGS`'s pattern).
Smoke-tested locally on a real 12-row train / 8-row val PAD-UFES-20
subset via `.venv` (CPU): confirmed strict-mode Stage 1 checkpoint
loading into both embedders, a real image+metadata batch through the
dataset/model pipeline, one train and one eval epoch via
`run_epoch_fusion`, and a checkpoint save/reload round trip — all
passed. Kaggle notebook written to
`notebooks/pad_ufes20_fusion_kaggle_notebook.md` for the 3 fusion
training runs (seeds 0/1/2). **New requirement vs. the Stage 1
notebooks:** this run needs a third Kaggle "Add Data" source — the
Stage 1 checkpoints (`logs/PAD_UFES20/checkpoints/*.pt` from this
machine, zipped and uploaded as a new private Kaggle dataset) — since
`/kaggle/working` is wiped fresh per session and warm-start cannot
proceed without them.
