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
24. [Literature Review Reconciliation — 16 Papers (2026-07-17)](#literature-review-reconciliation)
25. [Phase 7 Stage 1 — COMPLETE (2026-07-18)](#phase-7-stage-1-complete)
26. [MetaBlock Mechanism Confirmed; Phase 7 Stage 2 Proposal — APPROVED (2026-07-18)](#phase-7-stage-2-proposal)
27. [Phase 7 Stage 2 — COMPLETE (2026-07-18)](#phase-7-stage-2-complete)
28. [Phase 8 Experiment 1 — Anatomical-Site Mapping Approved; Reduced-Feature Models + Eval Script Implemented (2026-07-18)](#phase-8-experiment-1-implementation)
29. [Negative Result — Improved Cross-Attention Variant Underperforms Original (2026-07-23)](#negative-result-improved-cross-attention)
30. [Phase 8, Experiment 1 — Cross-Dataset Generalization Scope, Final Model Confirmed (2026-07-23)](#phase-8-experiment-1-scope-final-model)

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
| 2. Literature Review | 🟡 In Progress | — | 16 papers reconciled (8 abstract-only, pending full-text read) — see "Literature Review Reconciliation" below and `docs/Literature_Review.md`. Not yet complete: 8 papers still need full-text reading before the Literature Review chapter can be written. |
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
| 7. Multimodal Model Development | ✅ Completed — both stages done, PAD-UFES-20 only | 2026-07-18 | Stage 1 (late fusion, mean macro-F1 0.5731+/-0.0021) and Stage 2 (cross-attention fusion, mean macro-F1 0.6209+/-0.0143) both complete and verified — see "Phase 7 Stage 1 — COMPLETE" and "Phase 7 Stage 2 — COMPLETE" entries. Cross-attention clearly outperforms all 3 prior variants (image/metadata/late-fusion), confirming the pre-registered dimensionality-imbalance hypothesis. Next: Phase 8 (Experiments & Evaluation). |
| 8. Experiments & Evaluation | ⏳ In progress | — | PAD-UFES-20<->HAM10000 cross-dataset generalization: ✅ complete 2026-07-25 (bootstrap significance included — note: only cross_attention vs. metadata is statistically significant; cross_attention's edge over image-only and late-fusion is **not** significant in this transfer direction, see "Phase 8, Experiment 1 ... COMPLETE" entry). Fitzpatrick fairness analysis: ✅ complete 2026-07-25, also stands as PAD-UFES-20's official final Stage 1 test-set result (see "Test-Split Single-Use Safeguard Added"). Both dataset's test splits now locked. Ensemble/TTA on cross-attention explored and permanently rejected (Melanoma F1 collapses 0.364->0.20 under ensemble+TTA — disqualifying, not just "no improvement" — see "Ensemble/TTA Exploration" entry). Remaining: HAM10000->ISIC external validation — both previously-open ISIC gaps (Archive 2's 3 unverified anatomical_site-adjacent whitelist fields, 3 cross-archive label conflicts) are now **resolved** (2026-07-25), scope approved, `evaluate_external_isic.py` implemented and smoke-tested (1 of 9 combinations run: Archive 1 image seed0). Not blocking — 8 of 9 combinations remain, to be run on Kaggle. |
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

**Note (added 2026-07-25):** these are **val**-split numbers only. The
official, one-time PAD-UFES-20 **test**-split result for all 4 variants
(image/metadata/fusion/cross_attention) is logged under "Phase 8 —
Fitzpatrick Fairness Analysis — COMPLETE; PAD-UFES-20 Test Split Now
Spent (2026-07-25)" below and cross-referenced in
`docs/Phase8_Fitzpatrick_Fairness_Results.md`. The test split is now
locked (see "Test-Split Single-Use Safeguard Added" entry, 2026-07-25).

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

---

<a id="literature-review-reconciliation"></a>
## Literature Review Reconciliation — 16 Papers (2026-07-17)

**Documentation/writing-support update only** — does not touch `data/`,
`src/models/`, `src/evaluation/`, or any logs/checkpoints; no dataset,
split, `feature_whitelist.md`, architecture, or hyperparameter changed.

`docs/Literature_Review_Master.xlsx` reconciles this project's original 3
tracked papers (Mridha & Islam 2026; Suresh et al. 2026 TG-CAVNet; Watson et
al. 2026), the user's 8 Excel-reviewed papers, and 8 new web-search-found
candidates into **16 unique papers, not 19** (see duplicate resolution
below). A readable summary of the reconciled table, per-paper limitations,
and relevance-to-thesis notes is in `docs/Literature_Review.md` — the xlsx
remains the source of record if the two ever disagree.

**Status: 8 of the 16 papers are abstract-only, pending full-text read.**
Phase 2 (Literature Review) is **not** being marked complete — full-text
reading of those 8 is required before the Literature Review chapter can be
written. Updated in the Progress Tracker table above.

**Duplicate resolved (logged so it is not re-investigated later):** the
user's Excel rows 1 and 2 are the *same* paper listed twice under slightly
different titles (identical AUC=0.9818/AUPRC=0.9924/F1=0.9769) — confirmed
via GitHub author match to be the medRxiv preprint already tracked as
**"Mridha & Islam 2026"**. Net effect: the Excel's 8 rows contribute only 6
new unique papers, not 8, giving 3 + 6 + 8 = 16 total.

**Still open, not a duplicate finding:** one *unconfirmed* possible match —
Excel row 4 ("Multimodal Skin Lesion Classification Using Deep Learning",
ISIC Archive, 2018) may be the same paper as Yap, Yolland & Tschandl (2018,
Experimental Dermatology), found independently via the web search. Needs a
full-text author check; if confirmed, paper count drops to 15.

**Literature gap — Fitzpatrick/skin-tone fairness:** none of the 16 papers
focus on Fitzpatrick/skin-tone fairness in dermatology AI. This is relevant
because `PROJECT_PLAN.md`'s Phase 8 already plans a dedicated Fitzpatrick
fairness analysis — the absence of prior work specifically on skin-tone bias
in dermatology AI means Phase 8's fairness analysis is a citable
contribution in its own right, not just a routine evaluation step, and
should be stated explicitly as such in the thesis (e.g. Related Work or
Contributions section). Recommend a targeted search for 2-3
fairness-specific papers before Phase 9 (Thesis Writing Support) begins.

**Two priority full-text reads flagged for the user**, both by PAD-UFES-20's
own dataset creators and near-mandatory citations for a thesis using their
dataset:

1. **Pacheco & Krohling (2021)** — MetaBlock (attention-based
   image+metadata fusion) — highest priority, direct architectural
   comparison point for Phase 7.
2. **Pacheco & Krohling (2020)** — clinical-information impact on automated
   skin cancer detection — foundational justification for this thesis's
   multimodal premise, citable in the Introduction.

---

<a id="phase-7-stage-1-complete"></a>
## Phase 7 Stage 1 — COMPLETE (2026-07-18)

All 3 Stage 1 late-fusion runs (seeds 0/1/2) for PAD-UFES-20 ran to
completion on Kaggle. Verified on disk directly this session, not trusted
from pasted numbers alone — the first verification pass (before any files
existed) correctly caught that the fusion checkpoints/summaries were not
actually present yet; they were re-checked only after the user placed them
and confirmed via VS Code. Verification method: all 3 checkpoint files
(`logs/PAD_UFES20/checkpoints/fusion_seed{0,1,2}_best.pt`, ~17.2MB each,
sane for an EfficientNet-B0 image branch + MLP metadata branch + joint
head) and all 3 summary files
(`logs/PAD_UFES20/train_fusion_seed{0,1,2}_summary.json`) exist; each
summary's `best_val_macro_f1` was read directly and matches the reported
number exactly (seed0=0.572250, seed1=0.576023, seed2=0.571108); mean/std
recomputed independently from these exact values using the same
(population, ddof=0) method as the Stage 1 image/metadata baselines, giving
0.5731/0.0021 — the pasted 0.0022 was a trivial rounding difference, not a
discrepancy in the underlying numbers.

**Final 3-way comparison (best val macro-F1 per run, mean +/- std across 3
seeds, PAD-UFES-20):**

| branch | seed0 | seed1 | seed2 | mean | std |
|---|---|---|---|---|---|
| image (EfficientNet-B0) | 0.5529 | 0.5741 | 0.5840 | 0.5703 | 0.0130 |
| metadata (MLP) | 0.5861 | 0.5694 | 0.5732 | 0.5762 | 0.0072 |
| **late fusion** (warm-started, joint fine-tune) | 0.5723 | 0.5760 | 0.5711 | **0.5731** | **0.0021** |

**Finding — late fusion did not clearly beat either single-modality
baseline in raw macro-F1, but is markedly more stable:**

- Late fusion's mean (0.5731) lands between the two unimodal baselines —
  slightly below metadata-only (0.5762) and slightly above image-only
  (0.5703) — a difference well within the unimodal branches' own std
  (0.0130 for image), so this is **not** a demonstrated improvement over
  either baseline on its own terms.
- What late fusion clearly does deliver is variance reduction: std=0.0021,
  roughly 3.4x tighter than metadata-only (0.0072) and 6.2x tighter than
  image-only (0.0130). All 3 fusion seeds land in a narrow 0.5711-0.5760
  band, vs. image-only's 0.5529-0.5840 spread.
- This is the expected outcome flagged in advance in "Phase 7 Stage 1 —
  Late Fusion Scope Approved" above: the 1280:64 image:metadata embedding
  dimensionality imbalance was predicted to let the image branch
  numerically dominate simple concatenation even with a deeper joint head,
  diluting metadata's contribution rather than combining both branches'
  genuine signal. The result is consistent with that prediction — fusion
  behaves more like a regularized/stabilized version of the image branch
  (much lower variance, similar mean) than like a genuine combination that
  captures metadata's higher standalone score.
- **Framing for the thesis: not a failed experiment.** This is the
  pre-registered, literal purpose of running late fusion as the Phase 7
  Stage 1 baseline step before cross-attention — it empirically
  demonstrates *why* naive concatenation fusion is insufficient on this
  dataset's embedding-dimension imbalance, giving Stage 2 (cross-attention
  fusion) a concrete, quantified motivation rather than an assumed one.

**Status:** Phase 7 Stage 1 (late fusion, PAD-UFES-20) is now complete.
Next: Phase 7 Stage 2 (cross-attention fusion) scope proposal — pending
your review/approval before code starts, per the same two-stage process
used for every prior phase.

---

<a id="phase-7-stage-2-proposal"></a>
## MetaBlock Mechanism Confirmed; Phase 7 Stage 2 Proposal — APPROVED (2026-07-18)

**MetaBlock's actual mechanism (Pacheco & Krohling 2021, row #10 in
`Literature_Review.md`) confirmed before locking Stage 2's design, per your
instruction not to assume abstract-level fidelity.** Verified via the
paper's official code repository (`github.com/paaatcha/MetaBlock`),
cross-checked against an independently-agreeing secondary paper's
description. **Caveat, stated plainly: the primary IEEE JBHI PDF is
paywalled and was not directly read** — this is a code + secondary-source
confirmation, not a full-text read. If the thesis needs to cite specific
numeric results or methodology details beyond the mechanism itself, the
primary PDF should still be obtained before finalizing those claims.

**Finding: our abstract-level guess was correct — MetaBlock is not
Transformer-style Q/K/V attention.** It is a channel-wise gated
feature-modulation block: a metadata vector `U` passes through two
independent `Linear + BatchNorm` branches producing `t1`, `t2` (same
channel-dim as the CNN's feature vector `V`); output
`V' = sigmoid(tanh(V·t1) + t2)` — a multiplicative gate (`t1`) plus an
additive bias (`t2`), squashed by sigmoid, broadcast **uniformly across
spatial positions within each channel** (no per-spatial-location
weighting). Their own simpler baseline, MetaNet, is even closer to
squeeze-and-excitation: metadata → conv+ReLU+sigmoid → per-channel scale
map → elementwise multiply, no tanh/additive term. Per the abstract,
MetaBlock beats MetaNet + plain concatenation in 6/10 tested scenarios
across ISIC 2019 and PAD-UFES-20.

**Answering your question 2 directly: our proposed cross-attention design
is a related-but-meaningfully-different mechanism, not a MetaBlock
reproduction.** Both pursue the same goal (let metadata do more than get
concatenated/ignored), but:

- MetaBlock modulates **channels**, uniformly across all spatial locations
  within a channel — it cannot make the network attend to one image region
  over another.
- Our proposed design (metadata as Query, EfficientNet-B0's 49 spatial
  tokens from `backbone.features(x)` as Key/Value, standard multi-head
  scaled dot-product attention) computes explicit **per-spatial-location**
  attention weights — genuinely different image regions can be weighted
  differently depending on metadata, which channel-wise gating structurally
  cannot do.

**Corrected framing for all future references (this doc,
`Literature_Review.md`, eventual thesis text): call Stage 2 "cross-attention
fusion, contrasted with MetaBlock's channel-gating approach" — never
"MetaBlock-inspired" or "MetaBlock-style," since that would overclaim
architectural fidelity to a mechanism we now know is different.**
`Literature_Review.md` row #10 and its "Priority Full-Text Reads" section
have been updated accordingly.

### Stage 2 architecture — reconfirmed proposal, pending your approval

Unchanged from what was proposed before this verification, now with
accurate framing and grounded in the actual `fusion_model.py`/
`image_model.py` code:

- **Fusion point:** replace the pooled 1280-d image vector with
  EfficientNet-B0's pre-pool spatial feature map (`backbone.features(x)`,
  shape `[B, 1280, 7, 7]` → 49 spatial tokens × 1280-d). The 64-d metadata
  embedding (from the existing `MetadataEmbedder`) is projected into a
  query and cross-attends over the 49 image tokens (Q=metadata, K/V=image
  tokens, standard multi-head scaled dot-product attention, projected to a
  shared `d_model` so the 1280:64 raw-dimension imbalance no longer
  mechanically dominates the result the way Stage 1's concatenation did).
- **Optional dual-mechanism add-on** (Suresh et al. TG-CAVNet-inspired,
  already flagged in "Future Improvements" above): a metadata-conditioned
  channel gate applied before cross-attention, so metadata both reweights
  channels *and* spatially attends — TG-CAVNet itself remains only
  partially captured in `Literature_Review.md` (row #2), so this stays a
  secondary, not primary, design input pending its own full-text read.
- **New class** `CrossAttentionFusionModel` added alongside the existing
  `FusionModel` in `fusion_model.py` (not replacing it), so Stage 1's
  late-fusion checkpoints/results remain reproducible for the eventual
  3-way (image/metadata/late-fusion) vs. cross-attention comparison.
- **Everything else unchanged from Stage 1's discipline:** warm-start both
  branches from Stage 1 checkpoints (seed-matched), fine-tune end-to-end at
  a low LR, class-weighted CE loss, macro-F1 for model selection/reporting,
  seeds 0/1/2, PAD-UFES-20 only (same reasoning as Stage 1 — HAM10000's
  3-column metadata whitelist is too thin), test split untouched until
  final evaluation.

**Approved 2026-07-18.** Implementation status below.

### Stage 2 implementation status (2026-07-18)

`src/models/cross_attention_fusion_model.py` written:
`SpatialImageEmbedder` (wraps the full `build_efficientnet_b0()` model,
forward stops at `backbone.features(x)` to return the 49 pre-pool spatial
tokens instead of the pooled 1280-d vector — same full-architecture-wrap
approach as Stage 1's `ImageEmbedder`, so a Stage 1 image checkpoint's
state_dict still loads with `strict=True`), `MetadataChannelGate` (the
optional TG-CAVNet-inspired channel gate, enabled by default via
`use_channel_gate=True`), and `CrossAttentionFusionModel` (reuses Stage
1's `MetadataEmbedder` unchanged from `fusion_model.py`; metadata
projected to a `d_model=256` query, image tokens projected to
`d_model=256` key/value, `nn.MultiheadAttention` with 8 heads; joint head
concatenates the attended 256-d vector with the raw 64-d metadata
embedding before the same `Linear→BatchNorm→ReLU→Dropout→Linear` head
style as Stage 1). Added alongside `FusionModel`, not replacing it.

`src/models/train_cross_attention_fusion.py` written, mirroring
`train_fusion.py`'s structure exactly (same warm-start-then-fine-tune
flow, same `compute_class_weights`/`set_seed` reuse from `train.py`, same
CLI convention — `--dataset PAD_UFES20 --seed {0,1,2}`), writing
`cross_attention_seed{N}_best.pt` checkpoints and
`train_cross_attention_seed{N}_summary.json`/`.csv` logs, distinct
filenames from Stage 1's `fusion_seed{N}_*` so both stages' outputs coexist
in the same `logs/PAD_UFES20/` folder. `src/models/config.py` extended
with `LEARNING_RATE_CROSS_ATTENTION = 1e-5` (same conservative rationale
as Stage 1's `LEARNING_RATE_FUSION` — both embedders are Stage
1-converged, only the new cross-attention/head parameters start random).

**Smoke-tested locally on a real 12-row train / 12-row val PAD-UFES-20
subset (2 rows per class, all 6 classes present) via `.venv` (CPU):**
confirmed strict-mode Stage 1 checkpoint loading into
`CrossAttentionFusionModel` (both the image and metadata embedders),
single-sample and batch-of-4 forward passes with correct output shapes,
one real train epoch + one real eval epoch via
`run_epoch_cross_attention`, and a checkpoint save/reload round trip — all
passed.

**Kaggle notebook generated, not hand-assembled:**
`scripts/generate_cross_attention_kaggle_notebook.py` reads the real
`src/models/*.py` file contents at generation time and writes
`notebooks/pad_ufes20_cross_attention_kaggle_notebook.md`, so the notebook
cannot silently drift from the source-of-truth `.py` files the way a
hand-typed copy could. Same 3 Kaggle "Add Data" sources as Stage 1's
fusion notebook (raw PAD-UFES-20 mirror, processed metadata,
`pad-ufes20-stage1-checkpoints`) — Stage 2 warm-starts from the identical
Stage 1 image/metadata checkpoints Stage 1 fusion did, so no new upload is
needed. Same folder-verification → setup → 8 `%%writefile` cells (config,
dataset, image_model, metadata_model, fusion_model,
cross_attention_fusion_model, train, train_cross_attention_fusion) →
sanity check → full model/GPU check → 3 training cells (seeds 0/1/2)
pattern as the Stage 1 notebook.

**Not yet done:** the 3 actual Kaggle training runs (seeds 0/1/2) have not
been executed — notebook is ready to paste in, per the same "Save & Run
All (Commit)" process adopted after the Stage 1 session-loss incident.

---

<a id="phase-7-stage-2-complete"></a>
## Phase 7 Stage 2 — COMPLETE (2026-07-18)

All 3 Stage 2 cross-attention-fusion runs (seeds 0/1/2) for PAD-UFES-20 ran
to completion on Kaggle. Verified on disk directly this session, not
trusted from pasted numbers alone — the first verification pass (before
any files existed) correctly caught that the cross-attention
checkpoints/summaries were not actually present yet; they were re-checked
only after the user placed them and confirmed via VS Code. Verification
method: all 3 checkpoint files
(`logs/PAD_UFES20/checkpoints/cross_attention_seed{0,1,2}_best.pt`,
~19.4MB each — larger than Stage 1 fusion's ~17.2MB, consistent with the
added query/key/value projections and attention module) and all 3 summary
files (`logs/PAD_UFES20/train_cross_attention_seed{0,1,2}_summary.json`)
exist; each summary's `best_val_macro_f1` was read directly and matches
the reported number exactly (seed0=0.604949, seed1=0.618178,
seed2=0.639662); mean/std recomputed independently using the same
population (ddof=0) method as every prior stage, giving 0.6209/0.0143.

**Final 4-way comparison (best val macro-F1 per run, mean +/- std across 3
seeds, PAD-UFES-20):**

| branch | seed0 | seed1 | seed2 | mean | std |
|---|---|---|---|---|---|
| image (EfficientNet-B0) | 0.5529 | 0.5741 | 0.5840 | 0.5703 | 0.0130 |
| metadata (MLP) | 0.5861 | 0.5694 | 0.5732 | 0.5762 | 0.0072 |
| late fusion (concat, warm-started) | 0.5723 | 0.5760 | 0.5711 | 0.5731 | 0.0021 |
| **cross-attention fusion** (metadata=Q, image tokens=K/V) | 0.6049 | 0.6182 | 0.6397 | **0.6209** | 0.0143 |

**Note (added 2026-07-25):** these are **val**-split numbers only. The
official, one-time PAD-UFES-20 **test**-split result for all 4 variants
is logged under "Phase 8 — Fitzpatrick Fairness Analysis — COMPLETE;
PAD-UFES-20 Test Split Now Spent (2026-07-25)" below and cross-referenced
in `docs/Phase8_Fitzpatrick_Fairness_Results.md`. The test split is now
locked (see "Test-Split Single-Use Safeguard Added" entry, 2026-07-25).

**Finding — cross-attention clearly outperforms all 3 prior variants, a
clean separation, not just a mean-level difference within noise:**

- Cross-attention's minimum seed result (0.6049) exceeds every other
  variant's *maximum* seed result (image max 0.5840, metadata max 0.5861,
  late-fusion max 0.5760) — all 3 cross-attention seeds land strictly above
  the entire range of all 3 prior variants' 9 combined runs, with zero
  overlap. This is a materially stronger result than a mean-difference
  claim resting on overlapping distributions.
- Magnitude: +0.0506 over image-only, +0.0447 over metadata-only, +0.0478
  over late-fusion (mean-to-mean) — roughly 4x the size of late-fusion's
  own std (0.0021) and comparable to or larger than image-only's std
  (0.0130), i.e. a jump too large to attribute to run-to-run noise at this
  seed count.
- **Confirms the pre-registered hypothesis from Phase 7 Stage 1's scope
  approval** (dimensionality-imbalance limitation, logged in advance, not
  discovered after the fact): late fusion's 1280:64 raw-dimension
  concatenation let the image branch numerically dominate, diluting
  metadata's contribution (Stage 1's finding: fusion ~= regularized image
  branch, not a genuine combination). Cross-attention's shared-`d_model`
  projection (both modalities projected to 256-d before any interaction)
  removes that mechanical imbalance, and the result is consistent with
  that structural fix actually working — not merely a different
  hyperparameter producing a better number by chance.
- **Honest caveat on variance:** cross-attention's std (0.0143) is higher
  than late fusion's (0.0021) and close to image-only's (0.0130) — Stage 2
  trades away Stage 1's variance-tightening property in exchange for a
  substantially higher mean. Both properties are true simultaneously and
  should be reported together in the thesis, not just the mean gain.

**Status:** Phase 7 (Multimodal Model Development) is now complete —
both Stage 1 (late fusion) and Stage 2 (cross-attention fusion) done and
verified for PAD-UFES-20. Per `PROJECT_PLAN.md`'s roadmap, next is Phase 8
(Experiments & Evaluation): PAD-UFES-20<->HAM10000 cross-dataset
generalization (headline result), HAM10000->ISIC external validation
(with the documented exclusion lists applied), Fitzpatrick fairness
analysis, and bootstrap significance testing — scope proposal pending,
same two-stage process as every prior phase.

---

<a id="ensemble-tta-exploration"></a>
## Ensemble/TTA Exploration on Cross-Attention Fusion — No Adoption (2026-07-19)

**Backfilled 2026-07-25** — this entry documents `reports/PAD_UFES20/
score_experiments/` and `reports/PAD_UFES20/fusion/eval_fusion_seed0_val.json`
(all file-dated 2026-07-19), which existed on disk without a corresponding
status entry until flagged by the 2026-07-25 project audit. Contents read
and verified directly from the JSON files for this entry — not from
memory or the audit's summary.

**What was tested:** after Phase 7 Stage 2 completed (2026-07-18), a
follow-up exploration asked whether ensembling the 3 independently-trained
cross-attention seeds (and/or adding test-time augmentation) could beat
per-seed reporting as the "final" way to present/deploy the cross-attention
result, before committing to Phase 8. Val split, PAD-UFES-20.

**Files and verified contents:**

| File | Config | macro-F1 (val) |
|---|---|---|
| `eval_cross_attention_seed0_val.json` | single seed 0 | 0.604949 |
| `eval_cross_attention_seed1_val.json` | single seed 1 | 0.618178 |
| `eval_cross_attention_seed2_val.json` | single seed 2 | 0.639662 |
| `eval_cross_attention_ensemble3_val.json` | 3-seed logit-average ensemble | **0.621266** |
| `eval_cross_attention_ensemble3_tta_val.json` | 3-seed ensemble + TTA | **0.588628** |
| `../fusion/eval_fusion_seed0_val.json` | late-fusion, single seed 0 | 0.572250 |

The 3 single-seed values reproduce the already-reported Phase 7 Stage 2
numbers exactly (seed0=0.604949, seed1=0.618178, seed2=0.639662 — see
"Phase 7 Stage 2 — COMPLETE" entry above), confirming these are the same
underlying evaluations, not a divergent re-run. The fusion seed0 value
(0.572250) likewise matches the already-reported late-fusion seed0
(0.5723) to within float/eval-pass noise.

**New information from these files — the ensemble and TTA results:**
- **3-seed ensemble (no TTA): 0.6213** vs. the individual-seed mean of
  0.6209 — a +0.0004 difference, well inside run-to-run noise (individual
  seeds already span 0.6049-0.6397, std 0.0143). Ensembling the 3 seeds
  does **not** meaningfully outperform simply reporting their mean.
- **3-seed ensemble + TTA: 0.5886** — **worse** than every individual
  cross-attention seed (all 3 of which exceed 0.60) and worse than the
  ensemble-without-TTA. TTA hurts this model/dataset rather than helping.
- **Per-class breakdown reveals the real reason this is disqualifying, not
  just "no improvement":** Melanoma F1 **collapses from 0.3636 (ensemble,
  no TTA) to 0.20 (ensemble + TTA)** — verified directly from
  `eval_cross_attention_ensemble3_val.json` and
  `eval_cross_attention_ensemble3_tta_val.json`. Melanoma is the clinically
  critical minority class in this task; a test-time augmentation scheme
  that nearly halves its F1 is not a neutral/no-op choice, it is actively
  harmful to the metric that matters most.

**Conclusion / decision (inferred from the fact no later entry adopts an
ensemble or TTA checkpoint, and Phase 8 evaluation code loads single-seed
checkpoints only):** neither ensembling nor TTA was adopted. The
project's existing practice — reporting mean +/- std across 3
independently trained single-model seeds, and using single-seed
checkpoints for all downstream work — remained the right call; this
exploration is negative-result evidence supporting that choice, not a
change to it. No checkpoint files exist for the ensemble/TTA
configurations (there's nothing to checkpoint — an eval-time ensemble
of the 3 existing single-seed checkpoints), consistent with them not
being adopted.

**Explicit, permanent confirmation (2026-07-26):** TTA (ensemble+TTA) is
**permanently excluded from consideration anywhere in this project**, at
any future phase, for any variant. This is not a "revisit later" deferral
— the Melanoma collapse above is a disqualifying failure mode on the most
clinically important class, not a marginal or dataset-specific quirk that
different hyperparameters might fix. Any future proposal to reintroduce
TTA should be treated as requiring a fresh justification against this
finding, not a default option.

---

<a id="phase-8-experiment-1-implementation"></a>
## Phase 8 Experiment 1 — Anatomical-Site Mapping Approved; Reduced-Feature Models + Eval Script Implemented (2026-07-18)

**Scope (approved earlier this session):** PAD-UFES-20 -> HAM10000 is the
primary/headline cross-dataset generalization direction (reverse direction
deferred, do-if-time-allows); all 4 variants (image, metadata, late-fusion,
cross-attention) evaluated, not just cross-attention, to see whether the
generalization gap differs by architecture; Protocol A (full native
6-way argmax, scored only on the 3 shared classes, spillover reported, not
masked/hidden); Fitzpatrick fairness via per-group macro-F1 (equalized
odds deferred - small per-group samples make it unstable); significance
testing = cross-attention vs. each of the other 3 variants, 1000 bootstrap
resamples, 95% CI.

### Anatomical-site mapping — reviewed and approved before any training

`docs/Phase8_Anatomical_Site_Mapping.csv` (mirrors the `label_mapping.csv`
format) reviewed and approved before locking. PAD-UFES-20 has 14 total
`anatomical_site` categories: **9 clean** (casing-only difference: ABDOMEN,
BACK, CHEST, EAR, FACE, FOOT, HAND, NECK, SCALP), **3 lossy/coarsened**
(ARM and FOREARM both collapse into HAM10000's single "upper extremity"
category - collision flagged explicitly, not hidden; THIGH -> "lower
extremity"), **2 ambiguous with no HAM10000 equivalent** (LIP, NOSE -
**approved decision: left unmapped**, falling to the existing
unseen-category "__MISSING__" bucket rather than being force-mapped to
"face", which would have been an anatomically imprecise stretch not
reflected in either dataset's real taxonomy).

### Implementation (2026-07-18)

**`src/models/config.py`** extended with `REDUCED_NUMERIC_FEATURES`
(`["age"]`), `REDUCED_CATEGORICAL_FEATURES` (`["sex", "anatomical_site"]`),
`ANATOMICAL_SITE_CROSS_DATASET_MAP` (the approved mapping, coded directly
from the CSV above), and `normalize_anatomical_site_for_cross_dataset()`.
`DatasetConfig.with_features()` added (shallow copy overriding just the
feature lists) so the reduced-feature variant reuses all of PAD-UFES-20's
existing paths/checkpoint dirs without a near-duplicate second
`DatasetConfig` entry.

**`src/models/dataset.py`**: `MetadataPreprocessor` extended with an
optional `column_transforms` dict (`{column: callable}`, applied in both
`fit()` and `transform_row()`) and a `without_transforms()` method
(shallow copy with transforms cleared, keeping the already-fitted
means/stds/categories) - the latter is what lets the same fitted
preprocessor be reused at HAM10000 evaluation time without re-applying
PAD-UFES-20's normalization to HAM10000's already-correctly-vocabularied
values. Verified backward-compatible: re-ran the existing rich-feature
preprocessor with no `column_transforms` argument, confirmed
`output_dim` still returns 89 exactly as before.

**3 new training scripts** (`train_metadata_reduced.py`,
`train_fusion_reduced.py`, `train_cross_attention_fusion_reduced.py`),
mirroring their Stage 1/7 counterparts exactly (same architecture,
hyperparameters, loss/metric/seed discipline) except: metadata restricted
to the 3 reduced features with anatomical_site normalization applied;
image side of fusion/cross-attention warm-starts from the *existing,
unchanged* `image_seed{N}_best.pt` (no schema issue there); metadata side
warm-starts from the new `metadata_reduced_seed{N}_best.pt`. Checkpoints
saved as `{metadata,fusion,cross_attention}_reduced_seed{N}_best.pt`,
alongside - never overwriting - the existing rich-feature Stage 1/7
checkpoints used for every already-reported PAD-UFES-20-internal result.

**Smoke-tested locally on a real 12-row train/12-row val PAD-UFES-20
subset** (2 rows/class, deliberately including CHEST/FOREARM+ARM/THIGH/NOSE
to exercise clean, lossy-collision, and unmapped-fallback cases all at
once): confirmed `normalize_anatomical_site_for_cross_dataset()` maps each
category correctly (verified `ARM`/`FOREARM` -> `upper extremity`,
`THIGH` -> `lower extremity`, `NOSE`/`LIP`/missing -> `__MISSING__`,
9 clean cases -> lowercase identity); real train+eval epochs for all 3
reduced-feature models (metadata_reduced standalone, then fusion_reduced
and cross_attention_reduced warm-started from it plus the real
`image_seed0_best.pt`); checkpoint save/reload round trip. All passed.

**Kaggle notebook generated** (`scripts/generate_reduced_feature_kaggle_notebook.py`
-> `notebooks/pad_ufes20_reduced_feature_kaggle_notebook.md`), same
read-real-files-at-generation-time approach as the Stage 2 notebook.
Reuses the identical 3 "Add Data" sources as Phase 7 (no new upload
needed). 24 cells: folder verification -> setup -> 12 `%%writefile` cells
-> sanity check (normalization spot-check + Stage 1 checkpoint
resolution) -> 9 training cells in dependency order (metadata_reduced x3
must complete before fusion_reduced/cross_attention_reduced x3, since the
latter two warm-start from the former's output).

**`src/evaluation/evaluate_cross_dataset.py` written**, implementing
Protocol A for all 4 variants: `CrossDatasetEvalDataset` (filters
HAM10000's test split to the 3 shared classes, encodes labels in
PAD-UFES-20's label space), `build_pad_to_ham_eval_preprocessor()` (fits
on PAD-UFES-20 train, returns the `without_transforms()` eval copy),
`coverage_diagnostic()` (reports what fraction of HAM10000's sex/
anatomical_site values matched a known category vs. fell to
"__MISSING__" - e.g. HAM10000's "genital"/"trunk" categories have no
PAD-UFES-20 counterpart and are expected to fall through), macro-F1 +
per-class F1 restricted to the 3 shared classes via sklearn's `labels=`,
full 6-class confusion matrix, and spillover rate.

**Validated with a complete real run (image variant, seed 0) - no Kaggle
dependency, since `image_seed0_best.pt` already exists and evaluation is
inference-only:** 1,253 HAM10000 test images (all 3 shared classes),
macro-F1 (shared classes) = **0.4577** (Basal Cell Carcinoma 0.1987,
Melanoma 0.3642, Nevus 0.8102), spillover rate 16.5%. Lower than
PAD-UFES-20's own within-dataset image macro-F1 (0.5703) - the expected
generalization-gap direction for true cross-dataset transfer, not a bug.
This exercises the entire shared dataset/protocol/reporting code path
used by all 4 variants; the metadata/late-fusion/cross-attention variants
share this same code and differ only in which checkpoint/preprocessor
they load, but full verification of those 3 awaits the 9 Kaggle-trained
reduced-feature checkpoints below.

**Not yet done:** the 9 actual Kaggle training runs (via
`notebooks/pad_ufes20_reduced_feature_kaggle_notebook.md`) have not been
executed. Once run and the checkpoints brought back and verified (same
process as every prior stage), remaining work is: run
`evaluate_cross_dataset.py` for the metadata/late_fusion/cross_attention
variants x 3 seeds, then assemble the 4-way PAD-UFES-20->HAM10000
generalization comparison table.

---

<a id="negative-result-improved-cross-attention"></a>
## Negative Result — Improved Cross-Attention Variant Underperforms Original (2026-07-23)

An "improved" cross-attention fusion variant was trained (3 seeds, via
`notebooks/pad_ufes20_cross_attention_improved_kaggle_notebook.md` /
`src/models/train_cross_attention_improved.py`) as an attempt to beat the
original Phase 7 Stage 2 cross-attention result (0.6209 +/- 0.0143 mean
macro-F1, see "Phase 7 Stage 2 — COMPLETE"). It did not.

**Result (user-reported, unverified from file):** seed macro-F1 scores
0.4804 / 0.5447 / 0.5028, mean **0.509**. These numbers were provided
directly by the user in this session rather than confirmed against
checkpoint/summary files in this pass — flagged explicitly as
**unverified-from-file**, but consistent with what was observed during
the improved-variant runs. Standard file-verification (checkpoint
existence, summary JSON cross-check) was deliberately skipped for this
entry per user instruction, since the result is negative and not being
carried forward as a reported/final number.

**Decision:** the improved cross-attention variant is **not** adopted.
The original Phase 7 Stage 2 cross-attention model (0.6209 +/- 0.0143)
remains the final fusion architecture and is what Phase 8 experiments
(cross-dataset generalization, fairness analysis, etc.) evaluate against.
No further tuning of the "improved" variant is planned — recorded here
only so the attempt and its outcome aren't lost, per this project's
practice of logging negative results alongside positive ones.

**Re-confirmed 2026-07-25 (project audit):** re-checked disk for any
`cross_attention_improved` checkpoint or summary JSON — `find . -iname
"*improved*"` returns only the training script/notebook
(`src/models/train_cross_attention_improved.py`,
`notebooks/pad_ufes20_cross_attention_improved_kaggle_notebook.md`,
`scripts/generate_improved_cross_attention_kaggle_notebook.py`); no
`.pt` or summary file exists anywhere in the repo. The
0.4804/0.5447/0.5028 numbers above remain exactly as flagged —
user-reported and unverified-from-file, not silently upgraded to
verified status anywhere else in this document.

---

<a id="phase-8-experiment-1-scope-final-model"></a>
## Phase 8, Experiment 1 — Cross-Dataset Generalization Scope, Final Model Confirmed (2026-07-23)

With the improved cross-attention variant ruled out (see previous entry),
Phase 8 proceeds using the original Phase 7 Stage 2 cross-attention model
(0.6209 +/- 0.0143 mean macro-F1, PAD-UFES-20 internal) as the final
fusion architecture. This does not change the Experiment 1 scope already
approved on 2026-07-18 ("Phase 8 Experiment 1 — Anatomical-Site Mapping
Approved" entry above); it confirms which checkpoint set the
cross-attention arm of that comparison uses going forward:
`logs/PAD_UFES20/checkpoints/cross_attention_seed{0,1,2}_best.pt` (the
original variant, already validated — see Phase 7 Stage 2 entry), not any
checkpoint from the improved-variant experiment.

**Experiment 1 recap — PAD-UFES-20 -> HAM10000 cross-dataset
generalization:**
- **Direction:** PAD-UFES-20 (train) -> HAM10000 (eval) is primary/
  headline; reverse direction deferred, do-if-time-allows.
- **Variants evaluated:** all 4 — image, metadata, late-fusion,
  cross-attention (original, not improved) — to see whether the
  generalization gap differs by architecture, not just report
  cross-attention alone.
- **Protocol:** Protocol A — full native 6-way argmax, scored only on the
  3 classes shared between PAD-UFES-20 and HAM10000, spillover rate
  reported (not masked/hidden).
- **Anatomical-site mapping:** already reviewed and locked
  (`docs/Phase8_Anatomical_Site_Mapping.csv`) — 9 clean, 3
  lossy/coarsened (ARM+FOREARM -> upper extremity, THIGH -> lower
  extremity), 2 unmapped (LIP, NOSE -> `__MISSING__`).
- **Fairness:** per-Fitzpatrick-group macro-F1 (equalized odds deferred —
  small per-group samples make it unstable).
- **Significance testing:** cross-attention vs. each of the other 3
  variants, 1000 bootstrap resamples, 95% CI.

**Status of implementation** (from the 2026-07-18 entry, unchanged by
today's decision): reduced-feature training scripts, config/dataset
support, and `evaluate_cross_dataset.py` are written and smoke-tested;
one full real run (image variant, seed 0) is validated end-to-end
(macro-F1 0.4577 on shared classes vs. 0.5703 within-dataset — expected
generalization-gap direction). **Not yet done:** the 9 Kaggle training
runs for the reduced-feature metadata/fusion/cross-attention variants
(3 variants x 3 seeds), after which `evaluate_cross_dataset.py` runs for
each and the 4-way PAD-UFES-20->HAM10000 comparison table is assembled.

**Next step:** run the 9 reduced-feature training jobs via
`notebooks/pad_ufes20_reduced_feature_kaggle_notebook.md` on Kaggle,
bring back and verify checkpoints (file-verified, per normal practice —
today's skip applies only to the negative-result entry above), then
evaluate all 4 variants cross-dataset and assemble the comparison table.

---

<a id="phase-8-reduced-feature-training-complete"></a>
## Phase 8 — Reduced-Feature Training — COMPLETE (2026-07-25)

All 9 reduced-feature training runs (metadata_reduced, fusion_reduced,
cross_attention_reduced x 3 seeds each) completed via Kaggle commit,
using `notebooks/pad_ufes20_reduced_feature_kaggle_notebook.md`. All 9
checkpoints and matching CSV/summary files verified present in
`logs/PAD_UFES20/` and cross-checked against the summary JSON contents
(file-verified, not user-reported-only).

**Results (PAD-UFES-20 internal, best validation macro-F1):**

| Branch | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| metadata_reduced | 0.4789 | 0.4832 | 0.4736 | 0.4786 |
| fusion_reduced | 0.5734 | 0.5835 | 0.5945 | 0.5838 |
| cross_attention_reduced | 0.5921 | 0.6777 | 0.6650 | 0.6449 |

**Important caveat — not comparable to the original 21-feature models:**
these are schema-matched (3-feature: age, sex, anatomical_site) versions
built specifically so their metadata preprocessing lines up with what
HAM10000 provides, for Phase 8 Experiment 1 cross-dataset evaluation.
Reducing from 21 features to 3 naturally changes the task the metadata/
fusion/cross-attention branches are solving, so these numbers are not a
regression relative to the original Phase 7 models (metadata/fusion/
cross_attention_seed{0,1,2}, e.g. cross-attention's original 0.6209 mean)
— they answer a different question (how well does a 3-feature-schema
model do) and exist only to support the cross-dataset generalization
experiment, not to replace the original within-dataset headline numbers.

**Next step:** run `evaluate_cross_dataset.py` for all 4 variants (image
using the original unmodified checkpoints; metadata/late_fusion/
cross_attention using these new reduced-feature checkpoints) and assemble
the 4-way PAD-UFES-20->HAM10000 generalization comparison table.

---

<a id="phase-8-experiment-1-complete"></a>
## Phase 8, Experiment 1 — PAD-UFES-20 -> HAM10000 Cross-Dataset Generalization — COMPLETE (2026-07-25)

All 12 runs (4 variants x 3 seeds) of `evaluate_cross_dataset.py` Protocol
A completed: image uses the original unmodified
`image_seed{0,1,2}_best.pt` checkpoints; metadata/late_fusion/
cross_attention use the new reduced-feature checkpoints logged above.
1,253 of HAM10000's 1,510 test rows (3 shared classes: Basal Cell
Carcinoma, Melanoma, Nevus) evaluated per run, full native 6-way argmax,
scored only on the 3 shared classes.

**Bug found and fixed during this run:** `evaluate_cross_dataset.py`'s
eval loop called `model(images, metadata)` unconditionally whenever
`need_metadata` was true, but `MetadataMLP.forward()` only accepts a
single metadata tensor (matches its usage in `evaluate.py`) - the
metadata-only variant doesn't take an image input at all. This raised
`TypeError: MetadataMLP.forward() takes 2 positional arguments but 3
were given` on first run. Fixed by branching on `variant == "metadata"`
to call `model(metadata)` only for that variant, leaving late_fusion/
cross_attention's `model(images, metadata)` call unchanged (their
`forward()` signatures do take both).

**Results — macro-F1 (3 shared classes), mean +/- population std across
3 seeds:**

| Variant | seed0 | seed1 | seed2 | mean | std | spillover (mean) |
|---|---|---|---|---|---|---|
| image | 0.4577 | 0.4247 | 0.5149 | 0.4658 | 0.0373 | 17.7% |
| metadata | 0.3079 | 0.2787 | 0.2893 | 0.2920 | 0.0121 | 13.1% |
| late_fusion | 0.4521 | 0.4715 | 0.4557 | 0.4597 | 0.0084 | 12.2% |
| cross_attention | 0.4916 | 0.4442 | 0.4604 | 0.4654 | 0.0197 | 11.6% |

**Per-class F1 (shared classes), mean across 3 seeds:**

| Variant | Basal Cell Carcinoma | Melanoma | Nevus |
|---|---|---|---|
| image | 0.2609 | 0.3335 | 0.8028 |
| metadata | 0.1393 | 0.2071 | 0.5296 |
| late_fusion | 0.2417 | 0.3360 | 0.8015 |
| cross_attention | 0.2399 | 0.3526 | 0.8037 |

**Reading the table:**
- **Metadata alone generalizes worst** (0.2920) - unsurprising given it's
  the reduced 3-feature schema (age/sex/anatomical_site) and metadata
  distributions (especially anatomical_site coverage - ~17% of HAM10000
  rows fell to `__MISSING__`, see per-run `metadata_coverage` in the JSON
  reports) shift more across datasets than pixel statistics do.
- **Image, late_fusion, and cross_attention cluster tightly** (0.4597-
  0.4658 mean) - fusing in the weak reduced-metadata signal neither helps
  much nor hurts much relative to image-alone under zero-shot transfer;
  none of the three differences look likely to be significant by eye
  (overlapping seed-to-seed ranges), though the formal bootstrap
  significance test (cross-attention vs. each of the other 3, 1000
  resamples, 95% CI - scoped in the 2026-07-23 entry above) has not been
  run yet.
- **Nevus dominates every variant's per-class F1** (0.53-0.80) while
  Basal Cell Carcinoma and Melanoma lag well behind (0.14-0.35) - this
  mirrors PAD-UFES-20's own class imbalance and is consistent across all
  4 architectures, not an artifact of one branch.
- **All 4 variants underperform their PAD-UFES-20-internal macro-F1**
  (image 0.5703, cross_attention 0.6209 original / 0.6449 reduced-feature
  mean) - the expected generalization-gap direction for genuine zero-shot
  cross-dataset transfer.

**Not yet done (at time of this entry):** bootstrap significance testing
(cross-attention vs. each other variant), per-Fitzpatrick-group fairness
breakdown, and the reverse-direction (HAM10000 -> PAD-UFES-20)
experiment, all still scoped as deferred/next-step per the 2026-07-23
entry.

**Update — bootstrap significance testing completed (2026-07-25):** see
`docs/Phase8_CrossDataset_Generalization_Results.md` §"Bootstrap
significance testing". Result confirms the "cluster tightly" read above
was correct, with one important nuance for the thesis writeup:
**cross_attention's edge over image and over late_fusion is NOT
statistically significant** (p=0.970 and p=0.590 respectively, both 95%
CIs straddling zero) — **only cross_attention vs. metadata is
significant** (p<0.001). **This caveat must be preserved wherever these
cross-dataset results are summarized for the thesis** (abstract,
methodology/results chapters, any headline-result slide) — the correct
claim is "cross-attention matches image/late-fusion performance under
zero-shot transfer while significantly beating metadata-alone," not
"cross-attention generalizes better than image-only or late-fusion."
Overclaiming generalization superiority here would not survive a viva
question that asks for the significance test.

**Clean writeup:** full per-seed/mean/std/spillover/per-class-F1 tables
assembled into `docs/Phase8_CrossDataset_Generalization_Results.md`.

---

<a id="pad-ufes20-image-path-integrity-check"></a>
## Due-Diligence — PAD-UFES-20 Image Path Integrity, Exhaustive Check (2026-07-25)

Every row (not a sample) in PAD-UFES-20's `train.csv`/`val.csv`/`test.csv`
was checked: `resolve_image_path(row["image_path"])` resolved and
`Path.exists()` verified for all 2,298 rows (train 1,606 + val 338 + test
354). **0 missing/unresolvable paths.** This closes out any residual
doubt about silent image-loading failures affecting the Phase 6-8 results
reported above.

---

<a id="phase-8-bootstrap-significance-complete"></a>
## Phase 8 — Bootstrap Significance Testing — COMPLETE (2026-07-25)

**Scope (approved earlier this session):** cross-attention vs. each of
the other 3 variants (image, metadata, late_fusion) - 3 comparisons, all
sharing the cross-attention anchor. Paired bootstrap, row-level
resampling of the 1,253 HAM10000 eval rows, seed-averaged per iteration,
1,000 resamples, percentile-method 95% CI, fixed RNG seed (42, distinct
from the 3 model-training seeds) for reproducibility. Both the
uncorrected alpha=0.05 and the Bonferroni-adjusted alpha=0.05/3~=0.0167
reported by default (3 comparisons share one anchor - flagged in advance
as a predictable reviewer question, costs nothing extra to compute).

**Prerequisite implemented first:** `evaluate_cross_dataset.py` never
saved per-row predictions (only the aggregate confusion matrix/macro-F1),
so bootstrap resampling had nothing to resample. Added a non-invasive
per-row prediction CSV dump
(`reports/PAD_UFES20/cross_dataset/predictions_{variant}_seed{seed}_pad_to_ham.csv`)
after the existing forward pass - the existing aggregate JSON output and
metrics logic are untouched. All 12 evals (4 variants x 3 seeds) were
re-run to produce these CSVs; every re-run macro-F1 matched the
already-logged number exactly, confirming this was purely additive, not
a re-verification catching a discrepancy. Row order was independently
confirmed identical across all 12 CSVs (same HAM10000-filtered rows in
the same order) before it was relied on for paired resampling.

**Script:** `src/evaluation/bootstrap_significance.py`. **Output:**
`reports/PAD_UFES20/cross_dataset/bootstrap_significance.json`.

**Results:**

| Comparison | Observed diff | 95% CI | p-value (2-sided) | Sig. alpha=0.05 | Sig. Bonferroni alpha~=0.0167 |
|---|---|---|---|---|---|
| cross_attention vs. image | -0.0004 | [-0.0189, +0.0196] | 0.970 | No | No |
| cross_attention vs. metadata | **+0.1734** | **[+0.1341, +0.2097]** | **0.000** | **Yes** | **Yes** |
| cross_attention vs. late_fusion | +0.0057 | [-0.0139, +0.0247] | 0.590 | No | No |

**Interpretation:** cross-attention's advantage over metadata-alone on
this cross-dataset transfer task is real and highly significant under
both thresholds. Cross-attention is **not** statistically distinguishable
from image-alone or late_fusion here - both CIs comfortably straddle 0.
**This means the apparent 0.4597-0.4658 mean-macro-F1 ranking among
image/late_fusion/cross_attention in the cross-dataset generalization
table should not be reported as one architecture "beating" the others on
this specific transfer task** - only that all 3 clearly and significantly
outperform metadata-alone, while remaining statistically tied with each
other on HAM10000 transfer specifically. This does not change
cross-attention's status as the strongest architecture within-dataset
(0.6209 vs. 0.5703 image / 0.5731 late-fusion, PAD-UFES-20-internal,
already a clean non-overlapping-range separation per the Phase 7 Stage 2
entry) - it only tempers what can be claimed about the cross-dataset
transfer numbers specifically.

**Full writeup:** results and interpretation also added to
`docs/Phase8_CrossDataset_Generalization_Results.md`.

**Next step (approved order):** Fitzpatrick fairness analysis, then
external validation via ISIC (blocked on the 2 open ISIC gaps - see
project audit, 2026-07-25).

---

<a id="phase-8-fitzpatrick-fairness-complete"></a>
## Phase 8 — Fitzpatrick Fairness Analysis — COMPLETE; PAD-UFES-20 Test Split Now Spent (2026-07-25)

**Scope (approved earlier this session):** all 4 variants (image,
metadata, fusion, cross_attention), original full-feature checkpoints
(not the reduced-feature HAM10000-schema ones - not needed here, this
stays within PAD-UFES-20). Per-group macro-F1 on PAD-UFES-20's own test
split (Fitzpatrick only exists in PAD-UFES-20 - HAM10000 has no such
column, so this cannot extend the cross-dataset experiment). Small-sample
tiering: n<15 excluded from the macro-F1 table (count-only), 15<=n<30
included with a small-sample caution flag, n>=30 included with no flag;
missing Fitzpatrick reported as its own count/percentage, not treated as
a fairness group. Per-group 95% CIs via the same paired-bootstrap
machinery built for the cross-dataset significance test (1000 resamples,
percentile method, RNG seed 42).

**Process issue caught and resolved during implementation:** the initial
script draft asserted in its own docstring that PAD-UFES-20's test split
"had already been used once for Phase 6/7 Stage 1's final evaluation" -
this was an unverified assumption, and it was wrong. Checked directly:
no `eval_*_test.json` exists anywhere in `reports/PAD_UFES20/`, and
`evaluate.py`'s `--confirm-final` guard (in place since Phase 6 Stage 1,
per the 2026-07-09 entry) had never actually been invoked - only ever
described in code. The 12 inference runs for this fairness analysis were
therefore **the first-ever evaluation of any PAD-UFES-20 checkpoint on
the test split**, run through a new script that doesn't go through
`evaluate.py`'s guard at all - a real deviation from the project's own
"test touched only once, after all training/model-selection decisions
are finalized" discipline (decision 4), caught only after the runs had
already executed, not before. Flagged to the user immediately once
discovered, before writing up or logging any results. **User decision:**
keep the results and treat them as both the fairness breakdown and
PAD-UFES-20's official final Stage 1 test-set result - the single
sanctioned test-split use, retroactively justified since all Stage
1/Phase 7 training and model-selection decisions were already finalized
before this analysis ran (Phase 7 Stage 2 completed 2026-07-18, well
before this 2026-07-25 run).

**Official PAD-UFES-20 Stage 1 test-set result (first and only test-set
evaluation of these checkpoints):**

| Variant | seed0 | seed1 | seed2 | mean | std |
|---|---|---|---|---|---|
| image | 0.6019 | 0.6382 | 0.6123 | 0.6175 | 0.0153 |
| metadata | 0.5897 | 0.5975 | 0.6360 | 0.6077 | 0.0202 |
| fusion | 0.6261 | 0.6830 | 0.6606 | 0.6566 | 0.0234 |
| **cross_attention** | 0.6862 | 0.6721 | 0.7349 | **0.6977** | 0.0269 |

Architecture ranking matches the val-split ranking from Phase 7 Stage 2
(cross-attention > fusion > image ~= metadata) - consistent, not a
reversal. All 4 score higher on test than val; no other test-set number
exists for these checkpoints to cross-check against, since this is the
first time the split was touched.

**Fitzpatrick group sizes, test split (n=354):** 1=22, 2=120, 3=59, 4=15,
5=2, 6=1, missing=135 (38.1%). **Headline finding: the two darkest-skin
groups (5, 6) have only 2 and 1 rows total - too few to report any rate**,
and 38% of the test set has no Fitzpatrick value recorded at all. This
means **the fairness question that matters most clinically (performance
on the darkest skin tones) cannot be answered from this dataset** - an
absence-of-data finding, not a negative result, and the single most
important conclusion of this analysis. Any fairness claim drawn from this
work is bounded to Fitzpatrick types I-IV (and even IV is small-sample,
n=15) - must be stated explicitly wherever these results are cited, not
implied to cover the full clinical population.

**Per-group macro-F1, mean across 3 seeds (reportable groups only):**

| Variant | Group 1 (n=22) | Group 2 (n=120) | Group 3 (n=59) | Group 4 (n=15) |
|---|---|---|---|---|
| image | 0.2525 | 0.5446 | 0.4637 | 0.4051 |
| metadata | 0.3586 | 0.3690 | 0.5351 | 0.4053 |
| fusion | 0.2785 | 0.5638 | 0.5228 | 0.4190 |
| **cross_attention** | 0.3307 | 0.6212 | 0.5706 | 0.4365 |

Groups 1 and 4 carry a small-sample caution flag (n<30); bootstrap CIs
(in `reports/PAD_UFES20/fairness/fairness_results.json`) are wide across
every group, typically spanning 0.1-0.3 macro-F1 even for the
best-scoring group. cross_attention is the best or tied-best performer
in every reportable group, not just on the overall mean - a positive
signal that its overall lead isn't coming at the expense of the
weaker-performing groups, though still bounded by the same small-sample
caveats. No equalized-odds or other parity metric computed (deferred, as
scoped - per-group instability at this n would make such a metric itself
unreliable).

**Script:** `src/evaluation/evaluate_fairness.py`. **Outputs:**
`reports/PAD_UFES20/fairness/fairness_results.json`, per-row predictions
`reports/PAD_UFES20/fairness/predictions_{variant}_seed{seed}_test.csv`.
**Full writeup:** `docs/Phase8_Fitzpatrick_Fairness_Results.md`.

**Next step (approved order):** external validation via ISIC - the 2
open ISIC gaps flagged by the 2026-07-25 project audit (Archive 2's 3
unverified anatomical_site-adjacent whitelist fields, and the 3
cross-archive label conflicts) are now **resolved** (see "Gap B" entry
and "Proposed ISIC External Validation Scope" below) - no longer
blocking. Scope approved 2026-07-25, `evaluate_external_isic.py`
implemented and smoke-tested; 8 of 9 combinations remain to be run on
Kaggle.

---

<a id="test-split-safeguard-added"></a>
## Test-Split Single-Use Safeguard Added (2026-07-25)

**What happened:** `evaluate_fairness.py` read PAD-UFES-20's
`metadata_test.csv` without going through `evaluate.py`'s
`--confirm-final` flag (see "Phase 8 — Fitzpatrick Fairness Analysis —
COMPLETE" above) - that flag only ever gated `evaluate.py` itself, so a
second, independent script reading the same file bypassed the "test
split touched only once" discipline (decision 4) with no error raised.
Caught and flagged before any results were written up, per the "never
silently resolve" rule.

**Why the existing results still stand:** all Stage 1/Phase 7
training and model-selection decisions for PAD-UFES-20 were already
finalized (Phase 7 Stage 2 completed 2026-07-18, a week before this run);
the 12 fairness-analysis inference runs did not feed back into any
training or hyperparameter choice (post-hoc diagnostic only); and the
resulting test-split ranking (cross-attention > fusion > image ~=
metadata) matches the val-split ranking already established, i.e. no
post-hoc tuning happened and no surprising reversal needs explaining
away. Per user decision (2026-07-25), these 12 runs are now **the
official, final, one-time Stage 1 test-set evaluation** for
image/metadata/fusion/cross_attention on PAD-UFES-20 - locked, no further
hyperparameter changes, retraining, or re-evaluation of these checkpoints
on the test split. Full numbers: "Phase 8 — Fitzpatrick Fairness Analysis
— COMPLETE" above and `docs/Phase8_Fitzpatrick_Fairness_Results.md`;
cross-referenced from the Phase 6 Stage 1 and Phase 7 Stage 2 val-result
tables above so anyone reading only those sections is pointed at the
official test number instead of assuming val is final.

**Same gap existed for HAM10000:** `evaluate_cross_dataset.py` reads
`metadata_test.csv` for HAM10000 (zero-shot evaluation of PAD-UFES-20-
trained models, Protocol A - see "Phase 8, Experiment 1" above) and had
the identical unguarded read. That experiment is also already complete
and its ranking claims are unaffected (see "Phase 8, Experiment 1 —
COMPLETE"), so it is being locked retroactively for the same reason, not
re-run.

**Safeguard implemented:** a shared, dataset-scoped marker file rather
than a per-script flag, since the flag-based approach is exactly what
was bypassed here - a marker file is checked by every script that reads
a test split, not just the one that happens to remember to check.

- `src/evaluation/test_split_guard.py` (new): `check_test_split_available(ds_config, caller)`
  raises `SystemExit` if `data/processed/<Dataset>/TEST_SPLIT_CONSUMED.json`
  exists, naming the dataset, who consumed it, when, and where the
  official numbers live. `mark_test_split_consumed(...)` writes that
  marker.
- Wired into all 3 scripts that can read a `*_test.csv`:
  `evaluate.py` (both the single-checkpoint `evaluate()` path and the
  ensemble/TTA `evaluate_ensemble()` path, whenever `--split test`),
  `evaluate_fairness.py` (unconditional - it only ever touches
  PAD-UFES-20's test split), and `evaluate_cross_dataset.py` (reads
  HAM10000's test split unconditionally).
- Marker files written for both already-consumed splits:
  `data/processed/PAD_UFES20/TEST_SPLIT_CONSUMED.json` (consumed by
  `evaluate_fairness.py`, 2026-07-25) and
  `data/processed/HAM10000/TEST_SPLIT_CONSUMED.json` (consumed by
  `evaluate_cross_dataset.py`, 2026-07-25).
- Verified: `evaluate.py --dataset PAD_UFES20 --branch image --seed 0
  --split test --confirm-final` now fails fast with a clear message
  before loading any model or data; `--split val` runs are unaffected
  and reproduce the already-recorded seed0 image val macro-F1 (0.5529)
  exactly.
- The marker is deliberately not something a script can silently work
  around - deleting it to re-run would itself have to be a logged,
  user-approved decision to reopen decision 4 for that dataset, not a
  code change.

**Thesis-writing note (2026-07-26):** when writing the methodology
chapter, be precise about where PAD-UFES-20's **official** Stage 1
test-set number comes from. It is **not** the output of a separate,
dedicated test-set evaluation script — it is sourced from
`evaluate_fairness.py`'s 12 inference runs (4 variants x 3 seeds),
which were repurposed on 2026-07-25 to serve double duty as both the
Fitzpatrick fairness breakdown *and* the official one-time test-split
result (see "Important process note — first use of the test split" in
`docs/Phase8_Fitzpatrick_Fairness_Results.md`, and the "Why the existing
results still stand" paragraph above). If the methodology chapter
describes a distinct "test evaluation" step/script, that description
would not match what actually happened and should instead say: test-set
numbers were obtained as a by-product of the fairness-analysis run,
justified because no training/model-selection decisions were made
afterward. cross-attention's official test macro-F1 is **0.6977**
(mean of 0.6862/0.6721/0.7349), reported in "Phase 8 — Fitzpatrick
Fairness Analysis — COMPLETE" above.

---

<a id="isic-gaps-resolved"></a>
## ISIC External-Validation Gaps A & B — Resolved (2026-07-25)

Both gaps blocking ISIC external validation (flagged in the 2026-07-25
project audit, see "Test-Split Single-Use Safeguard Added" and Phase 8
fairness entries above) are now resolved.

### Gap A — 3 unverified anatomical_site-adjacent whitelist fields

`anatom_site_2`, `anatom_site_special`, and `dermoscopic_type` were left
"allowed-in-principle but deferred" in `feature_whitelist.md` on
2026-07-09, pending the same missingness x `attribution` crosstab used
to exclude the other 6 sparse fields that day - never run until now. Ran
it, on the identical train+val+test-combined methodology, and showed the
evidence to the user before excluding anything (same discipline as the
2026-07-09 audit and the `melanocytic` exclusion):

| Field | Overall missing | Hospital Clínic present | ViDIR present | Anonymous present | Verdict |
|---|---|---|---|---|---|
| `anatom_site_2` | 52.91% | 40.98% | 51.68% | 57.39% | keep - populated across all 3 institutions, ordinary sparsity |
| `anatom_site_special` | 96.21% | 3.53% | 4.65% | 1.96% | keep - uniformly sparse across all 3 institutions, ordinary sparsity |
| `dermoscopic_type` | 94.30% | **0.00%** | **0.00%** | 49.29% | **exclude** - reproduces the exact 0%/0%/nonzero institution-proxy pattern of the 6 already-excluded fields |

**User-confirmed decision:** exclude `dermoscopic_type` as a source-leak
risk (same category/reasoning as `anatom_site_3`/`4`/`5`,
`family_hx_mm`, `personal_hx_mm`, `clin_size_long_diam_mm`); keep
`anatom_site_2` and `anatom_site_special` in the whitelist -
`data/processed/ISIC_Archive_2/feature_whitelist.md` updated
accordingly (allowed-feature count 7 -> 6). Both remain outside the
active Phase 6 Stage 1 4-feature baseline, but that's a scoping artifact
of Stage 1 already being locked before this check ran, not a leak
finding.

### Gap B — 3 cross-archive label conflicts

`ISIC_0028619`, `ISIC_0011126`, `ISIC_0011118` (documented under
"Cross-Dataset Verification" above, 2026-07-08) each have disagreeing
`disease_label` values between ISIC Archive 1 and ISIC Archive 2.
Consistent with this project's existing "ambiguous label -> exclude,
never guess" precedent (ISIC Archive 1's 155 conflicting-folder images;
ISIC Archive 2's 255 missing-`diagnosis_3` rows), these 3 images are now
excluded from external-validation scoring in both archives, rather than
picking one archive's label as authoritative.

**Implementation:** `src/data_cleaning/label_conflict_filter.py` (new)
re-verifies all 3 image_ids are present with disagreeing labels in both
archives (fails loudly if the conflict list is ever stale, e.g. if a
future data refresh resolves the disagreement), then writes:
`data/processed/ISIC_Archive_1/label_conflict_exclusions.csv` and
`data/processed/ISIC_Archive_2/label_conflict_exclusions.csv` (each: one
`image_id` column, the same 3 rows). Re-verified labels at write time:
`ISIC_0011118` (Seborrheic Keratosis vs. Melanoma), `ISIC_0011126`
(Seborrheic Keratosis vs. Melanoma), `ISIC_0028619` (Nevus vs. Actinic
Keratosis) - matches the 2026-07-08 finding exactly. This is a separate
file from `external_validation_exclusions.csv` (the pre-existing
HAM10000-overlap exclusion) since it's a distinct exclusion reason with
its own provenance - see "Proposed ISIC External Validation Scope"
below for how the two combine at evaluation time.

**Status:** both gaps resolved. ISIC external validation is now
unblocked pending scope approval (see next entry).

---

<a id="isic-external-validation-scope-proposed"></a>
## Proposed ISIC External Validation Scope — APPROVED (2026-07-25)

Same two-stage process as every prior experiment: proposed here, not
started until approved.

**Direction:** HAM10000-trained models evaluated zero-shot on the ISIC
archives — the direction the project has documented as the planned use
case throughout (Cross-Dataset Verification entry, PROJECT_PLAN.md).
Not PAD-UFES-20-trained models: PAD-UFES-20 has zero image overlap with
the ISIC archives (unlike HAM10000, which is 66.5-98.6% overlapping),
so it needs no exclusion-list machinery, but the project's stated
external-validation use case is specifically "HAM10000->ISIC", and
that's what the exclusion lists (both existing ones) were built for.

**Archives, evaluated separately, never pooled:** ISIC Archive 1 and
ISIC Archive 2, each its own independent evaluation run against the same
HAM10000 checkpoints - continuing fix (b)'s existing per-archive
protocol (2026-07-08 decision) rather than merging them, which is also
what makes Gap B's resolution sufficient (each archive scores with its
own label for the 3 conflicting images once they're excluded, no need to
reconcile the two archives' disagreeing labels into one true value).

**Models:** `image` and `metadata` Stage 1 branches only, 3 seeds each
(the only branches trained on HAM10000 - fusion/cross_attention are
PAD-UFES-20-only, Phase 7 scope, never trained on HAM10000). 2 branches
x 3 seeds x 2 archives = 12 runs.

- **Archive 1:** image branch only - Archive 1 has 0 usable metadata
  columns (image-only by necessity, `feature_whitelist.md`), so the
  metadata branch cannot be evaluated there at all.
- **Archive 2:** both branches. Metadata branch needs `anatom_site_1`
  (or `anatom_site_general`) normalized into HAM10000's `anatomical_site`
  vocabulary before scoring - same normalization *mechanism* as
  `normalize_anatomical_site_for_cross_dataset()` (Phase 8 Experiment 1),
  but that specific mapping table was built for PAD-UFES-20's vocabulary,
  not Archive 2's - a **new mapping table for Archive 2's anatom_site
  values would need review before this specific sub-run**, same
  precedent as the PAD-UFES-20 mapping's 2026-07-18 approval. Flagging
  this now rather than assuming the existing map transfers.

**Class taxonomy - restrict to exact-string-matching shared classes only
(Protocol A precedent), not a semantic best-guess merge:**

| | HAM10000 ∩ Archive 1 | HAM10000 ∩ Archive 2 |
|---|---|---|
| Shared classes (exact string match) | Basal Cell Carcinoma, Dermatofibroma, Melanoma, Nevus, Vascular Lesion (5) | Basal Cell Carcinoma, Dermatofibroma, Melanoma, Nevus (4 - Archive 2 has no Vascular Lesion class) |

**Known, deliberately-unresolved naming/granularity mismatches (flagged,
not merged):** HAM10000's `"Actinic Keratosis / Intraepithelial
Carcinoma"` vs. both archives' `"Actinic Keratosis"` are very likely the
same clinical category under a different label string; HAM10000's
`"Benign Keratosis-like Lesion"` (a known coarse bucket covering
seborrheic keratosis, solar lentigo, and lichen-planus-like keratosis in
the source ISIC taxonomy) vs. the archives' finer `"Pigmented Benign
Keratosis"`/`"Seborrheic Keratosis"`/`"Solar Lentigo"` categories is a
real hierarchy mismatch, not just a spelling difference. Per this
project's "don't guess, exclude or ask" precedent (same one used for
`melanocytic`, the 155/255 ambiguous-label exclusions, and Gap B just
above), these are **not** auto-merged into the shared-class set now -
scoring stays restricted to the 5/4 exact-match classes above unless you
approve an explicit mapping for these two categories, which would
expand shared-class coverage meaningfully (AK and the keratosis-family
classes are common in all 3 datasets).

**Exclusions, applied per archive independently (union of two lists,
each archive scored using only its own two files):**

1. `external_validation_exclusions.csv` (existing, 2026-07-08) - drops
   images already seen by the HAM10000-trained model as training data
   (or leaking into HAM10000's own val/test split) - Archive 1: 1,362 of
   2,047 dropped, 685 remain; Archive 2: 9,873 of 25,076 dropped, 15,203
   remain.
2. `label_conflict_exclusions.csv` (new, this session) - drops the 3
   images with disagreeing cross-archive ground truth, from both
   archives' evaluation sets regardless of which archive is used, since
   the ground truth is unreliable either way.

**Protocol:** same as `evaluate_cross_dataset.py`'s Protocol A - each
HAM10000-trained model's native, unmodified classifier (full argmax over
its own 7-class output, never masked) is run on the filtered archive
images; macro-F1 and per-class F1 are reported restricted to that
archive's shared-class list (`labels=` parameter, so spillover into a
non-shared HAM10000 class still counts against the true class); full
7-class confusion matrix reported for transparency; per-row predictions
saved for reuse by `bootstrap_significance.py`, same as the PAD-UFES-20
version. Proposed implementation: extend/adapt
`evaluate_cross_dataset.py`'s pattern into a new
`evaluate_external_isic.py` rather than duplicating the mechanics from
scratch, since the two experiments share the same shape (native argmax,
foreign-dataset evaluation, shared-class-restricted scoring, row-level
CSV output).

**Test-split discipline:** this is HAM10000's *training* checkpoints
evaluated against *ISIC's* own data, not a second read of HAM10000's own
test split - no interaction with the `test_split_guard.py` marker
already placed for HAM10000. Each ISIC archive's own train/val/test
split is not being touched as a "test split" in the single-use sense
either, since ISIC's own splits were never used for any ISIC-internal
model training in this project (no ISIC-trained models exist) - this is
purely external validation of already-finalized HAM10000 checkpoints.

**Approved 2026-07-25**, with one adjustment to the open question: AK
and keratosis-family naming/granularity mismatches deferred (not
merged) - exact-string-matching shared classes only, per Protocol A
precedent. Logged as a documented future-work item, not needed for the
core result.

---

<a id="isic-archive2-metadata-mapping-approved"></a>
## ISIC Archive 2 -> HAM10000 Metadata Mapping — Proposed and Approved (2026-07-25)

Proposed as its own artifact before any code ran, same process as the
PAD-UFES-20 anatomical-site mapping (2026-07-18). `sex` and `age_approx`
needed no mapping - `sex` vocabulary (MALE/FEMALE) and numeric scale
already match HAM10000's exactly, verified directly rather than assumed.
`anatom_site_general` (Archive 2's finer of its two location fields)
needed a mapping into HAM10000's `anatomical_site` vocabulary -
`docs/Phase8_ISIC_Archive2_Anatomical_Site_Mapping.csv`:

| Archive 2 category | n | HAM10000 target | Type |
|---|---|---|---|
| lower extremity | 4,941 | lower extremity | clean |
| upper extremity | 2,887 | upper extremity | clean |
| palms/soles | 385 | acral | clean (definitional - "acral" is the standard dermatology term for palms/soles/nail-unit) |
| posterior torso | 2,765 | back | clean (standard synonym, no information loss) |
| anterior torso | 6,837 | trunk | lossy_coarsened (same precedent as PAD-UFES-20's ARM/FOREARM->upper extremity: anterior torso is genuinely part of the trunk region, mapped to HAM10000's own generic "trunk" catch-all) |
| lateral torso | 54 | trunk | lossy_coarsened (same reasoning) |
| head/neck | 4,550 | unmapped -> `__MISSING__` | ambiguous - HAM10000 splits this into 4 categories (face/scalp/neck/ear), no way to disambiguate, no single catch-all like "trunk" covers this split |
| oral/genital | 106 | unmapped -> `__MISSING__` | ambiguous - bundles "oral" (no HAM10000 equivalent, same gap as PAD-UFES-20's LIP) with "genital" |
| missing/NaN | 2,551 | `__MISSING__` | not a mapping question |

**Two rounds of user review before finalizing:** first round questioned
whether anterior/lateral torso should map to "trunk" or stay unmapped -
approved for "trunk" specifically because it's a genuine anatomical
hierarchy relationship (trunk subsumes anterior/posterior/lateral torso)
matching the already-approved ARM/FOREARM/THIGH coarsening precedent,
not an invented equivalence like the deferred AK/keratosis case above.
Second round confirmed posterior torso->back at the **clean** tier
(same concept, different label, no information loss), not the
coarsening tier.

**Implementation:** `src/models/config.py` -
`ISIC_ARCHIVE2_ANATOMICAL_SITE_CROSS_DATASET_MAP` and
`normalize_isic_archive2_anatomical_site_for_ham10000()`, mirroring the
shape of the existing PAD-UFES-20 map/function. Wired into
`evaluate_external_isic.py`'s `ExternalIsicEvalDataset._adapted_row()`.

---

<a id="evaluate-external-isic-implemented"></a>
## `evaluate_external_isic.py` Implemented and Smoke-Tested (2026-07-25)

Implements the approved scope above. Structure mirrors
`evaluate_cross_dataset.py`'s Protocol A: native unmasked argmax over
each HAM10000-trained model's full 7-class output, scored restricted to
that archive's exact-string-match shared classes (`labels=` parameter),
full 7-class confusion matrix for transparency, per-row predictions
saved to `reports/HAM10000/external_isic/predictions_{archive}_{variant}_seed{seed}.csv`
for reuse by `bootstrap_significance.py`.

Both exclusion lists (`external_validation_exclusions.csv`,
`label_conflict_exclusions.csv`) applied per archive before scoring, as
their union. Archive 1: 678 shared-class rows remain after filtering +
exclusion (from 2,047 total). Archive 2: 12,508 remain (from 25,076).

**Smoke-tested, not yet run to completion for all 12 combinations:**

- `--archive ISIC_Archive_1 --variant image --seed 0` run to completion
  locally (678 images, CPU) - macro-F1 (shared classes) 0.2563. Sanity
  numbers only, not a final result - single seed, not yet compared
  against anything.
- Archive 2 metadata path verified directly (not a full run, this
  machine is CPU-only per the 2026-07-09 Kaggle-move decision and 12,508
  images x EfficientNet-B0 would be slow): confirmed the fitted
  preprocessor's `anatomical_site` categories include `trunk`/`back`/
  `acral`/`__MISSING__` as expected, confirmed sample Archive 2 rows map
  correctly (`anterior torso`->`trunk`, `head/neck`->`__MISSING__`,
  missing->`__MISSING__`), and ran one real batch through the loaded
  `metadata_seed0_best.pt` checkpoint end-to-end (output shape `[8, 7]`,
  as expected for HAM10000's 7 classes).
- Archive 1's image branch (678 rows) is the only combination cheap
  enough to run fully on this machine; the other 11 (Archive 1 has no
  metadata branch; Archive 2 image x 3 seeds + metadata x 3 seeds =
  12,508 rows each) should move to Kaggle, same as every prior
  full-scale run in this project, rather than run slowly/riskily on CPU
  here.

**Next step:** run all 12 combinations (Archive 1: image x 3 seeds;
Archive 2: image x 3 seeds + metadata x 3 seeds) - proposed as a Kaggle
notebook, following the same "Save & Run All (Commit)" discipline
adopted after the Stage 1 session-loss incident.

---

<a id="isic-archive2-kaggle-mirror-quirks"></a>
## ISIC Archive 2 Kaggle Mirror (`andrewmvd/isic-2019`) - Known Packaging Quirks (2026-07-27)

Found while running `external_isic_evaluation_kaggle_notebook.md`'s
sanity-check cell against the live Kaggle mount. Recorded here so
neither quirk gets re-investigated from scratch by a future session -
both are fixed in `src/models/config.py`'s `resolve_image_path()`, not
worked around per-caller.

1. **Images folder is renamed.** Our local
   `data/raw/ISIC_Archive_2/images/` is packaged on this mirror as
   `ISIC_2019_Training_Input/` instead of `images/` - a plain rename, not
   a nesting difference. Fixed via `KAGGLE_REST_FOLDER_RENAME`.

2. **Some files are additionally renamed with a `_downsampled` suffix.**
   ~2,074 of our 25,076 processed IDs (8.3%) exist on this mirror only as
   `<image_id>_downsampled.jpg`, not `<image_id>.jpg` - e.g.
   `ISIC_0016058.jpg` only exists as `ISIC_0016058_downsampled.jpg`.
   Presumably the mirror uploader's own downsizing of oversized
   originals; undocumented upstream. This is a systematic naming
   convention affecting a real subset of files, not stray/corrupt
   entries - confirmed via a full ID-set comparison
   (`scripts/isic_archive2_id_comparison_cell.py`), not assumed from the
   3 IDs that happened to surface in a 20-row sanity-check sample. Fixed
   via `KAGGLE_FILENAME_FALLBACK_SUFFIX["ISIC_Archive_2"] = "_downsampled"`,
   tried as a fallback (checked with `.exists()`, never assumed) after
   the plain filename fails to resolve, at both the flat and doubled
   candidate locations.

**Verification discipline:** total image count matching our expected
25,331 (passed during initial mirror verification) was NOT sufficient to
rule out ID-level mismatches - a mirror can have the right *count* while
missing/renaming specific files, as happened here. `resolve_image_path()`
now has three independent, individually-`.exists()`-checked fallback
layers for Kaggle mirrors, in order: (a) `KAGGLE_DATASET_SUBPATH` (whole
dataset root shifted deeper - Archive 1's case), (b)
`KAGGLE_REST_FOLDER_RENAME` (top-level folder renamed) +
imgs_part_N-style doubling (folder nested inside itself), (c)
`KAGGLE_FILENAME_FALLBACK_SUFFIX` (individual filenames renamed). None of
these are assumed to apply beyond the specific dataset/mirror they were
confirmed for.
