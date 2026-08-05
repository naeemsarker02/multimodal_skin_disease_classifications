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
24a. [Literature Review — Fairness Papers Added, 19 Total (2026-07-28)](#lit-review-fairness-papers-added)
25. [Phase 7 Stage 1 — COMPLETE (2026-07-18)](#phase-7-stage-1-complete)
26. [MetaBlock Mechanism Confirmed; Phase 7 Stage 2 Proposal — APPROVED (2026-07-18)](#phase-7-stage-2-proposal)
27. [Phase 7 Stage 2 — COMPLETE (2026-07-18)](#phase-7-stage-2-complete)
28. [Phase 8 Experiment 1 — Anatomical-Site Mapping Approved; Reduced-Feature Models + Eval Script Implemented (2026-07-18)](#phase-8-experiment-1-implementation)
29. [Negative Result — Improved Cross-Attention Variant Underperforms Original (2026-07-23)](#negative-result-improved-cross-attention)
30. [Phase 8, Experiment 1 — Cross-Dataset Generalization Scope, Final Model Confirmed (2026-07-23)](#phase-8-experiment-1-scope-final-model)
31. [Phase 8B+8C — Master Plan Adopted; Step 0 Backup — COMPLETE (2026-07-29)](#phase8bc-step0-backup)
32. [Phase 8C, Step 1 — Dataset Expansion Candidate Research (2026-07-29)](#phase8bc-step1-dataset-candidates)
33. [Phase 8C, Step 1 — Source Verification: DERM12345, MED-NODE, DDI (2026-07-29)](#phase8bc-step1-verification)
34. [Imbalance-Ablation WIP Folded In as Step 3a; Naming Collision Fixed (2026-07-29)](#step3a-fold-in-and-rename)
35. [Step 2 Binding Rule — Locked Test Split Frozen (2026-07-29)](#step2-binding-test-split-rule)
36. [Forward-Reference Note for Step 4 — `backbone_fusion_model.py` WIP (2026-07-29)](#step4-forward-reference-backbone-fusion-wip)
37. [Step 2 — Dataset Integration Plan (PROPOSED) (2026-07-29)](#step2-integration-plan-proposed)
38. [Lesion Segmentation/Cropping — Reasoned Through, Recommend Defer (2026-07-29)](#segmentation-cropping-reasoning)
39. [Step 2 Integration Plan — FINAL APPROVAL (2026-07-29)](#step2-plan-final-approval)
40. [Step 2 — PAD-UFES-20-Expanded Built — COMPLETE (2026-07-29)](#step2-implementation-complete)
41. [Pre-Step-3 Wiring Verification — Real Bug Found and Fixed (2026-07-29)](#step3-wiring-verification-and-bug-fix)
42. [Phase 8B Step 3 — Backbone Comparison Kaggle Notebook Generated (2026-07-29)](#step3-kaggle-notebook-generated)

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
| 2. Literature Review | 🟡 In Progress | — | 19 papers reconciled (16 + 3 fairness-focused papers added 2026-07-28, rows 17-19 — see "Literature Review — Fairness Papers Added" below); row #11 full-text read 2026-07-28 (free arXiv preprint), row #10 (MetaBlock) mechanism-confirmed but primary text still paywalled with no free alternative found — see "Literature Review — Priority Full-Text Reads, Progress" below and `docs/Literature_Review.md`. Not yet complete: rows 12, 14, 16 still abstract-only, plus a dataset-citation gap (PAD-UFES-20's own Data in Brief paper) newly found and not yet added. |
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
| 8. Experiments & Evaluation | ✅ Completed | 2026-07-27 | PAD-UFES-20<->HAM10000 cross-dataset generalization: ✅ complete 2026-07-25 (bootstrap significance included — note: only cross_attention vs. metadata is statistically significant; cross_attention's edge over image-only and late-fusion is **not** significant in this transfer direction, see "Phase 8, Experiment 1 ... COMPLETE" entry). Fitzpatrick fairness analysis: ✅ complete 2026-07-25, also stands as PAD-UFES-20's official final Stage 1 test-set result (see "Test-Split Single-Use Safeguard Added"). Both dataset's test splits now locked. Ensemble/TTA on cross-attention explored and permanently rejected (Melanoma F1 collapses 0.364->0.20 under ensemble+TTA — disqualifying, not just "no improvement" — see "Ensemble/TTA Exploration" entry). HAM10000->ISIC external validation: ✅ complete 2026-07-27, all 9 combinations run (image seeds0-2 x Archive1, image+metadata seeds0-2 x Archive2), bootstrap significance included (image vs. metadata on Archive 2: +0.2502, 95% CI [+0.2368,+0.2641], p<0.001, significant) — see "Phase 8.2 — ISIC External Validation — COMPLETE" entry and `docs/Phase8_ISIC_External_Validation_Results.md`. All 4 planned Phase 8 components now complete. |
| 9. Thesis Writing Support | ⏳ Pending | — | |
| 10. arXiv preprint / submission | ⏳ Pending | — | |
| **8B. Backbone Comparison** (new, 2026-07-29) | 🟡 In Progress — Step 4 (Option B) complete | — | 5-backbone image-only comparison, extends Phase 8. Step 4 Option B (ConvNeXt-Tiny + DenseNet121 cross-attention fusion models, val-selected, one-time test-evaluated, paired-bootstrapped against the locked 0.6977 headline) complete 2026-08-01 — see "Step 4 (Option B) — Final Test Results and Bootstrap Comparison — COMPLETE". Best result: dual-backbone ensemble, test macro-F1 0.7321 (+0.0343 vs. locked headline, **not** statistically significant, p=0.062). Locked 0.6977 (EfficientNet-B0) remains the thesis headline. Next: Phase 8E (Option A, single joint three-way fusion), not yet started. |
| **8C. Dataset Expansion** (new, 2026-07-29) | 🟡 In Progress — Steps 1-2 done | — | `PAD_UFES20_Expanded` built 2026-07-29: DERM12345 (Melanoma 400 + SCC 266, ISIC-overlap check clean) + MED-NODE (Melanoma 70) added to TRAIN only; VAL/TEST verified byte-identical to original PAD-UFES-20 (never touched). Melanoma 38→508, SCC 135→401 in the image-only training file. New rows have no compatible clinical metadata — image-branch-only, never joint fusion training (`metadata_train_image_only.csv` vs. `metadata_train.csv`, see feature_whitelist.md). See "Step 2 — PAD-UFES-20-Expanded Built — COMPLETE". Next: Step 3 (5-backbone comparison, using Phase 8B's WIP). |

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
**RESOLVED 2026-07-28 — see "Literature Review — Fairness Papers Added, 19
Total" below; this gap description is left here only as a historical
record of the 2026-07-17 state.**

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

---

<a id="phase-8-2-isic-external-validation-complete"></a>
## Phase 8.2 — ISIC External Validation — COMPLETE (2026-07-27)

All 9 planned runs (3 seeds × [Archive 1 image + Archive 2 image + Archive 2 metadata]) executed on Kaggle per the approved scope ("Proposed ISIC External Validation Scope — APPROVED", 2026-07-25) and `evaluate_external_isic.py`. Output files verified read directly from disk (not from any pasted log) — every JSON's `macro_f1_shared_classes` is exactly the mean of its own `per_class_f1_shared_classes` values, confirming internal consistency between the stored macro-F1 and per-class numbers.

**Headline results — macro-F1 (shared classes):**

| Archive | Variant | seed0 | seed1 | seed2 | mean | std (population) | spillover rate (mean) |
|---|---|---|---|---|---|---|---|
| Archive 1 (n=678) | image | 0.2563 | 0.2275 | 0.2426 | **0.2421** | 0.0118 | 19.1% |
| Archive 2 (n=12,508) | image | 0.5029 | 0.4908 | 0.4799 | **0.4912** | 0.0094 | 23.0% |
| Archive 2 (n=12,508) | metadata | 0.2009 | 0.2710 | 0.2510 | **0.2410** | 0.0295 | 27.3% |

**Notable finding, verified directly from each run's confusion matrix:** Archive 1's 678-row eval set has **zero** true instances of 2 of its 5 shared classes (Basal Cell Carcinoma, Vascular Lesion) and only 7 of Dermatofibroma — Archive 1's low macro-F1 (0.2421) is substantially a consequence of this eval-set composition, not a weaker model than Archive 2's (0.4912), which has hundreds-to-thousands of true rows in 3 of its 4 shared classes. Flagged explicitly rather than left implicit, per this project's "don't silently average away a support artifact" precedent.

**Bootstrap significance (Archive 2, image vs. metadata — the only archive with both branches, so the only place this comparison is possible):** paired bootstrap, 1,000 resamples, RNG seed 42, same method as the PAD→HAM bootstrap test. `src/evaluation/bootstrap_significance_isic.py` (new). Result: **+0.2502** observed diff (image − metadata), 95% CI **[+0.2368, +0.2641]**, p<0.001 — significant. Image is a materially and significantly stronger generalizer than metadata on this transfer task, consistent with the same qualitative finding already established for PAD→HAM (there, cross_attention vs. metadata: +0.1734, [+0.1341, +0.2097]).

**Full writeup:** `docs/Phase8_ISIC_External_Validation_Results.md` (per-seed macro-F1, mean±std, per-class F1 for all 3 archive/variant combinations, spillover rates, bootstrap test, and reading notes).

**Fitzpatrick fairness doc checked for dangling follow-up before declaring Phase 8 complete:** `docs/Phase8_Fitzpatrick_Fairness_Results.md` has exactly one deferred item — an equalized-odds/parity metric, explicitly deferred as scoped (2026-07-25) because per-group instability at PAD-UFES-20's small Fitzpatrick group sizes would make such a metric itself unreliable. This was an intentional scope decision, not an open TODO, and is not a blocker for Phase 8 completion.

**Phase 8 status — all 4 planned components now complete:**
1. PAD-UFES-20↔HAM10000 cross-dataset generalization — ✅ complete 2026-07-25
2. HAM10000→ISIC external validation — ✅ complete 2026-07-27 (this entry)
3. Fitzpatrick skin-type fairness analysis — ✅ complete 2026-07-25
4. Bootstrap significance testing — ✅ complete for both cross-dataset directions (PAD→HAM 2026-07-25, ISIC Archive 2 image-vs-metadata 2026-07-27)

Phase 8 (Experiments & Evaluation) is now fully complete. Next: Phase 9 (Thesis Writing Support).

---

<a id="lit-review-priority-reads-progress"></a>
## Literature Review — Priority Full-Text Reads, Progress (2026-07-28)

Full project status audit run this session (comprehensive re-check of all 10 phases against `PROJECT_PLAN.md`, similar to the 2026-07-25 audit). Confirmed Phase 8 fully complete per the entry above; confirmed the two remaining blockers before Phase 9 can start in earnest are (1) the literature review's 8 abstract-only papers and (2) a pending git commit of the Phase 8.2 ISIC work — both actioned this session.

**Git commit:** the uncommitted Phase 8.2 output-reorg (`reports/HAM10000/*` → `reports/HAM10000/external_isic/*`), `bootstrap_significance_isic.py`, and `docs/Phase8_ISIC_External_Validation_Results.md` were committed (`79939d7`) before any further work, per explicit instruction — uncommitted work is a standing risk given this project's prior Kaggle-session-loss incident (see "Phase 6 Stage 1 — COMPLETE" above).

**Row #11 (Pacheco & Krohling 2020) — full text read.** Freely available via arXiv (`arxiv.org/abs/1909.12912`), no institutional access needed. Full methodology, results, and limitations now captured in `docs/Literature_Review.md`'s reconciled table — not repeated here in full, see that file. Headline: image+clinical fusion improves avg balanced accuracy from 0.650→0.718 (6 CNN backbones, 5-fold CV) via a tunable-ratio concatenation fusion (a `combination factor` balancing image-feature-reducer output size against fixed clinical-feature count) — a simpler mechanism than both MetaBlock (channel-gating) and this thesis's cross-attention, and a direct historical precursor to this thesis's own Phase 7 Stage 1 late-fusion baseline. **Notable finding worth checking against our own results:** the paper found clinical metadata improves 4 of 6 classes but does *not* help separate SCC vs. BCC (near-identical clinical profiles) — worth checking whether our own PAD-UFES-20 confusion matrices show the same BCC/SCC-adjacent confusion (note: our taxonomy doesn't include SCC as a separate PAD-UFES-20 class in the same way — needs a direct check against our `label_mapping.csv`/confusion matrices before citing this as a confirmed parallel, not assumed).

**Side finding — dataset citation gap identified, not yet fixed:** row #11's dataset (1,612 images) is the direct precursor to the public PAD-UFES-20 dataset this thesis uses (2,298 images). The public dataset's own citation paper — Pacheco et al., *"PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones,"* Data in Brief — was found during this search but is **not currently cited anywhere in this project's literature review or docs**, and is not part of the 16-paper comparative table (it's a dataset-release paper, not related work to compare against). **Action needed before the thesis Methodology chapter is written: add this as the formal PAD-UFES-20 dataset citation.** Flagged here rather than silently added, since it changes what "the literature review" should contain (16 comparative papers + this dataset citation, not 16 alone).

**Row #10 (MetaBlock, Pacheco & Krohling 2021) — still blocked on institutional access.** Re-checked for a free full text this session (arXiv: none exists; ResearchGate: HTTP 403; author's own publications page: links only to the paywalled IEEE Xplore page, no self-hosted PDF). The mechanism remains confirmed via the official code repo + a secondary source (per the 2026-07-18 entry) but the primary text itself has still never been read. **Open question for you:** do you have institutional/library access to fetch the IEEE JBHI PDF, or should the thesis proceed citing the mechanism as verified via the code repository + secondary source (with that caveat stated explicitly wherever it's cited)?

**Update (2026-07-28, continued session):** rows 12, 14, 15, 16 processed (12/14/16 abstract-level, genuinely paywalled with no free full text anywhere; 15 read in full via a free PMC copy — Scientific Reports is fully OA). Row 14 caught a title mislabel: the tracked "JI-ADF" name belonged to an unrelated paper by different authors; the actual tracked paper is Zuo/Wang/Wang's "CosCatNet" (AIIM 2025). Row 15 (Islam et al., 79,246-image UK teledermatology dataset) yielded two citable parallels: its unconfirmed-ground-truth limitation mirrors this thesis's own `biopsed`-leakage concern from the opposite direction, and its independent finding of Fitzpatrick V/VI under-representation corroborates this thesis's own Phase 8 fairness finding in a completely different cohort/country.

**Row #5 duplicate check — RESOLVED 2026-07-28.** Confirmed "Multimodal Skin Lesion Classification Using Deep Learning" = Yap, Yolland & Tschandl (2018, Experimental Dermatology) via exact title/year/topic match. **This does not reduce the paper count to 15** — there was no second table row duplicating this paper to merge away, only row #5's factual description needed correcting (dataset and reported metrics were both wrong in the original Excel summary). **Final reconciled count: 16 unique papers, confirmed.** While resolving this, also caught and fixed a pre-existing arithmetic error in `Literature_Review.md`'s intro paragraph (claimed 3+6+8=16, which doesn't reconcile; corrected to 3+6+7=16, matching the table's actual row composition).

**Not yet done:** row 2 (TG-CAVNet) and rows 6-9 still need a full-text pass (currently at abstract/secondary-source depth from the 2026-07-17 reconciliation), and the Fitzpatrick-fairness-literature search recommended 2026-07-17 (next up).

---

<a id="lit-review-fairness-papers-added"></a>
## Literature Review — Fairness Papers Added, 19 Total (2026-07-28)

The Fitzpatrick-fairness-literature search recommended above (and at
2026-07-17) was run this session. 3 papers added to `docs/Literature_Review.md`
and `docs/Literature_Review_Master.xlsx` as rows 17-19, bringing the
reconciled total from **16 to 19 unique papers**. Arithmetic: 3
(`Project_Tracking.md`-tracked) + 6 (net-new from user's Excel) + 7
(web-search, original round) + 3 (fairness-focused, this round) = 19 —
double-checked given the earlier 3+6+8 slip caught and corrected on
2026-07-28 (see "Row #5 duplicate check" above).

1. **Daneshjou et al. (2022)** — *Disparities in Dermatology AI Performance
   on a Diverse, Curated Clinical Image Set* (*Science Advances*;
   arXiv:2203.08807, free full text). Introduces the DDI dataset (656
   images, pathologically confirmed, diverse Fitzpatrick representation) and
   benchmarks SOTA dermatology classifiers against it: **27-36 percentage-point
   ROC-AUC drop**, concentrated on dark skin tones and uncommon diseases;
   dermatologists show the same pattern; fine-tuning on DDI narrows but
   doesn't close the gap. **Included despite being outside the 2023-2025
   preferred window** — logged exception: it's the field's foundational
   fairness-benchmark paper and the closest methodological analogue to this
   thesis's own Phase 8.3 analysis (external, pathologically-confirmed test
   set built specifically to expose skin-tone gaps).
2. **Alipour, Burke & Courtney (2024)** — *Skin Type Diversity in Skin
   Lesion Datasets: A Review* (*Current Dermatology Reports*; PMC11343783,
   free full text). Systematic review confirming Fitzpatrick
   under-representation is a **systemic, field-wide dataset problem**, not
   specific to PAD-UFES-20 or any single dataset. **This is now the primary
   citation for the "Literature Gap: Fitzpatrick/Skin-Tone Fairness"
   section in `Literature_Review.md`**, replacing the prior framing that the
   gap was entirely unaddressed in prior work — it directly corroborates
   this thesis's own Phase 8.3 finding (PAD-UFES-20 test split: only 3 rows
   across Fitzpatrick V/VI, 38% missing).
3. **Xu, Gui, Rotemberg, Wang, Chen & Daneshjou (2024)** — *A Framework for
   Evaluating the Efficacy of Foundation Embedding Models in Healthcare*
   (medRxiv 2024.04.17.24305983, free preprint). Evaluates Google's Derm
   Foundation Model; finds lower sensitivity for darker Fitzpatrick tones
   (4-6) even in the pretrained embedding space before fine-tuning.
   Contributes a **3-axis evaluation framework (general performance /
   bias-fairness / confounders)** worth citing as a structure for this
   thesis's own Phase 8.3 write-up in the Discussion/Results chapter.

All three are full-text-read already (2 open-access journal articles, 1 free
preprint) — none are abstract-only, unlike several rows in the original
16-paper set.

**Updated elsewhere for consistency:** Progress Tracker table (Phase 2 row,
above), TOC (new entry after "Literature Review Reconciliation — 16
Papers"), and the "Literature gap" paragraph in the 2026-07-17 reconciliation
section (marked resolved, left in place as historical record).

---

<a id="phase8bc-step0-backup"></a>
## Phase 8B+8C — Master Plan Adopted; Step 0 Backup — COMPLETE (2026-07-29)

**Scope adopted this session:** a combined Phase 8B (backbone comparison)
+ Phase 8C (dataset expansion) roadmap, 6 steps, user-directed, with
explicit user review/approval gates before Steps 0, 1, 2, and 4. Phase 8
(Experiments & Evaluation, closed 2026-07-27) is **not reopened** by this
— 8B/8C are new, additional phases, and the existing Phase 8 result
(cross-attention fusion, val macro-F1 0.6209±0.0143, test macro-F1
**0.6977**, locked 2026-07-25) remains the presentable thesis result
unless and until 8B/8C produce something that beats it *and* is shown
statistically significant (see Step 5/6 below, not yet reached).
`docs/PROJECT_PLAN.md` future-phases list updated accordingly (this
entry).

**Naming collision flagged, not yet resolved:** pre-existing uncommitted
code (`src/models/config.py`, `dataset.py`) already used the label
"Phase 8B" for a *different*, undocumented piece of work — a
class-imbalance ablation (`WeightedRandomSampler` + a `strong=True`
augmentation path gated to Melanoma/SCC via
`STRONG_AUGMENT_TARGET_CLASSES`), citing "docs/PROJECT_PLAN.md Phase 8B
scoping" for its rationale. No such scoping exists in `PROJECT_PLAN.md`
or anywhere in this file — the citation is currently a dangling
reference. Per user instruction (2026-07-29), this ablation work is left
in place as-is (not reverted) but is **out of scope for the Phase
8B/8C plan below**; "Phase 8B" from this point forward in project docs
refers exclusively to the 5-backbone comparison. **Open item for a
future session:** either retire the class-imbalance ablation's "Phase
8B" labeling (rename in code comments to avoid collision) or fold it
into this plan as an explicit Step 3a — not decided yet, flagged so it
isn't silently lost.

**Step 0 — Backup — COMPLETE, verified restorable:**

1. **Git tag:** `pre-phase8bc-baseline-2026-07-29`, on commit `c81c85d`
   (annotated, message states the locked val/test numbers). Everything
   under `logs/`, `reports/`, `docs/` was already clean/committed at
   this commit — the tag is an exact, ordinary-git-history snapshot of
   every locked checkpoint and result file, not a special copy.
2. **Physical archive** (belt-and-suspenders, outside the working tree,
   protects against repo corruption — not just git):
   `D:\Naeem\thesis-v2\backups\pre-phase8bc-baseline-2026-07-29.tar.gz`
   (357,573,414 bytes, MD5 `9f394b9b3fbfc44b175931bf6a118f83`, 550
   files). Contains `docs/`, `logs/` (all checkpoints, both datasets),
   `reports/` (all result JSON/CSV/figures), `data/processed/`,
   `data/interim/`, `src/`, `scripts/`, `notebooks/`. Deliberately
   excludes `data/raw/` (untouched originals, re-obtainable from source)
   and `.venv/`/`__pycache__` (regenerable). Verified: archive entry
   list includes `logs/PAD_UFES20/checkpoints/cross_attention_seed0_best.pt`,
   `docs/Project_Tracking.md`, `docs/Phase8_ISIC_External_Validation_Results.md`.
   **Note:** this archive also captured the uncommitted Phase 8B-labeled
   WIP files present in the working tree at backup time (both the
   backbone-comparison code and the class-imbalance ablation code above)
   — intentional, so nothing in flight is lost, but it means the archive
   is *not* a purely clean baseline the way the git tag is.
3. **`D:\Naeem\thesis-v2\backups\RESTORE_INSTRUCTIONS.md`** written:
   two restore paths (git tag for a guaranteed-clean baseline restore;
   archive for full disaster recovery), and explicit steps to confirm a
   restore reproduces 0.6209 val / 0.6977 test (checkpoint presence,
   `Project_Tracking.md` line citation, and a deterministic
   re-evaluation check against the locked test-set prediction CSVs).

**Verification shown to user before proceeding (2026-07-29):** `git
tag -l -n5` output confirming the tag and its message; archive folder
listing (`tar.gz` + `RESTORE_INSTRUCTIONS.md` present); `tar -tzf` spot
check confirming the checkpoint, tracking doc, and results doc are
inside the archive. User reviewed and approved proceeding to Step 1
before this doc-logging pass.

---

<a id="phase8bc-step1-dataset-candidates"></a>
## Phase 8C, Step 1 — Dataset Expansion Candidate Research (scoping only) (2026-07-29)

Web research run (background fork, 9 tool calls) per the approved Step 1
scope: real, license-clear, non-ISIC/HAM10000/PAD-UFES-20-overlapping
public dermatology sources to expand PAD-UFES-20's weakest classes
(Melanoma: 38 train images; SCC: 135 train images). **No integration or
code written — research/reporting only, pending user approval before
Step 2.**

**Ranked candidates:**

1. **DERM12345** (Harvard Dataverse, DOI 10.7910/DVN/DAXZ7P) — 12,345
   dermatoscopic images, 1,627 patients (Turkey, 2008–2021), 40
   subclasses. **Melanoma: 400 images, SCC: 266 images** — by far the
   strongest class-relevant counts of any candidate found. Malignant
   labels biopsy-confirmed; benign labels by two dermatologists
   (20+ yrs experience) where no histology/follow-up existed. License:
   **CC BY 4.0**. Has patient ID, modality, 3-level taxonomic labels,
   malignancy status, and an official train/test split.
   **⚠️ BLOCKING RISK, not yet resolved:** per the dataset's own
   secondary coverage (PMC11604664), DERM12345 is independently
   collected from three Turkish institutions but **"also accessible
   through ISIC Archive."** If our existing ISIC Archive 1/2 pulls
   include this mirrored copy, using DERM12345 as new training data
   could silently reinsert images we currently hold out as
   cross-dataset validation targets (Phase 8.1/8.2) — this would
   invalidate those already-locked results, not just this new work.
   **Must be checked against our actual ISIC Archive 1/2 image
   lists/hashes before this source can be approved**, not assumed safe
   from the license/label quality alone.
2. **MED-NODE** (Univ. Medical Center Groningen) — 170 macroscopic
   (non-dermoscopic) clinical images: 70 melanoma, 100 nevi, no SCC.
   License: **CC BY 4.0**, direct download, no registration. Confirmed
   independent origin (no ISIC/HAM10000 derivation found). Notably,
   modality-matched to PAD-UFES-20 (both macroscopic/clinical photos,
   not dermoscopic like ISIC/HAM10000) — a genuine compatibility point
   the other candidates don't share. Image-only, no tabular metadata.
   Too small to be a primary source; a clean, low-risk melanoma-only
   supplement.
3. **DDI — Diverse Dermatology Images** (Stanford AIMI; also the source
   of Daneshjou et al. 2022, already in this project's literature
   review — see "Literature Review — Fairness Papers Added" above,
   corroborates the 656-image, pathologically-confirmed description
   independently). 656 images (+ 665 more via a DDI-2 extension,
   self-identified-Asian patients). All pathologically/biopsy
   confirmed. License: **signed Research Use Agreement**, non-commercial
   — not a clean CC license, needs citation care. Built explicitly for
   Fitzpatrick-skin-tone diversity, directly relevant to re-running our
   own Phase 8.3 fairness analysis on an expanded set (see Step 2's
   open question below). Melanoma/SCC-specific per-class counts not
   yet confirmed from the dataset's own tables — needed before ranking
   this above MED-NODE with confidence.
4. **Fitzpatrick17k** — not recommended: labels are atlas/clinical-
   diagnosis based, not biopsy-confirmed (below this project's stated
   bar); melanoma coverage unclear/likely minimal; license unconfirmed
   in research.

**Excluded — confirmed overlap:** BCN20000 (confirmed part of the ISIC
Challenge 2019/2020 aggregate, served through the ISIC Archive itself —
excluded outright, no further evaluation needed).

**Deprioritized — overlap/provenance unconfirmed, not excluded outright:**
PH2 (200 images, too small to matter at this project's scale regardless
of overlap status), Derm7pt (license terms and ISIC-family overlap both
unconfirmed in research).

**Recommendation given to user:** pursue DERM12345 first, *contingent
on* the ISIC-overlap blocking check above, paired with MED-NODE as a
zero-risk melanoma supplement; treat DDI as a strong third option
pending its per-class counts and Research Use Agreement terms being
confirmed directly from the dataset page. Fitzpatrick17k and the
deprioritized items not recommended without further verification.

**Not yet done (as of first pass, 2026-07-29):** user approval of
specific sources; the DERM12345/ISIC-overlap check; DDI's exact
Melanoma/SCC counts and RUA terms. **All actioned same day — see next
entry.**

---

<a id="phase8bc-step1-verification"></a>
## Phase 8C, Step 1 — Source Verification: DERM12345 Overlap Check, MED-NODE, DDI (2026-07-29)

Per explicit user direction: DERM12345's ISIC-overlap risk treated as a
hard blocker (reject outright if found, not "exclude with a workaround"
— avoids adding a second exclusion-list mechanism to maintain). Also
renamed the pre-existing imbalance-ablation WIP's "Phase 8B" labeling to
avoid collision with this plan's Phase 8B (backbone comparison) — see
next entry.

**DERM12345 vs. ISIC Archive 1/2 — exact ID-match check, not pixel
hashing.** Reused the same method as the original HAM10000↔ISIC Archive
1/2 overlap discovery (`src/data_cleaning/cross_dataset_leakage_filter.py`
— exact `image_id` string matching across full train+val+test, not
perceptual/file hashing of pixel data). Chose this over a real pixel-hash
comparison because (a) it's the literal precedent method already used
and trusted for this exact class of decision, (b) DERM12345's images
aren't downloaded and pixel-hashing would require pulling all 12,345+
images for a check that ID-matching answers definitively if the ID
namespaces are what they claim to be, which was verified first, not
assumed.

- Downloaded DERM12345's own metadata files directly from Harvard
  Dataverse (`derm12345_metadata_{train,test}.tab`, via the Dataverse
  API, `doi:10.7910/DVN/DAXZ7P` file IDs 10736043/10736044) — 12,345
  rows, confirmed `image_id` format is `DERM_XXXXXX` (verified by direct
  inspection of the downloaded file, not the paper's prose, which doesn't
  document the exact scheme).
- Loaded every `image_id` from `data/processed/ISIC_Archive_1/metadata_{train,val,test}.csv`
  and `data/processed/ISIC_Archive_2/metadata_{train,val,test}.csv`
  (2,047 + 25,076 = 27,123 combined) — confirmed both use `ISIC_XXXXXXX`
  format by direct inspection.
- **Result: 0 overlapping `image_id`s against ISIC_Archive_1, 0 against
  ISIC_Archive_2** (exact set-intersection over the full 12,345 DERM12345
  IDs vs. the full 27,123 combined ISIC Archive IDs). Disjoint ID
  namespaces by construction (`DERM_*` vs `ISIC_*`) makes accidental
  collision structurally implausible, and the exhaustive check confirms
  it directly rather than relying on the namespace argument alone.
- **Corroborating context, not the basis for the decision:** the
  DERM12345 paper (PMC11604664) states *"This dataset will also be
  accessible from the ISIC archive..."* — future tense, describing a
  planned distribution channel as of publication (2024), not a claim
  that the images were already merged into any existing ISIC Archive
  snapshot. Our `ISIC_Archive_1`/`ISIC_Archive_2` pulls are both
  pre-2024 (Archive 2 = the `andrewmvd/isic-2019` Kaggle mirror), so
  even absent the ID check, temporal precedence alone makes inclusion
  implausible — the ID check turns "implausible" into "confirmed absent."
- **Verdict: DERM12345 APPROVED.** Zero overlap found, per the user's
  binding rule this is a clean pass, not a workaround-and-proceed.

**MED-NODE — partially confirmed, one gap flagged honestly.**
- License: **CC BY 4.0** — confirmed via a secondary source (Papers with
  Code / dataset aggregator listing), not found stated on the dataset's
  own page (`cs.rug.nl/~imaging/databases/melanoma_naevi/`, which lists
  download links but no explicit license text) or in the freely
  accessible parts of the original paper (Giotis et al. 2015,
  ScienceDirect — abstract-only, full text paywalled).
- Image counts: confirmed **70 melanoma, 100 nevus, 170 total**, from
  the dataset's own page.
- **Gap, not resolved:** the exact labeling methodology (histopathology/
  biopsy-confirmed vs. clinical/visual diagnosis) could **not** be
  confirmed from any freely accessible source — the primary paper is
  paywalled and no secondary source found states this explicitly. Given
  MED-NODE is a hospital dermatology department dataset (University
  Medical Center Groningen), biopsy-confirmation for the melanoma cases
  specifically would be clinically typical, but this is an inference,
  not a verified fact, and is reported as such rather than overclaimed.
- Overlap: no evidence found connecting MED-NODE to ISIC Archive or
  HAM10000 in any source checked; independent single-institution origin,
  2015 publication, non-dermoscopic clinical photography (distinct
  modality from both ISIC-family archives). No ID-based check was
  possible/needed (no shared ID namespace question — different
  imaging modality and pre-dates the datasets it would need to overlap
  with by original collection method, not just ID scheme).
- **Verdict: MED-NODE conditionally approved** — license and non-overlap
  are reasonably confirmed; the labeling-methodology gap is disclosed to
  the user as unresolved rather than silently assumed favorable.

**DDI — RUA terms and biopsy-confirmation now confirmed; exact
Melanoma/SCC counts still not obtainable without dataset registration.**
- License: confirmed via `ddi-dataset.github.io` — a **signed Research
  Use Agreement**, explicitly: personal/non-commercial research only, no
  redistribution, no sharing the download link, and explicitly **"data or
  images generated through the use of the... Dataset [may not] be used
  or relied upon in the diagnosis or provision of patient care."** Access
  requires individual registration via the Stanford AIMI portal.
- Labeling: confirmed via the full paper (Daneshjou et al. 2022, *Science
  Advances*, arXiv:2203.08807, read in full) — every lesion biopsy-proven,
  reviewed by a board-certified dermatologist and dermatopathologist;
  **Supplemental Table 1** gives the only breakdown available in the
  public paper: 656 images total, 171 malignant / 485 benign, split by
  Fitzpatrick group (FST I-II: 49 malignant/159 benign; FST III-IV: 74/167;
  FST V-VI: 48/159). **No per-diagnosis (Melanoma vs. SCC vs. BCC, etc.)
  breakdown appears anywhere in the paper's main text, tables, or
  supplement** — only the aggregate malignant/benign counts and the
  common-vs-uncommon diagnosis name lists (which confirm Melanoma and SCC
  are both present among the "common malignant" category, but not their
  individual counts).
- Overlap: no statement found connecting DDI to ISIC/HAM10000; DDI is
  explicitly built from Stanford Clinics EHR-sourced biopsy records
  (2010-2020), an independent origin.
- **Verdict: still not approved or rejected, per user instruction.**
  Getting exact Melanoma/SCC counts requires registering for and
  downloading the actual RUA-gated dataset (its metadata file, not the
  images, would likely suffice) — this is a real next step, not a dead
  end, but wasn't done here since it requires creating an account under
  the user's identity/institution, which needs the user's go-ahead.

**Open decision for user:** register for DDI access to get the exact
counts (who registers — user, given institutional affiliation may
matter for RUA terms), or proceed with DERM12345 + MED-NODE now and
treat DDI as a possible later addition once counts are in hand.

---

<a id="step3a-fold-in-and-rename"></a>
## Imbalance-Ablation WIP Folded In as Step 3a; "Phase 8B" Naming Collision Fixed in Code (2026-07-29)

Per user instruction: the pre-existing uncommitted class-imbalance
ablation WIP (flagged as a dangling-citation naming collision in the
"Phase 8B+8C — Master Plan Adopted" entry above) is now formally **Step
3a** of the Phase 8B/8C plan — runs alongside/before Step 3's 5-backbone
comparison, on EfficientNet-B0 only, as originally scoped in the code
comments themselves (isolate the sampler/strong-augment ablations before
stacking them onto the 5-backbone runs). This is a scope decision, not
new work — the ablation code itself is unchanged, still uncommitted, and
still unrun.

**Renamed in code (comments/help text only, no logic changed) to stop
saying "Phase 8B" for ablation-specific content:**
- `src/models/config.py`: `STRONG_AUGMENT_TARGET_CLASSES` block header
  and citation now say "Step 3a" and point at this doc entry (was citing
  a nonexistent `PROJECT_PLAN.md` "Phase 8B scoping" section — that
  citation was already dangling before this rename; now fixed to point
  somewhere real).
- `src/models/dataset.py`: `build_image_transform`'s `strong=True`
  docstring and `ImageDataset`'s `strong_augment_classes` docstring.
- `src/models/train.py`: the `sampler == "weighted"` inline comment, and
  both `--sampler`/`--strong-augment` CLI help strings (`main()`).
- `src/evaluation/evaluate.py`: the matching `--sampler`/`--strong-augment`
  CLI help strings.
- `src/models/train_backbone_fusion.py`: one cross-reference in the
  module docstring ("If Phase 8B's imbalance ablations..." →
  "If Step 3a's imbalance ablations...").

**Left as "Phase 8B" (correct, unchanged) — these genuinely are the
backbone comparison:** `src/models/backbones.py`, `backbone_fusion_model.py`,
the `--backbone`/`--backbone-a`/`--backbone-b` CLI help text in
`train.py`/`evaluate.py`, and `_image_run_name`'s general "Phase 8B's
5-backbone runs" reference (train.py). Also fixed `backbones.py`'s own
dangling citation (was pointing at the same nonexistent `PROJECT_PLAN.md`
"Phase 8B scoping" section; now points at the "Phase 8B+8C — Master Plan
Adopted" entry).

**Discovery made while reading this code for the rename, flagged for
Step 4 (not acted on yet):** `backbone_fusion_model.py`'s own docstring
describes it as fusing **two image backbones together, with no metadata
branch at all** — this matches neither of the two Step 4 fusion designs
(A: three-way image+image+metadata; B: two full backbone+metadata models
then ensembled) proposed for your decision. It's a third, narrower thing
(image-only two-backbone fusion) that may have been an earlier/exploratory
step by whoever wrote this WIP, not a finished answer to Step 4. Not
touched — flagging so it isn't mistaken for a pre-made Step 4 decision
when we reach that step.

**Progress Tracker (8B row) updated** to reflect Step 3a fold-in.

---

<a id="step2-binding-test-split-rule"></a>
## Step 2 Binding Rule — Locked Test Split Frozen, Expansion Train/Val-Only (2026-07-29)

**Formalized as the binding design principle for Step 2's dataset
integration plan, before that plan is written:**

- PAD-UFES-20's original, already-locked **TEST split stays
  byte-for-byte identical, forever** — never touched, never expanded,
  never re-split. This is the same file (`data/processed/PAD_UFES20/metadata_test.csv`)
  already spent under "Test-Split Single-Use Safeguard Added"
  (2026-07-25) for the existing 0.6977 result.
- Any approved external source (DERM12345, MED-NODE, and DDI if
  approved later) is added **only to TRAIN, and optionally VAL** — never
  to TEST.
- **Rationale (raised as a rigor concern by the assistant, accepted by
  the user 2026-07-29):** this is what makes a valid *paired* bootstrap
  comparison possible between the Phase 8C expanded-dataset model and
  the existing locked 0.6977 test result — same method already used for
  cross_attention vs. metadata and the PAD→HAM comparisons (resampling
  matched predictions on identical test instances). A new, re-split test
  set for the expanded dataset would make that comparison invalid (or at
  best an unpaired, weaker-evidence comparison) without any offsetting
  benefit, since protecting the existing result was already this
  project's stated top priority for this phase.
- This rule is now also recorded in `docs/PROJECT_PLAN.md`'s "Confirmed
  design decisions" section (this entry cross-referenced there) so it
  carries the same "do not re-litigate" status as this project's other
  locked methodology choices (patient/lesion-wise splitting, no
  image-copying, etc.).

**Not yet done:** the Step 2 integration plan itself (how DERM12345/
MED-NODE's images get mapped to the 6-class taxonomy, how the new
combined TRAIN/VAL splits are built patient/lesion-wise, whether
DERM12345's metadata columns need their own leakage audit) — next, once
DDI's status is resolved per the open decision above, or the user says
to proceed without waiting on DDI.

**Update (2026-07-29, same day):** User approved proceeding with
DERM12345 + MED-NODE now; DDI deferred as a parallel, possible-future
addition (user registering under their own identity, not blocking Step
2). **MED-NODE's conditional approval accepted as final**, with the
label-acquisition-method gap to be disclosed explicitly in the thesis
Methodology/Limitations section (exact language agreed): *"MED-NODE's
exact label-confirmation method could not be independently verified due
to the source publication being paywalled; the dataset's label quality
is accepted based on its established use in prior peer-reviewed
dermatology-AI literature."* This is a documented limitation, not a
disqualifying gap — flagged here so it isn't forgotten before the
Methodology chapter is written.

**Approved sources for Step 2, final: DERM12345 (clean, verified 2026-07-29) + MED-NODE (conditional, caveat above).**

---

<a id="step4-forward-reference-backbone-fusion-wip"></a>
## Forward-Reference Note for Step 4 — Existing `backbone_fusion_model.py` WIP (2026-07-29)

Per user instruction: `src/models/backbone_fusion_model.py` (pre-existing
uncommitted WIP, discovered during the Step 3a rename pass — see
"Imbalance-Ablation WIP Folded In as Step 3a" above) is left **untouched**
for now — not extended, not decided as anyone's Step 4 answer yet.

**When Step 4 is reached, present this file explicitly as a third
consideration alongside options A and B:** this pre-existing
dual-image-backbone fusion (concatenates two backbones' penultimate
embeddings, no metadata branch) could serve as the **foundation for
building Option A** (three-way image+image+metadata fusion) by adding a
metadata branch and a cross-attention mechanism to it, rather than
building Option A from scratch. Logged now specifically so this isn't
forgotten by the time Step 4 discussion happens (per user: "good catch,
thank you for flagging rather than silently repurposing it").

---

<a id="step2-integration-plan-proposed"></a>
## Step 2 — Dataset Integration Plan (PROPOSED, pending user review) (2026-07-29)

Full integration plan for merging DERM12345 + MED-NODE into a new
`PAD-UFES-20-Expanded` variant, per the master plan's Step 2 scope.
**Not yet approved or implemented** — presented for review per the
project's two-stage process.

**1. New variant, originals untouched.** New CSVs written under
`data/processed/PAD_UFES20_Expanded/` (mirroring the existing
`metadata_{train,val,test}.csv` / `label_mapping.csv` /
`feature_whitelist.md` / `dataset_description.md` convention).
`data/processed/PAD_UFES20/` is never modified. `image_path` for new
rows points back to `data/raw/DERM12345/` / `data/raw/MED-NODE/`
(new raw folders, added read-only, same no-copy convention as the
existing 4 datasets) — new images are not copied into `data/raw/PAD_UFES20/`.

**2. Taxonomy mapping — Melanoma/SCC only, everything else excluded on
purpose, not by omission.** DERM12345's own `main_class_1`/`sub_class`
breakdown (verified directly from its downloaded metadata) does have
clean mappings available for other PAD-UFES-20 classes too (e.g.
`keratinocytic/basal_cell_carcinoma`, n=423 → Basal Cell Carcinoma;
`keratinocytic/seborrheic_keratosis`, n=607 → Seborrheic Keratosis), but
**this plan proposes using only the two priority classes**, matching the
plan's own stated goal (Melanoma: 38 train images; SCC: 135) rather than
opportunistically pulling in classes PAD-UFES-20 is already reasonably
supported on. Reason, not just scope discipline: pulling in
dermoscopic images for well-supported classes too would mean *every*
class gets some modality-mixed data, which actually would have been the
*safer* choice re: point 3 below — restricting to 2 of 6 classes is a
deliberate tradeoff (see risk flagged in point 3), open for
reconsideration if the user prefers the safer/broader alternative.

| Source | `main_class_1` / `sub_class` | Count | → PAD-UFES-20 class |
|---|---|---|---|
| DERM12345 | `melanoma` (all 5 sub_classes: `melanoma`, `lentigo_maligna`, `acral_nodular`, `acral_lentiginious`, `lentigo_maligna_melanoma`) | 400 | Melanoma |
| DERM12345 | `keratinocytic`/`squamous_cell_carcinoma` | 266 | Squamous Cell Carcinoma |
| MED-NODE | `melanoma` (binary label) | 70 | Melanoma |

**Excluded from this mapping, flagged not guessed:** DERM12345's
`bowen_disease` (n=37, keratinocytic) — Bowen's disease is SCC-in-situ
in some taxonomies, but this is a medical-judgment call PAD-UFES-20's
own SCC label definition doesn't resolve for us; excluded per the
"don't guess" precedent (same treatment as the anatomical-site mapping
work) unless the user confirms it should be folded into SCC.
`cutaneous_horn` (n=12) — a morphological descriptor, not a specific
diagnosis (can indicate ACK, SCC, or benign lesions); excluded, same
reasoning. MED-NODE has no SCC images at all (melanoma/nevus only).

**Combined effect on Melanoma/SCC train supply:** Melanoma 38 → 508
(38 + 400 + 70); SCC 135 → 401 (135 + 266). Both remain PAD-UFES-20's
two smallest classes relative to Basal Cell Carcinoma (586) and Actinic
Keratosis (513), but the gap narrows substantially.

**3. Real risk found while drafting this — modality confound between
malignancy-priority classes and image source, needs a decision.**
DERM12345 is **dermoscopic** (specialized magnified/polarized-light
imaging); PAD-UFES-20 is **macroscopic** (smartphone clinical photos);
MED-NODE is macroscopic (modality-matched). If DERM12345 is added, then
**after expansion, Melanoma and SCC are the only two classes with
mixed-modality training images — every other class (BCC, ACK, SEK,
Nevus) stays 100% macroscopic.** A backbone could learn "dermoscopic-style
texture/vignette → malignant-priority class" as a shortcut correlated
with, but not equivalent to, real lesion morphology. This is
structurally the same *category* of risk as the already-documented
"ISIC Archive 2 sparse-field = institution proxy" leakage pattern (Phase
6 pre-condition decision), just in image-pixel-statistics space instead
of metadata-column space, so it's flagged with the same seriousness.
**Concretely, this risk would show up as:** inflated Melanoma/SCC
val-set F1 (if new data touches val) or apparent train-time
learnability that doesn't transfer to the **frozen, macro-only** locked
test set — i.e., exactly the failure mode where a claimed improvement
turns out to be a shortcut, not real generalization.
**Mitigation proposed (see point 4): new data goes to TRAIN ONLY,
never VAL** — this doesn't remove the shortcut risk during training,
but it means model *selection* (checkpoint choice via val macro-F1)
and *evaluation* both stay on pure macro-only PAD-UFES-20 data, so the
shortcut can't inflate the numbers we actually report or use for
early stopping. It also sets up a specific check for Step 6 (below).

**4. Split discipline — proposal: new data is TRAIN-ONLY, not
train+val (tightens the already-agreed "train/val, never test" rule).**
Original binding rule (already agreed, still holds): TEST never
touched. This plan proposes going one step further for VAL too, for
three reasons: (a) directly limits the modality-shortcut risk above to
training data only, keeping the val-based model-selection signal
faithful to the real target distribution; (b) simpler — no need to
patient-wise split DERM12345's `patient_id` (1,627 patients) or figure
out MED-NODE's (undocumented) patient linkage, since 100% of approved
new images go to one bucket; (c) more conservative, consistent with
this phase's "protect existing results first" priority. **If approved,
this becomes the binding rule going forward, superseding the "train
and optionally val" wording from the earlier Step 2 binding-rule entry.**

**5. Leakage-audit process for new columns — applied, result: moot by
design, documented anyway.** DERM12345's own metadata
(`patient_id`, `image_type`, `copyright-license`, `split`, `super_class`,
`malignancy`, `main_class_1`, `main_class_2`, `sub_class`, `label`) has
**no overlap with PAD-UFES-20's 21-feature clinical whitelist** (age,
sex, itch, grew, hurt, changed, bleed, elevation, anatomical_site,
fitspatrick, family/personal cancer history, etc. — DERM12345 simply
doesn't collect this kind of clinical-history data). **This means the
fusion/cross-attention model's metadata branch cannot be trained on the
new images at all — there is no compatible metadata to give it.**
Proposed resolution (not imputation — see reasoning below):
- **New images (DERM12345 Melanoma/SCC subset + MED-NODE Melanoma)
  train the IMAGE branch only** — used in Step 3's backbone comparison
  and Step 3a's ablations (image-only training), and to strengthen the
  image embedder that Step 5's fusion model warm-starts from (same
  warm-start pattern already used for every existing fusion model in
  this project).
- **The fusion/cross-attention model's own joint (image+metadata)
  training in Step 5 uses PAD-UFES-20's ORIGINAL multimodal train
  rows only** — every row needs a real image+metadata pair, so rows
  without compatible metadata can't participate in joint training
  regardless of the val/test question above.
- **Why not impute placeholder metadata + a missingness flag for the
  new rows instead:** rejected on the same reasoning as the already-locked
  "ISIC Archive 2 sparse-field exclusion" decision — missingness would be
  a near-perfect proxy for "which dataset this row came from," which is
  exactly the kind of source-identity leakage this project has already
  established a precedent against. This isn't a new principle, it's the
  same one reapplied to a new situation.
- `malignancy` (DERM12345) is, as expected, a 1:1 derivation of the
  class label (same category as `diagnostic_code`/`biopsed`) — verified,
  documented, **never used as model input**, matching this project's
  standard treatment of label-source columns. No other DERM12345/MED-NODE
  column is a candidate for model input under this plan, since none of
  them get used at all (image-only usage, per above).

**6. Fitzpatrick fairness — no rerun needed for the reason expected,
but a new check recommended.** Because the locked TEST split never
changes (point 4), Phase 8.3's existing fairness numbers on that split
remain exactly as valid as before — nothing to rerun there. **New
recommendation, not part of the original Step 2 scope:** once Step 5
produces the expanded-training fusion model, run Fitzpatrick-stratified
per-class F1 for *that* model on the same frozen test set as a genuinely
new analysis (not a rerun of old numbers) — this checks whether
training on DERM12345 (Turkish population, likely different Fitzpatrick
distribution than PAD-UFES-20's Brazilian cohort) shifts the new
model's fairness profile, for better or worse, relative to the existing
locked model's already-documented fairness result.

**Open decisions for user before implementation:**
1. Approve or reject the "train-only, not train+val" tightening in
   point 4.
2. Approve or reject excluding `bowen_disease` from the SCC mapping
   (point 2).
3. Approve or reject restricting to Melanoma/SCC only vs. also pulling
   in DERM12345's other clean-mapping classes (point 2's tradeoff note).
4. Approve the image-branch-only / warm-start integration design
   (point 5) as the resolution to the metadata-incompatibility problem.

---

<a id="segmentation-cropping-reasoning"></a>
## Lesion Segmentation/Cropping — Reasoned Through, Recommendation: Defer (2026-07-29)

User raised segmentation/cropping as an additional improvement lever,
explicitly asking for it to be reasoned through, not assumed. Assistant
recommendation: **defer, do not add to the current Phase 8B/8C plan.**

**1. Augmentation (Step 3a) — confirmed unchanged.** No new scope; the
existing class-targeted augmentation ablation for Melanoma/SCC stands as
already designed.

**2a. Isolation risk — confirmed, agrees with user's own framing.**
Adding segmentation at the same time as dataset expansion (Step 2) and
the 5-backbone comparison (Step 3) would make any score change
unattributable to a single cause — the same failure mode already lived
through once in this project (the "Improved Cross-Attention Variant"
negative result, 2026-07-23, where stacked changes made the regression
uninterpretable until unwound). **Recommendation: segmentation must be
its own separate, later, single-variable experiment**, applied only
after Step 5's fusion result is in, not bundled into Steps 2-5.

**2b. Feasibility — a real caveat, not just a green light.** A
legitimate pretrained option exists: U-Net-style lesion-segmentation
models trained on the ISIC 2018 Task 1 segmentation challenge data are
publicly available (multiple pretrained implementations) — training a
segmentation model from scratch would not be necessary. **However**,
those models are trained on **dermoscopic** images (ISIC), while
PAD-UFES-20 is macroscopic smartphone photos — the same modality
mismatch already flagged for DERM12345 (see Step 2 plan above). Applying
an ISIC-pretrained segmentation model to PAD-UFES-20's images is itself
a domain-shift application (different framing, background skin/clothing,
lighting, capture distance) that may segment unreliably without
fine-tuning — this needs its own feasibility check (e.g. visually
spot-checking segmentation output on a sample of PAD-UFES-20 images)
before being trusted, not assumed to transfer cleanly just because a
pretrained model exists. A simpler alternative (heuristic center-crop,
since PAD-UFES-20 photos are typically already lesion-centered
close-ups) may capture much of the benefit at far lower risk and should
be considered as a first, cheaper thing to try if this is ever pursued.

**2c. Test-set validity — agrees with user's concern, recommends option
(i) over (ii) if/when this is pursued.** If segmentation/cropping is
ever applied to the locked test set's images, that changes the pixel
input those images present, which breaks a fair single-variable
bootstrap comparison against the existing 0.6977 result (computed on
uncropped images) unless handled deliberately. Between the user's two
proposed options: **(i) re-evaluate the existing, already-trained
cross-attention checkpoint on cropped versions of the same locked test
images** is recommended over **(ii) treat segmentation as fully out of
scope for any comparison against 0.6977**. Reason: (i) is cheap (no
retraining — same frozen model weights, just different test-time
preprocessing, direct paired comparison) and it's the only option that
lets a future segmentation experiment actually claim segmentation
helped or didn't; (ii) would make the experiment's own results
permanently uncomparable to anything, which defeats the point of
running it.

**Overall recommendation given to user:** lesion segmentation is a
legitimate future lever but should **not** be added to the current
Phase 8B/8C plan. Propose it as a distinct future phase (tentatively
"Phase 8D"), sequenced strictly after Steps 2-6 complete, applied on top
of whichever configuration Phase 8B/8C lands on, using recommendation
(2c)(i)'s comparison methodology, and starting with a feasibility
spot-check (2b) before committing resources to it.

**User decision (2026-07-29): APPROVED as scoped — deferred.** "Phase
8D: Lesion Segmentation (Deferred)" is now a placeholder future phase
(see `PROJECT_PLAN.md`). Confirmed plan for when it starts: sequenced
strictly after Phase 8B/8C completes; feasibility spot-check of an
ISIC-pretrained segmentation model against PAD-UFES-20's macroscopic
images first; comparison methodology is option (i) — re-evaluate the
existing frozen cross-attention checkpoint on cropped versions of the
same locked test images, not a fresh out-of-scope claim. **No
implementation work started — placeholder only.**

---

<a id="step2-plan-final-approval"></a>
## Step 2 Integration Plan — FINAL APPROVAL (2026-07-29)

All 4 open items from the proposed plan approved by user, as scoped:

1. **Train-only (not train+val) domain-shift mitigation — APPROVED.**
   Val and test both stay pure, untouched, original PAD-UFES-20. This is
   now the binding rule for Step 2, superseding the earlier "train and
   optionally val" wording.
2. **Bowen's disease / cutaneous horn exclusion — APPROVED.**
3. **Restrict to Melanoma/SCC only this pass, do NOT pull DERM12345's
   other clean-mapping classes — APPROVED.** Explicit reasoning
   (user's own): this expansion's stated purpose is fixing the two
   identified bottleneck classes; pulling additional classes expands
   scope/QC surface/domain-shift risk to more of the dataset without a
   clearly stated need. Other clean-mapping classes (BCC n=423, SEK
   n=607, ACK n=58 — all identified during Step 2 planning) are
   explicitly deferred as a **separate, future, separately-evaluated
   addition**, not bundled into this pass.
4. **Image-branch-only / warm-start integration design — APPROVED.**

**Final integration spec locked in:** DERM12345 melanoma-family (400) +
SCC (266), MED-NODE melanoma (70) → `PAD_UFES20_Expanded` TRAIN only.
VAL/TEST = PAD-UFES-20's original files, byte-for-byte, untouched. New
images used for image-branch training (Step 3/3a) and fusion warm-start
pretraining only — never for joint image+metadata training, which stays
restricted to original PAD-UFES-20 rows.

**Proceeding to implementation.**

---

<a id="step2-implementation-complete"></a>
## Step 2 — PAD-UFES-20-Expanded Built — COMPLETE (2026-07-29)

**Image acquisition.** DERM12345's own metadata (`derm12345_metadata_{train,test}.tab`)
confirmed the authoritative row set for the approved mapping: labels
`{mel, lm, lmm, alm, anm}` (main_class_1="melanoma", all 5 sub_classes)
→ 400 rows; label `scc` → 266 rows; 666 total, cross-checked against the
group-by counts from the original Step 1 research (400/266 — exact
match). Rather than downloading the full DERM12345 archive (3 zips,
~6.6GB, to get 666 of 12,345 images — impractical given this machine's
~23GB free disk), used the `remotezip` package to fetch only the 666
needed JPEGs directly via HTTP range requests against Harvard
Dataverse's zip files (`test.zip` 136, `train_part_1.zip` 108,
`train_part_2.zip` 422 — zip-membership determined empirically by
listing each remote zip's folder contents, not assumed). All 666
extracted successfully, 0 missing, saved to `data/raw/DERM12345/images/`
(502MB). MED-NODE's full zip (26MB) downloaded directly (small enough
not to need the range-request approach); 70 melanoma images used from
its `melanoma/` folder (the `naevus/` folder, 100 images, extracted but
unused — out of scope per the approved Melanoma/SCC-only restriction).

**Dataset build.** `src/data_cleaning/pad_ufes20_expanded/c01_build_expanded_dataset.py`
(run via `src/data_cleaning/run_build_pad_ufes20_expanded.py`) built
`data/processed/PAD_UFES20_Expanded/`. Verified directly (not assumed):
- `metadata_train.csv`, `metadata_val.csv`, `metadata_test.csv` are
  byte-for-byte identical to `data/processed/PAD_UFES20/`'s originals
  (`DataFrame.equals()` check, all 3 True) — the frozen-split rule
  holds by direct verification, not just by construction.
- `metadata_train_image_only.csv`: 2,342 rows = 1,606 original + 666
  DERM12345 + 70 MED-NODE. `dataset_source` value_counts confirms the
  exact split. Class counts: **Melanoma 38→508, SCC 135→401** (both
  exactly matching the plan's predicted totals), all other 4 classes
  unchanged (Basal Cell Carcinoma 586, Actinic Keratosis 513, Nevus 167,
  Seborrheic Keratosis 167).
- 0 missing-image warnings (every `image_path` for all 736 new rows
  resolves to a real file).
- Spot-checked 5 random new-row images load correctly via PIL (RGB,
  varied resolutions from 576×768 to 3024×4032 — consistent with
  DERM12345 being dermoscopic/high-resolution vs. PAD-UFES-20's
  smartphone photos, the documented modality difference).

**Docs written:** `data/processed/PAD_UFES20_Expanded/dataset_description.md`,
`feature_whitelist.md` (explicitly warns `dataset_source` is a leakage
risk in this variant specifically, not just an identifier — ties back to
the modality-confound flag), `label_mapping.csv`.

**Not yet done:** actually training anything on this dataset (Step 3 —
5-backbone comparison — is next, once scoped/approved to proceed).

---

<a id="step3-wiring-verification-and-bug-fix"></a>
## Pre-Step-3 Wiring Verification — Real Bug Found and Fixed (2026-07-29)

Per user instruction, before starting the 15 backbone-comparison runs:
verified the train-only-not-val mitigation is actually wired into the
training code, not just true by file-naming convention. **This surfaced
a real, would-have-crashed-on-first-run bug**, not just a
confirmation.

**Bug found:** `c01_build_expanded_dataset.py` wrote `image_path` for
all 736 new rows as **absolute Windows paths**
(`D:\Naeem\thesis-v2\...\data\raw\DERM12345\images\mel\DERM_602864.jpg`),
not the `"data/raw/<Dataset>/..."` relative format every other row in
this project uses. `src/models/config.py`'s `resolve_image_path()`
unconditionally requires that relative format (checks
`parts[0]=="data", parts[1]=="raw"` before even checking
`IS_KAGGLE`) — confirmed via direct test that the old CSV would raise
`ValueError: Unexpected image_path format` the moment training touched
any new-source row, locally or on Kaggle. Caught before any training
run, not after.

**Fixed:** `_image_path()`/`_load_mednode_rows()` now write
`Path.relative_to(PROJECT_ROOT).as_posix()` — same format as every
other row. Rebuilt `PAD_UFES20_Expanded`; re-verified: **736/736 new
rows resolve via the real `resolve_image_path()` function and exist on
disk** (not just string-format-checked — actually resolved and
`.exists()`-checked).

**Dataset wiring added (didn't exist before — `PAD_UFES20_Expanded` had
no entry in `src/models/config.py`'s `DATASETS` registry at all):**
- New `DatasetConfig(name="PAD_UFES20_Expanded", ..., train_csv_name="metadata_train_image_only.csv", image_branch_only=True)`.
  `train_csv` now correctly points at the image-only file;
  `val_csv`/`test_csv` point at this variant's own copies (verified
  byte-identical to the original in the prior entry).
- **`image_branch_only=True` is a hard guard, not documentation**:
  `train.py`'s `train_one_run()` and `evaluate.py`'s `main()` both now
  raise (`ValueError`/`SystemExit`) if anyone requests `--branch`
  other than `image` for this dataset — makes the
  metadata-incompatibility problem (flagged in the Step 2 plan) fail
  loudly instead of silently training on all-NaN rows. Verified: calling
  `train_one_run('PAD_UFES20_Expanded', 'metadata', seed=0)` raises with
  a clear message pointing back to this decision.
- `KAGGLE_DATASET_SLUGS["DERM12345"]`, `["MED-NODE"]`, and
  `KAGGLE_PROCESSED_SLUGS["PAD_UFES20_Expanded"]` added as
  `REPLACE_WITH_*` placeholders (same convention as every other
  not-yet-uploaded Kaggle dataset in this file) — **the user must upload
  `data/raw/DERM12345/`, `data/raw/MED-NODE/`, and
  `data/processed/PAD_UFES20_Expanded/` as 3 new private Kaggle
  datasets before the Step 3 notebook can actually run**, same as every
  prior phase's Kaggle-migration step. Not done yet — flagged, not
  silently assumed.

**End-to-end smoke test (not just unit-level):** built a real
`ImageDataset` from `PAD_UFES20_Expanded`'s `train_csv` (2,342 rows),
pulled a real batch through a `DataLoader` (`torch.Size([8, 3, 224, 224])`,
labels present) — confirms the whole load path works, not just that
files exist. Separately loaded `val_csv` directly and confirmed **all
338 rows have `dataset_source == "PAD_UFES20"`, zero contamination** —
the train-only-not-val rule verified by inspecting the actual data, not
inferred from the build script's logic alone.

---

<a id="step3-kaggle-notebook-generated"></a>
## Phase 8B Step 3 — Backbone Comparison Kaggle Notebook Generated (2026-07-29)

`scripts/generate_backbone_comparison_kaggle_notebook.py` (new, mirrors
every prior notebook generator's structure exactly — reads real
`src/models/` source into `%%writefile` cells, never hand-typed) →
`notebooks/pad_ufes20_expanded_backbone_comparison_kaggle_notebook.md`.
24 cells: folder verification, setup, 5 `%%writefile` cells
(`config.py`, `dataset.py`, `backbones.py`, `metadata_model.py`,
`train.py`), a sanity-check cell, a full model/GPU/dependency check
cell, and 15 training cells (5 backbones × 3 seeds,
`--dataset PAD_UFES20_Expanded --branch image`).

**Verified before presenting:**
- All 5 `%%writefile` cells have `%%writefile` as the exact first line
  (checked programmatically, not by eye — this exact class of mistake
  has silently broken a cell in a past session).
- The embedded `config.py` content includes today's `image_branch_only`
  fix and the `PAD_UFES20_Expanded` entry — confirmed by grepping the
  generated notebook, not assumed from "the generator reads the current
  file" alone.
- Sanity-check cell (Cell 8) asserts `val_df["dataset_source"]` is
  exactly `{"PAD_UFES20"}` and fails loudly if not — the train-only-not-val
  rule gets re-verified on Kaggle itself, every run, not just locally
  once.
- Full-check cell (Cell 9) forward-passes a real batch through all 5
  backbones AND calls `train_one_run("PAD_UFES20_Expanded", "metadata", seed=0)`
  to confirm the `image_branch_only` guard fires on Kaggle too, before
  any of the 15 real training runs start.

**Blocking, not yet done — 3 Kaggle "Add Data" uploads required before
this notebook can run:** `data/raw/DERM12345/`, `data/raw/MED-NODE/`,
and `data/processed/PAD_UFES20_Expanded/` all need to be zipped and
uploaded as new private Kaggle datasets (this assistant has no Kaggle
upload capability). `src/models/config.py`'s `KAGGLE_DATASET_SLUGS["DERM12345"]`,
`["MED-NODE"]`, and `KAGGLE_PROCESSED_SLUGS["PAD_UFES20_Expanded"]` are
`REPLACE_WITH_OWNER/REPLACE_WITH_SLUG` placeholders until then — Cell 1
will fail loudly (not silently) if run before this is done. Notebook
header spells out the exact 3 uploads needed.

## Phase 8B backbone normalization verified (2026-07-29)

`src/models/backbones.py` builds all 5 comparison backbones with a
single hardcoded `IMAGENET_MEAN`/`IMAGENET_STD` (defined in
`src/models/dataset.py`) applied uniformly. This was flagged for
explicit confirmation rather than assumption: different
`torchvision` pretrained-weights enums can in principle ship
different per-weights normalization stats.

**Verified, not assumed** — queried `weights.transforms().mean/std`
directly (project `.venv`, `torchvision==0.28.0+cpu`) for the exact
5 weights enums used in `backbones.py`:
- `EfficientNet_B0_Weights.IMAGENET1K_V1`
- `MobileNet_V3_Large_Weights.IMAGENET1K_V2`
- `DenseNet121_Weights.IMAGENET1K_V1`
- `ResNet50_Weights.IMAGENET1K_V2`
- `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`

All 5 resolve to identical `mean=[0.485, 0.456, 0.406]`,
`std=[0.229, 0.224, 0.225]` — the standard ImageNet1K recipe. (Their
`resize_size`/`crop_size` differ slightly per weights enum, but the
dataset pipeline here uses its own fixed `ResizePad` + resize, not
`weights.transforms()`, so that variation doesn't apply.) No fix
needed. Citation comment added at `src/models/dataset.py`'s
`IMAGENET_MEAN`/`IMAGENET_STD` definition pointing back to this
entry, so this doesn't silently drift into "assumed" territory in a
future session.

Notebook `notebooks/pad_ufes20_expanded_backbone_comparison_kaggle_notebook.md`
is approved to paste into Kaggle.

## Kaggle Commit Ran, Zero Training Happened — Stale Notebook `--dataset` Choices — Fixed (2026-07-29)

**Symptom:** the Kaggle commit "succeeded" (no crash reported by the
runner) but all 15 training cells (Cell 10–24) individually failed with
`train.py: error: argument --dataset: invalid choice: 'PAD_UFES20_Expanded'
(choose from PAD_UFES20, HAM10000)`. A "successful" commit with silently
failed subprocess cells — same failure shape as the earlier slug-wiring
bug (Pre-Step-3 Wiring Verification entry above): looks done, isn't.

**Root cause, checked not assumed:** `src/models/train.py`'s and
`src/evaluation/evaluate.py`'s live `main()` already build `--dataset`
choices as `choices=list(DATASETS)` — derived from the registry, not a
separate literal list — confirmed by reading both files directly. The
bug was **not** in the source; it was that
`notebooks/pad_ufes20_expanded_backbone_comparison_kaggle_notebook.md`
had been generated from an *older* `train.py` snapshot with a
hand-written `choices=["PAD_UFES20", "HAM10000"]`, before that file was
switched to derive from `DATASETS` — the notebook generator embeds
`%%writefile` cells from `src/models/` at generation time, so a stale
notebook silently drifts from current source until regenerated.

**Fix applied:**
1. Confirmed `train.py`/`evaluate.py`'s `--dataset` argparse already
   reads `choices=list(DATASETS)` — the registry-derived form item 3
   asked for was already in place; no separate literal list exists in
   either file to drift again.
2. Grepped every other CLI script's `choices=` for a hardcoded dataset
   list. `train_backbone_fusion.py`, `train_fusion.py`,
   `train_fusion_reduced.py`, `train_cross_attention_fusion.py`,
   `train_cross_attention_fusion_reduced.py`,
   `train_cross_attention_improved.py`, `train_metadata_reduced.py` all
   hardcode `choices=["PAD_UFES20"]` — but that's intentional and
   correct (fusion/cross-attention are PAD_UFES20-only per Phase 7/8B
   scope, enforced elsewhere too), not the same bug class. No other
   drifted list found.
3. Re-ran `scripts/generate_backbone_comparison_kaggle_notebook.py` to
   re-embed current source. Verified via grep that the regenerated
   notebook's Cell 10 (`train.py`) now reads
   `choices=list(DATASETS)`, not the stale hardcoded list.
4. Smoke-tested locally before reporting ready:
   `python -m src.models.train --help` shows
   `--dataset {PAD_UFES20,HAM10000,PAD_UFES20_Expanded}`; passing a
   bogus `--dataset` value correctly raises
   `SystemExit 2` listing all three; `evaluate.py --help` shows the
   same three-way choice.

**Lesson for future notebook generators:** a `%%writefile`-embedding
notebook generator is only as current as the last time it was run —
regenerate it as the final step of any change to an embedded file, not
just when first creating the notebook.

## Step 4 — Cross-Attention Backbone Fusion (ConvNeXt-Tiny + DenseNet121) Implemented (2026-07-31)

Approved plan: Option B (two full multimodal models via the existing
Phase 7 Stage 2 cross-attention mechanism, one per Phase 8B top-2
backbone, then an unweighted-probability-averaging ensemble) — not
Option A (a novel three-way joint model), and not extending
`backbone_fusion_model.py`'s concatenation pattern, which would have
reproduced the exact raw-dimension-dominance flaw Stage 2's
cross-attention was built to fix.

**New files:**
- `src/models/spatial_backbone_embedder.py` — `SpatialBackboneEmbedder`
  for `convnext_tiny`/`densenet121` only (the existing EfficientNet-B0
  `SpatialImageEmbedder` in `cross_attention_fusion_model.py` stays
  untouched for Phase 7 Stage 2 reproducibility). Per-backbone pre-pool
  extraction verified directly against torchvision's real `forward`
  source, not assumed: `convnext_tiny`'s `.features(x)` is already the
  natural pre-pool representation (`LayerNorm2d` sits inside
  `classifier[0]`, applied *after* pooling); `densenet121` requires an
  explicit `F.relu(.features(x))` before treating the map as spatial
  tokens, since `DenseNet.forward` applies that ReLU *outside*
  `.features` (which ends in a bare `norm5` BatchNorm) — omitting it
  would feed the cross-attention un-activated, possibly-negative
  BatchNorm output, which is not what the pretrained/warm-started
  weights were ever trained to have pooled.
- `src/models/cross_attention_backbone_fusion_model.py` —
  `CrossAttentionBackboneFusionModel(backbone_name, ...)`, structurally
  identical to `CrossAttentionFusionModel` (reuses `MetadataEmbedder`
  and `MetadataChannelGate` unchanged), parameterized over the image
  side via `SpatialBackboneEmbedder`.
- `src/models/train_cross_attention_backbone_fusion.py` — dataset is
  fixed to `PAD_UFES20` (not a CLI choice): `PAD_UFES20_Expanded` is
  `image_branch_only=True` and structurally has no usable metadata for
  its DERM12345/MED-NODE rows. Cross-dataset warm start, verified
  against what's actually on disk: image embedder loads
  `logs/PAD_UFES20_Expanded/checkpoints/image_{backbone}_seed{N}_best.pt`
  (Step 3/Phase 8B — confirmed these exist there and nowhere under
  `logs/PAD_UFES20/checkpoints/`), metadata embedder loads
  `ds_config.stage1_checkpoints_dir/metadata_seed{N}_best.pt` (Phase 7
  Stage 1, PAD_UFES20's own). The train/val fine-tuning loop itself runs
  entirely on PAD_UFES20's original `metadata_train.csv`/`metadata_val.csv`
  — zero expanded rows are directly seen in this step; their only
  influence is indirect, already baked into the warm-started image
  weights.
- `src/evaluation/evaluate.py` extended (additive only) with
  `--branch cross_attention_backbone` (single checkpoint,
  `--backbone {convnext_tiny,densenet121} --seed N`) and
  `--branch cross_attention_backbone_ensemble` (`evaluate_dual_backbone_ensemble`
  — unweighted softmax-probability averaging between the two
  seed-matched checkpoints, no training, no learned combiner: every
  split already has a fixed role in this project's discipline — train
  fits the base models, val selects/early-stops them, test is locked —
  so a parameter-free combination step is the only one that doesn't
  need a new place to be fit without leaking one of those roles).

**Smoke-tested locally on a real 12-row train / 12-row val PAD-UFES-20
subset (2 rows per class, all 6 classes present) via `.venv` (CPU), for
both backbones:** confirmed strict-mode warm-start checkpoint loading
(both the Step 3 image checkpoint and the Stage 1 metadata checkpoint),
single-sample and batch-of-4 forward passes with correct output shapes,
one real train epoch + one real eval epoch via
`run_epoch_cross_attention_backbone`, a checkpoint save/reload round
trip, and the dual-backbone ensemble's softmax-averaging path on the
12-row val subset — all passed. (Metadata preprocessor was fit on the
*real* full PAD_UFES20 train CSV, not the 12-row subset — fitting on the
tiny subset produces a smaller one-hot vocabulary and a dimension
mismatch against the warm-started metadata checkpoint, which expects the
full 89-dim encoding.)

**Kaggle notebook generated, not hand-assembled:**
`scripts/generate_cross_attention_backbone_fusion_kaggle_notebook.py`
reads the real `src/models/*.py` file contents at generation time and
writes
`notebooks/pad_ufes20_cross_attention_backbone_fusion_kaggle_notebook.md`.
Requires **four** Kaggle "Add Data" sources, one more than Stage 2's
notebook: the usual raw mirror, processed metadata, and
`pad-ufes20-stage1-checkpoints`, plus a **new, not-yet-published**
private dataset holding this machine's
`logs/PAD_UFES20_Expanded/checkpoints/image_{convnext_tiny,densenet121}_seed{0,1,2}_best.pt`
(6 files) — `PAD_UFES20_Expanded` has no Kaggle-slug indirection in
`config.py` the way `PAD_UFES20`/`HAM10000`'s `stage1_checkpoints_dir`
does, so the notebook's Cell 2 copies those 6 files into
`/kaggle/working/logs/PAD_UFES20_Expanded/checkpoints/` itself before
training starts. The generator script's
`EXPANDED_BACKBONE_CHECKPOINTS_SLUG` constant is a `REPLACE_WITH_*`
placeholder (same convention as `config.py`'s
`KAGGLE_STAGE1_CHECKPOINT_SLUGS`) — **must be edited to the real slug
after uploading**, then the notebook regenerated. 19 cells total: folder
verification → setup/copy → 9 `%%writefile` cells (config, dataset,
backbones, fusion_model, cross_attention_fusion_model,
spatial_backbone_embedder, cross_attention_backbone_fusion_model, train,
train_cross_attention_backbone_fusion) → sanity check → full
model/GPU check → 6 training cells (2 backbones × 3 seeds).

---

## Step 4 Scope Gap Found and Resolved — Option B vs. Option A, Phase 8E Added (2026-07-31)

**Gap identified:** the supervisor's literal instruction for Step 4 was
"build ONE final fusion model from the top-2 backbones, then
train/val/test that single model" — a genuine three-way joint
architecture, both image backbones (ConvNeXt-Tiny, DenseNet121) and
metadata fused into one trainable model ("Option A"). What was actually
scoped and implemented (see "Step 4 — Cross-Attention Backbone Fusion
Implemented" above) is "Option B": two independent, full
backbone+metadata cross-attention models (one per backbone), combined
only at prediction time via unweighted softmax-probability averaging —
no joint training, no shared parameters across backbones. Option B was
chosen earlier for lower technical risk (reuses the already-validated
Phase 7 Stage 2 cross-attention mechanism as-is, rather than requiring
new joint-architecture design), but it does not literally satisfy "one
final fusion model."

**Decision (senior-researcher risk/reward assessment, both options
weighed on time, risk, defensibility, and novelty):**

1. **Finish Option B first, uninterrupted.** It is already training on
   Kaggle (partial results in — see live log, epoch 1 of
   `convnext_tiny` seed 0: `val_macroF1=0.1385`, warm-start and sanity
   checks all passed). This becomes a primary, safe, well-documented
   result on its own: two independently-reportable multimodal models
   (ConvNeXt-Tiny+metadata, DenseNet121+metadata cross-attention
   fusion) plus their prediction-time ensemble, compared against the
   existing locked headline (val 0.6209±0.0143, test 0.6977) via paired
   bootstrap on the frozen test split. Final results to be logged in a
   new `docs/Phase8B_Backbone_Fusion_Results.md` (or similarly named
   file) once all 6 runs + ensemble + bootstrap are complete.
2. **Then start Option A as a new, separately-scoped phase: "Phase
   8E — Single Joint Three-Way Fusion."** This is what directly
   fulfills the supervisor's literal instruction. Framed explicitly as
   an exploratory, higher-risk/higher-reward addition — NOT a
   replacement for Option B's result, and NOT the thesis's
   make-or-break outcome. `src/models/backbone_fusion_model.py` (the
   existing WIP concatenation-based dual-backbone model, image-only —
   see "Forward-Reference Note for Step 4" above) is a possible
   starting point to revisit, but the actual joint architecture (how
   metadata interacts with two image modalities simultaneously) is to
   be proposed and justified before any code is written for this phase
   — not assumed to be a straightforward extension of the existing
   cross-attention mechanism to a third input.
3. **Presentation:** Option B's and Option A's results will be
   presented together in the thesis as a comparison/ablation.
   Whichever performs better becomes the reported "improved" result —
   but only if it beats 0.6977 test *and* that improvement is
   statistically significant per paired bootstrap. If neither does, the
   original locked cross-attention (EfficientNet-B0, test 0.6977)
   remains the headline result, and both Option A and Option B become
   documented exploratory findings rather than the thesis's primary
   claim.

**Status:** Option B (already in progress) continues uninterrupted —
no action taken on Option A/Phase 8E yet beyond this scoping and
documentation. Phase 8E added to `PROJECT_PLAN.md`'s future-phases list
as a new, clearly-scoped, not-yet-started phase.

---

## Future Ablation — Isolating Dataset-Expansion Effect from Backbone-Change Effect (2026-07-31)

**Observation:** EfficientNet-B0 on the expanded dataset (image-only)
scored 0.5882±0.0059 vs. 0.5703±0.0130 on the original dataset — a
modest +0.018 gain from dataset expansion alone (architecture held
constant). This suggests most of ConvNeXt-Tiny's larger improvement
(0.6542 val on seed 0, a +0.033 gain over the original headline's
0.6209) is likely attributable to the backbone architecture change, not
the expanded dataset alone.

**Caveat — not yet rigorously isolated:** this is an inference from
comparing 2 different image-only ablation numbers, not a controlled
same-architecture-different-dataset cross-attention comparison. Treat
as a hypothesis, not a confirmed finding.

**Proposed follow-up (if time allows later):** train the ORIGINAL
cross-attention architecture (EfficientNet-B0 + metadata) on the
EXPANDED dataset — same warm-start pattern as Step 4, just swapping
which backbone checkpoint gets loaded — and compare directly against
both the 0.6209 original and whichever final Option A/B result (see
"Step 4 Scope Gap Found and Resolved" above) is landed on. This would
cleanly separate "dataset helped" from "backbone helped" as two
distinct, individually-cited findings for the thesis Discussion
chapter.

**Status:** not blocking anything currently in progress — flagged here
so this research idea isn't lost. Not scheduled as a phase; revisit
after Option B (and, if pursued, Phase 8E) are complete.

---

## PAD-UFES-20 Test-Split Guard Reopened for Step 4 — Second Sanctioned Read Authorized (2026-08-01)

**Context:** Step 4 (Option B) training completed — all 6
`cross_attention_backbone` checkpoints (ConvNeXt-Tiny + DenseNet121, 3
seeds each) verified on disk against their summary JSONs (val macro-F1
0.6542/0.6731/0.6856 and 0.6363/0.6714/0.6601 respectively, matching
exactly). Val-split dual-backbone ensemble evaluated
(`--branch cross_attention_backbone_ensemble`): seed0 0.6661, seed1
0.6952, seed2 0.6956 (mean 0.6856). Per the "Step 4 Scope Gap
Found and Resolved" entry above, the approved plan requires comparing
these against the existing locked headline (val 0.6209±0.0143, test
0.6977) via paired bootstrap **on the frozen test split** — which
requires a second read of PAD-UFES-20's `metadata_test.csv`.

**Conflict found:** `src/evaluation/test_split_guard.py`'s
dataset-scoped marker
(`data/processed/PAD_UFES20/TEST_SPLIT_CONSUMED.json`) already recorded
PAD-UFES-20's test split as consumed on 2026-07-25 by
`evaluate_fairness.py` (the run producing the locked 0.6977). The guard
blocks *any* further script from reading that file, by design (see
"Test-Split Single-Use Safeguard Added", 2026-07-25) — "deleting it to
re-run would itself have to be a logged, user-approved decision to
reopen decision 4 for that dataset, not a code change." This entry is
that logged decision.

**Decision (explicit, user-approved 2026-08-01):** reopen the guard for
exactly one additional, sanctioned read — the Step 4 backbone-fusion
evaluation set (`cross_attention_backbone` convnext_tiny x3 seeds,
densenet121 x3 seeds, `cross_attention_backbone_ensemble` x3 seeds; 9
evaluation runs total) — immediately followed by paired bootstrap
comparison against the already-existing, unchanged
`predictions_cross_attention_seed{0,1,2}_test.csv` files from the
original 2026-07-25 run (those files are read-only reused, not
regenerated). This does **not** reopen decision 4 generally or permit
arbitrary future re-reads: the marker is updated (not deleted) to
record both consumption events, so a third, unsanctioned read is still
blocked by the same mechanism afterward.

**Rationale:** this is the specific comparison the Step 4 plan already
called for (2026-07-31, before this gap was noticed), not a new or
expanded use of the test split, and not a tuning decision — the 6
backbone-fusion checkpoints and the ensemble combination rule were
already fully fixed (via val-split selection only) before this test
read happens, satisfying the same "no test-set influence on model
selection" discipline decision 4 was written to protect.

**Mechanics:** `data/processed/PAD_UFES20/TEST_SPLIT_CONSUMED.json`
updated from a single consumed_by record to a `consumption_events` list
containing both the original 2026-07-25 entry and this session's new
entry (dataset/test CSV file itself never modified, never re-split —
only the marker's bookkeeping changes). `test_split_guard.py`'s error
message updated to read the latest event from either marker schema, so
it still names the correct consumer/date/reference instead of `None`.

---

## Step 4 (Option B) — Final Test Results and Bootstrap Comparison — COMPLETE (2026-08-01)

**Verification (from disk, not trusted from printed numbers alone):**
all 6 `cross_attention_backbone` checkpoints and summary JSONs
(`logs/PAD_UFES20/checkpoints/cross_attention_backbone_{backbone}_seed{N}_best.pt`,
`logs/PAD_UFES20/train_cross_attention_backbone_{backbone}_seed{N}_summary.json`)
confirmed present and exactly matching the reported numbers:

| backbone | seed0 | seed1 | seed2 |
|---|---|---|---|
| convnext_tiny (val) | 0.654151 | 0.673132 | 0.685567 |
| densenet121 (val) | 0.636304 | 0.671400 | 0.660099 |

Checkpoint sizes sane and internally consistent (convnext_tiny ~113.7MB
x3, densenet121 ~31.2MB x3, matching each backbone's parameter count).

**Val-split dual-backbone ensemble** (`--branch
cross_attention_backbone_ensemble`, unweighted softmax-probability
averaging, seed-matched): seed0 0.6661, seed1 0.6952, seed2 0.6956
(mean 0.6856).

**Test-split guard reopened** (see entry above) for one additional
sanctioned read, then all 9 final test evaluations run with
`--confirm-final`:

| variant | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| cross_attention_backbone (convnext_tiny) | 0.6897 | 0.6994 | 0.7425 | **0.7105** |
| cross_attention_backbone (densenet121) | 0.6630 | 0.6946 | 0.6980 | **0.6852** |
| cross_attention_backbone_ensemble | 0.7081 | 0.7338 | 0.7542 | **0.7321** |
| cross_attention (original, locked 2026-07-25, re-used unchanged) | 0.6862 | 0.6721 | 0.7349 | 0.6977 |

Per-row test predictions written to
`reports/PAD_UFES20/cross_attention_backbone/predictions_*_test.csv`
(new `evaluate.py` capability, additive only - `_predictions_df` built
in `evaluate()`/`evaluate_dual_backbone_ensemble()` and written to CSV
in `main()` only when `--split test`, mirroring
`evaluate_fairness.py`'s existing predictions-CSV convention so row
order/schema pairs directly against
`reports/PAD_UFES20/fairness/predictions_cross_attention_seed{N}_test.csv`).

**Paired bootstrap significance** (new
`src/evaluation/bootstrap_significance_backbone_fusion.py`, same method
as `bootstrap_significance.py`: 1,000 resamples, row-level paired,
seed-averaged, all 6 classes since this is PAD-UFES-20's own test split,
not a cross-dataset shared-class comparison). Recomputed anchor
(0.697746) matched the locked 0.6977 exactly, confirming correct
row-pairing:

| comparison | observed diff | 95% CI | p (two-sided) | significant (α=0.05) | significant (Bonferroni α=0.0167) |
|---|---|---|---|---|---|
| convnext_tiny vs. original | +0.0128 | [-0.0251, +0.0506] | 0.470 | No | No |
| densenet121 vs. original | -0.0126 | [-0.0689, +0.0509] | 0.656 | No | No |
| **ensemble vs. original** | **+0.0343** | **[-0.0020, +0.0771]** | **0.062** | **No** | No |

**Conclusion:** the dual-backbone ensemble (0.7321 test macro-F1) is the
numerically best result to date, +0.0343 over the locked headline, and
closest to significance of the three comparisons — but its 95% CI still
crosses zero (barely: -0.0020) and p=0.062 exceeds even the
uncorrected alpha=0.05, so **it is not statistically significant** at
n=354 test rows. Per the "Step 4 Scope Gap Found and Resolved" decision
(2026-07-31), since no Option B variant beats 0.6977 significantly, the
**original locked cross-attention (EfficientNet-B0, test 0.6977)
remains the thesis headline result**; Option B (both single-backbone
models and their ensemble) is documented here as a positive but
not-yet-significant exploratory finding, to be presented alongside
Option A (Phase 8E, not yet started) once that is also complete.

**Status:** Step 4 Option B fully complete (training, val selection,
one-time test evaluation, bootstrap significance). Next: Phase 8E
(Option A, single joint three-way fusion) remains not-yet-started, per
existing scope.

---

**Low-priority note (2026-08-01):** commit `070a034` deleted several
`.zip` files under `data/processed/` (Kaggle-upload staging copies,
redundant duplicates of already-present `data/raw/`/`data/processed/`
data) — intentional, done directly by the user to reduce repo/push
size, not a bug or accidental loss. Noted here so it isn't mistaken for
a mystery by a future session.

---

## Pre-Registered Prediction — Dataset-Expansion-Only Ablation (Original Architecture) (2026-08-01)

**Ablation being run (approved, not yet trained as of this entry):**
train the ORIGINAL Phase 7 Stage 2 architecture (`CrossAttentionFusionModel`,
EfficientNet-B0 image embedder) on PAD-UFES-20, warm-starting the image
embedder from Step 3's **expanded**-dataset EfficientNet-B0 checkpoint
(`logs/PAD_UFES20_Expanded/checkpoints/image_seed{N}_best.pt`) instead
of the original (unexpanded) Stage 1 image checkpoint, metadata embedder
warm-started as usual from PAD-UFES-20's own Stage 1 checkpoint. Fusion
fine-tuning itself still runs on PAD-UFES-20's original, unexpanded
train/val split (expanded rows have no metadata - same pattern as Step
4). Purpose: isolate the dataset-expansion effect from the
backbone-architecture-change effect already observed in Step 4, holding
architecture constant at EfficientNet-B0. New files:
`train_cross_attention_efficientnet_expanded.py`,
`cross_attention_efficientnet_expanded_seed{0,1,2}_best.pt` - verified
non-colliding with the locked `cross_attention_seed{0,1,2}_best.pt`
(0.6977) or Step 4's `cross_attention_backbone_*` checkpoints before any
training started (see naming-collision safety check, this session).
Test split **not** touched (val-only; also still blocked twice-over by
`test_split_guard.py` even if mistakenly attempted).

**Prediction, logged before seeing any result from this run:**

- **Expected val macro-F1 range: 0.61-0.66**, likely landing near or
  modestly above the original locked headline's val macro-F1
  (0.6209±0.0143), and clearly below Step 4's backbone-driven fusion
  results (ConvNeXt-Tiny 0.6542-0.6856, DenseNet121 0.6363-0.6714,
  ensemble test 0.7321).
- **Reasoning:** Step 3's image-only ablation isolated the
  dataset-expansion effect at only **+0.018** for EfficientNet-B0 alone
  (0.5703->0.5882, architecture held constant, no fusion/fine-tuning
  involved) - much smaller than the architecture-change effect inferred
  for ConvNeXt-Tiny (~+0.033 over the original headline, per "Future
  Ablation" entry above). Since this run's fusion fine-tuning loop
  re-trains on the *original* unexpanded train set (identical to how
  Step 4's own warm-started backbones were fine-tuned), some but not
  necessarily all of the expanded initialization's benefit should
  survive fine-tuning - hence "near or modestly above" 0.6209 rather
  than "no different from" it.
  **Working hypothesis: architecture (backbone choice) is the dominant
  driver of Step 4's gains, not dataset expansion alone** - this run is
  the direct test of that hypothesis, not a formality, and could
  disconfirm it if the result lands materially higher than 0.66.

**Status:** prediction logged 2026-08-01, before training starts.
Actual result to be logged in a follow-up entry once the 3 Kaggle seed
runs complete, compared explicitly against this range.

---

## Phase 8E (Option A) — Genuine Joint Three-Way Fusion — Plan Approved, Prediction Logged (2026-08-02)

**Scientific rationale (updated given new evidence):** the
dataset-expansion-only ablation (see prior entry) confirmed
architecture change is the dominant driver of Step 4's improvement
(0.6186 dataset-alone vs. 0.6710 architecture-change). This makes
Option A's real question: does JOINTLY fusing two good backbones (not
late-ensembling two separately-trained models, which is what Step 4
Option B's `cross_attention_backbone_ensemble` already did) capture
complementary signal beyond Step 4's ensemble (0.7321 test)? This is a
genuine hypothesis test, not a formality, per the "Step 4 Scope Gap
Found and Resolved" decision (2026-07-31) that scoped Phase 8E
separately from Option B.

**Naming collision check (repeated per established safety drill):**
verified by reading `spatial_backbone_embedder.py`,
`cross_attention_backbone_fusion_model.py`,
`train_cross_attention_backbone_fusion.py`, `src/evaluation/evaluate.py`,
and directory listings of `logs/PAD_UFES20/checkpoints/`,
`logs/PAD_UFES20/*.csv`, `reports/PAD_UFES20/**` directly (not assumed).
New prefix chosen: **`cross_attention_joint_convnext_densenet_seed{N}`**
- confirmed non-colliding with `cross_attention_seed{N}` (locked
headline), `cross_attention_backbone_convnext_tiny_seed{N}` /
`cross_attention_backbone_densenet121_seed{N}` (Step 4 Option B),
`cross_attention_backbone_ensemble_convnext_tiny_densenet121_seed{N}`
(Step 4 Option B ensemble, eval-only), and
`cross_attention_efficientnet_expanded_seed{N}` (dataset-effect
ablation). New files: `cross_attention_joint_fusion_model.py`,
`train_cross_attention_joint_fusion.py`,
`cross_attention_joint_convnext_densenet_seed{0,1,2}_best.pt`, plus a
new `--branch cross_attention_joint` case in `src/evaluation/evaluate.py`.

**Design (dimension-mismatch explicitly solved):** ConvNeXt-Tiny's
spatial tokens are `[B,49,768]`, DenseNet121's are `[B,49,1024]` -
verified from `spatial_backbone_embedder.py`, both reused unchanged.
Each backbone's tokens are projected by its own `kv_proj` Linear
(768->256 and 1024->256 respectively, `d_model=256`) into the *shared*
space **before** concatenation along the token axis, producing a
combined 98-token (49+49) Key/Value sequence; metadata's `query_proj`
(64->256) supplies the single Query token. This feeds the same
`nn.MultiheadAttention` module and downstream head structure already
validated in `CrossAttentionBackboneFusionModel` (Step 4) - the only
new mechanism is the per-backbone projection + token-concatenation
step, no new attention variant.

**Reuse:** `SpatialImageEmbedder`/`SpatialBackboneEmbedder` and
`MetadataEmbedder` reused unchanged. Warm-start: image embedders from
Step 3's `image_convnext_tiny_seed{N}_best.pt` /
`image_densenet121_seed{N}_best.pt`
(`logs/PAD_UFES20_Expanded/checkpoints/`), metadata embedder from
PAD-UFES-20's own Phase 7 Stage 1 `metadata_seed{N}_best.pt` - identical
warm-start pattern to Step 4.

**Test-split discipline (explicit, before training):** this round
evaluates on **validation only**. The locked test split has already
been consumed twice (Stage 1 final result; Step 4 Option B's sanctioned
second read). A third consumption is **not automatic** - it will only
be considered as a separate, explicitly-approved decision if validation
clearly and meaningfully exceeds Step 4's best single-backbone val
result (0.6710, ConvNeXt-Tiny), not a marginal beat.

**Prediction, logged before seeing any result from this run:**

- **Expected val macro-F1 range: 0.66-0.71.**
- **Reasoning:** Step 4's ensemble (0.7321 test) gains from decorrelated
  errors between two *independently*-trained models, softmax-averaged
  at prediction time. A joint model instead compresses both backbones'
  signal through one shared 256-dim attention bottleneck and one
  Query - in principle it can learn backbone-specific weighting a fixed
  average can't, but joint training typically captures less
  complementary diversity than late averaging of independently-optimized
  models. Genuinely uncertain whether this clears the single-backbone
  0.6710 val bar at all, let alone Step 4's ensemble - this is a real
  experiment, not a foregone conclusion.

**Status:** plan and prediction approved 2026-08-02, before any code is
written. Next: implement `cross_attention_joint_fusion_model.py` and
`train_cross_attention_joint_fusion.py`, then a local smoke test (real
small subset, verify shapes/warm-start/forward-pass) before generating
the Kaggle notebook.

**Implementation + local smoke test (2026-08-02):**
`cross_attention_joint_fusion_model.py`, `train_cross_attention_joint_fusion.py`,
and evaluate.py's new `--branch cross_attention_joint` case (val-only,
refuses `--split test` unconditionally) written per the approved design
above. End-to-end smoke test (not just unit-level, per this project's
established drill) run on CPU against real data: a real `FusionDataset`
batch pulled from PAD-UFES-20's actual `train_csv` via `DataLoader`
(`images=(8,3,224,224)`, `metadata=(8,89)`, `labels=(8,)`); all three
seed-0 warm-start checkpoints (`image_convnext_tiny_seed0_best.pt`,
`image_densenet121_seed0_best.pt` from `PAD_UFES20_Expanded`,
`metadata_seed0_best.pt` from PAD-UFES-20's own Stage 1) confirmed to
exist and load without error; forward pass produced the expected
`(8, 6)` output shape; a real `loss.backward()` confirmed non-zero
gradient norms reached `kv_proj_a`, `kv_proj_b`, `query_proj`, and the
final head layer specifically (not just that `.backward()` didn't
throw) - confirms both backbones' projections and the metadata query
path are actually wired into the trainable graph. `optimizer.step()`
completed without error. Smoke test script was scratch-only, not
committed. Next: generate the Kaggle notebook and run the 3 seeds.

---

## Phase 8E (Option A) — Joint Fusion Val Results — Condition for Test Read Not Met (2026-08-03)

**Result (Kaggle run, 3 seeds, `cross_attention_joint_convnext_densenet_seed{0,1,2}`,
val split only):**

| seed | best val macro-F1 |
|---|---|
| 0 | 0.6575 |
| 1 | 0.6927 |
| 2 | 0.6660 |
| **mean** | **0.6721** |

Falls inside the pre-registered prediction range (0.66-0.71, logged
2026-08-02).

**Checked against the pre-registered test-split-read condition** (from
the same 2026-08-02 entry: "a third consumption is not automatic - it
will only be considered ... if validation clearly and meaningfully
exceeds Step 4's best single-backbone val result (0.6710,
ConvNeXt-Tiny), not a marginal beat"): the joint model's mean
(0.6721) exceeds 0.6710 by only **+0.0011** - not a clear or
meaningful margin - and remains **below** Step 4's dual-backbone
ensemble val mean (0.6856). Seed 1 alone (0.6927) does clearly clear
0.6710, but the plan's comparison point is the 3-seed mean, not a
single favorable seed.

**Decision (2026-08-03):** the pre-registered condition for a third
test-split read is **not met**. No test-split evaluation is performed
for Phase 8E. The locked headline (original cross-attention,
EfficientNet-B0, test 0.6977) and Step 4 Option B's ensemble (test
0.7321, not statistically significant vs. the headline) remain the
project's reportable results; Option A is documented as a completed,
val-only exploratory result that did not clear its own pre-registered
bar for further evaluation.

**Interpretation:** consistent with the 2026-08-02 working hypothesis -
joint training through a single shared attention bottleneck captures
less complementary signal than late softmax-averaging of two
independently-trained backbones (Step 4's ensemble). Phase 8E is
complete; no further training or evaluation planned for this variant
unless a future decision explicitly reopens it.

---

## Limitation, Honestly Documented — Val-to-Test Gap Pattern and the Un-Chased Marginal Miss (2026-08-03)

**Observation (not a mistake, not a reason to reopen the test split):**
every variant in this project that has gone through both a val and a
locked test-split evaluation has scored **higher on test than on val**:

| variant | val | test | gap |
|---|---|---|---|
| Cross-attention (headline, §3.4) | 0.6209 | 0.6977 | **+0.0768** |
| Step 4 dual-backbone ensemble | 0.6856 | 0.7321 | **+0.0465** |

Phase 8E's joint-fusion val mean (0.6721) missed the pre-registered
0.6710 decision-rule bar by only **+0.0011**. Under the pattern above,
if Phase 8E were carried through to a test-split evaluation, its test
score would plausibly also land higher than its val score - possibly
enough to beat the 0.6977 headline.

**We are explicitly NOT reopening the test split for this.** Deciding
to re-test specifically because of a post-hoc "it might score higher"
observation would defeat the entire purpose of the pre-registered
0.6710 bar (set 2026-08-02, before any Phase 8E result existed) -
which exists precisely to prevent rationalized re-testing driven by a
result that came in close but didn't clear the bar. Acting on this
observation now would be exactly the kind of decision the
pre-registration was designed to rule out, regardless of how
plausible the reasoning sounds in isolation.

**Documented honestly as a limitation/future-work point, not chased
this thesis cycle:** a disciplined choice to preserve test-set
integrity, at the possible cost of a marginal improvement we chose not
to chase. A legitimate future-work path (not a re-read of this
project's already-twice-consumed PAD-UFES-20 test split) would be a
fresh, independently held-out evaluation set, pre-registered before
looking at it, specifically to test whether Phase 8E's val-test gap
follows the same pattern observed here.

**Caveat on the pattern itself:** n=2 prior val/test pairs is a small
basis for a "pattern" - both existing gaps are consistent in direction
and roughly similar in magnitude (+0.077, +0.047), but this is
observational, not a statistically established property of this
project's split/training pipeline. Framed as an honest observation
worth disclosing, not a proven effect.

---

## Phase 8E — Local Verification of the 3 Summary JSONs (2026-08-03)

User placed the 3 `cross_attention_joint` summary JSONs at
`logs/PAD_UFES20/`. Confirmed present and read directly:
`train_cross_attention_joint_convnext_densenet_seed{0,1,2}_summary.json`
- `best_val_macro_f1` = 0.657466 (seed0) / 0.692674 (seed1) / 0.665969
(seed2), mean **0.672036** (rounds to 0.6721) - matches the previously
reported numbers exactly, same verification standard used for the
dataset-expansion ablation's 0.6186 (§3.7 of
`THESIS_OWNERSHIP_MASTER.md`). `THESIS_OWNERSHIP_MASTER.md` updated to
remove the "not yet locally verified" caveat on Phase 8E throughout
(intro, §3.8, §8.5). No change to the decision outcome - the mean
still misses the 0.6710 bar's "clear and meaningful" threshold by only
+0.0011, so the test split remains unread for this variant.

---
