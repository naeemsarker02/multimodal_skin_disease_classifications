# PROJECT PLAN (Canonical — 2026-07-08)

This is the single source of truth for this project going forward. All
earlier planning drafts from outside this project are superseded by this
file and should not be added.

Project: Multimodal Skin Lesion Classification Using Image and Clinical
Metadata. Goal: Master's thesis + conference/journal publication +
scholarship portfolio.

---

## Table of Contents

1. [Confirmed design decisions (do not re-litigate)](#confirmed-design-decisions)
2. [Canonical folder structure](#canonical-folder-structure)
3. [Current phase — finish dataset readiness](#current-phase)
4. [Future phases (reference only)](#future-phases)
5. [Ground rules (every phase, no exceptions)](#ground-rules)

---

<a id="confirmed-design-decisions"></a>
## Confirmed design decisions (do not re-litigate)

- **Datasets:** PAD-UFES-20 (primary, multimodal), HAM10000 (benchmark),
  ISIC Archive 1 & 2 (external validation, with leakage exclusions applied).
- **Multi-class** taxonomy (not binary) — deliberate differentiation vs.
  reviewed literature.
- **Patient-wise / lesion-wise** splitting only, never image-wise.
- **No image copying** — `processed/` holds CSVs only; `image_path` points
  back to `data/raw/.../`. This is correct and intentional (disk savings,
  `data/raw/` stays untouched, standard practice for this kind of dataset).
  Do not build an image-copy pipeline. (Resizing/caching for training speed
  may be revisited later as a pure performance optimization, not now.)
- **Cross-dataset leakage (HAM10000 ↔ ISIC Archive 1/2):** resolved via
  filtering (`external_validation_exclusions.csv` per ISIC archive), not
  global re-split. Must be applied whenever an ISIC archive is used as
  external validation against a HAM10000-trained model.
- **Label-leakage features — must NEVER be used as model input, any
  dataset:**
  - `biopsed` (PAD-UFES-20) — 100% of malignant cases have `biopsed=True`,
    zero exceptions (Phi = 0.80).
  - `diagnosis_confirm_type` (HAM10000, ISIC Archive 2) — same
    near-deterministic pattern.
  - `diagnostic_code` (PAD-UFES-20), `class_label` (ISIC Archive 1),
    `diagnosis_1`/`diagnosis_2`/`diagnosis_3` (ISIC Archive 2) — these are
    the raw fields `disease_label` was directly derived from (1:1 mapping).
    Using them as input = reading the answer key, not learning.
  - All of the above are kept in the CSVs as documentation/audit-trail
    columns only — never as training features.
- **Metrics:** macro-F1 + per-class F1 + confusion matrix, always — never
  plain accuracy alone (imbalance ranges 16:1 to 393:1 across datasets —
  updated 2026-07-09 from a stale "239:1" upper bound; ISIC Archive 1's
  train split has a single seborrheic keratosis image, 393:1 against its
  largest class — see `PROJECT_OWNERSHIP.md` Appendix and
  `Project_Tracking.md`'s Phase 5 summary).
- **Reproducibility:** fixed seeds (3–5 per experiment), results reported
  as mean ± std.

---

<a id="canonical-folder-structure"></a>
## Canonical folder structure

```
Multimodal_skin_disease_classification_system/
├── README.md
├── docs/
│   ├── PROJECT_PLAN.md                # this file — canonical
│   ├── Project_Tracking.md            # living status + decision log
│   ├── Dataset_Strategy.md            # kept, corrected of stale sections
│   ├── AI_Assistant_Instructions.md   # kept if still accurate
│   └── Dataset_Preparation_Final_Report.md  # frozen historical report
├── data/
│   ├── raw/                           # IMMUTABLE
│   └── processed/
│       ├── PAD_UFES20/  HAM10000/  ISIC_Archive_1/  ISIC_Archive_2/
│           # metadata_{train,val,test}.csv, label_mapping.csv,
│           # split_quality_report.csv, dataset_description.md,
│           # feature_whitelist.md (NEW — see Current Phase below)
├── src/
│   ├── data_cleaning/   ├── data_audit/
│   ├── eda/              # not started
│   ├── models/           # not started
│   └── evaluation/       # not started
├── notebooks/  ├── reports/<dataset>/  ├── logs/<dataset>/
└── _archive/                           # superseded files, never hard-deleted
```

---

<a id="current-phase"></a>
## Current phase — finish dataset readiness (do this now, nothing beyond it)

Dataset preparation (audit → cleaning → split → cross-dataset leakage fix)
is functionally complete for all 4 datasets. Before this phase is declared
closed, these remain:

1. **Record the label-leakage decision** (above) formally in
   `Project_Tracking.md` — same treatment as the `biopsed` decision.
2. **Produce a `feature_whitelist.md` per dataset** in
   `data/processed/<Dataset>/` — the explicit, final list of columns
   allowed as model input (everything else excluded, with a one-line reason
   per exclusion: identifier, path, label-source, or leakage feature).
   This becomes the single reference for Phase 6 model-building — no
   feature-selection ambiguity later.
3. **Cleanup pass** on project docs: read `Dataset_Strategy.md`,
   `Project_Cleanup_Review.md`, and any other planning `.md` files still in
   the project; archive (to `_archive/`, never hard-delete) anything stale
   or conflicting with this PROJECT_PLAN.md; merge any still-valid content
   into `Dataset_Strategy.md`/`Project_Tracking.md`. Log what was archived.
4. **Full verification printout** (already partially run): for each
   dataset — column list, 3 sample rows, image_path existence + dimensions
   check, split row counts, class distribution, and the final allowed
   feature list from step 2 — all printed in full, nothing truncated.

**Do not start EDA, model code, or any Phase 5+ work until all 4 items
above are done and I've reviewed them.**

---

<a id="future-phases"></a>
## Future phases (reference only — do not start yet)

5. EDA — class/demographic distributions, Fitzpatrick distribution,
   missing-value visualization, sample image grids.
6. Baseline models (PAD-UFES-20) — image-only, metadata-only, using each
   dataset's `feature_whitelist.md`, class-weighted loss.
7. Multimodal fusion — late fusion, then cross-attention.
8. Experiments — PAD-UFES-20↔HAM10000 generalization (headline result),
   HAM10000→ISIC external validation (with exclusions applied), Fitzpatrick
   fairness analysis, bootstrap significance testing.
9. Thesis writing support.
10. arXiv preprint → workshop/journal submission.

---

<a id="ground-rules"></a>
## Ground rules (every phase, no exceptions)

- `data/raw/` never written to.
- Every file change explained in plain language, not just "done."
- Any ambiguous decision gets flagged to me, never silently resolved.
- Delete = archive to `_archive/`, never hard-delete.
- Each phase ends with a status entry in `Project_Tracking.md` before the
  next phase starts.
