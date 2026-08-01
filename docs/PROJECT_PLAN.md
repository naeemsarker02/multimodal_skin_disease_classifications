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
- **Phase 8C dataset expansion — locked test split is never touched**
  (added 2026-07-29): PAD-UFES-20's existing, already-spent test split
  (`data/processed/PAD_UFES20/metadata_test.csv`) stays byte-for-byte
  identical through any dataset-expansion work — new external sources
  are added to TRAIN (and optionally VAL) only, never TEST. This is what
  keeps a valid paired-bootstrap comparison possible against the locked
  0.6977 result. See `Project_Tracking.md`, "Step 2 Binding Rule —
  Locked Test Split Frozen" (2026-07-29).

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
   fairness analysis, bootstrap significance testing. **Closed 2026-07-27**
   — see `Project_Tracking.md`.
8B. Backbone comparison (added 2026-07-29) — 5 pretrained image
   backbones (EfficientNet-B0 baseline + MobileNetV3-Large, DenseNet121,
   ResNet50, ConvNeXt-Tiny), image-only, 3 seeds each, ranked by val
   macro-F1, each backbone's own pretrained normalization verified (not
   assumed identical to EfficientNet-B0's). Full scope, decision log, and
   step-by-step approval gates: `Project_Tracking.md`, "Phase 8B+8C —
   Master Plan Adopted" (2026-07-29) onward. Results:
   `docs/Phase8B_Backbone_Comparison_Results.md` (not yet written).
8C. Dataset expansion (added 2026-07-29) — vetted external dermatology
   sources merged into a new `PAD-UFES-20-Expanded` variant (original
   `data/processed/PAD_UFES20/` never overwritten), targeting Melanoma/SCC
   under-representation; top-2 backbones from 8B + metadata fused (design
   TBD, see Step 4 of the plan — three-way fusion vs. two-model ensemble,
   not yet decided); trained/evaluated on a new patient-wise split; bootstrap
   significance vs. the locked 0.6977 test result. Full scope:
   `Project_Tracking.md`, same entry as 8B. Results:
   `docs/Phase8C_Expanded_Dataset_Results.md` (not yet written).
   **Neither 8B nor 8C supersedes Phase 8** — the existing locked
   cross-attention result (val 0.6209±0.0143, test 0.6977) remains the
   presentable thesis result unless 8B/8C produce a significantly better one.
8D. Lesion segmentation/cropping (placeholder, added 2026-07-29,
   **deferred — no implementation started**) — reasoned through and
   explicitly sequenced strictly *after* Phase 8B/8C completes, not
   bundled in now (isolation-risk precedent: the 2026-07-23 "Improved
   Cross-Attention Variant" negative result, where stacked changes made
   a regression unattributable). Plan for when it starts: (1) a
   feasibility spot-check of an ISIC-pretrained (dermoscopic-trained)
   segmentation model against PAD-UFES-20's macroscopic images, since
   that's an unverified domain-shift application, not an assumed fit;
   (2) if pursued, any test-set comparison must use option (i) —
   re-evaluate the existing frozen cross-attention checkpoint on cropped
   versions of the same locked test images — never a fresh, unpaired
   claim against 0.6977. Full reasoning: `Project_Tracking.md`, "Lesion
   Segmentation/Cropping — Reasoned Through" (2026-07-29).
8E. Single Joint Three-Way Fusion (added 2026-07-31, **not yet
   started — Step 4/Option B below must finish first**) — a genuine
   single trainable model jointly fusing both top-2 Phase 8B backbones
   (ConvNeXt-Tiny, DenseNet121) and metadata ("Option A"), directly
   fulfilling the supervisor's literal Step 4 instruction ("build ONE
   final fusion model... then train/val/test that single model"), as
   distinct from Step 4's actual implementation ("Option B": two
   independent per-backbone cross-attention models combined only via
   prediction-time ensemble averaging — lower-risk, reuses the existing
   Phase 7 Stage 2 mechanism unchanged). Framed as exploratory,
   higher-risk/higher-reward — not a replacement for Option B's result.
   `src/models/backbone_fusion_model.py` is a possible starting point
   to revisit, but the actual three-way joint architecture (how
   metadata interacts with two image modalities simultaneously) must be
   proposed and justified before any code is written, not assumed.
   Option A and Option B results will be presented together as a
   comparison/ablation; whichever beats the locked 0.6977 test result
   (with bootstrap significance) becomes the reported "improved"
   result, otherwise both remain documented exploratory findings and
   0.6977 stays the headline. Full scope and reasoning:
   `Project_Tracking.md`, "Step 4 Scope Gap Found and Resolved — Option
   B vs. Option A, Phase 8E Added" (2026-07-31).
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
