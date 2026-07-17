# PROJECT OWNERSHIP DOCUMENT — Dataset Preparation & EDA

**Scope:** Everything from project start through "dataset ready" (Phase 1
Planning → Phase 4 Dataset Preparation → Phase 5 EDA). Phase 6 model code
(`src/models/`, `src/evaluation/`) is deliberately **out of scope** here.

**Purpose:** A single, self-contained reference so the project owner can
fully understand and personally defend this work in meetings and the viva,
assuming no knowledge of the project and no AI-tool context.

**Accuracy note:** Every number, filename, and behaviour below was re-read
directly from the actual files on disk on 2026-07-09, not from memory. Where
two documents in the repo phrase a number differently, that is flagged
explicitly in-line rather than smoothed over. Basic Python is assumed; no
project-specific knowledge is assumed.

---

<a id="quick-summary"></a>
## Quick Summary

*Orientation aid, added during the docs readability pass — every detailed
section below is unchanged and remains the source of truth.*

- **Four datasets, all "dataset-ready":** PAD-UFES-20 (2,298 images, 6
  classes, 21 allowed features), HAM10000 (10,015 images, 7 classes, 3
  allowed features), ISIC Archive 1 (2,047 images, 9 classes, 0 allowed
  features — image-only), ISIC Archive 2 (25,076 images, 9 classes, 7
  allowed features, 4 active baseline).
- **Only PAD-UFES-20 is fully independent** — HAM10000 and both ISIC
  archives physically share large fractions of the same images (up to
  98.6% overlap), fixed via per-archive `external_validation_exclusions.csv`
  files rather than a global re-split.
- **Several near-perfect leakage/shortcut features were found and excluded**
  from every dataset's `feature_whitelist.md` — e.g. PAD-UFES-20's
  `biopsed` (phi = 0.80) and ISIC Archive 2's `melanocytic` (a perfect
  deterministic split of the label).
- **All splits are patient-/lesion-/group-wise, never image-wise**, and each
  split's leakage-freedom is asserted (not just claimed) in code.
- **Phase 6 pre-conditions are resolved:** 224×224 aspect-preserving
  resize-and-pad at load time, and ISIC Archive 2's sparse metadata fields
  excluded as source-institution leak risk.
- **Every number in this document was independently re-derived from the
  actual processed CSVs** on 2026-07-09 (see the Appendix) — not copied
  from an earlier report or from memory.
- **Section 7 gives copy-pasteable commands** to reproduce the entire
  audit → cleaning → EDA pipeline from a fresh checkout.

---

## Table of Contents

1. [Full current folder / file tree](#section-1)
2. [File-by-file walkthrough](#section-2)
   - 2.1 [`docs/`](#section-2-1)
   - 2.2 [`src/data_audit/`](#section-2-2)
   - 2.3 [`src/data_cleaning/`](#section-2-3)
   - 2.4 [`src/eda/`](#section-2-4)
   - 2.5 [`data/processed/<Dataset>/` files](#section-2-5)
3. [Chronological narrative](#section-3)
4. [Current dataset state (summary table)](#section-4)
5. [What a manual (non-AI) researcher would have had to do](#section-5)
6. [Open items / not yet done](#section-6)
7. [Appendix — Independent re-verification](#appendix)
8. [How to reproduce this pipeline from scratch](#section-7)
   - 7.1 [Environment setup](#section-7-1)
   - 7.2 [Run the audits](#section-7-2)
   - 7.3 [Run the cleaning steps](#section-7-3)
   - 7.4 [Run the EDA](#section-7-4)
   - 7.5 [End-to-end sanity check](#section-7-5)

---

<a id="section-1"></a>
## Section 1 — Full current folder / file tree

Raw image files are omitted (there are ~39,000 of them); `data/raw/` is
shown only down to its dataset-level structure. `.venv/`, `__pycache__/`,
and `.git/` are omitted as environment/tooling noise.

```
Multimodal_Skin_Disease_Research/
├── README.md                       # project front page
├── requirements.txt                # pinned Python dependencies
│
├── data/
│   ├── raw/                        # IMMUTABLE — never written to by any script
│   │   ├── PAD_UFES20/             #   imgs_part_1..3/ + metadata.csv
│   │   ├── HAM10000/               #   HAM10000_images_part_1..2/ + HAM10000_metadata.csv + hmnist_*.csv (unused)
│   │   ├── ISIC_Archive_1/         #   Train/<class>/*.jpg + Test/<class>/*.jpg (no metadata.csv)
│   │   └── ISIC_Archive_2/         #   images/ + metadata.csv
│   │
│   ├── interim/                    # intermediate cleaning output (not final)
│   │   ├── PAD_UFES20/             #   metadata_standardized.csv, value_validation_report.csv
│   │   ├── HAM10000/               #   metadata_standardized.csv, value_validation_report.csv
│   │   ├── ISIC_Archive_1/         #   metadata_standardized.csv
│   │   └── ISIC_Archive_2/         #   metadata_standardized.csv, value_validation_report.csv
│   │
│   └── processed/                  # FINAL model-ready metadata (CSVs only, no images)
│       ├── PAD_UFES20/             #   8 files (see §2)
│       ├── HAM10000/               #   8 files
│       ├── ISIC_Archive_1/         #   8 files
│       └── ISIC_Archive_2/         #   9 files
│
├── docs/
│   ├── PROJECT_PLAN.md             # canonical plan (source of truth for decisions)
│   ├── Project_Tracking.md         # living status + decision log
│   ├── Dataset_Strategy.md         # dataset roles + processing philosophy
│   ├── AI_Assistant_Instructions.md# original working directive
│   ├── Dataset_Preparation_Final_Report.md  # frozen cross-dataset verification report
│   ├── PROJECT_OWNERSHIP.md        # THIS document
│   └── Multimodal Skin Lesion ... .xlsx     # literature-review working spreadsheet
│
├── src/
│   ├── __init__.py
│   ├── data_audit/                 # Phase 4a — read-only dataset auditing
│   │   ├── config.py
│   │   ├── common/                 #   io_utils.py, logging_utils.py
│   │   ├── run_audit_pad_ufes20.py     (+ ham10000, isic_archive_1, isic_archive_2)
│   │   ├── pad_ufes20/             #   m01..m11 audit modules
│   │   ├── ham10000/               #   m01..m11 audit modules
│   │   ├── isic_archive_1/         #   m01..m04 audit modules
│   │   └── isic_archive_2/         #   m01..m08 audit modules
│   │
│   ├── data_cleaning/              # Phase 4b — cleaning, labelling, splitting
│   │   ├── config.py
│   │   ├── common_label_mapping.py
│   │   ├── cross_dataset_leakage_filter.py
│   │   ├── run_cleaning_pad_ufes20.py  (+ ham10000, isic_archive_1, isic_archive_2)
│   │   ├── pad_ufes20/             #   c01..c06 cleaning steps
│   │   ├── ham10000/               #   c01..c06 cleaning steps
│   │   ├── isic_archive_1/         #   c01..c05 cleaning steps
│   │   └── isic_archive_2/         #   c01..c06 cleaning steps
│   │
│   ├── eda/                        # Phase 5 — exploratory data analysis
│   │   ├── config.py
│   │   ├── common/                 #   image_stats.py, plotting.py
│   │   └── eda_pad_ufes20.py  eda_ham10000.py  eda_isic_archive_1.py
│   │       eda_isic_archive_2.py  eda_cross_dataset.py
│   │
│   ├── models/                     # Phase 6 (OUT OF SCOPE here)
│   └── evaluation/                 # Phase 6 (OUT OF SCOPE here)
│
├── notebooks/                      # 01..05_eda_*.ipynb (thin EDA notebooks)
├── reports/                        # audit report CSVs/figures + reports/eda/
├── logs/                           # per-dataset timestamped run logs
├── papers/                         # reviewed literature (PDFs, docx)
└── _archive/                       # superseded docs (Research_Plan.md, Project_Cleanup_Review.md)
```

---

<a id="section-2"></a>
## Section 2 — File-by-file walkthrough

Every file in `docs/`, `src/data_audit/`, `src/data_cleaning/`, `src/eda/`,
and every file in each `data/processed/<Dataset>/` gets its own entry. The
16 `__init__.py` files across `src/` are all **empty (0 bytes)** — they are
standard Python "this folder is an importable package" markers and carry no
logic; they are not listed individually below.

<a id="section-2-1"></a>
### 2.1 `docs/`

**`docs/PROJECT_PLAN.md`** — The canonical, do-not-re-litigate plan. Records
the confirmed design decisions (four datasets and their roles; multi-class
not binary; patient-/lesion-wise splitting only; no image copying;
label-leakage features that must never be model inputs; macro-F1 as the
headline metric; fixed seeds with mean±std reporting), the canonical folder
structure, and the definition of "current phase." This is the document all
others defer to.

**`docs/Project_Tracking.md`** — The living status log and decision journal.
Opens with a "Session Handoff" block describing exactly where work stopped
and what is decided vs. still open. Contains the phase-by-phase progress
tracker, the full record of the cross-dataset leakage decision (fix b), the
`biopsed` leakage audit, the label-leakage exclusion table for all four
datasets, the documentation-cleanup log, and (most recently) the ISIC
Archive 2 sparse-field crosstab decision and the 224×224 resize decision.
This is the file to read first to know the current state.

**`docs/Dataset_Strategy.md`** — Plain-language description of each dataset's
research role (PAD-UFES-20 primary/multimodal; HAM10000 benchmark; ISIC
Archives 1 & 2 external validation) and the raw→audit→cleaning→standardization
→processed philosophy. Was corrected on 2026-07-08 to describe the two ISIC
archives separately and to use the real output filenames.

**`docs/AI_Assistant_Instructions.md`** — The original project directive
that framed the work: the researcher role, the pipeline phases (audit →
standardization → image processing → label standardization → splitting →
final structure), and the hard rules ("never modify raw data," "never invent
missing metadata," "never randomly split patient images," "don't optimise for
accuracy alone"). Useful as the statement of intent behind the pipeline.

**`docs/Dataset_Preparation_Final_Report.md`** — A frozen, independent
verification report (dated 2026-07-08) written after all four datasets were
individually prepared. It re-derived every dataset's numbers from scratch and
confirmed each is internally consistent and leakage-free on its own split —
**except** for its headline finding (§6): HAM10000, ISIC Archive 1, and ISIC
Archive 2 physically share 40–99% of the same images, so using the ISIC
archives as "external validation" against a HAM10000-trained model is invalid
for the overlapping images. This report is what triggered the cross-dataset
leakage fix.

**`docs/PROJECT_OWNERSHIP.md`** — This document.

**`docs/Multimodal Skin Lesion Classification Using Image and Clinical
Metadata.xlsx`** — A two-sheet literature-review working spreadsheet, opened
and read for this document. Sheet **"Dataset Comparison Table"** has columns
Dataset, Year, Total Images/Rows, Classes, Diseases, Columns/metadata, Source —
comparing the candidate datasets (HAM10000: 2018, 10,015 images, 7 classes;
PAD-UFES-20: 2020, 2,298 images / 1,641 lesions / 1,373 patients, 6 classes;
and others). Sheet **"Paper summary"** has columns Year, Paper Title, dataset
link, Task/Problem, Model/Methodology summary, Accuracy/F1 score, Limitations —
one row per reviewed paper, and it is where the "binary-only / limited metadata"
gaps that motivate this thesis were first written down. It is a working
reference artifact, not part of the data pipeline; no script reads it.

<a id="section-2-2"></a>
### 2.2 `src/data_audit/` — the read-only auditing pipeline

**`config.py`** — Central path constants for auditing: the read-only raw
directories and metadata-file locations for all four datasets, the valid
image extensions (`.png/.jpg/.jpeg`), and the `reports/` and `logs/` output
directories. All paths are resolved relative to the file itself, so scripts
run from any working directory. Produces nothing; imported by every audit
module.

**`common/io_utils.py`** — The safety layer. Provides `assert_not_raw_path`,
which raises `RawDataWriteError` if any write target resolves to inside
`data/raw/`; every save helper (`save_csv`, `save_text`, `ensure_dir`) calls
it first. This is what turns "never modify raw data" from a convention into a
runtime-enforced guarantee. Also provides `list_files` (a recursive,
read-only file walk). Input: a DataFrame/text + a target path. Output: files
written safely outside `data/raw/`.

**`common/logging_utils.py`** — Builds a logger that writes simultaneously to
the console and to a timestamped log file under `logs/<Dataset>/`. Every
audit run's full trace is captured for reproducibility. Input: a logger name
and log-file path. Output: a configured `logging.Logger`.

**`run_audit_pad_ufes20.py`** (and `run_audit_ham10000.py`,
`run_audit_isic_archive_1.py`, `run_audit_isic_archive_2.py`) — The four
orchestrators. Each creates a timestamped logger, then calls its dataset's
audit modules **in order**, passing results from one module to the next
(e.g. the image inventory feeds verification, which feeds size stats), and
wraps everything in a try/except that logs any failure. Input: none (reads
`data/raw/`). Output: the full set of report CSVs/figures + a summary
markdown, plus a run log. Run with e.g.
`.venv/Scripts/python.exe -m src.data_audit.run_audit_pad_ufes20`.

#### PAD-UFES-20 audit modules (`pad_ufes20/m01…m11`)

- **`m01_folder_structure.py`** — Walks `data/raw/PAD_UFES20/` and records,
  per folder, the file count, extensions present, and total size. Confirms the
  raw layout matches what the pipeline assumes. Output: `01_folder_structure.csv`.
- **`m02_image_inventory.py`** — Builds the single source-of-truth list of
  every image under `imgs_part_1..3/` (filename, extension, size, relative
  path); warns on any filename appearing in more than one folder. Output:
  `02_image_inventory.csv`.
- **`m03_image_verification.py`** — Opens and fully decodes every inventoried
  image with Pillow (a `verify()` pass then a `load()` pass), recording
  status, dimensions, and colour mode in one read. Detects corruption
  (0 found). Output: `03_image_verification.csv`, `03_corrupted_images.csv`.
- **`m04_missing_image_detection.py`** — Reconciles `metadata.csv`'s `img_id`
  against the on-disk inventory both ways (missing = in metadata but not on
  disk; orphan = on disk but not in metadata) and flags duplicate `img_id`s.
  Output: `04_missing_images.csv`, `04_orphan_images.csv`,
  `04_duplicate_img_id.csv` (all empty — a clean 1:1 mapping).
- **`m05_image_size_stats.py`** — Reuses the dimensions from m03 (no reopen)
  to compute width/height/aspect-ratio statistics and a resolution-frequency
  table, plus a size-distribution figure. Output: `05_image_size_stats.csv`,
  `05_resolution_frequency.csv`, `05_image_mode_frequency.csv`, a figure.
- **`m06_metadata_overview.py`** — Structural profile of `metadata.csv`:
  shape, per-column dtype/unique/missing counts, duplicate-row count, and
  numeric summaries for age and the two diameters. Output:
  `06_metadata_overview.csv`, `06_metadata_numeric_describe.csv`.
- **`m07_column_description.py`** — A human data dictionary: every column
  paired with its clinical meaning (drawn from the published PAD-UFES-20 paper,
  Pacheco et al. 2020) plus computed stats and sample values. Output:
  `07_column_description.csv`.
- **`m08_missing_value_analysis.py`** — Per-column missing count/percentage
  via `isna()`, sorted, to show which fields are structurally sparse. Output:
  `08_missing_value_report.csv`. (Note: this `isna()` measure later proved to
  under-count the symptom columns — see §3, the `UNK` finding.)
- **`m09_class_distribution.py`** — Value counts, percentages, and the
  majority:minority imbalance ratio for the `diagnostic` target, plus a bar
  chart. Output: `09_class_distribution.csv`, a figure.
- **`m10_patient_statistics.py`** — Unique patient/lesion counts,
  images-per-patient stats, patients with multiple lesions, and (crucially)
  patients whose lesions carry more than one diagnosis — needed before a
  patient-wise split can be trusted. Output: `10_patient_statistics.csv`,
  `10_patients_multiple_diagnoses.csv`.
- **`m11_dataset_summary.py`** — Rolls all module outputs into the readable
  `11_dataset_audit_summary.md`, the primary citable audit artifact.

#### HAM10000 audit modules (`ham10000/m01…m11`)

Structurally a near-verbatim mirror of the PAD-UFES-20 audit (same module
names and sequence), with two differences that matter:
- **`m02_image_inventory.py`** walks **two** folders (`HAM10000_images_part_1..2`)
  rather than three.
- **`m10_lesion_statistics.py`** replaces patient statistics: HAM10000 has
  **no patient identifier**, so the grouping unit is `lesion_id`. It reports
  unique lesions, images-per-lesion, lesions with multiple images, and lesions
  with inconsistent diagnoses (found: 0). Output: `10_lesion_statistics.csv`,
  `10_lesions_multiple_diagnoses.csv`.
The other modules (`m01`, `m03`–`m09`, `m11`) do the same jobs as their
PAD-UFES-20 counterparts against HAM10000 paths and columns, writing to
`reports/HAM10000/`.

#### ISIC Archive 1 audit modules (`isic_archive_1/m01…m04`)

This archive has **no metadata.csv** — the folder path *is* the label — so
its audit is a compressed four-module version:
- **`m01_folder_inventory.py`** — Walks `Train/<class>/` and `Test/<class>/`
  in one pass, treating the class-folder name as the label; builds the
  inventory and detects filenames appearing under more than one location,
  distinguishing simple Train/Test repeats from genuine cross-label conflicts.
  Output: `01_folder_structure.csv`, `02_image_inventory.csv`,
  `02_duplicate_filenames.csv`.
- **`m02_image_verification.py`** — Decodes every image and, in the same pass,
  computes size statistics (combines PAD's m03+m05). Output:
  `03_image_verification.csv`, `03_corrupted_images.csv`,
  `04_image_size_stats.csv`, `04_resolution_frequency.csv`, a figure.
- **`m03_class_distribution.py`** — Class counts per Train/Test split and
  overall, plus imbalance ratio and a bar chart. Output:
  `05_class_distribution.csv`, a figure.
- **`m04_dataset_summary.py`** — Writes `06_dataset_audit_summary.md` **and**
  the critical `02_duplicate_filename_label_conflicts.csv`: the row-level list
  of every image filed under two conflicting labels (78 melanoma↔seborrheic
  keratosis + 77 actinic keratosis↔nevus = 155 conflicting filenames).

#### ISIC Archive 2 audit modules (`isic_archive_2/m01…m08`)

A flat `images/` folder plus a rich 27-column `metadata.csv`:
- **`m01_folder_inventory.py`** — Inventories the single flat `images/`
  folder. Output: `01_folder_structure.csv`, `02_image_inventory.csv`.
- **`m02_image_verification.py`** — Decode + size stats in one pass (returns
  both verification and size-stats results). Output: `03_image_verification.csv`,
  `03_corrupted_images.csv`, `04_image_size_stats.csv`,
  `04_resolution_frequency.csv`, a figure.
- **`m03_missing_orphan_detection.py`** — Reconciles `isic_id` (+`.jpg`)
  against the inventory both ways and flags duplicate `isic_id`s. Output:
  `05_missing_images.csv`, `05_orphan_images.csv`, `05_duplicate_isic_id.csv`
  (all empty).
- **`m04_metadata_overview.py`** — Combined structural profile + data
  dictionary for all 27 columns (documents the `diagnosis_1..5` and
  `anatom_site_1..5` specificity hierarchies). Output: `06_column_description.csv`,
  `06_metadata_numeric_describe.csv`.
- **`m05_missing_value_analysis.py`** — `isna()`-based missingness (notes that,
  unlike HAM10000/PAD-UFES-20, no hidden "unknown" string markers exist here so
  `isna()` is reliable). Output: `07_missing_value_report.csv`.
- **`m06_class_distribution.py`** — Both the coarse (`diagnosis_1`:
  Benign/Malignant/Indeterminate) and fine (`diagnosis_3`: specific disease)
  breakdowns, since `diagnosis_3` becomes the class label. Output:
  `08_class_distribution_coarse.csv`, `08_class_distribution_fine.csv`, a figure.
- **`m07_lesion_patient_statistics.py`** — Establishes that `lesion_id` covers
  23,664/25,331 rows but `patient_id` only 417/25,331 — so patient-wise
  splitting is impossible for the bulk, and the split must fall back to
  lesion/image grouping. Output: `09_lesion_patient_statistics.csv`,
  `09_lesions_multiple_diagnoses.csv`.
- **`m08_dataset_summary.py`** — Writes `10_dataset_audit_summary.md`.

<a id="section-2-3"></a>
### 2.3 `src/data_cleaning/` — cleaning, labelling, splitting

**`config.py`** — Path constants for cleaning: raw inputs, `data/interim/`
(intermediate) and `data/processed/` (final) directories, per-dataset
`dataset_source` name strings, and the two reproducibility constants
**`SPLIT_SEED = 42`** and **`SPLIT_RATIOS = {train 0.70, val 0.15, test 0.15}`**.

**`common_label_mapping.py`** — The single shared disease taxonomy
(`ISIC_LABEL_MAPPING`) used by **both** ISIC archives, so that a disease common
to both gets the identical canonical string. Keys are lowercased source labels;
values are `(canonical_name, reason)`. Encodes the deliberate merges (e.g.
"melanoma in situ"/"melanoma invasive"/"melanoma, nos" → Melanoma; the two
1-occurrence nevus variants → Nevus) and the deliberate non-merge (Pigmented
Benign Keratosis kept distinct from Seborrheic Keratosis).

**`cross_dataset_leakage_filter.py`** — Implements fix (b). Reads the already-
final HAM10000 and ISIC-archive split CSVs (read-only — modifies nothing) and,
for each ISIC archive, writes `external_validation_exclusions.csv` listing the
`image_id`s that also appear in HAM10000. Input: the processed metadata CSVs.
Output: `ISIC_Archive_1/external_validation_exclusions.csv` (1,362 ids) and
`ISIC_Archive_2/external_validation_exclusions.csv` (9,873 ids).

**`run_cleaning_*.py`** (four orchestrators) — Each runs its dataset's cleaning
steps in order (column standardization → value validation → label
standardization → split → split-quality report → dataset description) under a
timestamped logger. Input: `data/raw/`. Output: everything in that dataset's
`data/processed/` folder.

#### PAD-UFES-20 cleaning (`pad_ufes20/c01…c06`)

- **`c01_column_standardization.py`** — Renames columns to the unified schema
  (`gender`→`sex`, `region`→`anatomical_site`, `img_id`→`image_id`,
  `diagnostic`→`diagnostic_code`), casts booleans to pandas nullable boolean,
  **maps the literal string `UNK` → missing** in the six symptom columns
  (real missingness the audit's `isna()` could not see), normalises text, and
  resolves `image_path` back into `data/raw/` (no bytes copied). Changes
  representation only — no rows dropped, nothing imputed. Output:
  `interim/PAD_UFES20/metadata_standardized.csv`.
- **`c02_value_validation.py`** — Flags (never edits) implausible values:
  age outside 0–110, sex outside {MALE,FEMALE}, Fitzpatrick outside 1–6,
  diameters outside 0–100 mm. Output: `interim/…/value_validation_report.csv`
  (empty — none flagged).
- **`c03_label_standardization.py`** — Maps the six diagnostic codes
  (BCC/SCC/ACK/NEV/SEK/MEL) to full disease names and writes `label_mapping.csv`
  with a reason per mapping; raises if any code is unmapped.
- **`c04_patient_split.py`** — The **patient-wise** split. Computes each
  patient's dominant diagnosis, groups patients by it, shuffles with seed 42,
  and greedily assigns whole patients to train/val/test by cumulative image
  count toward 70/15/15. Guarantees no patient spans two splits. Output:
  `patient_split_assignment.csv` + `metadata_{train,val,test}.csv`.
- **`c05_split_quality_report.py`** — Verifies zero patient overlap across
  splits (raises if any) and tabulates per-split class balance. Output:
  `split_quality_report.csv`.
- **`c06_dataset_description.py`** — Writes the human-readable
  `dataset_description.md` (schema, split method, seed, limitations, the
  `biopsed` audit, and the boolean-casting loading tip).

#### HAM10000 cleaning (`ham10000/c01…c06`)

Same six-step shape as PAD-UFES-20, differing where the data differs:
- **`c01`** renames `dx`→`diagnostic_code`, `dx_type`→`diagnosis_confirm_type`,
  `localization`→`anatomical_site`, and maps the literal string **`unknown` →
  missing** in `sex` (57 rows) and `anatomical_site` (234 rows).
- **`c02`** flags age/sex anomalies (none found).
- **`c03`** maps the seven `dx` codes, keeping the three shared with
  PAD-UFES-20 identical and giving HAM-specific classes (akiec, bkl, df, vasc)
  their own distinct, deliberately-not-force-merged names.
- **`c04_lesion_split.py`** is **lesion-wise** (no patient id exists);
  stratification is exact because 0 lesions have inconsistent labels. Output:
  `lesion_split_assignment.csv` + the three metadata CSVs.
- **`c05`/`c06`** verify lesion-level leakage-freedom and write the description.

#### ISIC Archive 1 cleaning (`isic_archive_1/c01…c05`)

Five steps, because there is no metadata.csv to standardize:
- **`c01_schema_construction.py`** — Builds the metadata table by walking the
  class folders, then **excludes the conflicting images**: any filename that
  appears under more than one class label is dropped entirely (its true class
  is unknowable). It removes 310 rows (= 155 conflicting filenames × 2 label
  folders). Output: `interim/…/metadata_standardized.csv`, and returns the
  excluded-row count.
- **`c02_label_standardization.py`** — Maps the folder-name labels through the
  shared `ISIC_LABEL_MAPPING`; writes `label_mapping.csv`.
- **`c03_val_split.py`** — Keeps the archive's **own Test set untouched** and
  carves a 15%-of-Train, seed-42, stratified validation set out of Train only.
- **`c04_split_quality_report.py`** — Leakage check keyed on **filename** (no
  lesion/patient id exists). Output: `split_quality_report.csv`.
- **`c05_dataset_description.py`** — Writes `dataset_description.md` including
  the exclusion count and the small-imbalanced-Test caveat.

#### ISIC Archive 2 cleaning (`isic_archive_2/c01…c06`)

- **`c01_column_standardization.py`** — Renames `isic_id`→`image_id`, trims
  text, resolves `image_path`. No rows dropped here.
- **`c02_value_validation.py`** — Flags age/sex/diameter anomalies (none).
- **`c03_label_standardization.py`** — Uses **`diagnosis_3`** (the finest
  level populated for nearly all rows) mapped through the shared taxonomy;
  rows with no `diagnosis_3` (255) get no label and are excluded from splitting.
- **`c04_lesion_split.py`** — Group-wise split where the group is `lesion_id`
  when present (93.4%) and the row's own `image_id` as a singleton group
  otherwise, so no labelled image is dropped and no group spans two splits.
  Output: `group_split_assignment.csv` + three metadata CSVs.
- **`c05_split_quality_report.py`** — Leakage check on the lesion/image group
  key.
- **`c06_dataset_description.py`** — Writes `dataset_description.md`.

<a id="section-2-4"></a>
### 2.4 `src/eda/` — Phase 5 exploratory analysis

**`config.py`** — Paths for EDA outputs (`reports/eda/<Dataset>/` + figures)
and the two sampling constants (`IMAGE_DIM_SAMPLE_SIZE = 250`,
`IMAGES_PER_CLASS_IN_GRID = 4`, seed 42).

**`common/image_stats.py`** — `sample_image_dimensions`: samples up to N images
per dataset, opens each with PIL, and returns width/height/aspect-ratio/file-
size, skipping (and logging) any missing file. Used because measuring all
~39k images is unnecessary for a distribution estimate.

**`common/plotting.py`** — All matplotlib helpers (headless, `Agg` backend):
class-distribution bars, categorical bars, histograms, multi-panel grids,
missingness bars, an image-dimension triptych (width-vs-height scatter, aspect
histogram, size histogram), a sample-image grid, split-balance grouped bars,
and the cross-dataset shared-label comparison. Uses only matplotlib (no new
dependency).

**`eda_pad_ufes20.py`** — Runs the richest EDA (PAD-UFES-20 has the most
metadata): class distribution, demographics + symptom + lifestyle panels,
Fitzpatrick distribution and Fitzpatrick×disease crosstab, missingness across
all whitelisted features, sampled image dimensions, a per-class sample grid,
and a train/val/test split-balance check. Reads only `metadata_train.csv`
(val/test only for the split-balance counts). Output: the CSVs/figures under
`reports/eda/PAD_UFES20/`.

**`eda_ham10000.py`** — Same shape but limited to HAM10000's three usable
features (age, sex, anatomical_site); no Fitzpatrick step (no such column).

**`eda_isic_archive_1.py`** — The thinnest EDA: with **0 usable metadata
columns**, only class distribution, image dimensions, sample grid, and split
balance are possible.

**`eda_isic_archive_2.py`** — Class distribution; demographics plus the
anatomical-site-hierarchy and clinical-history panels; missingness across the
whitelisted features (this is the report that surfaced the 88–100% sparse
fields); image dimensions (revealed the bimodal 600×450 / 1024×1024 split);
sample grid; split balance.

**`eda_cross_dataset.py`** — Compares train-set counts for `disease_label`
values shared across 2+ datasets, with a prominent caption warning that the
chart is descriptive only and must not be read as four independent pools
(because of the documented image overlap). Output:
`reports/eda/cross_dataset/01_shared_label_comparison.csv` + figure.

<a id="section-2-5"></a>
### 2.5 `data/processed/<Dataset>/` files

The three `metadata_{train,val,test}.csv` per dataset are the model-ready
tables; every image row keeps its `image_path` pointing back into
`data/raw/`. Column counts and row counts:

- **PAD-UFES-20** — `metadata_train.csv` (1,606 rows), `metadata_val.csv`
  (338), `metadata_test.csv` (354); **29 columns** each (patient_id, lesion_id,
  age, sex, anatomical_site, two diameters, Fitzpatrick, 6 symptom flags, 8
  lifestyle/clinical flags, diagnostic_code, biopsed, image_id, image_path,
  dataset_source, disease_label). `label_mapping.csv` (6 code→name rows).
  `patient_split_assignment.csv` (1,373 rows: patient_id, dominant label,
  n_images, split). `split_quality_report.csv` (18 rows: split × class counts
  and percentages). `dataset_description.md`. `feature_whitelist.md` (**21
  allowed** input features; excludes `biopsed`, `diagnostic_code`, and the
  identifier/path/label columns).
- **HAM10000** — train 7,004 / val 1,501 / test 1,510; **10 columns**
  (lesion_id, image_id, diagnostic_code, diagnosis_confirm_type, age, sex,
  anatomical_site, dataset_source, image_path, disease_label).
  `label_mapping.csv` (7 rows). `lesion_split_assignment.csv` (7,470 rows).
  `split_quality_report.csv` (21 rows). `dataset_description.md`.
  `feature_whitelist.md` (**3 allowed**: age, sex, anatomical_site; excludes
  `diagnostic_code` and `diagnosis_confirm_type` as label-source/leakage).
- **ISIC Archive 1** — train 1,655 / val 292 / test 100; **6 columns**
  (image_id, filename, class_label, image_path, dataset_source, disease_label).
  `label_mapping.csv` (9 rows). `split_quality_report.csv` (26 rows).
  `external_validation_exclusions.csv` (1,362 image_ids).
  `dataset_description.md`. `feature_whitelist.md` (**0 allowed** — image +
  label only; any model here is necessarily image-only).
- **ISIC Archive 2** — train 17,535 / val 3,769 / test 3,772; **30 columns**
  (image_id, attribution, copyright_license, age_approx, anatom_site_1..5,
  anatom_site_general, anatom_site_special, clin_size_long_diam_mm,
  concomitant_biopsy, dermoscopic_type, diagnosis_1..5, diagnosis_confirm_type,
  family_hx_mm, image_type, lesion_id, melanocytic, patient_id, personal_hx_mm,
  sex, dataset_source, image_path, disease_label). `label_mapping.csv`
  (13 rows). `group_split_assignment.csv` (13,794 rows: group_id, label,
  n_images, split). `split_quality_report.csv` (27 rows).
  `external_validation_exclusions.csv` (9,873 image_ids).
  `dataset_description.md`. `feature_whitelist.md` (**7 allowed**, of which
  **4 are the active Phase 6 baseline set**: age_approx, sex, anatom_site_1,
  anatom_site_general — see §3 for why the other columns are excluded).

---

<a id="section-3"></a>
## Section 3 — Chronological narrative: what actually happened, in order

**1. Planning (completed 2026-06-29).** The project was scoped as a master's
thesis on multimodal skin-lesion classification combining images with clinical
metadata, targeting the full multi-class disease taxonomy rather than a binary
benign/malignant split (a deliberate differentiation from the reviewed
literature). The README, `Dataset_Strategy.md`, and `AI_Assistant_Instructions.md`
were authored, fixing the non-negotiable rules: raw data is read-only, missing
values are never invented, splitting is patient-/lesion-wise (never image-wise),
and every decision is documented. A literature review of three multimodal
skin-lesion papers (Mridha & Islam 2026; Suresh et al. 2026 TG-CAVNet;
Watson et al. 2026) identified concrete gaps (multi-class fusion, leakage
auditing) the thesis targets. Watson et al.'s warning about
diagnosis-derived-after-the-fact fields directly motivated the later leakage
audits.

**2. Dataset collection.** Four public datasets were acquired into
`data/raw/`: PAD-UFES-20 (primary; smartphone clinical images with rich patient
metadata), HAM10000 (dermoscopic benchmark), and two ISIC Archive exports
(external validation).

**3. Auditing, per dataset (2026-07-07).** A read-only audit pipeline profiled
each dataset. Key findings:
- **PAD-UFES-20** (audited 19:51): 2,298 images across three folders, all
  decoded successfully, **0 corrupted, 0 missing, 0 orphan**. Six diagnostic
  classes with imbalance BCC 845 / ACK 730 / NEV 244 / SEK 235 / SCC 192 /
  MEL 52 (~16:1). 1,373 patients, 1,641 lesions, 355 patients with multiple
  lesions, and **179 patients with more than one distinct diagnosis** (which
  forced the "dominant diagnosis" stratification approach). Widely varying
  image sizes (width 147–3,474 px).
- **HAM10000** (audited 21:19): 10,015 images, all decoded, 0 corrupted/
  missing/orphan. Seven classes, heavily imbalanced (Nevus 6,705 = 66.95%
  down to Dermatofibroma 115 = 1.15%, ~58:1). **No patient identifier** — only
  7,470 lesions, 0 with inconsistent labels. Uniform 600×450 images. The audit
  flagged that `sex` (57 rows) and `localization` (234 rows) encode missingness
  as the literal string "unknown".
- **ISIC Archive 1** (audited 23:01): No metadata.csv — labels come from
  `Train|Test/<class>/` folders. 2,357 images on disk, 0 corrupted. Nine
  classes. **Genuine data defect found:** 155 filenames each appear under two
  conflicting class labels, systematically — 78 melanoma↔seborrheic keratosis
  and 77 actinic keratosis↔nevus. No patient/lesion identifier exists.
- **ISIC Archive 2** (audited 23:39): Flat `images/` + 27-column metadata.
  25,331 images, 0 corrupted/missing/orphan. `diagnosis_3` chosen as the class
  label (populated for all but 255 rows). Severe imbalance (Nevus ~51% down to
  two 1-image classes). `lesion_id` covers 23,664/25,331 rows but `patient_id`
  only 417 — so patient-wise splitting is impossible for the bulk. Image sizes
  bimodal (median 1,024 but a large 600×450 cluster too).

**4. Cleaning, labelling, and splitting, per dataset (2026-07-07 → 2026-07-08).**
- **PAD-UFES-20:** columns standardized; the hidden `UNK` symptom-missingness
  was surfaced (revised missingness: grew 17.49%, changed 17.23%, others <0.5%);
  no implausible values; six codes mapped to disease names; a **patient-wise
  70/15/15** split (seed 42) produced train 1,606 (948 patients) / val 338
  (205) / test 354 (220), verified zero patient overlap.
- **HAM10000:** the "unknown" strings mapped to missing; seven codes mapped
  (three shared names with PAD-UFES-20, four kept deliberately distinct — akiec
  as "Actinic Keratosis / Intraepithelial Carcinoma" and bkl as "Benign
  Keratosis-like Lesion", both broader than the narrower ISIC/PAD categories);
  a **lesion-wise** split gave train 7,004 / val 1,501 / test 1,510, zero
  lesion overlap.
- **ISIC Archive 1:** the 155 conflicting filenames were **excluded** (310
  physical rows removed: 2,357 − 310 = 2,047), rather than guessed at; labels
  mapped through the shared ISIC taxonomy; the archive's own Test set (100
  images) kept untouched with a 15%-of-Train validation carved out → train
  1,655 / val 292 / test 100.
- **ISIC Archive 2:** the 255 rows with no `diagnosis_3` excluded (25,331 −
  255 = 25,076); labels mapped (including deliberate melanoma-subtype and
  two 1-image nevus-variant merges); **group-wise** split on lesion_id with
  image_id fallback → train 17,535 / val 3,769 / test 3,772.

**5. Independent cross-dataset verification (2026-07-08).** After all four
were individually done, a fresh verification re-derived every number and found
each dataset internally sound — but uncovered the **critical cross-dataset
image overlap**: comparing `image_id`s, HAM10000 ↔ ISIC Archive 2 share
**9,873** images (98.6% of HAM10000), ISIC Archive 1 ↔ ISIC Archive 2 share
**1,673** (81.7% of Archive 1), and HAM10000 ↔ ISIC Archive 1 share **1,362**
(66.5% of Archive 1). PAD-UFES-20 has **zero** overlap with anything.
Because each split was computed independently, 40–46% of the overlapping
images land in *different* splits across datasets — so using an ISIC archive
as "external validation" for a HAM10000-trained model would silently score it
on images seen in training.

**6. Cross-dataset leakage fix — decision (2026-07-08).** Rather than
re-splitting all three datasets globally (fix a, which would discard verified
splits), **fix (b)** was chosen: keep every internal split untouched and,
at evaluation time only, exclude the overlapping images. `cross_dataset_leakage_filter.py`
generated `external_validation_exclusions.csv` for each archive — **1,362**
excluded ids for ISIC Archive 1 (685 remain valid) and **9,873** for ISIC
Archive 2 (15,203 remain). A residual 3-image label disagreement between the
two archives (`ISIC_0028619`, `ISIC_0011126`, `ISIC_0011118`) was documented,
not "fixed," as it is inherited from the source archives.

**7. Label-leakage / shortcut feature audit (2026-07-08).** Every candidate
feature that could let a model cheat was tested with real numbers and then
excluded from model input (kept in the CSVs only as documentation):
- **PAD-UFES-20 `biopsed`** — every one of the 1,089 malignant-labelled
  images has `biopsed=True` (0 exceptions) vs. only 21% of non-malignant
  images. **Phi = 0.80, chi-square = 1,474.5, n = 2,298.** A near-perfect
  proxy for malignancy; excluded.
- **PAD-UFES-20 `diagnostic_code`** — the raw code the label was derived from,
  1:1 with `disease_label` (6/6). Label-source; excluded.
- **HAM10000 `diagnosis_confirm_type`** — every malignant image (1,627/1,627)
  is confirmed via histopathology; non-malignant images spread across four
  confirmation types. **Phi = 0.41, chi-square = 1,700.67, n = 10,015.**
  Excluded (weaker than `biopsed` because histo is also used for many benign
  cases, but the malignant→histo direction is deterministic).
- **HAM10000 `diagnostic_code`** — 1:1 with the label (7/7). Excluded.
- **ISIC Archive 1 `class_label`** — the folder label the target is a direct
  rename of (9/9). Excluded (leaving 0 usable features).
- **ISIC Archive 2 `diagnosis_confirm_type`** — **Phi = 0.36, chi-square =
  3,171.74, n = 25,076**; malignant cases are never confirmed by serial
  imaging or single-image consensus, though histopathology is used for both
  malignant (7,685/8,473 = 90.7%) and benign (9,205/16,603 = 55.4%) cases.
  Excluded. (Precisely: these figures come from binarising the column as
  `histopathology` vs. everything else. Treating all confirmation methods as
  separate categories, and missing values as their own category, gives a
  somewhat stronger phi = 0.39, chi-square = 3,773.40 on the same n = 25,076;
  2,440 rows have no recorded confirmation type. Either way the column is a
  shortcut and is excluded.)
- **ISIC Archive 2 `diagnosis_1/2/3`** (and, by extension, `diagnosis_4/5`) —
  the diagnostic hierarchy the label was built from; `diagnosis_3` is 1:1 with
  the label. Excluded.
- **ISIC Archive 2 `concomitant_biopsy`** — produces the *identical*
  contingency table to `diagnosis_confirm_type == histopathology` (same
  Phi = 0.36); a duplicate encoding of the confirmation signal. Excluded.
- **ISIC Archive 2 `melanocytic`** — a **perfect deterministic split** of the
  label: Melanoma and Nevus are 100% `melanocytic=True` (17,395/17,395), all
  seven other classes 100% `False` (7,681/7,681), zero exceptions. Excluded.
- **ISIC Archive 2 `attribution`, `copyright_license`, `image_type`** —
  source-identifying/administrative, not clinical (`attribution`'s "ViDIR
  Group, Vienna" value = exactly 9,873 rows, matching the HAM10000 overlap
  count; `image_type` is constant). Excluded.

**8. Feature whitelists + documentation cleanup + phase closure (2026-07-08).**
A `feature_whitelist.md` was written for each dataset listing exactly which
columns may be model inputs (PAD-UFES-20: 21; HAM10000: 3; ISIC Archive 1: 0;
ISIC Archive 2: 13 at this point) with a one-line reason per exclusion. Stale
docs were archived to `_archive/` (never deleted), `Dataset_Strategy.md` was
corrected in place, and Dataset Preparation (Phase 4) was formally closed.

**9. Exploratory Data Analysis (Phase 5, completed 2026-07-09).** The EDA
pipeline ran on all four datasets plus a cross-dataset comparison; every output
was verified non-empty and mirrored into five notebooks and `eda_summary.md`.
Findings: every dataset needs class-weighted loss (train-set imbalance from
~15:1 in PAD-UFES-20 up to ~393:1 in ISIC Archive 1, whose train set has a
single seborrheic keratosis image after the conflict exclusion); PAD-UFES-20's
richest fields co-miss together in ~34–35% of patients while its symptom flags
are >99.5% complete; ISIC Archive 1 has no usable metadata at all; and ISIC
Archive 2's image resolution is **bimodal (600×450 vs ~1024×1024)** aligning
exactly with its two `attribution` sources — an independent visual confirmation
of the HAM10000 image overlap found earlier by id-matching. Two items were
flagged as open, not decided: ISIC Archive 2's sparse-field handling and the
Phase 6 input-resolution choice.

**10. Post-EDA decisions closing the two open items (2026-07-09).**
- **ISIC Archive 2 sparse fields:** a crosstab of "is this field missing?"
  against `attribution` was run for the six sparse fields (`anatom_site_3/4/5`,
  `family_hx_mm`, `personal_hx_mm`, `clin_size_long_diam_mm`). Five of the six
  are **never** populated for the Hospital Clínic or ViDIR rows (present only,
  and only partially, for "Anonymous" rows); `anatom_site_3` is present for
  0.00% of Hospital Clínic rows. Because a field's mere presence identifies the
  source institution, all six were reclassified as **source-leak risk and
  excluded** (not imputed). This reduced ISIC Archive 2's whitelist from 13 to
  **7 allowed**, with a **4-field active baseline** (age_approx, sex,
  anatom_site_1, anatom_site_general). `clin_size_long_diam_mm`, initially
  proposed for the baseline, was corrected out for the same reason.
- **Input resolution:** **224×224, aspect-ratio-preserving resize-and-pad,
  applied at data-load time** (not stored as resized copies — consistent with
  the no-image-copying rule). 224 matches pretrained ImageNet backbones;
  resize-and-pad (vs. stretch) avoids distorting lesion shape given the
  documented heterogeneity.

At this point the datasets are "ready": four processed, individually
leakage-verified, feature-whitelisted datasets, with the cross-dataset caveat
handled by exclusion lists and the two Phase-6 pre-conditions resolved.

---

<a id="section-4"></a>
## Section 4 — Current dataset state (summary table)

All counts are the **cleaned/processed** figures (post-exclusion), verified
from the CSVs on 2026-07-09.

| Dataset | Total images | Classes | Train / Val / Test | Allowed features | Key limitation |
|---|---|---|---|---|---|
| PAD-UFES-20 | 2,298 | 6 | 1,606 / 338 / 354 | 21 | ~34–35% co-missing lifestyle/measurement fields; 179 multi-diagnosis patients; imbalance ~16:1 (train ~15:1) |
| HAM10000 | 10,015 | 7 | 7,004 / 1,501 / 1,510 | 3 | No patient id (lesion-wise only); imbalance ~58:1; ~99% image overlap with ISIC Archive 2 |
| ISIC Archive 1 | 2,047 (155 conflicting filenames = 310 files excluded) | 9 | 1,655 / 292 / 100 | 0 | No metadata at all (image-only); tiny/imbalanced Test (train has 1 seborrheic keratosis); ~66% overlap with HAM10000 |
| ISIC Archive 2 | 25,076 (255 unlabeled rows excluded) | 9 | 17,535 / 3,769 / 3,772 | 7 (4 active baseline) | Mostly-sparse metadata; patient id <2%; train imbalance 63.5:1 (Nevus 9,011 vs Solar Lentigo 142); ~99% overlap with HAM10000 |

Cross-dataset: **PAD-UFES-20 is the only fully independent source** (zero
image overlap). When an ISIC archive is used as external validation against a
HAM10000-trained model, apply its `external_validation_exclusions.csv` first
(ISIC Archive 1: 1,362 excluded → 685 usable; ISIC Archive 2: 9,873 excluded →
15,203 usable).

---

<a id="section-5"></a>
## Section 5 — What a manual (non-AI) researcher would have had to do

This section states the actual technical work the pipeline performs, so it can
be explained as genuine methodology rather than "a tool did it." In order:

1. **Establish a read-only discipline.** Write a guard that refuses any write
   whose path falls inside `data/raw/`, and route all output through it, so no
   audit/cleaning step can ever mutate source data.
2. **Inventory images.** Recursively walk each dataset's image folders and
   build one canonical list of every file (name, size, relative path); detect
   filenames appearing in more than one location.
3. **Verify image integrity.** Open and fully decode every image with a
   library like Pillow, recording dimensions and colour mode and marking any
   file that fails to decode as corrupted.
4. **Reconcile images against metadata.** For datasets with a metadata file,
   compare the id column against the on-disk inventory both ways to find
   missing files (in metadata, not on disk) and orphans (on disk, not in
   metadata); flag duplicate ids.
5. **Profile the metadata.** Compute per-column dtype, unique count, and
   missing count; summarise numeric columns; count fully-duplicate rows.
6. **Write a data dictionary by hand.** Record each column's clinical meaning
   from the source dataset's publication — this is domain knowledge that cannot
   be inferred from values.
7. **Detect *hidden* missingness.** Recognise that `isna()` misses
   "unknown"/"UNK" string sentinels; scan categorical columns for such markers
   and map them to true missing values (a standardization, not an invention),
   then recompute missingness.
8. **Compute class distributions and imbalance ratios** for every target, to
   justify class-weighted loss later.
9. **Compute grouping statistics** — unique patients/lesions, images per
   group, and (critically) any patient/lesion carrying more than one diagnosis
   — because these determine whether an exact or approximate stratification is
   possible.
10. **Harmonise labels across four datasets.** Design one canonical disease
    taxonomy, decide every merge and non-merge with a written clinical reason
    (e.g. keep HAM10000's broader "akiec"/"bkl" distinct; collapse melanoma
    subtypes), and record it as `label_mapping.csv` per dataset.
11. **Handle datasets without metadata.** For ISIC Archive 1, derive labels
    from folder names, and detect+exclude images filed under two conflicting
    labels rather than guessing.
12. **Handle a label hierarchy.** For ISIC Archive 2, pick the finest fully-
    populated diagnosis level as the label and exclude rows too coarse to
    classify.
13. **Implement grouped, stratified, seeded splitting by hand.** Group by the
    correct unit (patient / lesion / lesion-with-image-fallback / filename),
    stratify by class, shuffle with a fixed seed, and assign whole groups to
    train/val/test greedily by cumulative image count toward 70/15/15 — never
    a naive per-image random split.
14. **Prove the split is leakage-free.** Assert zero overlap of the grouping
    key across the three splits, and tabulate per-split class balance.
15. **Detect cross-dataset leakage.** Compare image ids across datasets to
    discover that HAM10000 and the ISIC archives share physical images, quantify
    how many overlapping images land in different splits, and decide a remedy
    (here: build per-archive exclusion lists).
16. **Test candidate leakage/shortcut features statistically.** For each
    suspicious column, build a contingency table against malignant/non-malignant
    (or against the label) and compute the phi coefficient and chi-square, then
    exclude any near-deterministic proxy — this is what produced the phi=0.80
    (`biopsed`), 0.41 (HAM confirm-type), 0.36 (ISIC-2 confirm-type) figures and
    the perfect `melanocytic` split.
17. **Detect source-identity leakage in missingness.** Crosstab "is this field
    present?" against a source-identifier column to catch fields whose presence
    silently encodes the contributing institution.
18. **Produce a feature whitelist** per dataset — the explicit final list of
    allowed model inputs with a reason for every exclusion.
19. **Run EDA and plot everything** — class/demographic/Fitzpatrick
    distributions, missingness bars, sampled image-dimension distributions,
    per-class sample grids, split-balance charts — headless and reproducibly.
20. **Document every dataset** in a human-readable description and keep a
    living decision log, so each choice is defensible and reproducible with a
    fixed seed.

---

<a id="section-6"></a>
## Section 6 — Open items / not yet done

**Dataset-preparation and EDA are complete and reviewed.** As of dataset-ready
state, the following are the honest caveats and the hand-off into Phase 6.

Known limitations that remain (documented, not defects):
- **Cross-dataset image overlap is mitigated, not eliminated.** The exclusion
  lists must be *actively applied* whenever an ISIC archive is used as external
  validation against a HAM10000-trained model. Nothing enforces this
  automatically at train time yet — it is the modeller's responsibility.
- **No patient-level control for HAM10000 and ISIC Archive 2.** HAM10000 has no
  patient id (lesion-wise only) and ISIC Archive 2's `patient_id` covers <2% of
  rows, so same-patient leakage across splits cannot be fully ruled out for
  those two datasets (only lesion/group-level leakage is controlled).
- **ISIC Archive 1 is image-only** (0 usable metadata) and its Test set is tiny
  and imbalanced; its train set has a single seborrheic keratosis image.
- **ISIC Archive 2 deferred features:** three whitelisted-in-principle columns
  (`anatom_site_2`, `anatom_site_special`, `dermoscopic_type`) are held out of
  the active baseline set pending their own missingness/attribution check,
  which has not been run.
- **HAM10000 boolean caveat / dtype note:** PAD-UFES-20 boolean columns
  round-trip through CSV as `True`/`False`/blank and should be explicitly
  re-cast after loading (documented in its `dataset_description.md`).
- **Three residual cross-archive label disagreements** (`ISIC_0028619`,
  `ISIC_0011126`, `ISIC_0011118`) remain as inherited annotation noise if those
  specific images are ever pooled.

Not part of dataset preparation (correctly deferred):
- The Literature Review is marked "in progress" (3 papers reviewed) and
  continues as new papers are added.
- Fairness/bias analysis across Fitzpatrick and age (only PAD-UFES-20 has
  Fitzpatrick, ~34% missing) is a Phase 8 activity; the crosstab is already
  saved for it.

What Phase 6 (Baseline Model Development) will take from this foundation:
- **PAD-UFES-20 as the starting dataset** (it is the only fully independent
  source and needs no exclusion handling), building an image-only and a
  metadata-only baseline.
- **Each dataset's `feature_whitelist.md`** as the definitive, non-negotiable
  list of allowed model inputs (so no excluded leakage feature is ever fed to a
  model).
- **Class-weighted loss** driven by the documented imbalance, with macro-F1 +
  per-class F1 + confusion matrix as the reporting metrics.
- **224×224 aspect-preserving resize-and-pad at load time** as the agreed image
  input transform.
- **Fixed seeds with mean±std reporting**, and strict val/test discipline (the
  test split untouched until a single final evaluation).

*(Note: Phase 6 code has in fact since begun to be written in `src/models/`
and `src/evaluation/`, but that is outside the dataset-preparation scope this
document covers and is intentionally not described here.)*

---

<a id="appendix"></a>
## Appendix — Independent re-verification (2026-07-09)

Every quantitative claim in this document was recomputed directly from the
processed CSVs, rather than copied from any earlier report. The following
reproduced **exactly**:

- All 25 processed-CSV row and column counts (§2.5) — e.g. PAD-UFES-20
  1,606/338/354 × 29 cols; HAM10000 7,004/1,501/1,510 × 10; ISIC Archive 1
  1,655/292/100 × 6; ISIC Archive 2 17,535/3,769/3,772 × 30.
- **PAD-UFES-20 `biopsed`:** phi = 0.801, chi-square = 1,474.54, n = 2,298;
  all 1,089 malignant images have `biopsed=True`; non-malignant rate 20.9%.
- **HAM10000 `diagnosis_confirm_type`:** phi = 0.412, chi-square = 1,700.67,
  n = 10,015; 1,627/1,627 malignant confirmed by histopathology.
- **ISIC Archive 2 `melanocytic`:** a perfect partition — 17,395 Melanoma/Nevus
  rows all `True`, 7,681 other-class rows all `False`, zero exceptions.
- **ISIC Archive 2 `concomitant_biopsy`:** contingency table identical to
  binarised `diagnosis_confirm_type` (7,685 malignant / 9,205 benign), hence
  identical phi = 0.3556 and chi-square = 3,171.74.
- **Cross-dataset overlap:** HAM10000 ∩ Archive 2 = 9,873 (98.6% of HAM10000);
  Archive 1 ∩ Archive 2 = 1,673 (81.7% of Archive 1); HAM10000 ∩ Archive 1 =
  1,362 (66.5% of Archive 1); PAD-UFES-20 ∩ (all others) = **0**.
- Each `external_validation_exclusions.csv` is **exactly** the corresponding
  HAM10000 intersection (set-equality confirmed), leaving 685 usable images in
  Archive 1 and 15,203 in Archive 2.
- **Residual label disagreements:** exactly 3, and exactly the ids named —
  `ISIC_0028619` (Nevus vs. Actinic Keratosis), `ISIC_0011126` and
  `ISIC_0011118` (both Seborrheic Keratosis vs. Melanoma).
- **ISIC Archive 1 conflicts:** 155 unique filenames across 310 rows, split
  78 melanoma↔seborrheic keratosis and 77 actinic keratosis↔nevus; each
  conflicting filename appears once in `Train/` and once in `Test/`.
- **Class imbalance:** PAD-UFES-20 845/52 = 16.3:1; HAM10000 6,705/115 = 58.3:1;
  ISIC Archive 1 train 393/1 = 393:1 (one seborrheic keratosis image).

Three corrections were applied to this document as a result:

1. ISIC Archive 2's `label_mapping.csv` has **13** rows, not 14.
2. ISIC Archive 2's train imbalance is **63.5:1**, not "~62:1".
3. The ISIC Archive 2 `diagnosis_confirm_type` phi of 0.36 is specifically the
   *histopathology-vs-rest* binarisation; the full multi-category table gives
   phi = 0.39. Both are now stated in §3.

The literature-review spreadsheet was also opened and its two sheets and their
column headers confirmed, so §2.1 now describes it from the file itself rather
than from the tracking log. **No claim in this document is left unverified.**

---

<a id="section-7"></a>
## Section 7 — How to reproduce this pipeline from scratch

Everything below is a literal, copy-pasteable command sequence. All commands
are run from the project root (`Multimodal_Skin_Disease_Research/`) in
Windows PowerShell, which is what this project was built and run on. Each
script is a proper Python module (has an `if __name__ == "__main__":` guard)
and is invoked with `-m`, never run as a bare file path — the `-m` form is
what lets the scripts' internal `from src...import` statements resolve.

<a id="section-7-1"></a>
### 7.1 Environment setup (one-time)

```powershell
# 1. Create the virtual environment (only needed once)
python -m venv .venv

# 2. Activate it (every new terminal session)
.venv\Scripts\Activate.ps1

# 3. Install the pinned dependencies
pip install -r requirements.txt
```

`requirements.txt` pins seven packages: `pandas==3.0.3`, `Pillow==12.3.0`,
`numpy==2.5.1`, `matplotlib==3.11.0`, `torch==2.13.0`, `torchvision==0.28.0`,
`scikit-learn==1.9.0`. (`torch`/`torchvision`/`scikit-learn` are Phase 6
dependencies, not used by the audit/cleaning/EDA scripts covered here, but
are installed together since they share the one requirements file.)

**Verify success:** `python --version` and `pip show pandas` both resolve
inside the activated `.venv`, with no import errors.

<a id="section-7-2"></a>
### 7.2 Run the audits (Phase 4a) — any order, each independent

Each audit script only reads its own dataset's folder under `data/raw/` and
writes only to its own `reports/<Dataset>/` and `logs/<Dataset>/` folders, so
the four can run in any order (including in parallel). Order given here is
the order they were actually run in:

```powershell
.venv\Scripts\python.exe -m src.data_audit.run_audit_pad_ufes20
.venv\Scripts\python.exe -m src.data_audit.run_audit_ham10000
.venv\Scripts\python.exe -m src.data_audit.run_audit_isic_archive_1
.venv\Scripts\python.exe -m src.data_audit.run_audit_isic_archive_2
```

**Observed real run times** (measured from this project's own log files —
timestamp of the log's first line to the log file's last-modified time):
PAD-UFES-20 ≈ 65 s, HAM10000 ≈ 2 min, ISIC Archive 1 ≈ 16 s, ISIC Archive 2 ≈
5 min. The dominant cost is `m0x_image_verification`/`m0x_image_inventory`,
which fully decodes every image with Pillow — so runtime scales with image
count (ISIC Archive 2 has the most images at 25,331, hence the longest run).

**Verify success (per dataset):**
- Its summary markdown exists and is non-empty, e.g.
  `reports/PAD_UFES20/11_dataset_audit_summary.md` (or `06_...` for ISIC
  Archive 1, `10_...` for ISIC Archive 2).
- Its `03_corrupted_images.csv` (or equivalent) is empty — 0 corrupted images
  is the expected baseline result for all four datasets.
- A new timestamped log file appears under `logs/<Dataset>/` and ends with a
  `... AUDIT - COMPLETE` banner rather than a Python traceback.

<a id="section-7-3"></a>
### 7.3 Run the cleaning steps (Phase 4b) — 4 datasets, then the leakage filter

The four `run_cleaning_*.py` scripts are independent of each other (each
reads only its own `data/raw/` + writes only its own `data/interim/` and
`data/processed/` folder) and can run in any order, but **all four must
finish before `cross_dataset_leakage_filter.py` runs**, because that script
reads the *finished* `metadata_{train,val,test}.csv` output of HAM10000 and
both ISIC archives to compute image-id overlap — it will silently produce
wrong (or missing-file) results if run against partial output.

```powershell
# Step 1 — the four dataset-specific cleaning pipelines, any order:
.venv\Scripts\python.exe -m src.data_cleaning.run_cleaning_pad_ufes20
.venv\Scripts\python.exe -m src.data_cleaning.run_cleaning_ham10000
.venv\Scripts\python.exe -m src.data_cleaning.run_cleaning_isic_archive_1
.venv\Scripts\python.exe -m src.data_cleaning.run_cleaning_isic_archive_2

# Step 2 — ONLY after all four above have completed successfully:
.venv\Scripts\python.exe -m src.data_cleaning.cross_dataset_leakage_filter
```

**Observed real run times:** each of the four dataset-cleaning scripts
completes in 1–2 seconds (they only manipulate the already-decoded metadata
CSVs, not image bytes — no re-decoding happens in this phase).
`cross_dataset_leakage_filter.py` is similarly near-instant (it is a set
intersection over `image_id` columns).

**Verify success:**
- Each dataset's `data/processed/<Dataset>/` folder is fully populated (see
  §2.5 for the exact expected file list and row counts per dataset).
- Each `split_quality_report.csv` exists — its generation step
  (`c05_split_quality_report.py`) `raise`s immediately if it finds any
  patient/lesion/group overlap across train/val/test, so the script
  completing at all is itself a leakage-free proof, not just a file check.
- After step 2, `data/processed/ISIC_Archive_1/external_validation_exclusions.csv`
  has 1,362 rows and `data/processed/ISIC_Archive_2/external_validation_exclusions.csv`
  has 9,873 rows (§4 / Appendix).

<a id="section-7-4"></a>
### 7.4 Run the EDA (Phase 5) — 4 per-dataset scripts, then the cross-dataset one

The four per-dataset EDA scripts are independent of each other and of the
cross-dataset script, but **`eda_cross_dataset.py` should run last**, since it
reads the `metadata_train.csv` of all four datasets (already produced in
§7.3) — it does not depend on the other EDA scripts' output, only on Phase
4b's, but running it last mirrors how the pipeline was actually exercised and
keeps the log timestamps meaningfully ordered.

```powershell
.venv\Scripts\python.exe -m src.eda.eda_pad_ufes20
.venv\Scripts\python.exe -m src.eda.eda_ham10000
.venv\Scripts\python.exe -m src.eda.eda_isic_archive_1
.venv\Scripts\python.exe -m src.eda.eda_isic_archive_2
.venv\Scripts\python.exe -m src.eda.eda_cross_dataset
```

**Observed real run times:** PAD-UFES-20 ≈ 2 s, HAM10000 ≈ 7 s, ISIC Archive
1 ≈ 6 s, ISIC Archive 2 ≈ 21 s (longest, since it has the most rows and the
richest metadata to plot), cross-dataset ≈ 1 s. All five together run in
well under a minute — EDA only samples up to `IMAGE_DIM_SAMPLE_SIZE = 250`
images per dataset for the dimension analysis rather than opening every
image, which is why this phase is far faster than the audit phase despite
covering the same datasets.

**Verify success:**
- `reports/eda/<Dataset>/figures/` contains PNG files (e.g.
  `01_class_distribution.png`, `06_sample_image_grid.png`) and is not empty.
- `reports/eda/cross_dataset/01_shared_label_comparison.csv` and its
  figure exist after the last command.
- No script raises; each prints/logs a completion line and exits 0.

<a id="section-7-5"></a>
### 7.5 End-to-end sanity check

After 7.1–7.4 complete, the dataset is "ready" in the same sense as §3.9–§3.10
of this document. A quick way to confirm the whole pipeline reproduced
correctly without re-reading every file by hand:

```powershell
.venv\Scripts\python.exe -c "import pandas as pd; [print(ds, len(pd.read_csv(f'data/processed/{ds}/metadata_train.csv'))) for ds in ['PAD_UFES20','HAM10000','ISIC_Archive_1','ISIC_Archive_2']]"
```

Expected output: `PAD_UFES20 1606`, `HAM10000 7004`, `ISIC_Archive_1 1655`,
`ISIC_Archive_2 17535` — matching §2.5/§4 exactly. Any different number means
something upstream (raw data, a dependency version, or the random seed
handling) diverged from this document's reference run.
