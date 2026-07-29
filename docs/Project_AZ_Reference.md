# A-to-Z Project Reference: Datasets, Models, Pipeline

**Purpose:** a complete, file-verified reference for the user's own understanding (supervisor/teammate explanation, viva prep) — not a thesis chapter draft. Every fact below was extracted by directly reading the project's actual files (CSVs, audit reports, `.py` source, `config.py`, result JSON/logs), not summarized from memory. Items that could not be verified are explicitly marked **NOT FOUND**.

**Compiled:** 2026-07-28, via 7 parallel file-verification passes over `D:\Naeem\thesis-v2\Multimodal_Skin_Disease_Research`.

---

# PART A — Datasets

## A.1 PAD-UFES-20

### 1. Source
- **No source URL, portal, or download date is documented anywhere in the repo.** No readme/citation/license file exists inside `data/raw/PAD_UFES20/` (only `metadata.csv` + 3 image folders). `docs/Project_Tracking.md:230` only states it was "acquired into `data/raw/`" with no further detail.
- **Dataset citation paper (identified but not yet added to the lit review):** Pacheco et al., *"PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones,"* **Data in Brief** (`docs/Literature_Review.md:198-211`). Distinct from the two Pacheco & Krohling *methods* papers already read in full (2021 MetaBlock; 2020 arXiv:1909.12912, whose 1,612-image/6-class/8-field dataset is the direct precursor to this public 2,298-image release — not the same dataset object).
- **License: NOT FOUND** — checked `README.md`, `docs/PROJECT_PLAN.md`, `docs/Dataset_Strategy.md`, `docs/Project_Tracking.md`, `docs/PROJECT_OWNERSHIP.md`, `docs/Literature_Review.md`, `data/raw/PAD_UFES20/`. No license text or SPDX identifier exists anywhere in this repo for this dataset.

### 2. Raw structure
- `data/raw/PAD_UFES20/imgs_part_1/` (911 files) + `imgs_part_2/` (659) + `imgs_part_3/` (728) = **2,298 `.png` images** (e.g. `PAT_100_393_595.png`).
- `metadata.csv`: 2,299 lines (2,298 rows + header), **26 columns**: `patient_id, lesion_id, smoke, drink, background_father, background_mother, age, pesticide, gender, skin_cancer_history, cancer_history, has_piped_water, has_sewage_system, fitspatrick, region, diameter_1, diameter_2, diagnostic, itch, grew, hurt, changed, bleed, elevation, img_id, biopsed`.
- Image mode: 1,440 RGBA / 858 RGB (`reports/PAD_UFES20/05_image_mode_frequency.csv`).

### 3. What audit found
Source: `reports/PAD_UFES20/11_dataset_audit_summary.md` (2026-07-07), cross-checked against `03_corrupted_images.csv` / `04_duplicate_img_id.csv` / `04_orphan_images.csv` / `04_missing_images.csv` (all empty — headers only).
- Corrupted/unreadable images: **0/2,298**. Missing images: **0**. Orphan images: **0**. Duplicate `img_id`: **0**.
- Image size: width min=147 max=3474 mean=933.7 median=780 px (height nearly identical).
- Class distribution (`diagnostic`): BCC 845 (36.77%), ACK 730 (31.77%), NEV 244 (10.62%), SEK 235 (10.23%), SCC 192 (8.36%), MEL 52 (2.26%). **Imbalance ratio 16.2:1** (BCC:MEL).
- Patients: 1,373 unique; lesions: 1,641 unique; 355 patients have multiple lesions; **179 patients have images spanning more than one distinct diagnosis** (relevant to split methodology, §8).
- Missing values (plain `isna()`, `08_missing_value_report.csv`): `background_mother` 822 (35.77%), `background_father` 818 (35.60%); 9 columns tied at 804 (34.99%) — `pesticide, gender, drink, smoke, skin_cancer_history, has_sewage_system, has_piped_water, cancer_history, diameter_1, diameter_2, fitspatrick`. All symptom/identifier columns 0% missing under this raw scan.
- Value-validation report is empty (0 flagged rows) — no implausible age/sex/Fitzpatrick/diameter values found.
- **Caveat:** the audit's `isna()` scan undercounted symptom-column missingness because "not assessed" is encoded as the literal string `"UNK"`, not blank, in the raw CSV. True missingness after cleaning (`data/processed/PAD_UFES20/dataset_description.md:24-33`): `itch` 0.26%, `grew` 17.49%, `hurt` 0.44%, `changed` 17.23%, `bleed` 0.26%, `elevation` 0.09%.

### 4. Cleaning steps applied
Pipeline: `src/data_cleaning/pad_ufes20/c01_column_standardization.py` → `c02_value_validation.py` → `c03_label_standardization.py` → `c04_patient_split.py` → `c05_split_quality_report.py` → `c06_dataset_description.py` (driven by `src/data_cleaning/run_cleaning_pad_ufes20.py`).
- **c01**: renames `gender→sex`, `region→anatomical_site`, `img_id→image_id`, `diagnostic→diagnostic_code`; casts 8 boolean columns (`smoke, drink, pesticide, skin_cancer_history, cancer_history, has_piped_water, has_sewage_system, biopsed`) to pandas nullable `boolean` dtype (with a missingness-count-unchanged assertion); maps the 6 symptom columns' string `"TRUE"/"FALSE"/"UNK"` → `True/False/pd.NA`; uppercases+strips `background_father, background_mother, sex, anatomical_site, diagnostic_code`; adds `dataset_source="PAD_UFES20"` and resolves `image_path` by searching the 3 `imgs_part_*` folders (no images copied — path points back into `data/raw/`). **No rows dropped, no values imputed.**
- **c02**: flags-only validation (age 0-110, sex∈{MALE,FEMALE}, Fitzpatrick 1-6, diameters 0-100mm) — 0 flags raised, nothing altered.
- **c03**: applies the fixed label-mapping dict (§5), raises on any unmapped code.
- **c04**: patient-wise, dominant-diagnosis-stratified split (§8).
- Intermediate output: `data/interim/PAD_UFES20/metadata_standardized.csv`; final outputs in `data/processed/PAD_UFES20/`.

### 5. Label mapping
`data/processed/PAD_UFES20/label_mapping.csv` (also hardcoded in `c03_label_standardization.py`):

| original_label | standardized_label |
|---|---|
| BCC | Basal Cell Carcinoma |
| SCC | Squamous Cell Carcinoma |
| ACK | Actinic Keratosis |
| NEV | Nevus |
| SEK | Seborrheic Keratosis |
| MEL | Melanoma |

### 6. Leakage audit
Two columns excluded (`data/processed/PAD_UFES20/dataset_description.md:80-133`, `feature_whitelist.md:15-28`):
- **`biopsed`** — leakage feature. 100% of malignant cases (BCC 845/845, MEL 52/52, SCC 192/192 = 1,089/1,089) have `biopsed=True`, **zero counter-examples** (verified independently per split: train 0/759, val 0/157, test 0/173). Non-malignant biopsy rate only ~21% (ACK 24.4%, NEV 24.6%, SEK 6.4%). **Phi = 0.80, chi-square = 1474.5, n=2,298.** Motivated by Watson et al. (2026)'s leakage-methodology caution (`docs/Literature_Review.md` row 3). Retained in CSVs for documentation only, never as model input.
- **`diagnostic_code`** — excluded as the raw label source: 1:1 with `disease_label` ("using it as input reads the answer key").

### 7. Feature whitelist
`data/processed/PAD_UFES20/feature_whitelist.md`: 29 total columns in `metadata_train.csv`; **21 allowed**: `smoke, drink, background_father, background_mother, age, pesticide, sex, skin_cancer_history, cancer_history, has_piped_water, has_sewage_system, fitspatrick, anatomical_site, diameter_1, diameter_2, itch, grew, hurt, changed, bleed, elevation`. **Excluded (8):** `patient_id, lesion_id, image_id, image_path, dataset_source, disease_label, diagnostic_code, biopsed`.

### 8. Split methodology
Script: `src/data_cleaning/pad_ufes20/c04_patient_split.py`; seed/ratios in `src/data_cleaning/config.py` (`SPLIT_SEED=42`, ratios 0.70/0.15/0.15).
- **Patient-wise** (not lesion- or image-wise): each patient's *dominant* (most-frequent) `disease_label` determines a stratification group; within each group, `np.random.default_rng(42)` shuffles patient IDs, then patients are greedily assigned to train/val/test by cumulative image count.
- **Verified row counts** (`wc -l` minus header): **train 1,606 / val 338 / test 354** (total 2,298, matches raw). Patient counts: train 948, val 205, test 220 (total 1,373, matches unique patients).
- No overlap confirmed twice: `split_quality_report.csv` and independently in `docs/Dataset_Preparation_Final_Report.md:164`.
- `data/processed/PAD_UFES20/TEST_SPLIT_CONSUMED.json`: test split consumed 2026-07-25 by `evaluate_fairness.py` (12 runs: 4 variants × 3 seeds) — this run also stands as PAD-UFES-20's official final Stage 1 test-set result (single sanctioned test-set use).

### 9. Known limitations/caveats
- Class imbalance ~16.2:1 (BCC/ACK majority vs. MEL minority).
- ~35% missingness across lifestyle/socioeconomic/measurement fields (real, not imputed).
- 179 patients have images spanning >1 diagnosis — since split stratifies by *dominant* diagnosis, a minority diagnosis for those patients may land in a different split than their dominant one (accepted, documented limitation of patient-wise splitting).
- `lesion_id` is unique **only within a patient**, not globally — must be joined on `(patient_id, lesion_id)` together.
- Boolean columns round-trip through CSV as literal `True`/`False`/blank, requiring explicit downstream re-casting on reload.
- Fitzpatrick under-representation is present here too, but framed (via `docs/Literature_Review.md` row 18, Alipour et al. 2024) as a systemic, field-wide issue, not unique to PAD-UFES-20.
- **Positive note:** PAD-UFES-20 has **zero image overlap** with HAM10000 or either ISIC archive — the only one of the 4 datasets unaffected by the cross-dataset leakage problem described in A.2-A.4.

---

## A.2 HAM10000

### 1. Source
- **No HAM10000-specific readme/license/citation file exists in this repo.** `data/raw/HAM10000/` contains only image folders, `HAM10000_metadata.csv`, and 4 out-of-scope Kaggle pixel-matrix CSVs (`reports/HAM10000/11_dataset_audit_summary.md:8`).
- **Acquisition path** (from notebooks, not a repo doc): `notebooks/ham10000_kaggle_notebook.md:30` references the Kaggle mirror `kmader/skin-cancer-mnist-ham10000`. Processed output is separately published as Kaggle dataset `naeemsarkertracer/ham10000-processed` (line 78).
- **No academic dataset citation (e.g. Tschandl et al. 2018) appears anywhere in `docs/` or `README.md`.** The only "Tschandl" mention in the repo is an unrelated 2018 paper in the literature review (`docs/Literature_Review.md:65,87`), noted only because that author also happens to be a HAM10000 co-creator — not a dataset citation.
- **License: NOT FOUND.**

### 2. Raw structure
- `HAM10000_images_part_1/` (5,000 `.jpg`) + `HAM10000_images_part_2/` (5,015 `.jpg`) = **10,015 images**.
- `HAM10000_metadata.csv`: 10,016 lines, header `lesion_id, image_id, dx, dx_type, age, sex, localization`.
- Uniform image size **600×450 px** (no variance).
- `data/processed/HAM10000/` has no image copies — `image_path` resolves back to `data/raw/`.

### 3. What audit found
Source: `reports/HAM10000/11_dataset_audit_summary.md` (2026-07-07).
- Images found = metadata rows = 10,015; no missing/orphan images; **0 duplicate `image_id`**.
- Corrupted/unreadable images: **0/10,015**.
- Missingness not caught by plain `isna()`: `sex` has 57 rows of literal string `"unknown"`; `localization` has 234 such rows.
- Class distribution (`dx`): nv 6,705 (66.95%), mel 1,113 (11.11%), bkl 1,099 (10.97%), bcc 514 (5.13%), akiec 327 (3.27%), vasc 142 (1.42%), df 115 (1.15%). **Imbalance ratio ~58.3:1** (nv:df).
- No patient identifier — `lesion_id` is the only grouping key, not 1:1 with images: 7,470 unique lesions, 1,956 multi-image lesions, **0 lesions with inconsistent `dx` across images**.

### 4. Cleaning steps applied
Pipeline: `src/data_cleaning/ham10000/c01_column_standardization.py` → `c02_value_validation.py` → `c03_label_standardization.py` → `c04_lesion_split.py` → `c05_split_quality_report.py` → `c06_dataset_description.py`.
- **c01**: renames `dx→diagnostic_code`, `dx_type→diagnosis_confirm_type`, `localization→anatomical_site`; strips/lowercases string columns; maps the literal string `"unknown"` in `sex`/`anatomical_site` to true NA (since it was hidden missingness the audit's `isna()` missed); `sex` then uppercased; adds `dataset_source`, `image_path`. **No rows dropped, no values imputed.**
- **c02**: flags implausible age/sex, result: no anomalies flagged.

### 5. Label mapping
`data/processed/HAM10000/label_mapping.csv`:

| original | standardized | note |
|---|---|---|
| bcc | Basal Cell Carcinoma | shared string with PAD-UFES-20 |
| mel | Melanoma | shared string with PAD-UFES-20 |
| nv | Nevus | shared string with PAD-UFES-20 |
| akiec | Actinic Keratosis / Intraepithelial Carcinoma | broader than PAD-UFES-20's ACK — kept distinct |
| bkl | Benign Keratosis-like Lesion | broader than PAD-UFES-20's SEK — kept distinct |
| df | Dermatofibroma | HAM10000-specific |
| vasc | Vascular Lesion | HAM10000-specific |

**Only 3 of 7 classes are shared verbatim strings with PAD-UFES-20's taxonomy** (Basal Cell Carcinoma, Melanoma, Nevus) — this is the exact set used for PAD→HAM cross-dataset generalization scoring (see Part C).

### 6. Leakage audit
- **`diagnosis_confirm_type`** (from `dx_type`): **phi = 0.41, chi2 = 1700.67, n = 10,015.** 100% of malignant images (BCC/MEL, 1,627/1,627) are confirmed via `histo` with zero exceptions; non-malignant images (4,675) split across `histo`/`consensus`/`follow_up`/`confocal` (3,713/4,675 also `histo`). Noted as weaker than PAD-UFES-20's `biopsed` (since `histo` isn't exclusive to malignant cases) but still excluded — the malignant→histo direction is fully deterministic.
- **`diagnostic_code`** excluded as label-source (1:1 with `disease_label`), not a leakage column.

### 7. Feature whitelist
`data/processed/HAM10000/feature_whitelist.md`: **3 allowed** — `age`, `sex`, `anatomical_site`. Excluded (6): `lesion_id`, `image_id`, `image_path`, `dataset_source`, `disease_label`, `diagnostic_code`, `diagnosis_confirm_type` (leakage).

### 8. Split methodology
- **Lesion-wise**, not patient-wise (no patient ID exists). Stratified by each lesion's `disease_label` (unambiguous — 0 inconsistent lesions found), seeded shuffle within label group, greedy assignment to hit ratios.
- **Seed: 42** (`src/data_cleaning/config.py:42`). Ratios: 0.70/0.15/0.15.
- **Verified counts:** train **7,004** rows (5,247 lesions), val **1,501** (1,114 lesions), test **1,510** (1,109 lesions). Sum = 10,015.
- No lesion appears in >1 split.
- `TEST_SPLIT_CONSUMED.json`: test split consumed 2026-07-25 by the PAD→HAM cross-dataset generalization run (4 variants × 3 seeds = 12 runs) — now locked.

### 9. Known limitations/caveats
- Imbalance ~58:1 majority:minority; no patient identifier (only lesion-level leakage is controlled); `sex` missing 57 rows, `anatomical_site` missing 234; no clinical/lifestyle metadata beyond age/sex/site.
- **Dual role**: own Stage 1 baseline dataset AND cross-dataset generalization target for PAD-UFES-20 models (`docs/Phase8_CrossDataset_Generalization_Results.md`) — scored only on the 3 shared classes (1,253/1,510 test rows fall in these 3 classes).
- **Critical cross-dataset caveat** (`docs/Dataset_Preparation_Final_Report.md` §6): HAM10000 is **not independent** from the ISIC archives also used in this project — 98.6% of its images (9,873/10,015) overlap with ISIC Archive 2, 66.5% with ISIC Archive 1. Of the Archive-2 overlap, 4,588 images (46%) are cross-split (e.g. HAM10000-train but Archive-2-test) — a real leakage risk if the datasets were ever pooled naively. Repo policy: never pool HAM10000 with ISIC Archive 1/2 in training/validation; resolved via documented exclusion lists (see Part C, stage 6), not re-splitting.

---

## A.3 ISIC Archive 1

### 1. Source
- **No citation, provenance URL, DOI, or license text found anywhere in the repo.** Checked `README.md`, `docs/PROJECT_PLAN.md`, `docs/Dataset_Preparation_Final_Report.md`, `data/raw/ISIC_Archive_1/` (only `Train/`/`Test/` folders, no readme/license/citation file), and `src/data_cleaning/isic_archive_1/*.py` (no source URL string). **NOT FOUND.**
- `docs/Dataset_Preparation_Final_Report.md:187-201` infers (from image-overlap analysis, not a citation) that it was "bulk-exported from the same underlying ISIC image pool" that HAM10000 and Archive 2 also draw from.

### 2. Raw structure
- `data/raw/ISIC_Archive_1/{Train,Test}/<class-name-folder>/*.jpg` — **folder name = label, no metadata.csv at all.**
- **Test** (9 folders): actinic keratosis 16, basal cell carcinoma 16, dermatofibroma 16, melanoma 16, nevus 16, pigmented benign keratosis 16, seborrheic keratosis 3, squamous cell carcinoma 16, vascular lesion 3 → **128 total**.
- **Train** (9 folders): actinic keratosis 114, basal cell carcinoma 376, dermatofibroma 95, melanoma 438, nevus 357, pigmented benign keratosis 462, seborrheic keratosis 77, squamous cell carcinoma 181, vascular lesion 139 → **2,239 total**.
- **Raw total: 2,357 images**, all `.jpg`.
- `data/processed/ISIC_Archive_1/` has no image copies; `image_path` points back to `data/raw/`.

### 3. What audit found
Source: `reports/ISIC_Archive_1/06_dataset_audit_summary.md` (2026-07-07).
- Corrupted images: **0/2,357**.
- **155 unique filenames filed under two conflicting class-label folders** (systematic, not random): `melanoma↔seborrheic keratosis` 78 images, `actinic keratosis↔nevus` 77 images (78+77=155). The detail file has **310 rows** because each of the 155 filenames appears twice (once per conflicting folder) — 155 unique images × 2 locations = 310. (`data/processed/ISIC_Archive_1/dataset_description.md:12` states "310 images filed under conflicting labels" — same finding, counted as folder-entries rather than unique images.)
- Image size: width 576-6688px, height 450-4479px, mean aspect ratio 1.33; most common resolution 600×450 (1,512/2,357 images).

### 4. Cleaning steps applied
Pipeline: `src/data_cleaning/isic_archive_1/c01_schema_construction.py` (schema *built* rather than standardized, since no raw metadata.csv exists) → `c02_label_standardization.py` → `c03_val_split.py` → `c04_split_quality_report.py` → `c05_dataset_description.py`.
- The 155 conflicting-label images (310 folder-entries) were **excluded**, not guessed.
- No lesion/patient ID exists — each image treated independently.
- The archive's own **Test** split (128 raw / 100 post-exclusion) kept exactly as provided.
- Val carved from **Train only**, 15%, stratified by `disease_label`, seed 42.
- Resulting totals: 2,357 raw − 310 conflict rows = **2,047**, matches exactly.
- No filename appears in more than one split (re-verified independently in `docs/Dataset_Preparation_Final_Report.md:166`).

### 5. Label mapping
`data/processed/ISIC_Archive_1/label_mapping.csv` (9 rows): actinic keratosis→Actinic Keratosis, basal cell carcinoma→Basal Cell Carcinoma, dermatofibroma→Dermatofibroma, melanoma→Melanoma, nevus→Nevus, pigmented benign keratosis→Pigmented Benign Keratosis, seborrheic keratosis→Seborrheic Keratosis, squamous cell carcinoma→Squamous Cell Carcinoma, vascular lesion→Vascular Lesion. Uses the same central `src/data_cleaning/common_label_mapping.py` shared with Archive 2. "Pigmented Benign Keratosis" deliberately kept distinct from "Seborrheic Keratosis."

### 6. Leakage audit
- **`label_conflict_exclusions.csv`** (4 lines, 3 data rows, column `image_id`): `ISIC_0011118`, `ISIC_0011126`, `ISIC_0028619`. **Not** the 155 within-archive conflicts — this is the list of 3 images with **disagreeing labels between Archive 1 and Archive 2** on their 1,673 shared images (e.g. `ISIC_0028619`: Nevus in Archive 1 vs. Actinic Keratosis in Archive 2). Generated by `src/data_cleaning/label_conflict_filter.py`.
- **`external_validation_exclusions.csv`** (1,363 lines = 1,362 `image_id`s + header): images shared with HAM10000. Of 2,047 total, 1,362 overlap with HAM10000, leaving **685** valid for HAM10000-external-validation use. Generated by `src/data_cleaning/cross_dataset_leakage_filter.py`. Applies only when Archive 1 is scored as external validation against a HAM10000-trained model — not to its own internal train/val/test, and not relevant to PAD-UFES-20 (zero overlap there).
- Phase 8 headline eval set is **n=678** (`docs/Phase8_ISIC_External_Validation_Results.md:27`) — this is 685 further narrowed by the union with the label-conflict list and restriction to the 5 classes shared with HAM10000's taxonomy (0 true Basal Cell Carcinoma or Vascular Lesion instances, 7 Dermatofibroma among the 678).

### 7. Feature whitelist
`data/processed/ISIC_Archive_1/feature_whitelist.md`: 6 total columns in `metadata_train.csv` (`image_id, filename, image_path, dataset_source, disease_label, class_label`); **0 allowed as model input** — image-only, no clinical/demographic metadata exists at all. `class_label` retained for documentation only (1:1 with `disease_label`), explicitly flagged "never pass into a model."

### 8. Split methodology
- Archive's own Test kept untouched; val carved from Train only (15%, stratified, seed 42).
- **Verified counts:** train **1,655**, val **292**, test **100** (total **2,047**).
- Split ratio is intentionally "archive-preserving" (80.85/14.26/4.89%), not 70/15/15 like the other 3 datasets, because the archive's own Test set (100 post-exclusion images) is smaller than 15% of the total.
- No grouping key beyond bare `filename` (no lesion/patient ID) — confirmed no filename appears in >1 split.
- Test-split class counts are stark: BCC/Dermatofibroma/Melanoma/Nevus/Pigmented Benign Keratosis/SCC each 16, but Actinic Keratosis only 1, Vascular Lesion 3, Seborrheic Keratosis 0 (all 77 train-side SK images went to train/val).

### 9. Known limitations/caveats
- Small, imbalanced test set (per above).
- No clinical/demographic metadata whatsoever.
- No patient identifier — cannot rule out the same patient contributing to both Train and Test.
- **Not independent from the project's other datasets**: shares 1,362/2,047 images (66.5%) with HAM10000 and 1,673/2,047 (81.7%) with ISIC Archive 2 — all drawn from the same underlying ISIC image pool. 41% of the HAM10000-overlap images land in different splits between the two datasets when split independently.
- In the actual HAM10000→Archive1 zero-shot transfer eval (image-only, n=678, 3 seeds): mean macro-F1 only **0.2421**, much lower than Archive 2's **0.4912** on the identical checkpoints — attributed explicitly to Archive 1's eval-set class-support composition (near-zero support for 2 of the 5 shared classes), not evidence of worse true generalization; the project's own docs state Archive 2's number is "the more representative measurement of this checkpoint's true transfer quality."

---

## A.4 ISIC Archive 2

### 1. Source
- `data/raw/ISIC_Archive_2/attribution.txt`: three contributing institutions — **Hospital Clínic de Barcelona**, **ViDIR Group, Dept. of Dermatology, Medical University of Vienna**, and **Anonymous** — also present as a 3-value `attribution` column in `metadata.csv` (12,302 / 9,873 / 2,901 rows respectively).
- **Provenance**: `docs/Project_Tracking.md:2238` — Kaggle mirror `andrewmvd/isic-2019`, i.e. this archive corresponds to the **ISIC 2019 Challenge training-image pool**. Note: 8.3% of files (2,074/25,076) carry a `_downsampled` suffix on that mirror.
- **License**: `data/raw/ISIC_Archive_2/licenses/CC-0.txt` and `CC-BY-NC.txt` present (CC0 1.0 Universal, CC BY-NC 4.0). Per-row license recorded in `metadata.csv`'s `copyright_license` column — `CC-0` co-occurs exactly with `attribution="Anonymous"` (2,901/2,901 rows).

### 2. Raw structure
- `data/raw/ISIC_Archive_2/{attribution.txt, metadata.csv, licenses/, images/}` — flat folder, no class subfolders.
- **25,331 `.jpg` images** on disk (matches audit).
- `metadata.csv`: 25,332 lines, **27 columns**: `isic_id, attribution, copyright_license, age_approx, anatom_site_1..5, anatom_site_general, anatom_site_special, clin_size_long_diam_mm, concomitant_biopsy, dermoscopic_type, diagnosis_1..5, diagnosis_confirm_type, family_hx_mm, image_type, lesion_id, melanocytic, patient_id, personal_hx_mm, sex`.
- `diagnosis_3` is a real metadata column — the finest-populated diagnosis level (only 1.01% missing), not synthetic — part of ISIC's `diagnosis_1`→`diagnosis_5` coarsest→finest hierarchy.

### 3. What audit found
Source: `reports/ISIC_Archive_2/10_dataset_audit_summary.md` (2026-07-07, "8 modules" confirmed — 8 numbered report sections).
- Corrupted images: **0/25,331**. Missing/orphan images: **0/0**. Duplicate `isic_id`: **0**.
- Image size: width min=576 max=6748 mean=1105.3 median=1024px; height min=450 max=4499 mean=928.8 median=1024px. (Bimodal ~600×450/1024×1024 clustering used separately as visual confirmation of HAM10000 image-source overlap, per `docs/Project_Tracking.md:80`.)
- Class distribution (`diagnosis_3`): Nevus 12,871 (50.81%), Melanoma NOS 4,140 (16.34%), BCC 3,323 (13.12%), Seborrheic Keratosis 1,316 (5.20%), Pigmented Benign Keratosis 1,099 (4.34%), Actinic/Solar Keratosis 867 (3.42%), SCC 628 (2.48%), missing 255 (1.01%), Dermatofibroma 239 (0.94%), Melanoma in situ 211 (0.83%), Solar Lentigo 209 (0.83%), Melanoma Invasive 171 (0.68%), Epidermal Nevus 1, Atypical Melanocytic Neoplasm 1. **Imbalance ratio 61.6:1.**
- `lesion_id` populated 23,664/25,331 (93.4%, 12,264 unique lesions); `patient_id` populated only **417/25,331** (369 unique patients).
- **0 lesions with inconsistent `diagnosis_3` labels.**

### 4. Cleaning steps applied
Pipeline: `src/data_cleaning/isic_archive_2/c01_column_standardization.py` (`isic_id→image_id`, adds `image_path`, `dataset_source`) → `c02_value_validation.py` (no implausible values flagged) → `c03_label_standardization.py` (`diagnosis_3→disease_label` via the shared `common_label_mapping.py`; the 255 rows with no `diagnosis_3` are **excluded**, since coarser fields alone were judged "too coarse") → `c04_lesion_split.py` (group-wise, see §8) → `c05_split_quality_report.py` (re-verifies zero cross-split group overlap).

### 5. Label mapping
`data/processed/ISIC_Archive_2/label_mapping.csv` (13 rows): solar/actinic keratosis→Actinic Keratosis, basal cell carcinoma→Basal Cell Carcinoma, squamous cell carcinoma NOS→Squamous Cell Carcinoma, nevus→Nevus, epidermal nevus→Nevus (n=1), atypical melanocytic neoplasm→Nevus (n=1), seborrheic keratosis→Seborrheic Keratosis, melanoma NOS / melanoma in situ / melanoma invasive → all collapsed to Melanoma, dermatofibroma→Dermatofibroma, pigmented benign keratosis→Pigmented Benign Keratosis (kept distinct from Seborrheic Keratosis), solar lentigo→Solar Lentigo (archive-2-specific). **Result: 9 final classes.**

### 6. Leakage audit
- **`label_conflict_exclusions.csv`** (3 rows, same `ISIC_0011118`/`ISIC_0011126`/`ISIC_0028619` as Archive 1 — the cross-archive disagreement list).
- **`external_validation_exclusions.csv`** (9,874 lines = 9,873 `image_id`s + header): the images shared with HAM10000 (98.6% of HAM10000's total). Applied only when Archive 2 is used as external validation against a HAM10000-trained model.
- Leakage columns excluded from the feature whitelist (§7 below) with quantified evidence:
  - **`diagnosis_confirm_type`**: phi=0.36, chi2=3171.74, n=25,076 — malignant cases 90.7% histopathology-confirmed vs. 55.4% non-malignant, never confirmed via "serial imaging"/"single image consensus."
  - **`concomitant_biopsy`**: identical contingency table to `diagnosis_confirm_type` — duplicate signal.
  - **`melanocytic`**: perfect deterministic split — Melanoma+Nevus = 100% True, all other 7 classes = 100% False, 0 exceptions.
  - Institution-proxy fields (`anatom_site_3/4/5`, `family_hx_mm`, `personal_hx_mm`, `clin_size_long_diam_mm`, `dermoscopic_type`): near-perfectly correlated with which of the 3 contributing institutions supplied the row (e.g. 0.00% present for Hospital Clínic vs. 23-28% for others) — excluded as source-leak risk.
- At evaluation time, Archive 2's exclusion lists are applied as their **union**: 25,076 labeled rows → 12,508 remaining after exclusions + shared-class filtering.

### 7. Feature whitelist
`data/processed/ISIC_Archive_2/feature_whitelist.md`: 30 total columns; **6 allowed**: `age_approx, anatom_site_1, anatom_site_2, anatom_site_general, anatom_site_special, sex`. **Active Stage 1 baseline uses only 4 of the 6** (`age_approx, sex, anatom_site_1, anatom_site_general`) — the other 2 have 52.9%/96.2% missingness and were excluded from Stage 1 by scoping timing, not a leak finding (their institution-crosstab found no leak signal when checked 2026-07-25).

### 8. Split methodology
- **Group-wise**, group = `lesion_id` when present (93.4%), else the row's own `image_id` as a singleton group. Stratified by group's `disease_label` (0 inconsistent groups). Seed 42, ratios 0.70/0.15/0.15.
- **Verified counts** (of 25,076 labeled rows): train **17,535**, val **3,769**, test **3,772**. Group counts: train 9,612, val 2,075, test 2,107 groups (13,794 total).
- Zero cross-split group overlap verified in `split_quality_report.csv`; per-split class % near-identical across splits (e.g. Nevus 51.39/51.21/51.22%).

### 9. Known limitations/caveats
- Severe imbalance (61.6:1); `patient_id` populated for <2% of rows.
- Deep-hierarchy fields >96% missing by design (`anatom_site_4/5`, `family_hx_mm`, `personal_hx_mm`, `clin_size_long_diam_mm`).
- **Massive overlap with HAM10000** (9,873 images, 98.6% of HAM10000) and with Archive 1 (1,673 images, 81.7% of Archive 1) — HAM10000 is essentially a subset of Archive 2's image pool.
- 3 pre-existing label disagreements with Archive 1 on shared images.
- Kaggle-mirror packaging quirks (renamed images folder, `_downsampled` suffix on 8.3% of files) handled in `src/models/config.py`'s `resolve_image_path()`.
- **External validation results** (HAM10000-trained checkpoints, n=12,508, 4 shared classes: BCC, Dermatofibroma, Melanoma, Nevus): image branch mean macro-F1 **0.4912** (23.0% mean spillover); metadata branch **0.2410** (27.3% spillover) — image's advantage over metadata confirmed highly significant via bootstrap (95% CI +0.24 to +0.26, excludes 0).
- Anatomical-site cross-dataset mapping documented in `docs/Phase8_ISIC_Archive2_Anatomical_Site_Mapping.csv` (e.g. "head/neck"→`__MISSING__` for 4,550 rows, genuinely ambiguous against HAM10000's finer site vocabulary).

---

## A.5 Dataset Licenses & Citations (for the thesis References section)

**Added 2026-07-28** to close the gap flagged in the original NOT-FOUND section — none of these citations existed anywhere in this repo's own docs before this. Confidence level stated for each; verify DOIs/license text directly before submission where flagged.

### PAD-UFES-20
- **Citation** (high confidence, confirmed via web search this session): Pacheco, A. G. C., Lima, G. R., Salomão, A. S., Krohling, B., Biral, I. P., de Angelo, G. G., Alves Jr, F. C. R., Esgario, J. G. M., Simora, A. C., Castro, P. B. C., Rodrigues, F. B., Frasson, P. H. L., Krohling, R. A., Knidel, H., Santos, M. C. S., do Espírito Santo, R. B., Macedo, T. L. S. G., Canuto, T. R. P., & de Barros, L. F. S. (2020). **PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones.** *Data in Brief*, 32, 106221. https://doi.org/10.1016/j.dib.2020.106221. Also on arXiv:2007.00478. Hosted at Mendeley Data, DOI 10.17632/zr7vgbcyr2.1 (https://data.mendeley.com/datasets/zr7vgbcyr2/1).
- **License: NOT independently confirmed.** Web search did not surface an explicit license string on the Mendeley Data page itself (only found license info for a *different*, derived preprint dataset, CC-BY-NC-ND 4.0, which is not the same thing and should not be cited as PAD-UFES-20's own license). **Action for user before thesis submission: open `data.mendeley.com/datasets/zr7vgbcyr2/1` directly and copy the exact license shown there.**

### HAM10000
- **Citation** (high confidence, confirmed via web search this session): Tschandl, P., Rosendahl, C., & Kittler, H. (2018). **The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.** *Scientific Data*, 5, 180161. https://doi.org/10.1038/sdata.2018.161.
- **License** (high confidence): **CC BY-NC 4.0** — data deposited at Harvard Dataverse; images/metadata also mirrored on the ISIC Archive. Attribution must reference the Tschandl et al. 2018 data descriptor above.

### ISIC Archive 2
This project's own `data/raw/ISIC_Archive_2/attribution.txt` records 3 contributing sources with row counts 12,302 / 9,873 / 2,901 (Hospital Clínic de Barcelona / ViDIR Group, Medical University of Vienna / Anonymous) — matching the well-documented structure of the **ISIC 2019 Challenge training set**, which is standardly cited as a composite of 3 sources. Per-source citations (medium-high confidence — the 3-way institutional split matches, but exact row-for-row correspondence to each paper's own reported image counts was not independently re-verified against this repo's `metadata.csv`):
- **ViDIR Group (Vienna) portion** — same HAM10000 citation as above (Tschandl et al. 2018), consistent with this repo's own finding that 9,873 of Archive 2's images (98.6% of all of HAM10000) are shared with HAM10000 (`docs/Project_AZ_Reference.md` §A.2/§A.4).
- **Hospital Clínic de Barcelona portion**: Combalia, M., Codella, N. C. F., Rotemberg, V., Helba, B., Vilaplana, V., Reiter, O., Carrera, C., Barreiro, A., Halpern, A. C., Puig, S., & Malvehy, J. (2019). **BCN20000: Dermoscopic Lesions in the Wild.** arXiv:1908.02288.
- **"Anonymous" portion (2,901 rows)** — **NOT independently confirmed.** The ISIC 2019 training set's third standard component is commonly identified in the field as the Memorial Sloan Kettering (MSK) contribution, itself typically cited via the broader ISIC challenge papers (e.g. Codella, N. C. F. et al. (2019). *Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC).* arXiv:1902.03368) — but no source in this repo confirms the "Anonymous" attribution label maps specifically to MSK. **Flagged for the user to verify directly (e.g. via the ISIC Archive collection page for this data) before citing a specific MSK paper for this portion.**
- **License** (high confidence, confirmed from files already in this repo): `data/raw/ISIC_Archive_2/licenses/CC-0.txt` and `CC-BY-NC.txt` — **CC0 1.0 Universal** for rows where `copyright_license="CC-0"` (these are exactly the 2,901 "Anonymous"-attributed rows, confirmed 1:1 in `metadata.csv`), **CC BY-NC 4.0** for the rest. Per-row license is in `metadata.csv`'s `copyright_license` column — cite at the granularity your thesis needs.

### ISIC Archive 1
- **Citation: NOT FOUND, and NOT safely inferable.** No metadata.csv, no readme, no license file exists anywhere under `data/raw/ISIC_Archive_1/` (§A.3). The dataset's 9 class-named folders share the exact same disease taxonomy as ISIC 2019 (Archive 2), and `docs/Dataset_Preparation_Final_Report.md` independently found 1,362/2,047 images (66.5%) overlap with HAM10000 and 1,673/2,047 (81.7%) overlap with Archive 2 — strong circumstantial evidence this is *some* repackaging of the same underlying ISIC image pool, but this repo contains no direct evidence of which specific release/mirror it was repackaged from.
- **Do not guess a Kaggle dataset slug or specific paper for this in the thesis** — that would be a fabricated citation. **Recommended before thesis submission:** either (a) manually trace the archive back to its original download source if that's still recoverable (browser history, download folder metadata, an old email, etc.), or (b) cite it generically as "a repackaging of images from the ISIC Archive (isic-archive.com), exact release unconfirmed" and disclose the overlap-based inference above as the evidentiary basis, or (c) if a specific source truly cannot be recovered, discuss this as a known provenance gap in the Methodology/Limitations chapter rather than presenting a guessed citation as fact.
- **License: NOT FOUND**, for the same reason.

---

# PART B — Models

All 4 variants share the same overall training scaffold (`src/models/train*.py`, hyperparameters in `src/models/config.py`) but differ in architecture, warm-start behavior, and (for image branch) learning rate. **No scheduler is used anywhere** — checked all training scripts, no `torch.optim.lr_scheduler` import exists in the base (non-"improved") variants.

## B.1 Image-Only

### 1. Full architecture
`src/models/image_model.py` — EfficientNet-B0, ImageNet-pretrained, **fully fine-tuned** (no frozen layers — `model.parameters()` passed unfiltered to the optimizer):
```python
def build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features  # 1280
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
```
Only modification to stock torchvision EfficientNet-B0 is replacing the final `Linear` head. Backbone choice rationale (`docs/Project_Tracking.md:91-98`): EfficientNet-B0 (~5.3M params) chosen over ResNet-50 (~25.6M) given PAD-UFES-20's small size (~2,298 images) — lower overfitting risk, smaller footprint for free-tier Kaggle/Colab GPUs, comparable ImageNet top-1 (~77.1% vs ~76.1%).

### 2. Input/output
- Input: `ResizePad(224)` (aspect-preserving resize + zero-pad to square) → train-only `RandomHorizontalFlip`, `RandomRotation(20)`, `ColorJitter(0.15,0.15)` → `ToTensor` + ImageNet `Normalize` → **[B, 3, 224, 224]**.
- Output: logits **[B, num_classes]** — 6 for PAD-UFES-20, 7 for HAM10000 (independent per-dataset heads, no shared taxonomy).

### 3. Training procedure
Adam (plain, not AdamW), **`LEARNING_RATE_IMAGE = 1e-4`**, `weight_decay=1e-4`, class-weighted `CrossEntropyLoss` (inverse train-split frequency), `BATCH_SIZE=32`, `NUM_EPOCHS=30` max, `EARLY_STOPPING_PATIENCE=7` (on val macro-F1), 3 seeds `[0,1,2]`, no scheduler.

### 4. What makes it different
Stage 1 unimodal baseline establishing the visual-only performance level, using a compact CNN chosen specifically to avoid overfitting on a small dataset.

### 5. Warm-start/checkpoint dependencies
**None.** Trains from ImageNet-pretrained weights only — confirmed zero `load_state_dict` calls in `image_model.py`/`train.py`.

### 6. Final verified results
| Split | seed0 | seed1 | seed2 | mean | std |
|---|---|---|---|---|---|
| PAD-UFES-20 val | 0.5529 | 0.5741 | 0.5840 | 0.5703 | 0.0130 |
| PAD-UFES-20 test (official, one-time) | 0.6019 | 0.6382 | 0.6123 | 0.6175 | 0.0153 |
| HAM10000 val | 0.6961 | 0.6882 | 0.6977 | 0.6940 | 0.0041 |
| PAD→HAM cross-dataset (3 shared classes) | 0.4577 | 0.4247 | 0.5149 | 0.4658 | 0.0373 |
| HAM→ISIC Archive 1 (n=678, image-only) | 0.2563 | 0.2275 | 0.2426 | 0.2421 | 0.0118 |
| HAM→ISIC Archive 2 (n=12,508) | 0.5029 | 0.4908 | 0.4799 | 0.4912 | 0.0094 |

Sources: `logs/PAD_UFES20/train_image_seed{N}_summary.json`, `logs/HAM10000/train_image_seed{N}_summary.json`, `docs/Phase8_Fitzpatrick_Fairness_Results.md`, `docs/Phase8_CrossDataset_Generalization_Results.md`, `docs/Phase8_ISIC_External_Validation_Results.md`.

---

## B.2 Metadata-Only

### 1. Full architecture
`src/models/metadata_model.py`:
```python
class MetadataMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.net(x)
```
No activation after the final linear (raw logits). `input_dim` is dataset- and feature-set-dependent, computed at fit time by `MetadataPreprocessor` (`dataset.py:83-163`): numeric columns z-scored on train-split stats (std floored at 1.0 to avoid div-by-zero), categorical columns one-hot encoded against train-split-observed categories only, missing categorical → literal `"__MISSING__"` bucket, unseen categories at val/test → all-zero one-hot row. **PAD-UFES-20's fitted `input_dim` = 89** (3 numeric + one-hot of 18 categorical whitelisted columns, confirmed via checkpoint metadata). **HAM10000's whitelist has only 3 columns** (1 numeric + 2 categorical) → much smaller `input_dim`.

### 2. Input/output
Input: `[B, input_dim]` (89 for full PAD-UFES-20, 3-column-derived for HAM10000/reduced variants). Output: logits `[B, num_classes]`.

### 3. Training procedure
Same scaffold as image-only except **`LEARNING_RATE_METADATA = 1e-3`** (10× higher than image's 1e-4). Same weight decay, loss, batch size, epochs, patience, seeds.

### 4. What makes it different
Explicitly "establishes the metadata-alone performance floor for Phase 6 — not intended to be competitive with the image branch alone" (`metadata_model.py` docstring). The real fusion comparison is deferred to Phase 7.

### 5. Warm-start/checkpoint dependencies
**None** — random PyTorch init, trained from scratch, confirmed via the same `load_state_dict` grep as image-only.

### 6. Final verified results
| Split | seed0 | seed1 | seed2 | mean | std |
|---|---|---|---|---|---|
| PAD-UFES-20 val | 0.5861 | 0.5694 | 0.5732 | 0.5762 | 0.0072 |
| PAD-UFES-20 test (official) | 0.5897 | 0.5975 | 0.6360 | 0.6077 | 0.0202 |
| HAM10000 val (3-feature) | 0.2503 | 0.2657 | 0.2403 | 0.2521 | 0.0104 |
| PAD→HAM cross-dataset | 0.3079 | 0.2787 | 0.2893 | 0.2920 | 0.0121 |
| HAM→ISIC Archive 2 (n=12,508) | 0.2009 | 0.2710 | 0.2510 | 0.2410 | 0.0295 |

HAM10000's metadata branch is much weaker than PAD-UFES-20's, consistent with its 3-column vs. 21-column whitelist. Bootstrap significance: metadata is significantly weaker than image on both cross-dataset transfers tested (PAD→HAM diff +0.1734, 95% CI [0.1341,0.2097], p<0.001; HAM→ISIC-Archive2 diff +0.2502, 95% CI [0.2368,0.2641], p<0.001) — the largest, most significant effect of any bootstrap comparison in the project.

---

## B.3 Late Fusion

### 1. Full architecture
`src/models/fusion_model.py`. Two branches, each a full Stage 1 model with its own-seed Stage 1 checkpoint loaded `strict=True`, stopped short of its final classification layer:
```python
class ImageEmbedder(nn.Module):        # wraps build_efficientnet_b0; forward runs features→avgpool→flatten→dropout, returns [B,1280]
class MetadataEmbedder(nn.Module):     # wraps MetadataMLP, drops final Linear(64,num_classes), returns [B,64]

class FusionModel(nn.Module):
    def __init__(self, metadata_input_dim, num_classes):
        self.image_embedder = ImageEmbedder(num_classes)
        self.metadata_embedder = MetadataEmbedder(metadata_input_dim, num_classes)
        joint_dim = 1280 + 64  # 1344
        self.head = nn.Sequential(nn.Linear(1344,128), nn.BatchNorm1d(128), nn.ReLU(),
                                   nn.Dropout(0.3), nn.Linear(128, num_classes))
    def forward(self, image, metadata):
        joint = torch.cat([self.image_embedder(image), self.metadata_embedder(metadata)], dim=1)
        return self.head(joint)
```
Fusion = plain `torch.cat` at the penultimate-embedding level (1280-d image, 64-d metadata), followed by a two-layer joint head (not a bare linear combination). No attention, no gating.

### 2. Input/output
Image `[B,3,224,224]`, metadata `[B,89]` (full PAD-UFES-20 whitelist) → embeddings `[B,1280]` + `[B,64]` → joint `[B,1344]` → head hidden `[B,128]` → logits `[B,6]`.

### 3. Training procedure
`LEARNING_RATE_FUSION = 1e-5` (much lower than either baseline's LR — deliberately conservative given warm-starting), `weight_decay=1e-4`, class-weighted CE, batch 32, 30 max epochs, patience 7, seeds [0,1,2]. No scheduler.

### 4. What makes it different from the baselines
Per `docs/Project_Tracking.md` "Phase 7 Stage 1" entry: the deeper joint head (vs. a single `Linear(1344,num_classes)`) gives "room to learn a real weighting between modalities instead of the 1280-d image branch mechanically dominating a bare linear combination." Warm-start (not from-scratch) chosen because joint from-scratch training of a 5.3M-param CNN + fusion head is a higher overfitting risk on ~2,298 images, and because it "directly tests the actual thesis question — does fusion improve on the already-optimized unimodal branches."
**Deliberately logged limitation** (not treated as a bug): the 1280:64 dimensionality imbalance is expected to let the image branch numerically dominate even with the deeper head — explicitly stated as the motivation for the cross-attention upgrade (B.4).

### 5. Warm-start/checkpoint dependencies
`FusionModel.load_stage1_checkpoints()` loads seed-matched `image_seed{N}_best.pt` and `metadata_seed{N}_best.pt` via `strict=True`, then **fine-tunes the entire model end-to-end — nothing frozen.**

### 6. Final verified results
| Split | seed0 | seed1 | seed2 | mean | std |
|---|---|---|---|---|---|
| PAD-UFES-20 val | 0.572250 | 0.576023 | 0.571108 | 0.5731 | 0.0021 |
| PAD-UFES-20 test (official) | 0.6261 | 0.6830 | 0.6606 | 0.6566 | 0.0234 |
| PAD→HAM cross-dataset (`fusion_reduced`) | 0.4521 | 0.4715 | 0.4557 | 0.4597 | 0.0084 |

`fusion_reduced` (3-feature schema-matched variant for cross-dataset use) val: 0.5734/0.5835/0.5945, mean≈0.5838. **ISIC external-validation results do not exist for this model, by scope, not by a failed attempt**: the ISIC external-validation experiment (Part A.3/A.4, Part C stage 9) is defined specifically as "HAM10000-trained checkpoints evaluated on the ISIC archives" — late fusion was never trained on HAM10000 at all (Phase 7 was scoped to PAD-UFES-20 only, per `docs/Project_Tracking.md`'s Phase 7 entries), so it was never a candidate for that evaluation in the first place, not a model that was run against ISIC and underperformed.

---

## B.4 Cross-Attention Fusion

### 1. Full architecture
`src/models/cross_attention_fusion_model.py`:
```python
class SpatialImageEmbedder(nn.Module):
    def forward(self, x):
        x = self.backbone.features(x)           # [B,1280,7,7]
        return x.flatten(2).transpose(1,2)        # [B,49,1280] - 49 conv-grid spatial tokens

class MetadataChannelGate(nn.Module):              # ON by default (use_channel_gate=True)
    def __init__(self, metadata_dim=64, num_channels=1280):
        self.gate = nn.Linear(64, 1280)
    def forward(self, image_tokens, metadata_embedding):
        channel_scale = torch.sigmoid(self.gate(metadata_embedding))
        return image_tokens * channel_scale.unsqueeze(1)

class CrossAttentionFusionModel(nn.Module):
    def __init__(self, metadata_input_dim, num_classes, d_model=256, num_heads=8, dropout=0.3):
        self.image_embedder = SpatialImageEmbedder(num_classes)
        self.metadata_embedder = MetadataEmbedder(metadata_input_dim, num_classes)  # reused from fusion_model.py
        self.channel_gate = MetadataChannelGate(64, 1280)
        self.query_proj = nn.Linear(64, 256)     # metadata -> Q
        self.kv_proj = nn.Linear(1280, 256)      # image tokens -> K,V
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1, batch_first=True)
        joint_dim = 256 + 64  # 320
        self.head = nn.Sequential(nn.Linear(320,128), nn.BatchNorm1d(128), nn.ReLU(),
                                   nn.Dropout(0.3), nn.Linear(128, num_classes))
    def forward(self, image, metadata):
        image_tokens = self.image_embedder(image)               # [B,49,1280]
        metadata_embedding = self.metadata_embedder(metadata)   # [B,64]
        image_tokens = self.channel_gate(image_tokens, metadata_embedding)  # if enabled
        query = self.query_proj(metadata_embedding).unsqueeze(1)  # [B,1,256]
        key_value = self.kv_proj(image_tokens)                     # [B,49,256]
        attended, _ = self.attention(query, key_value, key_value)  # [B,1,256]
        joint = torch.cat([attended.squeeze(1), metadata_embedding], dim=1)  # [B,320]
        return self.head(joint)
```
**Mechanism confirmed directly in code**: metadata = Query, image's 49 spatial tokens = Key/Value, **one** `nn.MultiheadAttention` layer (not stacked), `num_heads=8`, `d_model=256` (head_dim=32). Image encoder = EfficientNet-B0's 7×7 conv feature map (not a ViT patch embedder) — "spatial tokens" = the 49 conv-grid positions. Also includes an **optional metadata-conditioned channel gate before attention** (on by default, described in code comments as "TG-CAVNet-inspired") — this is a dual-mechanism design (channel gate + spatial cross-attention), not pure cross-attention alone.
**Explicitly not a MetaBlock reproduction**: MetaBlock is `sigmoid(tanh(V·t1)+t2)`, uniform across spatial positions; this model computes genuine per-spatial-location attention weights, which channel gating structurally cannot do.

### 2. Input/output
Image → `[B,49,1280]`; metadata `[B,89]` → `[B,64]`; after optional gate `[B,49,1280]` (reweighted); Q `[B,1,256]`, K/V `[B,49,256]`; attention output `[B,1,256]`→squeeze→`[B,256]`; joint (attended ⊕ raw metadata) `[B,320]`; head hidden `[B,128]`; logits `[B,6]`.

### 3. Training procedure
Same scaffold as late fusion: `LEARNING_RATE_CROSS_ATTENTION = 1e-5` (numerically identical to fusion's LR, same "conservative warm-start" rationale), `weight_decay=1e-4`, class-weighted CE, batch 32, 30 max epochs, patience 7, seeds [0,1,2], no scheduler.

An **"improved" variant** (`train_cross_attention_improved.py`) additionally stacked: `FocalLoss(gamma=2.0, alpha=class_weights, label_smoothing=0.1)` instead of plain CE, `WeightedRandomSampler` instead of shuffle, stronger augmentation (`RandomRotation(30)` up from 20, plus `RandomAffine`), and `CosineAnnealingLR(T_max=30, eta_min=1e-7)`.

### 4. What makes it different from late fusion
Per `docs/Project_Tracking.md` "Phase 7 Stage 2 Proposal — APPROVED" (2026-07-18): directly addresses late fusion's diagnosed limitation — its 1280:64 raw-dimension concatenation let the image branch numerically dominate. Here, both modalities are projected into a shared `d_model` **before** any interaction, so raw dimension counts no longer mechanically bias the result. Compared explicitly against MetaBlock before approval; correct framing per project history is "cross-attention, contrasted with MetaBlock's channel-gating approach," never "MetaBlock-inspired."

### 5. Warm-start/checkpoint dependencies
Same discipline as late fusion: loads `image_seed{N}_best.pt` + `metadata_seed{N}_best.pt` strict-mode, then fine-tunes **everything end-to-end including the new query/kv projections, attention, channel gate, and head** — nothing frozen. Rationale: "both embedders are Stage 1-converged and only the new cross-attention/head parameters are randomly initialized."

**Reduced-feature variant**: a separate checkpoint set (`cross_attention_reduced_seed{N}_best.pt`, `metadata_input_dim=16`) built for PAD→HAM cross-dataset generalization. The image embedder still warm-starts from the **unchanged original** `image_seed{N}_best.pt`; the metadata embedder warm-starts from a new `metadata_reduced_seed{N}_best.pt` trained on only 3 schema-matched columns (age, sex, anatomical_site, the latter renormalized into HAM10000's vocabulary). These two checkpoint sets (full vs. reduced) are never mixed and saved under distinct filenames. **The official Fitzpatrick fairness / test-set result uses the original full-feature checkpoints, not the reduced ones.**

**Negative result — "improved" variant confirmed NOT adopted**: trained 3 seeds, scored 0.4804/0.5447/0.5028 (mean 0.509), well below the original's 0.6209±0.0143. **These numbers are user-reported and unverified from file** — independently confirmed **no checkpoint or summary JSON exists anywhere in the repo for `cross_attention_improved`** (only `train_fusion_*`, `train_cross_attention_*`, `*_reduced_*` files exist), so this specific mean/per-seed breakdown cannot be re-derived from a logged artifact and should be cited with that caveat every time it's mentioned (this matches `docs/Project_Tracking.md`'s own framing, lines 1502/1520-1526). **Decision: the improved variant was never adopted regardless of the exact numbers. The final cross-attention model used for all of Phase 8 is the original architecture** (`cross_attention_seed{0,1,2}_best.pt`).

### 6. Final verified results
| Split | seed0 | seed1 | seed2 | mean | std |
|---|---|---|---|---|---|
| PAD-UFES-20 val | 0.604949 | 0.618178 | 0.639662 | 0.6209 | 0.0143 |
| PAD-UFES-20 test (official) | 0.6862 | 0.6721 | 0.7349 | 0.6977 | 0.0269 |
| PAD→HAM cross-dataset (`cross_attention_reduced`) | 0.4916 | 0.4442 | 0.4604 | 0.4654 | 0.0197 |

Cross-attention's minimum val seed (0.6049) exceeds every other variant's maximum val seed (image 0.5840, metadata 0.5861, fusion 0.5760) — zero overlap across the 4-way comparison. `cross_attention_reduced` val (schema-matched, for the cross-dataset run): seed0=0.5921, seed1=0.6777, seed2=0.6650, mean=0.6449.

Per-Fitzpatrick-group macro-F1 (test split, mean of 3 seeds): group 1 (n=22) 0.3307, group 2 (n=120) 0.6212, group 3 (n=59) 0.5706, group 4 (n=15) 0.4365 — best or tied-best of all 4 architectures in every reportable group (groups 5/6, darkest skin, unreportable: n=2 and n=1).

Bootstrap significance on PAD→HAM (paired, 1000 resamples): cross_attention vs. image diff −0.0004 (not significant); vs. metadata +0.1734 (significant, p<0.001); vs. late_fusion +0.0057 (not significant) — cross-attention, image, and late-fusion are statistically indistinguishable on this specific transfer task even though cross-attention leads within-dataset. **ISIC external-validation results do not exist for this model, by scope, not by a failed attempt** — same reason as late fusion above: the ISIC external-validation experiment only evaluates HAM10000-trained checkpoints, and cross-attention (like late fusion) was never trained on HAM10000, so it was never in scope for that specific experiment, not a model that was tried and underperformed there.

---

# PART C — Full Pipeline Map

**Note on the roadmap vs. this pipeline:** `docs/Project_Tracking.md`'s 10-phase "Project Roadmap" is coarser than the 11 stages below — e.g. its Phase 4 "Dataset Preparation" fans out into pipeline stages 2-6 here (nested as per-dataset sub-rows), and its Phase 8 "Experiments & Evaluation" absorbs pipeline stages 8-11 (evaluation, cross-dataset validation, fairness, significance testing) under one label. Phases 9-10 (Thesis Writing, arXiv preprint) have no technical-pipeline counterpart.

1. **Raw data acquisition** → `data/raw/{HAM10000,PAD_UFES20,ISIC_Archive_1,ISIC_Archive_2}/`. **No acquisition/download script exists in this repo** — acquisition was manual; Kaggle "Add Data" mirror slugs are referenced in notebook headers but no local fetch script exists (`NOT FOUND — checked src/, scripts/, docs/Dataset_Strategy.md`).

2. **Audit** → `src/data_audit/<dataset>/m0N_*.py` (common helpers in `src/data_audit/common/`). PAD-UFES-20: 12 modules; HAM10000: 11 modules (same pattern, lesion- not patient-statistics); ISIC Archive 1: 4 modules (folder-derived, no metadata.csv); ISIC Archive 2: 8 modules. Each dataset's final module (`m11`/`m11`/`m04`/`m08`) rolls all prior modules into one markdown summary report under `reports/<dataset>/`.

3. **Cleaning** → `src/data_cleaning/<dataset>/c0N_*.py`. Common pattern: column standardization → value validation → label standardization → split → split-quality report → dataset description doc. (ISIC Archive 1 has only 5 steps, no separate value-validation step since it has no metadata.csv to validate.)

4. **Splitting** → the `c0{3,4}_*_split.py` script per dataset, all reading `SPLIT_SEED = 42` from `src/data_cleaning/config.py:42`, all using `np.random.default_rng(SPLIT_SEED)`. PAD-UFES-20 = patient-wise; HAM10000 & ISIC Archive 2 = lesion-wise/group-wise (Archive 2 uses `lesion_id.fillna(image_id)` for the 1,667 rows lacking a lesion_id); ISIC Archive 1 = archive's own Test kept untouched, val carved from Train only (no grouping key exists).

5. **Feature whitelist generation** → **manual/hand-authored, not script-generated.** Grepped all of `src/` for writers of `feature_whitelist.md` — none found; each dataset's file is a hand-authored markdown doc (the phi/chi-square numbers cited in each were presumably computed ad hoc during a documented "review and approve" process, `docs/Project_Tracking.md:236`, but no reusable script produces the artifact).

6. **Cross-dataset leakage fix** → `src/data_cleaning/cross_dataset_leakage_filter.py` (generates `external_validation_exclusions.csv` for both ISIC archives, by computing `image_id` overlap against HAM10000's full image set) + `src/data_cleaning/label_conflict_filter.py` (generates `label_conflict_exclusions.csv`, the 3 images with disagreeing labels between the two ISIC archives). Docstring of the leakage-filter script cites "**Fix (b)**: restrict external-validation claims rather than re-splitting globally (fix a)" — fix (a), a global re-split, was rejected (`docs/Project_Tracking.md:288-313`) because it would discard already-verified individual dataset splits to solve a problem specific only to the external-validation use case.

7. **Training** → `src/models/train*.py`. **Verified source-of-truth relationship for later notebooks**: `scripts/generate_cross_attention_kaggle_notebook.py` (+ 3 sibling generators) literally embed the real `.py` files into Kaggle-notebook `%%writefile` cells, so those notebooks cannot silently drift from source. However, `notebooks/ham10000_kaggle_notebook.md` and `notebooks/pad_ufes20_fusion_kaggle_notebook.md` have **no** corresponding generator script — these are earlier, hand-maintained notebooks. Distinct entry points: `train.py` (image+metadata, both datasets, shared recipe by design), `train_fusion.py` and `train_cross_attention_fusion.py` (PAD-UFES-20 only), `train_cross_attention_improved.py` (abandoned, see B.4), plus `train_metadata_reduced.py`/`train_fusion_reduced.py`/`train_cross_attention_fusion_reduced.py` (3-column schema-matched variants for the PAD→HAM experiment).

8. **Evaluation** → `src/evaluation/evaluate.py` — main entrypoint for all 4 variants, either split, both datasets. The documented `--confirm-final` guard is real (`if args.split=="test" and not args.confirm_final: raise SystemExit(...)`). **Confirmed gap**: `evaluate_fairness.py` bypassed this flag entirely on 2026-07-25 by calling `evaluate.py`'s helper functions directly rather than its CLI — this led to `src/evaluation/test_split_guard.py` being added the same day, a dataset-scoped marker file (`TEST_SPLIT_CONSUMED.json`) now enforced across `evaluate.py`, `evaluate_fairness.py`, and `evaluate_cross_dataset.py`.

9. **Cross-dataset validation** → `src/evaluation/evaluate_cross_dataset.py` (PAD-UFES-20-trained models → HAM10000 test split, 3 shared classes, writes `reports/PAD_UFES20/cross_dataset/`) and `src/evaluation/evaluate_external_isic.py` (HAM10000-trained models → both ISIC archives, applying both exclusion lists, writes `reports/HAM10000/external_isic/`).

10. **Fairness testing** → `src/evaluation/evaluate_fairness.py`. Runs all 4 variants × 3 seeds on PAD-UFES-20's test split only (Fitzpatrick doesn't exist in HAM10000). Confirmed in code: `N_RESAMPLES=1000`, `RNG_SEED=42`, `np.random.default_rng(RNG_SEED)`, percentile-method 95% CI. Small-sample rule confirmed: n<15 excluded, 15≤n<30 flagged, n≥30 clean.

11. **Significance testing** → `src/evaluation/bootstrap_significance.py` (PAD→HAM cross-dataset comparison, paired row-level resampling of `evaluate_cross_dataset.py`'s predictions) and `src/evaluation/bootstrap_significance_isic.py` (HAM→ISIC Archive 2 image-vs-metadata comparison only — Archive 1 has no metadata to compare). Both: 1,000 resamples, seed 42, paired (same resampled row-index set applied to both variants per iteration), per-iteration macro-F1 averaged across 3 seeds, two-sided p-value, significance = 95% CI excludes 0.

---

## Items flagged NOT FOUND (do not assume/guess these in a viva without checking further)

- **RESOLVED 2026-07-28 (see §A.5 "Dataset Licenses & Citations"):** citations and licenses were externally researched (not present in this repo's own docs, but now verified via web search and added to §A.5) for PAD-UFES-20, HAM10000, and ISIC Archive 2. **Still genuinely unresolved: ISIC Archive 1** — no citation or license was found, and §A.5 explicitly recommends against guessing one; **PAD-UFES-20's exact license text also still needs the user to open the Mendeley Data page directly**, since web search could not surface it. This repo's own docs still contain no license/citation text for any of the 4 datasets — §A.5's citations came from external verification, not from anything discovered in-repo.
- **No raw-data acquisition/download script exists** anywhere in the repo (Part C, stage 1) — all 4 datasets' presence in `data/raw/` is undocumented beyond "acquired."
- **PAD-UFES-20's `MetadataMLP` exact fitted `input_dim` (89) was confirmed via checkpoint metadata**, not hardcoded in `config.py` — if this number is ever needed for a schema not yet checked, re-verify against the actual checkpoint rather than assuming 89 generalizes.
