# HAM10000 Dataset Audit Summary

Audit date: 2026-07-07 21:19:40

## 1. Folder Structure

- Folders analyzed: 3
- Raw folder also contains 4 hmnist_*.csv files (Kaggle-provided precomputed pixel matrices) which are **out of scope** for this pipeline and were not processed.
- See: `01_folder_structure.csv`

## 2. Image Inventory

- Total images found on disk: 10015
- See: `02_image_inventory.csv`

## 3. Image Verification & Corrupted Images

- Images successfully decoded (OK): 10015
- Corrupted / unreadable images: 0
- See: `03_image_verification.csv`, `03_corrupted_images.csv`

## 4. Missing / Orphan Images

- Metadata rows: 10015
- Inventory files: 10015
- Missing images (in metadata, not on disk): 0
- Orphan images (on disk, not in metadata): 0
- Duplicate image_id entries in metadata: 0
- See: `04_missing_images.csv`, `04_orphan_images.csv`, `04_duplicate_image_id.csv`

## 5. Image Size Statistics

- Width: min=600 max=600 mean=600.0 median=600 px
- Height: min=450 max=450 mean=450.0 median=450 px
- See: `05_image_size_stats.csv`, `05_resolution_frequency.csv`, `figures/image_size_distribution.png`

## 6-9. Metadata Analysis, Column Descriptions, Missing Values

- See: `06_metadata_overview.csv`, `07_column_description.csv`, `08_missing_value_report.csv`

- **Important:** `isna()` alone under-reports missingness. The following columns use the literal string `"unknown"` instead of a blank value:
  - `sex`: 57 rows
  - `localization`: 234 rows

## 10. Class Distribution (dx)

| dx | count | pct |
|---|---|---|
| nv | 6705 | 66.95% |
| mel | 1113 | 11.11% |
| bkl | 1099 | 10.97% |
| bcc | 514 | 5.13% |
| akiec | 327 | 3.27% |
| vasc | 142 | 1.42% |
| df | 115 | 1.15% |

- See: `09_class_distribution.csv`, `figures/class_distribution.png`

## 11. Lesion Statistics

- HAM10000 has **no patient identifier** - `lesion_id` is the only grouping key available, and it is not 1:1 with images.
- Unique lesions: 7470
- Lesions with multiple images: 1956
- Lesions with inconsistent (multiple) dx labels: 0
- See: `10_lesion_statistics.csv`, `10_lesions_multiple_diagnoses.csv`

## Notes

- This audit is READ-ONLY. `data/raw/` was not modified.
- No cleaning or processed dataset creation was performed in this phase.
- Because there is no patient identifier, the upcoming cleaning phase must split by `lesion_id` (lesion-wise), not patient-wise.