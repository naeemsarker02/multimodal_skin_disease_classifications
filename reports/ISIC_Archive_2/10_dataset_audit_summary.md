# ISIC Archive 2 Dataset Audit Summary

Audit date: 2026-07-07 23:39:56

## 1. Folder Structure & Image Inventory

- Flat `images/` folder, single `metadata.csv` (27 columns), no pre-existing split.
- Total images found on disk: 25331
- See: `01_folder_structure.csv`, `02_image_inventory.csv`

## 2. Image Verification & Corrupted Images

- Images successfully decoded (OK): 25331
- Corrupted / unreadable images: 0
- See: `03_image_verification.csv`, `03_corrupted_images.csv`

## 3. Missing / Orphan Images

- Metadata rows: 25331
- Inventory files: 25331
- Missing images (in metadata, not on disk): 0
- Orphan images (on disk, not in metadata): 0
- Duplicate isic_id entries in metadata: 0
- See: `05_missing_images.csv`, `05_orphan_images.csv`, `05_duplicate_isic_id.csv`

## 4. Image Size Statistics

- Width: min=576 max=6748 mean=1105.3 median=1024 px
- Height: min=450 max=4499 mean=928.8 median=1024 px
- See: `04_image_size_stats.csv`, `04_resolution_frequency.csv`, `figures/image_size_distribution.png`

## 5-6. Metadata Analysis, Column Descriptions, Missing Values

- 27 columns, several forming a specificity hierarchy (`diagnosis_1`->`diagnosis_5`, `anatom_site_1`->`anatom_site_5`) that gets sparser at deeper levels by design, not by defect.
- See: `06_column_description.csv`, `06_metadata_numeric_describe.csv`, `07_missing_value_report.csv`

## 7. Class Distribution

`diagnosis_3` is used as the primary class label (populated for nearly every row, and names concrete diseases comparable to the other three datasets in this project).

| diagnosis_3 | count | pct |
|---|---|---|
| Nevus | 12871 | 50.81% |
| Melanoma, NOS | 4140 | 16.34% |
| Basal cell carcinoma | 3323 | 13.12% |
| Seborrheic keratosis | 1316 | 5.2% |
| Pigmented benign keratosis | 1099 | 4.34% |
| Solar or actinic keratosis | 867 | 3.42% |
| Squamous cell carcinoma, NOS | 628 | 2.48% |
| nan | 255 | 1.01% |
| Dermatofibroma | 239 | 0.94% |
| Melanoma in situ | 211 | 0.83% |
| Solar lentigo | 209 | 0.83% |
| Melanoma Invasive | 171 | 0.68% |
| Epidermal nevus | 1 | 0.0% |
| Atypical melanocytic neoplasm | 1 | 0.0% |

- See: `08_class_distribution_coarse.csv`, `08_class_distribution_fine.csv`, `figures/class_distribution.png`

## 8. Lesion / Patient Statistics

- `lesion_id` populated for 23664/25331 rows (12264 unique lesions); `patient_id` populated for only 417/25331 rows (369 unique patients).
- Unlike PAD-UFES-20, patient-wise splitting is not possible for the bulk of this archive; the cleaning phase must split by `lesion_id` where available, and treat rows with neither identifier as their own singleton group (as with HAM10000's lesion-wise approach).
- Lesions with inconsistent `diagnosis_3` labels: 0
- See: `09_lesion_patient_statistics.csv`, `09_lesions_multiple_diagnoses.csv`

## Notes

- This audit is READ-ONLY. `data/raw/` was not modified.
- This archive overlaps substantially in disease taxonomy with ISIC Archive 1's class-folder labels and with PAD-UFES-20/HAM10000; label harmonization across all four sources happens in the cleaning phase (`label_mapping.csv`).