# PAD-UFES-20 Dataset Audit Summary

Audit date: 2026-07-07 19:51:14

## 1. Folder Structure

- Folders analyzed: 4
- See: `01_folder_structure.csv`

## 2. Image Inventory

- Total images found on disk: 2298
- See: `02_image_inventory.csv`

## 3. Image Verification & Corrupted Images

- Images successfully decoded (OK): 2298
- Corrupted / unreadable images: 0
- See: `03_image_verification.csv`, `03_corrupted_images.csv`

## 4. Missing / Orphan Images

- Metadata rows: 2298
- Inventory files: 2298
- Missing images (in metadata, not on disk): 0
- Orphan images (on disk, not in metadata): 0
- Duplicate img_id entries in metadata: 0
- See: `04_missing_images.csv`, `04_orphan_images.csv`, `04_duplicate_img_id.csv`

## 5. Image Size Statistics

- Width: min=147 max=3474 mean=933.7 median=780 px
- Height: min=147 max=3476 mean=933.6 median=780 px
- See: `05_image_size_stats.csv`, `05_resolution_frequency.csv`, `figures/image_size_distribution.png`

## 6-9. Metadata Analysis, Column Descriptions, Missing Values

- See: `06_metadata_overview.csv`, `07_column_description.csv`, `08_missing_value_report.csv`

## 10. Class Distribution (diagnostic)

| diagnostic | count | pct |
|---|---|---|
| BCC | 845 | 36.77% |
| ACK | 730 | 31.77% |
| NEV | 244 | 10.62% |
| SEK | 235 | 10.23% |
| SCC | 192 | 8.36% |
| MEL | 52 | 2.26% |

- See: `09_class_distribution.csv`, `figures/class_distribution.png`

## 11. Patient Statistics

- Unique patients: 1373
- Unique lesions: 1641
- Patients with multiple lesions: 355
- Patients with multiple distinct diagnoses: 179
- See: `10_patient_statistics.csv`, `10_patients_multiple_diagnoses.csv`

## Notes

- This audit is READ-ONLY. `data/raw/` was not modified.
- No cleaning or processed dataset creation was performed in this phase.
- Findings above should inform the upcoming cleaning and standardization phase.