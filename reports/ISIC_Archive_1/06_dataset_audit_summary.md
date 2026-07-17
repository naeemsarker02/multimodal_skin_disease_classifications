# ISIC Archive 1 Dataset Audit Summary

Audit date: 2026-07-07 23:01:48

## Structure

No metadata.csv ships with this archive - it is pre-split into `Train/` and `Test/`, each with one subfolder per class; the folder name itself is the label. Folder structure + image inventory were built in one pass (`01_folder_structure.csv`, `02_image_inventory.csv`).
- Total images found on disk: 2357
- Classes: ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']

## Duplicate Filenames Across Locations

- 155 filenames appear in more than one location. See `02_duplicate_filenames.csv`.
- **Data-quality finding:** 155 of those are the *same image* filed under two *different* class labels - a genuine label conflict, not merely a Train/Test split repeat. The conflicts are systematic, not random noise:

  - `melanoma` <-> `seborrheic keratosis`: 78 images
  - `actinic keratosis` <-> `nevus`: 77 images

  See `02_duplicate_filename_label_conflicts.csv` for the full row-level detail. These images should be excluded or resolved (not silently kept under either label) during the cleaning phase, since their true class is ambiguous.

## Image Verification & Corrupted Images

- Images successfully decoded (OK): 2357
- Corrupted / unreadable images: 0
- See: `03_image_verification.csv`, `03_corrupted_images.csv`

## Image Size Statistics

- See: `04_image_size_stats.csv`, `04_resolution_frequency.csv`, `figures/image_size_distribution.png`

## Class Distribution

| split | class_label | count | pct_of_split |
|---|---|---|---|
| Test | actinic keratosis | 16 | 13.56% |
| Test | basal cell carcinoma | 16 | 13.56% |
| Test | dermatofibroma | 16 | 13.56% |
| Test | melanoma | 16 | 13.56% |
| Test | nevus | 16 | 13.56% |
| Test | pigmented benign keratosis | 16 | 13.56% |
| Test | squamous cell carcinoma | 16 | 13.56% |
| Test | seborrheic keratosis | 3 | 2.54% |
| Test | vascular lesion | 3 | 2.54% |
| Train | pigmented benign keratosis | 462 | 20.63% |
| Train | melanoma | 438 | 19.56% |
| Train | basal cell carcinoma | 376 | 16.79% |
| Train | nevus | 357 | 15.94% |
| Train | squamous cell carcinoma | 181 | 8.08% |
| Train | vascular lesion | 139 | 6.21% |
| Train | actinic keratosis | 114 | 5.09% |
| Train | dermatofibroma | 95 | 4.24% |
| Train | seborrheic keratosis | 77 | 3.44% |

- See: `05_class_distribution.csv`, `figures/class_distribution.png`

## Notes

- This audit is READ-ONLY. `data/raw/` was not modified.
- No patient or lesion identifier exists in this archive - only a bare `ISIC_xxxxxxx` filename per image, so no patient/lesion leakage check is possible here. The provided Train/Test split is used as-is; the cleaning phase further carves a validation set out of Train.
- Class names overlap heavily with ISIC Archive 2's `diagnosis_3` field and with PAD-UFES-20/HAM10000's disease taxonomy - label harmonization across all four sources is handled in the cleaning phase (`label_mapping.csv`).