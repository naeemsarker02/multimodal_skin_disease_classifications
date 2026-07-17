# HAM10000 Processed Dataset Description

Generated: 2026-07-07 22:57:19

## Source

Raw data: `data/raw/HAM10000/` (untouched, read-only).
Images are **not copied**. `image_path` in the metadata CSVs points back to the original file under `data/raw/HAM10000/HAM10000_images_part_*/` (disk space constraint - consistent with PAD-UFES-20). The Kaggle-provided `hmnist_*.csv` pixel-matrix files are out of scope and untouched.

## Schema

All original HAM10000 columns are retained. Renamed for the unified cross-dataset schema: `dx`->`diagnostic_code`, `dx_type`->`diagnosis_confirm_type`, `localization`->`anatomical_site`. Added: `image_path`, `dataset_source`, `disease_label`.

**No patient identifier exists in this dataset** - `lesion_id` is the only grouping key, and it is not 1:1 with images (7,470 unique lesions across 10,015 images).

## Label Standardization

See `label_mapping.csv` for the dx code -> disease_label mapping. Labels for diseases shared with PAD-UFES-20 (Basal Cell Carcinoma, Melanoma, Nevus) use identical standardized names for cross-dataset consistency; HAM10000-specific categories (Actinic Keratosis / Intraepithelial Carcinoma, Benign Keratosis-like Lesion, Dermatofibroma, Vascular Lesion) are kept distinct rather than force-mapped onto PAD-UFES-20's narrower codes.

## Missing Values

No values were imputed or invented. For most columns, missingness matches the dataset audit exactly (see `reports/HAM10000/08_missing_value_report.csv`).

One exception: the audit measured missingness with `isna()`, which could not detect that the raw `sex` and `localization` columns encode "not assessed" as the literal string `unknown` rather than a blank value. During cleaning, `unknown` was mapped to a proper missing value (no information was invented). Actual missingness for these columns:

- `sex`: 0.57% missing
- `anatomical_site`: 2.34% missing

## Value Validation

No implausible values were flagged.

## Train / Validation / Test Split

- Method: **lesion-wise** split (not patient-wise - no patient identifier exists), stratified by each lesion's `disease_label`. The audit confirmed 0 lesions have inconsistent dx labels across their images, so stratification is exact (no dominant-label approximation needed, unlike PAD-UFES-20).
- Target ratios: {'train': 0.7, 'val': 0.15, 'test': 0.15}
- Random seed: 42 (fixed, for reproducibility)
- Total images: 10015 | Total lesions: 7470

- **train**: 7004 images, 5247 lesions
- **val**: 1501 images, 1114 lesions
- **test**: 1510 images, 1109 lesions

No lesion_id appears in more than one split (verified in `split_quality_report.csv`).

## Known Limitations

- Class imbalance: `nv` (Nevus) is ~67% of the dataset; `df` (Dermatofibroma) is ~1.15%, a ~58:1 majority:minority ratio.
- No patient identifier - cannot rule out the same patient contributing lesions to more than one split (only lesion-level leakage is controlled for).
- `sex` missing for 57 rows, `anatomical_site` missing for 234 rows (both encoded as `unknown` in the raw data).
- No clinical/lifestyle metadata (unlike PAD-UFES-20) - only age, sex, anatomical_site, and diagnosis confirmation method are available.

## Generated Files

- `metadata_train.csv`, `metadata_val.csv`, `metadata_test.csv`
- `label_mapping.csv`
- `lesion_split_assignment.csv`
- `split_quality_report.csv`