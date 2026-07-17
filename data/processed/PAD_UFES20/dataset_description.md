# PAD-UFES-20 Processed Dataset Description

Generated: 2026-07-07 20:27:16

## Source

Raw data: `data/raw/PAD_UFES20/` (untouched, read-only).
Images are **not copied**. `image_path` in the metadata CSVs points back to the original file under `data/raw/PAD_UFES20/imgs_part_*/` (disk space constraint - see project decision log).

## Schema

All original PAD-UFES-20 columns are retained. Renamed for the unified cross-dataset schema: `gender`->`sex`, `region`->`anatomical_site`, `img_id`->`image_id`, `diagnostic`->`diagnostic_code`. Added: `image_path`, `dataset_source`, `disease_label`.

The README/Dataset_Strategy conceptual groupings map to these concrete columns:
- **symptoms**: itch, grew, hurt, changed, bleed, elevation
- **clinical_features**: smoke, drink, pesticide, skin_cancer_history, cancer_history, has_piped_water, has_sewage_system, fitspatrick, background_father, background_mother, diameter_1, diameter_2, biopsed

## Label Standardization

See `label_mapping.csv` for the diagnostic code -> disease_label mapping.

## Missing Values

No values were imputed or invented. For most columns, missingness percentages match the dataset audit exactly (see `reports/PAD_UFES20/08_missing_value_report.csv`).

One exception: the audit measured missingness with `isna()`, which could not detect that the raw symptom columns (itch, grew, hurt, changed, bleed, elevation) encode "not assessed" as the literal string `UNK` rather than a blank value. During cleaning, `UNK` was mapped to a proper missing value (no information was invented - an existing "unknown" marker was simply standardized). Actual missingness for these columns is therefore higher than the audit reported:

- `itch`: 0.26% missing
- `grew`: 17.49% missing
- `hurt`: 0.44% missing
- `changed`: 17.23% missing
- `bleed`: 0.26% missing
- `elevation`: 0.09% missing

## Value Validation

No implausible values were flagged.

## Train / Validation / Test Split

- Method: patient-wise split, stratified by each patient's dominant `disease_label`.
- Target ratios: {'train': 0.7, 'val': 0.15, 'test': 0.15}
- Random seed: 42 (fixed, for reproducibility)
- Total images: 2298 | Total patients: 1373

- **train**: 1606 images, 948 patients
- **val**: 338 images, 205 patients
- **test**: 354 images, 220 patients

No patient appears in more than one split (verified in `split_quality_report.csv`).

## `lesion_id` Uniqueness

`lesion_id` is unique **per patient only**, not globally - two different patients can share the same `lesion_id` value. Always group or join on `(patient_id, lesion_id)` together, never on `lesion_id` alone.

## Recommended Metadata Loading Strategy

Boolean columns are written to CSV as `True` / `False` / empty. `pandas.read_csv` parses these as plain Python `bool`/`NaN` objects but does not assign the nullable `boolean` dtype automatically, and which columns come back as `bool` vs `object` can vary by file depending on whether that file happens to contain any missing values. Downstream code should cast explicitly after loading:

```python
boolean_columns = [
    "smoke",
    "drink",
    "pesticide",
    "skin_cancer_history",
    "cancer_history",
    "has_piped_water",
    "has_sewage_system",
    "biopsed",
    "itch",
    "grew",
    "hurt",
    "changed",
    "bleed",
    "elevation",
]
df[boolean_columns] = df[boolean_columns].astype("boolean")
```

## `biopsed` Leakage/Shortcut Audit (2026-07-08)

Per Watson et al. (2026)'s finding (cited in `docs/Project_Tracking.md`) that
clinically-derived-after-diagnosis fields can let a model shortcut
classification, `biopsed` ("whether the diagnosis was confirmed via biopsy
(True) or clinical judgment only (False)") was checked against
`disease_label` across all 2,298 images (train+val+test combined).

**Contingency table (biopsy rate by disease):**

| disease_label | biopsed=False | biopsed=True | biopsy rate |
|---|---|---|---|
| Basal Cell Carcinoma | 0 | 845 | 100.0% |
| Melanoma | 0 | 52 | 100.0% |
| Squamous Cell Carcinoma | 0 | 192 | 100.0% |
| Actinic Keratosis | 552 | 178 | 24.4% |
| Nevus | 184 | 60 | 24.6% |
| Seborrheic Keratosis | 220 | 15 | 6.4% |

Collapsing to malignant (BCC, Melanoma, SCC) vs. non-malignant (Actinic
Keratosis, Nevus, Seborrheic Keratosis):

| | biopsed=False | biopsed=True |
|---|---|---|
| Non-malignant | 956 | 253 |
| Malignant | 0 | 1,089 |

**Every single malignant-labeled image in this dataset has `biopsed=True`**
(0 counter-examples), while non-malignant images are biopsied only 21% of
the time. This holds independently within each split (train: 0/759 malignant
have `biopsed=False`; val: 0/157; test: 0/173 — verified separately, not
just in the pooled total). Association strength: phi coefficient = 0.80,
chi-square = 1474.5 (n=2,298) — a very strong association bordering on
deterministic for the malignant class.

**Interpretation:** this is not a data error — it reflects real clinical
practice (biopsy is near-mandatory to confirm a cancer diagnosis, while
benign lesions are frequently diagnosed on clinical inspection alone). But
it means `biopsed` is a **near-perfect proxy for the malignant/non-malignant
distinction** this project's disease taxonomy ultimately encodes. A model
given `biopsed` as an input feature could achieve high apparent accuracy on
the malignant-vs-not distinction by reading this single flag, without using
the image or any genuine pre-diagnosis symptom/clinical signal — this would
not reflect real diagnostic capability and would invalidate the model's
scientific value for the thesis.

**Recommendation: exclude `biopsed` from model input entirely.** Keep it in
`metadata_{train,val,test}.csv` as a retained column for dataset
documentation/analysis purposes only (e.g., to describe cohort
characteristics or to sanity-check that a trained model isn't implicitly
recovering it), but never pass it into any image, metadata, or fusion model
as a training feature. This decision is also recorded in
`docs/Project_Tracking.md`.

## Known Limitations

- Class imbalance: BCC/ACK are far more frequent than MEL (~16:1 majority:minority in the full dataset).
- ~35% missingness in lifestyle/socioeconomic/measurement fields (smoke, drink, pesticide, cancer history, piped water, sewage system, diameters, Fitzpatrick).
- 179 patients have images spanning more than one diagnosis; the split uses each patient's dominant diagnosis for stratification, so minority diagnoses for those patients may land in a different split than their dominant one.

## Generated Files

- `metadata_train.csv`, `metadata_val.csv`, `metadata_test.csv`
- `label_mapping.csv`
- `patient_split_assignment.csv`
- `split_quality_report.csv`