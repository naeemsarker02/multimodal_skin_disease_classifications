# HAM10000 — Model-Input Feature Whitelist

Generated: 2026-07-08. This is the single reference for feature selection in
any future model (image-only, metadata-only, or fusion). `metadata_train.csv`
has 10 total columns; 3 are allowed as model input.

## Allowed features (3)

`age`, `sex`, `anatomical_site`

## Excluded columns and reasons

| Column | Reason |
|---|---|
| `lesion_id` | identifier |
| `image_id` | identifier |
| `image_path` | path, not a feature |
| `dataset_source` | constant tag within this dataset's files, not discriminative |
| `disease_label` | the prediction target itself |
| `diagnostic_code` | label-source — verified 1:1 with `disease_label` (all 7 dx codes: akiec/bcc/bkl/df/mel/nv/vasc each map to exactly one `disease_label`, zero ambiguity). Using it as input reads the answer key. |
| `diagnosis_confirm_type` | leakage feature — every malignant image (Basal Cell Carcinoma/Melanoma, 1,627/1,627) is confirmed via `histo`, zero exceptions, while non-malignant images are confirmed via `histo`/`consensus`/`follow_up`/`confocal` in a mixed 4,675-way split. Formally verified: phi=0.41, chi2=1700.67 (n=10,015). Weaker than PAD-UFES-20's `biopsed` (phi=0.80) since `histo` is also used for many non-malignant cases (3,713/4,675), but the malignant→histo direction is fully deterministic — a real, if partial, shortcut risk. |

Retained in the CSV for documentation/audit purposes only: `diagnostic_code`, `diagnosis_confirm_type`. Never pass either into a model.
