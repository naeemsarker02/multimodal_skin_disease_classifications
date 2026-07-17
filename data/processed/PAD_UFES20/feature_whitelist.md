# PAD-UFES-20 — Model-Input Feature Whitelist

Generated: 2026-07-08. This is the single reference for feature selection in
any future model (image-only, metadata-only, or fusion). `metadata_train.csv`
has 29 total columns; 21 are allowed as model input.

## Allowed features (21)

`smoke`, `drink`, `background_father`, `background_mother`, `age`,
`pesticide`, `sex`, `skin_cancer_history`, `cancer_history`,
`has_piped_water`, `has_sewage_system`, `fitspatrick`, `anatomical_site`,
`diameter_1`, `diameter_2`, `itch`, `grew`, `hurt`, `changed`, `bleed`,
`elevation`

## Excluded columns and reasons

| Column | Reason |
|---|---|
| `patient_id` | identifier |
| `lesion_id` | identifier (also not globally unique — see `dataset_description.md`) |
| `image_id` | identifier |
| `image_path` | path, not a feature |
| `dataset_source` | constant tag within this dataset's files, not discriminative |
| `disease_label` | the prediction target itself |
| `diagnostic_code` | label-source — verified 1:1 with `disease_label` (ACK/BCC/MEL/NEV/SCC/SEK each map to exactly one `disease_label`, zero ambiguity). Using it as input reads the answer key. |
| `biopsed` | leakage feature — 100% of malignant cases (1,089/1,089) have `biopsed=True` with zero exceptions vs. 21% for non-malignant; phi=0.80, chi2=1474.5 (n=2,298). See `dataset_description.md` and `Project_Tracking.md` for full audit. |

Retained in the CSV for documentation/audit purposes only: `diagnostic_code`, `biopsed`. Never pass either into a model.
