# ISIC Archive 1 — Model-Input Feature Whitelist

Generated: 2026-07-08. This is the single reference for feature selection in
any future model. `metadata_train.csv` has 6 total columns; **0 are allowed
as model input** — this archive carries no clinical/demographic metadata at
all (image + class label only, per `dataset_description.md`'s Known
Limitations). Any model using this archive is necessarily image-only.

## Allowed features (0)

None.

## Excluded columns and reasons

| Column | Reason |
|---|---|
| `image_id` | identifier |
| `filename` | identifier (redundant with `image_id` — same value plus `.jpg`) |
| `image_path` | path, not a feature |
| `dataset_source` | constant tag within this dataset's files, not discriminative |
| `disease_label` | the prediction target itself |
| `class_label` | label-source — verified 1:1 with `disease_label` (all 9 folder-derived class labels each map to exactly one `disease_label`, zero ambiguity — `disease_label` is a direct rename of this column). Using it as input reads the answer key. |

Retained in the CSV for documentation/audit purposes only: `class_label`. Never pass it into a model.
