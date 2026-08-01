"""Builds the PAD-UFES-20-Expanded variant per the approved Step 2 plan
(docs/Project_Tracking.md, "Step 2 Integration Plan - FINAL APPROVAL",
2026-07-29).

Adds DERM12345's melanoma-family (400) + squamous_cell_carcinoma (266)
images and MED-NODE's melanoma (70) images to PAD-UFES-20's two
bottleneck classes - image-branch training data only, never for joint
image+metadata training (neither external source has clinical metadata
compatible with PAD-UFES-20's 21-feature whitelist).

Never touches data/processed/PAD_UFES20/ (read-only source). Never adds
new rows to VAL or TEST (approved mitigation for the dermoscopic/
macroscopic modality-confound risk - see Project_Tracking.md).

Outputs (data/processed/PAD_UFES20_Expanded/):
    metadata_train.csv                  - byte-identical copy of the
                                           original PAD-UFES-20 train
                                           split. Safe for metadata/fusion
                                           branch training.
    metadata_train_image_only.csv       - metadata_train.csv rows PLUS
                                           the new DERM12345/MED-NODE
                                           rows. Image branch ONLY - new
                                           rows have no usable clinical
                                           metadata (all-NaN).
    metadata_val.csv, metadata_test.csv - byte-identical copies of the
                                           original PAD-UFES-20 val/test
                                           splits. Never modified, never
                                           expanded.
    label_mapping.csv                   - the DERM12345/MED-NODE ->
                                           PAD-UFES-20 class mapping
                                           actually applied.
    feature_whitelist.md, dataset_description.md
"""

from pathlib import Path

import pandas as pd

from src.data_audit.common.io_utils import save_csv, save_text
from src.data_cleaning.config import PAD_UFES20_PROCESSED_DIR, PROCESSED_ROOT, PROJECT_ROOT

OUT_DIR = PROCESSED_ROOT / "PAD_UFES20_Expanded"

DERM12345_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "DERM12345"
DERM12345_TRAIN_TAB = DERM12345_RAW_DIR / "derm12345_metadata_train.tab"
DERM12345_TEST_TAB = DERM12345_RAW_DIR / "derm12345_metadata_test.tab"
DERM12345_IMAGES_DIR = DERM12345_RAW_DIR / "images"

MEDNODE_MELANOMA_DIR = (
    PROJECT_ROOT / "data" / "raw" / "MED-NODE" / "complete_mednode_dataset" / "melanoma"
)

# Authoritative label -> PAD-UFES-20 class mapping, per the approved
# Step 2 plan (Melanoma/SCC only; bowen_disease and cutaneous_horn
# excluded as medically ambiguous, not guessed).
DERM12345_LABEL_TO_DISEASE = {
    "mel": "Melanoma",
    "lm": "Melanoma",
    "lmm": "Melanoma",
    "alm": "Melanoma",
    "anm": "Melanoma",
    "scc": "Squamous Cell Carcinoma",
}
DERM12345_LABEL_TO_CODE = {
    "mel": "MEL", "lm": "MEL", "lmm": "MEL", "alm": "MEL", "anm": "MEL",
    "scc": "SCC",
}

# train1/train2 zip split, discovered by inspecting the remote zips
# (src/data_cleaning/pad_ufes20_expanded README note - see dataset_description.md).
TRAIN1_LABELS = {"alm", "anm"}


def _load_derm12345_rows(logger) -> pd.DataFrame:
    df = pd.concat(
        [
            pd.read_csv(DERM12345_TRAIN_TAB, sep="\t", encoding="utf-8-sig"),
            pd.read_csv(DERM12345_TEST_TAB, sep="\t", encoding="utf-8-sig"),
        ],
        ignore_index=True,
    )
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip('"')

    sub = df[df["label"].isin(DERM12345_LABEL_TO_CODE)].copy()
    logger.info("DERM12345: %d rows match the approved Melanoma/SCC label set", len(sub))

    def _image_path(row):
        # Stored relative to PROJECT_ROOT as "data/raw/DERM12345/...",
        # matching every other dataset's image_path convention exactly -
        # src/models/config.py's resolve_image_path() requires this exact
        # "data/raw/<Dataset>/..." shape (it raises ValueError otherwise,
        # both locally and on Kaggle) and looks up dataset_dir == "DERM12345"
        # in KAGGLE_DATASET_SLUGS when running on Kaggle.
        abs_path = DERM12345_IMAGES_DIR / row["label"] / f"{row['image_id']}.jpg"
        rel_path = abs_path.relative_to(PROJECT_ROOT)
        return rel_path.as_posix()

    sub["image_path"] = sub.apply(_image_path, axis=1)
    n_missing = sum(
        1 for p in sub["image_path"] if not (PROJECT_ROOT / p).exists()
    )
    if n_missing:
        logger.warning("DERM12345: %d expected image files not found on disk", n_missing)

    out = pd.DataFrame({
        "patient_id": sub["patient_id"],
        "lesion_id": pd.NA,
        "image_id": sub["image_id"],
        "diagnostic_code": sub["label"].map(DERM12345_LABEL_TO_CODE),
        "biopsed": True,  # all malignant DERM12345 labels are biopsy-confirmed per the source paper
        "dataset_source": "DERM12345",
        "image_path": sub["image_path"],
        "disease_label": sub["label"].map(DERM12345_LABEL_TO_DISEASE),
    })
    return out


def _load_mednode_rows(logger) -> pd.DataFrame:
    files = sorted(MEDNODE_MELANOMA_DIR.glob("*.jpg"))
    logger.info("MED-NODE: %d melanoma images found on disk", len(files))
    out = pd.DataFrame({
        "patient_id": [f"MEDNODE_{f.stem}" for f in files],
        "lesion_id": pd.NA,
        "image_id": [f.name for f in files],
        "diagnostic_code": "MEL",
        "biopsed": pd.NA,  # not independently confirmed - see MED-NODE caveat in Project_Tracking.md
        "dataset_source": "MED-NODE",
        "image_path": [f.relative_to(PROJECT_ROOT).as_posix() for f in files],
        "disease_label": "Melanoma",
    })
    return out


def run(logger) -> None:
    original_train = pd.read_csv(PAD_UFES20_PROCESSED_DIR / "metadata_train.csv")
    original_val = pd.read_csv(PAD_UFES20_PROCESSED_DIR / "metadata_val.csv")
    original_test = pd.read_csv(PAD_UFES20_PROCESSED_DIR / "metadata_test.csv")

    derm_rows = _load_derm12345_rows(logger)
    mednode_rows = _load_mednode_rows(logger)
    new_rows = pd.concat([derm_rows, mednode_rows], ignore_index=True)

    # New rows have no clinical whitelist features at all (age, sex, itch,
    # grew, ... - the 21 columns in feature_whitelist.md) - left absent
    # entirely (NaN on concat), not imputed, so it's structurally obvious
    # (and enforced by the two-file split below) that these rows can only
    # ever feed the image branch.
    combined_train = pd.concat([original_train, new_rows], ignore_index=True)

    save_csv(original_train, OUT_DIR / "metadata_train.csv")
    save_csv(original_val, OUT_DIR / "metadata_val.csv")
    save_csv(original_test, OUT_DIR / "metadata_test.csv")
    save_csv(combined_train, OUT_DIR / "metadata_train_image_only.csv")

    logger.info(
        "metadata_train.csv (original, multimodal-safe): %d rows", len(original_train)
    )
    logger.info(
        "metadata_train_image_only.csv (expanded, IMAGE BRANCH ONLY): %d rows "
        "(%d original + %d new)",
        len(combined_train), len(original_train), len(new_rows),
    )
    logger.info("metadata_val.csv: %d rows (untouched)", len(original_val))
    logger.info("metadata_test.csv: %d rows (untouched)", len(original_test))

    by_class = combined_train["disease_label"].value_counts()
    logger.info("Expanded TRAIN (image-only file) class counts:\n%s", by_class.to_string())

    mapping_df = pd.DataFrame(
        [
            {"source": "DERM12345", "source_label": k, "diagnostic_code": DERM12345_LABEL_TO_CODE[k],
             "disease_label": v}
            for k, v in DERM12345_LABEL_TO_DISEASE.items()
        ]
        + [{"source": "MED-NODE", "source_label": "melanoma", "diagnostic_code": "MEL",
            "disease_label": "Melanoma"}]
    )
    save_csv(mapping_df, OUT_DIR / "label_mapping.csv")
    logger.info("Saved -> %s", OUT_DIR / "label_mapping.csv")

    whitelist_md = f"""# PAD-UFES-20-Expanded — Model-Input Feature Whitelist

Generated by `src/data_cleaning/pad_ufes20_expanded/c01_build_expanded_dataset.py`.

**Identical to `data/processed/PAD_UFES20/feature_whitelist.md`'s 21
allowed clinical features** for any row originating from the original
PAD-UFES-20 dataset (`dataset_source == "PAD_UFES20"`):
`smoke`, `drink`, `background_father`, `background_mother`, `age`,
`pesticide`, `sex`, `skin_cancer_history`, `cancer_history`,
`has_piped_water`, `has_sewage_system`, `fitspatrick`, `anatomical_site`,
`diameter_1`, `diameter_2`, `itch`, `grew`, `hurt`, `changed`, `bleed`,
`elevation`.

**Rows from DERM12345 / MED-NODE (`dataset_source` in `{{"DERM12345",
"MED-NODE"}}`) have none of the above columns populated (NaN) and MUST
NEVER be used for metadata-branch or fusion (image+metadata) training.**
They exist only in `metadata_train_image_only.csv`, never in
`metadata_train.csv` — use `metadata_train.csv` for any metadata/fusion
model, `metadata_train_image_only.csv` only for image-only backbone
training (Phase 8B Step 3 / Step 3a).

## Additional excluded columns (all sources)

| Column | Reason |
|---|---|
| `patient_id`, `lesion_id`, `image_id`, `image_path` | identifiers / paths |
| `dataset_source` | **not just an identifier here — actively a leakage risk.** Since this expanded train file mixes image sources for only 2 of 6 classes (Melanoma/SCC get DERM12345's dermoscopic images; all other classes stay pure PAD-UFES-20 macroscopic), `dataset_source` (or anything correlated with it, like image modality/texture) is disproportionately predictive of the malignant-priority classes for reasons unrelated to real lesion morphology. Never use as a feature. See `Project_Tracking.md`, "Step 2 — Dataset Integration Plan" (2026-07-29), point 3. |
| `disease_label` | the prediction target itself |
| `diagnostic_code` | label-source (same treatment as the original PAD-UFES-20 whitelist) |
| `biopsed` | leakage feature, same reasoning as the original PAD-UFES-20 whitelist; DERM12345 rows are set `True` (per the source paper's biopsy-confirmation of all malignant labels), MED-NODE rows are left `NaN` (not independently confirmed — see the MED-NODE caveat in `Project_Tracking.md`, disclosed in the thesis Limitations section) |

Retained in the CSV for documentation/audit purposes only. Never pass any
of the excluded columns into a model.
"""
    save_text(whitelist_md, OUT_DIR / "feature_whitelist.md")
    logger.info("Saved -> %s", OUT_DIR / "feature_whitelist.md")

    description_md = f"""# PAD-UFES-20-Expanded Processed Dataset Description

Generated by `src/data_cleaning/pad_ufes20_expanded/c01_build_expanded_dataset.py`.
Full plan and approval trail: `docs/Project_Tracking.md`, "Step 2 —
Dataset Integration Plan" and "Step 2 Integration Plan — FINAL APPROVAL"
(2026-07-29).

## Source

`data/processed/PAD_UFES20/` (original, **read-only, never modified**) +
DERM12345 (Harvard Dataverse, `doi:10.7910/DVN/DAXZ7P`, CC BY 4.0) +
MED-NODE (`cs.rug.nl/~imaging/databases/melanoma_naevi/`, CC BY 4.0).

DERM12345/MED-NODE images are stored under `data/raw/DERM12345/` and
`data/raw/MED-NODE/` respectively (new raw source folders, read-only,
same no-image-copying convention as the project's other 4 datasets —
`image_path` points back to these locations, nothing is duplicated into
`PAD_UFES20_Expanded/`). DERM12345's images were pulled via targeted
HTTP range requests (only the {len(new_rows) - len(mednode_rows)}
Melanoma/SCC-mapped images extracted, not the full 12,345-image
archive) — its own metadata files
(`derm12345_metadata_{{train,test}}.tab`) are kept alongside as the
permanent record of the full dataset's labels, even though only a
subset of the images were downloaded.

## Class mapping applied (`label_mapping.csv`)

DERM12345 `main_class_1 == "melanoma"` (all 5 sub_classes: `melanoma`,
`lentigo_maligna`, `acral_nodular`, `acral_lentiginious`,
`lentigo_maligna_melanoma`) → **Melanoma** (400 images). DERM12345
`sub_class == "squamous_cell_carcinoma"` → **Squamous Cell Carcinoma**
(266 images). MED-NODE `melanoma` folder → **Melanoma** (70 images).

**Excluded, not guessed:** DERM12345's `bowen_disease` (n=37) and
`cutaneous_horn` (n=12) — medically ambiguous mappings to PAD-UFES-20's
SCC/ACK classes, excluded per the project's established
exclusion-over-guessing precedent (same treatment as the anatomical-site
mapping work). **Restricted to Melanoma/SCC only** — DERM12345 has other
classes with clean mappings (e.g. `basal_cell_carcinoma` n=423,
`seborrheic_keratosis` n=607) that were deliberately NOT pulled in this
pass, to keep scope tightly isolated to the two identified bottleneck
classes (user decision, 2026-07-29) — a candidate for a future,
separately-evaluated addition, not bundled here.

## Train / Validation / Test Split

- **`metadata_val.csv`, `metadata_test.csv`: byte-for-byte copies of
  the original PAD-UFES-20 val/test splits. Never touched, never
  expanded.** This is what keeps a valid paired-bootstrap comparison
  possible against the locked 0.6977 test result.
- **`metadata_train.csv`: byte-for-byte copy of the original PAD-UFES-20
  train split.** Use this file for any metadata-only or fusion
  (image+metadata) model — every row has real clinical metadata.
- **`metadata_train_image_only.csv`: `metadata_train.csv` rows PLUS all
  new DERM12345/MED-NODE rows.** Use this file ONLY for image-branch
  training (backbone comparison, Step 3a ablations, and as pretraining
  for the fusion model's image embedder before it warm-starts and
  fine-tunes on `metadata_train.csv`). New rows have no clinical
  metadata (all whitelist columns NaN) — they must never reach a
  metadata-consuming branch. See `feature_whitelist.md`.

## Known risk, disclosed not hidden

Melanoma/SCC are the only two classes with mixed-modality training
images post-expansion (dermoscopic DERM12345 + macroscopic PAD-UFES-20
for those two classes; every other class stays pure macroscopic). This
is a documented, deliberate tradeoff (see `feature_whitelist.md`'s
`dataset_source` entry and `Project_Tracking.md`'s Step 2 plan, point 3)
— val/test staying pure-macroscopic-PAD-UFES-20 is the chosen mitigation,
not a fix for the underlying training-time confound.

## Class counts, `metadata_train_image_only.csv`

{by_class.to_string()}
"""
    save_text(description_md, OUT_DIR / "dataset_description.md")
    logger.info("Saved -> %s", OUT_DIR / "dataset_description.md")
