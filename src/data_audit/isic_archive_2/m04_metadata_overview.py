"""Module 4: Metadata analysis + column descriptions.

Structural profile of metadata.csv (27 columns) combined with the
meaning of each column, since with this many mostly-sparse columns
(diagnosis_1..5, anatom_site_1..5 form a hierarchy of increasing
specificity where deeper levels are populated for fewer rows) a single
combined data dictionary is more useful than two separate reports.
"""

import pandas as pd

from src.data_audit.config import ISIC2_METADATA_FILE, ISIC2_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv

NUMERIC_COLUMNS = ["age_approx", "clin_size_long_diam_mm"]

COLUMN_DESCRIPTIONS = {
    "isic_id": "Unique image identifier; corresponding file is f'{isic_id}.jpg' under images/.",
    "attribution": "Contributing institution/source.",
    "copyright_license": "Image usage license (e.g. CC-0, CC-BY-NC).",
    "age_approx": "Patient age in years, rounded/approximated.",
    "anatom_site_1": "Anatomical site, coarsest level (e.g. Trunk, Head/neck).",
    "anatom_site_2": "Anatomical site, level 2 (more specific than anatom_site_1).",
    "anatom_site_3": "Anatomical site, level 3.",
    "anatom_site_4": "Anatomical site, level 4 (rarely populated).",
    "anatom_site_5": "Anatomical site, level 5 (rarely populated).",
    "anatom_site_general": "Anatomical site, ISIC's standard general category.",
    "anatom_site_special": "Anatomical site special-case flag (e.g. acral, mucosal).",
    "clin_size_long_diam_mm": "Clinically measured longest lesion diameter, in mm.",
    "concomitant_biopsy": "Whether a concomitant biopsy was taken (boolean).",
    "dermoscopic_type": "Dermoscopy imaging sub-type, when recorded.",
    "diagnosis_1": "Coarsest diagnosis level: Benign / Malignant / Indeterminate.",
    "diagnosis_2": "Diagnosis level 2: broad pathological category (e.g. 'Benign melanocytic proliferations').",
    "diagnosis_3": "Diagnosis level 3: specific disease name (e.g. Nevus, Melanoma NOS, Basal cell carcinoma) - the finest level populated for nearly all rows; used as the primary class label for this archive.",
    "diagnosis_4": "Diagnosis level 4: disease sub-type detail (mostly populated only for nevi/melanoma).",
    "diagnosis_5": "Diagnosis level 5: finest sub-type detail (rarely populated).",
    "diagnosis_confirm_type": "How the diagnosis was confirmed (histopathology, serial imaging, expert consensus, confocal microscopy).",
    "family_hx_mm": "Family history of melanoma (boolean/blank).",
    "image_type": "Imaging modality; constant 'dermoscopic' for this archive.",
    "lesion_id": "Lesion identifier grouping multiple images of the same lesion; NOT populated for every row.",
    "melanocytic": "Whether the lesion is of melanocytic origin (boolean).",
    "patient_id": "Patient identifier; populated for only a small subset of rows.",
    "personal_hx_mm": "Personal history of melanoma (boolean/blank).",
    "sex": "Patient sex.",
}


def run(logger) -> pd.DataFrame:
    metadata = pd.read_csv(ISIC2_METADATA_FILE)
    n_rows = len(metadata)

    rows = []
    for col in metadata.columns:
        n_missing = metadata[col].isna().sum()
        sample_values = metadata[col].dropna().unique()[:5]
        rows.append(
            {
                "column": col,
                "description": COLUMN_DESCRIPTIONS.get(col, "UNKNOWN - not documented, verify with source."),
                "dtype": str(metadata[col].dtype),
                "n_unique": metadata[col].nunique(dropna=True),
                "missing_count": n_missing,
                "missing_pct": round(100 * n_missing / n_rows, 2),
                "sample_values": ", ".join(str(v) for v in sample_values),
            }
        )
    df = pd.DataFrame(rows)

    save_csv(df, ISIC2_REPORTS_DIR / "06_column_description.csv")

    numeric_desc = metadata[NUMERIC_COLUMNS].describe().transpose().reset_index().rename(columns={"index": "column"})
    save_csv(numeric_desc, ISIC2_REPORTS_DIR / "06_metadata_numeric_describe.csv")

    n_duplicate_rows = metadata.duplicated().sum()

    logger.info("Metadata shape: %d rows x %d columns", *metadata.shape)
    logger.info("Fully duplicate rows: %d", n_duplicate_rows)
    logger.info("Column data dictionary built for %d columns", len(df))
    logger.info("Saved -> %s", ISIC2_REPORTS_DIR / "06_column_description.csv")

    return df
