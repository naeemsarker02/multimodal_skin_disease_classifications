"""Module 8: Metadata column descriptions.

Combines computed statistics (dtype, % missing, unique count, sample
values) with the clinical/domain meaning of each column. The meanings
below are drawn from the published PAD-UFES-20 dataset documentation
(Pacheco et al., 2020) - this is domain knowledge that cannot be
inferred from the CSV values alone, so it is recorded explicitly here
rather than guessed at.
"""

import pandas as pd

from src.data_audit.config import PAD_UFES20_METADATA_FILE, PAD_UFES20_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv

COLUMN_DESCRIPTIONS = {
    "patient_id": "Unique identifier for the patient.",
    "lesion_id": "Unique identifier for the lesion (a patient may have multiple lesions).",
    "smoke": "Whether the patient is a smoker (boolean, self-reported).",
    "drink": "Whether the patient consumes alcohol (boolean, self-reported).",
    "background_father": "Father's ethnic/regional background (self-reported).",
    "background_mother": "Mother's ethnic/regional background (self-reported).",
    "age": "Patient age in years.",
    "pesticide": "Whether the patient has occupational/residential pesticide exposure (boolean).",
    "gender": "Patient gender (MALE/FEMALE).",
    "skin_cancer_history": "Whether the patient has a personal history of skin cancer (boolean).",
    "cancer_history": "Whether the patient has a personal history of any cancer (boolean).",
    "has_piped_water": "Household has piped water access (socioeconomic indicator, boolean).",
    "has_sewage_system": "Household has a sewage system (socioeconomic indicator, boolean).",
    "fitspatrick": "Fitzpatrick skin type scale (1-6; higher = darker skin, lower UV sensitivity).",
    "region": "Anatomical location of the lesion on the body.",
    "diameter_1": "Lesion diameter measurement, axis 1, in millimeters.",
    "diameter_2": "Lesion diameter measurement, axis 2, in millimeters.",
    "diagnostic": "Diagnostic label / target class (e.g. BCC, MEL, SCC, ACK, NEV, SEK).",
    "itch": "Whether the lesion itches (boolean symptom).",
    "grew": "Whether the lesion has grown recently (boolean symptom).",
    "hurt": "Whether the lesion is painful (boolean symptom).",
    "changed": "Whether the lesion has changed in appearance recently (boolean symptom).",
    "bleed": "Whether the lesion bleeds (boolean symptom).",
    "elevation": "Whether the lesion is elevated/raised (boolean symptom).",
    "img_id": "Filename of the corresponding lesion image.",
    "biopsed": "Whether the diagnosis was confirmed via biopsy (True) or clinical judgment only (False).",
}


def run(logger) -> pd.DataFrame:
    metadata = pd.read_csv(PAD_UFES20_METADATA_FILE)
    n_rows = len(metadata)

    rows = []
    for col in metadata.columns:
        n_missing = metadata[col].isna().sum()
        sample_values = metadata[col].dropna().unique()[:5]
        rows.append(
            {
                "column": col,
                "description": COLUMN_DESCRIPTIONS.get(col, "UNKNOWN - not documented, verify with source paper."),
                "dtype": str(metadata[col].dtype),
                "n_unique": metadata[col].nunique(dropna=True),
                "missing_count": n_missing,
                "missing_pct": round(100 * n_missing / n_rows, 2),
                "sample_values": ", ".join(str(v) for v in sample_values),
            }
        )
    df = pd.DataFrame(rows)

    undocumented = df[df["description"].str.startswith("UNKNOWN")]
    if not undocumented.empty:
        logger.warning(
            "%d column(s) have no documented description: %s",
            len(undocumented),
            list(undocumented["column"]),
        )

    out_path = PAD_UFES20_REPORTS_DIR / "07_column_description.csv"
    save_csv(df, out_path)

    logger.info("Column description data dictionary built for %d columns", len(df))
    logger.info("Saved -> %s", out_path)

    return df
