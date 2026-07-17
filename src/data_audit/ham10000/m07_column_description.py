"""Module 8: Metadata column descriptions.

Combines computed statistics (dtype, % missing, unique count, sample
values) with the meaning of each column. HAM10000 has only 7 columns
and no clinical/symptom fields - the domain knowledge below reflects
the dataset's published documentation (Tschandl et al., 2018) and is
recorded explicitly since it cannot be inferred from the CSV values
alone.
"""

import pandas as pd

from src.data_audit.config import HAM10000_METADATA_FILE, HAM10000_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv

COLUMN_DESCRIPTIONS = {
    "lesion_id": "Identifier for the lesion; not unique per row - a lesion may have multiple images (2-6) taken from different angles/times.",
    "image_id": "Unique identifier for the image (bare ISIC id, no file extension); the corresponding file is f'{image_id}.jpg'.",
    "dx": "Diagnostic label / target class (bkl, nv, df, mel, vasc, bcc, akiec).",
    "dx_type": "How the diagnosis was confirmed: 'histo' (histopathology, gold standard), 'follow_up' (clinical follow-up), 'consensus' (expert consensus), 'confocal' (confocal microscopy). Lower-confidence confirmation types mean some labels are less certain than others.",
    "age": "Patient age in years.",
    "sex": "Patient sex (male/female); 'unknown' is used as a literal string for missing values, not NaN.",
    "localization": "Anatomical location of the lesion; 'unknown' is used as a literal string for missing values, not NaN.",
}


def run(logger) -> pd.DataFrame:
    metadata = pd.read_csv(HAM10000_METADATA_FILE)
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

    out_path = HAM10000_REPORTS_DIR / "07_column_description.csv"
    save_csv(df, out_path)

    logger.info("Column description data dictionary built for %d columns", len(df))
    logger.info(
        "Note: this isna()-based missing_pct does NOT capture the literal "
        "'unknown' string values in sex/localization - see module 9 "
        "(missing value analysis) for the corrected figures."
    )
    logger.info("Saved -> %s", out_path)

    return df
