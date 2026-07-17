"""Module 7: Metadata analysis.

Structural profile of HAM10000_metadata.csv: shape, dtypes, per-column
unique counts, duplicate rows, and descriptive statistics for the only
numeric column (age).
"""

import pandas as pd

from src.data_audit.config import HAM10000_METADATA_FILE, HAM10000_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv

NUMERIC_COLUMNS = ["age"]


def run(logger) -> pd.DataFrame:
    metadata = pd.read_csv(HAM10000_METADATA_FILE)

    overview_rows = []
    for col in metadata.columns:
        overview_rows.append(
            {
                "column": col,
                "dtype": str(metadata[col].dtype),
                "n_unique": metadata[col].nunique(dropna=True),
                "n_missing": metadata[col].isna().sum(),
            }
        )
    overview_df = pd.DataFrame(overview_rows)

    save_csv(overview_df, HAM10000_REPORTS_DIR / "06_metadata_overview.csv")

    numeric_desc = metadata[NUMERIC_COLUMNS].describe().transpose().reset_index()
    numeric_desc = numeric_desc.rename(columns={"index": "column"})
    save_csv(numeric_desc, HAM10000_REPORTS_DIR / "06_metadata_numeric_describe.csv")

    n_duplicate_rows = metadata.duplicated().sum()

    logger.info("Metadata shape: %d rows x %d columns", *metadata.shape)
    logger.info("Fully duplicate rows: %d", n_duplicate_rows)
    logger.info("Numeric column summary (age) saved")
    logger.info("Saved -> %s", HAM10000_REPORTS_DIR / "06_metadata_overview.csv")

    return overview_df
