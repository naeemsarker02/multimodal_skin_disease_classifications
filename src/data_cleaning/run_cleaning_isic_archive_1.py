"""Orchestrator: runs the full ISIC Archive 1 cleaning pipeline.

Run from the project root with:
    .venv/Scripts/python.exe -m src.data_cleaning.run_cleaning_isic_archive_1
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit.common.logging_utils import get_logger
from src.data_cleaning.config import ISIC1_LOGS_DIR
from src.data_cleaning.isic_archive_1 import (
    c01_schema_construction,
    c02_label_standardization,
    c03_val_split,
    c04_split_quality_report,
    c05_dataset_description,
)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = ISIC1_LOGS_DIR / f"cleaning_run_{timestamp}.log"
    logger = get_logger("isic_archive_1_cleaning", log_file)

    logger.info("=" * 70)
    logger.info("ISIC ARCHIVE 1 DATA CLEANING - START")
    logger.info("=" * 70)

    try:
        logger.info("--- Step 1: Schema construction ---")
        df, n_excluded = c01_schema_construction.run(logger)

        logger.info("--- Step 2: Label standardization ---")
        df = c02_label_standardization.run(logger, df)

        logger.info("--- Step 3: Validation split ---")
        splits = c03_val_split.run(logger, df)

        logger.info("--- Step 4: Split quality report ---")
        c04_split_quality_report.run(logger, splits)

        logger.info("--- Step 5: Dataset description ---")
        c05_dataset_description.run(logger, splits, n_excluded)

    except Exception:
        logger.exception("Cleaning pipeline failed")
        raise

    logger.info("=" * 70)
    logger.info("ISIC ARCHIVE 1 DATA CLEANING - COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
