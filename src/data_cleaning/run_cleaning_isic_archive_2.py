"""Orchestrator: runs the full ISIC Archive 2 cleaning pipeline.

Run from the project root with:
    .venv/Scripts/python.exe -m src.data_cleaning.run_cleaning_isic_archive_2
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit.common.logging_utils import get_logger
from src.data_cleaning.config import ISIC2_LOGS_DIR
from src.data_cleaning.isic_archive_2 import (
    c01_column_standardization,
    c02_value_validation,
    c03_label_standardization,
    c04_lesion_split,
    c05_split_quality_report,
    c06_dataset_description,
)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = ISIC2_LOGS_DIR / f"cleaning_run_{timestamp}.log"
    logger = get_logger("isic_archive_2_cleaning", log_file)

    logger.info("=" * 70)
    logger.info("ISIC ARCHIVE 2 DATA CLEANING - START")
    logger.info("=" * 70)

    try:
        logger.info("--- Step 1: Column standardization ---")
        df = c01_column_standardization.run(logger)

        logger.info("--- Step 2: Value validation ---")
        validation_report = c02_value_validation.run(logger, df)

        logger.info("--- Step 3: Label standardization ---")
        df = c03_label_standardization.run(logger, df)
        n_unlabeled = df["disease_label"].isna().sum()

        logger.info("--- Step 4: Group-wise split ---")
        splits = c04_lesion_split.run(logger, df)

        logger.info("--- Step 5: Split quality report ---")
        c05_split_quality_report.run(logger, splits)

        logger.info("--- Step 6: Dataset description ---")
        c06_dataset_description.run(logger, splits, validation_report, n_unlabeled)

    except Exception:
        logger.exception("Cleaning pipeline failed")
        raise

    logger.info("=" * 70)
    logger.info("ISIC ARCHIVE 2 DATA CLEANING - COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
