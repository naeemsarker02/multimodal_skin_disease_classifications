"""Orchestrator: runs the full PAD-UFES-20 cleaning pipeline, steps 1-6.

data/raw/ is never written to (enforced by
common/io_utils.assert_not_raw_path, reused from the audit phase).
Intermediate outputs go to data/interim/PAD_UFES20/, final outputs to
data/processed/PAD_UFES20/.

Run from the project root with:
    .venv/Scripts/python.exe -m src.data_cleaning.run_cleaning_pad_ufes20
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit.common.logging_utils import get_logger
from src.data_cleaning.config import PAD_UFES20_LOGS_DIR
from src.data_cleaning.pad_ufes20 import (
    c01_column_standardization,
    c02_value_validation,
    c03_label_standardization,
    c04_patient_split,
    c05_split_quality_report,
    c06_dataset_description,
)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = PAD_UFES20_LOGS_DIR / f"cleaning_run_{timestamp}.log"
    logger = get_logger("pad_ufes20_cleaning", log_file)

    logger.info("=" * 70)
    logger.info("PAD-UFES-20 DATA CLEANING - START")
    logger.info("=" * 70)

    try:
        logger.info("--- Step 1: Column standardization ---")
        df = c01_column_standardization.run(logger)

        logger.info("--- Step 2: Value validation ---")
        validation_report = c02_value_validation.run(logger, df)

        logger.info("--- Step 3: Label standardization ---")
        df = c03_label_standardization.run(logger, df)

        logger.info("--- Step 4: Patient-wise split ---")
        splits = c04_patient_split.run(logger, df)

        logger.info("--- Step 5: Split quality report ---")
        c05_split_quality_report.run(logger, splits)

        logger.info("--- Step 6: Dataset description ---")
        c06_dataset_description.run(logger, splits, validation_report)

    except Exception:
        logger.exception("Cleaning pipeline failed")
        raise

    logger.info("=" * 70)
    logger.info("PAD-UFES-20 DATA CLEANING - COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
