"""Orchestrator: runs the full ISIC Archive 1 dataset audit.

Each module writes only to reports/ISIC_Archive_1/ and logs/ISIC_Archive_1/
- data/raw/ is never written to (enforced by common/io_utils.assert_not_raw_path).

Run from the project root with:
    .venv/Scripts/python.exe -m src.data_audit.run_audit_isic_archive_1
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit.config import ISIC1_LOGS_DIR
from src.data_audit.common.logging_utils import get_logger
from src.data_audit.isic_archive_1 import (
    m01_folder_inventory,
    m02_image_verification,
    m03_class_distribution,
    m04_dataset_summary,
)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = ISIC1_LOGS_DIR / f"audit_run_{timestamp}.log"
    logger = get_logger("isic_archive_1_audit", log_file)

    logger.info("=" * 70)
    logger.info("ISIC ARCHIVE 1 DATASET AUDIT - START")
    logger.info("=" * 70)

    results = {}
    try:
        logger.info("--- Module 1: Folder structure + image inventory ---")
        results["inventory"] = m01_folder_inventory.run(logger)

        logger.info("--- Module 2: Image verification / size stats ---")
        results["verification"] = m02_image_verification.run(logger, results["inventory"])

        logger.info("--- Module 3: Class distribution ---")
        results["class_distribution"] = m03_class_distribution.run(logger, results["inventory"])

        logger.info("--- Module 4: Dataset summary ---")
        m04_dataset_summary.run(logger, results)
    except Exception:
        logger.exception("Audit pipeline failed")
        raise

    logger.info("=" * 70)
    logger.info("ISIC ARCHIVE 1 DATASET AUDIT - COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
