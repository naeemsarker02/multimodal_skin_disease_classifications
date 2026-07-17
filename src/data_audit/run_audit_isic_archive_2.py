"""Orchestrator: runs the full ISIC Archive 2 dataset audit.

Each module writes only to reports/ISIC_Archive_2/ and logs/ISIC_Archive_2/
- data/raw/ is never written to (enforced by common/io_utils.assert_not_raw_path).

Run from the project root with:
    .venv/Scripts/python.exe -m src.data_audit.run_audit_isic_archive_2
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit.config import ISIC2_LOGS_DIR
from src.data_audit.common.logging_utils import get_logger
from src.data_audit.isic_archive_2 import (
    m01_folder_inventory,
    m02_image_verification,
    m03_missing_orphan_detection,
    m04_metadata_overview,
    m05_missing_value_analysis,
    m06_class_distribution,
    m07_lesion_patient_statistics,
    m08_dataset_summary,
)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = ISIC2_LOGS_DIR / f"audit_run_{timestamp}.log"
    logger = get_logger("isic_archive_2_audit", log_file)

    logger.info("=" * 70)
    logger.info("ISIC ARCHIVE 2 DATASET AUDIT - START")
    logger.info("=" * 70)

    results = {}
    try:
        logger.info("--- Module 1: Folder structure + image inventory ---")
        results["inventory"] = m01_folder_inventory.run(logger)

        logger.info("--- Module 2: Image verification / size stats ---")
        results["verification"], results["size_stats"] = m02_image_verification.run(logger, results["inventory"])

        logger.info("--- Module 3: Missing / orphan image detection ---")
        results["missing_image_stats"] = m03_missing_orphan_detection.run(logger, results["inventory"])

        logger.info("--- Module 4: Metadata overview + column descriptions ---")
        results["column_description"] = m04_metadata_overview.run(logger)

        logger.info("--- Module 5: Missing value analysis ---")
        results["missing_values"] = m05_missing_value_analysis.run(logger)

        logger.info("--- Module 6: Class distribution ---")
        results["class_distribution"] = m06_class_distribution.run(logger)

        logger.info("--- Module 7: Lesion / patient statistics ---")
        results["lesion_stats"] = m07_lesion_patient_statistics.run(logger)

        logger.info("--- Module 8: Dataset summary ---")
        m08_dataset_summary.run(logger, results)
    except Exception:
        logger.exception("Audit pipeline failed")
        raise

    logger.info("=" * 70)
    logger.info("ISIC ARCHIVE 2 DATASET AUDIT - COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
