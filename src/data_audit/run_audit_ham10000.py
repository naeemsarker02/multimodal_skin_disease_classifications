"""Orchestrator: runs the full HAM10000 dataset audit, modules 1-11 in
order. Each module writes only to reports/HAM10000/ and logs/HAM10000/
- data/raw/ is never written to (enforced by
common/io_utils.assert_not_raw_path).

Run from the project root with:
    .venv/Scripts/python.exe -m src.data_audit.run_audit_ham10000
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit.config import HAM10000_LOGS_DIR
from src.data_audit.common.logging_utils import get_logger
from src.data_audit.ham10000 import (
    m01_folder_structure,
    m02_image_inventory,
    m03_image_verification,
    m04_missing_image_detection,
    m05_image_size_stats,
    m06_metadata_overview,
    m07_column_description,
    m08_missing_value_analysis,
    m09_class_distribution,
    m10_lesion_statistics,
    m11_dataset_summary,
)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = HAM10000_LOGS_DIR / f"audit_run_{timestamp}.log"
    logger = get_logger("ham10000_audit", log_file)

    logger.info("=" * 70)
    logger.info("HAM10000 DATASET AUDIT - START")
    logger.info("=" * 70)

    results = {}

    try:
        logger.info("--- Module 1: Folder structure analysis ---")
        results["folder_structure"] = m01_folder_structure.run(logger)

        logger.info("--- Module 2: Image inventory ---")
        results["inventory"] = m02_image_inventory.run(logger)

        logger.info("--- Module 3 & 4: Image verification / corrupted detection ---")
        results["verification"] = m03_image_verification.run(logger, results["inventory"])

        logger.info("--- Module 5: Missing / orphan image detection ---")
        results["missing_image_stats"] = m04_missing_image_detection.run(logger, results["inventory"])

        logger.info("--- Module 6: Image size statistics ---")
        results["size_stats"] = m05_image_size_stats.run(logger, results["verification"])

        logger.info("--- Module 7: Metadata analysis ---")
        results["metadata_overview"] = m06_metadata_overview.run(logger)

        logger.info("--- Module 8: Metadata column descriptions ---")
        results["column_description"] = m07_column_description.run(logger)

        logger.info("--- Module 9: Missing value analysis ---")
        results["missing_values"] = m08_missing_value_analysis.run(logger)

        logger.info("--- Module 10: Class distribution ---")
        results["class_distribution"] = m09_class_distribution.run(logger)

        logger.info("--- Module 11: Lesion statistics ---")
        results["lesion_stats"] = m10_lesion_statistics.run(logger)

        logger.info("--- Module 12: Dataset summary ---")
        m11_dataset_summary.run(logger, results)

    except Exception:
        logger.exception("Audit pipeline failed")
        raise

    logger.info("=" * 70)
    logger.info("HAM10000 DATASET AUDIT - COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
