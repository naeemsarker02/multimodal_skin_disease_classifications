"""Orchestrator: builds the PAD-UFES-20-Expanded dataset variant.

data/processed/PAD_UFES20/ is never modified (read-only source).
data/raw/ is never written to for the original 4 datasets; DERM12345 and
MED-NODE's raw folders were populated once via a separate download step
(not part of this script - see docs/Project_Tracking.md, "Step 2
Integration Plan" for how those images were obtained).

Run from the project root with:
    .venv/Scripts/python.exe -m src.data_cleaning.run_build_pad_ufes20_expanded
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit.common.logging_utils import get_logger
from src.data_cleaning.config import LOGS_ROOT
from src.data_cleaning.pad_ufes20_expanded import c01_build_expanded_dataset


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_ROOT / "PAD_UFES20_Expanded" / f"build_run_{timestamp}.log"
    logger = get_logger("pad_ufes20_expanded_build", log_file)

    logger.info("=" * 70)
    logger.info("PAD-UFES-20-EXPANDED BUILD - START")
    logger.info("=" * 70)

    try:
        c01_build_expanded_dataset.run(logger)
    except Exception:
        logger.exception("Build failed")
        raise

    logger.info("=" * 70)
    logger.info("PAD-UFES-20-EXPANDED BUILD - COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
