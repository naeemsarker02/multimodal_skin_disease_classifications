"""Phase 5 EDA - cross-dataset shared disease_label comparison (optional,
approved 2026-07-08).

Compares train-set counts for disease_label values shared across 2+ of the
4 datasets. This is a class-taxonomy comparison only - it does NOT imply
these datasets can be pooled as independent training data. HAM10000, ISIC
Archive 1, and ISIC Archive 2 have documented image overlap (see
docs/Dataset_Preparation_Final_Report.md and each archive's
dataset_description.md); the exclusion lists in
data/processed/ISIC_Archive_{1,2}/external_validation_exclusions.csv apply
to any actual cross-dataset training/evaluation use, not to this
descriptive comparison.

Run from the project root with:
    .venv/Scripts/python.exe -m src.eda.eda_cross_dataset
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_audit.common.logging_utils import get_logger
from src.eda import config
from src.eda.common import plotting

CAPTION = (
    "Note: HAM10000, ISIC Archive 1, and ISIC Archive 2 share 40-99% of the same "
    "underlying images (see Dataset_Preparation_Final_Report.md §6). This chart "
    "compares train-set class taxonomy only - it is not a valid basis for pooling "
    "these datasets as independent training data without applying the documented "
    "external_validation_exclusions.csv filters."
)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger("cross_dataset_eda", config.CROSS_DATASET_LOGS_DIR / f"eda_run_{timestamp}.log")

    logger.info("=" * 70)
    logger.info("CROSS-DATASET EDA - START")
    logger.info("=" * 70)

    datasets = {
        "PAD-UFES-20": config.PAD_UFES20_PROCESSED_DIR,
        "HAM10000": config.HAM10000_PROCESSED_DIR,
        "ISIC Archive 1": config.ISIC1_PROCESSED_DIR,
        "ISIC Archive 2": config.ISIC2_PROCESSED_DIR,
    }

    counts_by_dataset = {}
    for name, processed_dir in datasets.items():
        train = pd.read_csv(processed_dir / "metadata_train.csv")
        counts_by_dataset[name] = train["disease_label"].value_counts()

    # Keep only labels that appear in 2+ datasets - a true "shared" comparison.
    all_labels = pd.concat(counts_by_dataset.values(), axis=1).fillna(0)
    all_labels.columns = list(counts_by_dataset.keys())
    shared_labels = all_labels[(all_labels > 0).sum(axis=1) >= 2]

    shared_counts_by_dataset = {
        name: counts_by_dataset[name].reindex(shared_labels.index).fillna(0)
        for name in counts_by_dataset
    }

    save_csv(shared_labels.reset_index().rename(columns={"index": "disease_label"}),
             config.CROSS_DATASET_EDA_DIR / "01_shared_label_comparison.csv")
    plotting.plot_cross_dataset_label_comparison(
        shared_counts_by_dataset,
        config.CROSS_DATASET_EDA_FIGURES_DIR / "01_shared_label_comparison.png",
        "Train-set counts for disease_label values shared across 2+ datasets",
        CAPTION,
    )
    logger.info("01_shared_label_comparison done: %d shared labels out of %d total", len(shared_labels), len(all_labels))

    logger.info("=" * 70)
    logger.info("CROSS-DATASET EDA - COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
