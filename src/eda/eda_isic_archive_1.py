"""Phase 5 EDA - ISIC Archive 1.

This archive has 0 whitelisted metadata columns (feature_whitelist.md -
image + class label only), so its EDA is necessarily limited to: class
distribution, image dimensions, sample image grid, and split balance.
There is no demographics/missingness/Fitzpatrick step - nothing exists to
plot. Never writes to data/raw/ or any existing processed CSV.

Run from the project root with:
    .venv/Scripts/python.exe -m src.eda.eda_isic_archive_1
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
from src.eda.common.image_stats import sample_image_dimensions


def run_class_distribution(logger, train: pd.DataFrame) -> None:
    counts = train["disease_label"].value_counts()
    save_csv(counts.rename("count").reset_index().rename(columns={"index": "disease_label"}),
             config.ISIC1_EDA_DIR / "01_class_distribution.csv")
    plotting.plot_class_distribution(
        counts, config.ISIC1_EDA_FIGURES_DIR / "01_class_distribution.png",
        "ISIC Archive 1 (train) - disease_label distribution",
    )
    logger.info("01_class_distribution done: %s", counts.to_dict())


def run_image_dimensions(logger, train: pd.DataFrame) -> None:
    stats = sample_image_dimensions(train, "image_path", config.IMAGE_DIM_SAMPLE_SIZE, config.IMAGE_DIM_SAMPLE_SEED, logger)
    save_csv(stats, config.ISIC1_EDA_DIR / "05_image_dimensions.csv")
    plotting.plot_image_dimensions(
        stats, config.ISIC1_EDA_FIGURES_DIR / "05_image_dimensions.png",
        "ISIC Archive 1 - image width/height/size (sampled)",
    )
    logger.info("05_image_dimensions done: %s", stats[["width", "height", "aspect_ratio", "size_kb"]].describe().to_dict())


def run_sample_grid(logger, train: pd.DataFrame) -> None:
    plotting.plot_sample_image_grid(
        train, "image_path", "disease_label",
        config.ISIC1_EDA_FIGURES_DIR / "06_sample_image_grid.png",
        config.IMAGES_PER_CLASS_IN_GRID, config.GRID_SAMPLE_SEED,
        "ISIC Archive 1 - sample images per class",
    )
    logger.info("06_sample_image_grid done")


def run_split_balance(logger, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    counts = {"train": train["disease_label"].value_counts(), "val": val["disease_label"].value_counts(),
              "test": test["disease_label"].value_counts()}
    plotting.plot_split_balance(
        counts, config.ISIC1_EDA_FIGURES_DIR / "07_split_balance_check.png",
        "ISIC Archive 1 - class proportion by split",
    )
    combined = pd.DataFrame(counts).fillna(0).astype(int)
    save_csv(combined.reset_index().rename(columns={"index": "disease_label"}),
             config.ISIC1_EDA_DIR / "07_split_balance_check.csv")
    logger.info("07_split_balance_check done")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger("isic_archive_1_eda", config.ISIC1_LOGS_DIR / f"eda_run_{timestamp}.log")

    logger.info("=" * 70)
    logger.info("ISIC ARCHIVE 1 EDA - START")
    logger.info("=" * 70)

    train = pd.read_csv(config.ISIC1_PROCESSED_DIR / "metadata_train.csv")
    val = pd.read_csv(config.ISIC1_PROCESSED_DIR / "metadata_val.csv")
    test = pd.read_csv(config.ISIC1_PROCESSED_DIR / "metadata_test.csv")

    try:
        run_class_distribution(logger, train)
        run_image_dimensions(logger, train)
        run_sample_grid(logger, train)
        run_split_balance(logger, train, val, test)
    except Exception:
        logger.exception("EDA pipeline failed")
        raise

    logger.info("=" * 70)
    logger.info("ISIC ARCHIVE 1 EDA - COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
