"""Phase 5 EDA - ISIC Archive 2.

Explores only data/processed/ISIC_Archive_2/metadata_train.csv, restricted
to columns in feature_whitelist.md (13 features) + disease_label. No
Fitzpatrick column exists in this dataset. Val/test are touched only for
the split-balance check. Never writes to data/raw/ or any existing
processed CSV.

Run from the project root with:
    .venv/Scripts/python.exe -m src.eda.eda_isic_archive_2
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

CORE_DEMOGRAPHIC_COLS = ["age_approx", "sex", "anatom_site_general", "clin_size_long_diam_mm"]
SITE_HIERARCHY_COLS = ["anatom_site_1", "anatom_site_2", "anatom_site_3", "anatom_site_4", "anatom_site_5"]
CLINICAL_HISTORY_COLS = ["family_hx_mm", "personal_hx_mm", "dermoscopic_type"]
ALL_WHITELIST_COLS = CORE_DEMOGRAPHIC_COLS + SITE_HIERARCHY_COLS + ["family_hx_mm", "personal_hx_mm", "dermoscopic_type"]


def run_class_distribution(logger, train: pd.DataFrame) -> None:
    counts = train["disease_label"].value_counts()
    save_csv(counts.rename("count").reset_index().rename(columns={"index": "disease_label"}),
             config.ISIC2_EDA_DIR / "01_class_distribution.csv")
    plotting.plot_class_distribution(
        counts, config.ISIC2_EDA_FIGURES_DIR / "01_class_distribution.png",
        "ISIC Archive 2 (train) - disease_label distribution",
    )
    logger.info("01_class_distribution done: %s", counts.to_dict())


def run_demographics(logger, train: pd.DataFrame) -> None:
    kinds = {"age_approx": "hist", "clin_size_long_diam_mm": "hist", "sex": "bar", "anatom_site_general": "bar"}
    plotting.plot_panel(
        train, CORE_DEMOGRAPHIC_COLS, kinds, config.ISIC2_EDA_FIGURES_DIR / "02_demographics.png",
        "ISIC Archive 2 (train) - core demographics", ncols=2,
    )
    summary = train[CORE_DEMOGRAPHIC_COLS].describe(include="all").transpose()
    save_csv(summary.reset_index().rename(columns={"index": "column"}),
             config.ISIC2_EDA_DIR / "02_demographics_summary.csv")

    plotting.plot_panel(
        train, SITE_HIERARCHY_COLS, {c: "bar" for c in SITE_HIERARCHY_COLS},
        config.ISIC2_EDA_FIGURES_DIR / "02b_site_hierarchy.png",
        "ISIC Archive 2 (train) - anatomical site hierarchy (levels 1-5)", ncols=3,
    )
    plotting.plot_panel(
        train, CLINICAL_HISTORY_COLS, {c: "bar" for c in CLINICAL_HISTORY_COLS},
        config.ISIC2_EDA_FIGURES_DIR / "02c_clinical_history.png",
        "ISIC Archive 2 (train) - family/personal history, dermoscopic type", ncols=3,
    )
    logger.info("02_demographics (+ site hierarchy, clinical history) done")


def run_missingness(logger, train: pd.DataFrame) -> None:
    missing_pct = train[ALL_WHITELIST_COLS].isna().mean().sort_values(ascending=False) * 100
    save_csv(missing_pct.rename("missing_pct").reset_index().rename(columns={"index": "column"}),
             config.ISIC2_EDA_DIR / "04_missing_value_report.csv")
    plotting.plot_missingness(
        train, ALL_WHITELIST_COLS, config.ISIC2_EDA_FIGURES_DIR / "04_missing_value_report.png",
        "ISIC Archive 2 (train) - missingness among whitelisted features",
    )
    logger.info("04_missing_value_report done")


def run_image_dimensions(logger, train: pd.DataFrame) -> None:
    stats = sample_image_dimensions(train, "image_path", config.IMAGE_DIM_SAMPLE_SIZE, config.IMAGE_DIM_SAMPLE_SEED, logger)
    save_csv(stats, config.ISIC2_EDA_DIR / "05_image_dimensions.csv")
    plotting.plot_image_dimensions(
        stats, config.ISIC2_EDA_FIGURES_DIR / "05_image_dimensions.png",
        "ISIC Archive 2 - image width/height/size (sampled)",
    )
    logger.info("05_image_dimensions done: %s", stats[["width", "height", "aspect_ratio", "size_kb"]].describe().to_dict())


def run_sample_grid(logger, train: pd.DataFrame) -> None:
    plotting.plot_sample_image_grid(
        train, "image_path", "disease_label",
        config.ISIC2_EDA_FIGURES_DIR / "06_sample_image_grid.png",
        config.IMAGES_PER_CLASS_IN_GRID, config.GRID_SAMPLE_SEED,
        "ISIC Archive 2 - sample images per class",
    )
    logger.info("06_sample_image_grid done")


def run_split_balance(logger, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    counts = {"train": train["disease_label"].value_counts(), "val": val["disease_label"].value_counts(),
              "test": test["disease_label"].value_counts()}
    plotting.plot_split_balance(
        counts, config.ISIC2_EDA_FIGURES_DIR / "07_split_balance_check.png",
        "ISIC Archive 2 - class proportion by split",
    )
    combined = pd.DataFrame(counts).fillna(0).astype(int)
    save_csv(combined.reset_index().rename(columns={"index": "disease_label"}),
             config.ISIC2_EDA_DIR / "07_split_balance_check.csv")
    logger.info("07_split_balance_check done")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger("isic_archive_2_eda", config.ISIC2_LOGS_DIR / f"eda_run_{timestamp}.log")

    logger.info("=" * 70)
    logger.info("ISIC ARCHIVE 2 EDA - START")
    logger.info("=" * 70)

    train = pd.read_csv(config.ISIC2_PROCESSED_DIR / "metadata_train.csv")
    val = pd.read_csv(config.ISIC2_PROCESSED_DIR / "metadata_val.csv")
    test = pd.read_csv(config.ISIC2_PROCESSED_DIR / "metadata_test.csv")

    try:
        run_class_distribution(logger, train)
        run_demographics(logger, train)
        run_missingness(logger, train)
        run_image_dimensions(logger, train)
        run_sample_grid(logger, train)
        run_split_balance(logger, train, val, test)
    except Exception:
        logger.exception("EDA pipeline failed")
        raise

    logger.info("=" * 70)
    logger.info("ISIC ARCHIVE 2 EDA - COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
