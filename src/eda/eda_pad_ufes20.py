"""Phase 5 EDA - PAD-UFES-20.

Explores only data/processed/PAD_UFES20/metadata_train.csv, restricted to
columns in feature_whitelist.md (+ disease_label, + fitspatrick). Val/test
are touched only for the split-balance check (existing split counts, no new
exploration of their content). Writes figures/tables to reports/eda/PAD_UFES20/
and a log to logs/PAD_UFES20/eda_run_<timestamp>.log. Never writes to
data/raw/ or to any existing processed CSV.

Run from the project root with:
    .venv/Scripts/python.exe -m src.eda.eda_pad_ufes20
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

WHITELIST_DEMOGRAPHIC_COLS = ["age", "sex", "anatomical_site", "diameter_1", "diameter_2"]
SYMPTOM_COLS = ["itch", "grew", "hurt", "changed", "bleed", "elevation"]
LIFESTYLE_COLS = [
    "smoke", "drink", "pesticide", "skin_cancer_history", "cancer_history",
    "has_piped_water", "has_sewage_system", "background_father", "background_mother",
]
ALL_WHITELIST_COLS = WHITELIST_DEMOGRAPHIC_COLS + SYMPTOM_COLS + LIFESTYLE_COLS + ["fitspatrick"]


def run_class_distribution(logger, train: pd.DataFrame) -> None:
    counts = train["disease_label"].value_counts()
    save_csv(counts.rename("count").reset_index().rename(columns={"index": "disease_label"}),
             config.PAD_UFES20_EDA_DIR / "01_class_distribution.csv")
    plotting.plot_class_distribution(
        counts, config.PAD_UFES20_EDA_FIGURES_DIR / "01_class_distribution.png",
        "PAD-UFES-20 (train) - disease_label distribution",
    )
    logger.info("01_class_distribution done: %s", counts.to_dict())


def run_demographics(logger, train: pd.DataFrame) -> None:
    kinds = {"age": "hist", "diameter_1": "hist", "diameter_2": "hist", "sex": "bar", "anatomical_site": "bar"}
    plotting.plot_panel(
        train, WHITELIST_DEMOGRAPHIC_COLS, kinds,
        config.PAD_UFES20_EDA_FIGURES_DIR / "02_demographics.png",
        "PAD-UFES-20 (train) - demographics", ncols=3,
    )
    summary = train[WHITELIST_DEMOGRAPHIC_COLS].describe(include="all").transpose()
    save_csv(summary.reset_index().rename(columns={"index": "column"}),
             config.PAD_UFES20_EDA_DIR / "02_demographics_summary.csv")

    plotting.plot_panel(
        train, SYMPTOM_COLS, {c: "bar" for c in SYMPTOM_COLS},
        config.PAD_UFES20_EDA_FIGURES_DIR / "02b_symptoms.png",
        "PAD-UFES-20 (train) - symptom flags", ncols=3,
    )
    plotting.plot_panel(
        train, LIFESTYLE_COLS, {c: "bar" for c in LIFESTYLE_COLS},
        config.PAD_UFES20_EDA_FIGURES_DIR / "02c_lifestyle.png",
        "PAD-UFES-20 (train) - lifestyle / socioeconomic", ncols=3,
    )
    logger.info("02_demographics (+ symptoms, lifestyle) done")


def run_fitzpatrick(logger, train: pd.DataFrame) -> None:
    plotting.plot_categorical_bar(
        train["fitspatrick"], config.PAD_UFES20_EDA_FIGURES_DIR / "03_fitzpatrick_distribution.png",
        "PAD-UFES-20 (train) - Fitzpatrick skin type distribution", "fitzpatrick type",
    )
    crosstab = pd.crosstab(train["disease_label"], train["fitspatrick"], dropna=False)
    save_csv(crosstab.reset_index(), config.PAD_UFES20_EDA_DIR / "03_fitzpatrick_by_disease_label.csv")
    logger.info("03_fitzpatrick_distribution done")


def run_missingness(logger, train: pd.DataFrame) -> None:
    missing_pct = train[ALL_WHITELIST_COLS].isna().mean().sort_values(ascending=False) * 100
    save_csv(missing_pct.rename("missing_pct").reset_index().rename(columns={"index": "column"}),
             config.PAD_UFES20_EDA_DIR / "04_missing_value_report.csv")
    plotting.plot_missingness(
        train, ALL_WHITELIST_COLS, config.PAD_UFES20_EDA_FIGURES_DIR / "04_missing_value_report.png",
        "PAD-UFES-20 (train) - missingness among whitelisted features",
    )
    logger.info("04_missing_value_report done")


def run_image_dimensions(logger, train: pd.DataFrame) -> None:
    stats = sample_image_dimensions(train, "image_path", config.IMAGE_DIM_SAMPLE_SIZE, config.IMAGE_DIM_SAMPLE_SEED, logger)
    save_csv(stats, config.PAD_UFES20_EDA_DIR / "05_image_dimensions.csv")
    plotting.plot_image_dimensions(
        stats, config.PAD_UFES20_EDA_FIGURES_DIR / "05_image_dimensions.png",
        "PAD-UFES-20 - image width/height/size (sampled)",
    )
    logger.info("05_image_dimensions done: %s", stats[["width", "height", "aspect_ratio", "size_kb"]].describe().to_dict())


def run_sample_grid(logger, train: pd.DataFrame) -> None:
    plotting.plot_sample_image_grid(
        train, "image_path", "disease_label",
        config.PAD_UFES20_EDA_FIGURES_DIR / "06_sample_image_grid.png",
        config.IMAGES_PER_CLASS_IN_GRID, config.GRID_SAMPLE_SEED,
        "PAD-UFES-20 - sample images per class",
    )
    logger.info("06_sample_image_grid done")


def run_split_balance(logger, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    counts = {"train": train["disease_label"].value_counts(), "val": val["disease_label"].value_counts(),
              "test": test["disease_label"].value_counts()}
    plotting.plot_split_balance(
        counts, config.PAD_UFES20_EDA_FIGURES_DIR / "07_split_balance_check.png",
        "PAD-UFES-20 - class proportion by split",
    )
    combined = pd.DataFrame(counts).fillna(0).astype(int)
    save_csv(combined.reset_index().rename(columns={"index": "disease_label"}),
             config.PAD_UFES20_EDA_DIR / "07_split_balance_check.csv")
    logger.info("07_split_balance_check done")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger("pad_ufes20_eda", config.PAD_UFES20_LOGS_DIR / f"eda_run_{timestamp}.log")

    logger.info("=" * 70)
    logger.info("PAD-UFES-20 EDA - START")
    logger.info("=" * 70)

    train = pd.read_csv(config.PAD_UFES20_PROCESSED_DIR / "metadata_train.csv")
    val = pd.read_csv(config.PAD_UFES20_PROCESSED_DIR / "metadata_val.csv")
    test = pd.read_csv(config.PAD_UFES20_PROCESSED_DIR / "metadata_test.csv")

    try:
        run_class_distribution(logger, train)
        run_demographics(logger, train)
        run_fitzpatrick(logger, train)
        run_missingness(logger, train)
        run_image_dimensions(logger, train)
        run_sample_grid(logger, train)
        run_split_balance(logger, train, val, test)
    except Exception:
        logger.exception("EDA pipeline failed")
        raise

    logger.info("=" * 70)
    logger.info("PAD-UFES-20 EDA - COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
