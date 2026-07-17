"""Module 2: Image verification, corrupted detection, and size stats."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.data_audit.config import ISIC2_FIGURES_DIR, ISIC2_RAW_DIR, ISIC2_REPORTS_DIR
from src.data_audit.common.io_utils import ensure_dir, save_csv


def _verify_one(rel_path: str) -> dict:
    full_path = ISIC2_RAW_DIR / rel_path
    result = {"relative_path": rel_path, "status": "OK", "error_message": "", "width": None, "height": None, "mode": None}
    try:
        with Image.open(full_path) as img:
            img.verify()
        with Image.open(full_path) as img:
            img.load()
            result["width"], result["height"] = img.size
            result["mode"] = img.mode
    except Exception as exc:  # noqa: BLE001 - any decode failure is "corrupted"
        result["status"] = "CORRUPTED"
        result["error_message"] = f"{type(exc).__name__}: {exc}"
    return result


def run(logger, inventory_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Verifying %d images (open + decode)", len(inventory_df))

    records = [_verify_one(rel_path) for rel_path in inventory_df["relative_path"]]
    df = pd.DataFrame(records)

    save_csv(df, ISIC2_REPORTS_DIR / "03_image_verification.csv")
    corrupted = df[df["status"] == "CORRUPTED"]
    save_csv(corrupted, ISIC2_REPORTS_DIR / "03_corrupted_images.csv")

    logger.info("Verification complete: %d OK, %d CORRUPTED", len(df) - len(corrupted), len(corrupted))
    if not corrupted.empty:
        for _, row in corrupted.iterrows():
            logger.warning("  CORRUPTED: %s -> %s", row["relative_path"], row["error_message"])

    ok = df[df["status"] == "OK"].copy()
    ok["aspect_ratio"] = ok["width"] / ok["height"]
    stats_rows = []
    for col in ["width", "height", "aspect_ratio"]:
        desc = ok[col].describe()
        stats_rows.append(
            {"metric": col, "count": desc["count"], "min": desc["min"], "max": desc["max"], "mean": round(desc["mean"], 2), "median": ok[col].median(), "std": round(desc["std"], 2)}
        )
    stats_df = pd.DataFrame(stats_rows)
    resolution_freq = ok.groupby(["width", "height"]).size().reset_index(name="count").sort_values("count", ascending=False)

    save_csv(stats_df, ISIC2_REPORTS_DIR / "04_image_size_stats.csv")
    save_csv(resolution_freq, ISIC2_REPORTS_DIR / "04_resolution_frequency.csv")

    ensure_dir(ISIC2_FIGURES_DIR)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(ok["width"], bins=30, color="steelblue")
    axes[0].set_title("Image width distribution")
    axes[1].hist(ok["height"], bins=30, color="darkorange")
    axes[1].set_title("Image height distribution")
    fig.tight_layout()
    fig_path = ISIC2_FIGURES_DIR / "image_size_distribution.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    logger.info(
        "Width range: %s-%s px | Height range: %s-%s px | Distinct resolutions: %d",
        ok["width"].min(), ok["width"].max(), ok["height"].min(), ok["height"].max(), len(resolution_freq),
    )
    logger.info("Saved -> %s", ISIC2_REPORTS_DIR / "03_image_verification.csv")
    logger.info("Saved -> %s", ISIC2_REPORTS_DIR / "04_image_size_stats.csv")
    logger.info("Saved figure -> %s", fig_path)

    return df, stats_df
