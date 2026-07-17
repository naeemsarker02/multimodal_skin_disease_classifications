"""Module 6: Image size statistics.

Consumes the width/height/mode already extracted by module 3
(image verification) - no image files are reopened here. Computes
descriptive statistics and a resolution frequency table, plus a
histogram figure of image dimensions.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_audit.config import PAD_UFES20_FIGURES_DIR, PAD_UFES20_REPORTS_DIR
from src.data_audit.common.io_utils import ensure_dir, save_csv


def run(logger, verification_df: pd.DataFrame) -> pd.DataFrame:
    ok = verification_df[verification_df["status"] == "OK"].copy()
    ok["aspect_ratio"] = ok["width"] / ok["height"]

    stats_rows = []
    for col in ["width", "height", "aspect_ratio"]:
        desc = ok[col].describe()
        stats_rows.append(
            {
                "metric": col,
                "count": desc["count"],
                "min": desc["min"],
                "max": desc["max"],
                "mean": round(desc["mean"], 2),
                "median": ok[col].median(),
                "std": round(desc["std"], 2),
            }
        )
    stats_df = pd.DataFrame(stats_rows)

    resolution_freq = (
        ok.groupby(["width", "height"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    mode_freq = ok["mode"].value_counts().reset_index()
    mode_freq.columns = ["image_mode", "count"]

    save_csv(stats_df, PAD_UFES20_REPORTS_DIR / "05_image_size_stats.csv")
    save_csv(resolution_freq, PAD_UFES20_REPORTS_DIR / "05_resolution_frequency.csv")
    save_csv(mode_freq, PAD_UFES20_REPORTS_DIR / "05_image_mode_frequency.csv")

    ensure_dir(PAD_UFES20_FIGURES_DIR)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(ok["width"], bins=30, color="steelblue")
    axes[0].set_title("Image width distribution")
    axes[0].set_xlabel("width (px)")
    axes[1].hist(ok["height"], bins=30, color="darkorange")
    axes[1].set_title("Image height distribution")
    axes[1].set_xlabel("height (px)")
    fig.tight_layout()
    fig_path = PAD_UFES20_FIGURES_DIR / "image_size_distribution.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    logger.info("Image size stats computed over %d successfully-decoded images", len(ok))
    logger.info(
        "Width range: %s-%s px | Height range: %s-%s px | Distinct resolutions: %d",
        ok["width"].min(),
        ok["width"].max(),
        ok["height"].min(),
        ok["height"].max(),
        len(resolution_freq),
    )
    logger.info("Image modes found: %s", dict(zip(mode_freq["image_mode"], mode_freq["count"])))
    logger.info("Saved -> %s", PAD_UFES20_REPORTS_DIR / "05_image_size_stats.csv")
    logger.info("Saved figure -> %s", fig_path)

    return stats_df
