"""Module 3 & 4: Image verification and corrupted image detection.

For every image in the inventory, attempts to fully open and decode the
file with Pillow. This single read-only pass produces both:
  - the full verification report (status + dimensions for every image)
  - the corrupted-images report (the subset that failed to decode)

Verification and corruption detection are combined here because
detecting corruption *is* attempting verification; splitting them into
two scripts would mean opening every image file twice for no benefit.
Image dimensions captured here are reused by module 5 (image size
statistics) so that module never has to reopen the image files.
"""

import pandas as pd
from PIL import Image

from src.data_audit.config import PAD_UFES20_RAW_DIR, PAD_UFES20_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv


def _verify_one(rel_path: str) -> dict:
    full_path = PAD_UFES20_RAW_DIR / rel_path
    result = {
        "relative_path": rel_path,
        "status": "OK",
        "error_message": "",
        "width": None,
        "height": None,
        "mode": None,
    }
    try:
        # verify() checks integrity but leaves the file unusable afterwards,
        # so a fresh handle is opened to actually decode pixel data.
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


def run(logger, inventory_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Verifying %d images (open + decode)", len(inventory_df))

    records = [_verify_one(rel_path) for rel_path in inventory_df["relative_path"]]
    df = pd.DataFrame(records)

    out_path = PAD_UFES20_REPORTS_DIR / "03_image_verification.csv"
    save_csv(df, out_path)

    corrupted = df[df["status"] == "CORRUPTED"]
    corrupted_path = PAD_UFES20_REPORTS_DIR / "03_corrupted_images.csv"
    save_csv(corrupted, corrupted_path)

    logger.info("Verification complete: %d OK, %d CORRUPTED", len(df) - len(corrupted), len(corrupted))
    if not corrupted.empty:
        for _, row in corrupted.iterrows():
            logger.warning("  CORRUPTED: %s -> %s", row["relative_path"], row["error_message"])
    logger.info("Saved -> %s", out_path)
    logger.info("Saved -> %s", corrupted_path)

    return df
