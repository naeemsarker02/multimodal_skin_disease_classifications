"""Image dimension/size sampling, shared across all 4 datasets' EDA.

Samples a fixed number of images per dataset (not the full dataset - would
be slow and is unnecessary for a distribution estimate) and reads their
real width/height/size directly from data/raw/ via PIL. Never writes
anything under data/raw/.
"""

import os

import pandas as pd
from PIL import Image


def sample_image_dimensions(
    df: pd.DataFrame,
    image_path_col: str,
    n: int,
    seed: int,
    logger,
) -> pd.DataFrame:
    """Sample up to n rows from df, open each image_path with PIL, and
    return a DataFrame of [image_path, width, height, aspect_ratio, size_kb].
    Rows whose image is missing on disk are skipped and logged, not silently
    dropped."""
    sample = df.sample(n=min(n, len(df)), random_state=seed)

    records = []
    missing = 0
    for path in sample[image_path_col]:
        if not os.path.exists(path):
            missing += 1
            continue
        with Image.open(path) as im:
            width, height = im.size
        size_kb = os.path.getsize(path) / 1024
        records.append(
            {
                "image_path": path,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 4),
                "size_kb": round(size_kb, 2),
            }
        )

    if missing:
        logger.warning("%d/%d sampled images were missing on disk and were skipped", missing, len(sample))

    return pd.DataFrame.from_records(records)
