"""Step 3: Validation split.

This archive ships with a fixed Train/Test split (no lesion or patient
identifier exists to group on - each image is treated as independent).
The provided Test set is kept exactly as-is as the final held-out test
set. A validation set is carved out of Train only, stratified by
disease_label, with a fixed seed for reproducibility.
"""

import numpy as np
import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.config import ISIC1_PROCESSED_DIR, SPLIT_RATIOS, SPLIT_SEED

# Val fraction taken out of Train, sized so the overall train:val:test
# ratio is close to the project's standard 70/15/15 once combined with
# the archive's own fixed Test set.
VAL_FRACTION_OF_TRAIN = 0.15


def run(logger, df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SPLIT_SEED)

    train_df = df[df["provided_split"] == "Train"].copy()
    test_df = df[df["provided_split"] == "Test"].copy()

    val_indices = []
    for label, group in train_df.groupby("disease_label"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * VAL_FRACTION_OF_TRAIN))
        val_indices.extend(idx[:n_val])

    val_df = train_df.loc[val_indices].reset_index(drop=True)
    train_only_df = train_df.drop(index=val_indices).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    splits = {"train": train_only_df, "val": val_df, "test": test_df}
    for split_name, split_df in splits.items():
        out_cols = [c for c in split_df.columns if c != "provided_split"]
        save_csv(split_df[out_cols], ISIC1_PROCESSED_DIR / f"metadata_{split_name}.csv")
        logger.info("%s: %d images", split_name, len(split_df))

    logger.info("Split seed: %d | Val fraction of Train: %s | Test: archive-provided, untouched", SPLIT_SEED, VAL_FRACTION_OF_TRAIN)
    for split_name in splits:
        logger.info("Saved -> %s", ISIC1_PROCESSED_DIR / f"metadata_{split_name}.csv")

    return splits
