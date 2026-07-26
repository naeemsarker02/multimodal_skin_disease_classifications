"""
Gap B from the 2026-07-25 project audit: 3 images shared between ISIC
Archive 1 and ISIC Archive 2 have disagreeing `disease_label` values
between the two archives (documented in Project_Tracking.md's
Cross-Dataset Verification entry, 2026-07-08). Consistent with this
project's existing "ambiguous label -> exclude, never guess" precedent
(ISIC Archive 1's 155 conflicting-folder images, ISIC Archive 2's 255
missing-`diagnosis_3` rows), these 3 images are excluded from
external-validation scoring rather than assigning either archive's label
as authoritative.

This script does not modify any existing processed split file. It only
verifies the 3 known conflicting image_ids are present with disagreeing
disease_label values in both archives, then writes a new reference file
per archive.

Output (new files only, no existing file is modified):
    data/processed/ISIC_Archive_1/label_conflict_exclusions.csv
    data/processed/ISIC_Archive_2/label_conflict_exclusions.csv
"""

import pandas as pd

CONFLICTING_IMAGE_IDS = ["ISIC_0011118", "ISIC_0011126", "ISIC_0028619"]

ARCHIVES = {
    "ISIC_Archive_1": [
        "data/processed/ISIC_Archive_1/metadata_train.csv",
        "data/processed/ISIC_Archive_1/metadata_val.csv",
        "data/processed/ISIC_Archive_1/metadata_test.csv",
    ],
    "ISIC_Archive_2": [
        "data/processed/ISIC_Archive_2/metadata_train.csv",
        "data/processed/ISIC_Archive_2/metadata_val.csv",
        "data/processed/ISIC_Archive_2/metadata_test.csv",
    ],
}


def load_metadata(paths):
    return pd.concat(
        [pd.read_csv(p, usecols=["image_id", "disease_label"]) for p in paths],
        ignore_index=True,
    )


def main():
    labels_by_archive = {}
    for archive_name, paths in ARCHIVES.items():
        df = load_metadata(paths)
        found = df[df["image_id"].isin(CONFLICTING_IMAGE_IDS)]
        labels_by_archive[archive_name] = dict(zip(found["image_id"], found["disease_label"]))
        missing = set(CONFLICTING_IMAGE_IDS) - set(found["image_id"])
        if missing:
            raise SystemExit(
                f"{archive_name}: expected conflicting image_ids not found: {missing} - "
                "the 3-conflict list may be stale, re-verify before excluding."
            )

    for image_id in CONFLICTING_IMAGE_IDS:
        a1_label = labels_by_archive["ISIC_Archive_1"][image_id]
        a2_label = labels_by_archive["ISIC_Archive_2"][image_id]
        if a1_label == a2_label:
            raise SystemExit(
                f"{image_id}: labels now agree (Archive 1={a1_label!r}, "
                f"Archive 2={a2_label!r}) - this image is no longer a conflict, "
                "remove it from CONFLICTING_IMAGE_IDS rather than excluding it "
                "unnecessarily."
            )
        print(f"{image_id}: Archive 1={a1_label!r} vs Archive 2={a2_label!r} - confirmed disagreement")

    for archive_name in ARCHIVES:
        out_path = f"data/processed/{archive_name}/label_conflict_exclusions.csv"
        pd.DataFrame({"image_id": sorted(CONFLICTING_IMAGE_IDS)}).to_csv(out_path, index=False)
        print(f"{archive_name}: wrote {out_path}")


if __name__ == "__main__":
    main()
