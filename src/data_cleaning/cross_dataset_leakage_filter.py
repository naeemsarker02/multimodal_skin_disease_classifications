"""
Fix (b) from docs/Dataset_Preparation_Final_Report.md §9: restrict external-validation
claims rather than re-splitting globally (fix a). This script does not modify any
existing processed split file. It only reads the already-produced metadata CSVs for
HAM10000, ISIC Archive 1, and ISIC Archive 2, and writes a new reference file per ISIC
archive listing which image_ids must be excluded before that archive is used as
external validation against a HAM10000-trained model.

Output (new files only, no existing file is modified):
    data/processed/ISIC_Archive_1/external_validation_exclusions.csv
    data/processed/ISIC_Archive_2/external_validation_exclusions.csv

Each output file has one column, image_id, listing every image in that archive that
is also present in HAM10000 (across HAM10000's train+val+test). A future external-
validation evaluation script should drop these image_ids from the archive before
scoring a HAM10000-trained model.
"""

import pandas as pd

HAM_FILES = [
    "data/processed/HAM10000/metadata_train.csv",
    "data/processed/HAM10000/metadata_val.csv",
    "data/processed/HAM10000/metadata_test.csv",
]

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


def load_image_ids(paths):
    return pd.concat([pd.read_csv(p, usecols=["image_id"]) for p in paths])["image_id"]


def main():
    ham_ids = set(load_image_ids(HAM_FILES))
    print(f"HAM10000 total image_ids (train+val+test): {len(ham_ids)}")

    for archive_name, paths in ARCHIVES.items():
        archive_ids = load_image_ids(paths)
        total = len(archive_ids)
        overlap_mask = archive_ids.isin(ham_ids)
        overlap_ids = sorted(set(archive_ids[overlap_mask]))
        remaining = total - len(overlap_ids)

        print(f"\n{archive_name}: total images = {total}")
        print(f"{archive_name}: overlapping with HAM10000 = {len(overlap_ids)}")
        print(f"{archive_name}: remaining after exclusion (usable as external validation vs HAM10000) = {remaining}")

        out_path = f"data/processed/{archive_name}/external_validation_exclusions.csv"
        pd.DataFrame({"image_id": overlap_ids}).to_csv(out_path, index=False)
        print(f"{archive_name}: wrote {out_path}")


if __name__ == "__main__":
    main()
