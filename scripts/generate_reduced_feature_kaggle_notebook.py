"""Generates notebooks/pad_ufes20_reduced_feature_kaggle_notebook.md.

Phase 8's PAD-UFES-20 -> HAM10000 cross-dataset generalization experiment
needs schema-matched ("reduced-feature") metadata/fusion/cross-attention
models, since the existing rich-feature (21-column) checkpoints cannot
run on HAM10000 data at all (18 columns don't exist there). This notebook
runs all 9 new training runs (metadata_reduced/fusion_reduced/
cross_attention_reduced x 3 seeds) in dependency order: metadata_reduced
must finish first (fusion_reduced/cross_attention_reduced warm-start from
its checkpoints), then fusion_reduced and cross_attention_reduced (which
also warm-start their image side from the *existing* Stage 1
image_seed{N}_best.pt - already covered by the same Stage 1 checkpoint
Kaggle dataset used by the Phase 7 notebooks).

Reads the actual Phase 8 source files from src/models/ and embeds their
real content into %%writefile cells - never hand-typed, so it cannot
silently drift from the source-of-truth .py files. Mirrors the structure
of the Stage 1/Stage 2 fusion notebooks.

Run: python scripts/generate_reduced_feature_kaggle_notebook.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src" / "models"
OUT_PATH = PROJECT_ROOT / "notebooks" / "pad_ufes20_reduced_feature_kaggle_notebook.md"

FILES = [
    ("config.py", "/kaggle/working/src/models/config.py"),
    ("dataset.py", "/kaggle/working/src/models/dataset.py"),
    ("image_model.py", "/kaggle/working/src/models/image_model.py"),
    ("metadata_model.py", "/kaggle/working/src/models/metadata_model.py"),
    ("fusion_model.py", "/kaggle/working/src/models/fusion_model.py"),
    ("cross_attention_fusion_model.py", "/kaggle/working/src/models/cross_attention_fusion_model.py"),
    ("train.py", "/kaggle/working/src/models/train.py"),
    ("train_fusion.py", "/kaggle/working/src/models/train_fusion.py"),
    ("train_cross_attention_fusion.py", "/kaggle/working/src/models/train_cross_attention_fusion.py"),
    ("train_metadata_reduced.py", "/kaggle/working/src/models/train_metadata_reduced.py"),
    ("train_fusion_reduced.py", "/kaggle/working/src/models/train_fusion_reduced.py"),
    ("train_cross_attention_fusion_reduced.py", "/kaggle/working/src/models/train_cross_attention_fusion_reduced.py"),
]


def read_source(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8").rstrip("\n")


def writefile_cell(cell_num: int, source_name: str, kaggle_path: str) -> str:
    code = read_source(source_name)
    return (
        f"## Cell {cell_num} — `%%writefile {kaggle_path}`\n\n"
        f"```python\n%%writefile {kaggle_path}\n{code}\n```\n"
    )


def build_notebook() -> str:
    parts = []

    parts.append(
        "# PAD-UFES-20 Kaggle Notebook — Phase 8 Reduced-Feature Models "
        "(Cross-Dataset Generalization Prep)\n\n"
        "> **WARNING - read before pasting any cell into Kaggle:** for every "
        "`%%writefile` cell below, `%%writefile <path>` MUST be the exact first "
        "line of the Kaggle cell, with absolutely nothing above it. **After "
        "pasting each `%%writefile` cell, re-open it and visually confirm "
        "`%%writefile` is line 1 before running it.**\n\n"
        "Structured for \"Save & Run All (Commit)\" from the start. Trains "
        "**9 new schema-matched models** on PAD-UFES-20 (metadata_reduced, "
        "fusion_reduced, cross_attention_reduced x 3 seeds each), restricted "
        "to the 3 metadata columns HAM10000 also has (age, sex, "
        "anatomical_site — see `docs/Phase8_Anatomical_Site_Mapping.csv` for "
        "the approved anatomical_site normalization mapping), purpose-built "
        "for the Phase 8 PAD-UFES-20 → HAM10000 cross-dataset generalization "
        "experiment. **Training order matters:** metadata_reduced (seeds "
        "0/1/2) must complete first — fusion_reduced/cross_attention_reduced "
        "warm-start their metadata side from its checkpoints (their image "
        "side warm-starts from the *existing* Stage 1 image checkpoints, "
        "already in the Stage 1 checkpoints dataset).\n\n"
        "Requires the same **three** Kaggle \"Add Data\" sources as the Phase "
        "7 notebooks (no new upload needed):\n\n"
        "- `mahdavi1202/skin-cancer` (raw PAD-UFES-20 image mirror)\n"
        "- `naeemsarkertracer/pad-ufes20-processed` (our processed metadata CSVs)\n"
        "- `naeemsarkertracer/pad-ufes20-stage1-checkpoints` (Stage 1 "
        "`image_seed{0,1,2}_best.pt` — warm-start source for the image side "
        "of fusion_reduced/cross_attention_reduced)\n\n"
        "Paste each cell below into a separate Kaggle notebook cell, in order.\n\n"
        "---\n\n"
    )

    parts.append(
        "## Cell 1 — Folder verification (raw mirror + processed nesting + checkpoint dataset check)\n\n"
        "```python\n"
        "import os\n\n"
        "def show(path, label):\n"
        "    print(f\"--- {label}: {path} ---\")\n"
        "    if not os.path.isdir(path):\n"
        "        print(\"  !! NOT FOUND\")\n"
        "        return\n"
        "    for entry in sorted(os.listdir(path)):\n"
        "        full = os.path.join(path, entry)\n"
        "        kind = \"dir\" if os.path.isdir(full) else \"file\"\n"
        "        print(f\"  [{kind}] {entry}\")\n\n"
        "raw_root = \"/kaggle/input/datasets/mahdavi1202/skin-cancer\"\n"
        "show(\"/kaggle/input/datasets\", \"datasets root\")\n"
        "show(raw_root, \"raw PAD-UFES-20 mirror\")\n\n"
        "processed_root = \"/kaggle/input/datasets/naeemsarkertracer/pad-ufes20-processed\"\n"
        "show(processed_root, \"processed PAD-UFES-20 (root)\")\n"
        "wrapped = os.path.join(processed_root, \"PAD_UFES20\")\n"
        "show(wrapped, \"processed PAD-UFES-20 (wrapped candidate)\")\n"
        "assert os.path.isfile(os.path.join(wrapped, \"metadata_train.csv\")), (\n"
        "    \"Expected wrapped metadata_train.csv not found - check processed dataset packaging\"\n"
        ")\n\n"
        "checkpoint_root = \"/kaggle/input/datasets/naeemsarkertracer/pad-ufes20-stage1-checkpoints\"\n"
        "show(checkpoint_root, \"Stage 1 checkpoints (root)\")\n"
        "wrapped_ckpt = os.path.join(checkpoint_root, \"checkpoints\")\n"
        "show(wrapped_ckpt, \"Stage 1 checkpoints (wrapped candidate: root/checkpoints/)\")\n\n"
        "expected = [f\"image_seed{s}_best.pt\" for s in (0, 1, 2)]\n\n"
        "def all_present(base):\n"
        "    return all(os.path.isfile(os.path.join(base, f)) for f in expected)\n\n"
        "wrapped_ok = all_present(wrapped_ckpt)\n"
        "root_ok = all_present(checkpoint_root)\n"
        "print(f\"\\nAll 3 image checkpoints found at wrapped candidate ({wrapped_ckpt}): {wrapped_ok}\")\n"
        "print(f\"All 3 image checkpoints found at root ({checkpoint_root}): {root_ok}\")\n\n"
        "if wrapped_ok:\n"
        "    resolved_checkpoint_dir = wrapped_ckpt\n"
        "elif root_ok:\n"
        "    resolved_checkpoint_dir = checkpoint_root\n"
        "else:\n"
        "    resolved_checkpoint_dir = None\n\n"
        "assert resolved_checkpoint_dir is not None, (\n"
        "    \"Stage 1 image checkpoints not found in either expected layout - \"\n"
        "    \"fusion_reduced/cross_attention_reduced warm-start cannot proceed.\"\n"
        ")\n\n"
        "print(\"\\nOK: raw images present, processed dataset wrapped as expected, \"\n"
        "      f\"Stage 1 image checkpoints found at: {resolved_checkpoint_dir}\")\n"
        "```\n\n---\n\n"
    )

    parts.append(
        "## Cell 2 — Setup / os.makedirs\n\n"
        "```python\n"
        "import os, sys\n\n"
        "os.makedirs(\"/kaggle/working/src/models\", exist_ok=True)\n"
        "os.makedirs(\"/kaggle/working/src/evaluation\", exist_ok=True)\n"
        "open(\"/kaggle/working/src/__init__.py\", \"w\").close()\n"
        "open(\"/kaggle/working/src/models/__init__.py\", \"w\").close()\n"
        "open(\"/kaggle/working/src/evaluation/__init__.py\", \"w\").close()\n\n"
        "sys.path.insert(0, \"/kaggle/working\")\n"
        "print(\"setup OK, sys.path[0]:\", sys.path[0])\n"
        "```\n\n---\n\n"
    )

    cell_num = 3
    for source_name, kaggle_path in FILES:
        parts.append(writefile_cell(cell_num, source_name, kaggle_path))
        parts.append("\n---\n\n")
        cell_num += 1

    parts.append(
        f"## Cell {cell_num} — Sanity check (reduced-feature preprocessor + normalization)\n\n"
        "```python\n"
        "import sys\n"
        "for mod in list(sys.modules):\n"
        "    if mod.startswith(\"src.\"):\n"
        "        del sys.modules[mod]\n\n"
        "import pandas as pd\n"
        "from src.models.config import (\n"
        "    get_dataset, REDUCED_NUMERIC_FEATURES, REDUCED_CATEGORICAL_FEATURES,\n"
        "    normalize_anatomical_site_for_cross_dataset,\n"
        ")\n"
        "from src.models.dataset import MetadataPreprocessor\n\n"
        "ds_config = get_dataset(\"PAD_UFES20\")\n"
        "assert ds_config.train_csv.exists(), \"metadata_train.csv not found\"\n\n"
        "for site in (\"CHEST\", \"FOREARM\", \"ARM\", \"THIGH\", \"NOSE\", \"LIP\"):\n"
        "    print(site, \"->\", normalize_anatomical_site_for_cross_dataset(site))\n\n"
        "reduced_config = ds_config.with_features(REDUCED_NUMERIC_FEATURES, REDUCED_CATEGORICAL_FEATURES)\n"
        "train_df = pd.read_csv(reduced_config.train_csv)\n"
        "preprocessor = MetadataPreprocessor(\n"
        "    reduced_config, column_transforms={\"anatomical_site\": normalize_anatomical_site_for_cross_dataset}\n"
        ").fit(train_df)\n"
        "print(\"\\nreduced output_dim:\", preprocessor.output_dim)\n"
        "print(\"anatomical_site categories:\", preprocessor.categorical_values[\"anatomical_site\"])\n"
        "assert \"upper extremity\" in preprocessor.categorical_values[\"anatomical_site\"]\n"
        "assert \"__MISSING__\" in preprocessor.categorical_values[\"anatomical_site\"]\n\n"
        "for path in (ds_config.stage1_checkpoints_dir / f\"image_seed{s}_best.pt\" for s in (0, 1, 2)):\n"
        "    assert path.exists(), f\"Missing Stage 1 image checkpoint: {path}\"\n"
        "print(\"\\nSANITY CHECK PASSED\")\n"
        "```\n\n---\n\n"
    )
    cell_num += 1

    for seed in (0, 1, 2):
        parts.append(
            f"## Cell {cell_num} — Train: metadata_reduced, seed {seed}\n\n"
            "```python\n"
            f"!python -m src.models.train_metadata_reduced --dataset PAD_UFES20 --seed {seed}\n"
            "```\n\n---\n\n"
        )
        cell_num += 1

    for seed in (0, 1, 2):
        parts.append(
            f"## Cell {cell_num} — Train: fusion_reduced, seed {seed}\n\n"
            "```python\n"
            f"!python -m src.models.train_fusion_reduced --dataset PAD_UFES20 --seed {seed}\n"
            "```\n\n---\n\n"
        )
        cell_num += 1

    for i, seed in enumerate((0, 1, 2)):
        parts.append(
            f"## Cell {cell_num} — Train: cross_attention_reduced, seed {seed}\n\n"
            "```python\n"
            f"!python -m src.models.train_cross_attention_fusion_reduced --dataset PAD_UFES20 --seed {seed}\n"
            "```\n\n"
        )
        if i != 2:
            parts.append("---\n\n")
        cell_num += 1

    return "".join(parts)


def main():
    OUT_PATH.write_text(build_notebook(), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
