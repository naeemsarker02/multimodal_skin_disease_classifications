"""Generates notebooks/pad_ufes20_cross_attention_kaggle_notebook.md.

Reads the actual Phase 7 Stage 2 source files from src/models/ and embeds
their real content into %%writefile cells - the notebook is never
hand-typed, so it cannot silently drift from the source-of-truth .py
files. Mirrors the structure of the Stage 1 late-fusion notebook
(notebooks/pad_ufes20_fusion_kaggle_notebook.md): same "Add Data" sources
(raw mirror, processed metadata, Stage 1 checkpoints - Stage 2 warm-starts
from the identical Stage 1 checkpoints Stage 1 fusion did), same
folder-verification / sanity-check / full-model-check cell pattern, then
3 training cells (seeds 0/1/2).

Run: python scripts/generate_cross_attention_kaggle_notebook.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src" / "models"
OUT_PATH = PROJECT_ROOT / "notebooks" / "pad_ufes20_cross_attention_kaggle_notebook.md"

# (source file, path to %%writefile on Kaggle)
FILES = [
    ("config.py", "/kaggle/working/src/models/config.py"),
    ("dataset.py", "/kaggle/working/src/models/dataset.py"),
    ("image_model.py", "/kaggle/working/src/models/image_model.py"),
    ("metadata_model.py", "/kaggle/working/src/models/metadata_model.py"),
    ("fusion_model.py", "/kaggle/working/src/models/fusion_model.py"),
    ("cross_attention_fusion_model.py", "/kaggle/working/src/models/cross_attention_fusion_model.py"),
    ("train.py", "/kaggle/working/src/models/train.py"),
    ("train_cross_attention_fusion.py", "/kaggle/working/src/models/train_cross_attention_fusion.py"),
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
        "# PAD-UFES-20 Kaggle Notebook — Phase 7 Stage 2 (Cross-Attention Fusion)\n\n"
        "> **WARNING - read before pasting any cell into Kaggle:** for every "
        "`%%writefile` cell below, `%%writefile <path>` MUST be the exact first "
        "line of the Kaggle cell, with absolutely nothing above it - no blank "
        "line, no comment, no stray character. Kaggle (like Jupyter) only "
        "treats a line as a cell magic if it is the first line of the cell; "
        "anything preceding it turns `%%writefile` into a plain, non-magic "
        "line and the file silently never gets written. **After pasting each "
        "`%%writefile` cell, re-open it and visually confirm `%%writefile` is "
        "line 1 before running it.**\n\n"
        "Structured for \"Save & Run All (Commit)\" from the start, same "
        "process as the Stage 1 baseline and late-fusion notebooks. Requires "
        "**three** Kaggle \"Add Data\" sources attached to the notebook before "
        "running - identical sources to the Stage 1 late-fusion notebook, "
        "since Stage 2 warm-starts from the same Stage 1 image/metadata "
        "checkpoints:\n\n"
        "- `mahdavi1202/skin-cancer` (raw PAD-UFES-20 image mirror)\n"
        "- `naeemsarkertracer/pad-ufes20-processed` (our processed metadata CSVs)\n"
        "- `naeemsarkertracer/pad-ufes20-stage1-checkpoints` (published — "
        "https://www.kaggle.com/datasets/naeemsarkertracer/pad-ufes20-stage1-checkpoints), "
        "containing the Stage 1 checkpoints (`image_seed{0,1,2}_best.pt`, "
        "`metadata_seed{0,1,2}_best.pt`). Cross-attention fusion warm-starts "
        "from these; it cannot proceed without them.\n\n"
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
        "# --- Stage 1 checkpoint dataset check ------------------------------------\n"
        "checkpoint_root = \"/kaggle/input/datasets/naeemsarkertracer/pad-ufes20-stage1-checkpoints\"\n"
        "show(checkpoint_root, \"Stage 1 checkpoints (root)\")\n"
        "wrapped_ckpt = os.path.join(checkpoint_root, \"checkpoints\")\n"
        "show(wrapped_ckpt, \"Stage 1 checkpoints (wrapped candidate: root/checkpoints/)\")\n\n"
        "expected = [f\"{b}_seed{s}_best.pt\" for b in (\"image\", \"metadata\") for s in (0, 1, 2)]\n\n"
        "def all_present(base):\n"
        "    return all(os.path.isfile(os.path.join(base, f)) for f in expected)\n\n"
        "wrapped_ok = all_present(wrapped_ckpt)\n"
        "root_ok = all_present(checkpoint_root)\n"
        "print(f\"\\nAll 6 expected checkpoints found at wrapped candidate ({wrapped_ckpt}): {wrapped_ok}\")\n"
        "print(f\"All 6 expected checkpoints found at root ({checkpoint_root}): {root_ok}\")\n\n"
        "if wrapped_ok:\n"
        "    resolved_checkpoint_dir = wrapped_ckpt\n"
        "elif root_ok:\n"
        "    resolved_checkpoint_dir = checkpoint_root\n"
        "else:\n"
        "    resolved_checkpoint_dir = None\n\n"
        "assert resolved_checkpoint_dir is not None, (\n"
        "    \"Stage 1 checkpoints not found in either expected layout - cross-attention \"\n"
        "    \"warm-start cannot proceed. Re-check the uploaded dataset's contents/packaging.\"\n"
        ")\n\n"
        "print(\"\\nOK: raw images present, processed dataset wrapped as expected, \"\n"
        "      f\"all 6 Stage 1 checkpoints found at: {resolved_checkpoint_dir}\")\n"
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
        f"## Cell {cell_num} — Sanity check (config load + Stage 1 checkpoint resolution + real image path resolution)\n\n"
        "```python\n"
        "import sys\n"
        "for mod in list(sys.modules):\n"
        "    if mod.startswith(\"src.\"):\n"
        "        del sys.modules[mod]\n\n"
        "import pandas as pd\n"
        "from src.models.config import get_dataset, resolve_image_path\n\n"
        "ds_config = get_dataset(\"PAD_UFES20\")\n"
        "print(\"num_classes:\", ds_config.num_classes)\n"
        "print(\"train_csv:\", ds_config.train_csv, \"exists:\", ds_config.train_csv.exists())\n"
        "print(\"val_csv:  \", ds_config.val_csv, \"exists:\", ds_config.val_csv.exists())\n"
        "print(\"test_csv: \", ds_config.test_csv, \"exists:\", ds_config.test_csv.exists())\n"
        "assert ds_config.train_csv.exists(), \"metadata_train.csv not found - check processed dataset nesting\"\n\n"
        "print(\"\\nstage1_checkpoints_dir:\", ds_config.stage1_checkpoints_dir)\n"
        "for branch in (\"image\", \"metadata\"):\n"
        "    for seed in (0, 1, 2):\n"
        "        p = ds_config.stage1_checkpoints_dir / f\"{branch}_seed{seed}_best.pt\"\n"
        "        assert p.exists(), f\"Missing Stage 1 checkpoint: {p}\"\n"
        "print(\"all 6 Stage 1 checkpoints resolved OK\")\n\n"
        "df = pd.read_csv(ds_config.train_csv)\n"
        "sample_image_path = df.iloc[0][\"image_path\"]\n"
        "resolved = resolve_image_path(sample_image_path)\n"
        "print(\"\\nsample image_path (from CSV):\", sample_image_path)\n"
        "print(\"resolved filesystem path:    \", resolved)\n"
        "print(\"resolved path exists:        \", resolved.exists())\n"
        "assert resolved.exists(), f\"Resolved image path does not exist: {resolved}\"\n\n"
        "import random\n"
        "random.seed(0)\n"
        "sample_rows = df.sample(n=min(20, len(df)), random_state=0)\n"
        "missing = []\n"
        "for _, row in sample_rows.iterrows():\n"
        "    p = resolve_image_path(row[\"image_path\"])\n"
        "    if not p.exists():\n"
        "        missing.append((row[\"image_path\"], str(p)))\n"
        "print(f\"\\nChecked {len(sample_rows)} random rows, missing: {len(missing)}\")\n"
        "if missing:\n"
        "    print(\"First few missing:\", missing[:5])\n"
        "assert not missing, \"Some resolved image paths do not exist - check folder-verification cell output\"\n\n"
        "print(\"\\nSANITY CHECK PASSED\")\n"
        "```\n\n---\n\n"
    )
    cell_num += 1

    parts.append(
        f"## Cell {cell_num} — Full model/GPU/dependency check (cross-attention model, warm-start, one real batch)\n\n"
        "```python\n"
        "import torch, sys\n"
        "print(\"python:\", sys.version)\n"
        "print(\"torch:\", torch.__version__)\n"
        "print(\"cuda available:\", torch.cuda.is_available())\n"
        "if torch.cuda.is_available():\n"
        "    print(\"device:\", torch.cuda.get_device_name(0))\n\n"
        "from src.models.config import get_dataset\n"
        "from src.models.dataset import FusionDataset, MetadataPreprocessor\n"
        "from src.models.cross_attention_fusion_model import CrossAttentionFusionModel\n"
        "import pandas as pd\n\n"
        "ds_config = get_dataset(\"PAD_UFES20\")\n"
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n\n"
        "preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))\n"
        "model = CrossAttentionFusionModel(metadata_input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes)\n\n"
        "image_ckpt = ds_config.stage1_checkpoints_dir / \"image_seed0_best.pt\"\n"
        "metadata_ckpt = ds_config.stage1_checkpoints_dir / \"metadata_seed0_best.pt\"\n"
        "model.load_stage1_checkpoints(image_ckpt, metadata_ckpt, device)\n"
        "model.to(device)\n"
        "model.eval()  # single-sample batch below - BatchNorm1d needs eval mode, not train mode, to run on batch size 1\n"
        "print(\"Stage 1 checkpoints (seed 0) loaded into CrossAttentionFusionModel OK\")\n\n"
        "train_ds = FusionDataset(ds_config.train_csv, ds_config, preprocessor, train=True)\n"
        "image, metadata, label = train_ds[0]\n"
        "assert image.shape == (3, 224, 224)\n"
        "assert metadata.shape == (preprocessor.output_dim,)\n\n"
        "with torch.no_grad():\n"
        "    out = model(image.unsqueeze(0).to(device), metadata.unsqueeze(0).to(device))\n"
        "assert out.shape == (1, ds_config.num_classes)\n"
        "print(\"cross-attention model forward pass OK, output shape:\", out.shape)\n\n"
        "print(\"\\nALL CHECKS PASSED - ready to train\")\n"
        "```\n\n---\n\n"
    )
    cell_num += 1

    for seed in (0, 1, 2):
        parts.append(
            f"## Cell {cell_num} — Train: cross-attention fusion, seed {seed}\n\n"
            "```python\n"
            f"!python -m src.models.train_cross_attention_fusion --dataset PAD_UFES20 --seed {seed}\n"
            "```\n\n"
        )
        if seed != 2:
            parts.append("---\n\n")
        cell_num += 1

    return "".join(parts)


def main():
    OUT_PATH.write_text(build_notebook(), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
