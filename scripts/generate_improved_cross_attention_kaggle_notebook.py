"""Generates notebooks/pad_ufes20_cross_attention_improved_kaggle_notebook.md.

Score Improvement Experiments track's "best legitimate effort" combined
cross-attention run (Focal Loss + WeightedRandomSampler + stronger
augmentation + label smoothing + cosine annealing, all in ONE training
script - see src/models/train_cross_attention_improved.py) needs to run on
Kaggle (no local GPU). This notebook mirrors the Stage 2 cross-attention
notebook's structure exactly and reuses the same 3 "Add Data" sources - no
new Kaggle dataset upload needed.

Reads the actual source files from src/models/ and embeds their real content
into %%writefile cells - never hand-typed, so it cannot silently drift from
the source-of-truth .py files.

Run: python scripts/generate_improved_cross_attention_kaggle_notebook.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src" / "models"
OUT_PATH = PROJECT_ROOT / "notebooks" / "pad_ufes20_cross_attention_improved_kaggle_notebook.md"

FILES = [
    ("config.py", "/kaggle/working/src/models/config.py"),
    ("dataset.py", "/kaggle/working/src/models/dataset.py"),
    ("image_model.py", "/kaggle/working/src/models/image_model.py"),
    ("metadata_model.py", "/kaggle/working/src/models/metadata_model.py"),
    ("fusion_model.py", "/kaggle/working/src/models/fusion_model.py"),
    ("cross_attention_fusion_model.py", "/kaggle/working/src/models/cross_attention_fusion_model.py"),
    ("train.py", "/kaggle/working/src/models/train.py"),
    ("train_cross_attention_improved.py", "/kaggle/working/src/models/train_cross_attention_improved.py"),
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
        "# PAD-UFES-20 Kaggle Notebook — Score Improvement Experiments: "
        "Combined Best-Effort Cross-Attention Run\n\n"
        "> **WARNING - read before pasting any cell into Kaggle:** for every "
        "`%%writefile` cell below, `%%writefile <path>` MUST be the exact first "
        "line of the Kaggle cell, with absolutely nothing above it. **After "
        "pasting each `%%writefile` cell, re-open it and visually confirm "
        "`%%writefile` is line 1 before running it.**\n\n"
        "Structured for \"Save & Run All (Commit)\" from the start. Trains "
        "**3 new checkpoints** (`cross_attention_improved_seed{0,1,2}_best.pt`), "
        "one combined run stacking every legitimate training-strategy "
        "improvement on top of the existing Phase 7 Stage 2 cross-attention "
        "architecture: Focal Loss (gamma=2.0, existing class-weighted alpha), "
        "WeightedRandomSampler oversampling of rare classes, stronger "
        "augmentation (RandomRotation(30) + RandomAffine on top of the "
        "existing flip/color jitter), label smoothing (0.1), and cosine "
        "annealing LR (1e-5 -> 1e-7). This is the final reported number for "
        "this architecture/dataset - not a separate ablation track.\n\n"
        "Requires the same **three** Kaggle \"Add Data\" sources as the Phase "
        "7 Stage 2 notebook (no new upload needed):\n\n"
        "- `mahdavi1202/skin-cancer` (raw PAD-UFES-20 image mirror)\n"
        "- `naeemsarkertracer/pad-ufes20-processed` (our processed metadata CSVs)\n"
        "- `naeemsarkertracer/pad-ufes20-stage1-checkpoints` (Stage 1 "
        "`image_seed{0,1,2}_best.pt` / `metadata_seed{0,1,2}_best.pt` - "
        "warm-start source)\n\n"
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
        "expected = [f\"{branch}_seed{s}_best.pt\" for branch in (\"image\", \"metadata\") for s in (0, 1, 2)]\n\n"
        "def all_present(base):\n"
        "    return all(os.path.isfile(os.path.join(base, f)) for f in expected)\n\n"
        "wrapped_ok = all_present(wrapped_ckpt)\n"
        "root_ok = all_present(checkpoint_root)\n"
        "print(f\"\\nAll 6 Stage 1 checkpoints found at wrapped candidate ({wrapped_ckpt}): {wrapped_ok}\")\n"
        "print(f\"All 6 Stage 1 checkpoints found at root ({checkpoint_root}): {root_ok}\")\n\n"
        "if wrapped_ok:\n"
        "    resolved_checkpoint_dir = wrapped_ckpt\n"
        "elif root_ok:\n"
        "    resolved_checkpoint_dir = checkpoint_root\n"
        "else:\n"
        "    resolved_checkpoint_dir = None\n\n"
        "assert resolved_checkpoint_dir is not None, (\n"
        "    \"Stage 1 checkpoints not found in either expected layout - \"\n"
        "    \"cross-attention warm-start cannot proceed.\"\n"
        ")\n\n"
        "print(\"\\nOK: raw images present, processed dataset wrapped as expected, \"\n"
        "      f\"Stage 1 checkpoints found at: {resolved_checkpoint_dir}\")\n"
        "```\n\n---\n\n"
    )

    parts.append(
        "## Cell 2 — Setup / os.makedirs\n\n"
        "```python\n"
        "import os, sys\n\n"
        "os.makedirs(\"/kaggle/working/src/models\", exist_ok=True)\n"
        "open(\"/kaggle/working/src/__init__.py\", \"w\").close()\n"
        "open(\"/kaggle/working/src/models/__init__.py\", \"w\").close()\n\n"
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
        f"## Cell {cell_num} — Sanity check (warm-start checkpoints resolve)\n\n"
        "```python\n"
        "import sys\n"
        "for mod in list(sys.modules):\n"
        "    if mod.startswith(\"src.\"):\n"
        "        del sys.modules[mod]\n\n"
        "from src.models.config import get_dataset\n\n"
        "ds_config = get_dataset(\"PAD_UFES20\")\n"
        "assert ds_config.train_csv.exists(), \"metadata_train.csv not found\"\n\n"
        "for branch in (\"image\", \"metadata\"):\n"
        "    for s in (0, 1, 2):\n"
        "        path = ds_config.stage1_checkpoints_dir / f\"{branch}_seed{s}_best.pt\"\n"
        "        assert path.exists(), f\"Missing Stage 1 checkpoint: {path}\"\n"
        "print(\"\\nSANITY CHECK PASSED\")\n"
        "```\n\n---\n\n"
    )
    cell_num += 1

    for i, seed in enumerate((0, 1, 2)):
        parts.append(
            f"## Cell {cell_num} — Train: cross_attention_improved, seed {seed}\n\n"
            "```python\n"
            f"!python -m src.models.train_cross_attention_improved --dataset PAD_UFES20 --seed {seed}\n"
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
