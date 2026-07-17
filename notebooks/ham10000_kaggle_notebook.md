# HAM10000 Kaggle Notebook — Phase 6 Stage 1

Structured for "Save & Run All (Commit)" from the start (per the
process note logged after the PAD-UFES-20 session-loss incident — see
`Project_Tracking.md`). Requires two Kaggle "Add Data" sources attached
to the notebook before running:

- `kmader/skin-cancer-mnist-ham10000` (raw image mirror)
- `naeemsarkertracer/ham10000-processed` (our processed metadata CSVs)

Paste each cell below into a separate Kaggle notebook cell, in order.

---

## Cell 1 — Folder verification (raw mirror duplicate-case check + processed nesting check)

```python
import os

def show(path, label):
    print(f"--- {label}: {path} ---")
    if not os.path.isdir(path):
        print("  !! NOT FOUND")
        return
    for entry in sorted(os.listdir(path)):
        full = os.path.join(path, entry)
        kind = "dir" if os.path.isdir(full) else "file"
        print(f"  [{kind}] {entry}")

raw_root = "/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000"
show("/kaggle/input/datasets", "datasets root")
show(raw_root, "raw HAM10000 mirror")

# --- Duplicate-case folder check ---------------------------------------
# Listing showed BOTH HAM10000_images_part_1/2 (uppercase, matches our
# CSV's image_path values exactly) and ham10000_images_part_1/2
# (lowercase - not part of what we verified previously). Confirm the
# uppercase ones (the only ones resolve_image_path() will ever use,
# since Kaggle is case-sensitive) are complete, rather than assuming the
# lowercase addition didn't affect them.
counts = {}
for name in ["HAM10000_images_part_1", "HAM10000_images_part_2",
             "ham10000_images_part_1", "ham10000_images_part_2"]:
    d = os.path.join(raw_root, name)
    if os.path.isdir(d):
        files = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
        counts[name] = len(files)
        print(f"{name}: {len(files)} .jpg files, sample: {files[:2]}")
    else:
        print(f"{name}: NOT FOUND")

upper_total = counts.get("HAM10000_images_part_1", 0) + counts.get("HAM10000_images_part_2", 0)
lower_total = counts.get("ham10000_images_part_1", 0) + counts.get("ham10000_images_part_2", 0)
print(f"\nUppercase total: {upper_total} (expect 10015 - this is what our CSV/code actually uses)")
print(f"Lowercase total: {lower_total} (unused by our code, informational only)")
assert upper_total == 10015, (
    f"Uppercase HAM10000_images_part_1/2 total is {upper_total}, expected 10015 - "
    f"do not proceed, the mirror's uppercase folders may be incomplete."
)

# Cross-check against the actual CSV: every image_id path must resolve
metadata_csv = os.path.join(raw_root, "HAM10000_metadata.csv")
import pandas as pd
meta = pd.read_csv(metadata_csv)
print(f"\nHAM10000_metadata.csv rows: {len(meta)}")
missing = []
for image_id in meta["image_id"].sample(n=30, random_state=0):
    found = any(
        os.path.isfile(os.path.join(raw_root, part, f"{image_id}.jpg"))
        for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]
    )
    if not found:
        missing.append(image_id)
print(f"Spot-checked 30 image_ids against uppercase folders, missing: {len(missing)}")
assert not missing, f"Missing image_ids in uppercase folders: {missing}"

# --- Processed dataset nesting check ------------------------------------
processed_root = "/kaggle/input/datasets/naeemsarkertracer/ham10000-processed"
show(processed_root, "processed HAM10000 (root)")
wrapped = os.path.join(processed_root, "HAM10000")
show(wrapped, "processed HAM10000 (wrapped candidate)")
assert os.path.isfile(os.path.join(wrapped, "metadata_train.csv")), (
    "Expected wrapped metadata_train.csv not found - check processed dataset packaging"
)
print("\nOK: uppercase raw image folders complete (10015), CSV image_ids resolve, "
      "processed dataset confirmed wrapped under HAM10000/.")
```

---

## Cell 2 — Setup / os.makedirs

```python
import os, sys

os.makedirs("/kaggle/working/src/models", exist_ok=True)
os.makedirs("/kaggle/working/src/evaluation", exist_ok=True)
open("/kaggle/working/src/__init__.py", "w").close()
open("/kaggle/working/src/models/__init__.py", "w").close()
open("/kaggle/working/src/evaluation/__init__.py", "w").close()

sys.path.insert(0, "/kaggle/working")
print("setup OK, sys.path[0]:", sys.path[0])
```

---

## Cell 3 — `%%writefile /kaggle/working/src/models/config.py`

```python
%%writefile /kaggle/working/src/models/config.py
"""Central paths and constants for Phase 6 Stage 1 baseline models."""

from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()

IS_KAGGLE = Path("/kaggle/input").exists()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_DATASETS_ROOT = KAGGLE_INPUT_ROOT / "datasets"
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

KAGGLE_DATASET_SLUGS = {
    "PAD_UFES20": ("mahdavi1202", "skin-cancer"),
    "HAM10000": ("kmader", "skin-cancer-mnist-ham10000"),
}
KAGGLE_DATASET_SUBPATH = {
    "PAD_UFES20": "",
    "HAM10000": "",
}

RAW_ROOT = PROJECT_ROOT / "data" / "raw"


def resolve_image_path(image_path: str) -> Path:
    rel = Path(image_path)
    parts = rel.parts
    if len(parts) < 3 or parts[0] != "data" or parts[1] != "raw":
        raise ValueError(f"Unexpected image_path format: {image_path!r}")
    dataset_dir = parts[2]
    rest = Path(*parts[3:])

    if not IS_KAGGLE:
        return PROJECT_ROOT / rel

    owner_slug = KAGGLE_DATASET_SLUGS.get(dataset_dir)
    if owner_slug is None or owner_slug[0].startswith("REPLACE_WITH"):
        raise RuntimeError(
            f"No Kaggle dataset slug configured for {dataset_dir!r}. "
            f"Run `!ls /kaggle/input/datasets` in the notebook, then set "
            f"KAGGLE_DATASET_SLUGS[{dataset_dir!r}] in src/models/config.py."
        )
    owner, slug = owner_slug
    subpath = KAGGLE_DATASET_SUBPATH.get(dataset_dir, "")
    dataset_root = KAGGLE_DATASETS_ROOT / owner / slug / subpath

    rest_parts = rest.parts
    if rest_parts and rest_parts[0].startswith("imgs_part_"):
        top_dir = rest_parts[0]
        doubled = dataset_root / top_dir / top_dir / Path(*rest_parts[1:])
        if doubled.exists():
            return doubled

    return dataset_root / rest


KAGGLE_PROCESSED_SLUGS = {
    "PAD_UFES20": ("naeemsarkertracer", "pad-ufes20-processed"),
    "HAM10000": ("naeemsarkertracer", "ham10000-processed"),
}
KAGGLE_PROCESSED_WRAPPED = {
    "PAD_UFES20": True,
    "HAM10000": True,
}


def _processed_dir(dataset: str) -> Path:
    if not IS_KAGGLE:
        return PROJECT_ROOT / "data" / "processed" / dataset

    owner_slug = KAGGLE_PROCESSED_SLUGS.get(dataset)
    if owner_slug is None or owner_slug[0].startswith("REPLACE_WITH"):
        raise RuntimeError(
            f"Set KAGGLE_PROCESSED_SLUGS[{dataset!r}] in "
            f"src/models/config.py to the owner/slug of the private Kaggle "
            f"dataset holding data/processed/{dataset}/."
        )
    owner, slug = owner_slug
    root = KAGGLE_DATASETS_ROOT / owner / slug

    # Auto-detect wrapping (don't just trust KAGGLE_PROCESSED_WRAPPED):
    # if root/<Dataset>/metadata_train.csv exists, the zip wrapped the
    # folder itself; if root/metadata_train.csv exists, it didn't.
    wrapped_candidate = root / dataset
    if (wrapped_candidate / "metadata_train.csv").exists():
        return wrapped_candidate
    if (root / "metadata_train.csv").exists():
        return root
    return wrapped_candidate if KAGGLE_PROCESSED_WRAPPED.get(dataset, True) else root


OUTPUT_ROOT = KAGGLE_WORKING_ROOT if IS_KAGGLE else PROJECT_ROOT

IMAGE_INPUT_SIZE = 224

SEEDS = [0, 1, 2]

BATCH_SIZE = 32
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7
LEARNING_RATE_IMAGE = 1e-4
LEARNING_RATE_METADATA = 1e-3
WEIGHT_DECAY = 1e-4


class DatasetConfig:
    def __init__(self, name: str, class_names: list, numeric_features: list,
                 categorical_features: list):
        self.name = name
        self.class_names = class_names
        self.label_to_idx = {n: i for i, n in enumerate(class_names)}
        self.num_classes = len(class_names)
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        processed_dir = _processed_dir(name)
        self.train_csv = processed_dir / "metadata_train.csv"
        self.val_csv = processed_dir / "metadata_val.csv"
        self.test_csv = processed_dir / "metadata_test.csv"

        self.logs_dir = OUTPUT_ROOT / "logs" / name
        self.checkpoints_dir = self.logs_dir / "checkpoints"
        self.baseline_reports_dir = OUTPUT_ROOT / "reports" / name / "baseline"


PAD_UFES20 = DatasetConfig(
    name="PAD_UFES20",
    class_names=[
        "Actinic Keratosis", "Basal Cell Carcinoma", "Melanoma", "Nevus",
        "Seborrheic Keratosis", "Squamous Cell Carcinoma",
    ],
    numeric_features=["age", "diameter_1", "diameter_2"],
    categorical_features=[
        "smoke", "drink", "background_father", "background_mother",
        "pesticide", "sex", "skin_cancer_history", "cancer_history",
        "has_piped_water", "has_sewage_system", "fitspatrick",
        "anatomical_site", "itch", "grew", "hurt", "changed", "bleed",
        "elevation",
    ],
)

HAM10000 = DatasetConfig(
    name="HAM10000",
    class_names=[
        "Actinic Keratosis / Intraepithelial Carcinoma",
        "Basal Cell Carcinoma",
        "Benign Keratosis-like Lesion",
        "Dermatofibroma",
        "Melanoma",
        "Nevus",
        "Vascular Lesion",
    ],
    numeric_features=["age"],
    categorical_features=["sex", "anatomical_site"],
)

DATASETS = {"PAD_UFES20": PAD_UFES20, "HAM10000": HAM10000}


def get_dataset(name: str) -> DatasetConfig:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choices: {list(DATASETS)}")
    return DATASETS[name]
```

---

## Cell 4 — `%%writefile /kaggle/working/src/models/dataset.py`

```python
%%writefile /kaggle/working/src/models/dataset.py
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.models.config import IMAGE_INPUT_SIZE, DatasetConfig, resolve_image_path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResizePad:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        img = TF.resize(img, [new_h, new_w])
        pad_left = (self.size - new_w) // 2
        pad_right = self.size - new_w - pad_left
        pad_top = (self.size - new_h) // 2
        pad_bottom = self.size - new_h - pad_top
        return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)


def build_image_transform(train: bool) -> transforms.Compose:
    ops = [ResizePad(IMAGE_INPUT_SIZE)]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class ImageDataset(Dataset):
    def __init__(self, csv_path: Path, dataset_config: DatasetConfig, train: bool):
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = dataset_config.label_to_idx
        self.transform = build_image_transform(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = resolve_image_path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.label_to_idx[row["disease_label"]]
        return image, label


class MetadataPreprocessor:
    def __init__(self, dataset_config: DatasetConfig):
        self.numeric_features = dataset_config.numeric_features
        self.categorical_features = dataset_config.categorical_features
        self.numeric_means = {}
        self.numeric_stds = {}
        self.categorical_values = {}

    def fit(self, df: pd.DataFrame) -> "MetadataPreprocessor":
        for col in self.numeric_features:
            values = pd.to_numeric(df[col], errors="coerce")
            self.numeric_means[col] = values.mean()
            std = values.std()
            self.numeric_stds[col] = std if std and std > 0 else 1.0
        for col in self.categorical_features:
            values = df[col].astype("string").fillna("__MISSING__")
            self.categorical_values[col] = sorted(values.unique().tolist())
        return self

    @property
    def output_dim(self) -> int:
        numeric_dim = len(self.numeric_features)
        categorical_dim = sum(len(v) for v in self.categorical_values.values())
        return numeric_dim + categorical_dim

    def transform_row(self, row: pd.Series) -> torch.Tensor:
        parts = []
        for col in self.numeric_features:
            raw = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.isna(raw):
                raw = self.numeric_means[col]
            parts.append((raw - self.numeric_means[col]) / self.numeric_stds[col])
        for col in self.categorical_features:
            value = row[col]
            value = "__MISSING__" if pd.isna(value) else str(value)
            categories = self.categorical_values[col]
            one_hot = [1.0 if value == cat else 0.0 for cat in categories]
            if value not in categories:
                one_hot = [0.0] * len(categories)
            parts.extend(one_hot)
        return torch.tensor(parts, dtype=torch.float32)


class MetadataDataset(Dataset):
    def __init__(self, csv_path: Path, dataset_config: DatasetConfig,
                 preprocessor: MetadataPreprocessor):
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = dataset_config.label_to_idx
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        features = self.preprocessor.transform_row(row)
        label = self.label_to_idx[row["disease_label"]]
        return features, label
```

---

## Cell 5 — `%%writefile /kaggle/working/src/models/image_model.py`

```python
%%writefile /kaggle/working/src/models/image_model.py
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
```

---

## Cell 6 — `%%writefile /kaggle/working/src/models/metadata_model.py`

```python
%%writefile /kaggle/working/src/models/metadata_model.py
import torch.nn as nn


class MetadataMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)
```

---

## Cell 7 — `%%writefile /kaggle/working/src/models/train.py`

```python
%%writefile /kaggle/working/src/models/train.py
import argparse
import csv
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.models.config import (
    BATCH_SIZE, EARLY_STOPPING_PATIENCE, LEARNING_RATE_IMAGE,
    LEARNING_RATE_METADATA, NUM_EPOCHS, WEIGHT_DECAY, get_dataset,
)
from src.models.dataset import ImageDataset, MetadataDataset, MetadataPreprocessor
from src.models.image_model import build_efficientnet_b0
from src.models.metadata_model import MetadataMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(train_csv, class_names) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    counts = df["disease_label"].value_counts()
    freqs = np.array([counts.get(name, 0) for name in class_names], dtype=np.float64)
    weights = freqs.sum() / (len(class_names) * freqs)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def train_one_run(dataset_name: str, branch: str, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)

    if branch == "image":
        train_ds = ImageDataset(ds_config.train_csv, ds_config, train=True)
        val_ds = ImageDataset(ds_config.val_csv, ds_config, train=False)
        model = build_efficientnet_b0(num_classes=ds_config.num_classes).to(device)
        lr = LEARNING_RATE_IMAGE
    elif branch == "metadata":
        preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
        train_ds = MetadataDataset(ds_config.train_csv, ds_config, preprocessor)
        val_ds = MetadataDataset(ds_config.val_csv, ds_config, preprocessor)
        model = MetadataMLP(
            input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
        ).to(device)
        lr = LEARNING_RATE_METADATA
    else:
        raise ValueError(f"Unknown branch: {branch}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(
        ds_config.train_csv, ds_config.class_names
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{branch}_seed{seed}"
    checkpoint_path = ds_config.checkpoints_dir / f"{run_name}_best.pt"
    metrics_csv_path = ds_config.logs_dir / f"train_{run_name}.csv"

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1"]
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            start = time.time()
            train_loss, train_macro_f1 = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_macro_f1 = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )
            writer.writerow([epoch, train_loss, train_macro_f1, val_loss, val_macro_f1])
            f.flush()
            elapsed = time.time() - start
            print(
                f"[{dataset_name}/{run_name}] epoch {epoch:02d} "
                f"train_loss={train_loss:.4f} train_macroF1={train_macro_f1:.4f} "
                f"val_loss={val_loss:.4f} val_macroF1={val_macro_f1:.4f} "
                f"({elapsed:.1f}s)"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "dataset": dataset_name,
                        "branch": branch,
                        "seed": seed,
                        "epoch": epoch,
                        "val_macro_f1": val_macro_f1,
                        "num_classes": ds_config.num_classes,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"[{dataset_name}/{run_name}] early stopping at epoch {epoch}")
                    break

    summary = {
        "dataset": dataset_name,
        "branch": branch,
        "seed": seed,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": str(checkpoint_path),
    }
    summary_path = ds_config.logs_dir / f"train_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{dataset_name}/{run_name}] best val macro-F1: "
        f"{best_val_macro_f1:.4f} -> {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Stage 1 training")
    parser.add_argument("--dataset", choices=["PAD_UFES20", "HAM10000"], required=True)
    parser.add_argument("--branch", choices=["image", "metadata"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_one_run(args.dataset, args.branch, args.seed)


if __name__ == "__main__":
    main()
```

---

## Cell 8 — `%%writefile /kaggle/working/src/evaluation/evaluate.py`

```python
%%writefile /kaggle/working/src/evaluation/evaluate.py
import argparse
import json

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.models.config import BATCH_SIZE, get_dataset
from src.models.dataset import ImageDataset, MetadataDataset, MetadataPreprocessor
from src.models.image_model import build_efficientnet_b0
from src.models.metadata_model import MetadataMLP


def load_model(branch: str, checkpoint_path, ds_config, device):
    if branch == "image":
        model = build_efficientnet_b0(num_classes=ds_config.num_classes)
    else:
        preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
        model = MetadataMLP(
            input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_eval_dataset(branch: str, csv_path, ds_config):
    if branch == "image":
        return ImageDataset(csv_path, ds_config, train=False)
    preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
    return MetadataDataset(csv_path, ds_config, preprocessor)


def evaluate(dataset_name: str, branch: str, seed: int, split: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)
    checkpoint_path = ds_config.checkpoints_dir / f"{branch}_seed{seed}_best.pt"
    model = load_model(branch, checkpoint_path, ds_config, device)

    split_csv = {"val": ds_config.val_csv, "test": ds_config.test_csv}[split]
    dataset = build_eval_dataset(branch, split_csv, ds_config)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    class_names = ds_config.class_names
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, labels=list(range(len(class_names))),
        zero_division=0,
    )
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    return {
        "dataset": dataset_name,
        "branch": branch,
        "seed": seed,
        "split": split,
        "macro_f1": macro_f1,
        "accuracy_reference_only": accuracy,
        "per_class_f1": dict(zip(class_names, per_class_f1.tolist())),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_class_order": class_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Stage 1 evaluation")
    parser.add_argument("--dataset", choices=["PAD_UFES20", "HAM10000"], required=True)
    parser.add_argument("--branch", choices=["image", "metadata"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--confirm-final", action="store_true")
    args = parser.parse_args()

    if args.split == "test" and not args.confirm_final:
        raise SystemExit(
            "Refusing to evaluate on the test split without --confirm-final."
        )

    result = evaluate(args.dataset, args.branch, args.seed, args.split)

    ds_config = get_dataset(args.dataset)
    ds_config.baseline_reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        ds_config.baseline_reports_dir
        / f"eval_{args.branch}_seed{args.seed}_{args.split}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"macro-F1 ({args.split}): {result['macro_f1']:.4f}")
    for name, score in result["per_class_f1"].items():
        print(f"  {name}: {score:.4f}")
    print(f"written -> {out_path}")


if __name__ == "__main__":
    main()
```

---

## Cell 9 — Sanity check (config load + real image path resolution)

```python
import sys
for mod in list(sys.modules):
    if mod.startswith("src."):
        del sys.modules[mod]

import pandas as pd
from src.models.config import get_dataset, resolve_image_path

ds_config = get_dataset("HAM10000")
print("num_classes:", ds_config.num_classes)
print("train_csv:", ds_config.train_csv, "exists:", ds_config.train_csv.exists())
print("val_csv:  ", ds_config.val_csv, "exists:", ds_config.val_csv.exists())
print("test_csv: ", ds_config.test_csv, "exists:", ds_config.test_csv.exists())
assert ds_config.train_csv.exists(), "metadata_train.csv not found - check processed dataset nesting"

# Real image_path row from the actual CSV, not a guess
df = pd.read_csv(ds_config.train_csv)
sample_image_path = df.iloc[0]["image_path"]
resolved = resolve_image_path(sample_image_path)
print("\nsample image_path (from CSV):", sample_image_path)
print("resolved filesystem path:    ", resolved)
print("resolved path exists:        ", resolved.exists())
assert resolved.exists(), f"Resolved image path does not exist: {resolved}"

# Check a handful more, not just row 0
import random
random.seed(0)
sample_rows = df.sample(n=min(20, len(df)), random_state=0)
missing = []
for _, row in sample_rows.iterrows():
    p = resolve_image_path(row["image_path"])
    if not p.exists():
        missing.append((row["image_path"], str(p)))
print(f"\nChecked {len(sample_rows)} random rows, missing: {len(missing)}")
if missing:
    print("First few missing:", missing[:5])
assert not missing, "Some resolved image paths do not exist - check folder-verification cell output"

print("\nSANITY CHECK PASSED")
```

---

## Cell 10 — Full model/GPU/dependency check

```python
import torch, sys
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

from src.models.config import get_dataset
from src.models.dataset import ImageDataset, MetadataDataset, MetadataPreprocessor
from src.models.image_model import build_efficientnet_b0
from src.models.metadata_model import MetadataMLP
import pandas as pd

ds_config = get_dataset("HAM10000")

preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
meta_model = MetadataMLP(input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes)
dummy = torch.randn(4, preprocessor.output_dim)
out = meta_model(dummy)
assert out.shape == (4, ds_config.num_classes)
print("metadata model OK, output shape:", out.shape)

train_ds = ImageDataset(ds_config.train_csv, ds_config, train=True)
img, label = train_ds[0]
assert img.shape == (3, 224, 224)
img_model = build_efficientnet_b0(num_classes=ds_config.num_classes)
out = img_model(img.unsqueeze(0))
assert out.shape == (1, ds_config.num_classes)
print("image model OK, output shape:", out.shape)

print("\nALL CHECKS PASSED - ready to train")
```

---

## Cell 11 — Train: image branch, seed 0

```python
!python -m src.models.train --dataset HAM10000 --branch image --seed 0
```

---

## Cell 12 — Train: image branch, seed 1

```python
!python -m src.models.train --dataset HAM10000 --branch image --seed 1
```

---

## Cell 13 — Train: image branch, seed 2

```python
!python -m src.models.train --dataset HAM10000 --branch image --seed 2
```

---

## Cell 14 — Train: metadata branch, seed 0

```python
!python -m src.models.train --dataset HAM10000 --branch metadata --seed 0
```

---

## Cell 15 — Train: metadata branch, seed 1

```python
!python -m src.models.train --dataset HAM10000 --branch metadata --seed 1
```

---

## Cell 16 — Train: metadata branch, seed 2

```python
!python -m src.models.train --dataset HAM10000 --branch metadata --seed 2
```
