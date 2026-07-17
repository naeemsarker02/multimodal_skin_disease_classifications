"""Shared PyTorch Datasets for Phase 6 Stage 1 baselines.

Dataset-parameterized (extended 2026-07-13 for HAM10000). Reads
metadata_{train,val,test}.csv (never modifies them). image_path in those
CSVs already points into data/raw/<Dataset>/... - loaded directly, never
copied, per PROJECT_PLAN.md's no-image-copying rule.
"""

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
    """Aspect-ratio-preserving resize to a square canvas.

    Scales the longer side to `size`, then pads the shorter side with
    zeros (black) to reach size x size - avoids the distortion a naive
    stretch-to-square would introduce, given the documented image-size
    heterogeneity (see Project_Tracking.md decision 3).
    """

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
    """Image-only branch: returns (image_tensor, label_idx)."""

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
    """Fits standardization/encoding on the train split only, applies it
    identically to val/test - prevents any val/test statistic (mean,
    std, category set) from leaking into the transform.
    """

    def __init__(self, dataset_config: DatasetConfig):
        self.numeric_features = dataset_config.numeric_features
        self.categorical_features = dataset_config.categorical_features
        self.numeric_means = {}
        self.numeric_stds = {}
        self.categorical_values = {}  # col -> sorted list of seen categories

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
                # unseen category at val/test time (should not happen if
                # fit on train, but guard rather than crash)
                one_hot = [0.0] * len(categories)
            parts.extend(one_hot)
        return torch.tensor(parts, dtype=torch.float32)


class MetadataDataset(Dataset):
    """Metadata-only branch: returns (feature_tensor, label_idx)."""

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


class FusionDataset(Dataset):
    """Phase 7 late-fusion branch: returns (image_tensor, feature_tensor,
    label_idx) for the same row - same image transform as ImageDataset,
    same preprocessor contract as MetadataDataset.
    """

    def __init__(self, csv_path: Path, dataset_config: DatasetConfig,
                 preprocessor: MetadataPreprocessor, train: bool):
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = dataset_config.label_to_idx
        self.transform = build_image_transform(train)
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = resolve_image_path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        features = self.preprocessor.transform_row(row)
        label = self.label_to_idx[row["disease_label"]]
        return image, features, label
