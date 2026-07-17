"""EfficientNet-B0 wrapper for the PAD-UFES-20 image-only branch.

Chosen over ResNet-50 for this dataset size (~2,298 images) - see
Project_Tracking.md decision (4): far fewer parameters (~5.3M vs
~25.6M), lower overfitting risk, comparable ImageNet accuracy, smaller
memory/compute footprint for free-tier GPU training.
"""

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
