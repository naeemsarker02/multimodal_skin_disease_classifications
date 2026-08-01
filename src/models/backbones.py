"""Phase 8B backbone registry - 5 pretrained image backbones for the
image-only branch, plus a generic penultimate-embedding wrapper used by
backbone_fusion_model.py.

Same one-line-swap pattern as image_model.py's build_efficientnet_b0: load
ImageNet1K pretrained weights, replace only the final classification layer
with nn.Linear(in_features, num_classes). See docs/Project_Tracking.md,
"Phase 8B+8C — Master Plan Adopted" (2026-07-29) for why these 5 and not
others - spans ~5M-29M params so the comparison tests architecture capacity,
not just architecture-family trivia.
"""

import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    convnext_tiny,
    densenet121,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet50,
)


def _build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_mobilenet_v3_large(num_classes: int) -> nn.Module:
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_densenet121(num_classes: int) -> nn.Module:
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def _build_resnet50(num_classes: int) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _build_convnext_tiny(num_classes: int) -> nn.Module:
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# name -> (builder, final-linear locator). The locator finds the same
# nn.Linear the builder just replaced, given an already-built model -
# needed by BackboneEmbedder (backbone_fusion_model.py) to read
# embed_dim and hook the penultimate features, without duplicating each
# architecture's classifier layout in a second place.
BACKBONE_REGISTRY = {
    "efficientnet_b0": (_build_efficientnet_b0, lambda m: m.classifier[-1]),
    "mobilenet_v3_large": (_build_mobilenet_v3_large, lambda m: m.classifier[-1]),
    "densenet121": (_build_densenet121, lambda m: m.classifier),
    "resnet50": (_build_resnet50, lambda m: m.fc),
    "convnext_tiny": (_build_convnext_tiny, lambda m: m.classifier[-1]),
}

BACKBONE_NAMES = list(BACKBONE_REGISTRY)


def build_backbone(name: str, num_classes: int) -> nn.Module:
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone {name!r}; choices: {BACKBONE_NAMES}")
    builder, _ = BACKBONE_REGISTRY[name]
    return builder(num_classes)


def final_linear(name: str, model: nn.Module) -> nn.Linear:
    """Locates the same nn.Linear build_backbone(name, ...) just installed
    as the final classification layer, for an already-built model of
    architecture `name`."""
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone {name!r}; choices: {BACKBONE_NAMES}")
    _, locate = BACKBONE_REGISTRY[name]
    return locate(model)
