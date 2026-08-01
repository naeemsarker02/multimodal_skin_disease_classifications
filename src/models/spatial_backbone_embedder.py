"""Step 4 - generic spatial (pre-pool) embedder for the two Phase 8B
top-2 backbones (ConvNeXt-Tiny, DenseNet121), for use as the Key/Value
side of cross_attention_backbone_fusion_model.py's cross-attention.

Not a generalization of cross_attention_fusion_model.py's
SpatialImageEmbedder (kept unchanged there, hardcoded to EfficientNet-B0,
for Phase 7 Stage 2 reproducibility) - a separate, parallel implementation
covering only the two backbones Step 4 actually needs.

Per-backbone pre-pool extraction verified directly against torchvision's
real forward-pass source (not assumed):
  - convnext_tiny: ConvNeXt._forward_impl is features -> avgpool ->
    classifier; classifier[0] is a LayerNorm2d applied AFTER pooling. So
    raw model.features(x) is already the natural pre-pool representation -
    same pattern as EfficientNet-B0's SpatialImageEmbedder.
  - densenet121: DenseNet.forward is features -> F.relu(features) ->
    adaptive_avg_pool2d -> classifier. The ReLU is applied OUTSIDE
    .features (which ends in a bare BatchNorm, norm5, with no
    activation) - so .features(x) alone is not what the pretrained
    weights were trained to have pooled. F.relu must be applied
    explicitly before treating the map as spatial tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import build_backbone, final_linear

SPATIAL_BACKBONE_NAMES = ("convnext_tiny", "densenet121")


class SpatialBackboneEmbedder(nn.Module):
    """Wraps a full build_backbone(name, ...) model; forward returns the
    7x7 spatial tokens from that architecture's pre-pool feature map,
    instead of the pooled penultimate vector BackboneEmbedder
    (backbone_fusion_model.py) returns.
    """

    def __init__(self, name: str, num_classes: int):
        super().__init__()
        if name not in SPATIAL_BACKBONE_NAMES:
            raise ValueError(
                f"Unknown spatial backbone {name!r}; choices: {SPATIAL_BACKBONE_NAMES}"
            )
        self.name = name
        self.backbone = build_backbone(name, num_classes=num_classes)
        # in_features of the final classification Linear equals the
        # pre-pool feature map's channel count for both architectures
        # here (no channel-changing op between pooling and that Linear).
        self.embed_dim = final_linear(name, self.backbone).in_features

    def load_checkpoint(self, checkpoint_path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.backbone.load_state_dict(checkpoint["model_state_dict"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "convnext_tiny":
            feature_map = self.backbone.features(x)  # [B, 768, 7, 7]
        elif self.name == "densenet121":
            feature_map = F.relu(self.backbone.features(x))  # [B, 1024, 7, 7]
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"Unknown spatial backbone {self.name!r}")
        b, c, h, w = feature_map.shape
        tokens = feature_map.flatten(2)  # [B, C, 49]
        tokens = tokens.transpose(1, 2)  # [B, 49, C]
        return tokens
