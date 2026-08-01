"""Phase 8B fusion model - concatenates the top-2 backbones (by Phase 8B's
5-backbone comparison, src/models/train.py --backbone) penultimate
embeddings, then a joint classifier head. Same architectural pattern as
fusion_model.py's image+metadata FusionModel, applied to backbone+backbone
instead.

Unlike FusionModel's ImageEmbedder (which hand-replicates EfficientNet-B0's
forward pass up to its penultimate vector), BackboneEmbedder captures the
penultimate features generically via a forward hook on the final
classification Linear layer (src/models/backbones.py's final_linear) - this
works identically for all 5 backbone architectures without hand-writing a
separate forward-pass replica per architecture family.
"""

import torch
import torch.nn as nn

from src.models.backbones import build_backbone, final_linear


class BackboneEmbedder(nn.Module):
    """Wraps a full build_backbone(name, ...) model; forward returns the
    penultimate vector that model's final classification Linear consumed,
    instead of that layer's own output.
    """

    def __init__(self, name: str, num_classes: int):
        super().__init__()
        self.name = name
        self.backbone = build_backbone(name, num_classes=num_classes)
        self._final_linear = final_linear(name, self.backbone)
        self.embed_dim = self._final_linear.in_features
        self._captured = None
        self._final_linear.register_forward_hook(self._capture_input)

    def _capture_input(self, module, args, output):
        self._captured = args[0]

    def load_stage_checkpoint(self, checkpoint_path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.backbone.load_state_dict(checkpoint["model_state_dict"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone(x)  # full forward pass; hook captures the penultimate vector
        return self._captured


class BackboneFusionModel(nn.Module):
    """Runs the same input image through both backbones, concatenates their
    embeddings, and classifies with a joint head one hidden layer deep
    (128-d) - same head shape as fusion_model.py's FusionModel, for
    consistency across the thesis's fusion variants.
    """

    def __init__(self, backbone_a: str, backbone_b: str, num_classes: int):
        super().__init__()
        self.embedder_a = BackboneEmbedder(backbone_a, num_classes)
        self.embedder_b = BackboneEmbedder(backbone_b, num_classes)
        joint_dim = self.embedder_a.embed_dim + self.embedder_b.embed_dim
        self.head = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def load_stage_checkpoints(
        self, checkpoint_path_a, checkpoint_path_b, device: torch.device
    ) -> None:
        self.embedder_a.load_stage_checkpoint(checkpoint_path_a, device)
        self.embedder_b.load_stage_checkpoint(checkpoint_path_b, device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Both embedders see the same image - this fuses two architectural
        # "views" of one picture, not two different pictures. Both backbones
        # share IMAGE_INPUT_SIZE=224 and IMAGENET_MEAN/STD normalization
        # (dataset.py's build_image_transform), so no per-embedder
        # preprocessing split is needed.
        features_a = self.embedder_a(image)
        features_b = self.embedder_b(image)
        joint = torch.cat([features_a, features_b], dim=1)
        return self.head(joint)
