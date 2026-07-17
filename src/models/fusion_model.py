"""Late-fusion model for Phase 7 Stage 1 (PAD-UFES-20 only).

Concatenates the penultimate-layer embeddings from each Stage 1 branch -
EfficientNet-B0's 1280-d classifier[-1] input (image_model.py) and
MetadataMLP's 64-d pre-final-layer output (metadata_model.py) - then
feeds the 1344-d joint vector through its own small classifier head.

Each embedder wraps the *full* Stage 1 architecture (not a re-keyed
subset) so a Stage 1 checkpoint's state_dict loads with strict=True -
no manual key remapping to get wrong. The embedder's forward pass simply
stops short of the final Linear that Stage 1 used for its own
single-branch prediction.

Deliberate limitation, logged in Project_Tracking.md rather than treated
as a bug: 1280:64 is a large dimensionality imbalance, so the image
branch will likely dominate this concatenated representation numerically
even with a deeper joint head. That's acceptable for a late-fusion
baseline - it's expected motivation for Phase 7 Stage 2 (cross-attention
fusion), not something to fix here.
"""

import torch
import torch.nn as nn

from src.models.image_model import build_efficientnet_b0
from src.models.metadata_model import MetadataMLP


class ImageEmbedder(nn.Module):
    """Wraps a full build_efficientnet_b0() model; forward returns the
    1280-d vector that Stage 1's classifier[-1] consumed, instead of
    that layer's output.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = build_efficientnet_b0(num_classes=num_classes)
        self.embed_dim = self.backbone.classifier[-1].in_features

    def load_stage1(self, checkpoint_path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.backbone.load_state_dict(checkpoint["model_state_dict"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier[0](x)  # dropout only; identity at eval
        return x


class MetadataEmbedder(nn.Module):
    """Wraps a full MetadataMLP; forward returns the 64-d vector that
    Stage 1's final Linear(64, num_classes) consumed, instead of that
    layer's output.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.backbone = MetadataMLP(input_dim=input_dim, num_classes=num_classes)
        # Same module objects as self.backbone.net[:-1] - loading
        # self.backbone's state_dict updates these parameters in place.
        self.embedder_net = nn.Sequential(*list(self.backbone.net.children())[:-1])
        self.embed_dim = 64

    def load_stage1(self, checkpoint_path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.backbone.load_state_dict(checkpoint["model_state_dict"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedder_net(x)


class FusionModel(nn.Module):
    """Concatenates image + metadata embeddings, classifies with a joint
    head one hidden layer deep (128-d) rather than a single Linear, so
    the head has room to learn a real weighting between the 1280-d and
    64-d branches instead of the image branch dominating by dimension
    count alone.
    """

    def __init__(self, metadata_input_dim: int, num_classes: int):
        super().__init__()
        self.image_embedder = ImageEmbedder(num_classes)
        self.metadata_embedder = MetadataEmbedder(metadata_input_dim, num_classes)
        joint_dim = self.image_embedder.embed_dim + self.metadata_embedder.embed_dim
        self.head = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def load_stage1_checkpoints(
        self, image_checkpoint_path, metadata_checkpoint_path, device: torch.device
    ) -> None:
        self.image_embedder.load_stage1(image_checkpoint_path, device)
        self.metadata_embedder.load_stage1(metadata_checkpoint_path, device)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_features = self.image_embedder(image)
        metadata_features = self.metadata_embedder(metadata)
        joint = torch.cat([image_features, metadata_features], dim=1)
        return self.head(joint)
