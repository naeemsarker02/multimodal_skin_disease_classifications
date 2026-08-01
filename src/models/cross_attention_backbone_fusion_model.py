"""Step 4 fusion model - cross-attention fusion (Phase 7 Stage 2's
mechanism, not FusionModel/BackboneFusionModel's concatenation) between
metadata and one of the two Phase 8B top-2 backbones (ConvNeXt-Tiny,
DenseNet121), parameterized by backbone name.

Deliberately NOT built on backbone_fusion_model.py's BackboneFusionModel:
that model concatenates raw embeddings before a shallow head - the same
pattern cross_attention_fusion_model.py's docstring identifies as letting
one branch numerically dominate (there, image:metadata was 1280:64).
Extending it to a third concatenated metadata vector would reproduce that
already-diagnosed flaw. This module instead reuses the validated
cross-attention mechanism itself (metadata queries the image branch's
spatial tokens via multi-head attention, both projected into a shared
d_model first), applied per-backbone via SpatialBackboneEmbedder.

Structurally identical to CrossAttentionFusionModel
(cross_attention_fusion_model.py), which stays unchanged (hardcoded to
EfficientNet-B0) for Phase 7 Stage 2 reproducibility. MetadataEmbedder
and MetadataChannelGate are reused unmodified from there / from
fusion_model.py.
"""

import torch
import torch.nn as nn

from src.models.cross_attention_fusion_model import MetadataChannelGate
from src.models.fusion_model import MetadataEmbedder
from src.models.spatial_backbone_embedder import SpatialBackboneEmbedder


class CrossAttentionBackboneFusionModel(nn.Module):
    """Metadata (Query) cross-attends over one Phase 8B backbone's 49
    spatial image tokens (Key/Value), same mechanism as
    CrossAttentionFusionModel, with the image side generalized to any
    backbone in SPATIAL_BACKBONE_NAMES via SpatialBackboneEmbedder.
    """

    def __init__(
        self,
        backbone_name: str,
        metadata_input_dim: int,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 8,
        use_channel_gate: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.image_embedder = SpatialBackboneEmbedder(backbone_name, num_classes)
        self.metadata_embedder = MetadataEmbedder(metadata_input_dim, num_classes)

        self.use_channel_gate = use_channel_gate
        if use_channel_gate:
            self.channel_gate = MetadataChannelGate(
                metadata_dim=self.metadata_embedder.embed_dim,
                num_channels=self.image_embedder.embed_dim,
            )

        self.query_proj = nn.Linear(self.metadata_embedder.embed_dim, d_model)
        self.kv_proj = nn.Linear(self.image_embedder.embed_dim, d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=0.1, batch_first=True
        )

        joint_dim = d_model + self.metadata_embedder.embed_dim
        self.head = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def load_warm_start_checkpoints(
        self, image_checkpoint_path, metadata_checkpoint_path, device: torch.device
    ) -> None:
        # image_checkpoint_path: Step 3 (Phase 8B) PAD_UFES20_Expanded
        # backbone checkpoint. metadata_checkpoint_path: Phase 7 Stage 1
        # PAD_UFES20 metadata checkpoint. Deliberately not named
        # "load_stage1_checkpoints" (CrossAttentionFusionModel's name for
        # this method) - the image checkpoint here is a Step 3 artifact
        # from a different dataset, not a Stage 1 one.
        self.image_embedder.load_checkpoint(image_checkpoint_path, device)
        self.metadata_embedder.load_stage1(metadata_checkpoint_path, device)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_tokens = self.image_embedder(image)  # [B, 49, embed_dim]
        metadata_embedding = self.metadata_embedder(metadata)  # [B, 64]

        if self.use_channel_gate:
            image_tokens = self.channel_gate(image_tokens, metadata_embedding)

        query = self.query_proj(metadata_embedding).unsqueeze(1)  # [B, 1, d_model]
        key_value = self.kv_proj(image_tokens)  # [B, 49, d_model]
        attended, _ = self.attention(query, key_value, key_value)  # [B, 1, d_model]
        attended = attended.squeeze(1)  # [B, d_model]

        joint = torch.cat([attended, metadata_embedding], dim=1)
        return self.head(joint)
