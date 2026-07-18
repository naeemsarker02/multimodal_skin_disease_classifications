"""Cross-attention fusion model for Phase 7 Stage 2 (PAD-UFES-20 only).

Confirmed 2026-07-18 (Project_Tracking.md, "MetaBlock Mechanism Confirmed;
Phase 7 Stage 2 Proposal") that this is NOT a reproduction of Pacheco &
Krohling's MetaBlock - MetaBlock is a channel-wise gated affine transform
(sigmoid(tanh(V*t1) + t2)), uniform across spatial positions within a
channel. This module instead computes genuine per-spatial-location
attention weights: metadata queries EfficientNet-B0's 49 spatial tokens
(the 7x7 pre-pool feature map) via standard multi-head scaled dot-product
attention, so different image regions can be weighted differently
depending on metadata - something channel-wise gating cannot do. Framed
as "cross-attention, contrasted with MetaBlock's channel-gating approach,"
never "MetaBlock-inspired."

Directly addresses Phase 7 Stage 1's diagnosed limitation: late fusion's
1280:64 raw-dimension concatenation let the image branch numerically
dominate. Here, both modalities are projected into a shared d_model before
any interaction, so raw dimension counts no longer mechanically bias the
result.

Reuses MetadataEmbedder from fusion_model.py unchanged (same 64-d Stage 1
metadata embedding). Adds SpatialImageEmbedder (new: stops at the
pre-avgpool feature map instead of the pooled vector) and
CrossAttentionFusionModel (new: metadata-as-query cross-attention + joint
head), alongside - not replacing - Stage 1's ImageEmbedder/FusionModel, so
Stage 1's late-fusion results and checkpoints stay reproducible.
"""

import torch
import torch.nn as nn

from src.models.fusion_model import MetadataEmbedder
from src.models.image_model import build_efficientnet_b0


class SpatialImageEmbedder(nn.Module):
    """Wraps a full build_efficientnet_b0() model; forward returns the
    49 (7x7) spatial tokens of 1280-d each from the pre-avgpool feature
    map, instead of the pooled 1280-d vector ImageEmbedder returns.

    Same full-architecture-wrapping approach as ImageEmbedder (not a
    re-keyed subset), so a Stage 1 image checkpoint's state_dict loads
    with strict=True.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = build_efficientnet_b0(num_classes=num_classes)
        self.embed_dim = self.backbone.classifier[-1].in_features  # 1280

    def load_stage1(self, checkpoint_path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.backbone.load_state_dict(checkpoint["model_state_dict"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)  # [B, 1280, 7, 7]
        b, c, h, w = x.shape
        x = x.flatten(2)  # [B, 1280, 49]
        x = x.transpose(1, 2)  # [B, 49, 1280] - 49 spatial tokens
        return x


class MetadataChannelGate(nn.Module):
    """Optional dual-mechanism add-on (Suresh et al. TG-CAVNet-inspired,
    per Project_Tracking.md's "Future Improvements" - channel-wise gating
    + cross-attention). A metadata-conditioned sigmoid gate over the 1280
    image channels, applied before cross-attention, so metadata reweights
    channels *and* spatially attends rather than either alone.

    TG-CAVNet itself remains only partially captured in
    Literature_Review.md (row #2) - kept as a secondary, optional
    mechanism (can be disabled via use_channel_gate=False), not a primary
    design input pending its own full-text read.
    """

    def __init__(self, metadata_dim: int, num_channels: int):
        super().__init__()
        self.gate = nn.Linear(metadata_dim, num_channels)

    def forward(self, image_tokens: torch.Tensor, metadata_embedding: torch.Tensor) -> torch.Tensor:
        # image_tokens: [B, 49, C], metadata_embedding: [B, metadata_dim]
        channel_scale = torch.sigmoid(self.gate(metadata_embedding))  # [B, C]
        return image_tokens * channel_scale.unsqueeze(1)  # broadcast over 49 tokens


class CrossAttentionFusionModel(nn.Module):
    """Metadata (Query) cross-attends over EfficientNet-B0's 49 spatial
    image tokens (Key/Value) via standard multi-head scaled dot-product
    attention. Both modalities are projected into a shared d_model before
    interaction, so the 1280:64 raw-dimension imbalance that let Stage 1's
    concatenation-based fusion numerically favor the image branch no
    longer applies here.

    use_channel_gate=True (default) enables the optional TG-CAVNet-style
    channel gate ahead of attention (see MetadataChannelGate).
    """

    def __init__(
        self,
        metadata_input_dim: int,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 8,
        use_channel_gate: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.image_embedder = SpatialImageEmbedder(num_classes)
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

    def load_stage1_checkpoints(
        self, image_checkpoint_path, metadata_checkpoint_path, device: torch.device
    ) -> None:
        self.image_embedder.load_stage1(image_checkpoint_path, device)
        self.metadata_embedder.load_stage1(metadata_checkpoint_path, device)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_tokens = self.image_embedder(image)  # [B, 49, 1280]
        metadata_embedding = self.metadata_embedder(metadata)  # [B, 64]

        if self.use_channel_gate:
            image_tokens = self.channel_gate(image_tokens, metadata_embedding)

        query = self.query_proj(metadata_embedding).unsqueeze(1)  # [B, 1, d_model]
        key_value = self.kv_proj(image_tokens)  # [B, 49, d_model]
        attended, _ = self.attention(query, key_value, key_value)  # [B, 1, d_model]
        attended = attended.squeeze(1)  # [B, d_model]

        joint = torch.cat([attended, metadata_embedding], dim=1)
        return self.head(joint)
