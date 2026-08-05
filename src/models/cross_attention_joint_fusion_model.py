"""Phase 8E (Option A) - genuine single joint three-way fusion: both
Phase 8B top-2 backbones (ConvNeXt-Tiny, DenseNet121) AND metadata fused
into one trainable model, unlike Step 4 (Option B)'s two independent
per-backbone models combined only at prediction time via
`evaluate_dual_backbone_ensemble`.

Reuses cross_attention_backbone_fusion_model.py's mechanism (metadata
Query cross-attends over image Key/Value spatial tokens via
nn.MultiheadAttention, both projected into a shared d_model first) - the
only new step is running SpatialBackboneEmbedder for BOTH backbones,
projecting each backbone's tokens into the shared d_model with its OWN
kv_proj (768->d_model for ConvNeXt-Tiny, 1024->d_model for DenseNet121,
since the two backbones' raw token dims differ and cannot be
concatenated directly), and concatenating the two projected 49-token
sequences into a single 98-token Key/Value sequence before the same
attention module Step 4 already validated. No new attention variant.
"""

import torch
import torch.nn as nn

from src.models.cross_attention_fusion_model import MetadataChannelGate
from src.models.fusion_model import MetadataEmbedder
from src.models.spatial_backbone_embedder import SPATIAL_BACKBONE_NAMES, SpatialBackboneEmbedder

JOINT_BACKBONE_NAMES = ("convnext_tiny", "densenet121")


class CrossAttentionJointFusionModel(nn.Module):
    """Metadata (Query) cross-attends over the concatenation of
    ConvNeXt-Tiny's and DenseNet121's 49 spatial image tokens each
    (98 combined Key/Value tokens), each backbone's tokens projected into
    the shared d_model by its own kv_proj before concatenation.
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
        for name in JOINT_BACKBONE_NAMES:
            if name not in SPATIAL_BACKBONE_NAMES:
                raise ValueError(f"Unknown spatial backbone {name!r}")

        self.image_embedder_a = SpatialBackboneEmbedder("convnext_tiny", num_classes)
        self.image_embedder_b = SpatialBackboneEmbedder("densenet121", num_classes)
        self.metadata_embedder = MetadataEmbedder(metadata_input_dim, num_classes)

        self.use_channel_gate = use_channel_gate
        if use_channel_gate:
            self.channel_gate_a = MetadataChannelGate(
                metadata_dim=self.metadata_embedder.embed_dim,
                num_channels=self.image_embedder_a.embed_dim,
            )
            self.channel_gate_b = MetadataChannelGate(
                metadata_dim=self.metadata_embedder.embed_dim,
                num_channels=self.image_embedder_b.embed_dim,
            )

        self.query_proj = nn.Linear(self.metadata_embedder.embed_dim, d_model)
        self.kv_proj_a = nn.Linear(self.image_embedder_a.embed_dim, d_model)
        self.kv_proj_b = nn.Linear(self.image_embedder_b.embed_dim, d_model)
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
        self,
        image_checkpoint_a_path,
        image_checkpoint_b_path,
        metadata_checkpoint_path,
        device: torch.device,
    ) -> None:
        # image_checkpoint_a_path: Step 3 (Phase 8B) PAD_UFES20_Expanded
        # convnext_tiny checkpoint. image_checkpoint_b_path: same for
        # densenet121. metadata_checkpoint_path: Phase 7 Stage 1
        # PAD_UFES20 metadata checkpoint. Same warm-start pattern as
        # Step 4, just applied to both backbones.
        self.image_embedder_a.load_checkpoint(image_checkpoint_a_path, device)
        self.image_embedder_b.load_checkpoint(image_checkpoint_b_path, device)
        self.metadata_embedder.load_stage1(metadata_checkpoint_path, device)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        tokens_a = self.image_embedder_a(image)  # [B, 49, 768]
        tokens_b = self.image_embedder_b(image)  # [B, 49, 1024]
        metadata_embedding = self.metadata_embedder(metadata)  # [B, 64]

        if self.use_channel_gate:
            tokens_a = self.channel_gate_a(tokens_a, metadata_embedding)
            tokens_b = self.channel_gate_b(tokens_b, metadata_embedding)

        proj_a = self.kv_proj_a(tokens_a)  # [B, 49, d_model]
        proj_b = self.kv_proj_b(tokens_b)  # [B, 49, d_model]
        key_value = torch.cat([proj_a, proj_b], dim=1)  # [B, 98, d_model]

        query = self.query_proj(metadata_embedding).unsqueeze(1)  # [B, 1, d_model]
        attended, _ = self.attention(query, key_value, key_value)  # [B, 1, d_model]
        attended = attended.squeeze(1)  # [B, d_model]

        joint = torch.cat([attended, metadata_embedding], dim=1)
        return self.head(joint)
