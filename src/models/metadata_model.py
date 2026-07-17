"""MLP for the PAD-UFES-20 metadata-only branch.

Establishes the metadata-alone performance floor for Phase 6 - not
intended to be competitive with the image branch alone (that comparison,
plus fusion, is Phase 7).
"""

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
