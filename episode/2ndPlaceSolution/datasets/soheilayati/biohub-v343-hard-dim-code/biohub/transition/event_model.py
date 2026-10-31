"""Dense corrected continuation-event heads for the V34B3 decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class EventResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.layers(self.norm(values))


class CorrectedEventMLP(nn.Module):
    """Predict incoming-available and outgoing-available event logits."""

    def __init__(self, input_width: int = 67, width: int = 128, dropout: float = 0.05):
        super().__init__()
        self.input_width = int(input_width)
        self.width = int(width)
        self.dropout = float(dropout)
        self.input = nn.Sequential(
            nn.Linear(self.input_width, self.width),
            nn.LayerNorm(self.width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            EventResidualBlock(self.width, self.dropout),
            EventResidualBlock(self.width, self.dropout),
        )
        self.output = nn.Linear(self.width, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_width:
            raise ValueError(
                f"expected [batch,{self.input_width}] event features, got {tuple(values.shape)}"
            )
        return self.output(self.blocks(self.input(values)))


EVENT_FEATURE_COLUMNS = (
    "state_logit",
    "legacy_incoming_logit",
    "legacy_outgoing_logit",
    *(f"embedding_{index:02d}" for index in range(64)),
)

