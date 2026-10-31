"""Dual-scale temporal retention network used by V31."""

from __future__ import annotations

from typing import Any


def make_retention_model(
    n_structural_features: int,
    include_removal_utility: bool = False,
):
    """Create the preregistered 7-frame, dual-scale multi-task network."""
    import torch
    import torch.nn as nn

    class ResidualBlock3D(nn.Module):
        def __init__(self, incoming: int, outgoing: int, stride: int = 1):
            super().__init__()
            groups = min(8, outgoing)
            self.conv1 = nn.Conv3d(
                incoming, outgoing, 3, stride=stride, padding=1, bias=False
            )
            self.norm1 = nn.GroupNorm(groups, outgoing)
            self.conv2 = nn.Conv3d(
                outgoing, outgoing, 3, padding=1, bias=False
            )
            self.norm2 = nn.GroupNorm(groups, outgoing)
            self.skip = (
                nn.Identity()
                if incoming == outgoing and stride == 1
                else nn.Conv3d(incoming, outgoing, 1, stride=stride, bias=False)
            )
            self.activation = nn.SiLU()

        def forward(self, x):
            residual = self.skip(x)
            x = self.activation(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            return self.activation(x + residual)

    class SharedFrameEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.Sequential(
                ResidualBlock3D(1, 16, stride=2),
                ResidualBlock3D(16, 16),
                ResidualBlock3D(16, 32, stride=2),
                ResidualBlock3D(32, 32),
                ResidualBlock3D(32, 64, stride=2),
                ResidualBlock3D(64, 64),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
            )

        def forward(self, x):
            return self.blocks(x)

    class TemporalRetentionNetwork(nn.Module):
        HEADS = (
            "cell_presence",
            "incoming_link",
            "outgoing_link",
            "retention_harm",
            "abstention",
        )

        def __init__(self):
            super().__init__()
            self.n_structural_features = int(n_structural_features)
            self.frame_encoder = SharedFrameEncoder()
            self.scale_projection = nn.Sequential(
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
            )
            self.relative_time = nn.Parameter(torch.zeros(7, 128))
            nn.init.trunc_normal_(self.relative_time, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=512,
                dropout=0.10,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=2)
            self.temporal_norm = nn.LayerNorm(128)
            self.structural = nn.Sequential(
                nn.Linear(n_structural_features, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Dropout(0.10),
            )
            self.global_fusion = nn.Sequential(
                nn.Linear(128 + 128 + 64, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(0.10),
            )
            self.incoming_fusion = nn.Sequential(
                nn.Linear(128 + 128 + 64, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(0.10),
            )
            self.outgoing_fusion = nn.Sequential(
                nn.Linear(128 + 128 + 64, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(0.10),
            )
            self.heads = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(128, 64),
                    nn.SiLU(),
                    nn.Dropout(0.10),
                    nn.Linear(64, 1),
                )
                for name in self.HEADS
            })
            if include_removal_utility:
                self.heads["removal_utility"] = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.SiLU(),
                    nn.Dropout(0.10),
                    nn.Linear(64, 1),
                )

        @staticmethod
        def _masked_mean(x, valid):
            weight = valid.to(dtype=x.dtype).unsqueeze(-1)
            return (x * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

        def forward(self, local, context, valid, structural) -> dict[str, Any]:
            batch, frames = local.shape[:2]
            local_descriptor = self.frame_encoder(
                local.reshape(batch * frames, 1, *local.shape[-3:])
            ).reshape(batch, frames, 64)
            context_descriptor = self.frame_encoder(
                context.reshape(batch * frames, 1, *context.shape[-3:])
            ).reshape(batch, frames, 64)
            tokens = self.scale_projection(
                torch.cat([local_descriptor, context_descriptor], dim=-1)
            )
            tokens = tokens + self.relative_time.unsqueeze(0)
            tokens = self.temporal(tokens, src_key_padding_mask=~valid)
            tokens = self.temporal_norm(tokens)
            structural_descriptor = self.structural(structural)
            center = tokens[:, 3]
            global_descriptor = self._masked_mean(tokens, valid)
            incoming = self._masked_mean(tokens[:, :4], valid[:, :4])
            outgoing = self._masked_mean(tokens[:, 3:], valid[:, 3:])
            global_fused = self.global_fusion(
                torch.cat([center, global_descriptor, structural_descriptor], dim=-1)
            )
            incoming_fused = self.incoming_fusion(
                torch.cat([center, incoming, structural_descriptor], dim=-1)
            )
            outgoing_fused = self.outgoing_fusion(
                torch.cat([center, outgoing, structural_descriptor], dim=-1)
            )
            output = {
                "cell_presence": self.heads["cell_presence"](global_fused).squeeze(-1),
                "incoming_link": self.heads["incoming_link"](incoming_fused).squeeze(-1),
                "outgoing_link": self.heads["outgoing_link"](outgoing_fused).squeeze(-1),
                "retention_harm": self.heads["retention_harm"](global_fused).squeeze(-1),
                "abstention": self.heads["abstention"](global_fused).squeeze(-1),
            }
            if "removal_utility" in self.heads:
                output["removal_utility"] = self.heads["removal_utility"](
                    global_fused
                ).squeeze(-1)
            return output

    return TemporalRetentionNetwork()


ARCHITECTURE = {
    "family": "dual-scale trajectory-aligned temporal retention network",
    "timepoints": 7,
    "local_shape": [7, 9, 17, 17],
    "context_shape": [7, 17, 33, 33],
    "shared_spatial_widths": [16, 32, 64],
    "normalization": "GroupNorm",
    "activation": "SiLU",
    "temporal_layers": 2,
    "d_model": 128,
    "attention_heads": 4,
    "dropout": 0.10,
    "heads": [
        "cell_presence",
        "incoming_link",
        "outgoing_link",
        "retention_harm",
        "abstention",
    ],
}
