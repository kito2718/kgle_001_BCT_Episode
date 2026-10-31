"""Seven-frame candidate-state encoder for the V34B1 redesign."""

from __future__ import annotations

from typing import Any


def make_candidate_state_model(
    n_structural_features: int,
    spatial_widths: tuple[int, int, int] = (12, 24, 48),
    d_model: int = 96,
    embedding_dim: int = 64,
    dropout: float = 0.10,
):
    """Build the all-real temporal state model without action-utility heads."""

    import torch
    import torch.nn as nn

    def group_count(channels: int) -> int:
        for groups in (8, 6, 4, 3, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    class ResidualBlock3D(nn.Module):
        def __init__(self, incoming: int, outgoing: int, stride: int = 1):
            super().__init__()
            groups = group_count(outgoing)
            self.conv1 = nn.Conv3d(
                incoming, outgoing, 3, stride=stride, padding=1, bias=False
            )
            self.norm1 = nn.GroupNorm(groups, outgoing)
            self.conv2 = nn.Conv3d(outgoing, outgoing, 3, padding=1, bias=False)
            self.norm2 = nn.GroupNorm(groups, outgoing)
            self.skip = (
                nn.Identity()
                if incoming == outgoing and stride == 1
                else nn.Conv3d(incoming, outgoing, 1, stride=stride, bias=False)
            )
            self.activation = nn.SiLU()

        def forward(self, values):
            residual = self.skip(values)
            values = self.activation(self.norm1(self.conv1(values)))
            values = self.norm2(self.conv2(values))
            return self.activation(values + residual)

    class SharedFrameEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            first, second, third = spatial_widths
            self.blocks = nn.Sequential(
                ResidualBlock3D(1, first, stride=2),
                ResidualBlock3D(first, first),
                ResidualBlock3D(first, second, stride=2),
                ResidualBlock3D(second, second),
                ResidualBlock3D(second, third, stride=2),
                ResidualBlock3D(third, third),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
            )

        def forward(self, values):
            return self.blocks(values)

    class CandidateStateNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            frame_width = int(spatial_widths[-1])
            self.n_structural_features = int(n_structural_features)
            self.embedding_dim = int(embedding_dim)
            self.frame_encoder = SharedFrameEncoder()
            self.scale_projection = nn.Sequential(
                nn.Linear(2 * frame_width, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
            self.relative_time = nn.Parameter(torch.zeros(7, d_model))
            nn.init.trunc_normal_(self.relative_time, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=2)
            self.temporal_norm = nn.LayerNorm(d_model)
            self.structural = nn.Sequential(
                nn.Linear(n_structural_features, 48),
                nn.LayerNorm(48),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            fused_width = 2 * d_model + 48
            self.state_fusion = nn.Sequential(
                nn.Linear(fused_width, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.incoming_fusion = nn.Sequential(
                nn.Linear(fused_width, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.outgoing_fusion = nn.Sequential(
                nn.Linear(fused_width, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.state_head = nn.Sequential(
                nn.Linear(128, 64), nn.SiLU(), nn.Dropout(dropout), nn.Linear(64, 1)
            )
            self.incoming_head = nn.Sequential(
                nn.Linear(128, 64), nn.SiLU(), nn.Dropout(dropout), nn.Linear(64, 1)
            )
            self.outgoing_head = nn.Sequential(
                nn.Linear(128, 64), nn.SiLU(), nn.Dropout(dropout), nn.Linear(64, 1)
            )
            self.embedding_head = nn.Sequential(
                nn.Linear(128, embedding_dim),
                nn.LayerNorm(embedding_dim),
            )

        @staticmethod
        def _masked_mean(values, valid):
            weight = valid.to(dtype=values.dtype).unsqueeze(-1)
            return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

        def forward(self, local, context, valid, structural) -> dict[str, Any]:
            if local.shape[:2] != context.shape[:2] or local.shape[1] != 7:
                raise ValueError("candidate-state inputs must contain seven aligned frames")
            if valid.shape != local.shape[:2]:
                raise ValueError("candidate-state valid mask has an incompatible shape")
            if not bool(valid[:, 3].all()):
                raise ValueError("the centre candidate frame must always be valid")
            batch, frames = local.shape[:2]
            local_descriptor = self.frame_encoder(
                local.reshape(batch * frames, 1, *local.shape[-3:])
            ).reshape(batch, frames, -1)
            context_descriptor = self.frame_encoder(
                context.reshape(batch * frames, 1, *context.shape[-3:])
            ).reshape(batch, frames, -1)
            tokens = self.scale_projection(
                torch.cat([local_descriptor, context_descriptor], dim=-1)
            )
            tokens = tokens + self.relative_time.unsqueeze(0)
            tokens = self.temporal(tokens, src_key_padding_mask=~valid)
            tokens = self.temporal_norm(tokens)
            structural_descriptor = self.structural(structural)
            center = tokens[:, 3]
            global_descriptor = self._masked_mean(tokens, valid)
            incoming_descriptor = self._masked_mean(tokens[:, :4], valid[:, :4])
            outgoing_descriptor = self._masked_mean(tokens[:, 3:], valid[:, 3:])
            state = self.state_fusion(
                torch.cat([center, global_descriptor, structural_descriptor], dim=-1)
            )
            incoming = self.incoming_fusion(
                torch.cat([center, incoming_descriptor, structural_descriptor], dim=-1)
            )
            outgoing = self.outgoing_fusion(
                torch.cat([center, outgoing_descriptor, structural_descriptor], dim=-1)
            )
            return {
                "state_logit": self.state_head(state).squeeze(-1),
                "incoming_logit": self.incoming_head(incoming).squeeze(-1),
                "outgoing_logit": self.outgoing_head(outgoing).squeeze(-1),
                "embedding": self.embedding_head(state),
            }

    return CandidateStateNetwork()


STATE_ARCHITECTURE = {
    "family": "dual-scale trajectory-aligned candidate-state network",
    "timepoints": 7,
    "local_shape": [7, 9, 17, 17],
    "context_shape": [7, 17, 33, 33],
    "shared_spatial_widths": [12, 24, 48],
    "temporal_layers": 2,
    "d_model": 96,
    "attention_heads": 4,
    "embedding_dim": 64,
    "heads": ["state_logit", "incoming_logit", "outgoing_logit", "embedding"],
    "forbidden_heads": ["retention_harm", "abstention", "action_utility"],
}
