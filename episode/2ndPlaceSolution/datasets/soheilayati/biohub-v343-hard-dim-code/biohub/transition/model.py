"""Conditional predecessor energy model for the V34 transition graph."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransitionEnergyMLP(nn.Module):
    """Small residual MLP plus a learned explicit-null energy."""

    def __init__(
        self,
        n_features: int,
        width: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(n_features, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.output = nn.Linear(width, 1)
        self.null_energy = nn.Parameter(torch.tensor(0.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input(features)
        hidden = torch.nn.functional.silu(hidden + self.block(hidden))
        return self.output(hidden).squeeze(-1)


def listwise_null_loss(
    edge_energy: torch.Tensor,
    row_group: torch.Tensor,
    edge_target: torch.Tensor,
    null_target: torch.Tensor,
    null_energy: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    """Cross entropy over every group's edges plus one explicit null state."""

    if edge_energy.ndim != 1 or row_group.ndim != 1:
        raise ValueError("edge_energy and row_group must be one-dimensional")
    if len(edge_energy) != len(row_group) or len(edge_target) != len(edge_energy):
        raise ValueError("row tensors must have the same length")
    if len(null_target) != n_groups:
        raise ValueError("null_target must have one value per group")

    max_edge = torch.full(
        (n_groups,), -torch.inf, dtype=edge_energy.dtype, device=edge_energy.device
    )
    max_edge.scatter_reduce_(
        0, row_group, edge_energy, reduce="amax", include_self=True
    )
    null = null_energy.expand(n_groups)
    maximum = torch.maximum(max_edge, null)
    denominator = torch.zeros_like(maximum)
    denominator.scatter_add_(
        0, row_group, torch.exp(edge_energy - maximum[row_group])
    )
    denominator = denominator + torch.exp(null - maximum)
    log_partition = maximum + torch.log(denominator.clamp_min(1e-12))

    positive_edge = torch.zeros_like(maximum)
    positive_edge.scatter_add_(0, row_group, edge_energy * edge_target)
    target_energy = positive_edge + null * null_target
    return (log_partition - target_energy).mean()
