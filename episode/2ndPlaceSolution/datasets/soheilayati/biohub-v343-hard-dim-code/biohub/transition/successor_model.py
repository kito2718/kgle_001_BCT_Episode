"""Parent-group conditional successor model and pure listwise loss."""

from __future__ import annotations

import torch
import torch.nn as nn


class SuccessorEnergyMLP(nn.Module):
    """Residual pair-energy MLP without an event/null parameter."""

    def __init__(
        self,
        n_features: int,
        width: int = 192,
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input(features)
        hidden = torch.nn.functional.silu(hidden + self.block(hidden))
        return self.output(hidden).squeeze(-1)


def grouped_softmax_loss(
    edge_energy: torch.Tensor,
    row_group: torch.Tensor,
    edge_target: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    """Mean cross entropy for exactly-one-positive real-edge groups."""

    if edge_energy.ndim != 1 or row_group.ndim != 1 or edge_target.ndim != 1:
        raise ValueError("energy, group, and target must be one-dimensional")
    if not (len(edge_energy) == len(row_group) == len(edge_target)):
        raise ValueError("row tensors must have the same length")
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    maximum = torch.full(
        (n_groups,), -torch.inf, dtype=edge_energy.dtype, device=edge_energy.device
    )
    maximum.scatter_reduce_(
        0, row_group, edge_energy, reduce="amax", include_self=True
    )
    denominator = torch.zeros_like(maximum)
    denominator.scatter_add_(
        0, row_group, torch.exp(edge_energy - maximum[row_group])
    )
    target_count = torch.zeros_like(maximum)
    target_count.scatter_add_(0, row_group, edge_target)
    if not torch.allclose(target_count, torch.ones_like(target_count)):
        raise ValueError("every successor group must have exactly one positive")
    target_energy = torch.zeros_like(maximum)
    target_energy.scatter_add_(0, row_group, edge_energy * edge_target)
    return (maximum + torch.log(denominator.clamp_min(1e-12)) - target_energy).mean()
