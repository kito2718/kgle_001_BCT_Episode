"""Temporal 3D U-Net detector — PLAN.md §9.1.

Takes a short stack of consecutive frames and predicts, for the centre frame:
a cell-centre heatmap, a sub-grid offset, a backward motion vector to the same
cell (or its parent) in the previous frame, and a log-variance for that motion.

Designed for the `[1, 4, 4]` working grid, where the voxel spacing becomes
approximately isotropic at 1.625 um and a whole volume is only 64x64x64 — so
no patching is needed and a full volume fits comfortably in a batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch: int, groups: int = 8) -> nn.GroupNorm:
    # GroupNorm rather than BatchNorm: batches are small and, more importantly,
    # per-sample intensity statistics differ ~6x between embryos, so batch
    # statistics would mix incompatible domains.
    return nn.GroupNorm(min(groups, ch), ch)


class ResBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = _norm(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = _norm(out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class TemporalMix(nn.Module):
    """Light temporal mixing at the bottleneck (PLAN.md §9.1).

    The frames arrive as channels, so the encoder already sees them jointly;
    this adds an explicit cheap channel-wise mixing stage at the coarsest
    resolution, where the receptive field covers plausible inter-frame motion.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.fc1 = nn.Conv3d(ch, ch * 2, 1)
        self.fc2 = nn.Conv3d(ch * 2, ch, 1)
        self.norm = _norm(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(x + self.fc2(F.silu(self.fc1(x)))))


class TemporalUNet3d(nn.Module):
    """Encoder widths 24/48/96/192, decoder with skips, four prediction heads."""

    def __init__(
        self,
        in_frames: int = 3,
        widths: tuple[int, ...] = (24, 48, 96, 192),
        feature_ch: int = 48,
    ):
        super().__init__()
        self.in_frames = in_frames
        self.stem = ResBlock3d(in_frames, widths[0])

        self.downs = nn.ModuleList(
            ResBlock3d(widths[i], widths[i + 1]) for i in range(len(widths) - 1)
        )
        self.bottleneck = TemporalMix(widths[-1])

        rev = list(reversed(widths))
        self.ups = nn.ModuleList(
            ResBlock3d(rev[i] + rev[i + 1], rev[i + 1]) for i in range(len(rev) - 1)
        )
        self.feature = nn.Conv3d(widths[0], feature_ch, 1)

        self.head_heatmap = nn.Conv3d(feature_ch, 1, 1)
        self.head_offset = nn.Conv3d(feature_ch, 3, 1)
        self.head_flow = nn.Conv3d(feature_ch, 3, 1)
        self.head_logvar = nn.Conv3d(feature_ch, 1, 1)

        # Heatmap targets are near-zero almost everywhere; start biased low so
        # the model does not spend early training suppressing a hot volume.
        nn.init.constant_(self.head_heatmap.bias, -4.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.stem(x)
        skips = [h]
        for down in self.downs:
            h = down(F.avg_pool3d(h, 2))
            skips.append(h)

        h = self.bottleneck(h)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            h = up(torch.cat([h, skip], dim=1))

        f = F.silu(self.feature(h))
        return {
            "heatmap": self.head_heatmap(f),
            "offset": self.head_offset(f),
            "flow": self.head_flow(f),
            "flow_logvar": self.head_logvar(f),
            "features": f,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
