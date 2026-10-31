from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

AUX_FPN_CH = 16


def aux_grid_for_crop(crop_size):
    """Native-Z and half-YX occupancy grid for the supplied crop."""
    z, y, x = (int(s) for s in crop_size)
    if y % 2 or x % 2:
        raise ValueError(f"crop Y,X must be even for half-YX occupancy, got {crop_size}")
    return (z, y // 2, x // 2)


class AuxFPN(nn.Module):
    """Top-down FPN producing occupancy and division logits per frame."""

    def __init__(self, skip_channels, top_ch, out_grid, fpn_ch=AUX_FPN_CH, out_ch=2):
        super().__init__()
        self.out_grid = tuple((int(s) for s in out_grid))
        self.fpn_ch = int(fpn_ch)
        skip_channels = [int(c) for c in skip_channels]
        if not skip_channels:
            raise ValueError("AuxFPN needs at least one encoder skip")
        self.laterals = nn.ModuleList([nn.Conv3d(c, self.fpn_ch, 1) for c in skip_channels])
        self.top_proj = nn.Conv3d(int(top_ch), self.fpn_ch, 1)
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(self.fpn_ch, self.fpn_ch, 3, padding=1),
                    nn.InstanceNorm3d(self.fpn_ch, affine=True),
                    nn.ReLU(inplace=True),
                )
                for _ in skip_channels
            ]
        )
        self.head = nn.Conv3d(self.fpn_ch, int(out_ch), 1)

    def forward(self, skips, top):
        """``skips`` finest-first ``[(B, C, Z, Y, X), ...]``; ``top`` post-transformer grid."""
        x = self.top_proj(top)
        order = list(range(len(skips) - 1, -1, -1))
        for i in order:
            lat = self.laterals[i](skips[i])
            x = F.interpolate(x, size=tuple(lat.shape[2:]), mode="trilinear", align_corners=False)
            x = self.smooth[i](x + lat)
        if tuple(x.shape[2:]) != self.out_grid:
            x = F.interpolate(x, size=self.out_grid, mode="trilinear", align_corners=False)
        return self.head(x)


CLASS_AWARE_FRAMES = 3


def aux_occupancy_probs(aux_logits, class_aware=False):
    """``(B, 3, Z, Y, X)`` occupancy probabilities from raw aux logits (class-aware or not)."""
    p = torch.sigmoid(aux_logits.float())
    if not class_aware:
        return p
    return torch.maximum(p[:, :CLASS_AWARE_FRAMES], p[:, CLASS_AWARE_FRAMES:])


def aux_division_probs(aux_logits, class_aware=False):
    """``(B, 3, Z, Y, X)`` sigmoid(B) = [prev_B, t_B, next_B] from class-aware logits."""
    if not class_aware:
        raise ValueError("aux_division_probs needs class-aware logits (6 channels)")
    p = torch.sigmoid(aux_logits.float())
    if p.shape[1] < 2 * CLASS_AWARE_FRAMES:
        raise ValueError(f"class-aware aux logits need {2 * CLASS_AWARE_FRAMES} channels, got {tuple(p.shape)}")
    return p[:, CLASS_AWARE_FRAMES : 2 * CLASS_AWARE_FRAMES]


PREV_A, T_A, NEXT_A, PREV_B, T_B, NEXT_B = range(2 * CLASS_AWARE_FRAMES)
