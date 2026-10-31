from __future__ import annotations
import torch.nn as nn


class SharedSpatialEncoder(nn.Module):
    """3D CNN extracting localized spatial tokens."""

    def __init__(self, base=16, out_dim=96, in_ch=1):
        super().__init__()
        self.base = int(base)
        self.out_dim = int(out_dim)
        self.in_ch = int(in_ch)
        self.aniso_pool = "mid"
        self.dropout = float(0.0)
        chs = [self.in_ch, base, base * 2, base * 4]
        blocks = []
        for (a, b), pool in zip(zip(chs, chs[1:]), list(((2, 2, 2), (2, 2, 2), (1, 2, 2)))):
            blocks += [
                nn.Conv3d(a, b, 3, padding=1),
                nn.InstanceNorm3d(b, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(pool),
            ]
        blocks += [
            nn.Conv3d(base * 4, out_dim, 3, padding=1),
            nn.InstanceNorm3d(out_dim, affine=True),
            nn.ReLU(inplace=True),
        ]
        self.backbone = nn.Sequential(*blocks)
        self.drop = nn.Identity()

    def forward(self, x):
        return self.drop(self.backbone(x))

    def forward_with_skips(self, x):
        """Token grid plus per-pool-stage maps (finest-first) for the train-only aux FPN."""
        skips = []
        i = 0
        n = len(self.backbone)
        while i + 3 < n:
            x = self.backbone[i](x)
            x = self.backbone[i + 1](x)
            x = self.backbone[i + 2](x)
            nxt = self.backbone[i + 3]
            if isinstance(nxt, nn.MaxPool3d):
                x = nxt(x)
                skips.append(x)
                i += 4
            else:
                i += 3
                break
        while i < n:
            x = self.backbone[i](x)
            i += 1
        return (self.drop(x), skips)
