"""Small fully-convolutional 3-D flow network with reusable descriptors."""
import torch
from torch import nn
import torch.nn.functional as F


def local_correlation_flow(a, b, radius=(2, 3, 3), temperature=0.07):
    """Soft-argmax local matching on feature grids, output in grid voxels.

    For each location p in a, score b[p + delta]. Padding candidates are
    masked so borders cannot attract probability mass.
    """
    rz, ry, rx = (int(v) for v in radius)
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    bp = F.pad(b, (rx, rx, ry, ry, rz, rz))
    valid = F.pad(torch.ones_like(b[:, :1]), (rx, rx, ry, ry, rz, rz))
    z, y, x = a.shape[-3:]
    scores, offsets = [], []
    for dz in range(-rz, rz + 1):
        for dy in range(-ry, ry + 1):
            for dx in range(-rx, rx + 1):
                sl = (slice(rz + dz, rz + dz + z),
                      slice(ry + dy, ry + dy + y),
                      slice(rx + dx, rx + dx + x))
                candidate = bp[(...,) + sl]
                ok = valid[(...,) + sl]
                score = (a * candidate).sum(1) / temperature
                scores.append(score.masked_fill(ok[:, 0] == 0, -1e4))
                offsets.append((dz, dy, dx))
    prob = torch.softmax(torch.stack(scores, 1), 1)
    # Contract the candidate axis directly.  Materializing
    # ``prob[:, None] * delta`` creates a Bx3xKxDxHxW temporary (tens of GiB
    # for full-volume batch 16) although the requested output is only Bx3xD...
    delta = a.new_tensor(offsets).T
    return torch.einsum("bkzyx,ck->bczyx", prob, delta)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1), nn.GELU())

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Two-convolution residual block with projection when widths differ."""

    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1))
        self.skip = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, 1)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class FlowNet3D(nn.Module):
    """Pair network returning dense a->b flow and per-frame descriptors."""
    def __init__(self, base_channels=16, feature_dim=16,
                 max_displacement_vox=(4.0, 6.0, 6.0),
                 correlation_radius=(2, 3, 3), correlation_temperature=0.07,
                 multiscale_pair_fusion=False, residual_blocks=False):
        super().__init__()
        c = base_channels
        block = ResidualBlock if residual_blocks else Block
        self.enc0 = block(1, c)
        self.enc1 = block(c, 2 * c)
        self.enc2 = block(2 * c, 4 * c)
        self.multiscale_pair_fusion = bool(multiscale_pair_fusion)
        if self.multiscale_pair_fusion:
            self.pair_fuse = nn.ModuleList([
                nn.Sequential(nn.Conv3d(3 * width, width, 1), nn.GELU())
                for width in (c, 2 * c, 4 * c)
            ])
        else:
            self.fuse = block(8 * c, 4 * c)
        self.dec1 = block(6 * c, 2 * c)
        self.dec0 = block(3 * c, c)
        self.flow_head = nn.Conv3d(c, 3, 3, padding=1)
        self.feature_head = nn.Conv3d(c, feature_dim, 1)
        self.correlation_radius = tuple(correlation_radius)
        self.correlation_temperature = float(correlation_temperature)
        self.register_buffer("max_disp", torch.tensor(max_displacement_vox).view(1, 3, 1, 1, 1))
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def encode(self, x):
        f0 = self.enc0(x)
        f1 = self.enc1(F.avg_pool3d(f0, 2))
        f2 = self.enc2(F.avg_pool3d(f1, 2))
        return f0, f1, f2

    def forward(self, a, b, return_features=False):
        aa, bb = self.encode(a), self.encode(b)
        coarse = local_correlation_flow(aa[1], bb[1], self.correlation_radius,
                                        self.correlation_temperature)
        initial = F.interpolate(coarse, size=aa[0].shape[-3:], mode="trilinear",
                                align_corners=False) * 2.0
        if self.multiscale_pair_fusion:
            fused = [module(torch.cat((x, y, torch.abs(x - y)), 1))
                     for module, x, y in zip(self.pair_fuse, aa, bb)]
            x = fused[2]
            skip1, skip0 = fused[1], fused[0]
        else:
            x = self.fuse(torch.cat((aa[2], bb[2]), 1))
            skip1, skip0 = aa[1], aa[0]
        x = F.interpolate(x, size=aa[1].shape[-3:], mode="trilinear", align_corners=False)
        x = self.dec1(torch.cat((x, skip1), 1))
        x = F.interpolate(x, size=aa[0].shape[-3:], mode="trilinear", align_corners=False)
        x = self.dec0(torch.cat((x, skip0), 1))
        residual = torch.tanh(self.flow_head(x)) * self.max_disp
        flow = initial + residual
        if not return_features:
            return flow
        fa = F.normalize(self.feature_head(aa[0]), dim=1)
        fb = F.normalize(self.feature_head(bb[0]), dim=1)
        return {"flow": flow, "features_a": fa, "features_b": fb}


def build_flow_model(config):
    """FlowNet3D from a checkpoint's serialized model config."""
    params = dict(config)
    architecture = str(params.pop("architecture", "3d")).lower()
    if architecture == "3d":
        return FlowNet3D(**params)
    if architecture == "3d_multiscale":
        params["multiscale_pair_fusion"] = True
        return FlowNet3D(**params)
    if architecture == "3d_residual":
        params["residual_blocks"] = True
        return FlowNet3D(**params)
    raise ValueError(f"unsupported flow model architecture: {architecture}")
