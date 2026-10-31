"""3D U-Net model (reused from train_unet3d_lit)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cell_tracking.common.config import Channels, DEFAULT_N_RAYS, POOL_SCHEDULE

ARCHS = ("plain",)
NORMS = ("batch", "instance", "group")
ACTS = ("relu", "prelu")
GN_GROUPS = 8


def _norm_act(co, norm, act):
    """One (norm, activation) pair for `_block`."""
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
    if act not in ACTS:
        raise ValueError(f"act must be one of {ACTS}, got {act!r}")
    if norm == "batch":
        n = nn.BatchNorm3d(co)
    elif norm == "instance":
        n = nn.InstanceNorm3d(co, affine=True, eps=1e-05)
    else:
        if co % GN_GROUPS:
            raise ValueError(f"GroupNorm needs width divisible by {GN_GROUPS}, got {co}")
        n = nn.GroupNorm(GN_GROUPS, co, eps=1e-05)
    a = nn.ReLU(inplace=True) if act == "relu" else nn.PReLU(num_parameters=co)
    return (n, a)


def _block(ci, co, norm="batch", act="relu"):
    """Defaults MUST stay batch/relu: they reproduce the champion trunk bit-for-bit."""
    n1, a1 = _norm_act(co, norm, act)
    n2, a2 = _norm_act(co, norm, act)
    return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1), n1, a1, nn.Conv3d(co, co, 3, padding=1), n2, a2)


class _UnetCellPoseNet(nn.Module):
    """Shared output heads + input padding for every unetcellpose backbone."""

    def _build_heads(self, feat_ch, aux_in, horizons, rep, n_rays, contact=False):
        """``feat_ch``: channels of the finest decoder output."""
        self.ch = Channels(horizons, rep=rep, n_rays=n_rays, contact=contact)
        self.out_flow = nn.Conv3d(feat_ch, 3, 1) if self.ch.rep == "flow" else None
        self.out_dist = nn.Conv3d(feat_ch, self.ch.n_rays, 1) if self.ch.rep == "stardist" else None
        self.out_track = nn.Conv3d(feat_ch, 3 * horizons, 1) if horizons else None
        self.out_prob = nn.Conv3d(feat_ch, 1, 1)
        self.out_ttd = nn.Conv3d(feat_ch, 1, 1)
        self.out_contact = nn.Conv3d(feat_ch, 1, 1) if self.ch.contact is not None else None
        aux_ch = self.ch.aux().out_ch
        self.aux2 = nn.Conv3d(aux_in[0], aux_ch, 1)
        self.aux3 = nn.Conv3d(aux_in[1], aux_ch, 1)

    @property
    def out_inst(self):
        """The instance head for whichever representation this net was built for."""
        return self.out_flow if self.ch.rep == "flow" else self.out_dist

    def _heads(self, d1):
        parts = [self.out_inst(d1)]
        if self.out_track is not None:
            parts.append(self.out_track(d1))
        parts += [self.out_prob(d1), self.out_ttd(d1)]
        if self.out_contact is not None:
            parts.append(self.out_contact(d1))
        return torch.cat(parts, 1)

    def _set_mult(self, strides):
        m = [1, 1, 1]
        for st in strides:
            m = [a * b for a, b in zip(m, st)]
        self._mult = tuple(m)

    def _pad(self, x):
        z, y, xx = x.shape[-3:]
        mz, my, mx = self._mult
        pz, py, px = (-z % mz, -y % my, -xx % mx)
        if pz or py or px:
            x = F.pad(x, (0, px, 0, py, 0, pz))
        return (x, (z, y, xx))

    def _heads_and_d1(self, d1, orig):
        """Crop decoder features and run heads — shared by ``forward`` and ``forward_with_d1``."""
        z, y, xx = orig
        return (self._heads(d1)[..., :z, :y, :xx], d1[..., :z, :y, :xx])


class UNet3D(_UnetCellPoseNet):
    """3D U-Net with ANISOTROPIC per-stage pooling (POOL_SCHEDULE)."""

    def __init__(
        self,
        in_ch=1,
        base=24,
        horizons=0,
        pool_schedule=POOL_SCHEDULE,
        rep="flow",
        n_rays=DEFAULT_N_RAYS,
        contact=False,
        norm="batch",
        act="relu",
    ):
        super().__init__()
        self.pool_schedule = tuple((tuple((int(s) for s in st)) for st in pool_schedule))
        self.norm, self.act = (norm, act)
        b = lambda ci, co: _block(ci, co, norm=norm, act=act)
        self.e1 = b(in_ch, base)
        self.e2 = b(base, base * 2)
        self.e3 = b(base * 2, base * 4)
        self.pools = nn.ModuleList([nn.MaxPool3d(k) for k in self.pool_schedule])
        self.bott = b(base * 4, base * 8)
        s0, s1, s2 = self.pool_schedule
        self.u3 = nn.ConvTranspose3d(base * 8, base * 4, s2, stride=s2)
        self.d3 = b(base * 8, base * 4)
        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, s1, stride=s1)
        self.d2 = b(base * 4, base * 2)
        self.u1 = nn.ConvTranspose3d(base * 2, base, s0, stride=s0)
        self.d1 = b(base * 2, base)
        self._build_heads(base, (base * 2, base * 4), horizons, rep, n_rays, contact=contact)
        self._set_mult(self.pool_schedule)

    def _encode(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pools[0](e1))
        e3 = self.e3(self.pools[1](e2))
        return [e1, e2, e3, self.bott(self.pools[2](e3))]

    def _decode(self, e1, e2, e3, b):
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return (d1, d2, d3)

    def _trunk(self, x):
        xp, orig = self._pad(x)
        e1, e2, e3, b = self._encode(xp)
        d1, d2, d3 = self._decode(e1, e2, e3, b)
        return (d1, d2, d3, orig)

    def forward(self, x):
        d1, _, _, orig = self._trunk(x)
        return self._heads_and_d1(d1, orig)[0]

    def forward_with_d1(self, x):
        d1, _, _, orig = self._trunk(x)
        return self._heads_and_d1(d1, orig)

    def forward_with_aux(self, x):
        d1, d2, d3, (z, y, xx) = self._trunk(x)
        return (self._heads(d1)[..., :z, :y, :xx], [self.aux2(d2), self.aux3(d3)])


def build_backbone(arch="plain", in_offsets=None, **kw):
    if arch != "plain":
        raise ValueError(f"unsupported detector architecture: {arch!r}")
    return UNet3D(**kw)
