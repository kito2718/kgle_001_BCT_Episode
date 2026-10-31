from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from cell_tracking.common.config import SOON_CROP_SIZE as CROP_SIZE
from cell_tracking.soon.attention import EvaTransformerEncoder
from cell_tracking.soon.encoder import SharedSpatialEncoder
from cell_tracking.soon.attention import build_3d_rope_cat
from cell_tracking.soon.attention import timm_3d_rope_dim


def _gaussian_route(g, grid):
    gp = F.adaptive_avg_pool3d(g, tuple((int(v) for v in grid)))
    flat = gp.reshape(gp.shape[0], -1)
    alpha = F.softmax(flat * 3.0, dim=-1)
    return alpha.view(gp.shape[0], 1, *gp.shape[2:])


class _TemporalSoonNet(nn.Module):
    """Weight-shared 3-frame encoder plus an indicator-routed readout."""

    def __init__(self, base=16, feat_dim=96, num_classes=3):
        super().__init__()
        self.base = int(base)
        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.n_frames = 3
        self.encoder_kind = "cnn"
        self.encoder_kw = {}
        self.in_ch = 4
        self.attn = "softmax"
        self.cond_encoder = False
        self.aniso_pool = "mid"
        self.encoder = SharedSpatialEncoder(base=base, out_dim=feat_dim, in_ch=1)

    def _window(self, x, with_skips=False):
        """``(feats, alpha)`` for a ``(B, n_frames + 1, Z, Y, X)`` crop."""
        T = self.n_frames
        g = x[:, T : T + 1]
        frames = x[:, :T]
        inv = None
        if len(x) > 1:
            new = torch.ones(len(x), dtype=torch.bool, device=x.device)
            new[1:] = (frames[1:] != frames[:-1]).flatten(1).any(1)
            if not bool(new.all()):
                frames, inv = (frames[new], new.cumsum(0) - 1)
        frames = frames.unsqueeze(2)
        frames = frames.flatten(0, 1)
        if with_skips:
            f, skips = self.encoder.forward_with_skips(frames)
        else:
            f = self.encoder(frames)
        f = f.unflatten(0, (-1, T))
        if inv is not None:
            f = f[inv]
            if with_skips:
                skips = [m.unflatten(0, (-1, T))[inv].flatten(0, 1) for m in skips]
        feats = [f[:, i] for i in range(T)]
        mid = feats[T // 2]
        alpha = _gaussian_route(g, mid.shape[2:])
        if with_skips:
            return (feats, alpha, skips)
        return (feats, alpha)


class SiameseSpaceTimeSoonNet(_TemporalSoonNet):
    """Shared CNN and bidirectional space-time attention with class-aware occupancy."""

    def __init__(
        self,
        base=16,
        feat_dim=96,
        num_classes=3,
        nheads=4,
        num_layers=4,
        dropout=0.2,
        num_registers=4,
        rope_base=32.0,
        crop_size=CROP_SIZE,
    ):
        if (base, feat_dim, num_classes, nheads, num_layers, rope_base, tuple(crop_size)) != (
            16,
            96,
            3,
            4,
            4,
            32.0,
            (16, 64, 64),
        ) or num_registers not in (0, 4):
            raise ValueError("model settings differ from the packaged soon recipe")
        self.xy_pool = 1
        super().__init__(base=base, feat_dim=feat_dim, num_classes=num_classes)
        self.arch = "soon_stcausal"
        self.nheads = int(nheads)
        self.num_layers = int(num_layers)
        self.causal = False
        self.role_masked = False
        self.role_ball_thresh = 0.2
        self.num_registers = int(num_registers)
        self.use_rope = True
        self.rope_base = float(rope_base)
        self.use_swiglu = True
        self.layerscale_init = 0.0
        timm_3d_rope_dim(self.feat_dim // self.nheads)
        self.crop_size = tuple((int(s) for s in crop_size))
        with torch.no_grad():
            dummy = torch.zeros(1, self.encoder.in_ch, *self.crop_size)
            gtok = self.encoder(dummy)
        self._grid = tuple((int(v) for v in gtok.shape[2:]))
        self.n_tokens = int(gtok.shape[2] * gtok.shape[3] * gtok.shape[4])
        self.spatial_pos = nn.Parameter(torch.randn(1, self.n_tokens, feat_dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, self.n_frames, feat_dim) * 0.02)
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, self.num_registers, feat_dim) * 0.02)
        else:
            self.register_tokens = None
        self.in_ln = nn.LayerNorm(feat_dim)
        self.transformer = EvaTransformerEncoder(
            feat_dim,
            self.nheads,
            num_layers=self.num_layers,
            dropout=float(dropout),
            num_prefix_tokens=self.num_registers,
        )
        self.query = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)
        self.head = nn.Sequential(nn.Linear(feat_dim, feat_dim * 2), nn.ReLU(inplace=True), nn.Linear(feat_dim * 2, 3))
        self.q_ln = nn.LayerNorm(feat_dim)
        self.kv_ln = nn.LayerNorm(feat_dim)
        self.readout = nn.MultiheadAttention(feat_dim, self.nheads, dropout=float(dropout), batch_first=True)
        self.readout_drop = nn.Identity()
        self.head_drop = nn.Identity()
        if hasattr(self.encoder, "drop"):
            self.encoder.drop = nn.Identity()
        self._rope_cache = {}
        self.query_tag = True
        self.query_tag_vec = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)
        self.aux_spheres = True
        self.aux_class_aware = True
        self._build_aux_fpn()

    def _build_aux_fpn(self):
        from cell_tracking.soon.auxiliary import AuxFPN, aux_grid_for_crop

        if not hasattr(self.encoder, "forward_with_skips"):
            raise ValueError("aux_spheres needs encoder.forward_with_skips")
        with torch.no_grad():
            dummy = torch.zeros(1, self.encoder.in_ch, *self.crop_size)
            _, skips = self.encoder.forward_with_skips(dummy)
        skip_ch = [int(s.shape[1]) for s in skips]
        if not skip_ch:
            raise ValueError("encoder.forward_with_skips returned no skip maps")
        out_grid = aux_grid_for_crop(self.crop_size)
        self.aux_fpn = AuxFPN(skip_ch, top_ch=self.feat_dim, out_grid=out_grid, out_ch=2)
        self.aux_out_grid = out_grid

    def _get_rope_cache(self, grid, device, dtype):
        key = (grid, self.n_frames, self.nheads, self.rope_base, str(device), str(dtype))
        hit = self._rope_cache.get(key)
        if hit is not None:
            return hit
        d_head = self.feat_dim // self.nheads
        cat = build_3d_rope_cat(
            grid, n_frames=self.n_frames, d_head=d_head, theta=self.rope_base, device=device, dtype=dtype
        )
        self._rope_cache[key] = cat
        return cat

    def _readout(self, x, return_aux=False):
        """Query-attention readout ``h`` feeding the classifier heads."""
        if return_aux:
            feats, alpha, skips = self._window(x, with_skips=True)
        else:
            feats, alpha = self._window(x)
            skips = None
        B = x.shape[0]
        T, N = (self.n_frames, feats[0].shape[2] * feats[0].shape[3] * feats[0].shape[4])
        grid = tuple((int(v) for v in feats[0].shape[2:]))
        toks = []
        for i, f in enumerate(feats):
            t = f.flatten(2).transpose(1, 2)
            toks.append(t + self.temporal_pos[:, i : i + 1])
        g = x[:, self.n_frames : self.n_frames + 1]
        g_tok = F.adaptive_avg_pool3d(g, grid).reshape(B, 1, -1).transpose(1, 2)
        mid_i = T // 2
        toks[mid_i] = toks[mid_i] + g_tok * self.query_tag_vec
        spat_seq = torch.cat(toks, dim=1)
        if self.register_tokens is not None:
            regs = self.register_tokens.expand(B, -1, -1)
            seq = self.in_ln(torch.cat([regs, spat_seq], dim=1))
        else:
            seq = self.in_ln(spat_seq)
        mask = None
        rope = self._get_rope_cache(grid, x.device, seq.dtype)
        seq = self.transformer(seq, mask=mask, rope=rope)
        if self.register_tokens is not None:
            seq = seq[:, self.num_registers :]
        aux = None
        if return_aux:
            aux = self._aux_decode(skips, seq, B, T, N, grid)
        mid = feats[T // 2]
        centre = (mid * alpha).sum(dim=(2, 3, 4)).unsqueeze(1)
        q = self.q_ln(self.query.expand(B, -1, -1) + centre)
        h, _ = self.readout(q, self.kv_ln(seq), self.kv_ln(seq), need_weights=False)
        h = self.readout_drop(h.squeeze(1))
        return (h, aux) if return_aux else h

    def _aux_decode(self, skips, seq, B, T, N, grid):
        """Per-frame FPN on encoder skips + post-transformer tokens → ``(B, T, Z', Y', X')``."""
        top = seq.reshape(B * T, *grid, -1).permute(0, 4, 1, 2, 3)
        out = self.aux_fpn(skips, top)
        out = out.unflatten(0, (B, T))
        return out.permute(0, 2, 1, 3, 4, 5).reshape(B, -1, *out.shape[3:])

    def forward(self, x, return_aux=False):
        if return_aux:
            h, aux = self._readout(x, return_aux=True)
        else:
            h = self._readout(x)
            aux = None
        feat = self.head_drop(self.head[1](self.head[0](h)))
        logits = self.head[2](feat)
        return (logits, aux) if return_aux else logits
