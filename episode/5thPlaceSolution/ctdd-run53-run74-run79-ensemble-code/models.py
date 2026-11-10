# Detection and linking share one backbone: a temporal 3D U-Net produces a 32-dim
# descriptor per voxel, a 1x1x1 head turns it into a centre-probability, and a
# cross-attention transformer scores every (t, t+1) node pair from the same features.

import json
import math
import os
import sys
from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from config import *
from utils import *

sys.path.append(BaseConfig.vendor_src)

from tracking_cellmot.models.simple_node_transformer import CrossAttentionBlock


class LinkOutput(NamedTuple):
    edge: torch.Tensor
    appearance: torch.Tensor
    division: torch.Tensor


class TrackingLoss(nn.Module):
    """Detection, identity, appearance, and division losses."""

    def __init__(self, det_neg_weight=0.01, det_sigma_vox=1.0, id_tau=0.1,
                 appearance_pos_weight=10.0, division_pos_weight=20.0):
        super().__init__()
        self.det_neg_weight = det_neg_weight
        self.id_tau = id_tau
        self.appearance_pos_weight = appearance_pos_weight
        self.division_pos_weight = division_pos_weight

        radius = math.ceil(3 * det_sigma_vox)
        axis = torch.arange(-radius, radius + 1)
        offsets = torch.stack(
            torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1
        ).reshape(-1, 3)
        gaussian = torch.exp(
            -offsets.square().sum(-1).float() / (2 * det_sigma_vox ** 2)
        )
        self.register_buffer("offsets", offsets, persistent=False)
        self.register_buffer("gaussian", gaussian, persistent=False)

    @torch.no_grad()
    def _heatmap(self, logits, coords, mask):
        """Place a truncated 3-D Gaussian around every cell centre."""
        batch, _, depth, height, width = logits.shape
        heatmap = logits.new_zeros(batch, depth, height, width)
        limit = torch.tensor(
            [depth - 1, height - 1, width - 1], device=coords.device
        )

        for b in range(batch):
            centres = coords[b, mask[b]].round().long()
            if not len(centres):
                continue
            centres = centres.clamp(min=0)
            centres = torch.minimum(centres, limit)
            points = centres[:, None] + self.offsets[None]
            valid = ((points >= 0) & (points <= limit)).all(-1)
            points = points[valid]
            values = self.gaussian.expand(len(centres), -1)[valid]
            index = (points[:, 0] * height + points[:, 1]) * width + points[:, 2]
            heatmap[b].flatten().scatter_reduce_(
                0, index, values, reduce="amax", include_self=True
            )
        return heatmap

    def detection(self, logits, coords, mask):
        """CenterNet focal loss on a Gaussian cell-centre heatmap."""
        logits = logits.float()
        target = self._heatmap(logits, coords, mask)
        logits = logits[:, 0]
        probability = logits.sigmoid()
        positive = target == 1
        negative = ~positive

        positive_loss = (
            F.softplus(-logits) * (1 - probability).square() * positive
        )
        negative_loss = (
            F.softplus(logits)
            * probability.square()
            * (1 - target).pow(4)
            * negative
        )

        # Normalise the two classes separately: most real cells are unlabelled,
        # so the dense background must not overwhelm the annotated centres.
        positive_count = positive.flatten(1).sum(1).clamp(min=1)
        negative_count = negative.flatten(1).sum(1).clamp(min=1)
        positive_loss = positive_loss.flatten(1).sum(1) / positive_count
        negative_loss = negative_loss.flatten(1).sum(1) / negative_count
        return (positive_loss + self.det_neg_weight * negative_loss).mean()

    def identity(self, feat_t, feat_t1, real_t, real_t1, edge_target):
        """For each t+1 cell, classify its parent among real cells at t."""
        similarity = (
            feat_t.float() @ feat_t1.float().transpose(1, 2) / self.id_tau
        )
        similarity = similarity.masked_fill(~real_t[:, :, None], -1e4)

        logits = similarity.transpose(1, 2)
        targets = edge_target.transpose(1, 2)
        valid = (targets.sum(-1) == 1) & real_t1
        if not bool(valid.any()):
            return similarity.sum() * 0.0, 0

        labels = targets.argmax(-1)
        return F.cross_entropy(logits[valid], labels[valid]), int(valid.sum())

    @staticmethod
    def _binary_classification_loss(logits, labels, valid, pos_weight):
        if not bool(valid.any()):
            return logits.sum() * 0.0, 0, 0, 0, 0

        selected_logits = logits.float()[valid]
        selected_labels = labels.float()[valid]
        loss = F.binary_cross_entropy_with_logits(
            selected_logits, selected_labels,
            pos_weight=selected_logits.new_tensor(pos_weight))
        predicted = selected_logits.sigmoid() >= 0.5
        positive = selected_labels.bool()
        tp = int((predicted & positive).sum())
        fp = int((predicted & ~positive).sum())
        fn = int((~predicted & positive).sum())
        return loss, int(valid.sum()), tp, fp, fn

    def appearance(self, logits, edge_target, target_matches):
        """Classify targets with no detected source in the current frame pair."""
        labels = edge_target.sum(1) == 0
        valid = torch.zeros_like(labels, dtype=torch.bool)
        for b, matches in enumerate(target_matches):
            valid[b, :len(matches)] = matches >= 0
        return self._binary_classification_loss(
            logits, labels, valid, self.appearance_pos_weight)

    def division(self, logits, gt_edges, source_matches):
        """Classify division from the original GT parent out-degree.

        Mapping each detected source back to its GT node preserves a positive
        division label even when one daughter was missed by the detector.
        """
        labels = torch.zeros_like(logits, dtype=torch.float32)
        valid = torch.zeros_like(logits, dtype=torch.bool)

        for b, matches in enumerate(source_matches):
            n = len(matches)
            matched = matches.to(logits.device)
            matched_mask = matched >= 0
            valid[b, :n] = matched_mask
            if bool(matched_mask.any()):
                detected_ids = torch.arange(n, device=logits.device)[matched_mask]
                gt_ids = matched[matched_mask].long()
                labels[b, detected_ids] = (
                    gt_edges[b, gt_ids].sum(-1) == 2
                ).to(labels.dtype)

        return self._binary_classification_loss(
            logits, labels, valid, self.division_pos_weight)


def _conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class _TemporalAttention(nn.Module):
    def __init__(self, channels, n_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x):
        batch, time, channels = x.shape[:3]
        spatial = x.shape[3:]
        size = math.prod(spatial)

        h = x.reshape(batch, time, channels, size).permute(0, 3, 1, 2)
        h = self.norm(h)
        h = torch.stack([
            self.attn(sample, sample, sample, need_weights=False)[0]
            for sample in h
        ])
        h = h.permute(0, 2, 3, 1).reshape(batch, time, channels, *spatial)
        return x + h


class TemporalUNet3D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=32,
        layers: Sequence[int] = (32, 64, 128),
        gradient_checkpointing=True,
        skip_fullres_temporal=True,
    ):
        super().__init__()
        layers = list(layers)
        if len(layers) < 2:
            raise ValueError("layers must contain at least two stages")

        self.gradient_checkpointing = gradient_checkpointing
        self.encoder_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()

        previous = in_channels
        for i, channels in enumerate(layers):
            self.encoder_blocks.append(_conv_block(previous, channels))
            temporal = nn.Identity() if skip_fullres_temporal and i == 0 \
                else _TemporalAttention(channels)
            self.temporal_blocks.append(temporal)
            previous = channels

        self.pool = nn.MaxPool3d(2)
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(layers) - 1, 0, -1):
            self.upsamples.append(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
            self.decoder_blocks.append(
                _conv_block(layers[i] + layers[i - 1], layers[i - 1]))

        self.head = nn.Conv3d(layers[0], out_channels, 1)

    def _run(self, block, x):
        if self.gradient_checkpointing and self.training:
            return _checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x):
        batch, time = x.shape[:2]
        x = x.reshape(batch * time, *x.shape[2:])

        skips = []
        for i, (block, temporal) in enumerate(
                zip(self.encoder_blocks, self.temporal_blocks)):
            if i:
                x = self.pool(x)
            x = self._run(block, x)
            x = temporal(x.reshape(batch, time, *x.shape[1:]))
            x = x.reshape(batch * time, *x.shape[2:])
            if i < len(self.encoder_blocks) - 1:
                skips.append(x)

        bottleneck, decoded = x, []
        for upsample, block, skip in zip(
                self.upsamples, self.decoder_blocks, reversed(skips)):
            x = upsample(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = self._run(block, torch.cat([x, skip], dim=1))
            decoded.append(x)

        head = self.head(x)
        # pyramid, fine to coarse: full-res head, intermediate decoders, bottleneck
        pyramid = [t.reshape(batch, time, *t.shape[1:])
                   for t in (head, *reversed(decoded[:-1]), bottleneck)]
        return pyramid[0], pyramid


class NodeTransformer(nn.Module):
    """Cross-frame node matching with appearance and division heads."""

    def __init__(
        self,
        feat_dim=33,
        hidden_dim=128,
        n_heads=4,
        n_blocks=4,
        mlp_ratio=2.0,
        dropout=0.3,
        pair_chunk_size=32,
    ):
        super().__init__()
        self.pair_chunk_size = pair_chunk_size
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.appearance_head = nn.Linear(hidden_dim, 1)
        self.division_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _safe_mask(mask):
        if mask is None:
            return None
        mask = mask.clone()
        mask[:, 0] |= ~mask.any(dim=1)
        return mask

    def forward(
        self,
        feat_t,
        feat_t1,
        coords_t,
        coords_t1,
        mask_t=None,
        mask_t1=None,
    ):
        mask_t = self._safe_mask(mask_t)
        mask_t1 = self._safe_mask(mask_t1)
        q = self.norm_in(self.proj(feat_t))
        k = self.norm_in(self.proj(feat_t1))

        for block in self.blocks:
            def update(query, context, mask, current_block=block):
                return current_block(query, context, kv_mask=mask)

            if torch.is_grad_enabled():
                q = _checkpoint(update, q, k, mask_t1, use_reentrant=False)
                k = _checkpoint(update, k, q, mask_t, use_reentrant=False)
            else:
                q = update(q, k, mask_t1)
                k = update(k, q, mask_t)

        q = self.norm_out(q)
        k = self.norm_out(k)
        chunks = []
        chunk_size = self.pair_chunk_size or q.shape[1]

        for start in range(0, q.shape[1], chunk_size):
            q_chunk = q[:, start:start + chunk_size]
            coord_chunk = coords_t[:, start:start + chunk_size]

            def score_pairs(query, target, source_coord, target_coord):
                n_source, n_target = query.shape[1], target.shape[1]
                query = query.unsqueeze(2).expand(-1, -1, n_target, -1)
                target = target.unsqueeze(1).expand(-1, n_source, -1, -1)
                relative = (
                    source_coord.unsqueeze(2) - target_coord.unsqueeze(1)
                ) / 100.0
                return self.pair_mlp(
                    torch.cat([query, target, relative], dim=-1)
                ).squeeze(-1)

            if torch.is_grad_enabled():
                scores = _checkpoint(
                    score_pairs, q_chunk, k, coord_chunk, coords_t1,
                    use_reentrant=False,
                )
            else:
                scores = score_pairs(q_chunk, k, coord_chunk, coords_t1)
            chunks.append(scores)

        edge = torch.cat(chunks, dim=1)
        appearance = self.appearance_head(k).squeeze(-1)
        division = self.division_head(q).squeeze(-1)
        return LinkOutput(edge, appearance, division)


def _maybe_checkpoint(fn, *args):
    # recompute in backward, as the v1 linker does, to bound training memory
    if torch.is_grad_enabled():
        return _checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


class NodeTransformerV2(nn.Module):
    """Same-frame self-attention before cross-frame attention; pair scores see the
    displacement in micrometres."""

    def __init__(self, feat_dim, hidden_dim=128, n_heads=4, n_blocks=4,
                 mlp_ratio=2.0, dropout=0.1, distance_unit_um=5.0,
                 pair_chunk_size=32):
        super().__init__()
        self.distance_unit_um = distance_unit_um
        self.pair_chunk_size = pair_chunk_size
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.self_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        # pair input: h_i, h_j, displacement / unit (3), distance / unit (1)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.appearance_head = nn.Linear(hidden_dim, 1)
        self.division_head = nn.Linear(hidden_dim, 1)

    def forward(self, feat_t, feat_t1, um_t, um_t1, mask_t=None, mask_t1=None):
        mask_t = NodeTransformer._safe_mask(mask_t)
        mask_t1 = NodeTransformer._safe_mask(mask_t1)
        q = self.norm_in(self.proj(feat_t))
        k = self.norm_in(self.proj(feat_t1))
        for self_block, cross_block in zip(self.self_blocks, self.cross_blocks):
            q = _maybe_checkpoint(self_block, q, q, mask_t)
            k = _maybe_checkpoint(self_block, k, k, mask_t1)
            q, k = (_maybe_checkpoint(cross_block, q, k, mask_t1),
                    _maybe_checkpoint(cross_block, k, q, mask_t))
        q = self.norm_out(q)
        k = self.norm_out(k)

        def score(source, um_source, target, um_target):
            displacement = (
                (um_target.unsqueeze(1) - um_source.unsqueeze(2)) / self.distance_unit_um
            ).to(source.dtype)
            pair = torch.cat([
                source.unsqueeze(2).expand(-1, -1, target.shape[1], -1),
                target.unsqueeze(1).expand(-1, source.shape[1], -1, -1),
                displacement,
                displacement.norm(dim=-1, keepdim=True),
            ], dim=-1)
            return self.pair_mlp(pair).squeeze(-1)

        chunk = self.pair_chunk_size or q.shape[1]
        scores = [
            _maybe_checkpoint(score, q[:, s:s + chunk], um_t[:, s:s + chunk], k, um_t1)
            for s in range(0, q.shape[1], chunk)
        ]
        edge = torch.cat(scores, dim=1)
        appearance = self.appearance_head(k).squeeze(-1)
        division = self.division_head(q).squeeze(-1)
        return LinkOutput(edge, appearance, division)


class MyUnet(nn.Module):
    def __init__(self, config, cfg):
        super().__init__()

        self.cfg = cfg
        out_ch = int(config["unet_out_channels"])
        layers = tuple(config["unet_layers"])
        self.unet = TemporalUNet3D(
            in_channels=1,
            out_channels=out_ch,
            layers=layers,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )
        # 1x1x1: pure per-voxel projection, the U-Net already did the spatial mixing.
        self.detect_head = nn.Conv3d(out_ch, 1, kernel_size=1)

        identity_dim = int(config["identity_dim"])
        self.identity_head = nn.Sequential(
            nn.Linear(out_ch, out_ch),
            nn.GELU(),
            nn.Linear(out_ch, identity_dim),
        )

        # "v1": run53 linker on the full-res head feature at the integer peak.
        # "v2": self + cross attention linker on head, decoder and bottleneck
        #       features sampled at the refined centre.
        self.link_head = config.get("link_head", "v1")
        self.node_channels = [out_ch] if self.link_head == "v1" else [out_ch, *layers[1:]]
        feat_dim = sum(self.node_channels) + identity_dim + 4 * cfg.pos_per_dim
        if self.link_head == "v1":
            self.transformer = NodeTransformer(
                feat_dim=feat_dim,
                hidden_dim=cfg.hidden_dim,
                n_heads=cfg.n_heads,
                n_blocks=cfg.n_blocks,
                dropout=cfg.transformer_dropout,
            )
        else:
            self.transformer = NodeTransformerV2(
                feat_dim, cfg.hidden_dim, cfg.n_heads, cfg.n_blocks,
                dropout=cfg.transformer_dropout,
                distance_unit_um=cfg.pool_kernel_um,
            )

    def encode(self, image):
        head, pyramid = self.unet(image[:, :, None])
        point_logit = [self.detect_head(head[:, i]) for i in range(head.shape[1])]
        return pyramid[:len(self.node_channels)], point_logit

    def forward_unet(self, image):
        levels, point_logit = self.encode(image)
        # per frame, the scale maps the linker samples from
        frames = [[level[:, i] for level in levels] for i in range(levels[0].shape[1])]
        return frames, point_logit

    def sample_node_features(self, levels, peaks, zyx):
        # levels: one frame's scale maps (V, C, Z, Y, X); peaks / zyx: (N, 3) integer /
        # refined network coordinates. Returns (V, N, sum C).
        if self.link_head == "v1":
            z, y, x = (peaks[:, i].long().clamp(0, levels[0].shape[2 + i] - 1)
                       for i in range(3))
            return levels[0][:, :, z, y, x].transpose(1, 2)
        # trilinear sampling at the refined centre; with align_corners=False one
        # normalised grid addresses the same physical point at every scale
        shape = torch.tensor(levels[0].shape[2:], dtype=torch.float32, device=zyx.device)
        grid = ((zyx.float() + 0.5) / shape * 2 - 1).flip(-1)  # (x, y, z) order
        grid = grid.reshape(1, 1, 1, -1, 3).expand(levels[0].shape[0], -1, -1, -1, -1)
        return torch.cat([
            F.grid_sample(level, grid.to(level.dtype), mode="bilinear",
                          padding_mode="border", align_corners=False)
            .flatten(2).transpose(1, 2)
            for level in levels
        ], dim=-1)

    def node_features(self, levels, det_logit, frame, coords, pos, mask):
        # Batched training path over padded nodes. v2 refines the integer peaks first
        # and returns the refined coordinates with their position embedding.
        coords, pos = coords.clone(), pos.clone()
        feats = levels[0].new_zeros(*coords.shape[:2], sum(self.node_channels))
        for b in range(coords.shape[0]):
            n = int(mask[b].sum())
            if not n:
                continue
            peaks = coords[b, :n].clone()
            if self.link_head != "v1":
                coords[b, :n] = refine_peaks_parabolic(
                    det_logit[b].float().sigmoid(), peaks.long())
                pos[b, :n] = embed_position(coords[b, :n], frame, self.cfg)
            feats[b, :n] = self.sample_node_features(
                [level[b, frame][None] for level in levels], peaks, coords[b, :n])[0]
        return feats, coords, pos

    def identity_embedding(self, feature):
        # identity is learned on the full-res head feature only
        return F.normalize(self.identity_head(feature[..., :self.node_channels[0]]), dim=-1)

    def predict_links(self, feature0, feature1, coord0, coord1, pos0, pos1,
                      mask0=None, mask1=None, coord_scale=None):
        identity0 = self.identity_embedding(feature0)
        identity1 = self.identity_embedding(feature1)
        node0 = torch.cat([feature0, identity0, pos0.to(feature0.dtype)], dim=-1)
        node1 = torch.cat([feature1, identity1, pos1.to(feature1.dtype)], dim=-1)
        if self.link_head == "v1":
            return self.transformer(
                node0, node1, coord0.to(feature0.dtype), coord1.to(feature0.dtype),
                mask0, mask1)
        scale = torch.as_tensor(
            coord_scale, dtype=coord0.dtype, device=coord0.device).reshape(-1, 1, 3)
        return self.transformer(node0, node1, coord0 * scale, coord1 * scale, mask0, mask1)


def _load_model(cfg, checkpoint, config_path, device):
    if os.path.basename(os.path.dirname(os.path.dirname(checkpoint))) == "run79":
        from models_run79 import _load_model as load_run79
        return load_run79(cfg, checkpoint, config_path, device)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    trained_shape = config.get("volume_shape")
    if trained_shape and tuple(trained_shape) != tuple(cfg.volume_shape):
        raise ValueError(
            f"checkpoint was trained with volume_shape={trained_shape}, "
            f"but inference uses {cfg.volume_shape}"
        )
    model = MyUnet(config, cfg)
    model = load_checkpoint(model, checkpoint, device)
    steps = max((int(v) for k, v in model.state_dict().items()
                 if k.endswith("num_batches_tracked")),
                default=0)
    print(f"loaded weights: {checkpoint} | steps={steps:,} | config={config}")
    return model


def load_model(cfg=InferConfig, device=None):
    return load_models(cfg, device)[0]


def resolve_checkpoints(cfg=InferConfig):
    return [
        name if os.path.isabs(name) else os.path.join(cfg.weights_dir, name)
        for name in cfg.checkpoint_paths
    ]


def load_models(cfg=InferConfig, device=None, checkpoints=None):
    device = device or torch.device(
        cfg.device if torch.cuda.is_available() else "cpu")
    checkpoints = resolve_checkpoints(cfg) if checkpoints is None else checkpoints
    if not checkpoints:
        raise ValueError("checkpoint_paths must not be empty")
    models = []
    for checkpoint in checkpoints:
        folder = os.path.dirname(checkpoint)
        models.append(_load_model(
            cfg,
            checkpoint,
            os.path.join(folder, "config.json"),
            device,
        ))
    return models
