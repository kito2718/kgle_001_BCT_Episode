"""Transformer edge head over unetcellpose instance features."""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
from cell_tracking.common.config import SCALE as NATIVE_VOXEL_UM
from cell_tracking.linker.layers import SimpleNodeTransformer
from cell_tracking.linker.intensity import INTENSITY_PAIR_DIM
from cell_tracking.linker.occupancy import OCC_FORK_DIM, OCC_PAIR_DIM, occupancy_pair_features
from cell_tracking.linker.device_const import device_const
from cell_tracking.linker.motion import FLOW_PAIR_DIM, MOTION_PAIR_DIM, UM_DIST_PAIR_DIM
from cell_tracking.linker.pair_features import CONFIG_KIND_DIM, FORK_EDGE_DIM, FORK_GEOM_DIM, FOV_FEAT_DIM, KIND_DAUGHTER_SOON_DIM, PAIR_UM_SCALE, SHAPE_PAIR_DIM, SOON_N_CLASSES, _AXIS, _feat_block_mask, flow_pair_features, fov_distance_features, shape_pair_features
from cell_tracking.linker.fork_head import ForkHeadMixin


CHECKPOINT_METADATA = {
    "flow_feats": True,
    "shape_feats": True,
    "use_volume": True,
    "fork_head": True,
    "soon_fork": True,
    "fork_vol_conserv": True,
    "local_attention": True,
    "fork_hierarchical": True,
    "fork_kind_occupancy": True,
    "fork_kind_geometry": True,
    "fork_kind_expected": True,
    "intensity_pair": True,
    "intensity_fork": True,
    "occupancy_pair": True,
    "occupancy_fork": True,
    "occupancy_fork_daughter": True,
    "shape_residual_direction": True,
    "fov_feats": True,
}


class UnetCellPoseEdgeTransformer(ForkHeadMixin, SimpleNodeTransformer):
    def __init__(
        self,
        feat_dim: int = 39,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.3,
        pair_chunk_size: int | None = 32,
        fork_hidden_dim: int = 64,
        fork_topk: int = 16,
        fork_train_topk: int = 16,
        attn_radius_um: float = 30.0,
        attn_k: int = 16,
    ):
        super().__init__(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pair_chunk_size=pair_chunk_size,
            self_attention=False,
        )
        self.motion_feats = bool(False)
        self.flow_feats = bool(True)
        self.flow_mean_impute = False
        self.um_dist = bool(False)
        self.shape_feats = bool(True)
        self.use_volume = bool(True)
        self.no_parent = bool(False)
        self.symmetric_update = bool(False)
        self.matchability = bool(False)
        self.soon_pair = bool(False)
        self.intensity_pair = bool(True)
        self.occupancy_pair = bool(True)
        self.edge_context_pool = bool(False)
        self.context_block_dim = 0
        pair_extra = 4 + 0 + FLOW_PAIR_DIM + 0 + SHAPE_PAIR_DIM
        pair_extra_soon = pair_extra + 0
        pair_extra_int = pair_extra_soon + INTENSITY_PAIR_DIM
        pair_extra_occ = pair_extra_int + OCC_PAIR_DIM
        pair_mlp_extra = pair_extra_occ + 0
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + pair_mlp_extra, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        with torch.no_grad():
            self.pair_mlp[0].weight[:, hidden_dim * 2 + pair_extra_soon : hidden_dim * 2 + pair_extra_int].zero_()
        with torch.no_grad():
            self.pair_mlp[0].weight[:, hidden_dim * 2 + pair_extra_int : hidden_dim * 2 + pair_extra_occ].zero_()
        self.edge_attention = bool(False)
        self.edge_k = int(8)
        self.n_edge_blocks = int(2)
        self.edge_max_um = float(8.0)
        self.edge_rope = bool(False)
        self.edge_rope_base = float(10000.0)
        self.edge_checkpoint = bool(False)
        self.edge_topm = int(0)
        self.edge_stage = None
        self.fork_head = bool(True)
        self.fork_hidden_dim = int(fork_hidden_dim)
        self.fork_topk = int(fork_topk)
        if self.fork_topk < 1:
            raise ValueError(f"fork_topk must be >= 1, got {fork_topk}")
        self.fork_train_topk = int(fork_train_topk)
        if self.fork_train_topk < self.fork_topk:
            raise ValueError(f"fork_train_topk must be >= fork_topk, got {fork_train_topk} < {fork_topk}")
        self.soon_fork = bool(True)
        self.fork_hierarchical = bool(True)
        self.soon_fork_stats = bool(False)
        self.soon_temporal = bool(False)
        self.sister_feats = bool(False)
        self.intensity_fork = bool(True)
        self.occupancy_fork = bool(True)
        self.occupancy_fork_daughter = bool(True)
        self.fork_kind_occupancy = bool(True)
        self.fork_kind_geometry = bool(True)
        self.fork_kind_expected = bool(True)
        self.fork_kind_edge_context = bool(False)
        self.fork_vol_conserv = bool(True)
        self.soon_null = bool(False)
        self.fork_mlp = None
        self.fork_config_mlp = None
        self.fork_kind_mlp = None
        self.fork_null_head = None
        pair_hidden = hidden_dim // 2
        config_input_dim = (
            hidden_dim
            + pair_hidden
            + pair_hidden
            + FORK_GEOM_DIM
            + FORK_EDGE_DIM
            + CONFIG_KIND_DIM
            + self._fork_extra_width()
        )
        self.fork_config_mlp = nn.Sequential(
            nn.Linear(config_input_dim, self.fork_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.fork_hidden_dim, 1),
        )
        self.fork_kind_bias = nn.Parameter(torch.zeros(CONFIG_KIND_DIM))
        kind_in = SOON_N_CLASSES + OCC_FORK_DIM + FORK_GEOM_DIM + KIND_DAUGHTER_SOON_DIM + 0
        self.fork_kind_mlp = nn.Sequential(nn.Linear(kind_in, 16), nn.GELU(), nn.Linear(16, 2))
        nn.init.zeros_(self.fork_kind_mlp[0].weight)
        nn.init.zeros_(self.fork_kind_mlp[0].bias)
        with torch.no_grad():
            self.fork_kind_mlp[-1].bias.copy_(torch.tensor([4.0, 0.0]))
        null_in_dim = hidden_dim + 0
        self.fork_null_head = nn.Sequential(
            nn.Linear(null_in_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.zeros_(self.fork_config_mlp[-1].weight)
        nn.init.zeros_(self.fork_config_mlp[-1].bias)
        with torch.no_grad():
            self.fork_config_mlp[0].weight[:, -OCC_FORK_DIM:].zero_()
        self.local_attention = bool(True)
        self.attn_radius_um = float(attn_radius_um)
        self.attn_k = int(attn_k)
        if self.attn_radius_um <= 0 and self.attn_k <= 0:
            raise ValueError("local_attention needs attn_radius_um > 0 or attn_k > 0")
        self.soon_node = bool(False)
        self.shape_residual_direction = bool(True)
        self.fov_feats = bool(True)
        self.fov_node_proj = None
        self.fov_node_proj = nn.Linear(FOV_FEAT_DIM, hidden_dim)
        nn.init.zeros_(self.fov_node_proj.weight)
        nn.init.zeros_(self.fov_node_proj.bias)
        self.soon_node_proj = None
        self.soon_hidden_dim = int(0)
        self.soon_hidden_proj = None
        self.feat_block_dropout = float(0.0)
        self._node_block_slices = self._build_node_block_slices(feat_dim, True)
        self._pair_block_slices = self._build_pair_block_slices(hidden_dim, False, True, False, True, True)
        self.no_parent_mode = str("scalar")

    @staticmethod
    def locality_mask(
        coords_a: torch.Tensor,
        coords_b: torch.Tensor,
        *,
        radius_um: float,
        k: int,
        mask_a: torch.Tensor | None = None,
        mask_b: torch.Tensor | None = None,
        voxel_size=NATIVE_VOXEL_UM,
    ) -> torch.Tensor:
        """``(B, N_a, N_b)`` bool: b is inside *radius_um* of a, or among a's *k* nearest."""
        vs = device_const(voxel_size, coords_a.dtype, coords_a.device)
        d = torch.cdist(coords_a * vs, coords_b * vs)
        big = torch.finfo(d.dtype).max / 4
        if mask_b is not None:
            d = d.masked_fill(~mask_b.unsqueeze(1), big)
        keep = torch.zeros_like(d, dtype=torch.bool)
        if radius_um > 0:
            keep |= d <= radius_um
        n_b = d.shape[-1]
        if k > 0 and n_b > 0:
            idx = d.topk(min(int(k), n_b), dim=-1, largest=False).indices
            keep.scatter_(-1, idx, True)
        if mask_b is not None:
            keep &= mask_b.unsqueeze(1)
        if mask_a is not None:
            keep &= mask_a.unsqueeze(-1)
        return keep

    def _attention_masks(self, coords_t, coords_t1, mask_t, mask_t1) -> dict:
        """Cross- and self-attention locality masks for `encode_frames`, or empty."""
        r, k = (self.attn_radius_um, self.attn_k)
        fwd = self.locality_mask(coords_t, coords_t1, radius_um=r, k=k, mask_a=mask_t, mask_b=mask_t1)
        rev = self.locality_mask(coords_t1, coords_t, radius_um=r, k=k, mask_a=mask_t1, mask_b=mask_t)
        cross = fwd | rev.transpose(-1, -2)
        out = {"attn_t": cross, "attn_t1": cross.transpose(-1, -2)}
        if self.self_blocks is not None:
            st = self.locality_mask(coords_t, coords_t, radius_um=r, k=k, mask_a=mask_t, mask_b=mask_t)
            st1 = self.locality_mask(coords_t1, coords_t1, radius_um=r, k=k, mask_a=mask_t1, mask_b=mask_t1)
            out["self_attn_t"] = st | st.transpose(-1, -2)
            out["self_attn_t1"] = st1 | st1.transpose(-1, -2)
        return out

    def _node_extra(self, coords_t, coords_t1, soon_t, soon_t1, soon_hidden_t=None, soon_hidden_t1=None) -> dict:
        """Every additive node channel for `encode_frames`: soon, then field of view."""
        out = self._soon_extra(soon_t, soon_t1, soon_hidden_t, soon_hidden_t1)
        w = self.fov_node_proj.weight
        ft = self.fov_node_proj(fov_distance_features(coords_t.to(w.dtype)))
        ft1 = self.fov_node_proj(fov_distance_features(coords_t1.to(w.dtype)))
        out = {
            "extra_t": ft if "extra_t" not in out else out["extra_t"] + ft,
            "extra_t1": ft1 if "extra_t1" not in out else out["extra_t1"] + ft1,
        }
        return out

    def _soon_extra(self, soon_t, soon_t1, soon_hidden_t=None, soon_hidden_t1=None) -> dict:
        """The soon node channels for `encode_frames`, or empty."""
        extra_t = extra_t1 = None
        if extra_t is None:
            return {}
        return {"extra_t": extra_t, "extra_t1": extra_t1}

    @staticmethod
    def _build_node_block_slices(feat_dim: int, has_geom: bool) -> list[slice]:
        from cell_tracking.linker.features import GEOM_FEAT_DIM, POS_FEAT_DIM

        blocks: list[slice] = []
        geom_w = GEOM_FEAT_DIM if has_geom else 0
        pos_start = feat_dim - geom_w - POS_FEAT_DIM
        app_end = pos_start - 2
        if app_end > 0:
            blocks.append(slice(0, app_end))
        blocks.append(slice(app_end, pos_start))
        blocks.append(slice(pos_start, pos_start + POS_FEAT_DIM))
        if geom_w:
            blocks.append(slice(feat_dim - geom_w, feat_dim))
        return blocks

    @staticmethod
    def _build_pair_block_slices(
        hidden_dim: int, motion: bool, flow: bool, um_dist: bool, shape: bool, use_volume: bool
    ) -> list[slice]:
        blocks: list[slice] = []
        off = hidden_dim * 2
        off += 3
        if use_volume:
            off += 1
        if motion:
            blocks.append(slice(off, off + MOTION_PAIR_DIM))
            off += MOTION_PAIR_DIM
        if flow:
            blocks.append(slice(off, off + FLOW_PAIR_DIM))
            off += FLOW_PAIR_DIM
        if um_dist:
            blocks.append(slice(off, off + UM_DIST_PAIR_DIM))
            off += UM_DIST_PAIR_DIM
        if shape:
            blocks.append(slice(off, off + SHAPE_PAIR_DIM))
            off += SHAPE_PAIR_DIM
        return blocks

    def _geom_blocks(self, feat_t: torch.Tensor, feat_t1: torch.Tensor):
        """The trailing geometry block of each frame's node features."""
        from cell_tracking.linker.features import GEOM_FEAT_DIM

        w = feat_t.shape[-1]
        if w < GEOM_FEAT_DIM + 1:
            raise ValueError(
                f"shape_feats needs a cache built with include_geom=True (last {GEOM_FEAT_DIM} dims); got {w}-wide features"
            )
        return (feat_t[..., w - GEOM_FEAT_DIM :], feat_t1[..., w - GEOM_FEAT_DIM :])

    def _pair_encode(self, parts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(h, e)`` from concatenated pair features."""
        hidden_width = self.pair_mlp[-1].in_features
        if parts.shape[0] == 0:
            return (parts.new_zeros((0, hidden_width)), parts.new_zeros((0,)))
        h = self.pair_mlp[:-1](parts)
        e = self.pair_mlp[-1](h).squeeze(-1)
        return (h, e)

    @staticmethod
    def _batchify_context(ctx: dict, unbatched: bool) -> dict:
        """Add a batch dim to unbatched context tensors."""
        if not unbatched or not ctx:
            return ctx
        want_ndim = {
            "ctx_has_prev": 1,
            "ctx_has_next": 1,
            "ctx_feat_tm1": 3,
            "ctx_feat_t2": 3,
            "ctx_coords_tm1": 3,
            "ctx_coords_t2": 3,
            "ctx_volumes_tm1": 2,
            "ctx_volumes_t2": 2,
            "ctx_mask_tm1": 2,
            "ctx_mask_t2": 2,
            "ctx_vel_tm1": 3,
            "ctx_vel_t1": 3,
            "ctx_flow_tm1": 3,
            "ctx_flow_t1": 3,
            "ctx_prev_mask": 3,
            "ctx_next_mask": 3,
            "ctx_soon_tm1": 3,
            "ctx_soon_t2": 3,
            "ctx_soon_hidden_tm1": 3,
            "ctx_soon_hidden_t2": 3,
            "ctx_intensity_tm1": 3,
            "ctx_intensity_t2": 3,
            "ctx_prev_occ_next": 3,
            "ctx_prev_has_occ_next": 3,
            "ctx_prev_occ_prev": 3,
            "ctx_prev_has_occ_prev": 3,
            "ctx_prev_peak_next": 3,
            "ctx_next_occ_next": 3,
            "ctx_next_has_occ_next": 3,
            "ctx_next_occ_prev": 3,
            "ctx_next_has_occ_prev": 3,
            "ctx_next_peak_next": 3,
        }
        out = {}
        for key, value in ctx.items():
            if value is None:
                out[key] = None
                continue
            ndim = want_ndim.get(key)
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            if ndim is not None:
                while tensor.ndim < ndim:
                    tensor = tensor.unsqueeze(0)
            out[key] = tensor
        return out

    def _pair_parts_at(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        log_v_t: torch.Tensor,
        log_v_t1: torch.Tensor,
        vel_t: torch.Tensor | None,
        flow_t: torch.Tensor | None,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        b: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
        soon_t: torch.Tensor | None = None,
        soon_t1: torch.Tensor | None = None,
        intensity_t: torch.Tensor | None = None,
        intensity_t1: torch.Tensor | None = None,
        occ_next: torch.Tensor | None = None,
        has_next: torch.Tensor | None = None,
        occ_prev: torch.Tensor | None = None,
        has_prev: torch.Tensor | None = None,
        peak_next: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        prev_block: torch.Tensor | None = None,
        next_block: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pair-MLP input rows for the indexed ``(b, i, j)`` candidates."""
        in_dim = self.pair_mlp[0].in_features
        if b.numel() == 0:
            return q.new_zeros((0, in_dim))
        parts = [q[b, i], k[b, j], (coords_t[b, i] - coords_t1[b, j]) / 100.0]
        parts.append((log_v_t[b, i] - log_v_t1[b, j]).unsqueeze(-1))
        vs = device_const(NATIVE_VOXEL_UM, coords_t.dtype, coords_t.device)
        delta_um = coords_t1[b, j] * vs - coords_t[b, i] * vs
        if flow_t is None:
            flow_t = coords_t.new_zeros((*coords_t.shape[:2], 3))
        pred_um = coords_t[b, i] * vs + flow_t[b, i]
        residual_um = coords_t1[b, j] * vs - pred_um
        residual_norm = residual_um.norm(dim=-1, keepdim=True)
        flow_norm = flow_t[b, i].norm(dim=-1, keepdim=True)
        parts.append(torch.cat([residual_um / 100.0, residual_norm / PAIR_UM_SCALE, flow_norm / PAIR_UM_SCALE], dim=-1))
        gt, gt1 = self._geom_blocks(feat_t, feat_t1)
        axis_t = gt[b, i, _AXIS]
        axis_t1 = gt1[b, j, _AXIS]
        axis_t = axis_t / axis_t.norm(dim=-1, keepdim=True).clamp(min=1e-06)
        axis_t1 = axis_t1 / axis_t1.norm(dim=-1, keepdim=True).clamp(min=1e-06)
        move = residual_um
        direction = move / move.norm(dim=-1, keepdim=True).clamp(min=1e-06)
        cos_axes = (axis_t * axis_t1).sum(-1).abs()
        cos_t_move = (axis_t * direction).sum(-1).abs()
        cos_t1_move = (axis_t1 * direction).sum(-1).abs()
        d_logax = (gt[b, i, 0] - gt1[b, j, 0]).abs()
        d_elong = (gt[b, i, 3] - gt1[b, j, 3]).abs()
        parts.append(torch.stack([cos_axes, cos_t_move, cos_t1_move, d_logax, d_elong], dim=-1))
        if intensity_t is None or intensity_t1 is None:
            raise ValueError("intensity_pair=True requires intensity_t and intensity_t1")
        it = intensity_t[b, i]
        it1 = intensity_t1[b, j]
        d_lsum_signed = it1[..., 3] - it[..., 3]
        d_lsum_abs = d_lsum_signed.abs()
        d_mean = (it1[..., 0] - it[..., 0]).abs()
        d_max = it1[..., 1] - it[..., 1]
        parts.append(torch.stack([d_lsum_abs, d_lsum_signed, d_mean, d_max], dim=-1))
        if (
            occ_next is None
            or has_next is None
            or occ_prev is None
            or (has_prev is None)
            or (peak_next is None)
            or (pair_mask is None)
        ):
            raise ValueError(
                "occupancy_pair=True requires occ_next/has_next/occ_prev/has_prev/peak_next and pair_mask (point --occupancy-dir at a sidecar from build_occupancy_sidecar.py)"
            )
        occ_feats = occupancy_pair_features(occ_next, has_next, occ_prev, has_prev, peak_next, coords_t1, pair_mask)
        parts.append(occ_feats[b, i, j])
        return torch.cat(parts, dim=-1)

    def _sparse_pair_logits(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        log_v_t: torch.Tensor,
        log_v_t1: torch.Tensor,
        vel_t: torch.Tensor | None,
        flow_t: torch.Tensor | None,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        pair_mask: torch.Tensor,
        *,
        return_hidden: bool = False,
        soon_t: torch.Tensor | None = None,
        soon_t1: torch.Tensor | None = None,
        intensity_t: torch.Tensor | None = None,
        intensity_t1: torch.Tensor | None = None,
        occ_next: torch.Tensor | None = None,
        has_next: torch.Tensor | None = None,
        occ_prev: torch.Tensor | None = None,
        has_prev: torch.Tensor | None = None,
        peak_next: torch.Tensor | None = None,
        prev_block: torch.Tensor | None = None,
        next_block: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Score only supported pairs and scatter them into a dense-shaped tensor."""
        batch, n_t, n_t1 = (q.shape[0], q.shape[1], k.shape[1])
        b, i, j = torch.where(pair_mask)
        parts = self._pair_parts_at(
            q,
            k,
            coords_t,
            coords_t1,
            log_v_t,
            log_v_t1,
            vel_t,
            flow_t,
            feat_t,
            feat_t1,
            b,
            i,
            j,
            soon_t=soon_t,
            soon_t1=soon_t1,
            intensity_t=intensity_t,
            intensity_t1=intensity_t1,
            occ_next=occ_next,
            has_next=has_next,
            occ_prev=occ_prev,
            has_prev=has_prev,
            peak_next=peak_next,
            pair_mask=pair_mask,
            prev_block=prev_block,
            next_block=next_block,
        )
        pair_hidden, edge_scores = self._pair_encode(parts)
        linear = b * (n_t * n_t1) + i * n_t1 + j
        flat = q.new_zeros((batch * n_t * n_t1,))
        flat_hidden = q.new_zeros((batch * n_t * n_t1, pair_hidden.shape[-1])) if return_hidden else None
        if linear.numel():
            flat = flat.scatter(0, linear, edge_scores)
            if flat_hidden is not None:
                flat_hidden = flat_hidden.scatter(
                    0, linear.unsqueeze(-1).expand(-1, pair_hidden.shape[-1]), pair_hidden
                )
        edge_logits = flat.reshape(batch, n_t, n_t1)
        if flat_hidden is None:
            return edge_logits
        return (edge_logits, flat_hidden.reshape(batch, n_t, n_t1, -1))

    def forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        volumes_t: torch.Tensor,
        volumes_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
        vel_t: torch.Tensor | None = None,
        flow_t: torch.Tensor | None = None,
        return_birth: bool = False,
        return_matchability: bool = False,
        force_edges: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        return_fork: bool = False,
        fork_pairs: torch.Tensor | None = None,
        fork_pair_mask: torch.Tensor | None = None,
        soon_t: torch.Tensor | None = None,
        soon_t1: torch.Tensor | None = None,
        intensity_t: torch.Tensor | None = None,
        intensity_t1: torch.Tensor | None = None,
        intensity_raw_sum_t: torch.Tensor | None = None,
        intensity_raw_sum_t1: torch.Tensor | None = None,
        occ_next: torch.Tensor | None = None,
        has_next: torch.Tensor | None = None,
        occ_prev: torch.Tensor | None = None,
        has_prev: torch.Tensor | None = None,
        peak_next: torch.Tensor | None = None,
        next_maps: torch.Tensor | None = None,
        occ_fork_next: torch.Tensor | None = None,
        occ_fork_has_next: torch.Tensor | None = None,
        occ_fork_prev: torch.Tensor | None = None,
        occ_fork_has_prev: torch.Tensor | None = None,
        soon_hidden_t: torch.Tensor | None = None,
        soon_hidden_t1: torch.Tensor | None = None,
        soon_tm2: torch.Tensor | None = None,
        soon_tm1: torch.Tensor | None = None,
        soon_tp2: torch.Tensor | None = None,
        soon_has_tm2: torch.Tensor | None = None,
        soon_has_tm1: torch.Tensor | None = None,
        soon_has_tp2: torch.Tensor | None = None,
        fork_children: torch.Tensor | None = None,
        sister_key: torch.Tensor | None = None,
        sister_logit: torch.Tensor | None = None,
        edge_context: dict | None = None,
        _fork_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,
        _return_fork_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Predict edge logits between detections at consecutive frames."""
        if return_matchability and (not False):
            raise ValueError("return_matchability requires matchability=True")
        unbatched = feat_t.ndim == 2
        if unbatched:
            feat_t = feat_t.unsqueeze(0)
            feat_t1 = feat_t1.unsqueeze(0)
            coords_t = coords_t.unsqueeze(0)
            coords_t1 = coords_t1.unsqueeze(0)
            volumes_t = volumes_t.unsqueeze(0)
            volumes_t1 = volumes_t1.unsqueeze(0)
            if vel_t is not None:
                vel_t = vel_t.unsqueeze(0)
            if flow_t is not None:
                flow_t = flow_t.unsqueeze(0)
            if force_edges is not None and force_edges.ndim == 2:
                force_edges = force_edges.unsqueeze(0)
            if pair_mask is not None:
                pair_mask = torch.as_tensor(pair_mask).unsqueeze(0)
            if soon_t is not None and soon_t.ndim == 2:
                soon_t = soon_t.unsqueeze(0)
            if soon_t1 is not None and soon_t1.ndim == 2:
                soon_t1 = soon_t1.unsqueeze(0)
            if intensity_t is not None and intensity_t.ndim == 2:
                intensity_t = intensity_t.unsqueeze(0)
            if intensity_t1 is not None and intensity_t1.ndim == 2:
                intensity_t1 = intensity_t1.unsqueeze(0)
            if intensity_raw_sum_t is not None and intensity_raw_sum_t.ndim == 1:
                intensity_raw_sum_t = intensity_raw_sum_t.unsqueeze(0)
            if intensity_raw_sum_t1 is not None and intensity_raw_sum_t1.ndim == 1:
                intensity_raw_sum_t1 = intensity_raw_sum_t1.unsqueeze(0)
            if occ_next is not None and occ_next.ndim == 2:
                occ_next = occ_next.unsqueeze(0)
            if has_next is not None and has_next.ndim == 2:
                has_next = has_next.unsqueeze(0)
            if occ_prev is not None and occ_prev.ndim == 2:
                occ_prev = occ_prev.unsqueeze(0)
            if has_prev is not None and has_prev.ndim == 2:
                has_prev = has_prev.unsqueeze(0)
            if peak_next is not None and peak_next.ndim == 2:
                peak_next = peak_next.unsqueeze(0)
            if next_maps is not None and next_maps.ndim == 4:
                next_maps = next_maps.unsqueeze(0)
            if occ_fork_next is not None and occ_fork_next.ndim == 2:
                occ_fork_next = occ_fork_next.unsqueeze(0)
            if occ_fork_has_next is not None and occ_fork_has_next.ndim == 2:
                occ_fork_has_next = occ_fork_has_next.unsqueeze(0)
            if occ_fork_prev is not None and occ_fork_prev.ndim == 2:
                occ_fork_prev = occ_fork_prev.unsqueeze(0)
            if occ_fork_has_prev is not None and occ_fork_has_prev.ndim == 2:
                occ_fork_has_prev = occ_fork_has_prev.unsqueeze(0)
            if soon_hidden_t is not None and soon_hidden_t.ndim == 2:
                soon_hidden_t = soon_hidden_t.unsqueeze(0)
            if soon_hidden_t1 is not None and soon_hidden_t1.ndim == 2:
                soon_hidden_t1 = soon_hidden_t1.unsqueeze(0)
            if soon_tm2 is not None and soon_tm2.ndim == 2:
                soon_tm2 = soon_tm2.unsqueeze(0)
            if soon_tm1 is not None and soon_tm1.ndim == 2:
                soon_tm1 = soon_tm1.unsqueeze(0)
            if soon_tp2 is not None and soon_tp2.ndim == 2:
                soon_tp2 = soon_tp2.unsqueeze(0)
            if soon_has_tm2 is not None and soon_has_tm2.ndim == 1:
                soon_has_tm2 = soon_has_tm2.unsqueeze(0)
            if soon_has_tm1 is not None and soon_has_tm1.ndim == 1:
                soon_has_tm1 = soon_has_tm1.unsqueeze(0)
            if soon_has_tp2 is not None and soon_has_tp2.ndim == 1:
                soon_has_tp2 = soon_has_tp2.unsqueeze(0)
            if fork_children is not None and fork_children.ndim == 2:
                fork_children = fork_children.unsqueeze(0)
            if sister_key is not None and sister_key.ndim == 1:
                sister_key = sister_key.unsqueeze(0)
            if sister_logit is not None and sister_logit.ndim == 1:
                sister_logit = sister_logit.unsqueeze(0)
            edge_context = self._batchify_context(edge_context or {}, True)
        if self.training and 0.0 > 0:
            feat_t = _feat_block_mask(feat_t, self._node_block_slices, 0.0)
            feat_t1 = _feat_block_mask(feat_t1, self._node_block_slices, 0.0)
        if _fork_state is None:
            q, k = self.encode_frames(
                feat_t,
                feat_t1,
                mask_t,
                mask_t1,
                **self._attention_masks(coords_t, coords_t1, mask_t, mask_t1),
                **self._node_extra(coords_t, coords_t1, soon_t, soon_t1, soon_hidden_t, soon_hidden_t1),
            )
            log_v_t = volumes_t.float().clamp(min=1.0).log()
            log_v_t1 = volumes_t1.float().clamp(min=1.0).log()
            N_t = q.shape[1]
            pair_hidden = None
            who_pair_hidden = None
            prev_block = next_block = None
            if pair_mask is not None:
                pair_mask = torch.as_tensor(pair_mask, dtype=torch.bool, device=q.device)
                if pair_mask.shape != (q.shape[0], q.shape[1], k.shape[1]):
                    raise ValueError(
                        f"pair mask shape {tuple(pair_mask.shape)} does not match model output {(q.shape[0], q.shape[1], k.shape[1])}"
                    )
                if mask_t is not None:
                    pair_mask = pair_mask & mask_t.unsqueeze(-1)
                if mask_t1 is not None:
                    pair_mask = pair_mask & mask_t1.unsqueeze(1)
                sparse_scorer = self._sparse_pair_logits
                pair_result = sparse_scorer(
                    q,
                    k,
                    coords_t,
                    coords_t1,
                    log_v_t,
                    log_v_t1,
                    vel_t,
                    flow_t,
                    feat_t,
                    feat_t1,
                    pair_mask,
                    return_hidden=return_fork,
                    soon_t=soon_t,
                    soon_t1=soon_t1,
                    intensity_t=intensity_t,
                    intensity_t1=intensity_t1,
                    occ_next=occ_next,
                    has_next=has_next,
                    occ_prev=occ_prev,
                    has_prev=has_prev,
                    peak_next=peak_next,
                    prev_block=prev_block,
                    next_block=next_block,
                )
                if return_fork:
                    logits, pair_hidden = pair_result
                else:
                    logits = pair_result
            else:
                raise ValueError(
                    "intensity_pair requires a pair_mask (sparse pair scoring); the dense pair path does not build the intensity skip"
                )
                raise ValueError(
                    "occupancy_pair requires a pair_mask (sparse pair scoring); the dense pair path does not build the occupancy skip"
                )
                blocks = []
                if flow_t is None:
                    flow_t = coords_t.new_zeros((*coords_t.shape[:2], 3))
                flow_block = flow_pair_features(coords_t, coords_t1, flow_t)
                blocks.append(flow_block)
                raise ValueError("shape_residual_direction requires a pair_mask (sparse pair scoring)")
                gt, gt1 = self._geom_blocks(feat_t, feat_t1)
                blocks.append(
                    shape_pair_features(
                        gt[..., _AXIS],
                        gt1[..., _AXIS],
                        gt[..., 0],
                        gt1[..., 0],
                        gt[..., 3],
                        gt1[..., 3],
                        coords_t,
                        coords_t1,
                    )
                )
                motion = torch.cat(blocks, dim=-1) if blocks else None
                chunk = self.pair_chunk_size or N_t
                chunks = []
                pair_mlp = self.pair_mlp
                for i in range(0, N_t if chunk is not None else 0, chunk or 1):
                    q_c = q[:, i : i + chunk, :]
                    coords_c = coords_t[:, i : i + chunk, :]
                    log_v_c = log_v_t[:, i : i + chunk]
                    motion_c = motion[:, i : i + chunk] if motion is not None else None

                    def _chunk_fn(
                        qc: torch.Tensor,
                        kk: torch.Tensor,
                        cc: torch.Tensor,
                        cc1: torch.Tensor,
                        lvc: torch.Tensor,
                        lv1: torch.Tensor,
                        mo: torch.Tensor | None = motion_c,
                        _pm: nn.Module = pair_mlp,
                        _uv: bool = True,
                        _fbd: float = 0.0,
                        _pbs: list = self._pair_block_slices,
                        _trn: bool = self.training,
                    ) -> torch.Tensor:
                        nc_i = qc.shape[1]
                        n1 = kk.shape[1]
                        qe = qc.unsqueeze(2).expand(-1, -1, n1, -1)
                        ke = kk.unsqueeze(1).expand(-1, nc_i, -1, -1)
                        rel = (cc.unsqueeze(2) - cc1.unsqueeze(1)) / 100.0
                        parts = [qe, ke, rel]
                        if _uv:
                            parts.append((lvc.unsqueeze(2) - lv1.unsqueeze(1)).unsqueeze(-1))
                        if mo is not None:
                            parts.append(mo)
                        inp = torch.cat(parts, dim=-1)
                        if _trn and _fbd > 0 and _pbs:
                            inp = _feat_block_mask(inp, _pbs, _fbd)
                        return _pm(inp).squeeze(-1)

                    if torch.is_grad_enabled():
                        out = grad_ckpt(_chunk_fn, q_c, k, coords_c, coords_t1, log_v_c, log_v_t1, use_reentrant=False)
                    else:
                        out = _chunk_fn(q_c, k, coords_c, coords_t1, log_v_c, log_v_t1)
                    chunks.append(out)
                logits = torch.cat(chunks, dim=1)
        else:
            q, k, logits, pair_hidden, fork_support, *rest = _fork_state
            who_pair_hidden = None
            next_block = rest[0] if rest else None
            log_v_t = volumes_t.float().clamp(min=1.0).log()
            log_v_t1 = volumes_t1.float().clamp(min=1.0).log()
        birth = None
        if return_birth:
            raise ValueError("return_birth requires no_parent=True")
        match_t = match_t1 = None
        if return_matchability:
            match_t = self.match_head(q).squeeze(-1)
            match_t1 = self.match_head(k).squeeze(-1)
        fork_output = None
        if return_fork:
            if _fork_state is None:
                fork_support = pair_mask
                if fork_support is None and (mask_t is not None or mask_t1 is not None):
                    fork_support = torch.ones_like(logits, dtype=torch.bool)
                    if mask_t is not None:
                        fork_support = fork_support & mask_t.unsqueeze(-1)
                    if mask_t1 is not None:
                        fork_support = fork_support & mask_t1.unsqueeze(1)
            if soon_t is None or soon_t1 is None:
                raise ValueError("soon_fork=True requires soon_t and soon_t1 at forward")
            fork_output = self._fork_config_logits(
                q,
                k,
                coords_t,
                coords_t1,
                volumes_t,
                volumes_t1,
                flow_t,
                logits,
                pair_hidden,
                feat_t,
                feat_t1,
                vel_t,
                log_v_t,
                log_v_t1,
                fork_support,
                who_pair_hidden=who_pair_hidden,
                soon_t=soon_t,
                soon_t1=soon_t1,
                fork_children=fork_children,
                soon_tm2=soon_tm2,
                soon_tm1=soon_tm1,
                soon_tp2=soon_tp2,
                soon_has_tm2=soon_has_tm2,
                soon_has_tm1=soon_has_tm1,
                soon_has_tp2=soon_has_tp2,
                sister_key=sister_key,
                sister_logit=sister_logit,
                intensity_t=intensity_t,
                intensity_t1=intensity_t1,
                intensity_raw_sum_t=intensity_raw_sum_t,
                intensity_raw_sum_t1=intensity_raw_sum_t1,
                occ_next=occ_next,
                has_next=has_next,
                occ_prev=occ_prev,
                has_prev=has_prev,
                next_maps=next_maps,
                occ_fork_next=occ_fork_next,
                occ_fork_has_next=occ_fork_has_next,
                occ_fork_prev=occ_fork_prev,
                occ_fork_has_prev=occ_fork_has_prev,
                next_block=next_block,
            )
            if _return_fork_state:
                state = (q, k, logits, pair_hidden, fork_support)
                fork_output["_fork_state"] = state
        if unbatched:
            logits = logits.squeeze(0)
            if birth is not None:
                birth = birth.squeeze(0)
            if match_t is not None:
                match_t, match_t1 = (match_t.squeeze(0), match_t1.squeeze(0))
            if fork_output is not None:
                fork_output = {
                    name: value.squeeze(0) if name != "_fork_state" else value for name, value in fork_output.items()
                }
        if return_fork:
            output = {"logits": logits, "birth": birth, "match_t": match_t, "match_t1": match_t1}
            output.update(fork_output or {})
            return output
        if return_matchability:
            return (logits, match_t, match_t1)
        return (logits, birth) if return_birth else logits
