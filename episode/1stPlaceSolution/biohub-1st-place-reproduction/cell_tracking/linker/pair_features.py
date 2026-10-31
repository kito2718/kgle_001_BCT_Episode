"""Geometric pair and fork features and the linker feature widths."""

from __future__ import annotations
import torch
from cell_tracking.common.config import NATIVE_SHAPE
from cell_tracking.common.config import SCALE as NATIVE_VOXEL_UM
from cell_tracking.linker.device_const import device_const




PAIR_UM_SCALE = 10.0


def _feat_block_mask(feat: torch.Tensor, block_slices: list[slice], p: float) -> torch.Tensor:
    """Zero whole feature blocks with probability *p* per block per sample."""
    if p <= 0 or not block_slices:
        return feat
    mask = torch.ones(feat.shape[0], feat.shape[-1], device=feat.device, dtype=feat.dtype)
    for s in block_slices:
        drop = torch.rand(feat.shape[0], 1, device=feat.device) < p
        w = s.stop - s.start
        mask[:, s] = mask[:, s].masked_fill(drop.expand(-1, w), 0.0)
    shape = [feat.shape[0]] + [1] * (feat.ndim - 2) + [feat.shape[-1]]
    return feat * mask.view(*shape)


def flow_pair_features(
    coords_t: torch.Tensor, coords_t1: torch.Tensor, flow_t: torch.Tensor, *, voxel_size=NATIVE_VOXEL_UM
) -> torch.Tensor:
    """``(B, N_t, N_t1, FLOW_PAIR_DIM)`` displacement AFTER removing the tissue drift."""
    vs = device_const(voxel_size, coords_t.dtype, coords_t.device)
    ct = coords_t * vs
    ct1 = coords_t1 * vs
    pred = ct + flow_t
    res = ct1.unsqueeze(1) - pred.unsqueeze(2)
    res_um = res.norm(dim=-1)
    flow_um = flow_t.norm(dim=-1).unsqueeze(-1).expand(-1, -1, ct1.shape[1])
    return torch.cat(
        [res / 100.0, (res_um / PAIR_UM_SCALE).unsqueeze(-1), (flow_um / PAIR_UM_SCALE).unsqueeze(-1)], dim=-1
    )


SHAPE_PAIR_DIM = 5


_AXIS = slice(7, 10)


FOV_FEAT_DIM = 3


def fov_distance_features(coords: torch.Tensor, voxel_size=NATIVE_VOXEL_UM, shape=NATIVE_SHAPE) -> torch.Tensor:
    """Per-axis distance to the nearest volume face, ``(..., 3)`` in ``[0, 1]``."""
    vs = device_const(voxel_size, coords.dtype, coords.device)
    ext = device_const(shape, coords.dtype, coords.device) - 1.0
    d_um = torch.minimum(coords, ext - coords).clamp(min=0.0) * vs
    return (d_um / (0.5 * (ext * vs).min())).clamp(max=1.0)


def shape_pair_features(
    axis_t: torch.Tensor,
    axis_t1: torch.Tensor,
    logax_t: torch.Tensor,
    logax_t1: torch.Tensor,
    elong_t: torch.Tensor,
    elong_t1: torch.Tensor,
    coords_t: torch.Tensor,
    coords_t1: torch.Tensor,
    *,
    voxel_size=NATIVE_VOXEL_UM,
) -> torch.Tensor:
    """``(B, N_t, N_t1, SHAPE_PAIR_DIM)`` — HOCT's line-to-line bias, plus shape agreement."""
    dev = coords_t.device
    vs = device_const(voxel_size, coords_t.dtype, dev)
    delta = (coords_t1 * vs).unsqueeze(1) - (coords_t * vs).unsqueeze(2)
    dnorm = delta / delta.norm(dim=-1, keepdim=True).clamp(min=1e-06)
    at = axis_t / axis_t.norm(dim=-1, keepdim=True).clamp(min=1e-06)
    at1 = axis_t1 / axis_t1.norm(dim=-1, keepdim=True).clamp(min=1e-06)
    cos_axes = (at.unsqueeze(2) * at1.unsqueeze(1)).sum(-1).abs()
    cos_t_move = (at.unsqueeze(2) * dnorm).sum(-1).abs()
    cos_t1_move = (at1.unsqueeze(1) * dnorm).sum(-1).abs()
    d_logax = (logax_t.unsqueeze(2) - logax_t1.unsqueeze(1)).abs()
    d_elong = (elong_t.unsqueeze(2) - elong_t1.unsqueeze(1)).abs()
    return torch.stack([cos_axes, cos_t_move, cos_t1_move, d_logax, d_elong], dim=-1)


FORK_GEOM_DIM = 29


FORK_EDGE_DIM = 4


CONFIG_KIND_DIM = 3


CONFIG_EMPTY = 0


CONFIG_SINGLETON = 1


CONFIG_FORK = 2


SOON_N_CLASSES = 3


SOON_INT, SOON_MOTHER, SOON_DAUGHTER = (0, 1, 2)


SOON_FORK_DIM = 3 * SOON_N_CLASSES


SOON_PAIR_DIM = 5


SOON_FORK_STATS_DIM = 4


KIND_DAUGHTER_SOON_DIM = 3


SOON_TEMPORAL_MOTHER_DIM = 9


SOON_TEMPORAL_DAUGHTER_DIM = 6


SOON_TEMPORAL_DIM = SOON_TEMPORAL_MOTHER_DIM + 2 * SOON_TEMPORAL_DAUGHTER_DIM


SISTER_DIM = 2


def _unit(vec: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp(min=eps)


def fork_pair_features(
    coords_t: torch.Tensor,
    coords_t1: torch.Tensor,
    volumes_t: torch.Tensor,
    volumes_t1: torch.Tensor,
    fork_pairs: torch.Tensor,
    flow_t: torch.Tensor | None = None,
    feat_t: torch.Tensor | None = None,
    feat_t1: torch.Tensor | None = None,
    *,
    voxel_size=NATIVE_VOXEL_UM,
    use_geom: bool | None = None,
    vol_conserv: bool = True,
) -> torch.Tensor:
    """Permutation-invariant joint geometry for packed ``(i, j1, j2)`` forks."""
    if fork_pairs.ndim != 3 or fork_pairs.shape[-1] != 3:
        raise ValueError(f"fork_pairs must have shape (B, P, 3), got {fork_pairs.shape}")
    batch = fork_pairs.shape[0]
    batch_idx = torch.arange(batch, device=fork_pairs.device).view(batch, 1)
    batch_idx = batch_idx.expand(-1, fork_pairs.shape[1])
    src_i = fork_pairs[..., 0].long()
    child_a = fork_pairs[..., 1].long()
    child_b = fork_pairs[..., 2].long()
    vs = device_const(voxel_size, coords_t.dtype, coords_t.device)
    parent_um = coords_t[batch_idx, src_i] * vs
    child_a_um = coords_t1[batch_idx, child_a] * vs
    child_b_um = coords_t1[batch_idx, child_b] * vs
    rel_a = child_a_um - parent_um
    rel_b = child_b_um - parent_um
    sister_rel = child_b_um - child_a_um
    rel_sum = (rel_a + rel_b) / 2.0
    rel_abs = (rel_a - rel_b).abs() / 2.0
    dist_a = rel_a.norm(dim=-1, keepdim=True)
    dist_b = rel_b.norm(dim=-1, keepdim=True)
    dist_mean = (dist_a + dist_b) / (2.0 * PAIR_UM_SCALE)
    dist_gap = (dist_a - dist_b).abs() / PAIR_UM_SCALE
    sister_dist = sister_rel.norm(dim=-1, keepdim=True) / PAIR_UM_SCALE
    log_parent = volumes_t[batch_idx, src_i].float().clamp(min=1.0).log()
    vol_a = volumes_t1[batch_idx, child_a].float().clamp(min=1.0)
    vol_b = volumes_t1[batch_idx, child_b].float().clamp(min=1.0)
    log_a = vol_a.log()
    log_b = vol_b.log()
    log_vol_conserv = ((vol_a + vol_b).log() - log_parent).unsqueeze(-1)
    if not vol_conserv:
        log_vol_conserv = torch.zeros_like(log_vol_conserv)
    log_sister = (log_a - log_b).abs().unsqueeze(-1)
    if flow_t is None:
        flow = torch.zeros_like(coords_t)
    else:
        flow = flow_t
    flow_vec = flow[batch_idx, src_i]
    predicted = parent_um + flow_vec
    residual_a = child_a_um - predicted
    residual_b = child_b_um - predicted
    residual_sum = (residual_a + residual_b) / 2.0
    residual_abs = (residual_a - residual_b).abs() / 2.0
    n1 = residual_a.norm(dim=-1, keepdim=True)
    n2 = residual_b.norm(dim=-1, keepdim=True)
    residual_gap = (n1 - n2).abs() / PAIR_UM_SCALE
    residual_norm_mean = (n1 + n2) / (2.0 * PAIR_UM_SCALE)
    flow_norm = flow_vec.norm(dim=-1, keepdim=True)
    flow_scale = flow_norm / PAIR_UM_SCALE
    eps = 1e-06
    cos_sister = ((residual_a * residual_b).sum(-1, keepdim=True) / (n1.clamp(min=eps) * n2.clamp(min=eps))).clamp(
        -1.0, 1.0
    )
    flow_n = flow_norm.clamp(min=eps)
    sum_res = residual_a + residual_b
    sum_n = sum_res.norm(dim=-1, keepdim=True).clamp(min=eps)
    cos_sum_flow = ((sum_res * flow_vec).sum(-1, keepdim=True) / (sum_n * flow_n)).clamp(-1.0, 1.0)
    has_flow = flow_norm > 0.001
    cos_sum_flow = torch.where(has_flow, cos_sum_flow, torch.zeros_like(cos_sum_flow))
    diff_res = residual_a - residual_b
    diff_n = diff_res.norm(dim=-1, keepdim=True).clamp(min=eps)
    abs_cos_diff_flow = ((diff_res * flow_vec).sum(-1, keepdim=True).abs() / (diff_n * flow_n)).clamp(0.0, 1.0)
    abs_cos_diff_flow = torch.where(has_flow, abs_cos_diff_flow, torch.zeros_like(abs_cos_diff_flow))
    shape = residual_a.new_zeros((*residual_a.shape[:-1], 3))
    from cell_tracking.linker.features import GEOM_FEAT_DIM

    if use_geom is None:
        use_geom = feat_t is not None and feat_t1 is not None
    if (
        use_geom
        and feat_t is not None
        and (feat_t1 is not None)
        and (feat_t.shape[-1] >= GEOM_FEAT_DIM + 1)
        and (feat_t1.shape[-1] >= GEOM_FEAT_DIM + 1)
    ):
        gt = feat_t[..., -GEOM_FEAT_DIM:]
        gt1 = feat_t1[..., -GEOM_FEAT_DIM:]
        axis_p = _unit(gt[batch_idx, src_i, 7:10], eps)
        axis_a = _unit(gt1[batch_idx, child_a, 7:10], eps)
        axis_b = _unit(gt1[batch_idx, child_b, 7:10], eps)
        abs_cos_div_axis = (axis_p * (diff_res / diff_n)).sum(-1, keepdim=True).abs()
        abs_cos_sister_axes = (axis_a * axis_b).sum(-1, keepdim=True).abs()
        cos_a_res = (axis_a * (residual_a / n1.clamp(min=eps))).sum(-1, keepdim=True).abs()
        cos_b_res = (axis_b * (residual_b / n2.clamp(min=eps))).sum(-1, keepdim=True).abs()
        shape = torch.cat([abs_cos_div_axis, abs_cos_sister_axes, (cos_a_res + cos_b_res) / 2.0], dim=-1)
    geometric = torch.cat(
        [
            rel_sum / 100.0,
            rel_abs / 100.0,
            sister_rel.abs() / 100.0,
            dist_mean,
            dist_gap,
            sister_dist,
            log_vol_conserv,
            log_sister,
            residual_sum / 100.0,
            residual_abs / 100.0,
            residual_gap,
            flow_scale,
            cos_sister,
            residual_norm_mean,
            cos_sum_flow,
            abs_cos_diff_flow,
            shape,
        ],
        dim=-1,
    )
    if geometric.shape[-1] != FORK_GEOM_DIM:
        raise RuntimeError(f"fork geometry width drifted: {geometric.shape[-1]} != {FORK_GEOM_DIM}")
    return geometric
