"""Flow-follow decode: predicted flow field -> instance label volume."""

import numpy as np
from cell_tracking.common.io import pooled_voxel_um


def _cluster_um(pts_um, eps_um):
    """Single-linkage cluster of points (µm): two points are linked iff <= eps_um apart, and a cluster is a connected chain of such links."""
    n = len(pts_um)
    if n == 0:
        return np.zeros(0, np.int32)
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    pairs = cKDTree(pts_um).query_pairs(eps_um, output_type="ndarray")
    if len(pairs):
        data = np.ones(len(pairs), dtype=bool)
        graph = coo_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    else:
        graph = coo_matrix((n, n), dtype=bool)
    _, roots = connected_components(graph, directed=False)
    _, lab = np.unique(roots, return_inverse=True)
    return (lab + 1).astype(np.int32)


def track_landing_points_um(mask, track_vec, split_um=3.0, spacing=None):
    """Horizon-1 landing point (µm) of every voxel in `mask`, plus its `_cluster_um` mode label."""
    from cell_tracking.common.config import TRACK_SCALE

    spacing = np.asarray(pooled_voxel_um() if spacing is None else spacing, np.float64)
    mask = np.asarray(mask, bool)
    if not mask.any():
        return (np.zeros((0, 3)), np.zeros(0, np.int32))
    land = np.argwhere(mask).astype(np.float64) * spacing + np.asarray(track_vec, np.float64)[:, mask].T * TRACK_SCALE
    lab = _cluster_um(land, split_um)
    return (land, lab)


def gated_flow_magnitude(pred, ch, fg_thresh=0.5):
    """|flow| magnitude, SUPPRESSED (zeroed) outside the same foreground gate `flow_follow_decode` uses to seed trajectories -- i.e."""
    flow = pred[ch.flow]
    prob_logit = pred[ch.prob]
    mag = np.linalg.norm(flow, axis=0)
    fg = 1.0 / (1.0 + np.exp(-prob_logit)) > fg_thresh
    return np.where(fg, mag, 0.0)


def _follow_euler_numpy(flow, start, *, niter, step_um, spacing, flow_scale, unit_flow, quantile_exit=False):
    """SciPy map_coordinates Euler follow."""
    from scipy import ndimage as ndi

    Z, Y, X = flow.shape[1:]
    p = start.astype(np.float64)
    hi = np.array([Z - 1, Y - 1, X - 1], np.float64)
    vox_per_step = np.asarray(step_um, np.float64) / np.asarray(spacing, np.float64)
    flow = np.ascontiguousarray(flow.astype(np.float64)) / max(float(flow_scale), 1e-06)
    for _ in range(int(niter)):
        coords = p.T
        u = np.stack([ndi.map_coordinates(flow[a], coords, order=1, mode="nearest") for a in range(3)], 1)
        n = np.linalg.norm(u, axis=1, keepdims=True)
        u = u / np.maximum(n, 1e-06) if unit_flow else u / np.maximum(n, 1.0)
        newp = np.clip(p + u * vox_per_step, 0.0, hi)
        d = np.abs(newp - p)
        stop = (d < 0.01).mean() >= 0.9999 if quantile_exit else d.max() < 0.01
        if stop:
            p = newp
            break
        p = newp
    return p


def _resolve_decode_device(device):
    import torch

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _follow_euler_torch(flow, start, *, niter, step_um, spacing, flow_scale, unit_flow, device, quantile_exit=False):
    """CUDA grid_sample Euler follow."""
    import torch
    import torch.nn.functional as F

    Z, Y, X = flow.shape[1:]
    flow_np = np.ascontiguousarray(flow.astype(np.float32)) / max(float(flow_scale), 1e-06)
    with torch.no_grad():
        im = torch.from_numpy(flow_np).to(device=device, dtype=torch.float32)[None]
        p = torch.from_numpy(start.astype(np.float32)).to(device=device)
        hi = torch.tensor([Z - 1, Y - 1, X - 1], device=device, dtype=torch.float32)
        shape_m1 = hi
        denom = torch.clamp(shape_m1, min=1.0)
        vox_per_step = torch.as_tensor(step_um, device=device, dtype=torch.float32) / torch.as_tensor(
            np.asarray(spacing, np.float64), device=device, dtype=torch.float32
        )
        for _ in range(int(niter)):
            z = p[:, 0] / denom[0] * 2.0 - 1.0
            y = p[:, 1] / denom[1] * 2.0 - 1.0
            x = p[:, 2] / denom[2] * 2.0 - 1.0
            grid = torch.stack((x, y, z), dim=-1).view(1, -1, 1, 1, 3)
            u = F.grid_sample(im, grid, mode="bilinear", padding_mode="border", align_corners=True)[
                0, :, :, 0, 0
            ].transpose(0, 1)
            n = torch.linalg.norm(u, dim=1, keepdim=True)
            if unit_flow:
                u = u / torch.clamp(n, min=1e-06)
            else:
                u = u / torch.clamp(n, min=1.0)
            newp = torch.clamp(p + u * vox_per_step, torch.zeros_like(hi), hi)
            d = (newp - p).abs()
            if quantile_exit:
                stop = (d < 0.01).to(torch.float32).mean() >= 0.9999
            else:
                stop = d.max() < 0.01
            if stop:
                p = newp
                break
            p = newp
        return p.detach().cpu().numpy().astype(np.float64)


def flow_follow_decode(
    flow,
    prob_logit,
    fg_thresh=0.5,
    step_um=None,
    sink_merge_um=3.0,
    min_size=100,
    niter=200,
    spacing=None,
    flow_scale=1.0,
    unit_flow=False,
    device=None,
    contact_logit=None,
    contact_cut=0.5,
    quantile_exit=False,
):
    """Follow predicted PHYSICAL-direction flows to their sinks and group convergent voxels into instances."""
    Z, Y, X = prob_logit.shape
    spacing = np.asarray(pooled_voxel_um() if spacing is None else spacing, np.float64)
    step_um = float(spacing.min()) if step_um is None else float(step_um)
    fg_full = prob_logit > np.log(fg_thresh / (1.0 - fg_thresh))
    if contact_logit is not None:
        fg = fg_full & (contact_logit < np.log(contact_cut / (1.0 - contact_cut)))
    else:
        fg = fg_full
    out = np.zeros((Z, Y, X), np.int32)
    if not fg.any():
        return out
    start = np.argwhere(fg)
    dev = _resolve_decode_device(device)
    if dev.type == "cuda":
        p = _follow_euler_torch(
            flow,
            start,
            niter=niter,
            step_um=step_um,
            spacing=spacing,
            flow_scale=flow_scale,
            unit_flow=unit_flow,
            device=dev,
            quantile_exit=quantile_exit,
        )
    else:
        p = _follow_euler_numpy(
            flow,
            start,
            niter=niter,
            step_um=step_um,
            spacing=spacing,
            flow_scale=flow_scale,
            unit_flow=unit_flow,
            quantile_exit=quantile_exit,
        )
    land = np.round(p).astype(np.intp)
    flat = land[:, 0].astype(np.int64) * (Y * X) + land[:, 1] * X + land[:, 2]
    uflat, inv = np.unique(flat, return_inverse=True)
    inv = inv.reshape(-1)
    uland = np.stack([uflat // (Y * X), uflat // X % Y, uflat % X], 1)
    cluster = _cluster_um(uland.astype(np.float64) * spacing, sink_merge_um)
    lab = cluster[inv]
    if contact_logit is not None:
        from scipy import ndimage as ndi

        comp, _ = ndi.label(fg)
        cid = comp[start[:, 0], start[:, 1], start[:, 2]]
        lab = np.unique(np.stack([lab, cid], 1), axis=0, return_inverse=True)[1].astype(np.intp) + 1
    counts = np.bincount(lab)
    keep = np.flatnonzero(counts >= min_size)
    keep = keep[keep != 0]
    if not keep.size:
        return out
    relabel = np.zeros(counts.size, np.int32)
    relabel[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    out[start[:, 0], start[:, 1], start[:, 2]] = relabel[lab]
    if contact_logit is not None:
        gap = fg_full & (out == 0)
        if gap.any() and out.any():
            from scipy import ndimage as ndi

            _, (iz, iy, ix) = ndi.distance_transform_edt(out == 0, sampling=spacing, return_indices=True)
            out[gap] = out[iz[gap], iy[gap], ix[gap]]
    return out


def track_displacement_um(instances, track_vec, spacing=None, split_um=None):
    """Per-instance predicted DISPLACEMENT (µm), pooled from the horizon-1 track field."""
    from cell_tracking.common.config import TRACK_SCALE

    spacing = np.asarray(pooled_voxel_um() if spacing is None else spacing, np.float64)
    instances = np.asarray(instances)
    track_vec = np.asarray(track_vec, np.float64)
    ids = np.unique(instances)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return (ids, np.zeros((0, 3), np.float64))
    vox = np.argwhere(instances > 0)
    labs = instances[vox[:, 0], vox[:, 1], vox[:, 2]]
    order = np.argsort(labs, kind="stable")
    vox = vox[order]
    labs = labs[order]
    d_all = track_vec[:, vox[:, 0], vox[:, 1], vox[:, 2]].T * TRACK_SCALE
    lo = np.searchsorted(labs, ids, side="left")
    hi = np.searchsorted(labs, ids, side="right")
    if split_um is None:
        return (ids, np.stack([np.median(d_all[lo[j] : hi[j]], axis=0) for j in range(len(ids))]))
    pos_all = vox.astype(np.float64) * spacing
    disp = np.zeros((len(ids), 3), np.float64)
    for j in range(len(ids)):
        d = d_all[lo[j] : hi[j]]
        lab = _cluster_um(pos_all[lo[j] : hi[j]] + d, split_um)
        sizes = np.bincount(lab)[1:]
        keep = int(np.argmax(sizes)) + 1
        disp[j] = np.median(d[lab == keep], axis=0)
    return (ids, disp)


def decode_links_and_divisions(
    instances, track_vec, ttd=None, spacing=None, split_um=3.0, min_frac=0.25, ttd_max=0.5, mode_min_vox=10
):
    """Turn the horizon-1 track field into link and division candidates, per decoded instance."""
    from cell_tracking.common.config import TRACK_SCALE

    spacing = np.asarray(pooled_voxel_um() if spacing is None else spacing, np.float64)
    instances = np.asarray(instances)
    track_vec = np.asarray(track_vec, np.float64)
    ids = np.unique(instances)
    ids = ids[ids > 0]
    links, divisions, n_multimodal, n_ttd_vetoed = ([], [], 0, 0)
    for i in ids:
        m = instances == i
        n = int(m.sum())
        land = np.argwhere(m).astype(np.float64) * spacing + track_vec[:, m].T * TRACK_SCALE
        lab = _cluster_um(land, split_um)
        sizes = np.bincount(lab)[1:]
        order = list(np.argsort(sizes)[::-1] + 1)
        big = [c for c in order if sizes[c - 1] >= min_frac * n]
        candidates = [c for c in order if sizes[c - 1] >= mode_min_vox]
        mean_ttd = float(np.asarray(ttd)[m].mean()) if ttd is not None else None
        ttd_ok = mean_ttd is None or mean_ttd <= ttd_max
        if len(big) == 2 and ttd_ok:
            divisions.append(
                {
                    "label": int(i),
                    "n_vox": n,
                    "ttd": mean_ttd,
                    "children_um": [np.median(land[lab == c], axis=0) for c in big],
                }
            )
            continue
        if len(big) == 2 and (not ttd_ok):
            n_ttd_vetoed += 1
        if len(candidates) >= 3:
            n_multimodal += 1
        keep = order[0]
        links.append({"label": int(i), "n_vox": n, "ttd": mean_ttd, "centre_um": np.median(land[lab == keep], axis=0)})
    return (links, divisions, n_multimodal, n_ttd_vetoed)
