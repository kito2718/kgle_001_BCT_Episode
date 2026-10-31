"""Build original and fixed rotated-heat candidate banks with shared forward passes."""
import numpy as np
import pandas as pd
import torch
from skimage.feature import peak_local_max

from biohub.detector.infer import (
    BASE_COLUMNS, SCORE_FLOOR, TOP_K_PER_FRAME, VOXEL_UM, SCALE_ARR,
    _stacks, _neighbour_features, _axis_sigmas, _boundary_um,
    grid_to_src, emb_columns, _empty_frame,
)


def _append_rows(heat, flow, logvar, offset, feats, ts, grid_shape, n_emb, rows):
    """The existing default peak extraction, with no alternate hypotheses."""
    for i, t in enumerate(ts):
        peaks = peak_local_max(heat[i], min_distance=1, threshold_abs=SCORE_FLOOR,
                               exclude_border=False)
        if not len(peaks):
            continue
        s = heat[i][tuple(peaks.T)]
        keep = np.argsort(-s)[:TOP_K_PER_FRAME]
        peaks, s = peaks[keep], s[keep]

        idx = tuple(peaks.T)
        f = flow[i][:, idx[0], idx[1], idx[2]].T          # (N, 3) um
        sigma = np.sqrt(np.exp(np.clip(logvar[i][idx], -8, 8)))

        # Offset head refines the peak inside its voxel; both are in um, so
        # the conversion to source voxels happens once, here.
        off = offset[i][:, idx[0], idx[1], idx[2]].T
        src = grid_to_src(peaks) + off / SCALE_ARR

        um = src * SCALE_ARR
        near, dens = _neighbour_features(um, s)
        axis_sig = _axis_sigmas(heat[i], peaks, s)
        width = axis_sig.mean(axis=1) * VOXEL_UM
        bound = _boundary_um(src, grid_shape)

        # The embedding is read at the peak's grid voxel — the offset moves
        # the reported centre, not the sampling site. Clipped into float16
        # range so a stray activation degrades to a saturated value, not inf.
        pk = torch.from_numpy(np.ascontiguousarray(peaks)).long().to(feats.device)
        emb = feats[i][:, pk[:, 0], pk[:, 1], pk[:, 2]].T.cpu().numpy()
        emb = np.clip(emb, -65504.0, 65504.0).astype(np.float16)

        data = {
            "t": np.full(len(peaks), t, dtype=np.int32),
            "z": src[:, 0].astype(np.float32),
            "y": src[:, 1].astype(np.float32),
            "x": src[:, 2].astype(np.float32),
            "score": s.astype(np.float32),
            "rank_in_frame": np.arange(len(peaks), dtype=np.int32),
            "flow_z": f[:, 0].astype(np.float32),
            "flow_y": f[:, 1].astype(np.float32),
            "flow_x": f[:, 2].astype(np.float32),
            "flow_sigma_um": sigma.astype(np.float32),
            "near_um": np.minimum(near, 1e4).astype(np.float32),
            "density_10um": dens,
            "peak_width_um": width.astype(np.float32),
            "boundary_um": bound.astype(np.float32),
        }
        data.update({c: emb[:, j] for j, c in enumerate(emb_columns(n_emb))})
        rows.append(pd.DataFrame(data))




def detect_original_and_rotation(models, vols, device, amp_dtype=None, batch=8, return_flow=False):
    """Return exact original and V183 banks with two forwards per model/batch.

    V183's probability-to-logit-to-probability round trip is retained so the
    optimized bank is bit-identical to HeatRotation180 plus the normal detector.
    Directed heads and embeddings are shared and averaged in original model order.
    No full-movie dense feature tensor is retained.
    """
    if not models or batch < 1:
        raise ValueError("nonempty models and positive batch required")
    context = int(getattr(models[0], "in_frames", 3))
    n_emb = models[0].feature.out_channels
    if any(int(getattr(m, "in_frames", 3)) != context or m.feature.out_channels != n_emb for m in models):
        raise ValueError("all models must share context and feature dimensions")
    rows = [[], []]
    flow_batches = []
    for start in range(0, len(vols), batch):
        ts = list(range(start, min(start + batch, len(vols))))
        x = torch.from_numpy(_stacks(vols, ts, context)).to(device)
        heat = rotated_heat = flow = logvar = offset = feats = None
        with torch.no_grad():
            for model in models:
                with torch.autocast("cuda", dtype=amp_dtype,
                                    enabled=amp_dtype is not None and device.type == "cuda"):
                    original = model(x)
                    rotated = model(torch.flip(x, dims=(-2, -1)))
                h = torch.sigmoid(original["heatmap"][:, 0].float()).cpu().numpy()
                probability = 0.5 * (torch.sigmoid(original["heatmap"].float())
                    + torch.flip(torch.sigmoid(rotated["heatmap"].float()), dims=(-2, -1)))
                averaged_logit = torch.logit(probability.clamp(1e-6, 1.-1e-6))
                rh = torch.sigmoid(averaged_logit[:, 0].float()).cpu().numpy()
                heat = h if heat is None else heat + h
                rotated_heat = rh if rotated_heat is None else rotated_heat + rh
                f = original["flow"].float().cpu().numpy()
                lv = original["flow_logvar"][:, 0].float().cpu().numpy()
                off = original["offset"].float().cpu().numpy()
                fe = original["features"].float()
                if flow is None:
                    flow, logvar, offset, feats = f, lv, off, fe
                else:
                    flow += f
                    logvar += lv
                    offset += off
                    feats += fe
                del original, rotated, probability, averaged_logit
        k = float(len(models))
        heat, rotated_heat = heat / k, rotated_heat / k
        flow, logvar, offset, feats = flow / k, logvar / k, offset / k, feats / k
        if return_flow:
            flow_batches.append(flow.copy())
        for target, values in zip(rows, (heat, rotated_heat)):
            _append_rows(values, flow, logvar, offset, feats, ts, vols.shape[1:], n_emb, target)
    columns = list(BASE_COLUMNS) + emb_columns(n_emb)
    banks = tuple(pd.concat(part, ignore_index=True)[columns] if part else _empty_frame(columns) for part in rows)
    if return_flow:
        return (*banks, np.concatenate(flow_batches) if flow_batches else np.empty((0, 3, *vols.shape[1:]), np.float32))
    return banks
