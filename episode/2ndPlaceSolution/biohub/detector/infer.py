"""Test-time detection: a zarr volume in, a candidate table out.

This began as the inference half of `scripts/detect_candidates.py`, moved into
the package because **only `src/` ships to Kaggle** — anything left in
`scripts/` cannot run in the notebook. Since the Stage 3 schema extension the
script *delegates* here, so there is one implementation and the pool the
overnight cascade writes and the pool the notebook builds are the same artifact
by construction rather than by kept promises.

Ensembling averages the **heatmaps**, not the extracted peaks. Peak extraction,
ranking and the graph score are each nonlinear, so averaging afterwards would be
a different and worse operation — and averaging before extraction is what lets
several fold models behave as one detector rather than as a vote.

The model's `features` head is averaged the same way and sampled at each
selected peak: `emb_0..emb_47` is the per-candidate embedding Stage 4's edge
model consumes (stage-03 §2.1). It is trained and computed on every voxel of
every forward pass already; this file only reads it out. Three cheap
shape/context features ride in the same pass (stage-03 §2.3):

    density_10um    other candidates within 10 um in the same frame — the
                    crowding covariate of stage-02 §13, computed on this pool
    peak_width_um   Gaussian-equivalent sigma of the heatmap peak, from the
                    ±1-voxel neighbour ratio per axis, averaged; um
    boundary_um     distance to the nearest volume face, um

**No ensemble score variance, deliberately** — stage-03 §2.1 suggested it as
free. It is free only at deployment: under the out-of-fold discipline exactly
one fold model is clean per training sample (the other four trained on it), so
a locally computed variance is either leaked — and distribution-shifted against
deployment, where all five models are clean — or absent from the training pool,
and a feature the edge model cannot train with is dead weight at inference.
`flow_sigma_um` and `peak_width_um` carry the uncertainty signal instead.
Recorded in `Progress/stage-03-candidates.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biohub.detector.targets import VOXEL_UM, grid_to_src
from biohub.metrics.gt import SCALE_ARR
from biohub.pipeline.baseline import DOWNSAMPLE

TOP_K_PER_FRAME = 3000       # matches candidates_v1/v2, so pools stay comparable
SCORE_FLOOR = 0.01
NEAR_SCORE_FRAC = 0.5        # "comparable" neighbour, for the close-pair flag
DENSITY_RADIUS_UM = 10.0     # the stage-02 §13 crowding covariate's radius

# Two-centre hypotheses (stage-03 §2.2) — MEASURED DEAD 2026-08-09, kept as an
# instrument behind `detect(..., return_alt=True)`, never written in
# production. Over all 151 training divisions only 10 are pool-merged (one
# peak where two daughters are; the other 141 have distinct candidates — the
# division problem is selection, not synthesis). The covariance splitter below
# reaches at most 6 of the 10 and only while emitting 250–330 pairs/frame; the
# covering pairs rank #16–#458 by every available ordering inside their own
# frames, so no operating point separates them from junk. Full sweep:
# `reports/two_centre_sweep.parquet`, `scripts/measure_two_centre.py`,
# `Progress/stage-03-candidates.md`.
ALT_RANK_MAX = 1500          # beyond this rank a peak can never be selected
ALT_PRE_SIGMA_UM = 2.6       # cheap axis-width pre-filter before the covariance
ALT_WINDOW_VOX = 3           # ±3 grid voxels ≈ ±4.9 um of local heat mass
ALT_MASS_FRAC = 0.25         # mass floor, fraction of the peak value
ALT_MIN_RATIO = 1.2          # major/minor sigma ratio — isotropic blobs drop out
ALT_SEP_RANGE_UM = (2.5, 12.0)   # plausible daughter separations (q10 6.36)
ALT_MIN_NEAR_UM = 4.0        # a comparable twin already extracted ⇒ skip
ALT_MAX_PER_FRAME = 64       # pairs; volume guard, strongest parents first

BASE_COLUMNS = ("t", "z", "y", "x", "score", "rank_in_frame",
                "flow_z", "flow_y", "flow_x", "flow_sigma_um", "near_um",
                "density_10um", "peak_width_um", "boundary_um")
ALT_DIAG_COLUMNS = ("alt_of", "alt_sep_um", "alt_ratio", "alt_major_um")


def emb_columns(n: int) -> list[str]:
    return [f"emb_{j}" for j in range(n)]


def load_models(paths, device):
    """Load one or more checkpoints onto `device`, in eval mode.

    Every checkpoint must share an architecture — they are averaged in heatmap
    space, which is meaningless across different widths.
    """
    import torch

    from biohub.detector.model import TemporalUNet3d

    models, spec = [], None
    for p in paths:
        ck = torch.load(str(p), map_location=device, weights_only=False)
        in_frames = int(ck.get("in_frames", 3))
        here = (in_frames, tuple(ck["widths"]), ck["feature_ch"])
        if spec is None:
            spec = here
        elif here != spec:
            raise ValueError(f"{Path(p).name}: architecture {here} != {spec}; "
                             "heatmap averaging requires identical models")
        m = TemporalUNet3d(in_frames, tuple(ck["widths"]), ck["feature_ch"]).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)
    return models


def volumes_from_zarr(zarr_path: Path, norm_lo_q: float = 0.02,
                      norm_hi_q: float = 0.999) -> np.ndarray:
    """(T, Z, Y, X) float16, normalised then downsampled — training's order.

    The order matters and is not interchangeable: normalisation quantiles are
    estimated on the full-resolution frame, so downsampling first would change
    them. `biohub.detector.dataset` documents the same sequence.
    """
    import zarr

    from biohub.pipeline.baseline import BaselineConfig, downsample, normalise

    cfg = BaselineConfig(norm_lo_q=norm_lo_q, norm_hi_q=norm_hi_q)
    arr = zarr.open_array(str(Path(zarr_path) / "0"), mode="r")
    return np.stack([
        downsample(normalise(np.asarray(arr[t]), cfg)) for t in range(arr.shape[0])
    ]).astype(np.float16)


def _stacks(vols: np.ndarray, ts: list[int], in_frames: int = 3) -> np.ndarray:
    """Odd centred context per frame, edge-replicated at the ends.

    Replication rather than zeros: a zero frame is a scene with every cell gone,
    which is a motion signal the flow head would have to learn to ignore.
    """
    if in_frames < 1 or in_frames % 2 != 1:
        raise ValueError("in_frames must be a positive odd integer")
    n = len(vols)
    radius = in_frames // 2
    out = np.empty((len(ts), in_frames, *vols.shape[1:]), dtype=np.float32)
    for i, t in enumerate(ts):
        for k, dt in enumerate(range(-radius, radius + 1)):
            out[i, k] = vols[min(max(t + dt, 0), n - 1)]
    return out


def _frame_aligned_heat(logits: np.ndarray) -> np.ndarray:
    """Symmetrically align model logits per frame, average, then sigmoid.

    ``logits`` is ``(models, batch, z, y, x)``. Each model/frame is mapped to
    the arithmetic mean of model means and geometric mean of model standard
    deviations. The transform has no fitted state and preserves each model's
    within-frame spatial ordering. See Stage 5 §2.
    """
    if logits.ndim != 5 or logits.shape[0] < 1:
        raise ValueError("expected logits shaped (models, batch, z, y, x)")
    axes = (2, 3, 4)
    means = logits.mean(axis=axes, keepdims=True, dtype=np.float64)
    scales = logits.std(axis=axes, keepdims=True, dtype=np.float64)
    scales = np.maximum(scales, 1e-6)
    ref_mean = means.mean(axis=0)
    ref_scale = np.exp(np.log(scales).mean(axis=0))
    aligned = (logits - means) * (ref_scale / scales) + ref_mean
    mean_logit = aligned.mean(axis=0)
    return (1.0 / (1.0 + np.exp(-np.clip(mean_logit, -80, 80)))).astype(
        np.float32
    )


def _neighbour_features(um: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(near_um, density_10um)` from one KD-tree over the frame's candidates.

    `near_um` is the distance to the nearest neighbour whose score is at least
    half this one's: a much weaker peak nearby is a shoulder, not a second
    cell. `density_10um` counts the others within `DENSITY_RADIUS_UM`,
    unconditionally on score, matching stage-02 §13's `crowd10`.
    """
    from scipy.spatial import cKDTree

    n = len(um)
    near = np.full(n, np.inf)
    dens = np.zeros(n, dtype=np.int32)
    if n < 2:
        return near, dens

    tree = cKDTree(um)
    dens = np.asarray(
        tree.query_ball_point(um, DENSITY_RADIUS_UM, return_length=True),
        dtype=np.int32) - 1
    d, j = tree.query(um, k=min(8, n))
    for a in range(n):
        for b in range(1, d.shape[1]):
            if s[j[a, b]] >= NEAR_SCORE_FRAC * s[a]:
                near[a] = d[a, b]
                break
    return near, dens


def _axis_sigmas(heat: np.ndarray, peaks: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(N, 3) Gaussian-equivalent sigma of each heatmap peak, grid voxels.

    Per axis, the mean of the two ±1-voxel neighbours over the peak value
    inverts to `sigma = sqrt(-1 / (2 ln r))` for a Gaussian. The ratio is
    clipped at 0.999 (sigma ≈ 22 voxels), so saturated or plateaued peaks
    read as "very wide" — the informative direction. At a volume edge the
    clipped neighbour is the peak itself, which also biases wide;
    `boundary_um` carries that flag downstream.
    """
    shape = heat.shape
    sig = np.empty((len(peaks), 3), dtype=np.float64)
    for a in range(3):
        lo = peaks.copy()
        lo[:, a] = np.maximum(peaks[:, a] - 1, 0)
        hi = peaks.copy()
        hi[:, a] = np.minimum(peaks[:, a] + 1, shape[a] - 1)
        r = (heat[tuple(lo.T)] + heat[tuple(hi.T)]) / (2.0 * s)
        r = np.clip(r, 1e-6, 0.999)
        sig[:, a] = np.sqrt(-1.0 / (2.0 * np.log(r)))
    return sig


def _peak_width_um(heat: np.ndarray, peaks: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Mean of the three axis sigmas, micrometres — the isotropic grid makes
    one `VOXEL_UM` valid for every axis."""
    return _axis_sigmas(heat, peaks, s).mean(axis=1) * VOXEL_UM


def _split_two_centres(heat: np.ndarray, peak: np.ndarray, s: float,
                       window: int = ALT_WINDOW_VOX,
                       mass_frac: float = ALT_MASS_FRAC):
    """Principal-axis split of one peak's local heat mass.

    The mass above `mass_frac * s` in a ±`window` box is treated as a weighted
    point cloud; for two Gaussians of width sigma_g separated by d along the
    major axis, the covariance eigenvalues satisfy
    `lambda_major - lambda_minor ≈ (d/2)^2`, so the separation estimate is
    `2 * sqrt(lambda_major - lambda_minor)` and the two centres sit at the
    centroid ± half that along the major eigenvector. The mass floor truncates
    both tails, which shrinks the estimate somewhat — a hypothesis generator,
    not a localiser; the edge model rescores whatever this proposes.

    Returns `(c1, c2, sep_um, ratio, major_um)` in fractional grid coords, or
    None when the local mass is degenerate.
    """
    shape = heat.shape
    lo = np.maximum(peak - window, 0)
    hi = np.minimum(peak + window + 1, np.asarray(shape))
    sub = heat[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float64)
    w = sub - mass_frac * float(s)
    np.maximum(w, 0.0, out=w)
    tot = w.sum()
    if tot <= 0.0:
        return None

    axes = [np.arange(lo[a], hi[a], dtype=np.float64) for a in range(3)]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
    wf = w.ravel() / tot
    mu = wf @ pts
    d = pts - mu
    cov = (d * wf[:, None]).T @ d
    evals, evecs = np.linalg.eigh(cov)              # ascending
    lam_major = max(float(evals[2]), 0.0)
    lam_minor = max(0.5 * float(evals[0] + evals[1]), 1e-9)
    sep_vox = 2.0 * np.sqrt(max(lam_major - lam_minor, 0.0))
    v = evecs[:, 2]
    c1 = mu + 0.5 * sep_vox * v
    c2 = mu - 0.5 * sep_vox * v
    ratio = float(np.sqrt(lam_major / lam_minor))
    return c1, c2, sep_vox * VOXEL_UM, ratio, np.sqrt(lam_major) * VOXEL_UM


def _boundary_um(src: np.ndarray, grid_shape: tuple[int, ...]) -> np.ndarray:
    """Distance to the nearest volume face, micrometres, floored at zero —
    the offset head can push a refined centre fractionally outside."""
    extent = np.asarray(grid_shape) * DOWNSAMPLE          # source-voxel shape
    d_lo = src * SCALE_ARR
    d_hi = (extent - 1 - src) * SCALE_ARR
    return np.maximum(np.minimum(d_lo, d_hi).min(axis=1), 0.0)


def detect(models, vols: np.ndarray, device, amp_dtype=None,
           batch: int = 8, return_alt: bool = False,
           heat_alignment: str = "probability_mean",
           vector_model_count: int | None = None):
    """Candidate table for one dataset. `models` may hold one model or several.

    With `return_alt=True` also returns the two-centre hypothesis table:
    `(main, alt)` — an INSTRUMENT, measured dead for production (see the ALT_*
    constants' comment). Alt rows are synthesized daughters, two per flagged
    broad peak, in a separate table so the main pool is untouched. `alt_of`
    joins an alt row to its parent via `(t, alt_of == parent.rank_in_frame)`.
    """
    import torch
    from skimage.feature import peak_local_max

    if heat_alignment not in {"probability_mean", "frame_logit_z"}:
        raise ValueError(f"unknown heat alignment: {heat_alignment}")
    if vector_model_count is None:
        vector_model_count = len(models)
    if not 1 <= vector_model_count <= len(models):
        raise ValueError("vector_model_count must be between 1 and len(models)")

    n_emb = models[0].feature.out_channels
    rows, alt_rows = [], []
    n_t = len(vols)
    for start in range(0, n_t, batch):
        ts = list(range(start, min(start + batch, n_t)))
        context = int(getattr(models[0], "in_frames", 3))
        if any(int(getattr(model, "in_frames", 3)) != context for model in models):
            raise ValueError("all detector models must share in_frames")
        x = torch.from_numpy(_stacks(vols, ts, context)).to(device)

        heat = flow = logvar = offset = feats = None
        heat_logits = []
        with torch.no_grad():
            for model_index, m in enumerate(models):
                use_amp = amp_dtype is not None and device.type == "cuda"
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    out = m(x)
                raw_heat = out["heatmap"][:, 0].float().cpu().numpy()
                if heat_alignment == "frame_logit_z":
                    heat_logits.append(raw_heat)
                else:
                    h = torch.sigmoid(out["heatmap"][:, 0].float()).cpu().numpy()
                    heat = h if heat is None else heat + h
                # A heatmap-only secondary view must not shift the primary
                # detector's directed heads or learned embedding coordinates.
                if model_index >= vector_model_count:
                    continue
                f = out["flow"].float().cpu().numpy()
                lv = out["flow_logvar"][:, 0].float().cpu().numpy()
                o = out["offset"].float().cpu().numpy()
                # Features stay on the device: (B, C, Z, Y, X) float32 is
                # ~0.4 GB at batch 8, and only the peak samples ever leave.
                fe = out["features"].float()
                if flow is None:
                    flow, logvar, offset, feats = f, lv, o, fe
                else:
                    flow += f
                    logvar += lv
                    offset += o
                    feats += fe
        heat_k = float(len(models))
        vector_k = float(vector_model_count)
        if heat_alignment == "frame_logit_z":
            heat = _frame_aligned_heat(np.stack(heat_logits, axis=0))
        else:
            heat = heat / heat_k
        flow, logvar, offset = flow / vector_k, logvar / vector_k, offset / vector_k
        feats = feats / vector_k

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
            bound = _boundary_um(src, vols.shape[1:])

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

            if not return_alt:
                continue

            # ---- two-centre hypotheses (see the constants' comment) --------
            flagged = np.flatnonzero(
                (np.arange(len(peaks)) < ALT_RANK_MAX)
                & (axis_sig.max(axis=1) * VOXEL_UM >= ALT_PRE_SIGMA_UM)
                & (near >= ALT_MIN_NEAR_UM))
            pairs = []
            for a in flagged:                       # ascending index = by rank
                res = _split_two_centres(heat[i], peaks[a], float(s[a]))
                if res is None:
                    continue
                c1, c2, sep_um, ratio, major_um = res
                if ratio < ALT_MIN_RATIO:
                    continue
                if not ALT_SEP_RANGE_UM[0] <= sep_um <= ALT_SEP_RANGE_UM[1]:
                    continue
                pairs.append((a, c1, c2, sep_um, ratio, major_um))
                if len(pairs) >= ALT_MAX_PER_FRAME:
                    break
            if not pairs:
                continue

            par = np.repeat([p[0] for p in pairs], 2)
            centres = np.array([c for p in pairs for c in (p[1], p[2])])
            centres = np.clip(centres, 0, np.asarray(heat[i].shape) - 1)
            g = np.round(centres).astype(np.int64)
            gidx = tuple(g.T)
            asrc = grid_to_src(centres)             # no offset refinement here
            af = flow[i][:, gidx[0], gidx[1], gidx[2]].T
            asig = np.sqrt(np.exp(np.clip(logvar[i][gidx], -8, 8)))
            ge = torch.from_numpy(np.ascontiguousarray(g)).long().to(feats.device)
            aemb = feats[i][:, ge[:, 0], ge[:, 1], ge[:, 2]].T.cpu().numpy()
            aemb = np.clip(aemb, -65504.0, 65504.0).astype(np.float16)

            adata = {
                "t": np.full(len(par), t, dtype=np.int32),
                "z": asrc[:, 0].astype(np.float32),
                "y": asrc[:, 1].astype(np.float32),
                "x": asrc[:, 2].astype(np.float32),
                # The parent's evidence, inherited: the hypothesis is an
                # alternative reading of the SAME peak, not new detection.
                "score": s[par].astype(np.float32),
                "rank_in_frame": par.astype(np.int32),
                "flow_z": af[:, 0].astype(np.float32),
                "flow_y": af[:, 1].astype(np.float32),
                "flow_x": af[:, 2].astype(np.float32),
                "flow_sigma_um": asig.astype(np.float32),
                "near_um": np.repeat([p[3] for p in pairs], 2).astype(np.float32),
                "density_10um": (dens[par] + 1).astype(np.int32),
                "peak_width_um": width[par].astype(np.float32),
                "boundary_um": _boundary_um(asrc, vols.shape[1:]).astype(np.float32),
                "alt_of": par.astype(np.int32),
                "alt_sep_um": np.repeat([p[3] for p in pairs], 2).astype(np.float32),
                "alt_ratio": np.repeat([p[4] for p in pairs], 2).astype(np.float32),
                "alt_major_um": np.repeat([p[5] for p in pairs], 2).astype(np.float32),
            }
            adata.update({c: aemb[:, j] for j, c in enumerate(emb_columns(n_emb))})
            alt_rows.append(pd.DataFrame(adata))

    cols = list(BASE_COLUMNS) + emb_columns(n_emb)
    main = (pd.concat(rows, ignore_index=True)[cols] if rows
            else _empty_frame(cols))
    if not return_alt:
        return main
    alt_cols = (list(BASE_COLUMNS) + list(ALT_DIAG_COLUMNS) + emb_columns(n_emb))
    alt = (pd.concat(alt_rows, ignore_index=True)[alt_cols] if alt_rows
           else _empty_frame(alt_cols))
    return main, alt


def _empty_frame(cols) -> pd.DataFrame:
    def _dtype(c):
        if c in ("t", "rank_in_frame", "density_10um", "alt_of"):
            return np.int32
        return np.float16 if c.startswith("emb_") else np.float32
    return pd.DataFrame({c: np.zeros(0, dtype=_dtype(c)) for c in cols})
