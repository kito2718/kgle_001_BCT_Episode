"""Supervision targets for the temporal detector — PLAN.md §9.2.

Everything here is built for the centre frame of a `[t-1, t, t+1]` stack, on the
`[1, 4, 4]` grid where the voxel is isotropic at 1.625 um.

**One convention, stated once: every vector is in micrometres.** Offsets, flows
and radii alike. Source voxels are anisotropic by 4x (CLAUDE.md §7) and mixing
the two conventions is the single easiest way to produce a plausible-looking
wrong number, so the conversion happens at the boundary and nowhere else.

Three supervision zones, per PLAN.md §2.6:

    positive     a ball around each annotated centre. Target is the Gaussian,
                 which is ~0 at the ball's edge -- so the ring of near-zero
                 targets around each cell is what teaches the peak to be sharp.
    background   image-derived: dim in per-sample normalised units. Target 0.
    unknown      everything else, including every unannotated cell. NO LOSS.

The third zone is the whole point. The official baseline treats every
non-ground-truth voxel as a negative at `neg_weight=1e-2`; 44b6 annotates 0.77 %
of its cells, so that trains ~99 % of real cells toward background.
`tau = 0.02` is measured, not chosen: `scripts/audit_supervision_zones.py` found
the dimmest annotated centre anywhere sits at 0.0475 in these units, and tau
covers 0 of 107,324 temporally persistent maxima.

Heatmaps from several cells combine with **max, not sum**. That matters for
divisions: the max of two Gaussians has a dip at the midpoint for *any*
separation, where a sum is unimodal below `2*sigma`. The closest division pair in
the training set (2.87 um) still leaves a 23 % dip at `sigma = 2.0`, so the
target never destroys the information -- whether the network reproduces the dip
is a separate, empirical question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biohub.metrics.gt import SCALE_ARR
from biohub.pipeline.baseline import DOWNSAMPLE, GRID_UM

# The grid is isotropic after [1,4,4], so one scalar is the true spacing here.
# Asserted rather than assumed -- if DOWNSAMPLE ever changes, this must fail.
VOXEL_UM = float(GRID_UM[0])
assert np.allclose(GRID_UM, VOXEL_UM), f"grid is not isotropic: {GRID_UM}"


@dataclass(frozen=True)
class TargetConfig:
    """All radii in micrometres.

    `sigma_um` default 2.0: below the measured cell scale (Gaussian-equivalent
    sigma 3.2 um in y/x) so the target is sharper than the cell's appearance,
    which favours localisation, and comfortably inside the division bound.
    Screened over {1.5, 2.0, 3.0}.
    """

    sigma_um: float = 2.0
    # Ball supervised by the heatmap loss. Defaults to 3 sigma (99.7 % of the
    # Gaussian) with a floor, so the near-zero tail is inside the loss.
    positive_radius_um: float | None = None
    # Offset and flow are properties of the cell, supervised in a small ball so
    # a peak found one voxel off still receives a correction.
    vector_radius_um: float = 2.0
    # Reliable-background threshold in normalised units.
    background_tau: float = 0.02

    @property
    def positive_radius(self) -> float:
        if self.positive_radius_um is not None:
            return self.positive_radius_um
        return max(3.0 * self.sigma_um, 4.0)


def src_to_grid(zyx_src: np.ndarray) -> np.ndarray:
    """Source-resolution voxels -> fractional grid voxels.

    Exact inverse of `candidates.extract`'s grid -> source mapping, which places
    a grid cell's centre at the half-offset of the voxels it pools.
    """
    return (np.asarray(zyx_src, dtype=np.float64) - (DOWNSAMPLE - 1) / 2.0) / DOWNSAMPLE


def grid_to_src(zyx_grid: np.ndarray) -> np.ndarray:
    return np.asarray(zyx_grid, dtype=np.float64) * DOWNSAMPLE + (DOWNSAMPLE - 1) / 2.0


def _ball(centre_grid: np.ndarray, radius_um: float, shape: tuple[int, int, int]):
    """Voxel indices within `radius_um` of a fractional grid centre, and their
    displacement *to* that centre in micrometres.

    Returns `(index_tuple, delta_um)` where `delta_um[k]` is the vector from
    voxel k's centre to the cell centre — i.e. what a prediction at that voxel
    must add to land on the cell.
    """
    r_vox = radius_um / VOXEL_UM
    lo = np.maximum(np.floor(centre_grid - r_vox).astype(int), 0)
    hi = np.minimum(np.ceil(centre_grid + r_vox).astype(int) + 1, np.asarray(shape))
    if np.any(lo >= hi):
        return None, None

    grids = np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij")
    coords = np.stack([g.ravel() for g in grids], axis=1)
    delta_um = (centre_grid[None, :] - coords) * VOXEL_UM
    keep = np.linalg.norm(delta_um, axis=1) <= radius_um
    if not keep.any():
        return None, None
    coords, delta_um = coords[keep], delta_um[keep]
    return (coords[:, 0], coords[:, 1], coords[:, 2]), delta_um


def make_targets(
    centres_src: np.ndarray,
    shape: tuple[int, int, int],
    vol_norm: np.ndarray | None = None,
    parents_src: np.ndarray | None = None,
    cfg: TargetConfig | None = None,
) -> dict[str, np.ndarray]:
    """Targets for one centre frame.

    `centres_src`   (N, 3) annotated centres at this frame, source voxels.
    `shape`         grid shape, e.g. (64, 64, 64).
    `vol_norm`      the normalised, downsampled centre frame. Only the
                    background zone needs it; pass None to supervise the
                    positive zone alone.
    `parents_src`   (N, 3) each centre's predecessor position at `t-1`, source
                    voxels, or NaN where there is none. At a division both
                    daughters carry the *same* parent, which is what makes their
                    backward vectors converge.
    """
    cfg = cfg or TargetConfig()
    centres_src = np.asarray(centres_src, dtype=np.float64).reshape(-1, 3)
    z, y, x = shape

    heatmap = np.zeros(shape, dtype=np.float32)
    positive = np.zeros(shape, dtype=bool)
    offset = np.zeros((3, z, y, x), dtype=np.float32)
    flow = np.zeros((3, z, y, x), dtype=np.float32)
    vec_mask = np.zeros(shape, dtype=bool)
    flow_mask = np.zeros(shape, dtype=bool)

    centres_grid = src_to_grid(centres_src)
    two_sigma_sq = 2.0 * cfg.sigma_um**2

    for i, c in enumerate(centres_grid):
        idx, delta_um = _ball(c, cfg.positive_radius, shape)
        if idx is None:
            continue
        g = np.exp(-np.sum(delta_um**2, axis=1) / two_sigma_sq).astype(np.float32)
        # max, not sum: two cells never add up past 1, and the dip between a
        # division's daughters survives at any separation.
        np.maximum.at(heatmap, idx, g)
        positive[idx] = True

        vidx, vdelta = _ball(c, cfg.vector_radius_um, shape)
        if vidx is None:
            continue
        # Offset is per-voxel: the vector from this voxel to the cell centre.
        for a in range(3):
            offset[a][vidx] = vdelta[:, a]
        vec_mask[vidx] = True

        if parents_src is None:
            continue
        p = np.asarray(parents_src, dtype=np.float64).reshape(-1, 3)[i]
        if not np.all(np.isfinite(p)):
            continue
        # Backward flow: from THIS cell at t to its parent at t-1. A cell that
        # moved +x since the previous frame therefore has a negative x flow.
        # Constant over the ball -- it is a property of the cell, not the voxel.
        back_um = (p - centres_src[i]) * SCALE_ARR
        for a in range(3):
            flow[a][vidx] = back_um[a]
        flow_mask[vidx] = True

    if vol_norm is not None:
        background = (np.asarray(vol_norm) < cfg.background_tau) & ~positive
    else:
        background = np.zeros(shape, dtype=bool)

    return {
        "heatmap": heatmap,
        # 1 where the heatmap loss applies. Everything else -- every unannotated
        # cell included -- is unknown and contributes nothing.
        "heatmap_mask": (positive | background).astype(np.float32),
        "positive": positive,
        "background": background,
        "offset": offset,
        "offset_mask": vec_mask.astype(np.float32),
        "flow": flow,
        "flow_mask": flow_mask.astype(np.float32),
    }


def parents_for_frame(gt, t: int) -> tuple[np.ndarray, np.ndarray]:
    """`(centres, parents)` in source voxels for the annotated nodes at `t`.

    A node with no annotated predecessor gets NaN, so it is skipped by the flow
    loss rather than supervised toward zero motion. At a division both daughters
    resolve to the same parent row, which is the convergence property
    `tests/test_detector_targets.py` pins down.
    """
    rows = np.flatnonzero(gt.t == t)
    centres = gt.zyx[rows]
    parents = np.full((len(rows), 3), np.nan)
    if gt.n_edges:
        parent_of = {int(d): int(s) for s, d in gt.edges.tolist()}
        for k, r in enumerate(rows):
            p = parent_of.get(int(r))
            if p is not None and int(gt.t[p]) == t - 1:
                parents[k] = gt.zyx[p]
    return centres, parents


def decode_peaks(
    peaks_grid: np.ndarray,
    offset_um: np.ndarray | None = None,
) -> np.ndarray:
    """Grid peaks (+ predicted offset in um) -> source-resolution voxels.

    The inverse of the target convention, so a coordinate mistake shows up as a
    round-trip failure rather than as a slightly worse score.
    """
    src = grid_to_src(np.asarray(peaks_grid, dtype=np.float64))
    if offset_um is not None:
        src = src + np.asarray(offset_um, dtype=np.float64) / SCALE_ARR
    return src
