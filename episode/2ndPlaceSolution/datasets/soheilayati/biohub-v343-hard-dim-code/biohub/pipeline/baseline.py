"""Deliberately simple end-to-end tracker — PLAN.md §7, Sunday 9 August.

Its job is **not** to score well. It exists to prove that packaging, hidden-test
discovery, output format, memory behaviour and runtime all work, weeks before
the real model needs them. Everything here is classical: no weights to ship, so
the first Kaggle submission cannot fail on a missing checkpoint.

  discover -> normalise -> LoG peaks -> Hungarian nearest-neighbour links
           -> validate -> submission.csv

Hungarian assignment is one-to-one, which makes in-degree <= 1 and out-degree
<= 1 structurally true. The baseline therefore predicts no divisions at all --
a deliberate simplification, since divisions are 0.1 % of edges and carry a
0.1 weight.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biohub.io.submission import DatasetGraph

# Physical voxel size of the source data, micrometres (DATA.md §2).
SCALE_UM = np.array([1.625, 0.40625, 0.40625])
DOWNSAMPLE = np.array([1, 4, 4])
# After [1,4,4] the grid is isotropic at 1.625 um.
GRID_UM = SCALE_UM * DOWNSAMPLE


@dataclass
class BaselineConfig:
    smooth_sigma_um: float = 2.0
    # Half the 7 um matching tolerance: two detections closer than this would
    # compete for one ground-truth node, so one of them is redundant.
    nms_radius_um: float = 3.5
    # Peak threshold as a quantile of the per-frame normalised volume.
    # Swept against GT recall on 44b6_0113de3b: 0.9995 -> 31 peaks/frame and 0%
    # recall; 0.998 -> 71 and 21%; 0.995 -> 110 and 100%. Set at the knee, not
    # tuned for score.
    peak_quantile: float = 0.995
    # Linking gate. Set from the metric's own tolerance rather than measured
    # motion, which the oracle ladder has not yet produced.
    max_link_um: float = 7.0
    norm_lo_q: float = 0.50
    norm_hi_q: float = 0.999


def discover_datasets(test_root: Path) -> list[Path]:
    """Every `.zarr` under the test root.

    Never assume the example filenames or count -- PLAN.md §14.4. The hidden
    test is swapped in at rerun and is roughly the size of the training set.
    """
    found = sorted(p for p in test_root.glob("*.zarr") if (p / "0").exists())
    if not found:
        raise FileNotFoundError(f"no .zarr datasets under {test_root}")
    return found


def normalise(frame: np.ndarray, cfg: BaselineConfig) -> np.ndarray:
    """Per-frame robust quantile normalisation.

    Required, not optional: 44b6 is ~6x brighter than 6bba at the median and the
    gap varies by quantile (reports/annotation_regime_audit.md §2), so any
    absolute intensity threshold would not transfer between embryos -- let alone
    to unseen ones.
    """
    lo, hi = np.quantile(frame, [cfg.norm_lo_q, cfg.norm_hi_q])
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.float32)
    return np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0.0, None)


def downsample(frame: np.ndarray) -> np.ndarray:
    dz, dy, dx = DOWNSAMPLE
    z, y, x = frame.shape
    return frame.reshape(z // dz, dz, y // dy, dy, x // dx, dx).mean(axis=(1, 3, 5))


def detect_frame(vol: np.ndarray, cfg: BaselineConfig) -> np.ndarray:
    """Return (N, 3) peak coordinates in downsampled-grid voxels."""
    from scipy import ndimage
    from skimage.feature import peak_local_max

    # Laplacian-of-Gaussian: responds to blob centres rather than bright plateaus.
    sigma = cfg.smooth_sigma_um / GRID_UM
    smooth = ndimage.gaussian_filter(vol, sigma=sigma)
    log = -ndimage.gaussian_laplace(smooth, sigma=sigma)

    thresh = float(np.quantile(log, cfg.peak_quantile))
    min_dist = max(1, int(round(cfg.nms_radius_um / GRID_UM[1])))
    return peak_local_max(log, min_distance=min_dist, threshold_abs=thresh, exclude_border=False)


def link_frames(a: np.ndarray, b: np.ndarray, cfg: BaselineConfig) -> list[tuple[int, int, float]]:
    """One-to-one nearest-neighbour assignment between consecutive frames.

    Returns (index_in_a, index_in_b, confidence). Hungarian assignment on
    physical distance, gated at `max_link_um`.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    if len(a) == 0 or len(b) == 0:
        return []

    a_um, b_um = a * GRID_UM, b * GRID_UM
    # Sparse gate first: an all-pairs cost matrix is exactly what PLAN.md §18
    # forbids, and at ~800 detections per frame it is also needless.
    tree = cKDTree(b_um)
    pairs = tree.query_ball_point(a_um, r=cfg.max_link_um)

    rows, cols, costs = [], [], []
    for i, js in enumerate(pairs):
        for j in js:
            rows.append(i)
            cols.append(j)
            costs.append(float(np.linalg.norm(a_um[i] - b_um[j])))
    if not rows:
        return []

    # Dense matrix over only the gated candidates' index ranges.
    big = cfg.max_link_um * 10
    cost = np.full((len(a), len(b)), big, dtype=np.float32)
    cost[rows, cols] = costs
    ri, ci = linear_sum_assignment(cost)

    out = []
    for i, j in zip(ri, ci):
        d = cost[i, j]
        if d < cfg.max_link_um:
            # Confidence orders the output; closer link = more trusted.
            out.append((int(i), int(j), float(1.0 / (1.0 + d))))
    return out


def track_dataset(zarr_path: Path, cfg: BaselineConfig) -> DatasetGraph:
    """Detect and link one dataset, returning its predicted graph."""
    import zarr

    arr = zarr.open_array(str(zarr_path / "0"), mode="r")
    n_t = arr.shape[0]

    nodes: dict[int, tuple[int, int, int, int]] = {}
    edges: list[tuple[int, int, float]] = []
    next_id = 0
    prev_peaks: np.ndarray | None = None
    prev_ids: list[int] = []

    for t in range(n_t):
        vol = downsample(normalise(np.asarray(arr[t]), cfg))
        peaks = detect_frame(vol, cfg)

        # Map back to ORIGINAL voxel coordinates -- the metric applies the
        # physical scale itself and expects integer source-resolution voxels.
        ids = []
        for p in peaks:
            orig = np.round(p * DOWNSAMPLE + (DOWNSAMPLE - 1) / 2).astype(int)
            orig = np.clip(orig, 0, np.array(arr.shape[1:]) - 1)
            nodes[next_id] = (t, int(orig[0]), int(orig[1]), int(orig[2]))
            ids.append(next_id)
            next_id += 1

        if prev_peaks is not None and len(peaks):
            for i, j, conf in link_frames(prev_peaks, peaks, cfg):
                edges.append((prev_ids[i], ids[j], conf))

        prev_peaks, prev_ids = peaks, ids
        del vol

    gc.collect()
    return DatasetGraph(
        dataset=zarr_path.stem, nodes=nodes, edges=edges, shape=tuple(arr.shape)
    )
