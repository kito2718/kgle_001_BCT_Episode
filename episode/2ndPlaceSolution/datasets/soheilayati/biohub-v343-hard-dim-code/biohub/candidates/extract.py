"""High-recall candidate extraction, cached once and filtered afterwards.

The oracle ladder needs candidate recall *as a function of* threshold, NMS
radius and count — not at one operating point. Re-detecting per operating point
would cost an hour each time, so this extracts once at a deliberately permissive
setting and stores the score, so every later operating point is a filter over
the cache:

  threshold  -> `score >= s`
  count      -> `rank_in_frame < n`
  NMS radius -> greedy re-suppression in physical space (`nms.py`)

Permissive means: minimum spacing of one grid voxel (1.625 um, well under the
7 um matching radius), a low absolute threshold, and two blob scales unioned.
That is not a proposed operating point — it is the pool whose ceiling we want to
measure. `reports/baseline_floor.md` showed a *fixed* threshold emits ~110 peaks
per frame while true density spans 38-786, so the cache must be able to express
far more cells than we would ever submit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biohub.pipeline.baseline import DOWNSAMPLE, GRID_UM, downsample, normalise

# Blob scales in micrometres. Swept in `scripts/sweep_pool_recall.py` over 16
# density-stratified samples; results in `reports/pool_recall_sweep.parquet`.
#
#   sigmas             recall   worst sample   peaks/frame
#   (2.0,)             0.8958      0.333            467
#   (1.0, 1.5)         0.9876      0.967           2577
#   (1.0, 1.5, 2.5)    0.9889      0.967           2838
#
# The Stage 0 baseline used a single sigma of 2.0 um and lost 9 points of recall
# to it — and on one sample, two thirds. A single Laplacian-of-Gaussian scale is
# blind to cells much away from it, so a ceiling measured at one scale would be a
# property of the scale rather than of the images.
SIGMAS_UM: tuple[float, ...] = (1.0, 1.5, 2.5)

CANDIDATE_COLUMNS = (
    "t", "z", "y", "x",  # source-resolution voxels (float; sub-voxel from the grid map)
    "gz", "gy", "gx",  # downsampled-grid voxels
    "score",  # LoG response, comparable within a frame after per-frame normalisation
    "sigma_um",
    "intensity",  # normalised image value at the peak
    "rank_in_frame",  # 0 = strongest
)


@dataclass
class ExtractConfig:
    sigmas_um: tuple[float, ...] = SIGMAS_UM
    # Minimum peak spacing, in grid voxels. 1 == adjacent maxima allowed. The
    # pool is meant to be over-complete; deduplication happens downstream, where
    # it can be swept.
    min_distance: int = 1
    # Absolute floor as a quantile of the LoG response. Swept: 0.70 -> 0.9876
    # recall, 0.90 -> 0.9632, 0.95 -> 0.8576. Not free, so set at 0.70.
    threshold_quantile: float = 0.70
    top_k_per_frame: int = 3000
    # Normalisation floor. `normalise` clips below this quantile to exactly zero,
    # so any cell dimmer than it disappears entirely. The Stage 0 baseline used
    # 0.50 (the frame median) and on `6bba_05db0fb1` that annihilated 13 % of
    # annotated cells outright — their raw values (744-1326) sit below the frame
    # median (941). Lower is strictly safer for recall and measured to cost
    # nothing in peak count, which matters most for an unseen embryo.
    norm_lo_q: float = 0.02
    norm_hi_q: float = 0.999


def _peaks_one_scale(
    vol: np.ndarray, sigma_um: float, cfg: ExtractConfig
) -> tuple[np.ndarray, np.ndarray]:
    from scipy import ndimage
    from skimage.feature import peak_local_max

    sigma = sigma_um / GRID_UM
    smooth = ndimage.gaussian_filter(vol, sigma=sigma)
    # Negated Laplacian: responds to blob *centres*, not to bright plateaus.
    # Scale-normalised (x sigma^2) so responses from different sigmas are
    # comparable and the union is not dominated by the smaller scale.
    log = -ndimage.gaussian_laplace(smooth, sigma=sigma) * float(np.mean(sigma) ** 2)

    thresh = float(np.quantile(log, cfg.threshold_quantile))
    coords = peak_local_max(
        log,
        min_distance=cfg.min_distance,
        threshold_abs=thresh,
        exclude_border=False,
    )
    if len(coords) == 0:
        return coords.reshape(0, 3), np.zeros(0, dtype=np.float32)
    return coords, log[tuple(coords.T)].astype(np.float32)


def extract_frame(vol: np.ndarray, cfg: ExtractConfig) -> dict[str, np.ndarray]:
    """Candidates for one already-normalised, already-downsampled volume."""
    all_coords, all_scores, all_sigmas = [], [], []
    for s in cfg.sigmas_um:
        c, v = _peaks_one_scale(vol, s, cfg)
        all_coords.append(c)
        all_scores.append(v)
        all_sigmas.append(np.full(len(c), s, dtype=np.float32))

    coords = np.concatenate(all_coords) if all_coords else np.zeros((0, 3), int)
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0, np.float32)
    sigmas = np.concatenate(all_sigmas) if all_sigmas else np.zeros(0, np.float32)

    order = np.argsort(-scores)[: cfg.top_k_per_frame]
    coords, scores, sigmas = coords[order], scores[order], sigmas[order]

    intensity = vol[tuple(coords.T)].astype(np.float32) if len(coords) else np.zeros(0, np.float32)
    # Grid voxel -> source voxel: the grid cell spans DOWNSAMPLE voxels, so its
    # centre sits at the half-offset. Never assume isotropy here (CLAUDE.md §7).
    src = coords * DOWNSAMPLE + (DOWNSAMPLE - 1) / 2.0

    return {
        "z": src[:, 0], "y": src[:, 1], "x": src[:, 2],
        "gz": coords[:, 0], "gy": coords[:, 1], "gx": coords[:, 2],
        "score": scores,
        "sigma_um": sigmas,
        "intensity": intensity,
        "rank_in_frame": np.arange(len(coords), dtype=np.int32),
    }


def extract_dataset(zarr_path: Path, cfg: ExtractConfig | None = None):
    """Candidate table for every frame of one dataset."""
    import pandas as pd
    import zarr

    cfg = cfg or ExtractConfig()
    from biohub.pipeline.baseline import BaselineConfig

    norm_cfg = BaselineConfig(norm_lo_q=cfg.norm_lo_q, norm_hi_q=cfg.norm_hi_q)

    arr = zarr.open_array(str(Path(zarr_path) / "0"), mode="r")
    shape = tuple(arr.shape)

    frames = []
    for t in range(shape[0]):
        vol = downsample(normalise(np.asarray(arr[t]), norm_cfg))
        f = extract_frame(vol, cfg)
        f["t"] = np.full(len(f["score"]), t, dtype=np.int32)
        frames.append(pd.DataFrame(f))
        del vol

    df = pd.concat(frames, ignore_index=True)[list(CANDIDATE_COLUMNS)]
    return df.astype(
        {
            "t": "int32", "gz": "int16", "gy": "int16", "gx": "int16",
            "z": "float32", "y": "float32", "x": "float32",
            "rank_in_frame": "int32",
        }
    ), shape
