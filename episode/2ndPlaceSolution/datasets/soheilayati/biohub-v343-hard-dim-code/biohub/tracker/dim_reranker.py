"""Inference-only features and bounded rank transfer for dim cell candidates.

The reranker deliberately keeps ``candidate.score`` and ``rank_in_frame``
unchanged.  It emits a separate ``selection_score`` whose per-frame value
distribution is exactly the original score distribution.  Downstream edge,
division and post-pass models therefore continue to see their trained feature
semantics; only NMS/superset/solver priority changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from biohub.detector.targets import src_to_grid


IMAGE_FEATURES = (
    "image_center",
    "image_core_mean",
    "image_core_max",
    "image_core_std",
    "image_shell_median",
    "image_shell_std",
    "image_mean_contrast",
    "image_peak_contrast",
    "image_peak_snr",
)

MODEL_FEATURES = (
    "base_score",
    "base_logit",
    "rank_fraction",
    "frame_score_median",
    "frame_score_q90",
    "near_um",
    "density_10um",
    "peak_width_um",
    "boundary_um",
    "flow_sigma_um",
    "log_span",
    "centered3",
    "centered5",
    "chain_residual_um",
    "has_predecessor",
    "has_successor",
    *IMAGE_FEATURES,
)

DIM_MODEL_FEATURES = IMAGE_FEATURES


def _gather(volume: np.ndarray, times: np.ndarray, centres: np.ndarray,
            offsets: np.ndarray) -> np.ndarray:
    shape = np.asarray(volume.shape[1:], dtype=np.int64)
    out = np.empty((len(times), len(offsets)), dtype=np.float32)
    for j, off in enumerate(offsets):
        pos = np.clip(centres + off, 0, shape - 1)
        out[:, j] = np.asarray(
            volume[times, pos[:, 0], pos[:, 1], pos[:, 2]], dtype=np.float32
        )
    return out


def candidate_image_features(volume: np.ndarray, candidate: pd.DataFrame) -> pd.DataFrame:
    """Cheap local descriptors at prediction coordinates on the detector grid.

    The core is the centre plus its six one-grid-step axial neighbours
    (1.625 um).  The shell uses the six axes at three and four grid steps
    (4.875 and 6.5 um).  This approximates the registered 3-D core/shell
    contrast while requiring only 19 indexed samples per raw candidate.
    """
    required = {"t", "z", "y", "x"}
    if not required.issubset(candidate):
        raise ValueError(f"candidate lacks {sorted(required - set(candidate.columns))}")
    if volume.ndim != 4:
        raise ValueError("volume must have shape (time,z,y,x)")
    times = candidate.t.to_numpy(np.int64)
    source = candidate[["z", "y", "x"]].to_numpy(np.float64)
    if not np.isfinite(source).all() or (times < 0).any() or (times >= len(volume)).any():
        raise ValueError("invalid candidate image query")
    centres = np.rint(src_to_grid(source)).astype(np.int64)
    axes = np.eye(3, dtype=np.int64)
    core_offsets = np.concatenate([np.zeros((1, 3), np.int64), axes, -axes])
    shell_offsets = np.concatenate([3 * axes, -3 * axes, 4 * axes, -4 * axes])
    core = _gather(volume, times, centres, core_offsets)
    shell = _gather(volume, times, centres, shell_offsets)
    center = core[:, 0]
    core_mean = core.mean(axis=1)
    core_max = core.max(axis=1)
    core_std = core.std(axis=1)
    shell_median = np.median(shell, axis=1)
    shell_std = shell.std(axis=1)
    mean_contrast = core_mean - shell_median
    peak_contrast = core_max - shell_median
    return pd.DataFrame({
        "image_center": center,
        "image_core_mean": core_mean,
        "image_core_max": core_max,
        "image_core_std": core_std,
        "image_shell_median": shell_median,
        "image_shell_std": shell_std,
        "image_mean_contrast": mean_contrast,
        "image_peak_contrast": peak_contrast,
        "image_peak_snr": peak_contrast / np.maximum(shell_std, 0.02),
    }, dtype=np.float32)


def candidate_model_features(candidate: pd.DataFrame, state: pd.DataFrame,
                             image: pd.DataFrame) -> pd.DataFrame:
    """Assemble the fixed specialist feature table without ground truth."""
    if not (len(candidate) == len(state) == len(image)):
        raise ValueError("candidate, state and image rows must align")
    required_candidate = {
        "t", "score", "rank_in_frame", "near_um", "density_10um",
        "peak_width_um", "boundary_um", "flow_sigma_um",
    }
    required_state = {
        "span", "centered3", "centered5", "residual_um", "predecessor", "successor",
    }
    if not required_candidate.issubset(candidate):
        raise ValueError("candidate model columns missing")
    if not required_state.issubset(state):
        raise ValueError("candidate temporal columns missing")
    if not set(IMAGE_FEATURES).issubset(image):
        raise ValueError("candidate image columns missing")
    score = np.clip(candidate.score.to_numpy(np.float64), 1e-6, 1 - 1e-6)
    times = candidate.t.to_numpy(np.int64)
    frame_n = candidate.groupby("t", sort=False).t.transform("size").to_numpy(np.float64)
    frame_med = candidate.groupby("t", sort=False).score.transform("median").to_numpy(np.float64)
    frame_q90 = candidate.groupby("t", sort=False).score.transform(
        lambda x: x.quantile(.9)
    ).to_numpy(np.float64)
    residual = state.residual_um.to_numpy(np.float64)
    residual = np.where(np.isfinite(residual), np.minimum(residual, 20.0), 20.0)
    out = pd.DataFrame({
        "base_score": score.astype(np.float32),
        "base_logit": (np.log(score) - np.log1p(-score)).astype(np.float32),
        "rank_fraction": (
            candidate.rank_in_frame.to_numpy(np.float64) / np.maximum(frame_n - 1, 1)
        ).astype(np.float32),
        "frame_score_median": frame_med.astype(np.float32),
        "frame_score_q90": frame_q90.astype(np.float32),
        "near_um": candidate.near_um.to_numpy(np.float32),
        "density_10um": candidate.density_10um.to_numpy(np.float32),
        "peak_width_um": candidate.peak_width_um.to_numpy(np.float32),
        "boundary_um": candidate.boundary_um.to_numpy(np.float32),
        "flow_sigma_um": candidate.flow_sigma_um.to_numpy(np.float32),
        "log_span": np.log1p(state.span.to_numpy(np.float64)).astype(np.float32),
        "centered3": state.centered3.to_numpy(np.float32),
        "centered5": state.centered5.to_numpy(np.float32),
        "chain_residual_um": residual.astype(np.float32),
        "has_predecessor": (state.predecessor.to_numpy(np.int64) >= 0).astype(np.float32),
        "has_successor": (state.successor.to_numpy(np.int64) >= 0).astype(np.float32),
    })
    for col in IMAGE_FEATURES:
        out[col] = image[col].to_numpy(np.float32)
    if list(out.columns) != list(MODEL_FEATURES) or not np.isfinite(out.to_numpy()).all():
        raise RuntimeError("invalid specialist feature schema")
    if not np.array_equal(times, candidate.t.to_numpy(np.int64)):
        raise RuntimeError("candidate order changed")
    return out


def bounded_selection_score(candidate: pd.DataFrame, dim_probability: np.ndarray,
                            specialist_probability: np.ndarray,
                            max_shift: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Softly transfer each frame's original score ranks to specialist ranks.

    A candidate can move at most ``max_shift`` positions.  Its movement is
    multiplied by the inferred dim probability.  The final ordering receives
    the frame's original sorted score values, preserving the exact score
    histogram and scale used by the solver.
    """
    n = len(candidate)
    q = np.asarray(dim_probability, dtype=np.float64)
    p = np.asarray(specialist_probability, dtype=np.float64)
    if q.shape != (n,) or p.shape != (n,):
        raise ValueError("probability length differs from candidate rows")
    if not np.isfinite(q).all() or not np.isfinite(p).all() or \
       ((q < 0) | (q > 1)).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid reranker probability")
    if max_shift < 0:
        raise ValueError("max_shift must be nonnegative")
    t = candidate.t.to_numpy(np.int64)
    base_rank = candidate.rank_in_frame.to_numpy(np.int64)
    base_score = candidate.score.to_numpy(np.float64)
    selection = np.empty(n, dtype=np.float64)
    soft_rank = np.empty(n, dtype=np.float64)
    for frame in np.unique(t):
        rows = np.flatnonzero(t == frame)
        # Specialist ties resolve to the original order.
        spec_order = np.lexsort((base_rank[rows], -p[rows]))
        spec_rank = np.empty(len(rows), dtype=np.int64)
        spec_rank[spec_order] = np.arange(len(rows))
        delta = np.clip(spec_rank - base_rank[rows], -max_shift, max_shift)
        priority = base_rank[rows] + q[rows] * delta
        order = np.lexsort((base_rank[rows], priority))
        ranked_rows = rows[order]
        score_values = np.sort(base_score[rows], kind="stable")[::-1]
        selection[ranked_rows] = score_values
        soft_rank[rows] = priority
    return selection.astype(np.float32), soft_rank.astype(np.float32)
