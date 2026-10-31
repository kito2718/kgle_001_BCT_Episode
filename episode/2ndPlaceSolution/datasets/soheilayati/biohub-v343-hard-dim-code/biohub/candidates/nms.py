"""Physical-space deduplication of the candidate pool.

The cache is extracted at one grid voxel of spacing (1.625 um), far below the
7 um matching radius, so one cell typically produces several candidates. Two
candidates closer than the matching radius compete for the same ground-truth
node: the scorer matches one-to-one, so the loser is not a second chance, it is
node-count cost. Suppressing them is what turns the pool into a node set.

Radius is in **micrometres**, always. On the source grid 7 um is 4.3 voxels in z
but 17.2 in y/x (CLAUDE.md §7).
"""

from __future__ import annotations

import numpy as np

from biohub.metrics.gt import SCALE_ARR


def nms(points_um: np.ndarray, scores: np.ndarray, radius_um: float) -> np.ndarray:
    """Greedy non-maximum suppression. Returns kept indices, strongest first.

    Neighbour lists are precomputed once and then walked in score order, which
    is O(n log n + n*k) rather than the O(n^2) of re-querying a growing kept set.
    """
    from scipy.spatial import cKDTree

    n = len(points_um)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if radius_um <= 0:
        return np.argsort(-scores, kind="stable")

    neighbours = cKDTree(points_um).query_ball_point(points_um, r=radius_um)
    order = np.argsort(-scores, kind="stable")

    suppressed = np.zeros(n, dtype=bool)
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed[neighbours[i]] = True
    return np.asarray(keep, dtype=np.int64)


def select_frame(
    zyx: np.ndarray,
    scores: np.ndarray,
    radius_um: float,
    top_k: int | None = None,
    min_score: float | None = None,
) -> np.ndarray:
    """One frame's node set: threshold, then suppress, then cap. Kept indices.

    Order matters. Capping before suppression would spend the budget on a
    cluster of duplicates around the brightest cell instead of on distinct cells.
    """
    idx = np.arange(len(zyx))
    if min_score is not None:
        idx = idx[scores[idx] >= min_score]
    if len(idx) == 0:
        return idx
    kept = idx[nms(zyx[idx] * SCALE_ARR, scores[idx], radius_um)]
    return kept[:top_k] if top_k is not None else kept


def select_dataset(
    df,
    radius_um: float,
    top_k: int | None = None,
    min_score: float | None = None,
):
    """Apply `select_frame` per frame to a cached candidate table."""
    import pandas as pd

    out = []
    zyx_all = df[["z", "y", "x"]].to_numpy(dtype=np.float64)
    score_all = df["score"].to_numpy(dtype=np.float64)
    t_all = df["t"].to_numpy()
    for t in np.unique(t_all):
        rows = np.flatnonzero(t_all == t)
        kept = select_frame(zyx_all[rows], score_all[rows], radius_um, top_k, min_score)
        out.append(rows[kept])
    idx = np.concatenate(out) if out else np.zeros(0, dtype=np.int64)
    return df.iloc[idx].reset_index(drop=True) if isinstance(df, pd.DataFrame) else idx
