"""Links to TrackGraph, plus the post-processing chain."""

from __future__ import annotations
import numpy as np
from cell_tracking.linker import postprocess as A
from cell_tracking.common.tracks import TrackGraph, close_gaps, prune_isolated


def graph_from_pair_links(B, frames: list[np.ndarray], pair_links: list[list[tuple[int, int]]]):
    """Build a ``TrackGraph`` from per-pair local index links."""
    node_ids: list[int] = []
    node_t: list[int] = []
    node_z: list[float] = []
    node_y: list[float] = []
    node_x: list[float] = []
    frame_ids: list[list[int]] = []
    nid = 1
    for t, coords in enumerate(frames):
        ids: list[int] = []
        for z, y, x in np.asarray(coords, float).reshape(-1, 3):
            node_ids.append(nid)
            node_t.append(t)
            node_z.append(float(z))
            node_y.append(float(y))
            node_x.append(float(x))
            ids.append(nid)
            nid += 1
        frame_ids.append(ids)
    edges: list[tuple[int, int]] = []
    for t, links in enumerate(pair_links):
        for i, j in links:
            edges.append((frame_ids[t][i], frame_ids[t + 1][j]))
    return TrackGraph(
        node_t=np.array(node_t, np.int64),
        node_z=np.array(node_z, float),
        node_y=np.array(node_y, float),
        node_x=np.array(node_x, float),
        node_ids=np.array(node_ids, np.int64),
        edges=np.array(edges, np.int64).reshape(-1, 2) if edges else np.zeros((0, 2), np.int64),
        meta={},
    )


def postprocess_graph(
    B,
    frames: list[np.ndarray],
    g,
    *,
    linefit: bool = False,
    max_gap: int = 1,
    gap_dist_um: float = 6.0,
    gap_drift_um: float = 0.0,
    min_track_len: int = 3,
    keep_division_components: bool = False,
    safe_divisions: bool = False,
    linefit_window: int = 2,
    linefit_weight: float = 0.8,
    linefit_stop_at_forks: bool = True,
):
    """``close_gaps → prune_isolated → filter_short(3) [→ linefit]``."""
    g = close_gaps(frames, g, max_gap=int(max_gap), gap_dist_um=float(gap_dist_um), drift_radius_um=float(gap_drift_um))
    g = prune_isolated(g)
    if safe_divisions:
        g = A.add_safe_divisions(g)
    g = A.filter_short_tracks(g, int(min_track_len), keep_division_components=bool(keep_division_components))
    if linefit:
        g = A.linefit_smooth(
            g, window=int(linefit_window), weight=float(linefit_weight), stop_at_forks=bool(linefit_stop_at_forks)
        )
    return g
