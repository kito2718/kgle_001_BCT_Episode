"""Typed state graph and deploy-safe graph-universe construction for V34.

The graph contains hypotheses, not selected cells.  Node and edge scores remain
soft until :func:`biohub.transition.decoder.decode_graph` solves the lineage as
one constrained problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from biohub.metrics.gt import SCALE_ARR


def _pairs(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(value, dtype=np.int64).reshape(-1, 2)


@dataclass(frozen=True)
class TransitionGraph:
    """A scored hypothesis graph ready for constrained lineage decoding.

    Scores are additive log-odds or calibrated utilities: positive values favour
    admitting the variable.  Coordinates are source-resolution ``(z, y, x)``
    voxels; physical distances must always apply ``SCALE_ARR``.
    """

    t: np.ndarray
    zyx: np.ndarray
    node_score: np.ndarray
    edges: np.ndarray
    edge_score: np.ndarray
    duplicate_pairs: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int64)
    )
    birth_score: np.ndarray | None = None
    death_score: np.ndarray | None = None
    division_score: np.ndarray | None = None
    target_counts: dict[int, float] | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=np.int64).reshape(-1)
        zyx = np.asarray(self.zyx, dtype=np.float64).reshape(-1, 3)
        node_score = np.asarray(self.node_score, dtype=np.float64).reshape(-1)
        edges = _pairs(self.edges)
        edge_score = np.asarray(self.edge_score, dtype=np.float64).reshape(-1)
        duplicate_pairs = _pairs(self.duplicate_pairs)
        n = len(t)

        if len(zyx) != n or len(node_score) != n:
            raise ValueError("t, zyx, and node_score must have the same length")
        if len(edge_score) != len(edges):
            raise ValueError("edge_score must have one value per edge")
        if not np.all(np.isfinite(zyx)) or not np.all(np.isfinite(node_score)):
            raise ValueError("node coordinates and scores must be finite")
        if not np.all(np.isfinite(edge_score)):
            raise ValueError("edge scores must be finite")

        for name, pairs in (("edges", edges), ("duplicate_pairs", duplicate_pairs)):
            if len(pairs) and (pairs.min() < 0 or pairs.max() >= n):
                raise ValueError(f"{name} contain an out-of-range node index")
            if len(pairs) and np.any(pairs[:, 0] == pairs[:, 1]):
                raise ValueError(f"{name} cannot contain self-pairs")
            if len(pairs) and len(np.unique(pairs, axis=0)) != len(pairs):
                raise ValueError(f"{name} must be unique")

        if len(edges) and np.any(t[edges[:, 1]] != t[edges[:, 0]] + 1):
            raise ValueError("every transition edge must point to the next frame")
        if len(duplicate_pairs) and np.any(
            t[duplicate_pairs[:, 0]] != t[duplicate_pairs[:, 1]]
        ):
            raise ValueError("duplicate exclusions must be within one frame")

        optional = {}
        for name in ("birth_score", "death_score", "division_score"):
            value = getattr(self, name)
            arr = np.zeros(n, dtype=np.float64) if value is None else np.asarray(
                value, dtype=np.float64
            ).reshape(-1)
            if len(arr) != n or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite with one value per node")
            optional[name] = arr

        targets = None
        if self.target_counts is not None:
            targets = {int(k): float(v) for k, v in self.target_counts.items()}
            if any(not np.isfinite(v) or v < 0 for v in targets.values()):
                raise ValueError("target counts must be finite and nonnegative")
            missing = set(targets) - set(map(int, np.unique(t)))
            if missing:
                raise ValueError(f"target counts reference absent frames: {sorted(missing)}")

        object.__setattr__(self, "t", t)
        object.__setattr__(self, "zyx", zyx)
        object.__setattr__(self, "node_score", node_score)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "edge_score", edge_score)
        object.__setattr__(self, "duplicate_pairs", duplicate_pairs)
        for name, arr in optional.items():
            object.__setattr__(self, name, arr)
        object.__setattr__(self, "target_counts", targets)


def build_duplicate_pairs(
    t: np.ndarray,
    zyx: np.ndarray,
    radius_um: float = 5.0,
) -> np.ndarray:
    """All same-frame hypothesis pairs closer than ``radius_um``.

    These pairs become mutual-exclusion constraints in the decoder.  They are
    constructed from deploy-time geometry only and contain no ground-truth
    information.
    """

    from scipy.spatial import cKDTree

    t = np.asarray(t, dtype=np.int64).reshape(-1)
    zyx = np.asarray(zyx, dtype=np.float64).reshape(-1, 3)
    if len(t) != len(zyx):
        raise ValueError("t and zyx must have the same length")
    if radius_um <= 0:
        raise ValueError("radius_um must be positive")

    parts: list[np.ndarray] = []
    for frame in np.unique(t):
        rows = np.flatnonzero(t == frame)
        if len(rows) < 2:
            continue
        local = np.asarray(list(cKDTree(zyx[rows] * SCALE_ARR).query_pairs(radius_um)))
        if not len(local):
            continue
        parts.append(rows[local.astype(np.int64)])
    if not parts:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(parts).astype(np.int64, copy=False)


def build_local_transition_edges(
    t: np.ndarray,
    zyx: np.ndarray,
    gate_um: float = 14.0,
    k: int = 12,
    backward_flow_um: np.ndarray | None = None,
) -> np.ndarray:
    """Build a sparse predecessor universe without making a hard choice.

    For every child, retain up to ``k`` parents in the preceding frame.  If a
    backward motion field is supplied it changes only the query centre; all
    neighbours remain hypotheses for the learned transition model and global
    decoder.  No winner-take-all or score/rank cut occurs here.
    """

    from scipy.spatial import cKDTree

    t = np.asarray(t, dtype=np.int64).reshape(-1)
    zyx = np.asarray(zyx, dtype=np.float64).reshape(-1, 3)
    n = len(t)
    if len(zyx) != n:
        raise ValueError("t and zyx must have the same length")
    if gate_um <= 0 or k <= 0:
        raise ValueError("gate_um and k must be positive")
    flow = None if backward_flow_um is None else np.asarray(
        backward_flow_um, dtype=np.float64
    ).reshape(-1, 3)
    if flow is not None and len(flow) != n:
        raise ValueError("backward_flow_um must have one vector per node")

    um = zyx * SCALE_ARR
    at = {int(frame): np.flatnonzero(t == frame) for frame in np.unique(t)}
    parts: list[np.ndarray] = []
    for frame, children in at.items():
        parents = at.get(frame - 1)
        if parents is None or not len(parents) or not len(children):
            continue
        centres = um[children] if flow is None else um[children] + flow[children]
        kk = min(int(k), len(parents))
        distance, neighbour = cKDTree(um[parents]).query(
            centres, k=kk, distance_upper_bound=gate_um
        )
        distance = np.asarray(distance).reshape(len(children), kk)
        neighbour = np.asarray(neighbour).reshape(len(children), kk)
        child_local, slot = np.nonzero(np.isfinite(distance))
        if not len(child_local):
            continue
        parts.append(np.column_stack((parents[neighbour[child_local, slot]],
                                      children[child_local])))
    if not parts:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.concatenate(parts).astype(np.int64, copy=False), axis=0)
