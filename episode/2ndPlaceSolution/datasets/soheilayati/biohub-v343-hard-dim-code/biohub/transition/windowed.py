"""Windowed exact decoding and deterministic lineage stitching for V34B0.

The functions here are deliberately label-free. An incumbent graph may supply
soft per-frame count targets and protected division components, but no ground
truth or metric result is accepted by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from biohub.io.submission import DatasetGraph, drop_edgeless_nodes, validate_graph
from biohub.metrics.gt import SCALE_ARR
from biohub.transition.decoder import DecodeConfig, decode_ordinary_graph
from biohub.transition.graph import TransitionGraph, build_duplicate_pairs


@dataclass(frozen=True)
class WindowedDecodeConfig:
    window_frames: int = 10
    overlap_frames: int = 3
    duplicate_radius_um: float = 5.0
    protected_exclusion_radius_um: float = 5.0
    count_deviation_penalty: float = 20.0
    time_limit_seconds: float | None = 60.0
    mip_rel_gap: float = 0.001
    drop_edgeless: bool = True


@dataclass(frozen=True)
class WindowAudit:
    start: int
    stop: int
    candidates: int
    edges: int
    selected_nodes: int
    selected_edges: int
    objective: float
    optimal: bool
    status: int
    wall_seconds: float


def window_starts(
    total_frames: int,
    window_frames: int,
    overlap_frames: int,
) -> np.ndarray:
    """Return starts that cover every frame and always include the final frame."""

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if window_frames <= 1 or window_frames > total_frames:
        raise ValueError("window_frames must be in [2, total_frames]")
    if overlap_frames <= 0 or overlap_frames >= window_frames:
        raise ValueError("overlap_frames must be in [1, window_frames)")
    last = total_frames - window_frames
    starts = list(range(0, last + 1, window_frames - overlap_frames))
    if starts[-1] != last:
        starts.append(last)
    return np.asarray(starts, dtype=np.int64)


def owner_windows(
    total_frames: int,
    starts: np.ndarray,
    window_frames: int,
) -> np.ndarray:
    """Assign each frame to its nearest window centre, breaking ties earlier."""

    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    centres = starts.astype(np.float64) + (window_frames - 1) / 2.0
    frames = np.arange(total_frames, dtype=np.float64)
    valid = (
        (frames[:, None] >= starts[None, :])
        & (frames[:, None] < (starts + window_frames)[None, :])
    )
    distance = np.abs(frames[:, None] - centres[None, :])
    distance[~valid] = np.inf
    if np.any(~np.isfinite(distance).any(axis=1)):
        raise RuntimeError("decode windows do not cover every frame")
    return np.argmin(distance, axis=1).astype(np.int64)


def division_component_nodes(graph: DatasetGraph) -> set[int]:
    """Every node in a weak component containing an incumbent division."""

    node_ids = np.asarray(sorted(graph.nodes), dtype=np.int64)
    if not len(node_ids) or not graph.edges:
        return set()
    row_for_id = {int(node): row for row, node in enumerate(node_ids)}
    edge_rows = np.asarray(
        [
            (row_for_id[int(source)], row_for_id[int(target)])
            for source, target, _ in graph.edges
        ],
        dtype=np.int64,
    )
    outdegree = np.bincount(edge_rows[:, 0], minlength=len(node_ids))
    division_rows = np.flatnonzero(outdegree >= 2)
    if not len(division_rows):
        return set()

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    undirected = coo_matrix(
        (
            np.ones(2 * len(edge_rows), dtype=np.int8),
            (
                np.r_[edge_rows[:, 0], edge_rows[:, 1]],
                np.r_[edge_rows[:, 1], edge_rows[:, 0]],
            ),
        ),
        shape=(len(node_ids), len(node_ids)),
    )
    _, labels = connected_components(undirected, directed=False)
    protected_labels = np.unique(labels[division_rows])
    return set(map(int, node_ids[np.isin(labels, protected_labels)]))


def _candidate_keep_mask(
    t: np.ndarray,
    zyx: np.ndarray,
    incumbent: DatasetGraph,
    protected: set[int],
    radius_um: float,
) -> np.ndarray:
    """Remove raw hypotheses that could duplicate an exact protected node."""

    from scipy.spatial import cKDTree

    keep = np.ones(len(t), dtype=bool)
    if not protected:
        return keep
    protected_rows = np.asarray(
        [incumbent.nodes[node] for node in sorted(protected)], dtype=np.float64
    )
    protected_t = protected_rows[:, 0].astype(np.int64)
    protected_um = protected_rows[:, 1:] * SCALE_ARR
    candidate_um = zyx * SCALE_ARR
    for frame in np.unique(protected_t):
        candidate_rows = np.flatnonzero(t == frame)
        protected_frame = np.flatnonzero(protected_t == frame)
        if not len(candidate_rows):
            continue
        near = cKDTree(protected_um[protected_frame]).query_ball_point(
            candidate_um[candidate_rows], r=radius_um, return_length=True
        )
        keep[candidate_rows[np.asarray(near) > 0]] = False
    return keep


def _incumbent_frame_counts(
    incumbent: DatasetGraph,
    protected: set[int],
    total_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_counts = np.zeros(total_frames, dtype=np.int64)
    protected_counts = np.zeros(total_frames, dtype=np.int64)
    for node, coordinate in incumbent.nodes.items():
        frame = int(coordinate[0])
        all_counts[frame] += 1
        if node in protected:
            protected_counts[frame] += 1
    ordinary = all_counts - protected_counts
    if np.any(ordinary < 0):
        raise RuntimeError("protected counts exceed incumbent counts")
    return ordinary, protected_counts


def decode_windowed(
    dataset: str,
    t: np.ndarray,
    zyx: np.ndarray,
    node_score: np.ndarray,
    edges: np.ndarray,
    edge_score: np.ndarray,
    null_score: float | None,
    incumbent: DatasetGraph,
    cfg: WindowedDecodeConfig | None = None,
    birth_score: np.ndarray | None = None,
    death_score: np.ndarray | None = None,
) -> tuple[DatasetGraph, dict]:
    """Decode ordinary tracks in windows, then overlay exact division components.

    A scalar ``null_score`` preserves the original B0 behavior. Objective-aligned
    callers instead pass ``null_score=None`` plus per-node birth and death scores.
    Internal window boundaries then marginalize the unavailable outside event.
    """

    cfg = cfg or WindowedDecodeConfig()
    t = np.asarray(t, dtype=np.int64).reshape(-1)
    zyx = np.asarray(zyx, dtype=np.float64).reshape(-1, 3)
    node_score = np.asarray(node_score, dtype=np.float64).reshape(-1)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edge_score = np.asarray(edge_score, dtype=np.float64).reshape(-1)
    if not (len(t) == len(zyx) == len(node_score)):
        raise ValueError("candidate node arrays have inconsistent lengths")
    if len(edges) != len(edge_score):
        raise ValueError("edge arrays have inconsistent lengths")
    if birth_score is None:
        if null_score is None or not np.isfinite(null_score):
            raise ValueError("a finite null score or per-node birth scores are required")
        birth_score_values = np.full(len(t), float(null_score), dtype=np.float64)
        event_score_mode = "scalar_null"
    else:
        if null_score is not None:
            raise ValueError("null_score and per-node birth_score are mutually exclusive")
        birth_score_values = np.asarray(birth_score, dtype=np.float64).reshape(-1)
        if len(birth_score_values) != len(t):
            raise ValueError("birth-score and candidate-node lengths differ")
        event_score_mode = "per_node_objective_aligned"
    if death_score is None:
        death_score_values = np.zeros(len(t), dtype=np.float64)
    else:
        death_score_values = np.asarray(death_score, dtype=np.float64).reshape(-1)
        if len(death_score_values) != len(t):
            raise ValueError("death-score and candidate-node lengths differ")
    if not (
        np.isfinite(node_score).all()
        and np.isfinite(edge_score).all()
        and np.isfinite(birth_score_values).all()
        and np.isfinite(death_score_values).all()
    ):
        raise ValueError("decoder scores must all be finite")
    if incumbent.dataset != dataset:
        raise ValueError("incumbent dataset does not match candidate dataset")
    total_frames = int(
        max(t.max(initial=0) + 1, incumbent.shape[0] if incumbent.shape else 0)
    )
    starts = window_starts(total_frames, cfg.window_frames, cfg.overlap_frames)
    owners = owner_windows(total_frames, starts, cfg.window_frames)

    protected = division_component_nodes(incumbent)
    ordinary_keep = _candidate_keep_mask(
        t, zyx, incumbent, protected, cfg.protected_exclusion_radius_um
    )
    ordinary_target, protected_counts = _incumbent_frame_counts(
        incumbent, protected, total_frames
    )
    if np.any(
        [
            ordinary_keep[t == frame].sum() < ordinary_target[frame]
            for frame in range(total_frames)
        ]
    ):
        raise RuntimeError(f"{dataset}: too few ordinary hypotheses for a count target")

    selected_by_window: list[np.ndarray] = []
    edge_rows_by_window: list[np.ndarray] = []
    window_audits: list[WindowAudit] = []
    local_index = np.full(len(t), -1, dtype=np.int64)
    for start in starts:
        stop = int(start + cfg.window_frames)
        global_nodes = np.flatnonzero(
            ordinary_keep & (t >= int(start)) & (t < stop)
        )
        local_index[global_nodes] = np.arange(len(global_nodes), dtype=np.int64)
        edge_mask = (
            ordinary_keep[edges[:, 0]]
            & ordinary_keep[edges[:, 1]]
            & (t[edges[:, 0]] >= int(start))
            & (t[edges[:, 1]] < stop)
        )
        global_edge_rows = np.flatnonzero(edge_mask)
        local_edges = local_index[edges[global_edge_rows]]
        local_birth_score = birth_score_values[global_nodes].copy()
        local_death_score = death_score_values[global_nodes].copy()
        if event_score_mode == "per_node_objective_aligned":
            if int(start) > 0:
                local_birth_score[t[global_nodes] == int(start)] = 0.0
            if stop < total_frames:
                local_death_score[t[global_nodes] == stop - 1] = 0.0
        graph = TransitionGraph(
            t=t[global_nodes],
            zyx=zyx[global_nodes],
            node_score=node_score[global_nodes],
            edges=local_edges,
            edge_score=edge_score[global_edge_rows],
            duplicate_pairs=build_duplicate_pairs(
                t[global_nodes], zyx[global_nodes], cfg.duplicate_radius_um
            ),
            birth_score=local_birth_score,
            death_score=local_death_score,
            division_score=np.zeros(len(global_nodes), dtype=np.float64),
            target_counts={
                frame: float(ordinary_target[frame])
                for frame in range(int(start), stop)
            },
        )
        decode_cfg = DecodeConfig(
            birth_penalty=0.0,
            death_penalty=0.0,
            division_penalty=0.0,
            count_deviation_penalty=cfg.count_deviation_penalty,
            time_limit_seconds=cfg.time_limit_seconds,
            mip_rel_gap=cfg.mip_rel_gap,
            allow_divisions=False,
        )
        started = time.perf_counter()
        result = decode_ordinary_graph(graph, decode_cfg)
        elapsed = time.perf_counter() - started
        selected_by_window.append(global_nodes[result.selected_nodes])
        edge_rows_by_window.append(global_edge_rows[result.selected_edge_rows])
        window_audits.append(
            WindowAudit(
                start=int(start),
                stop=stop,
                candidates=int(len(global_nodes)),
                edges=int(len(local_edges)),
                selected_nodes=int(len(result.selected_nodes)),
                selected_edges=int(len(result.selected_edge_rows)),
                objective=float(result.objective),
                optimal=bool(result.optimal),
                status=int(result.status),
                wall_seconds=float(elapsed),
            )
        )
        local_index[global_nodes] = -1

    selected_parts = []
    for window_index, selected in enumerate(selected_by_window):
        selected_parts.append(selected[owners[t[selected]] == window_index])
    selected_nodes = np.unique(np.concatenate(selected_parts)).astype(np.int64)
    selected_mask = np.zeros(len(t), dtype=bool)
    selected_mask[selected_nodes] = True

    proposed_edge_rows = np.unique(np.concatenate(edge_rows_by_window)).astype(np.int64)
    proposed_edge_rows = proposed_edge_rows[
        selected_mask[edges[proposed_edge_rows, 0]]
        & selected_mask[edges[proposed_edge_rows, 1]]
    ]
    stitch_utility = (
        edge_score[proposed_edge_rows]
        - birth_score_values[edges[proposed_edge_rows, 1]]
        - death_score_values[edges[proposed_edge_rows, 0]]
    )
    order = np.lexsort((proposed_edge_rows, -stitch_utility))
    incoming: set[int] = set()
    outgoing: set[int] = set()
    stitched_edge_rows: list[int] = []
    for row in proposed_edge_rows[order]:
        source, target = map(int, edges[int(row)])
        if source in outgoing or target in incoming:
            continue
        outgoing.add(source)
        incoming.add(target)
        stitched_edge_rows.append(int(row))

    nodes = {node: incumbent.nodes[node] for node in sorted(protected)}
    next_id = max(incumbent.nodes, default=-1) + 1
    candidate_to_node: dict[int, int] = {}
    shape = incumbent.shape or (total_frames, 64, 256, 256)
    spatial_hi = np.asarray(shape[1:], dtype=np.int64) - 1
    for candidate in selected_nodes:
        coordinate = np.clip(np.rint(zyx[candidate]).astype(np.int64), 0, spatial_hi)
        node = next_id
        next_id += 1
        candidate_to_node[int(candidate)] = node
        nodes[node] = (int(t[candidate]), *map(int, coordinate))

    output_edges = [
        (int(source), int(target), float(confidence))
        for source, target, confidence in incumbent.edges
        if source in protected and target in protected
    ]
    output_edges.extend(
        (
            candidate_to_node[int(edges[row, 0])],
            candidate_to_node[int(edges[row, 1])],
            float(edge_score[row]),
        )
        for row in stitched_edge_rows
    )
    graph = DatasetGraph(dataset=dataset, nodes=nodes, edges=output_edges, shape=shape)
    before_drop = len(graph.nodes)
    if cfg.drop_edgeless:
        graph = drop_edgeless_nodes(graph)
    summary = validate_graph(graph)
    output_divisions = summary["forks"]
    incumbent_divisions = validate_graph(incumbent)["forks"]
    if output_divisions != incumbent_divisions:
        raise RuntimeError(
            f"{dataset}: protected division overlay changed fork count "
            f"{incumbent_divisions}->{output_divisions}"
        )

    audit = {
        "dataset": dataset,
        "event_score_mode": event_score_mode,
        "candidate_nodes": int(len(t)),
        "ordinary_candidate_nodes": int(ordinary_keep.sum()),
        "excluded_near_protected": int((~ordinary_keep).sum()),
        "protected_nodes": int(len(protected)),
        "protected_edges": int(
            sum(
                source in protected and target in protected
                for source, target, _ in incumbent.edges
            )
        ),
        "protected_divisions": int(incumbent_divisions),
        "selected_ordinary_nodes_before_edgeless_drop": int(len(selected_nodes)),
        "selected_ordinary_edges": int(len(stitched_edge_rows)),
        "edgeless_nodes_dropped": int(before_drop - len(graph.nodes)),
        "output_nodes": int(len(graph.nodes)),
        "output_edges": int(len(graph.edges)),
        "output_divisions": int(output_divisions),
        "incumbent_nodes": int(len(incumbent.nodes)),
        "incumbent_edges": int(len(incumbent.edges)),
        "ordinary_target_nodes": int(ordinary_target.sum()),
        "protected_target_nodes": int(protected_counts.sum()),
        "all_windows_optimal": bool(all(row.optimal for row in window_audits)),
        "decoder_wall_seconds": float(sum(row.wall_seconds for row in window_audits)),
        "windows": [asdict(row) for row in window_audits],
    }
    return graph, audit
