"""Label-free deployment helpers for the V55 residual bundle.

This module contains only prediction-time operations.  It deliberately has
no dependency on folds, annotations, ground truth, or the official metric.
The detector/ranker route is supplied by the caller so that local OOF replay
and hidden inference can share the exact same graph materializer.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from biohub.io.submission import DatasetGraph, validate_graph
from biohub.metrics.gt import SCALE_ARR
from biohub.metrics.matching import match_sequence
from biohub.pipeline.additive_segments import MAP_UM, _arrays


SCALAR_SUFFIXES = (
    "distance_um",
    "rank_fraction",
    "score",
    "flow_sigma_um",
    "near_um",
    "density_10um",
    "peak_width_um",
    "boundary_um",
)
CANDIDATE_FIELDS = (
    "score",
    "flow_sigma_um",
    "near_um",
    "density_10um",
    "peak_width_um",
    "boundary_um",
)
V55_FEATURES = (
    "origin_v21",
    "origin_v24",
    "origin_stage6",
    "v21_supported",
    "v24_supported",
    "stage6_supported",
    "v39_supported",
    "supporting_banks",
    "v21_distance_um",
    "v24_distance_um",
    "stage6_distance_um",
    "v39_distance_um",
    "indegree",
    "outdegree",
    "incident_degree",
    "history",
    "future",
    "track_support",
    "track_span",
    "incoming_step_um",
    "outgoing_step_um",
    *(
        f"{bank}_candidate_{suffix}"
        for bank in ("v21", "v24", "stage6", "v39")
        for suffix in SCALAR_SUFFIXES
    ),
    *(f"origin_candidate_{suffix}" for suffix in SCALAR_SUFFIXES),
)


def deployment_route(dataset: str, routes: int = 5) -> int:
    """Return a stable, name-only hidden route in ``[0, routes)``."""
    if routes <= 0:
        raise ValueError("routes must be positive")
    digest = hashlib.blake2b(dataset.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % routes


def graph_arrays(
    graph: DatasetGraph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(sorted(graph.nodes), dtype=np.int64)
    times = np.asarray(
        [graph.nodes[int(node)][0] for node in ids], dtype=np.int64
    )
    positions = np.asarray(
        [graph.nodes[int(node)][1:] for node in ids], dtype=np.float64
    )
    return ids, times, positions


def support_and_distance(
    base: DatasetGraph,
    bank: DatasetGraph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map base nodes to one witness and return support/distance/id arrays."""
    base_ids, base_t, base_zyx = graph_arrays(base)
    bank_ids, bank_t, bank_zyx = graph_arrays(bank)
    match = match_sequence(bank_t, bank_zyx, base_t, base_zyx, MAP_UM)
    supported = match >= 0
    matched_id = np.full(len(base_ids), -1, dtype=np.int64)
    matched_id[supported] = bank_ids[match[supported]]
    nearest = np.full(len(base_ids), 25.0, dtype=np.float32)
    for frame in np.unique(base_t):
        base_pos = np.flatnonzero(base_t == frame)
        bank_pos = np.flatnonzero(bank_t == frame)
        if len(bank_pos):
            nearest[base_pos] = np.minimum(
                cKDTree(bank_zyx[bank_pos] * SCALE_ARR).query(
                    base_zyx[base_pos] * SCALE_ARR, k=1
                )[0],
                25.0,
            ).astype(np.float32)
    return supported, nearest, matched_id


def action_anchors(actions: Iterable[dict[str, Any]]) -> set[int]:
    return {
        int(value)
        for action in actions
        for value in (action["mapped_start"], action["mapped_end"])
        if int(value) >= 0
    }


def action_quotas(
    witness: DatasetGraph,
    actions: Iterable[dict[str, Any]],
) -> Counter[int]:
    quotas: Counter[int] = Counter()
    for action in actions:
        path = [
            int(value)
            for value in json.loads(str(action["path_v24_nodes"]))
        ]
        quotas.update(int(witness.nodes[node][0]) for node in path)
    return quotas


def map_candidate_scalars(
    base: DatasetGraph,
    nodes: np.ndarray,
    times: np.ndarray,
    candidates: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Nearest same-frame detector scalars, matching the frozen V50 mapper."""
    result = {
        "distance_um": np.full(len(nodes), 25.0, dtype=np.float32),
        "rank_fraction": np.ones(len(nodes), dtype=np.float32),
    }
    for field in CANDIDATE_FIELDS:
        result[field] = np.full(len(nodes), np.nan, dtype=np.float32)
    required = {"t", "z", "y", "x", "rank_in_frame", *CANDIDATE_FIELDS}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"candidate table is missing fields: {missing}")
    for frame in np.unique(times):
        positions = np.flatnonzero(times == frame)
        part = candidates[candidates.t == frame]
        if not len(part):
            continue
        candidate_zyx = part[["z", "y", "x"]].to_numpy(np.float64)
        node_zyx = np.asarray(
            [base.nodes[int(node)][1:] for node in nodes[positions]],
            dtype=np.float64,
        )
        distance, nearest = cKDTree(candidate_zyx * SCALE_ARR).query(
            node_zyx * SCALE_ARR, k=1
        )
        nearest_frame = part.iloc[nearest]
        result["distance_um"][positions] = np.minimum(
            distance, 25.0
        ).astype(np.float32)
        result["rank_fraction"][positions] = (
            nearest_frame.rank_in_frame.to_numpy(np.float32)
            / max(1.0, float(len(part)))
        )
        for field in CANDIDATE_FIELDS:
            result[field][positions] = nearest_frame[field].to_numpy(
                np.float32
            )
    return result


def build_removal_features(
    base: DatasetGraph,
    phase19: DatasetGraph,
    v21: DatasetGraph,
    v24: DatasetGraph,
    stage6: DatasetGraph,
    v39: DatasetGraph,
    actions: list[dict[str, Any]],
    candidates: dict[str, pd.DataFrame],
    *,
    action_witness: DatasetGraph | None = None,
) -> pd.DataFrame:
    """Construct the exact frozen V51/V55 prediction-time feature matrix.

    ``action_witness`` may supply the path/time authority for a later action
    family while retaining V39 as the frozen support-feature bank. Omitting it
    preserves the byte-existing V51/V55 path.
    """
    if set(candidates) != {"v21", "v24", "stage6", "v39"}:
        raise ValueError("candidate banks must be v21/v24/stage6/v39")
    quotas = action_quotas(
        v39 if action_witness is None else action_witness, actions
    )
    anchors = action_anchors(actions)
    base_ids, base_t, _ = graph_arrays(base)
    support: dict[str, np.ndarray] = {}
    distance: dict[str, np.ndarray] = {}
    for label, bank in (
        ("v21", v21),
        ("v24", v24),
        ("stage6", stage6),
        ("v39", v39),
    ):
        support[label], distance[label], _ = support_and_distance(base, bank)

    topology = _arrays(base)
    indegree = topology["indegree"][base_ids].astype(np.int8)
    outdegree = topology["outdegree"][base_ids].astype(np.int8)
    incident = indegree + outdegree
    quota_frames = np.asarray(sorted(quotas), dtype=np.int64)
    relevant = np.isin(base_t, quota_frames)
    if anchors:
        relevant &= ~np.isin(
            base_ids, np.asarray(sorted(anchors), dtype=np.int64)
        )
    relevant &= outdegree <= 1
    relevant &= incident <= 2
    positions = np.flatnonzero(relevant)
    ids = base_ids[positions]

    phase19_nodes = set(phase19.nodes)
    v21_nodes = set(v21.nodes)
    origin = np.fromiter(
        (
            0
            if int(node) in v21_nodes
            else 1
            if int(node) in phase19_nodes
            else 2
            for node in ids
        ),
        dtype=np.int8,
        count=len(ids),
    )
    parent = topology["parent"][ids]
    child = topology["child"][ids]
    incoming_step = np.full(len(ids), np.nan, dtype=np.float32)
    outgoing_step = np.full(len(ids), np.nan, dtype=np.float32)
    has_parent = parent >= 0
    has_child = child >= 0
    if has_parent.any():
        incoming_step[has_parent] = np.linalg.norm(
            (
                topology["pos"][ids[has_parent]]
                - topology["pos"][parent[has_parent]]
            )
            * SCALE_ARR,
            axis=1,
        ).astype(np.float32)
    if has_child.any():
        outgoing_step[has_child] = np.linalg.norm(
            (
                topology["pos"][child[has_child]]
                - topology["pos"][ids[has_child]]
            )
            * SCALE_ARR,
            axis=1,
        ).astype(np.float32)

    frame = pd.DataFrame(
        {
            "dataset": base.dataset,
            "node_id": ids,
            "t": base_t[positions].astype(np.int16),
            "frame_quota": np.asarray(
                [quotas[int(t)] for t in base_t[positions]], dtype=np.int16
            ),
            "origin_v21": origin == 0,
            "origin_v24": origin == 1,
            "origin_stage6": origin == 2,
            "v21_supported": support["v21"][positions],
            "v24_supported": support["v24"][positions],
            "stage6_supported": support["stage6"][positions],
            "v39_supported": support["v39"][positions],
            "supporting_banks": (
                support["v21"][positions].astype(np.int8)
                + support["v24"][positions].astype(np.int8)
                + support["stage6"][positions].astype(np.int8)
                + support["v39"][positions].astype(np.int8)
            ),
            "v21_distance_um": distance["v21"][positions],
            "v24_distance_um": distance["v24"][positions],
            "stage6_distance_um": distance["stage6"][positions],
            "v39_distance_um": distance["v39"][positions],
            "indegree": indegree[positions],
            "outdegree": outdegree[positions],
            "incident_degree": incident[positions],
            "history": topology["history"][ids],
            "future": topology["future"][ids],
            "track_support": np.minimum(
                topology["history"][ids], topology["future"][ids]
            ),
            "track_span": (
                topology["history"][ids] + topology["future"][ids] - 1
            ),
            "incoming_step_um": incoming_step,
            "outgoing_step_um": outgoing_step,
        }
    )

    mapped: dict[str, dict[str, np.ndarray]] = {}
    for bank in ("v21", "v24", "stage6", "v39"):
        mapped[bank] = map_candidate_scalars(
            base, ids, frame.t.to_numpy(np.int64), candidates[bank]
        )
        for field, values in mapped[bank].items():
            frame[f"{bank}_candidate_{field}"] = values

    origin_bank = np.where(
        frame.origin_v21.to_numpy(bool),
        "v21",
        np.where(frame.origin_v24.to_numpy(bool), "v24", "stage6"),
    )
    for field in SCALAR_SUFFIXES:
        values = np.empty(len(frame), dtype=np.float32)
        for bank in ("v21", "v24", "stage6"):
            mask = origin_bank == bank
            values[mask] = mapped[bank][field][mask]
        frame[f"origin_candidate_{field}"] = values

    missing_features = sorted(set(V55_FEATURES) - set(frame.columns))
    if missing_features:
        raise RuntimeError(f"V55 feature construction drift: {missing_features}")
    frame.sort_values(["t", "node_id"], inplace=True)
    return frame


def select_removals(
    frame: pd.DataFrame,
    probability: np.ndarray,
) -> pd.DataFrame:
    """Frozen degree-one-first, same-frame quota selection."""
    held = frame.copy()
    held["p_risk"] = np.asarray(probability, dtype=np.float64)
    if len(held) != len(probability):
        raise ValueError("probability length differs from feature rows")
    parts: list[pd.DataFrame] = []
    for (_, _), group in held.groupby(["dataset", "t"], sort=True):
        quotas = group.frame_quota.unique()
        if len(quotas) != 1:
            raise RuntimeError("frame quota is not constant")
        quota = int(quotas[0])
        ordered = group.sort_values(
            [
                "incident_degree",
                "p_risk",
                "supporting_banks",
                "track_support",
                "node_id",
            ],
            ascending=[True, True, True, True, True],
            kind="stable",
        )
        if len(ordered) < quota:
            raise RuntimeError("V55 removal quota shortfall")
        parts.append(ordered.head(quota))
    if not parts:
        return held.iloc[:0].copy()
    result = pd.concat(parts, ignore_index=True)
    result.sort_values(["dataset", "t", "node_id"], inplace=True)
    return result


def predict_bagged_risk(
    frame: pd.DataFrame,
    boosters: Iterable[Any],
) -> np.ndarray:
    """Mean positive-class probability from serialized LightGBM boosters."""
    x = frame[list(V55_FEATURES)]
    probability = np.zeros(len(frame), dtype=np.float64)
    count = 0
    for booster in boosters:
        # A movie can have no complementary segments and therefore no removal
        # candidates. LightGBM rejects zero-row tables; the exchange is a no-op.
        if len(frame):
            probability += np.asarray(booster.predict(x), dtype=np.float64)
        count += 1
    if count == 0:
        raise ValueError("no V55 boosters supplied")
    return probability / count


def apply_exchange(
    base: DatasetGraph,
    witness: DatasetGraph,
    actions: list[dict[str, Any]],
    removed: set[int],
) -> DatasetGraph:
    """Apply V39 additions funded by same-frame V37c removals."""
    if removed & action_anchors(actions):
        raise RuntimeError(f"{base.dataset}: removal deleted a V39 anchor")
    nodes = {
        int(node): coordinate
        for node, coordinate in base.nodes.items()
        if int(node) not in removed
    }
    edges = [
        edge
        for edge in base.edges
        if int(edge[0]) not in removed and int(edge[1]) not in removed
    ]
    next_id = max(base.nodes, default=-1) + 1
    for action in actions:
        path = [
            int(value)
            for value in json.loads(str(action["path_v24_nodes"]))
        ]
        new_ids = list(range(next_id, next_id + len(path)))
        next_id += len(path)
        for new_id, witness_id in zip(new_ids, path):
            nodes[new_id] = tuple(int(value) for value in witness.nodes[witness_id])
        if int(action["mapped_start"]) >= 0:
            edges.append((int(action["mapped_start"]), new_ids[0], 1.0))
        edges.extend(
            (left, right, 1.0)
            for left, right in zip(new_ids[:-1], new_ids[1:])
        )
        if int(action["mapped_end"]) >= 0:
            edges.append((new_ids[-1], int(action["mapped_end"]), 1.0))
    candidate = DatasetGraph(
        dataset=base.dataset,
        nodes=nodes,
        edges=edges,
        shape=base.shape,
    )
    validate_graph(candidate)
    if len(candidate.nodes) != len(base.nodes):
        raise RuntimeError(f"{base.dataset}: candidate is not count-neutral")
    return candidate


def _mapped_ordinary_edges(
    base: DatasetGraph,
    witness: DatasetGraph,
) -> set[tuple[int, int]]:
    base_ids, base_t, base_pos = graph_arrays(base)
    witness_ids, witness_t, witness_pos = graph_arrays(witness)
    mapped = np.full(int(witness_ids.max(initial=-1)) + 1, -1, np.int64)
    match = match_sequence(base_t, base_pos, witness_t, witness_pos, MAP_UM)
    valid = match >= 0
    mapped[witness_ids[valid]] = base_ids[match[valid]]
    size = len(mapped)
    source = np.fromiter(
        (int(edge[0]) for edge in witness.edges),
        np.int64,
        count=len(witness.edges),
    )
    target = np.fromiter(
        (int(edge[1]) for edge in witness.edges),
        np.int64,
        count=len(witness.edges),
    )
    outdegree = np.bincount(source, minlength=size)
    indegree = np.bincount(target, minlength=size)
    result: set[tuple[int, int]] = set()
    for left, right in zip(source, target):
        if outdegree[left] != 1 or indegree[right] != 1:
            continue
        mapped_left, mapped_right = int(mapped[left]), int(mapped[right])
        if mapped_left < 0 or mapped_right < 0 or mapped_left == mapped_right:
            continue
        if base.nodes[mapped_right][0] != base.nodes[mapped_left][0] + 1:
            continue
        result.add((mapped_left, mapped_right))
    return result


def add_consensus_edges(
    base: DatasetGraph,
    witnesses: dict[str, DatasetGraph],
    *,
    output_base: DatasetGraph | None = None,
) -> tuple[DatasetGraph, list[dict[str, Any]], dict[str, Any]]:
    """Frozen V37 multi-family edge repair, optionally on output coords."""
    allowed = (
        {"stage6", "v24", "v25", "expanded"},
        {"stage6", "v24", "expanded"},
    )
    if set(witnesses) not in allowed:
        raise ValueError("multiwitness banks are incomplete")
    output_base = base if output_base is None else output_base
    if (
        set(output_base.nodes) != set(base.nodes)
        or output_base.edges != base.edges
        or output_base.dataset != base.dataset
    ):
        raise ValueError("output base changed the raw graph topology")
    size = max(base.nodes, default=-1) + 1
    source = np.fromiter(
        (int(edge[0]) for edge in base.edges), np.int64, count=len(base.edges)
    )
    target = np.fromiter(
        (int(edge[1]) for edge in base.edges), np.int64, count=len(base.edges)
    )
    outdegree = np.bincount(source, minlength=size)
    indegree = np.bincount(target, minlength=size)
    original = {(int(edge[0]), int(edge[1])) for edge in base.edges}
    bank_edges = {
        name: _mapped_ordinary_edges(base, graph)
        for name, graph in witnesses.items()
    }
    family_edges = {
        "strong_detector": bank_edges["stage6"],
        "detector_bank": bank_edges["v24"] | bank_edges.get("v25", set()),
        "crossbank_solver": bank_edges["expanded"],
    }
    support: dict[tuple[int, int], set[str]] = defaultdict(set)
    for family, edges in family_edges.items():
        for edge in edges:
            support[edge].add(family)
    qualifying: dict[tuple[int, int], set[str]] = {}
    for edge, families in support.items():
        left, right = edge
        if len(families) < 2 or edge in original:
            continue
        if outdegree[left] != 0 or indegree[right] != 0:
            continue
        if base.nodes[right][0] != base.nodes[left][0] + 1:
            continue
        qualifying[edge] = families
    by_source: dict[int, list[tuple[int, int]]] = defaultdict(list)
    by_target: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge in qualifying:
        by_source[edge[0]].append(edge)
        by_target[edge[1]].append(edge)
    selected = sorted(
        edge
        for edge in qualifying
        if len(by_source[edge[0]]) == 1 and len(by_target[edge[1]]) == 1
    )
    candidate = DatasetGraph(
        dataset=output_base.dataset,
        nodes=dict(output_base.nodes),
        edges=list(output_base.edges)
        + [(left, right, 1.0) for left, right in selected],
        shape=output_base.shape,
    )
    before, after = validate_graph(output_base), validate_graph(candidate)
    if before["forks"] != after["forks"]:
        raise RuntimeError(f"{base.dataset}: multiwitness repair changed forks")
    actions = [
        {
            "dataset": base.dataset,
            "source": left,
            "target": right,
            "families": "+".join(sorted(qualifying[(left, right)])),
        }
        for left, right in selected
    ]
    census = {
        "dataset": base.dataset,
        "mapped_stage6_edges": len(bank_edges["stage6"]),
        "mapped_v24_edges": len(bank_edges["v24"]),
        "mapped_v25_edges": len(bank_edges.get("v25", set())),
        "mapped_expanded_edges": len(bank_edges["expanded"]),
        "selected_mutual_unique": len(selected),
    }
    return candidate, actions, census


def make_expanded_superset(
    v21: pd.DataFrame,
    v24: pd.DataFrame,
    budget: int,
    *,
    map_um: float = 3.0,
    nms_um: float = 5.0,
    multiplier: int = 2,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the registered V29 cross-bank pre-WTA candidate union."""
    from biohub.tracker.edges import select_nodes

    v21_rows = select_nodes(v21, max(1, multiplier * budget), nms_um=nms_um)
    v24_rows = select_nodes(v24, max(1, multiplier * budget), nms_um=nms_um)
    v21_sup = v21.iloc[v21_rows].reset_index(drop=True)
    v24_sup = v24.iloc[v24_rows].reset_index(drop=True)
    mapped = match_sequence(
        v21_sup.t.to_numpy(np.int64),
        v21_sup[["z", "y", "x"]].to_numpy(np.float64),
        v24_sup.t.to_numpy(np.int64),
        v24_sup[["z", "y", "x"]].to_numpy(np.float64),
        map_um,
    )
    append = mapped < 0
    union = pd.concat([v21_sup, v24_sup.loc[append]], ignore_index=True)
    if list(union.columns) != list(v21.columns):
        raise RuntimeError("cross-bank candidate schema drift")
    return union, {
        "v21_superset": len(v21_sup),
        "v24_superset": len(v24_sup),
        "v24_mapped": int((mapped >= 0).sum()),
        "v24_appended": int(append.sum()),
        "union_superset": len(union),
    }


def bridges_for(
    nodes: dict[int, tuple[int, int, int, int]],
    edges: list[tuple[int, int, float]],
    pool: pd.DataFrame,
    *,
    gate_um: float = 7.0,
    place_um: float = 5.0,
    max_terminal_t: int = 97,
) -> list[dict[str, Any]]:
    """Frozen V44 mutually-nearest one-frame gap bridge actions."""
    node_t = {int(node): int(value[0]) for node, value in nodes.items()}
    pos = {
        int(node): np.asarray(value[1:], np.float64) * SCALE_ARR
        for node, value in nodes.items()
    }
    has_out: set[int] = set()
    has_in: set[int] = set()
    for source, target, *_ in edges:
        has_out.add(int(source))
        has_in.add(int(target))
    terms = [
        node
        for node in node_t
        if node not in has_out
        and node in has_in
        and 0 < node_t[node] < max_terminal_t
    ]
    roots = [
        node
        for node in node_t
        if node not in has_in and node in has_out and node_t[node] >= 2
    ]
    roots_by_t: dict[int, list[int]] = defaultdict(list)
    terms_by_t: dict[int, list[int]] = defaultdict(list)
    for root in roots:
        roots_by_t[node_t[root]].append(root)
    for term in terms:
        terms_by_t[node_t[term]].append(term)
    pairs: list[tuple[int, int, float]] = []
    for frame, frame_terms in terms_by_t.items():
        frame_roots = roots_by_t.get(frame + 2, [])
        if not frame_roots:
            continue
        root_points = np.asarray([pos[root] for root in frame_roots])
        term_points = np.asarray([pos[term] for term in frame_terms])
        root_tree = cKDTree(root_points)
        term_tree = cKDTree(term_points)
        distance_tr, nearest_tr = root_tree.query(term_points, k=1)
        _, nearest_rt = term_tree.query(root_points, k=1)
        for index, term in enumerate(frame_terms):
            if distance_tr[index] > gate_um:
                continue
            root_index = int(nearest_tr[index])
            if int(nearest_rt[root_index]) != index:
                continue
            pairs.append(
                (term, frame_roots[root_index], float(distance_tr[index]))
            )
    out: list[dict[str, Any]] = []
    for term, root, gap in pairs:
        middle_frame = node_t[term] + 1
        part = pool[pool.t == middle_frame]
        if not len(part):
            continue
        candidate_um = part[["z", "y", "x"]].to_numpy(np.float64) * SCALE_ARR
        middle = (pos[term] + pos[root]) / 2.0
        distance = np.linalg.norm(candidate_um - middle, axis=1)
        eligible = distance <= place_um
        if not eligible.any():
            continue
        best = part.index[eligible][
            np.argmax(part.score.to_numpy()[eligible])
        ]
        row = part.loc[best]
        out.append(
            {
                "term": int(term),
                "root": int(root),
                "gap_um": gap,
                "t": middle_frame,
                "z": float(row.z),
                "y": float(row.y),
                "x": float(row.x),
                "cand_score": float(row.score),
            }
        )
    return out


def add_bridges(
    base: DatasetGraph,
    actions: list[dict[str, Any]],
    *,
    confidence: float = 0.3,
) -> DatasetGraph:
    nodes = dict(base.nodes)
    edges = list(base.edges)
    next_id = max(nodes, default=-1) + 1
    for action in actions:
        nodes[next_id] = (
            int(action["t"]),
            int(np.rint(action["z"])),
            int(np.rint(action["y"])),
            int(np.rint(action["x"])),
        )
        edges.append((int(action["term"]), next_id, confidence))
        edges.append((next_id, int(action["root"]), confidence))
        next_id += 1
    candidate = DatasetGraph(
        dataset=base.dataset, nodes=nodes, edges=edges, shape=base.shape
    )
    validate_graph(candidate)
    return candidate


def load_boosters(paths: Iterable[Path]) -> list[Any]:
    """Load serialized V55 boosters without importing LightGBM at module load."""
    import lightgbm as lgb

    return [lgb.Booster(model_file=str(path)) for path in paths]
