"""Label-free temporal consensus witnesses from two raw candidate banks.

This module deliberately stops before graph solving.  Each bank selects the
same frozen high-recall candidate surface, chooses its own flow-centred top-1
parent, and contributes an edge only when the other bank made the identical
choice after one-to-one spatial mapping.  The resulting graph is a witness for
the additive segment policy; it is never used as a submission graph directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biohub.io.submission import DatasetGraph
from biohub.metrics.matching import match_sequence
from biohub.tracker.edges import build_edges, select_nodes


MAP_UM = 3.0
NMS_UM = 5.0
BUDGET_MULTIPLIER = 2
FLOW_GATE_UM = 10.0


def _selected_arrays(
    candidates: pd.DataFrame,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer frame and submission-coordinate arrays for a selection."""
    times = candidates["t"].to_numpy()[selected].astype(np.int64, copy=False)
    coordinates = np.rint(
        candidates[["z", "y", "x"]].to_numpy(dtype=np.float64)[selected]
    ).astype(np.int64)
    return times, coordinates


def _witness_graph(
    dataset: str,
    candidates: pd.DataFrame,
    selected: np.ndarray,
    pairs: set[tuple[int, int]],
    shape: tuple[int, int, int, int],
) -> DatasetGraph:
    """Convert selected-array edge pairs to a compact witness graph."""
    used = sorted({node for pair in pairs for node in pair})
    remap = {old: new for new, old in enumerate(used)}
    candidate_rows = selected[np.asarray(used, dtype=np.int64)] if used else np.zeros(0, int)
    raw = (
        np.rint(
            candidates[["t", "z", "y", "x"]]
            .to_numpy(dtype=np.float64)[candidate_rows]
        ).astype(np.int64)
        if used
        else np.zeros((0, 4), dtype=np.int64)
    )
    if len(raw):
        raw = np.clip(raw, np.zeros(4, dtype=np.int64), np.asarray(shape) - 1)
    nodes = {
        remap[old]: tuple(int(value) for value in coordinate)
        for old, coordinate in zip(used, raw)
    }
    edges = [
        (remap[parent], remap[child], 1.0)
        for parent, child in sorted(pairs)
    ]
    return DatasetGraph(dataset=dataset, nodes=nodes, edges=edges, shape=shape)


def top1_candidate_witness(
    dataset: str,
    candidates: pd.DataFrame,
    budget_per_frame: int,
    shape: tuple[int, int, int, int],
) -> tuple[DatasetGraph, dict[str, Any]]:
    """Build a frozen high-recall, flow-centred top-1 candidate witness."""
    budget = BUDGET_MULTIPLIER * int(budget_per_frame)
    selected = select_nodes(candidates, budget, nms_um=NMS_UM)
    edges = build_edges(
        candidates,
        selected,
        k=1,
        gate_um=FLOW_GATE_UM,
    )
    pairs = {
        (int(row.pi), int(row.ci))
        for row in edges[["pi", "ci"]].itertuples(index=False)
    }
    witness = _witness_graph(dataset, candidates, selected, pairs, shape)
    audit: dict[str, Any] = {
        "dataset": dataset,
        "budget_per_frame": int(budget_per_frame),
        "selection_budget": budget,
        "selected_nodes": int(len(selected)),
        "top1_edges": int(len(pairs)),
        "witness_nodes": int(len(witness.nodes)),
        "witness_edges": int(len(witness.edges)),
    }
    return witness, audit


def dual_bank_top1_witnesses(
    dataset: str,
    v21_candidates: pd.DataFrame,
    v24_candidates: pd.DataFrame,
    budget_per_frame: int,
    shape: tuple[int, int, int, int],
) -> tuple[DatasetGraph, DatasetGraph, dict[str, Any]]:
    """Build mapped-V21 control and dual-bank-agreement witness graphs.

    Node selection, mapping, and parent selection are fixed constants.  The
    control retains every V21 top-1 edge whose endpoints map to V24.  The
    primary is the subset for which V24 independently chooses the same mapped
    parent-child pair.  No labels or graph-solver output enter this function.
    """
    budget = BUDGET_MULTIPLIER * int(budget_per_frame)
    selected21 = select_nodes(v21_candidates, budget, nms_um=NMS_UM)
    selected24 = select_nodes(v24_candidates, budget, nms_um=NMS_UM)
    time21, coordinates21 = _selected_arrays(v21_candidates, selected21)
    time24, coordinates24 = _selected_arrays(v24_candidates, selected24)

    # match_sequence returns, for every V24 node, its position in V21.
    map24_to21 = match_sequence(
        time21,
        coordinates21,
        time24,
        coordinates24,
        MAP_UM,
    )
    map21_to24 = np.full(len(selected21), -1, dtype=np.int64)
    valid24 = np.flatnonzero(map24_to21 >= 0)
    if len(valid24):
        map21_to24[map24_to21[valid24]] = valid24

    edges21 = build_edges(
        v21_candidates,
        selected21,
        k=1,
        gate_um=FLOW_GATE_UM,
    )
    edges24 = build_edges(
        v24_candidates,
        selected24,
        k=1,
        gate_um=FLOW_GATE_UM,
    )
    pairs21 = {
        (int(row.pi), int(row.ci))
        for row in edges21[["pi", "ci"]].itertuples(index=False)
    }
    mapped_pairs24 = {
        (int(map24_to21[int(row.pi)]), int(map24_to21[int(row.ci)]))
        for row in edges24[["pi", "ci"]].itertuples(index=False)
        if map24_to21[int(row.pi)] >= 0 and map24_to21[int(row.ci)] >= 0
    }
    control_pairs = {
        (parent, child)
        for parent, child in pairs21
        if map21_to24[parent] >= 0 and map21_to24[child] >= 0
    }
    agreed_pairs = control_pairs & mapped_pairs24

    control = _witness_graph(
        dataset,
        v21_candidates,
        selected21,
        control_pairs,
        shape,
    )
    primary = _witness_graph(
        dataset,
        v21_candidates,
        selected21,
        agreed_pairs,
        shape,
    )
    audit: dict[str, Any] = {
        "dataset": dataset,
        "budget_per_frame": int(budget_per_frame),
        "selection_budget": budget,
        "v21_selected_nodes": int(len(selected21)),
        "v24_selected_nodes": int(len(selected24)),
        "mapped_nodes": int(len(valid24)),
        "v21_top1_edges": int(len(pairs21)),
        "v24_top1_edges": int(len(edges24)),
        "mapped_v24_top1_edges": int(len(mapped_pairs24)),
        "control_edges": int(len(control_pairs)),
        "agreed_edges": int(len(agreed_pairs)),
        "control_nodes": int(len(control.nodes)),
        "agreed_nodes": int(len(primary.nodes)),
        "agreement_rate": (
            float(len(agreed_pairs) / len(control_pairs)) if control_pairs else 0.0
        ),
    }
    return control, primary, audit


def dual_bank_spatially_confirm_witness(
    witness: DatasetGraph,
    v21_candidates: pd.DataFrame,
    v24_candidates: pd.DataFrame,
    budget_per_frame: int,
) -> tuple[DatasetGraph, dict[str, Any]]:
    """Keep witness edges whose endpoints have support in both raw banks.

    The two one-to-one mappings are independent.  Candidate scores, flows,
    parent choices, and labels do not enter the confirmation decision.
    """
    budget = BUDGET_MULTIPLIER * int(budget_per_frame)
    selected21 = select_nodes(v21_candidates, budget, nms_um=NMS_UM)
    selected24 = select_nodes(v24_candidates, budget, nms_um=NMS_UM)
    time21, coordinates21 = _selected_arrays(v21_candidates, selected21)
    time24, coordinates24 = _selected_arrays(v24_candidates, selected24)

    witness_ids = np.asarray(sorted(witness.nodes), dtype=np.int64)
    witness_coordinates = np.asarray(
        [witness.nodes[int(node)] for node in witness_ids],
        dtype=np.int64,
    ).reshape(-1, 4)
    witness_time = witness_coordinates[:, 0]
    witness_zyx = witness_coordinates[:, 1:]
    match21 = match_sequence(
        time21,
        coordinates21,
        witness_time,
        witness_zyx,
        MAP_UM,
    )
    match24 = match_sequence(
        time24,
        coordinates24,
        witness_time,
        witness_zyx,
        MAP_UM,
    )
    confirmed = {
        int(node)
        for node, in21, in24 in zip(witness_ids, match21 >= 0, match24 >= 0)
        if in21 and in24
    }
    retained_edges = [
        (int(parent), int(child), float(confidence))
        for parent, child, confidence in witness.edges
        if int(parent) in confirmed and int(child) in confirmed
    ]
    used = sorted(
        {node for parent, child, _ in retained_edges for node in (parent, child)}
    )
    remap = {old: new for new, old in enumerate(used)}
    filtered = DatasetGraph(
        dataset=witness.dataset,
        nodes={remap[node]: witness.nodes[node] for node in used},
        edges=[
            (remap[parent], remap[child], confidence)
            for parent, child, confidence in retained_edges
        ],
        shape=witness.shape,
    )
    audit: dict[str, Any] = {
        "dataset": witness.dataset,
        "budget_per_frame": int(budget_per_frame),
        "selection_budget": budget,
        "v21_selected_nodes": int(len(selected21)),
        "v24_selected_nodes": int(len(selected24)),
        "stage6_nodes": int(len(witness.nodes)),
        "stage6_edges": int(len(witness.edges)),
        "v21_supported_stage6_nodes": int(np.sum(match21 >= 0)),
        "v24_supported_stage6_nodes": int(np.sum(match24 >= 0)),
        "dual_supported_stage6_nodes": int(len(confirmed)),
        "retained_stage6_nodes": int(len(filtered.nodes)),
        "retained_stage6_edges": int(len(filtered.edges)),
    }
    return filtered, audit
