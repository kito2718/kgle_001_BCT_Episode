#!/usr/bin/env python3
"""Build sparse motion graphs from nnU-Net peaks and solve them with tracksdata ILP."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td
from scipy.spatial import cKDTree

SPACING_ZYX_UM = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-final", type=Path, required=True)
    parser.add_argument("--predicted-nodes-dir", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--det-threshold", type=float, default=0.3)
    parser.add_argument("--motion-mode", choices=("learned", "zero"), default="learned")
    parser.add_argument("--dilations", default="1")
    parser.add_argument("--gate-um", type=float, default=7.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--distance-sigma-um", type=float, default=3.0)
    parser.add_argument("--confidence-power", type=float, default=0.5)
    parser.add_argument("--edge-weight", type=float, default=-1.0)
    parser.add_argument("--appearance-weight", type=float, default=0.1)
    parser.add_argument("--disappearance-weight", type=float, default=0.1)
    parser.add_argument("--division-weight", type=float, default=1.0)
    parser.add_argument("--skip-ilp", action="store_true")
    parser.add_argument("--stem-limit", type=int)
    parser.add_argument(
        "--max-frame",
        type=int,
        help="Smoke-test only: ignore validation cases after this frame.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def case_stem_frame(case_id: str) -> tuple[str, int]:
    stem, frame_text = case_id.rsplit("_t", 1)
    return stem, int(frame_text)


def load_nodes(path: Path, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.empty((0, 3)), np.empty((0,))
    coords: list[list[float]] = []
    scores: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            score = float(row["score"])
            if score < threshold:
                continue
            coords.append([float(row["z"]), float(row["y"]), float(row["x"])])
            scores.append(score)
    return (
        np.asarray(coords, dtype=np.float64).reshape(-1, 3),
        np.asarray(scores, dtype=np.float64),
    )


def load_motion(
    path: Path,
    coords: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"motion export missing: {path}")
    with np.load(path) as payload:
        exported_coords = np.asarray(payload["zyx"], dtype=np.float64)
        exported_scores = np.asarray(payload["score"], dtype=np.float64)
        vectors = np.asarray(
            payload["motion_zyx_um_per_frame"], dtype=np.float64
        )
    keep = exported_scores >= threshold
    exported_coords = exported_coords[keep]
    exported_scores = exported_scores[keep]
    vectors = vectors[keep]
    if (
        exported_coords.shape != coords.shape
        or not np.allclose(exported_coords, coords, atol=1e-5)
        or not np.allclose(exported_scores, scores, atol=1e-5)
    ):
        raise RuntimeError(
            f"node/motion order mismatch for {path}: "
            f"{exported_coords.shape} versus {coords.shape}"
        )
    return vectors


def add_candidate_edges(
    edges: dict[tuple[int, int], tuple[float, float, int]],
    source_indices: np.ndarray,
    source_zyx: np.ndarray,
    source_scores: np.ndarray,
    target_indices: np.ndarray,
    target_zyx: np.ndarray,
    target_scores: np.ndarray,
    target_motion: np.ndarray,
    dilation: int,
    mode: str,
    gate_um: float,
    top_k: int,
    sigma_um: float,
    confidence_power: float,
) -> None:
    if not len(source_indices) or not len(target_indices):
        return
    source_um = source_zyx * SPACING_ZYX_UM
    target_um = target_zyx * SPACING_ZYX_UM
    projected_um = target_um.copy()
    if mode == "learned":
        projected_um += target_motion * float(dilation)

    tree = cKDTree(source_um)
    k = min(top_k, len(source_indices))
    distances, neighbors = tree.query(
        projected_um, k=k, distance_upper_bound=gate_um
    )
    if k == 1:
        distances = distances[:, None]
        neighbors = neighbors[:, None]
    for target_local in range(len(target_indices)):
        for distance, source_local in zip(
            distances[target_local], neighbors[target_local]
        ):
            if not np.isfinite(distance) or source_local >= len(source_indices):
                continue
            confidence = (
                source_scores[source_local] * target_scores[target_local]
            ) ** confidence_power
            probability = float(
                np.exp(-0.5 * (float(distance) / sigma_um) ** 2)
                * confidence
            )
            key = (
                int(source_indices[source_local]),
                int(target_indices[target_local]),
            )
            previous = edges.get(key)
            value = (probability, float(distance), dilation)
            if previous is None or probability > previous[0]:
                edges[key] = value


def build_graph(
    node_rows: list[dict[str, float | int]],
    edges: dict[tuple[int, int], tuple[float, float, int]],
) -> td.graph.InMemoryGraph:
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x", "node_score"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    node_ids = graph.bulk_add_nodes(node_rows)
    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.add_edge_attr_key("dilation", pl.Int64, 1)
        graph.bulk_add_edges(
            [
                {
                    "source_id": node_ids[source],
                    "target_id": node_ids[target],
                    "edge_prob": probability,
                    "edge_dist": distance,
                    "dilation": dilation,
                }
                for (source, target), (
                    probability,
                    distance,
                    dilation,
                ) in edges.items()
            ]
        )
    return graph


def save_graph(graph: td.graph.BaseGraph, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    graph.to_geff(output_path)


@contextlib.contextmanager
def suppress_solver_output():
    with open(os.devnull, "w") as null:
        with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
            yield


def main() -> None:
    args = arguments()
    dilations = tuple(
        int(value) for value in args.dilations.replace(",", " ").split()
    )
    if not dilations or any(value not in (1, 2) for value in dilations):
        raise ValueError("dilations must be a non-empty subset of 1,2")

    splits = json.loads(args.splits_final.read_text())
    cases = list(splits[args.fold]["val"])
    cases_by_stem: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for case_id in cases:
        stem, frame = case_stem_frame(case_id)
        if args.max_frame is not None and frame > args.max_frame:
            continue
        cases_by_stem[stem].append((frame, case_id))
    stems = sorted(cases_by_stem)
    if args.stem_limit is not None:
        stems = stems[: args.stem_limit]

    if args.output_dir.exists() and not args.resume:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_video: list[dict[str, float | int | str]] = []
    started = time.perf_counter()
    for stem_index, stem in enumerate(stems, start=1):
        output_path = args.output_dir / f"{stem}.geff"
        if args.resume and output_path.exists():
            continue
        frame_payload: dict[int, dict[str, np.ndarray]] = {}
        node_rows: list[dict[str, float | int]] = []
        for frame, case_id in sorted(cases_by_stem[stem]):
            coords, scores = load_nodes(
                args.predicted_nodes_dir / f"{case_id}.csv",
                args.det_threshold,
            )
            indices = np.arange(
                len(node_rows), len(node_rows) + len(coords), dtype=np.int64
            )
            node_rows.extend(
                {
                    "t": int(frame),
                    "z": float(coord[0]),
                    "y": float(coord[1]),
                    "x": float(coord[2]),
                    "node_score": float(score),
                }
                for coord, score in zip(coords, scores)
            )
            frame_payload[frame] = {
                "indices": indices,
                "coords": coords,
                "scores": scores,
            }

        edges: dict[tuple[int, int], tuple[float, float, int]] = {}
        for frame, case_id in sorted(cases_by_stem[stem]):
            target = frame_payload[frame]
            for dilation in dilations:
                if frame - dilation not in frame_payload:
                    continue
                vectors = load_motion(
                    args.motion_dir
                    / "cases"
                    / f"d{dilation}_{case_id}.npz",
                    target["coords"],
                    target["scores"],
                    args.det_threshold,
                )
                source = frame_payload[frame - dilation]
                add_candidate_edges(
                    edges,
                    source["indices"],
                    source["coords"],
                    source["scores"],
                    target["indices"],
                    target["coords"],
                    target["scores"],
                    vectors,
                    dilation,
                    args.motion_mode,
                    args.gate_um,
                    args.top_k,
                    args.distance_sigma_um,
                    args.confidence_power,
                )

        graph = build_graph(node_rows, edges)
        raw_edges = int(graph.num_edges())
        solve_seconds = 0.0
        if not args.skip_ilp and raw_edges:
            solver = td.solvers.ILPSolver(
                edge_weight=args.edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=args.appearance_weight,
                disappearance_weight=args.disappearance_weight,
                division_weight=args.division_weight,
            )
            solve_started = time.perf_counter()
            with suppress_solver_output():
                graph = solver.solve(graph)
            solve_seconds = time.perf_counter() - solve_started
        save_graph(graph, output_path)
        row = {
            "stem": stem,
            "nodes": int(graph.num_nodes()),
            "candidate_edges": raw_edges,
            "selected_edges": int(graph.num_edges()),
            "solve_seconds": solve_seconds,
        }
        per_video.append(row)
        print(
            f"[{stem_index}/{len(stems)}] {stem} "
            f"nodes={row['nodes']} edges={raw_edges}->{row['selected_edges']} "
            f"solve={solve_seconds:.2f}s",
            flush=True,
        )

    if per_video:
        with (args.output_dir / "graph_stats.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_video[0]))
            writer.writeheader()
            writer.writerows(per_video)
    summary = {
        "schema_version": 1,
        "fold": args.fold,
        "stems": len(stems),
        "max_frame": args.max_frame,
        "det_threshold": args.det_threshold,
        "motion_mode": args.motion_mode,
        "dilations": list(dilations),
        "gate_um": args.gate_um,
        "top_k": args.top_k,
        "distance_sigma_um": args.distance_sigma_um,
        "confidence_power": args.confidence_power,
        "skip_ilp": args.skip_ilp,
        "ilp": {
            "edge_weight": args.edge_weight,
            "appearance_weight": args.appearance_weight,
            "disappearance_weight": args.disappearance_weight,
            "division_weight": args.division_weight,
        },
        "videos_completed_now": len(per_video),
        "nodes": int(sum(int(row["nodes"]) for row in per_video)),
        "candidate_edges": int(
            sum(int(row["candidate_edges"]) for row in per_video)
        ),
        "selected_edges": int(
            sum(int(row["selected_edges"]) for row in per_video)
        ),
        "solve_seconds": float(
            sum(float(row["solve_seconds"]) for row in per_video)
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
