#!/usr/bin/env python3
"""Flatten validated per-video GEFF graphs into Biohub submission.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import tracksdata as td
import zarr


COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


def read_indexed_graph(path: Path):
    result = td.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result


def validate_sample_columns(path: Path | None) -> None:
    if path is None:
        return
    with path.open(newline="", encoding="utf-8") as handle:
        columns = next(csv.reader(handle))
    if columns != COLUMNS:
        raise RuntimeError(
            f"sample submission columns differ: {columns} versus {COLUMNS}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--diagnostics-json", type=Path)
    parser.add_argument("--limit-videos", type=int)
    args = parser.parse_args()
    validate_sample_columns(args.sample_submission)

    test_paths = sorted(args.test_dir.glob("*.zarr"))
    if args.limit_videos is not None:
        test_paths = test_paths[: args.limit_videos]
    expected = [path.name.removesuffix(".zarr") for path in test_paths]
    graph_paths = {path.stem: path for path in args.graphs_dir.glob("*.geff")}
    missing = sorted(set(expected) - set(graph_paths))
    extra = sorted(set(graph_paths) - set(expected))
    if missing or extra:
        raise RuntimeError(
            f"GEFF coverage mismatch: missing={missing}, extra={extra}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_csv.with_suffix(".tmp.csv")
    row_id = 0
    videos = []
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for test_path, stem in zip(test_paths, expected):
            shape = tuple(
                int(value)
                for value in zarr.open_group(str(test_path), mode="r")["0"].shape
            )
            graph = read_indexed_graph(graph_paths[stem])
            nodes = list(graph.node_attrs().iter_rows(named=True))
            edges = list(graph.edge_attrs().iter_rows(named=True))
            if not nodes:
                raise RuntimeError(f"{stem}: graph contains no nodes")

            node_time: dict[int, int] = {}
            node_ids = []
            for row in nodes:
                node_id = int(row["node_id"])
                t = int(row["t"])
                coords = tuple(float(row[axis]) for axis in ("z", "y", "x"))
                if node_id in node_time:
                    raise RuntimeError(f"{stem}: duplicate node_id {node_id}")
                if not all(math.isfinite(value) for value in coords):
                    raise RuntimeError(f"{stem}: non-finite node {node_id}")
                rounded = tuple(int(round(value)) for value in coords)
                if not (
                    0 <= t < shape[0]
                    and 0 <= rounded[0] < shape[1]
                    and 0 <= rounded[1] < shape[2]
                    and 0 <= rounded[2] < shape[3]
                ):
                    raise RuntimeError(
                        f"{stem}: out-of-bounds node {node_id}: "
                        f"tzyx={(t, *rounded)}, shape={shape}"
                    )
                node_time[node_id] = t
                node_ids.append(node_id)
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": stem,
                        "row_type": "node",
                        "node_id": node_id,
                        "t": t,
                        "z": rounded[0],
                        "y": rounded[1],
                        "x": rounded[2],
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
                row_id += 1

            edge_pairs = set()
            indegree = Counter()
            out_targets: dict[int, list[int]] = defaultdict(list)
            for row in edges:
                source = int(row["source_id"])
                target = int(row["target_id"])
                if source not in node_time or target not in node_time:
                    raise RuntimeError(
                        f"{stem}: dangling edge {source}->{target}"
                    )
                if (source, target) in edge_pairs:
                    raise RuntimeError(
                        f"{stem}: duplicate edge {source}->{target}"
                    )
                if node_time[source] >= node_time[target]:
                    raise RuntimeError(
                        f"{stem}: invalid time direction {source}->{target}"
                    )
                edge_pairs.add((source, target))
                indegree[target] += 1
                out_targets[source].append(target)
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": stem,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": source,
                        "target_id": target,
                    }
                )
                row_id += 1

            if indegree and max(indegree.values()) > 1:
                raise RuntimeError(f"{stem}: target indegree exceeds one")
            if out_targets and max(map(len, out_targets.values())) > 2:
                raise RuntimeError(f"{stem}: source outdegree exceeds two")
            for source, targets in out_targets.items():
                if len(targets) == 2 and len(
                    {node_time[target] for target in targets}
                ) != 1:
                    raise RuntimeError(
                        f"{stem}: division children have different times "
                        f"for source {source}"
                    )

            video = {
                "dataset": stem,
                "shape_tzyx": list(shape),
                "nodes": len(nodes),
                "edges": len(edges),
                "divisions": sum(
                    len(targets) == 2 for targets in out_targets.values()
                ),
            }
            videos.append(video)

    os.replace(temporary, args.output_csv)
    diagnostics = {
        "schema_version": 1,
        "submission_csv": str(args.output_csv),
        "columns": COLUMNS,
        "datasets": len(videos),
        "rows": row_id,
        "nodes": int(sum(item["nodes"] for item in videos)),
        "edges": int(sum(item["edges"] for item in videos)),
        "divisions": int(sum(item["divisions"] for item in videos)),
        "coverage_ok": True,
        "dangling_edges": 0,
        "duplicate_edges": 0,
        "invalid_time_edges": 0,
        "out_of_bounds_nodes": 0,
        "videos": videos,
    }
    diagnostics_path = (
        args.diagnostics_json
        if args.diagnostics_json is not None
        else args.output_csv.with_suffix(".diagnostics.json")
    )
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
