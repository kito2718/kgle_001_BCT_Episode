#!/usr/bin/env python3
"""Small regression checks for hidden-rerun postprocess safety guards."""

from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import biohub_highscore_postprocess_ablation as post


class _FakeArray:
    shape = (3, 2, 4, 5)


class _FakeGroup(dict):
    def __init__(self):
        super().__init__({"0": _FakeArray()})


def main() -> None:
    original_open_group = post.zarr.open_group
    post.zarr.open_group = lambda *args, **kwargs: _FakeGroup()
    try:
        nodes = {
            1: {"node_id": 1, "t": 0, "z": -0.25, "y": 1.0, "x": 9.0},
            2: {"node_id": 2, "t": 1, "z": 1.25, "y": 3.5, "x": 2.0},
        }
        edges = [{"source_id": 1, "target_id": 2}]
        stats = defaultdict(int)
        post.validate_and_clip_graph(nodes, edges, "synthetic", Path("/unused"), stats)
        assert nodes[1]["z"] == 0.0
        assert nodes[1]["x"] == 4.0
        assert nodes[2]["z"] == 1.0
        assert nodes[2]["y"] == 3.0
        assert stats["bounds_clipped_nodes"] == 2
        assert stats["bounds_clipped_coordinates"] == 4

        bad_nodes = {key: dict(value) for key, value in nodes.items()}
        bad_nodes[2]["x"] = np.nan
        try:
            post.validate_and_clip_graph(
                bad_nodes, edges, "synthetic", Path("/unused"), defaultdict(int)
            )
        except ValueError as exc:
            assert "non-finite" in str(exc)
        else:
            raise AssertionError("non-finite coordinate was not rejected")

        bad_edges = [{"source_id": 1, "target_id": 999}]
        try:
            post.validate_and_clip_graph(
                nodes, bad_edges, "synthetic", Path("/unused"), defaultdict(int)
            )
        except ValueError as exc:
            assert "dangling" in str(exc)
        else:
            raise AssertionError("dangling edge was not rejected")
    finally:
        post.zarr.open_group = original_open_group

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.geff"
        destination = root / "destination.geff"
        source.mkdir()
        (source / "sentinel.txt").write_text("original\n")
        destination.mkdir()
        (destination / "partial.txt").write_text("partial\n")
        post.copy_graph_fallback(source, destination)
        assert (destination / "sentinel.txt").read_text() == "original\n"
        assert not (destination / "partial.txt").exists()

    print("postprocess hidden-safety regression: PASS")


if __name__ == "__main__":
    main()
