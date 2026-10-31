"""Ground-truth loading — one place, so every oracle sees the same numbers.

A `.geff` is a zarr group. `tracksdata` can read it into a graph object, but the
oracle ladder needs plain arrays far more often than it needs a graph, and
building a graph per sample 199 times is wasted work. So this module reads the
arrays directly and builds the tracksdata graph only when the official scorer is
actually going to be called.

Coordinates stay in **source-resolution voxels** everywhere. The scorer applies
the physical scale itself, and mixing the two conventions is the single easiest
way to produce a plausible-looking wrong number here (DATA.md §2: z is 4x
coarser than y/x, so an isotropic distance is wrong by up to 4x).
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Micrometres per source-resolution voxel, (z, y, x).
SCALE_UM: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
SCALE_ARR = np.array(SCALE_UM)
# The official matching radius. Not a tunable.
MATCH_UM: float = 7.0


@dataclass
class GroundTruth:
    """One sample's annotated graph, in source-resolution voxels."""

    dataset: str
    embryo: str
    path: Path
    node_ids: np.ndarray  # (N,) int64 — the geff ids, needed to map edges back
    t: np.ndarray  # (N,) int64
    zyx: np.ndarray  # (N, 3) float64, voxel units
    edges: np.ndarray  # (E, 2) int64 — *row indices* into the node arrays
    n_total: float  # estimated_number_of_nodes: ALL real cells, not annotated ones

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def zyx_um(self) -> np.ndarray:
        return self.zyx * SCALE_ARR

    def divisions(self) -> np.ndarray:
        """Row indices of nodes with out-degree 2."""
        if self.n_edges == 0:
            return np.zeros(0, dtype=np.int64)
        src, counts = np.unique(self.edges[:, 0], return_counts=True)
        return src[counts >= 2]

    def to_tracksdata(self):
        """Build the tracksdata graph the official scorer expects."""
        import polars as pl
        import tracksdata as td

        g = td.graph.IndexedRXGraph()
        for axis in ("z", "y", "x"):
            g.add_node_attr_key(axis, pl.Float64, 0.0)
        remap = {}
        for i in range(self.n_nodes):
            z, y, x = self.zyx[i]
            remap[i] = g.add_node(
                {"t": int(self.t[i]), "z": float(z), "y": float(y), "x": float(x)}
            )
        for s, d in self.edges:
            g.add_edge(remap[int(s)], remap[int(d)], {})
        return g


def _read_geff_arrays(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, float]:
    """Read node ids, node attributes, edge id pairs and `estimated_number_of_nodes`.

    Reads the zarr arrays directly rather than going through tracksdata: this is
    called for all 199 samples in every oracle, and the graph object is only
    needed when the official scorer runs.
    """
    import zarr

    root = zarr.open_group(str(path), mode="r")
    node_ids = np.asarray(root["nodes"]["ids"][:]).astype(np.int64)

    props = root["nodes"]["props"]
    attrs = {k: np.asarray(props[k]["values"][:]) for k in ("t", "z", "y", "x")}

    edge_ids = np.asarray(root["edges"]["ids"][:]).astype(np.int64).reshape(-1, 2)

    meta = json.loads(Path(path, "zarr.json").read_text())
    extra = meta.get("attributes", {}).get("geff", {}).get("extra", {}) or {}
    n_total = float(extra.get("estimated_number_of_nodes", float("nan")))
    return node_ids, attrs, edge_ids, n_total


def load_gt(path: str | Path) -> GroundTruth:
    """Load one `.geff`. `path` may name the `.geff` or the bare stem."""
    path = Path(path)
    if path.suffix != ".geff":
        path = path.with_suffix(".geff")

    node_ids, attrs, edge_id_pairs, n_total = _read_geff_arrays(path)

    # Edges reference geff node ids; the rest of the code indexes rows.
    order = np.argsort(node_ids)
    lookup = node_ids[order]
    if len(edge_id_pairs):
        idx = order[np.searchsorted(lookup, edge_id_pairs.ravel())].reshape(-1, 2)
        if not (node_ids[idx].ravel() == edge_id_pairs.ravel()).all():
            raise ValueError(f"{path.name}: edge references an unknown node id")
    else:
        idx = np.zeros((0, 2), dtype=np.int64)

    t = attrs["t"].astype(np.int64)
    zyx = np.stack([attrs["z"], attrs["y"], attrs["x"]], axis=1).astype(np.float64)

    dataset = path.stem
    return GroundTruth(
        dataset=dataset,
        embryo=dataset.split("_")[0],
        path=path,
        node_ids=node_ids,
        t=t,
        zyx=zyx,
        edges=idx,
        n_total=n_total,
    )


def subset_gt(gt: GroundTruth, keep_rows: np.ndarray) -> GroundTruth:
    """A label subset of one sample: keep these node rows and re-index the edges.

    Only edges whose **both** endpoints survive are kept, so the returned graph is
    internally consistent and `parents_for_frame` cannot resolve a parent that is
    no longer a labelled node.

    `n_total` is deliberately carried through unchanged. It is the number of real
    cells in the sample — a fact about the embryo, not about the annotation — and
    the submission budget is derived from it. Thinning labels must not move it.
    """
    keep = np.unique(np.asarray(keep_rows, dtype=np.int64))
    if len(keep) and (keep[0] < 0 or keep[-1] >= gt.n_nodes):
        raise ValueError(f"{gt.dataset}: keep_rows out of range for {gt.n_nodes} nodes")
    remap = np.full(gt.n_nodes, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    if gt.n_edges:
        m = (remap[gt.edges[:, 0]] >= 0) & (remap[gt.edges[:, 1]] >= 0)
        edges = remap[gt.edges[m]]
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
    return GroundTruth(
        dataset=gt.dataset,
        embryo=gt.embryo,
        path=gt.path,
        node_ids=gt.node_ids[keep],
        t=gt.t[keep],
        zyx=gt.zyx[keep],
        edges=edges,
        n_total=gt.n_total,
    )


def discover_gt(gt_dir: str | Path) -> list[Path]:
    return sorted(Path(gt_dir).glob("*.geff"))


@functools.lru_cache(maxsize=None)
def load_all_gt(gt_dir: str = "data/train") -> tuple[GroundTruth, ...]:
    """Every sample under `gt_dir`, cached for the life of the process."""
    return tuple(load_gt(p) for p in discover_gt(gt_dir))
