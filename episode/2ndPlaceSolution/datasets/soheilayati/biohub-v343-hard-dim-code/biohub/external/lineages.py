"""Canonical readers for the downloaded Linajea and ITEC annotations.

The external packages use different column names, coordinate order, and edge
representations.  These readers intentionally preserve the published
coordinate values: physical rescaling is a model-specific concern and must
not be silently baked into the audit copy of the labels.

Canonical nodes contain ``source``, ``sequence``, ``node_id``, ``t``,
``z``, ``y``, and ``x``.  Canonical edges contain ``source_id`` and
``target_id`` within the same source/sequence.  All IDs are int64.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NODE_KEY = ["source", "sequence", "node_id"]
EDGE_KEY = ["source", "sequence", "source_id", "target_id"]


def _integral_ids(values: pd.Series, label: str) -> np.ndarray:
    """Return int64 IDs, rejecting non-integral or missing published IDs."""
    numeric = pd.to_numeric(values, errors="raise").to_numpy(np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{label} contains missing or non-finite IDs")
    rounded = np.rint(numeric)
    if not np.allclose(numeric, rounded, rtol=0, atol=1e-6):
        raise ValueError(f"{label} contains non-integral IDs")
    return rounded.astype(np.int64)


def _division_rows(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    degree = (edges.groupby(["source", "sequence", "source_id"], sort=False)
              .size().rename("out_degree").reset_index())
    div = degree[degree.out_degree == 2].rename(
        columns={"source_id": "node_id"})
    cols = ["source", "sequence", "node_id", "t", "z", "y", "x"]
    return div.merge(nodes[cols], on=["source", "sequence", "node_id"],
                     validate="one_to_one")[cols].reset_index(drop=True)


def division_triplets(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Return every binary fork with stable parent and child IDs."""
    children = (edges.groupby(["source", "sequence", "source_id"], sort=False)
                .target_id.agg(list).reset_index())
    children = children[children.target_id.map(len) == 2].copy()
    children["child1_id"] = children.target_id.map(lambda value: min(value))
    children["child2_id"] = children.target_id.map(lambda value: max(value))
    children = children.rename(columns={"source_id": "parent_id"})
    coords = nodes[["source", "sequence", "node_id", "t", "z", "y", "x"]].rename(
        columns={"node_id": "parent_id"})
    return (children.merge(
        coords, on=["source", "sequence", "parent_id"], validate="one_to_one")
        [["source", "sequence", "parent_id", "child1_id", "child2_id",
          "t", "z", "y", "x"]]
        .reset_index(drop=True))


def load_linajea_ground_truth(
    root: str | Path,
    sequences: Iterable[str] = (
        "tracks_side_1", "tracks_side_2", "discarded"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load manually traced Linajea zebrafish lineages and division census.

    Returns ``(nodes, edges, lineage_divisions, census_divisions)``.  The
    first three objects come from the sparse, fully traced manual lineages.
    The census is the separate set of manual pre-division points that is
    approximately exhaustive at its published target/adjacent frames.
    """
    root = Path(root)
    gt = root / "gt_tracks"
    node_parts: list[pd.DataFrame] = []
    edge_parts: list[pd.DataFrame] = []
    raw_cols = ["t", "z", "y", "x", "node_id", "parent_id", "lineage_id"]
    for sequence in sequences:
        path = gt / f"{sequence}.txt"
        frame = pd.read_csv(path, sep="\t", header=None, names=raw_cols)
        frame["source"] = "linajea"
        frame["sequence"] = sequence
        frame["node_id"] = _integral_ids(frame.node_id, f"{sequence}.node_id")
        frame["parent_id"] = _integral_ids(
            frame.parent_id, f"{sequence}.parent_id")
        frame["lineage_id"] = _integral_ids(
            frame.lineage_id, f"{sequence}.lineage_id")
        frame["t"] = _integral_ids(frame.t, f"{sequence}.t").astype(np.int32)
        for col in ["z", "y", "x"]:
            frame[col] = pd.to_numeric(frame[col], errors="raise").astype(np.float64)
        nodes = frame[["source", "sequence", "node_id", "t", "z", "y", "x",
                       "lineage_id"]].copy()
        linked = frame.parent_id != -1
        edges = frame.loc[linked, ["source", "sequence", "parent_id", "node_id"]].rename(
            columns={"parent_id": "source_id", "node_id": "target_id"})
        node_parts.append(nodes)
        edge_parts.append(edges)

    nodes = pd.concat(node_parts, ignore_index=True)
    edges = pd.concat(edge_parts, ignore_index=True)
    divisions = _division_rows(nodes, edges)

    census_path = gt / "full_frame_divisions.txt"
    census = pd.read_csv(census_path, sep="\t", header=None,
                         names=["t", "z", "y", "x", "node_id"])
    census["source"] = "linajea"
    census["sequence"] = "full_frame_divisions"
    census["node_id"] = _integral_ids(census.node_id, "division_census.node_id")
    census["t"] = _integral_ids(census.t, "division_census.t").astype(np.int32)
    census = census[["source", "sequence", "node_id", "t", "z", "y", "x"]]
    return nodes, edges, divisions, census


def load_celegans_curated_tracks(
    root: str | Path,
    *,
    microscope: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the six Zenodo C. elegans lineages into image-voxel coordinates.

    ``mskcc`` tracks are stored in lateral-pixel physical units, so their
    published z coordinate is divided by five to address the 0.75/0.15 um image
    grid. NIH diSPIM images and tracks are isotropic at 0.1625 um.
    """
    root = Path(root)
    specs = {
        "mskcc": {
            "source": "celegans_mskcc",
            "path_token": "mskcc_confocal_",
            "scale": (0.75, 0.15, 0.15),
            "z_divisor": 5.0,
        },
        "nih": {
            "source": "celegans_nih",
            "path_token": "nih_diSPIM_deconv_",
            "scale": (0.1625, 0.1625, 0.1625),
            "z_divisor": 1.0,
        },
    }
    if microscope not in specs:
        raise ValueError(f"unknown microscope {microscope!r}; expected {sorted(specs)}")
    spec = specs[microscope]
    paths = sorted(
        path for path in root.rglob("tracks.txt")
        if spec["path_token"] in path.as_posix())
    if len(paths) != 3:
        raise FileNotFoundError(
            f"expected three {microscope} tracks.txt files under {root}, found {len(paths)}")

    node_parts: list[pd.DataFrame] = []
    edge_parts: list[pd.DataFrame] = []
    scale_z, scale_y, scale_x = spec["scale"]
    for path in paths:
        sequence = path.parents[1].name
        frame = pd.read_csv(path, sep="\t")
        expected = {
            "t", "z", "y", "x", "cell_id", "parent_id", "track_id",
            "radius", "name", "div_state",
        }
        if missing := expected - set(frame.columns):
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        node_id = _integral_ids(frame.cell_id, f"{sequence}.cell_id")
        parent_id = _integral_ids(frame.parent_id, f"{sequence}.parent_id")
        t = _integral_ids(frame.t, f"{sequence}.t").astype(np.int32)
        z_native = pd.to_numeric(frame.z, errors="raise").to_numpy(np.float64)
        y = pd.to_numeric(frame.y, errors="raise").to_numpy(np.float64)
        x = pd.to_numeric(frame.x, errors="raise").to_numpy(np.float64)
        z = z_native / float(spec["z_divisor"])
        nodes = pd.DataFrame({
            "source": spec["source"],
            "sequence": sequence,
            "node_id": node_id,
            "track_id": _integral_ids(frame.track_id, f"{sequence}.track_id"),
            "t": t,
            "z": z,
            "y": y,
            "x": x,
            "z_um": z * scale_z,
            "y_um": y * scale_y,
            "x_um": x * scale_x,
            "z_published": z_native,
            "radius_published": pd.to_numeric(frame.radius, errors="raise"),
            "cell_name": frame.name.fillna("").astype(str),
            "division_state": _integral_ids(
                frame.div_state, f"{sequence}.div_state").astype(np.int16),
            "annotation_quality": "gold_manual_curated",
        })
        linked = parent_id != -1
        edges = pd.DataFrame({
            "source": spec["source"],
            "sequence": sequence,
            "source_id": parent_id[linked],
            "target_id": node_id[linked],
            "annotation_quality": "gold_manual_curated",
        })
        node_parts.append(nodes)
        edge_parts.append(edges)

    nodes = pd.concat(node_parts, ignore_index=True)
    edges = pd.concat(edge_parts, ignore_index=True)
    validate_lineage_graph(nodes, edges)
    divisions = division_triplets(nodes, edges)
    divisions["annotation_quality"] = "gold_manual_curated"
    return nodes, edges, divisions


def _itec_base(root: str | Path) -> Path:
    root = Path(root)
    nested = root / "Evaluation results of ITEC"
    return nested if nested.is_dir() else root


def load_itec_tracking_ground_truth(
    root: str | Path,
    sequences: Iterable[str] = ("FISH1", "FISH2"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ITEC's manually annotated sparse tracking ground truth."""
    base = _itec_base(root) / "TrackingGroundtruth"
    node_parts: list[pd.DataFrame] = []
    edge_parts: list[pd.DataFrame] = []
    for sequence in sequences:
        folder = base / sequence
        spots = pd.read_csv(folder / f"Groundtruth-{sequence}-Spot.csv")
        links = pd.read_csv(folder / f"Groundtruth-{sequence}-Link.csv")
        nodes = pd.DataFrame({
            "source": "itec",
            "sequence": sequence,
            "node_id": _integral_ids(spots["ID"], f"{sequence}.ID"),
            "t": _integral_ids(spots["Spot frame"],
                               f"{sequence}.Spot frame").astype(np.int32),
            "z": pd.to_numeric(spots["Z"], errors="raise").astype(np.float64),
            "y": pd.to_numeric(spots["Y"], errors="raise").astype(np.float64),
            "x": pd.to_numeric(spots["X"], errors="raise").astype(np.float64),
        })
        edges = pd.DataFrame({
            "source": "itec",
            "sequence": sequence,
            "source_id": _integral_ids(links["Source ID"],
                                       f"{sequence}.Source ID"),
            "target_id": _integral_ids(links["Target ID"],
                                       f"{sequence}.Target ID"),
        })
        node_parts.append(nodes)
        edge_parts.append(edges)
    nodes = pd.concat(node_parts, ignore_index=True)
    edges = pd.concat(edge_parts, ignore_index=True)
    return nodes, edges, _division_rows(nodes, edges)


def load_itec_detection_ground_truth(
    root: str | Path,
    sequences: Iterable[str] = ("FISH1", "FISH2"),
) -> pd.DataFrame:
    """Load ITEC's manually annotated full-frame detection censuses."""
    base = _itec_base(root) / "DetectionGroundtruth"
    parts: list[pd.DataFrame] = []
    for sequence in sequences:
        spots = pd.read_csv(base / f"Detectiongroundtruth-{sequence}.csv")
        parts.append(pd.DataFrame({
            "source": "itec",
            "sequence": sequence,
            "node_id": _integral_ids(spots["ID"], f"{sequence}.detection.ID"),
            "t": _integral_ids(spots["Spot frame"],
                               f"{sequence}.detection.frame").astype(np.int32),
            "z": pd.to_numeric(spots["Z"], errors="raise").astype(np.float64),
            "y": pd.to_numeric(spots["Y"], errors="raise").astype(np.float64),
            "x": pd.to_numeric(spots["X"], errors="raise").astype(np.float64),
        }))
    return pd.concat(parts, ignore_index=True)


def validate_lineage_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """Validate graph invariants and return an audit summary.

    The source datasets describe adjacent-frame lineage forests: node IDs are
    unique within a sequence, every edge advances exactly one frame, in-degree
    is at most one, and out-degree is at most two.
    """
    missing_nodes = set(NODE_KEY + ["t", "z", "y", "x"]) - set(nodes.columns)
    missing_edges = set(EDGE_KEY) - set(edges.columns)
    if missing_nodes or missing_edges:
        raise ValueError(f"missing canonical columns: nodes={sorted(missing_nodes)}, "
                         f"edges={sorted(missing_edges)}")
    if nodes.duplicated(NODE_KEY).any():
        raise ValueError("duplicate node IDs within an external sequence")
    if edges.duplicated(EDGE_KEY).any():
        raise ValueError("duplicate external edges")
    if not np.isfinite(nodes[["z", "y", "x"]].to_numpy()).all():
        raise ValueError("non-finite external node coordinates")

    lookup = nodes[["source", "sequence", "node_id", "t"]]
    joined = (edges.merge(
        lookup.rename(columns={"node_id": "source_id", "t": "source_t"}),
        on=["source", "sequence", "source_id"], how="left", validate="many_to_one")
        .merge(
            lookup.rename(columns={"node_id": "target_id", "t": "target_t"}),
            on=["source", "sequence", "target_id"], how="left", validate="many_to_one"))
    if joined[["source_t", "target_t"]].isna().any().any():
        raise ValueError("external edge references a missing node")
    dt = joined.target_t.to_numpy() - joined.source_t.to_numpy()
    if not np.all(dt == 1):
        bad = np.unique(dt[dt != 1]).tolist()
        raise ValueError(f"external edges are not adjacent-frame: dt={bad}")

    in_degree = edges.groupby(["source", "sequence", "target_id"]).size()
    out_degree = edges.groupby(["source", "sequence", "source_id"]).size()
    if int(in_degree.max()) > 1:
        raise ValueError("external graph contains a node with multiple parents")
    if int(out_degree.max()) > 2:
        raise ValueError("external graph contains an out-degree above two")

    by_sequence = []
    for (source, sequence), frame in nodes.groupby(["source", "sequence"], sort=True):
        seq_edges = edges[(edges.source == source) & (edges.sequence == sequence)]
        degree = seq_edges.groupby("source_id").size()
        by_sequence.append({
            "source": source,
            "sequence": sequence,
            "nodes": int(len(frame)),
            "edges": int(len(seq_edges)),
            "frames": int(frame.t.nunique()),
            "t_min": int(frame.t.min()),
            "t_max": int(frame.t.max()),
            "divisions": int((degree == 2).sum()),
        })
    return {
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "divisions": int((out_degree == 2).sum()),
        "max_in_degree": int(in_degree.max()),
        "max_out_degree": int(out_degree.max()),
        "sequences": by_sequence,
    }
