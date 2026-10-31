"""Canonical image/graph loaders for prepared external-data pilots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from .lineages import validate_lineage_graph


@dataclass(frozen=True)
class ExternalLineageSplit:
    """One frozen embryo-level graph split ready for model code."""

    name: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    divisions: pd.DataFrame

    @classmethod
    def open_celegans(
        cls,
        prepared_root: str | Path,
        split_contract: str | Path,
        name: str,
    ) -> "ExternalLineageSplit":
        prepared_root = Path(prepared_root)
        contract = json.loads(Path(split_contract).read_text(encoding="utf-8"))
        if name not in contract["splits"]:
            raise KeyError(f"unknown external split {name!r}")
        scope = contract["celegans_frame_scope"]
        t_min, t_max = int(scope["t_min"]), int(scope["t_max"])
        node_parts: list[pd.DataFrame] = []
        edge_parts: list[pd.DataFrame] = []
        division_parts: list[pd.DataFrame] = []
        for assignment in contract["splits"][name]:
            root = prepared_root / assignment["source"] / "canonical"
            nodes = pd.read_parquet(root / "nodes.parquet")
            edges = pd.read_parquet(root / "edges.parquet")
            divisions = pd.read_parquet(root / "divisions.parquet")
            sequence = assignment["sequence"]
            nodes = nodes[
                (nodes.sequence == sequence) & nodes.t.between(t_min, t_max)].copy()
            node_ids = set(nodes.node_id.astype(np.int64))
            edges = edges[
                (edges.sequence == sequence)
                & edges.source_id.isin(node_ids)
                & edges.target_id.isin(node_ids)].copy()
            divisions = divisions[
                (divisions.sequence == sequence)
                & divisions.parent_id.isin(node_ids)
                & divisions.child1_id.isin(node_ids)
                & divisions.child2_id.isin(node_ids)].copy()
            actual = (len(nodes), len(edges), len(divisions))
            expected = tuple(
                int(assignment[key]) for key in ("nodes", "edges", "divisions"))
            if actual != expected:
                raise ValueError(
                    f"split contract mismatch for {sequence}: expected {expected}, got {actual}")
            nodes["split"] = name
            edges["split"] = name
            divisions["split"] = name
            node_parts.append(nodes)
            edge_parts.append(edges)
            division_parts.append(divisions)
        nodes = pd.concat(node_parts, ignore_index=True)
        edges = pd.concat(edge_parts, ignore_index=True)
        divisions = pd.concat(division_parts, ignore_index=True)
        validate_lineage_graph(nodes, edges)
        return cls(name=name, nodes=nodes, edges=edges, divisions=divisions)

    def summary(self) -> dict[str, int | str | list[str]]:
        return {
            "split": self.name,
            "sequences": sorted(self.nodes.sequence.unique().tolist()),
            "nodes": int(len(self.nodes)),
            "edges": int(len(self.edges)),
            "divisions": int(len(self.divisions)),
        }


@dataclass(frozen=True)
class ExternalCrop:
    """One prepared TZYX crop and its canonical point-lineage graph."""

    root: Path
    images: zarr.Array
    nodes: pd.DataFrame
    edges: pd.DataFrame
    scale_um_zyx: tuple[float, float, float]
    annotation_quality: str

    @classmethod
    def open(cls, root: str | Path) -> "ExternalCrop":
        root = Path(root)
        images = zarr.open_array(str(root / "images.zarr"), mode="r")
        if images.ndim != 4:
            raise ValueError(f"external images must be TZYX, got shape {images.shape}")
        nodes = pd.read_parquet(root / "nodes.parquet")
        edges = pd.read_parquet(root / "edges.parquet")
        attrs = dict(images.attrs)
        scale = tuple(float(value) for value in attrs["scale_um_zyx"])
        quality = str(attrs["annotation_quality"])
        crop = cls(root, images, nodes, edges, scale, quality)
        crop.validate()
        return crop

    def validate(self) -> dict[str, int | float | str | list[int]]:
        node_required = {"node_id", "t", "z", "y", "x"}
        edge_required = {"source_id", "target_id"}
        if missing := node_required - set(self.nodes.columns):
            raise ValueError(f"external nodes are missing columns {sorted(missing)}")
        if missing := edge_required - set(self.edges.columns):
            raise ValueError(f"external edges are missing columns {sorted(missing)}")
        if self.nodes.node_id.duplicated().any():
            raise ValueError("external crop has duplicate node IDs")
        shape = np.asarray(self.images.shape, dtype=np.int64)
        coordinates = self.nodes[["t", "z", "y", "x"]].to_numpy(np.float64)
        if not np.isfinite(coordinates).all():
            raise ValueError("external crop has non-finite coordinates")
        inside = ((coordinates >= 0) & (coordinates < shape[None])).all(axis=1)
        if not inside.all():
            raise ValueError(f"{int((~inside).sum())} external nodes lie outside the crop")
        node_ids = set(self.nodes.node_id.astype(np.int64))
        referenced = set(self.edges.source_id.astype(np.int64)) | set(
            self.edges.target_id.astype(np.int64))
        if missing_ids := referenced - node_ids:
            raise ValueError(f"external edges reference {len(missing_ids)} missing nodes")
        time_by_id = self.nodes.set_index("node_id").t
        dt = (self.edges.target_id.map(time_by_id).to_numpy()
              - self.edges.source_id.map(time_by_id).to_numpy())
        if len(dt) and not np.all(dt == 1):
            raise ValueError("external crop contains non-adjacent edges")
        return {
            "shape_tzyx": list(map(int, self.images.shape)),
            "nodes": int(len(self.nodes)),
            "edges": int(len(self.edges)),
            "scale_um_zyx": list(self.scale_um_zyx),
            "annotation_quality": self.annotation_quality,
        }

    def five_frame_window(self, center_t: int) -> np.ndarray:
        if center_t < 2 or center_t + 2 >= self.images.shape[0]:
            raise IndexError(f"five-frame window is outside crop at t={center_t}")
        return np.asarray(self.images[center_t - 2:center_t + 3])

    def temporal_patch(
        self,
        t: int,
        z: int,
        y: int,
        x: int,
        *,
        time_radius: int = 1,
        radius_zyx: tuple[int, int, int] = (4, 8, 8),
    ) -> np.ndarray:
        rz, ry, rx = radius_zyx
        bounds = (
            t - time_radius, t + time_radius + 1,
            z - rz, z + rz + 1,
            y - ry, y + ry + 1,
            x - rx, x + rx + 1,
        )
        t0, t1, z0, z1, y0, y1, x0, x1 = bounds
        if min(t0, z0, y0, x0) < 0 or (
            t1 > self.images.shape[0]
            or z1 > self.images.shape[1]
            or y1 > self.images.shape[2]
            or x1 > self.images.shape[3]
        ):
            raise IndexError(f"patch falls outside crop: {bounds}")
        return np.asarray(self.images[t0:t1, z0:z1, y0:y1, x0:x1])
