"""Competition track graph, Zarr-v3 movie reader, and submission writer."""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)


@dataclass
class ImageVolume:
    path: str
    shape: tuple
    dtype: np.dtype
    chunk: tuple | None

    @property
    def n_t(self) -> int:
        return int(self.shape[0])

    def frame(self, t: int) -> np.ndarray:
        """Return the (Z, Y, X) volume for timepoint t."""
        return _read_chunk(self.path, t, self.shape, self.dtype)


def open_image(zarr_path: str) -> ImageVolume:
    with open(os.path.join(zarr_path, "0", "zarr.json")) as f:
        meta = json.load(f)
    shape = tuple((int(s) for s in meta["shape"]))
    dtype = np.dtype(meta["data_type"])
    chunk = None
    cg = meta.get("chunk_grid", {})
    conf = cg.get("configuration", {})
    if "chunk_shape" in conf:
        chunk = tuple((int(s) for s in conf["chunk_shape"]))
    return ImageVolume(path=zarr_path, shape=shape, dtype=dtype, chunk=chunk)


_BLOSC2 = None


def _blosc2():
    global _BLOSC2
    if _BLOSC2 is None:
        import blosc2

        _BLOSC2 = blosc2
    return _BLOSC2


def _read_chunk(zarr_path: str, t: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Read and decode one timepoint chunk -> (Z, Y, X)."""
    frame_shape = shape[1:]
    chunk_path = os.path.join(zarr_path, "0", "c", str(t), "0", "0", "0")
    with open(chunk_path, "rb") as f:
        raw = f.read()
    try:
        dec = _blosc2().decompress(raw)
        arr = np.frombuffer(dec, dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            return arr.reshape(frame_shape).copy()
    except Exception:
        pass
    import zarr

    z = zarr.open(os.path.join(zarr_path, "0"), mode="r")
    return np.asarray(z[t])


@dataclass
class TrackGraph:
    """A tracking graph: node coords + directed edges."""

    node_t: np.ndarray
    node_z: np.ndarray
    node_y: np.ndarray
    node_x: np.ndarray
    node_ids: np.ndarray
    edges: np.ndarray
    meta: dict

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def coords_by_id(self) -> dict:
        out = {}
        for i, nid in enumerate(self.node_ids):
            out[int(nid)] = (int(self.node_t[i]), float(self.node_z[i]), float(self.node_y[i]), float(self.node_x[i]))
        return out


def read_geff(geff_path: str) -> TrackGraph:
    """Read a .geff ground-truth graph directly from its zarr arrays."""
    import zarr

    g = zarr.open(geff_path, mode="r")

    def _arr(path):
        return np.asarray(g[path][:])

    node_ids = _arr("nodes/ids").astype(np.int64)
    t = _arr("nodes/props/t/values").astype(np.int64)
    z = _arr("nodes/props/z/values").astype(np.float64)
    y = _arr("nodes/props/y/values").astype(np.float64)
    x = _arr("nodes/props/x/values").astype(np.float64)
    edges = _arr("edges/ids").astype(np.int64)
    if edges.ndim == 1:
        edges = edges.reshape(-1, 2)
    meta = {}
    try:
        with open(os.path.join(geff_path, "zarr.json")) as f:
            zj = json.load(f)
        geff_meta = zj.get("attributes", {}).get("geff", {})
        extra = geff_meta.get("extra", {}) or {}
        meta = dict(geff_meta)
        if "estimated_number_of_nodes" in extra:
            meta["estimated_number_of_nodes"] = extra["estimated_number_of_nodes"]
    except Exception:
        pass
    return TrackGraph(node_t=t, node_z=z, node_y=y, node_x=x, node_ids=node_ids, edges=edges, meta=meta)


def list_datasets(root: str, kind: str = "train") -> list:
    """List dataset base-names (without .zarr) under root/<kind>."""
    d = os.path.join(root, kind)
    if not os.path.isdir(d):
        d = root
    names = sorted((n[:-5] for n in os.listdir(d) if n.endswith(".zarr")))
    return names


def embryo_id(dataset_name: str) -> str:
    """Folder names are {embryo_id}_{field_of_view}; embryo is the first segment."""
    return dataset_name.split("_")[0]


def _span_displacements(coords, succ, t, span):
    """``(starts_um, disp_um)`` for every node at ``t`` whose track runs ``span`` frames forward with exactly one successor per step (no division, no gap): its position at ``t`` and its displacement to ``t + span``, both in um."""
    pos, disp = ([], [])
    for nid, (tt, z, y, x) in coords.items():
        if tt != t:
            continue
        cur = nid
        for _ in range(span):
            nxt = succ.get(cur)
            if nxt is None or len(nxt) != 1:
                cur = None
                break
            cur = nxt[0]
        if cur is None or coords[cur][0] != t + span:
            continue
        p0 = np.array([z, y, x]) * SCALE
        pos.append(p0)
        disp.append(np.array(coords[cur][1:]) * SCALE - p0)
    if not pos:
        return (np.zeros((0, 3)), np.zeros((0, 3)))
    return (np.stack(pos), np.stack(disp))


def close_gaps(
    frames: list,
    g: TrackGraph,
    max_gap: int = 1,
    gap_dist_um: float = 8.0,
    drift_radius_um: float = 0.0,
    drift_min_neighbours: int = 3,
) -> TrackGraph:
    """Insert interpolated nodes to bridge single-frame detection gaps."""
    if g.n_edges == 0:
        return g
    coords = {int(nid): (int(g.node_t[i]), g.node_z[i], g.node_y[i], g.node_x[i]) for i, nid in enumerate(g.node_ids)}
    has_out = set((int(s) for s, _ in g.edges))
    has_in = set((int(t) for _, t in g.edges))
    ends_by_t = {}
    starts_by_t = {}
    for nid, (t, z, y, x) in coords.items():
        if nid not in has_out:
            ends_by_t.setdefault(t, []).append(nid)
        if nid not in has_in:
            starts_by_t.setdefault(t, []).append(nid)
    succ = {}
    for s_, t_ in g.edges:
        succ.setdefault(int(s_), []).append(int(t_))
    new_nodes = []
    new_edges = []
    next_id = int(g.node_ids.max()) + 1 if g.n_nodes else 1
    for gap in range(1, max_gap + 1):
        for t, ends in ends_by_t.items():
            starts = starts_by_t.get(t + gap + 1)
            if not starts:
                continue
            ec = np.array([[coords[e][1], coords[e][2], coords[e][3]] for e in ends]) * SCALE
            sc = np.array([[coords[s][1], coords[s][2], coords[s][3]] for s in starts]) * SCALE
            if drift_radius_um > 0:
                npos, ndisp = _span_displacements(coords, succ, t, gap + 1)
                if len(ndisp):
                    glob = np.median(ndisp, axis=0)
                    dn = np.sqrt(((ec[:, None, :] - npos[None, :, :]) ** 2).sum(axis=2))
                    shift = np.empty_like(ec)
                    for i in range(len(ec)):
                        near = dn[i] <= drift_radius_um
                        shift[i] = np.median(ndisp[near], axis=0) if int(near.sum()) >= drift_min_neighbours else glob
                    ec = ec + shift
            d = np.sqrt(((ec[:, None, :] - sc[None, :, :]) ** 2).sum(axis=2))
            thr = gap_dist_um * (gap + 1)
            big = thr * 1000 + 1
            cost = np.where(d <= thr, d, big)
            ri, ci = linear_sum_assignment(cost)
            used_s = set()
            for r, c in zip(ri, ci):
                if d[r, c] > thr or ends[r] in has_out or starts[c] in used_s:
                    continue
                e_id, s_id = (ends[r], starts[c])
                te, ze, ye, xe = coords[e_id]
                ts, zs, ys, xs = coords[s_id]
                prev = e_id
                for k in range(1, gap + 1):
                    frac = k / (gap + 1)
                    zi = ze + (zs - ze) * frac
                    yi = ye + (ys - ye) * frac
                    xi = xe + (xs - xe) * frac
                    nid = next_id
                    next_id += 1
                    new_nodes.append((te + k, zi, yi, xi, nid))
                    new_edges.append((prev, nid))
                    prev = nid
                new_edges.append((prev, s_id))
                has_out.add(e_id)
                used_s.add(c)
    if not new_nodes:
        return g
    nt = np.concatenate([g.node_t, np.array([n[0] for n in new_nodes], dtype=np.int64)])
    nz = np.concatenate([g.node_z, np.array([n[1] for n in new_nodes])])
    ny = np.concatenate([g.node_y, np.array([n[2] for n in new_nodes])])
    nx = np.concatenate([g.node_x, np.array([n[3] for n in new_nodes])])
    nid = np.concatenate([g.node_ids, np.array([n[4] for n in new_nodes], dtype=np.int64)])
    edges = np.concatenate([g.edges, np.array(new_edges, dtype=np.int64).reshape(-1, 2)])
    return TrackGraph(node_t=nt, node_z=nz, node_y=ny, node_x=nx, node_ids=nid, edges=edges, meta=g.meta)


def prune_isolated(g: TrackGraph) -> TrackGraph:
    """Remove nodes not referenced by any edge."""
    if g.n_edges == 0:
        return g
    used = set((int(x) for x in g.edges.reshape(-1)))
    keep = np.array([i for i, nid in enumerate(g.node_ids) if int(nid) in used])
    if len(keep) == len(g.node_ids):
        return g
    return TrackGraph(
        node_t=g.node_t[keep],
        node_z=g.node_z[keep],
        node_y=g.node_y[keep],
        node_x=g.node_x[keep],
        node_ids=g.node_ids[keep],
        edges=g.edges,
        meta=g.meta,
    )


def graph_to_rows(name: str, g: TrackGraph) -> list:
    rows = []
    for i in range(g.n_nodes):
        rows.append(
            {
                "dataset": name,
                "row_type": "node",
                "node_id": int(g.node_ids[i]),
                "t": int(g.node_t[i]),
                "z": int(round(g.node_z[i])),
                "y": int(round(g.node_y[i])),
                "x": int(round(g.node_x[i])),
                "source_id": -1,
                "target_id": -1,
            }
        )
    for s, t in g.edges:
        rows.append(
            {
                "dataset": name,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": int(s),
                "target_id": int(t),
            }
        )
    return rows


def write_submission(all_rows: list, path: str) -> pd.DataFrame:
    df = pd.DataFrame(
        all_rows, columns=["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    )
    df.index.name = "id"
    df.to_csv(path)
    return df
