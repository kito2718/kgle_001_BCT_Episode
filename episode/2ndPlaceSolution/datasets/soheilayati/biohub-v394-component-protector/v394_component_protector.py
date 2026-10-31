"""Deployment-only short-component cleanup validated by V394 strict CV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from biohub.io.submission import DatasetGraph, validate_graph
from biohub.metrics.gt import SCALE_ARR
from biohub.metrics.matching import match_sequence

OFF = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3)
QUANTILES = ("0.0", "0.001", "0.01", "0.1", "0.9", "0.99", "0.999", "1.0")


def load_assets(root: str | Path):
    from lightgbm import Booster
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text())
    model = Booster(model_file=str(root / "component_protector.txt"))
    with np.load(root / "acquisition_router.npz", allow_pickle=False) as data:
        router = {key: data[key].copy() for key in data.files}
    return model, metadata, router


def acquisition_cluster(path: str | Path, router: dict) -> str:
    import zarr
    root = zarr.open_group(str(path), mode="r")
    dataset = root.attrs["multiscales"][0]["datasets"][0]
    array = root[dataset["path"]]
    transform = next(x["scale"] for x in dataset["coordinateTransformations"] if x["type"] == "scale")
    quantiles = root.attrs["image_statistics"]["quantiles"]
    q = {key: float(quantiles[key]) for key in QUANTILES}
    values = dict(frames=int(array.shape[0]), depth=int(array.shape[1]),
                  height=int(array.shape[2]), width=int(array.shape[3]),
                  scale_z=float(transform[1]), scale_y=float(transform[2]),
                  scale_x=float(transform[3]), dtype_bits=int(np.dtype(array.dtype).itemsize * 8))
    for key, value in q.items():
        values[f"log1p_q{key.replace('.', 'p')}"] = float(np.log1p(max(value, 0.0)))
    values.update(log1p_range_q99_q01=float(np.log1p(max(q["0.99"] - q["0.01"], 0.0))),
                  log1p_range_q999_q001=float(np.log1p(max(q["0.999"] - q["0.001"], 0.0))),
                  contrast_q90_q10=float((q["0.9"] + 1) / (q["0.1"] + 1)),
                  tail_q999_q99=float((q["0.999"] + 1) / (q["0.99"] + 1)))
    names = [str(x) for x in router["feature_names"]]
    vector = np.asarray([values[name] for name in names], np.float64)
    scaled = (vector - router["scaler_mean"]) / router["scaler_scale"]
    label = int(np.argmin(np.linalg.norm(router["canonical_centers"] - scaled, axis=1)))
    return f"acq_{label:02d}"


def _stats(prefix, values):
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {prefix + key: np.nan for key in ("_mean", "_std", "_min", "_max")}
    return {prefix + "_mean": float(finite.mean()), prefix + "_std": float(finite.std()),
            prefix + "_min": float(finite.min()), prefix + "_max": float(finite.max())}


def _patches(volume, coords):
    out = np.empty((len(coords), len(OFF)), np.float32)
    for t in np.unique(coords[:, 0]):
        take = np.flatnonzero(coords[:, 0] == t)
        centre = coords[take, 1:].astype(float); centre[:, 1:] /= 4
        grid = centre[:, None] + OFF[None]
        out[take] = map_coordinates(np.asarray(volume[int(t)], np.float32),
                                    grid.reshape(-1, 3).T, order=1, mode="nearest",
                                    prefilter=False).reshape(len(take), -1)
    return out


def _ncc(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return np.sum(a*b, 1) / np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-8)


def prune_components(graph: DatasetGraph, volume: np.ndarray,
                     witnesses: dict[str, DatasetGraph], model, metadata: dict,
                     cluster: str) -> tuple[DatasetGraph, dict]:
    ids = np.asarray(list(graph.nodes), np.int64)
    coords = np.asarray(list(graph.nodes.values()), np.int64)
    lookup = {int(v): i for i, v in enumerate(ids)}
    pi = np.asarray([lookup[int(a)] for a, _, _ in graph.edges], int)
    ci = np.asarray([lookup[int(b)] for _, b, _ in graph.edges], int)
    uf = np.arange(len(ids))
    def find(x):
        while uf[x] != x: uf[x] = uf[uf[x]]; x = int(uf[x])
        return x
    for a, b in zip(pi, ci):
        a, b = find(int(a)), find(int(b))
        if a != b: uf[b] = a
    roots = np.asarray([find(i) for i in range(len(ids))]); sizes = np.bincount(roots, minlength=len(ids))
    outdegree = np.bincount(pi, minlength=len(ids)); fork_roots = set(roots[outdegree == 2])
    selected_roots = np.asarray([r for r in np.flatnonzero((sizes >= 2) & (sizes <= metadata["max_component_nodes"]))
                                 if r not in fork_roots], int)
    selected_nodes = np.isin(roots, selected_roots); selected_edges = np.isin(roots[pi], selected_roots)
    xyz = coords[:, 1:] * SCALE_ARR
    nearest = np.full(len(ids), np.nan)
    for t in np.unique(coords[:, 0]):
        take = np.flatnonzero(coords[:, 0] == t)
        if len(take) > 1: nearest[take] = cKDTree(xyz[take]).query(xyz[take], k=2)[0][:, 1]
    patch = np.full((len(ids), len(OFF)), np.nan, np.float32)
    take = np.flatnonzero(selected_nodes)
    if len(take): patch[take] = _patches(volume, coords[take])
    displacement = np.linalg.norm(xyz[ci] - xyz[pi], axis=1)
    selected_edge_ids = np.flatnonzero(selected_edges)
    continuity = _ncc(patch[pi[selected_edges]], patch[ci[selected_edges]])
    support = np.zeros(len(graph.edges), np.int8)
    base_pairs = {(int(ids[a]), int(ids[b])): j for j, (a, b) in enumerate(zip(pi, ci))}
    for witness in witnesses.values():
        wids = np.asarray(list(witness.nodes), np.int64)
        wc = np.asarray(list(witness.nodes.values()), np.float64)
        mapping = match_sequence(coords[:, 0], coords[:, 1:],
                                 wc[:, 0].astype(int), wc[:, 1:], max_distance=3.0)
        donor_to_base = {int(wids[i]): int(ids[j]) for i, j in enumerate(mapping) if j >= 0}
        seen = set()
        for a, b, _ in witness.edges:
            if a in donor_to_base and b in donor_to_base:
                key = (donor_to_base[a], donor_to_base[b])
                if key in base_pairs: seen.add(base_pairs[key])
        if seen: support[list(seen)] += 1

    edge_continuity = {edge: continuity[k] for k, edge in enumerate(selected_edge_ids)}
    rows = []
    max_t = max(int(coords[:, 0].max()), 1)
    for root in selected_roots:
        ni = np.flatnonzero(roots == root); ei = np.flatnonzero(roots[pi] == root)
        t = coords[ni, 0]
        row = dict(component_nodes=len(ni), component_edges=len(ei),
                   span=int(t.max() - t.min() + 1), start_fraction=float(t.min()/max_t),
                   end_fraction=float(t.max()/max_t),
                   z_fraction=float(coords[ni, 1].mean()/max(coords[:, 1].max(), 1)),
                   y_fraction=float(coords[ni, 2].mean()/max(coords[:, 2].max(), 1)),
                   x_fraction=float(coords[ni, 3].mean()/max(coords[:, 3].max(), 1)))
        for prefix, values in (
            ("nearest_um", nearest[ni]), ("node_mean", patch[ni].mean(1)),
            ("node_std", patch[ni].std(1)), ("edge_um", displacement[ei]),
            ("edge_ncc", [edge_continuity[j] for j in ei]),
            ("witnesses", support[ei]),
        ):
            row.update(_stats(prefix, values))
        if len(ei) >= 2:
            order = ei[np.argsort(coords[pi[ei], 0])]
            vector = xyz[ci[order]] - xyz[pi[order]]
            turn = np.sum(vector[:-1]*vector[1:], 1) / np.maximum(
                np.linalg.norm(vector[:-1], axis=1)*np.linalg.norm(vector[1:], axis=1), 1e-8)
            row.update(_stats("turn", turn))
            row.update(_stats("acceleration_um", np.linalg.norm(np.diff(vector, axis=0), axis=1)))
        else:
            row.update(_stats("turn", [])); row.update(_stats("acceleration_um", []))
        row["is_acq01"] = float(cluster == "acq_01")
        rows.append(row)

    if rows:
        x = np.asarray([[row.get(name, np.nan) for name in metadata["feature_columns"]]
                        for row in rows], np.float32)
        probability = np.asarray(model.predict(x, num_threads=1), np.float64)
        threshold = float(metadata["protect_thresholds"][cluster])
        remove_roots = selected_roots[probability < threshold]
    else:
        probability = np.zeros(0); threshold = float(metadata["protect_thresholds"][cluster])
        remove_roots = np.zeros(0, int)
    used = np.zeros(len(ids), bool); used[pi] = True; used[ci] = True
    remove_node = ~used | np.isin(roots, remove_roots)
    remove_ids = set(ids[remove_node].tolist())
    result = DatasetGraph(
        graph.dataset,
        {node: value for node, value in graph.nodes.items() if node not in remove_ids},
        [edge for edge in graph.edges if edge[0] not in remove_ids and edge[1] not in remove_ids],
        graph.shape,
    )
    audit = validate_graph(result)
    audit.update(acquisition_cluster=cluster, threshold=threshold,
                 candidates=len(selected_roots), removed_components=len(remove_roots),
                 edgeless_nodes=int((~used).sum()), removed_nodes=int(remove_node.sum()),
                 probability_min=float(probability.min()) if len(probability) else None,
                 probability_max=float(probability.max()) if len(probability) else None)
    return result, audit
