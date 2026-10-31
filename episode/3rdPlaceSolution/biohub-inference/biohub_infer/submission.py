"""Competition CSV: write the per-dataset lineage graphs, re-read and check the written file."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from biohub_infer.core.graph import validate_graph

COLUMNS = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
PLACEHOLDER_NODE = (0, 32, 128, 128)   # an empty graph still lists its dataset: one node at t=0, no edges


def write_submission(out_path, results):
    """results: iterable of (sid, nodes (N,4) t,z,y,x, edges (M,2) node indices) -> number of rows."""
    idx = 0
    with Path(out_path).open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for sid, nodes, edges in results:
            if len(nodes) == 0:
                w.writerow([idx, sid, 'node', 1, *PLACEHOLDER_NODE, -1, -1])
                idx += 1
                continue
            for gi in range(len(nodes)):
                t, z, y, x = (int(v) for v in nodes[gi])
                w.writerow([idx, sid, 'node', gi + 1, t, z, y, x, -1, -1])
                idx += 1
            for i, j in edges:
                w.writerow([idx, sid, 'edge', -1, -1, -1, -1, -1, int(i) + 1, int(j) + 1])
                idx += 1
    return idx


def export_submission(graph_dir, destination):
    """Every `<sid>.npz` (nodes, edges) of graph_dir, sorted by sid, -> destination CSV (written atomically)."""
    graph_dir, destination = Path(graph_dir), Path(destination)
    files = sorted(graph_dir.glob('*.npz'))
    if not files:
        raise ValueError(f'no graphs in {graph_dir}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False, suffix='.csv.tmp') as stream:
            temporary = Path(stream.name)

        def graphs():
            for path in files:
                with np.load(path, allow_pickle=False) as cache:
                    nodes, edges = validate_graph(cache['nodes'], cache['edges'], branching=True)
                yield path.stem, nodes, edges

        rows = write_submission(temporary, graphs())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return rows


def validate_submission(out_path, shapes, ids):
    """Structural checks of the written file; shapes: {sid: (T, Z, Y, X)}. Returns row / node / edge counts."""
    with Path(out_path).open() as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    if header != COLUMNS:
        raise ValueError(f'header {header} != {COLUMNS}')
    seen_ids, node_ids, edges_by_ds, n_nodes, n_edges = set(), {}, {}, 0, 0
    for r in body:
        rid, ds, rt = r[0], r[1], r[2]
        if rid in seen_ids:
            raise ValueError(f'duplicate row id {rid}')
        seen_ids.add(rid)
        if ds not in shapes:
            raise ValueError(f'unknown dataset in submission: {ds}')
        T, Z, Y, X = shapes[ds]
        if rt == 'node':
            nid, t, z, y, x = (int(v) for v in r[3:8])
            known = node_ids.setdefault(ds, set())
            if nid in known:
                raise ValueError(f'duplicate node_id {nid} in {ds}')
            known.add(nid)
            if not (0 <= t < T and 0 <= z < Z and 0 <= y < Y and 0 <= x < X):
                raise ValueError(f'node out of volume in {ds}: t={t} zyx=({z},{y},{x})')
            n_nodes += 1
        elif rt == 'edge':
            edges_by_ds.setdefault(ds, []).append((int(r[8]), int(r[9])))
            n_edges += 1
        else:
            raise ValueError(f'unknown row_type {rt}')
    missing = [s for s in ids if s not in node_ids]
    if missing:
        raise ValueError(f'datasets absent from submission: {missing}')
    for ds, es in edges_by_ds.items():
        known = node_ids.get(ds, set())
        bad = [e for e in es if e[0] not in known or e[1] not in known]
        if bad:
            raise ValueError(f'{len(bad)} edges reference unknown node_ids in {ds}: {bad[:3]}')
        if len(set(es)) != len(es):
            raise ValueError(f'duplicate edges in {ds}')
    return {'rows': len(body), 'nodes': n_nodes, 'edges': n_edges}
