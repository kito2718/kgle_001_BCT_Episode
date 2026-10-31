"""V533 post-pass on the final V507 graph: contrast-gated add-back of long raw-candidate chains, then xy edge trim.

Step A (acq_01 movies only). Raw primary-detector candidates that sit more than 4.5 um from every V507 node of their
frame, within the first 1.5x-node-count ranks of that frame, are thinned (3 um, by score) and linked frame to frame
(greedy nearest pairs within 5 um) into chains. A chain of >= 3 nodes may attach to a V507 track end one frame before
it and a V507 track start one frame after it (greedy, 5 um). A chain is admitted when it has >= 9 nodes, its nearest
V507 node is closer than 8 um in at least one frame, and the median full-resolution contrast of its nodes exceeds 0.1
(core r 1.2 um mean minus 5-8 um shell median on the frame normalized by its 50th/99.5th percentiles). Admitted chains
that would share a V507 attachment node are resolved longest first.

Step B (all movies). Every node within 1.5 px of its frame's non-zero data box in y or x is deleted with its edges.

No labels, no learned models, no neural forward pass. Pure numpy/scipy/zarr; deterministic (stable sorts).
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

VERSION = "v533_addback_edge_v1"
S = np.array([1.625, 0.40625, 0.40625])
PARAMS = {
    "cluster": "acq_01",
    "hole_um": 4.5,
    "rank_mult": 1.5,
    "nms_um": 3.0,
    "link_gate_um": 5.0,
    "attach_gate_um": 5.0,
    "min_chain": 3,
    "min_nodes": 9,
    "near_max_um": 8.0,
    "min_contrast": 0.1,
    "core_um": 1.2,
    "shell_um": [5.0, 8.0],
    "norm_percentiles": [50.0, 99.5],
    "edge_px": 1.5,
}


def _ball(r_um: float, r0_um: float = 0.0) -> np.ndarray:
    rz, ry = int(np.ceil(r_um / S[0])), int(np.ceil(r_um / S[1]))
    zz, yy, xx = np.mgrid[-rz:rz + 1, -ry:ry + 1, -ry:ry + 1]
    d = np.sqrt((zz * S[0]) ** 2 + (yy * S[1]) ** 2 + (xx * S[2]) ** 2)
    m = (d <= r_um) & (d >= r0_um)
    return np.stack([zz[m], yy[m], xx[m]], 1)


def _gather(vol: np.ndarray, zyx: np.ndarray, offs: np.ndarray) -> np.ndarray:
    p = zyx[:, None, :] + offs[None]
    for k, n in enumerate(vol.shape):
        np.clip(p[..., k], 0, n - 1, out=p[..., k])
    return vol[p[..., 0], p[..., 1], p[..., 2]]


def _greedy_pairs(a_xyz: np.ndarray, b_xyz: np.ndarray, gate: float) -> list:
    if not len(a_xyz) or not len(b_xyz):
        return []
    sd = cKDTree(a_xyz * S).sparse_distance_matrix(cKDTree(b_xyz * S), gate, output_type="ndarray")
    used_a, used_b, pairs = set(), set(), []
    for i, j, d in sorted(sd.tolist(), key=lambda r: r[2]):
        i, j = int(i), int(j)
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j, d))
    return pairs


def _holes(ct, cxyz, cscore, crank, nt, nxyz, K, p):
    """Row indices of thinned hole candidates, frame-ascending, score-descending within a frame."""
    out = []
    for t in np.unique(ct):
        rows = np.flatnonzero(ct == t)
        rows = rows[crank[rows] < p["rank_mult"] * K.get(int(t), 0)]
        n = nxyz[nt == t]
        if len(rows) and len(n):
            d, _ = cKDTree(n * S).query(cxyz[rows] * S)
            rows = rows[d > p["hole_um"]]
        if not len(rows):
            continue
        rows = rows[np.argsort(-cscore[rows], kind="stable")]
        pts = cxyz[rows] * S
        tree = cKDTree(pts)
        keep, taken = [], np.zeros(len(rows), bool)
        for i in range(len(rows)):
            if taken[i]:
                continue
            keep.append(i)
            taken[tree.query_ball_point(pts[i], p["nms_um"])] = True
        out.append(rows[keep])
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def _chains(ht, hxyz, gate):
    idx = {int(t): np.flatnonzero(ht == t) for t in np.unique(ht)}
    succ = {}
    for t, a in idx.items():
        b = idx.get(t + 1)
        if b is None:
            continue
        for i, j, _ in _greedy_pairs(hxyz[a], hxyz[b], gate):
            succ[int(a[i])] = int(b[j])
    has_pred = set(succ.values())
    chains = []
    for s in range(len(ht)):
        if s in has_pred:
            continue
        c = [s]
        while c[-1] in succ:
            c.append(succ[c[-1]])
        chains.append(c)
    return chains


def proposals(ids, coords, src, dst, candidates, frames, shape, p=PARAMS):
    """Label-free add-back proposals. ids/coords sorted by node id; coords (N, 4) t, z, y, x."""
    ct = candidates["t"].to_numpy(np.int64)
    cxyz = candidates[["z", "y", "x"]].to_numpy(float)
    cscore = candidates["score"].to_numpy()
    crank = candidates["rank_in_frame"].to_numpy()
    nt = coords[:, 0]
    nxyz = coords[:, 1:].astype(float)
    K = {int(t): int(c) for t, c in zip(*np.unique(nt, return_counts=True))}
    h = _holes(ct, cxyz, cscore, crank, nt, nxyz, K, p)
    ht, hxyz = ct[h], cxyz[h]
    chains = _chains(ht, hxyz, p["link_gate_um"])
    has_out = np.isin(ids, src)
    has_in = np.isin(ids, dst)
    ends = np.flatnonzero(~has_out & (nt < frames - 1))
    starts = np.flatnonzero(~has_in & (nt > 0))
    head, tail = {}, {}
    firsts = np.array([ht[c[0]] for c in chains], np.int64) if chains else np.zeros(0, np.int64)
    lasts = np.array([ht[c[-1]] for c in chains], np.int64) if chains else np.zeros(0, np.int64)
    for t in np.unique(firsts):
        cs = [c for c in chains if ht[c[0]] == t]
        e = ends[nt[ends] == t - 1]
        for i, j, _ in _greedy_pairs(hxyz[[c[0] for c in cs]], nxyz[e], p["attach_gate_um"]):
            head[cs[i][0]] = int(ids[e[j]])
    for t in np.unique(lasts):
        cs = [c for c in chains if ht[c[-1]] == t]
        s = starts[nt[starts] == t + 1]
        for i, j, _ in _greedy_pairs(hxyz[[c[-1] for c in cs]], nxyz[s], p["attach_gate_um"]):
            tail[cs[i][-1]] = int(ids[s[j]])
    hi = np.array(shape, float) - 1
    spec, nid = [], int(ids.max()) + 1 if len(ids) else 0
    for c in chains:
        if len(c) < p["min_chain"]:
            continue
        nodes, members = [], []
        for r in c:
            q = np.clip(np.rint(hxyz[r]), 0, hi).astype(int)
            nodes.append((nid, int(ht[r]), int(q[0]), int(q[1]), int(q[2])))
            members.append(nid)
            nid += 1
        edges = [(a, b, 1.0) for a, b in zip(members[:-1], members[1:])]
        if c[0] in head:
            edges.append((head[c[0]], members[0], 1.0))
        if c[-1] in tail:
            edges.append((members[-1], tail[c[-1]], 1.0))
        spec.append({"nodes": nodes, "edges": edges})
    return spec


def _normalized(arr, t, p):
    vol = np.asarray(arr[int(t)], np.float32)
    lo, hi = np.percentile(vol, p["norm_percentiles"])
    return (vol - lo) / max(hi - lo, 1.0)


def select(spec, coords, arr, p=PARAMS):
    """Proposal indices passing the length, proximity and contrast gates (ascending), plus a small audit."""
    nt = coords[:, 0]
    trees = {int(t): cKDTree(coords[nt == t, 1:].astype(float) * S) for t in np.unique(nt)}
    near_ok = []
    for i, sp in enumerate(spec):
        if len(sp["nodes"]) < p["min_nodes"]:
            continue
        near = [trees[t].query(np.array([z, y, x], float) * S)[0] for _, t, z, y, x in sp["nodes"] if t in trees]
        if near and min(near) < p["near_max_um"]:
            near_ok.append(i)
    core, shell = _ball(p["core_um"]), _ball(p["shell_um"][1], p["shell_um"][0])
    by_t = {}
    for i in near_ok:
        for k, (_, t, z, y, x) in enumerate(spec[i]["nodes"]):
            if t in trees:
                by_t.setdefault(int(t), []).append((i, k, z, y, x))
    values = {i: np.full(len(spec[i]["nodes"]), np.nan) for i in near_ok}
    for t in sorted(by_t):
        rows = by_t[t]
        vol = _normalized(arr, t, p)
        zyx = np.array([[z, y, x] for _, _, z, y, x in rows], np.int64)
        c = _gather(vol, zyx, core).mean(1) - np.median(_gather(vol, zyx, shell), 1)
        for (i, k, *_), v in zip(rows, c):
            values[i][k] = v
    chosen = []
    for i in near_ok:
        v = values[i][np.isfinite(values[i])]
        if len(v) and float(np.median(v)) > p["min_contrast"]:
            chosen.append(i)
    audit = {"proposals": len(spec), "length_near_pass": len(near_ok), "contrast_frames": len(by_t),
             "selected": len(chosen)}
    return chosen, audit


def resolve(spec, chosen, ids, src, dst):
    """Longest first; skip proposals whose attachment would give a V507 node a second parent or child."""
    existing = set(int(i) for i in ids)
    ind = set(int(d) for d in dst)
    outd = set(int(s) for s in src)
    keep, used, got_parent, got_child = [], set(), set(), set()
    for i in sorted(chosen, key=lambda i: -len(spec[i]["nodes"])):
        ns = {n[0] for n in spec[i]["nodes"]}
        ok = not (ns & used) and not (ns & existing)
        for s, d, _ in spec[i]["edges"]:
            if d in existing and (d in ind or d in got_parent):
                ok = False
            if s in existing and (s in outd or s in got_child):
                ok = False
        if not ok:
            continue
        keep.append(i)
        used |= ns
        for s, d, _ in spec[i]["edges"]:
            if d in existing:
                got_parent.add(d)
            if s in existing:
                got_child.add(s)
    return keep


def edge_boxes(arr) -> dict:
    """Per frame: (y0, y1, x0, x1) of rows/columns holding any non-zero voxel (array bounds if none)."""
    boxes = {}
    for t in range(arr.shape[0]):
        v = np.asarray(arr[t])
        rows = np.flatnonzero(v.any(axis=(0, 2)))
        cols = np.flatnonzero(v.any(axis=(0, 1)))
        y0, y1 = (int(rows[0]), int(rows[-1])) if len(rows) else (0, v.shape[1] - 1)
        x0, x1 = (int(cols[0]), int(cols[-1])) if len(cols) else (0, v.shape[2] - 1)
        boxes[t] = (y0, y1, x0, x1)
    return boxes


def apply(graph, candidates, cluster: str, zarr_path, p: dict | None = None):
    """Return (new DatasetGraph, audit). `graph` is the final V507 graph; `candidates` the primary raw pool."""
    import zarr
    from biohub.io.submission import DatasetGraph, validate_graph

    p = dict(PARAMS if p is None else p)
    started = time.perf_counter()
    arr = zarr.open_array(str(Path(zarr_path) / "0"), mode="r")
    frames, shape = int(arr.shape[0]), tuple(int(s) for s in arr.shape[1:])
    raw_ids = np.fromiter(graph.nodes.keys(), np.int64, count=len(graph.nodes))
    raw_coords = np.array(list(graph.nodes.values()), np.int64).reshape(-1, 4)
    order = np.argsort(raw_ids, kind="stable")
    ids, coords = raw_ids[order], raw_coords[order]
    src = np.array([e[0] for e in graph.edges], np.int64)
    dst = np.array([e[1] for e in graph.edges], np.int64)
    audit = {"version": VERSION, "cluster": cluster, "addback_applied": cluster == p["cluster"]}
    added_nodes, added_edges = [], []
    if cluster == p["cluster"] and len(ids):
        spec = proposals(ids, coords, src, dst, candidates, frames, shape, p)
        chosen, sel_audit = select(spec, coords, arr, p)
        keep = resolve(spec, chosen, ids, src, dst)
        for i in keep:  # resolution order (longest first), as validated
            added_nodes.extend(spec[i]["nodes"])
            added_edges.extend(spec[i]["edges"])
        audit.update(sel_audit, kept=len(keep), added_nodes=len(added_nodes), added_edges=len(added_edges))
    else:
        audit.update(proposals=0, length_near_pass=0, contrast_frames=0, selected=0, kept=0, added_nodes=0,
                     added_edges=0)
    t_add = time.perf_counter()
    boxes = edge_boxes(arr)
    points = [(int(n), int(c[0]), int(c[2]), int(c[3])) for n, c in zip(raw_ids, raw_coords)]
    points += [(n[0], n[1], n[3], n[4]) for n in added_nodes]
    drop = set()
    for n, t, y, x in points:
        y0, y1, x0, x1 = boxes[t]
        if min(y - y0, y1 - y, x - x0, x1 - x) <= p["edge_px"]:
            drop.add(n)
    nodes = {n: xyz for n, xyz in graph.nodes.items() if n not in drop}
    for n in added_nodes:
        if n[0] not in drop:
            nodes[n[0]] = (n[1], n[2], n[3], n[4])
    edges = [e for e in graph.edges if e[0] not in drop and e[1] not in drop]
    edges += [e for e in added_edges if e[0] not in drop and e[1] not in drop]
    out = DatasetGraph(graph.dataset, nodes, edges, graph.shape)
    valid = validate_graph(out)
    audit.update(trimmed_nodes=len(drop), trimmed_edges=len(graph.edges) + len(added_edges) - len(edges),
                 nodes=len(nodes), edges=len(edges), forks=int(valid.get("forks", -1)) if isinstance(valid, dict) else -1,
                 seconds_addback=round(t_add - started, 3), seconds_total=round(time.perf_counter() - started, 3))
    return out, audit
