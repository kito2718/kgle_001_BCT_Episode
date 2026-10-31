"""Close short gaps using fixed two-sided motion and existing detector peaks."""
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

from biohub.io.submission import DatasetGraph, validate_graph
from biohub.metrics.gt import SCALE_ARR
from biohub.tracker.postpass33 import changed_hop_guard


RADIUS_UM = 3.0
MAX_MISSING = 2


def propose_gaps(base, ids, backward, forward, candidates):
    """One fixed policy for adjacent ends and gaps of one or two missing cells.

    Both flanks have at least three consecutive nodes and lie outside the
    original two-hop fork guard. Every step is checked by backward motion;
    the first is independently checked by forward motion. Intermediate peaks
    must be spatially novel to the fixed baseline at the same frame.
    """
    ids = np.asarray(ids, np.int64)
    if not np.array_equal(ids, np.asarray(list(base.nodes), np.int64)):
        raise ValueError('motion and graph node order differ')
    nodes = np.asarray(list(base.nodes.values()), np.int64).reshape(-1, 4)
    backward, forward = np.asarray(backward, np.float64), np.asarray(forward, np.float64)
    if backward.shape != (len(ids), 3) or forward.shape != backward.shape:
        raise ValueError('motion vectors must align with all graph nodes')
    if not np.isfinite(backward).all() or not np.isfinite(forward).all():
        raise ValueError('motion must be finite')
    remap = {int(nid): row for row, nid in enumerate(ids)}
    edges = np.asarray([(remap[p], remap[c]) for p, c, _ in base.edges], np.int64).reshape(-1, 2)
    times, positions = nodes[:, 0], nodes[:, 1:] * SCALE_ARR
    n = len(ids)
    if not n:
        return [], {'proposals': 0, 'eligible_terminals': 0, 'eligible_roots': 0}
    pi, ci = edges.T
    indegree, outdegree = np.bincount(ci, minlength=n), np.bincount(pi, minlength=n)
    prev, nxt = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    prev[ci], nxt[pi] = pi, ci
    guard = changed_hop_guard(n, pi, ci, np.arange(len(edges)), hops=2)
    def supported(index, direction):
        start = index
        for step in range(2):
            index = direction[index]
            if index < 0 or abs(int(times[index]) - int(times[start])) != step + 1:
                return False
        return True
    terminals = [i for i in range(n) if outdegree[i] == 0 and not guard[i] and supported(i, prev)]
    roots = [i for i in range(n) if indegree[i] == 0 and not guard[i] and supported(i, nxt)]
    roots_by_t = defaultdict(list)
    for row in roots:
        roots_by_t[int(times[row])].append(row)
    root_trees = {t: (np.array(rows, np.int64), cKDTree(positions[rows] + backward[rows]))
                  for t, rows in roots_by_t.items()}
    raw_coord = candidates[['z', 'y', 'x']].to_numpy(np.float64)
    if not np.isfinite(raw_coord).all():
        raise ValueError('candidate coordinates must be finite')
    coord = np.rint(raw_coord).astype(np.int64)
    if base.shape is None:
        raise ValueError('source movie shape is required for coordinate bounds')
    coord = np.clip(coord, 0, np.asarray(base.shape[1:]) - 1)
    ctime = candidates.t.to_numpy(np.int64)
    cpos = coord * SCALE_ARR
    cback = candidates[['flow_z', 'flow_y', 'flow_x']].to_numpy(np.float64)
    if not np.isfinite(cback).all():
        raise ValueError('candidate motion must be finite')
    pools = {}
    for t in np.unique(ctime):
        rows = np.flatnonzero(ctime == t)
        old = positions[times == t]
        if len(old):
            distance, _ = cKDTree(old).query(cpos[rows], k=1)
            rows = rows[distance > RADIUS_UM]
        if len(rows):
            pools[int(t)] = (rows, cKDTree(cpos[rows]), cKDTree(cpos[rows] + cback[rows]))
    proposals = []
    for parent in terminals:
        t = int(times[parent])
        predicted = positions[parent] + forward[parent]
        # A zero-missing path checks both motion directions directly.
        if t + 1 in root_trees:
            rrows, tree = root_trees[t + 1]
            for j in tree.query_ball_point(positions[parent], RADIUS_UM):
                child = int(rrows[j])
                f = float(np.linalg.norm(positions[child] - predicted))
                b = float(np.linalg.norm(positions[child] + backward[child] - positions[parent]))
                if f <= RADIUS_UM:
                    proposals.append((.5 * (f*f + b*b), parent, child, ()))
        if t + 1 not in pools:
            continue
        rows, tree, _ = pools[t + 1]
        paths = []
        for j in tree.query_ball_point(predicted, RADIUS_UM):
            row = int(rows[j])
            b = float(np.linalg.norm(cpos[row] + cback[row] - positions[parent]))
            if b <= RADIUS_UM:
                f = float(np.linalg.norm(cpos[row] - predicted))
                paths.append(((row,), f*f + b*b))
        for missing in range(1, MAX_MISSING + 1):
            if t + missing + 1 in root_trees:
                rrows, tree = root_trees[t + missing + 1]
                for path, cost in paths:
                    for j in tree.query_ball_point(cpos[path[-1]], RADIUS_UM):
                        child = int(rrows[j])
                        b = float(np.linalg.norm(positions[child] + backward[child] - cpos[path[-1]]))
                        proposals.append(((cost + b*b) / (missing + 2), parent, child, path))
            if missing == MAX_MISSING or t + missing + 1 not in pools:
                break
            rows, _, tree = pools[t + missing + 1]
            extended = []
            for path, cost in paths:
                for j in tree.query_ball_point(cpos[path[-1]], RADIUS_UM):
                    row = int(rows[j])
                    b = float(np.linalg.norm(cpos[row] + cback[row] - cpos[path[-1]]))
                    extended.append((path + (row,), cost + b*b))
            paths = extended
    proposals.sort(key=lambda a: (a[0], len(a[3]), int(ids[a[1]]), int(ids[a[2]]), a[3]))
    best_parent, best_child = {}, {}
    for index, (_, parent, child, _) in enumerate(proposals):
        best_parent.setdefault(parent, index)
        best_child.setdefault(child, index)
    actions, used_candidates, added_by_t = [], set(), defaultdict(list)
    for index, (cost, parent, child, path) in enumerate(proposals):
        if best_parent[parent] != index or best_child[child] != index or any(r in used_candidates for r in path):
            continue
        if any(any(np.linalg.norm(cpos[r] - p) <= RADIUS_UM for p in added_by_t[int(ctime[r])]) for r in path):
            continue
        actions.append({'parent': int(ids[parent]), 'child': int(ids[child]),
            'mean_squared_residual_um2': float(cost), 'candidate_rows': list(path),
            'nodes': [(int(ctime[r]), *map(int, coord[r])) for r in path]})
        used_candidates.update(path)
        for row in path:
            added_by_t[int(ctime[row])].append(cpos[row])
    return actions, {'proposals': len(proposals), 'eligible_terminals': len(terminals),
        'eligible_roots': len(roots), 'selected_by_missing_cells': dict(Counter(len(a['nodes']) for a in actions))}


def materialize_gaps(base, actions):
    nodes, edges = dict(base.nodes), list(base.edges)
    next_id = max(nodes, default=-1) + 1
    for action in actions:
        chain = [int(action['parent'])]
        for point in action['nodes']:
            nodes[next_id] = tuple(map(int, point))
            chain.append(next_id)
            next_id += 1
        chain.append(int(action['child']))
        edges.extend((a, b, .30) for a, b in zip(chain[:-1], chain[1:]))
    result = DatasetGraph(base.dataset, nodes, edges, base.shape)
    audit = validate_graph(result)
    old_forks = {p for p, count in Counter(p for p, _, _ in base.edges).items() if count == 2}
    new_forks = {p for p, count in Counter(p for p, _, _ in edges).items() if count == 2}
    if old_forks != new_forks or {e for e in base.edges if e[0] in old_forks} != {e for e in edges if e[0] in new_forks}:
        raise RuntimeError('existing forks changed')
    if any(result.nodes[k] != v for k, v in base.nodes.items()) or result.edges[:len(base.edges)] != base.edges:
        raise RuntimeError('incumbent node or edge changed')
    return result, audit
