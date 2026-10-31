"""Prediction-only features and constrained ordinary-cell correspondence."""
from collections import Counter

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from biohub.io.submission import DatasetGraph, validate_graph
from biohub.detector.targets import src_to_grid
from biohub.pipeline.contested_pair import neighborhood

SCALE = np.asarray([1.625, .40625, .40625], np.float64)
OFFSETS = np.stack(np.meshgrid(*[np.arange(-4, 5)] * 3, indexing='ij'), axis=-1).reshape(-1, 3)
CENTER = (np.abs(OFFSETS) <= 2).all(axis=1)


def inventory(graph, motion_ids, motion_nodes, backward, forward):
    ids = np.asarray(list(graph.nodes), np.int64)
    coords = np.asarray(list(graph.nodes.values()), np.float64)
    lookup = {int(n): i for i, n in enumerate(ids)}
    n = len(ids)
    back, ahead = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    valid = np.zeros(n, bool)
    for j, node in enumerate(motion_ids):
        i = lookup.get(int(node))
        if i is not None:
            if not np.array_equal(coords[i], motion_nodes[j]):
                raise RuntimeError('motion coordinate does not match final node')
            back[i], ahead[i], valid[i] = backward[j], forward[j], True
    base_edges = np.asarray([(lookup[p], lookup[c]) for p, c, _ in graph.edges], np.int64).reshape(-1, 2)
    outdegree = np.bincount(base_edges[:, 0], minlength=n)
    forbidden = neighborhood(graph, ids[outdegree == 2], 2)
    # A missing field cannot silently turn an incumbent edge into a movable one.
    invalid = set(ids[~valid].tolist())
    forbidden.update(neighborhood(graph, invalid, 1))
    allowed = np.asarray([int(i) not in forbidden for i in ids]) & valid & (outdegree <= 1)
    movable = allowed[base_edges[:, 0]] & allowed[base_edges[:, 1]]
    # Lock endpoints of every fixed edge so candidate assignment cannot violate capacity.
    parent_ok, child_ok = allowed.copy(), allowed.copy()
    parent_ok[base_edges[~movable, 0]] = False
    child_ok[base_edges[~movable, 1]] = False
    return dict(ids=ids, coords=coords, xyz=coords[:, 1:] * SCALE,
        times=coords[:, 0].astype(int), backward=back, forward=ahead,
        base_edges=base_edges, movable=movable, parent_ok=parent_ok, child_ok=child_ok)


def candidates(st):
    xyz, times = st['xyz'], st['times']
    pairs = [st['base_edges'][st['movable']]]
    for t in np.unique(times):
        parents = np.flatnonzero((times == t) & st['parent_ok'])
        children = np.flatnonzero((times == t + 1) & st['child_ok'])
        if not len(parents) or not len(children):
            continue
        for reverse in (False, True):
            source, target = (children, parents) if reverse else (parents, children)
            field = st['backward'] if reverse else st['forward']
            distance, index = cKDTree(xyz[target]).query(xyz[source] + field[source], k=4, distance_upper_bound=10.)
            si = np.broadcast_to(source[:, None], index.shape)
            mask = np.isfinite(distance) & (index < len(target))
            a, b = si[mask], target[index[mask]]
            pairs.append(np.column_stack([b, a] if reverse else [a, b]))
    pairs = np.unique(np.concatenate(pairs), axis=0)
    return pairs.astype(np.int32)


def patches(volume, coords, pitch=1):
    result = np.empty((len(coords), len(OFFSETS)), np.float32)
    for frame in np.unique(coords[:, 0].astype(int)):
        rows = np.flatnonzero(coords[:, 0] == frame)
        for start in range(0, len(rows), 512):
            use = rows[start:start + 512]
            grid = src_to_grid(coords[use, 1:])[:, None] + pitch * OFFSETS[None]
            result[use] = map_coordinates(np.asarray(volume[frame], np.float32),
                grid.reshape(-1, 3).T, order=1, mode='nearest', prefilter=False).reshape(len(use), -1)
    return result


def unit_patch(values):
    centered = values - values.mean(axis=1, keepdims=True)
    return centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)


def features(st, pairs, volume, raw_probabilities=None, image_cache=None):
    p, c = pairs.T
    n = len(st['ids'])
    xyz, back, ahead = st['xyz'], st['backward'], st['forward']
    delta = xyz[c] - xyz[p]
    values = {}
    def put(key, value):
        values['f_' + key] = np.asarray(value, np.float32)
    def norm(key, value):
        put(key, np.linalg.norm(value, axis=1))
    norm('distance', delta)
    norm('back_residual', delta + back[c])
    norm('forward_residual', delta - ahead[p])
    norm('flow_disagreement', ahead[p] + back[c])
    norm('parent_motion_change', ahead[p] + back[p])
    norm('child_motion_change', ahead[c] + back[c])
    for side, vector in [('back', back[c]), ('forward', ahead[p])]:
        norm(side + '_speed', vector)
        put(side + '_alignment', np.sum(delta * vector, axis=1) /
            np.maximum(np.linalg.norm(delta, axis=1) * np.linalg.norm(vector, axis=1), 1e-8))
    base_code = st['base_edges'][:, 0].astype(np.int64) * n + st['base_edges'][:, 1]
    put('incumbent', np.isin(p.astype(np.int64) * n + c, base_code))
    parent = np.full(n, -1, np.int64)
    successor = np.full(n, -1, np.int64)
    parent[st['base_edges'][:, 1]] = st['base_edges'][:, 0]
    successor[st['base_edges'][:, 0]] = st['base_edges'][:, 1]
    put('parent_has_successor', successor[p] >= 0)
    put('child_has_parent', parent[c] >= 0)
    hist, future = p.copy(), c.copy()
    valid_h, valid_f = np.ones(len(p), bool), np.ones(len(p), bool)
    for depth in range(1, 5):
        previous, following = parent[np.maximum(hist, 0)], successor[np.maximum(future, 0)]
        valid_h &= previous >= 0
        valid_f &= following >= 0
        hist, future = np.where(valid_h, previous, 0), np.where(valid_f, following, 0)
        if depth in (1, 2, 4):
            h = (xyz[p] - xyz[hist]) / depth
            f = (xyz[future] - xyz[c]) / depth
            for label, v, valid in [('history', h, valid_h), ('future', f, valid_f)]:
                put(f'{label}{depth}_valid', valid)
                residual = np.linalg.norm(delta - v, axis=1)
                put(f'{label}{depth}_residual', np.where(valid, residual, np.nan))
            put(f'trajectory{depth}_disagreement', np.where(valid_h & valid_f, np.linalg.norm(h-f, axis=1), np.nan))
    if raw_probabilities is not None:
        for j in range(raw_probabilities.shape[1]):
            put('original_probability_' + str(j), raw_probabilities[:, j])
    # Patch extraction is limited to participating nodes; coordinates remain fixed.
    participating = np.unique(pairs)
    remap = np.full(n, -1, np.int32)
    remap[participating] = np.arange(len(participating))
    if image_cache is None:
        local = patches(volume, st['coords'][participating])
        blurred = gaussian_filter(np.asarray(volume, np.float32), sigma=(0., .5, .5, .5), mode='nearest')
        wide = patches(blurred, st['coords'][participating], pitch=2)
    else:
        local, wide = image_cache
        if local.shape != (len(participating), len(OFFSETS)) or wide.shape != local.shape:
            raise ValueError('participating image cache shape differs')
    for scale, patch in [('local5', local[:, CENTER]), ('local9', local), ('wide9', wide)]:
        patch = unit_patch(patch)
        result = np.empty(len(pairs), np.float32)
        for start in range(0, len(pairs), 4096):
            stop = min(start + 4096, len(pairs))
            result[start:stop] = np.einsum('ij,ij->i', patch[remap[p[start:stop]]], patch[remap[c[start:stop]]])
        put('ncc_' + scale, result)
    for scale, patch in [('local', local), ('wide', wide)]:
        means, stds = patch.mean(axis=1), patch.std(axis=1)
        for label, scalar in [('mean', means), ('std', stds)]:
            a, b = scalar[remap[p]], scalar[remap[c]]
            put(f'{scale}_{label}_difference', np.abs(a - b))
            put(f'{scale}_{label}_relative', np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-6))
    for key in ('ncc_local5', 'ncc_local9', 'ncc_wide9', 'back_residual', 'forward_residual'):
        value = values['f_' + key]
        desirability = -value if 'residual' in key else value
        for side, group in [('source', p), ('target', c)]:
            best = np.full(n, -np.inf, np.float32)
            np.maximum.at(best, group, desirability)
            put(key + '_' + side + '_deficit', best[group] - desirability)
    frame = pd.DataFrame(dict(pi=p, ci=c, **values))
    if np.isinf(frame.to_numpy()).any():
        raise RuntimeError('nonfinite correspondence feature')
    return frame


def assign(st, frame, probability, bonus):
    n = len(st['ids'])
    p, c = frame[['pi', 'ci']].to_numpy(np.int64).T
    incumbent = frame.f_incumbent.to_numpy(bool)
    if bonus == 'keep':
        return np.flatnonzero(incumbent)
    q = np.clip(np.asarray(probability, np.float64), 1e-6, 1-1e-6)
    utility = np.log(q) - np.log1p(-q) + float(bonus) * incumbent
    utility += incumbent * 1e-9
    chosen = []
    for t in np.unique(st['times'][p]):
        rows = np.flatnonzero(st['times'][p] == t)
        ps, cs = np.unique(p[rows]), np.unique(c[rows])
        pr, cr = np.searchsorted(ps, p[rows]), np.searchsorted(cs, c[rows])
        # All matched rows receive the same shift, so dummy and real utilities compare exactly.
        shift = max(1., 1. - float(utility[rows].min()))
        rr = np.concatenate([cr, np.arange(len(cs))])
        cc = np.concatenate([pr, len(ps) + np.arange(len(cs))])
        weights = np.concatenate([utility[rows] + shift, np.full(len(cs), shift)])
        matrix = coo_matrix((weights, (rr, cc)), shape=(len(cs), len(ps) + len(cs))).tocsr()
        a, b = min_weight_full_bipartite_matching(matrix, maximize=True)
        real = b < len(ps)
        desired = ps[b[real]] * n + cs[a[real]]
        code = p[rows] * n + c[rows]
        chosen.extend(rows[np.isin(code, desired)].tolist())
    return np.asarray(sorted(chosen), np.int64)


def materialize(base, st, frame, chosen):
    ids = st['ids']
    pairs = frame[['pi', 'ci']].to_numpy(np.int64)
    new = {(int(ids[p]), int(ids[c])) for p, c in pairs[chosen]}
    movable = {(int(ids[p]), int(ids[c])) for p, c in st['base_edges'][st['movable']]}
    old = {(p, c): weight for p, c, weight in base.edges}
    desired = (set(old) - movable) | new
    graph = DatasetGraph(base.dataset, dict(base.nodes),
        [(p, c, old.get((p, c), 1.)) for p, c in sorted(desired)], base.shape)
    audit = validate_graph(graph)
    old_forks = {p for p, count in Counter(p for p, c in old).items() if count == 2}
    new_forks = {p for p, count in Counter(p for p, c in desired).items() if count == 2}
    if old_forks != new_forks or graph.nodes != base.nodes or not (set(old) - movable).issubset(desired):
        raise RuntimeError('protected graph invariant failed')
    audit.update(added_edges=len(desired-set(old)), removed_edges=len(set(old)-desired),
        nodes_coordinates_forks_exact=True, protected_edges_exact=True)
    return graph, audit
