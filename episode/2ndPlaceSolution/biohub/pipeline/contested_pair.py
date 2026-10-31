"""Complete contested-fork proposals using a fixed primary edge universe."""
from collections import defaultdict

import numpy as np
import pandas as pd

from biohub.io.submission import DatasetGraph, validate_graph

SCALE = np.asarray([1.625, 0.40625, 0.40625], np.float64)
SCALARS = ('score', 'flow_sigma_um', 'near_um', 'density_10um',
           'peak_width_um', 'boundary_um')


def neighborhood(graph, nodes, hops=2):
    adjacency = defaultdict(set)
    for p, c, _ in graph.edges:
        adjacency[p].add(c)
        adjacency[c].add(p)
    found = set(nodes)
    frontier = set(nodes)
    for _ in range(hops):
        frontier = {v for u in frontier for v in adjacency[u]} - found
        found.update(frontier)
    return found


def graph_state(base, primary, selected, upstream):
    """Retain primary IDs only when both original and final coordinates agree."""
    n = max(max(base.nodes, default=-1), len(selected) - 1) + 1
    missing = n
    xyz = np.full((n + 1, 3), np.nan)
    times = np.full(n + 1, -1, np.int64)
    valid = np.zeros(n + 1, bool)
    degree = np.zeros(n + 1, np.int32)
    parent = np.full(n + 1, missing, np.int64)
    successor = np.full(n + 1, missing, np.int64)
    children = defaultdict(list)
    for node, coord in base.nodes.items():
        times[node], xyz[node] = coord[0], np.asarray(coord[1:]) * SCALE
        valid[node] = node < len(selected) and primary.nodes.get(node) == coord
    for p, c, _ in base.edges:
        degree[p] += 1
        parent[c] = p
        children[p].append(c)
    for p, cs in children.items():
        if len(cs) == 1:
            successor[p] = cs[0]
    forks = sorted(p for p, cs in children.items() if len(cs) == 2)
    guard = np.zeros(n + 1, bool)
    guard[list(neighborhood(base, forks))] = True
    edges = upstream.copy()
    pi, ci = edges.pi.to_numpy(np.int64), edges.ci.to_numpy(np.int64)
    keep = valid[pi] & valid[ci] & (times[ci] == times[pi] + 1)
    edges = edges.loc[keep].reset_index(drop=True)
    return dict(n=n, missing=missing, xyz=xyz, times=times, valid=valid,
                degree=degree, parent=parent, successor=successor,
                children=children, forks=forks, guard=guard, edges=edges)


def proposals(state):
    p = state['edges'].pi.to_numpy(np.int64)
    b = state['edges'].ci.to_numpy(np.int64)
    a = state['successor'][p]
    q = state['parent'][b]
    missing = state['missing']
    valid, guard = state['valid'], state['guard']
    keep = (state['degree'][p] == 1) & (a != b) & valid[a]
    keep &= (q == missing) | (state['degree'][q] == 1)
    keep &= ~(guard[p] | guard[a] | guard[b] | guard[q])
    p, a, b, q = p[keep], a[keep], b[keep], q[keep]
    actions = pd.DataFrame(dict(par=p, c1=np.minimum(a, b), c2=np.maximum(a, b),
                                added=b, displaced=np.where(q == missing, -1, q), is_action=True))
    refs = [(p, *sorted(state['children'][p]), -1, -1, False)
            for p in state['forks'] if all(valid[v] for v in [p, *state['children'][p]])]
    references = pd.DataFrame(refs, columns=actions.columns)
    result = pd.concat([actions, references], ignore_index=True)
    result['is_action'] = result['is_action'].astype(bool)
    for column in ('par', 'c1', 'c2', 'added', 'displaced'):
        result[column] = result[column].astype(np.int64)
    if result.duplicated(['par', 'c1', 'c2']).any():
        raise RuntimeError('duplicate complete pair')
    return result.sort_values(['par', 'c1', 'c2'], kind='stable').reset_index(drop=True)


def features(state, pairs, selected, candidates):
    """Vectorized, symmetric features; topology is not a target shortcut."""
    n, missing, xyz = state['n'], state['missing'], state['xyz']
    p, a, b = pairs[['par', 'c1', 'c2']].to_numpy(np.int64).T
    edges = state['edges']
    pi, ci = edges[['pi', 'ci']].to_numpy(np.int64).T
    code = pi * (n + 1) + ci
    order = np.argsort(code, kind='stable')
    sorted_code = code[order]
    if len(code) and np.any(np.diff(sorted_code) == 0):
        raise RuntimeError('duplicate upstream edge')
    scores = edges[['raw_p1', 'raw_p2']].to_numpy(np.float64)
    scores = np.concatenate([scores, np.full((1, 2), np.nan)])

    def lookup(left, right):
        query = left * (n + 1) + right
        index = np.searchsorted(sorted_code, query)
        safe = np.minimum(index, max(len(order) - 1, 0))
        if not len(order):
            return np.full((len(left), 2), np.nan)
        exists = (index < len(order)) & (sorted_code[safe] == query)
        return scores[np.where(exists, order[safe], len(order))]

    # Two best distinct upstream alternatives per child suffice to exclude P.
    ranked = edges.sort_values(['ci', 'raw_p2', 'pi'], ascending=[True, False, True], kind='stable')
    rank = ranked.groupby('ci', sort=False).cumcount().to_numpy()
    alternatives = np.full((n + 1, 2), missing, np.int64)
    top = rank < 2
    alternatives[ranked.ci.to_numpy(np.int64)[top], rank[top]] = ranked.pi.to_numpy(np.int64)[top]
    qa = alternatives[a, 0].copy()
    qb = alternatives[b, 0].copy()
    qa = np.where(qa == p, alternatives[a, 1], qa)
    qb = np.where(qb == p, alternatives[b, 1], qb)

    raw = {}
    pool_rows = selected.pool_row.to_numpy(np.int64)
    for column in [*SCALARS, 'flow_z', 'flow_y', 'flow_x']:
        arr = np.full(n + 1, np.nan)
        arr[:len(selected)] = candidates[column].to_numpy(np.float64)[pool_rows]
        arr[~state['valid']] = np.nan
        raw[column] = arr
    flow = np.column_stack([raw['flow_' + axis] for axis in 'zyx'])
    embedding = np.full((n + 1, 48), np.nan)
    embedding[:len(selected)] = candidates[[f'emb_{i}' for i in range(48)]].to_numpy(np.float64)[pool_rows]
    embedding[~state['valid']] = np.nan
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    output = {}

    def put(key, value):
        output[key] = np.asarray(value, np.float32)

    def symmetric(key, left, right):
        put(key + '_min', np.minimum(left, right))
        put(key + '_max', np.maximum(left, right))

    def norm(value):
        return np.linalg.norm(value, axis=1)

    def cosine(left, right):
        return np.sum(left * right, axis=1) / np.maximum(norm(left) * norm(right), 1e-12)

    def similarity(left, right):
        return np.sum(embedding[left] * embedding[right], axis=1)

    def chain(start, direction, steps):
        found = start.copy()
        for _ in range(steps):
            found = state[direction][found]
        return found

    da, db = xyz[a] - xyz[p], xyz[b] - xyz[p]
    fa, fb = xyz[a] + flow[a] - xyz[p], xyz[b] + flow[b] - xyz[p]
    separation = norm(xyz[a] - xyz[b])
    symmetric('arm_distance', norm(da), norm(db))
    symmetric('arm_flow_residual', norm(fa), norm(fb))
    symmetric('arm_flow_sigma_residual', norm(fa) / np.maximum(raw['flow_sigma_um'][a], .01),
              norm(fb) / np.maximum(raw['flow_sigma_um'][b], .01))
    symmetric('arm_embedding', similarity(p, a), similarity(p, b))
    put('daughter_embedding', similarity(a, b))
    put('daughter_separation', separation)
    put('arm_opposition_cosine', cosine(da, db))
    put('midpoint_distance', norm((xyz[a] + xyz[b]) / 2 - xyz[p]))
    put('daughter_flow_difference', norm(flow[a] - flow[b]))
    put('mean_flow_midpoint_residual', norm((fa + fb) / 2))
    pa, pb = lookup(p, a), lookup(p, b)
    aa, ab = lookup(qa, a), lookup(qb, b)
    for i, key in enumerate(('p1', 'p2')):
        symmetric('arm_' + key, pa[:, i], pb[:, i])
        symmetric('alternative_' + key, aa[:, i], ab[:, i])
        symmetric('advantage_' + key, pa[:, i] - aa[:, i], pb[:, i] - ab[:, i])
    symmetric('alternative_distance_advantage', norm(xyz[a] - xyz[qa]) - norm(da),
              norm(xyz[b] - xyz[qb]) - norm(db))
    symmetric('alternative_flow_advantage', norm(xyz[a] + flow[a] - xyz[qa]) - norm(fa),
              norm(xyz[b] + flow[b] - xyz[qb]) - norm(fb))
    symmetric('alternative_embedding_advantage', similarity(p, a) - similarity(qa, a),
              similarity(p, b) - similarity(qb, b))
    for key in SCALARS:
        value = raw[key]
        put('parent_' + key, value[p])
        symmetric('daughter_' + key, value[a], value[b])
        symmetric('parent_daughter_' + key + '_change', value[a] - value[p], value[b] - value[p])
        symmetric('alternative_parent_' + key + '_advantage', value[p] - value[qa], value[p] - value[qb])
    for horizon in (1, 2, 4):
        suffix = str(horizon)
        past = chain(p, 'parent', horizon)
        ahead_a, ahead_b = chain(a, 'successor', horizon), chain(b, 'successor', horizon)
        velocity = (xyz[p] - xyz[past]) / horizon
        va, vb = (xyz[ahead_a] - xyz[a]) / horizon, (xyz[ahead_b] - xyz[b]) / horizon
        past_qa, past_qb = chain(qa, 'parent', horizon), chain(qb, 'parent', horizon)
        vqa, vqb = (xyz[qa] - xyz[past_qa]) / horizon, (xyz[qb] - xyz[past_qb]) / horizon
        put('parent_speed_' + suffix, norm(velocity))
        symmetric('daughter_speed_' + suffix, norm(va), norm(vb))
        symmetric('motion_change_' + suffix, norm(va - velocity), norm(vb - velocity))
        symmetric('velocity_alignment_' + suffix, cosine(va, velocity), cosine(vb, velocity))
        symmetric('arm_extrapolation_' + suffix, norm(da - velocity), norm(db - velocity))
        symmetric('alternative_extrapolation_advantage_' + suffix,
                  norm(xyz[a] - xyz[qa] - vqa) - norm(da - velocity),
                  norm(xyz[b] - xyz[qb] - vqb) - norm(db - velocity))
        symmetric('alternative_velocity_advantage_' + suffix, norm(va - vqa) - norm(va - velocity),
                  norm(vb - vqb) - norm(vb - velocity))
        sep_future = norm(xyz[ahead_a] - xyz[ahead_b])
        put('future_separation_' + suffix, sep_future)
        put('separation_change_' + suffix, sep_future - separation)
        put('future_midpoint_residual_' + suffix,
            norm((xyz[ahead_a] + xyz[ahead_b]) / 2 - xyz[p] - (horizon + 1) * velocity))
        put('parent_embedding_history_' + suffix, similarity(p, past))
        symmetric('daughter_embedding_future_' + suffix, similarity(a, ahead_a), similarity(b, ahead_b))
        for key in ('score', 'peak_width_um'):
            value = raw[key]
            put('parent_history_' + key + '_' + suffix, value[p] - value[past])
            symmetric('daughter_future_' + key + '_' + suffix, value[ahead_a] - value[a], value[ahead_b] - value[b])
    frame = pd.DataFrame(output)
    if np.isinf(frame.to_numpy()).any():
        raise RuntimeError('nonfinite feature other than registered missing NaN')
    return frame


def select_actions(base, pairs, probabilities, threshold=.5):
    probability = np.asarray(probabilities, np.float64)
    if len(probability) != len(pairs) or not np.isfinite(probability).all():
        raise RuntimeError('invalid complete probabilities')
    eligible = pairs.loc[pairs.is_action & (probability >= threshold)].copy()
    eligible['probability'] = probability[eligible.index]
    eligible = eligible.sort_values(['probability', 'par', 'c1', 'c2'],
                                    ascending=[False, True, True, True], kind='stable')
    blocked, actions = set(), []
    # Build adjacency once; expanding only accepted actions keeps this inexpensive.
    adjacency = defaultdict(set)
    for p, c, _ in base.edges:
        adjacency[p].add(c)
        adjacency[c].add(p)
    for row in eligible.to_dict('records'):
        roles = {int(row[k]) for k in ('par', 'c1', 'c2', 'displaced') if int(row[k]) >= 0}
        if roles & blocked:
            continue
        actions.append({k: (float(v) if k == 'probability' else int(v))
                        for k, v in row.items() if k != 'is_action'})
        expanded, frontier = set(roles), set(roles)
        for _ in range(2):
            frontier = {v for u in frontier for v in adjacency[u]} - expanded
            expanded.update(frontier)
        blocked.update(expanded)
    return actions


def materialize(base, actions):
    edges = {(p, c): conf for p, c, conf in base.edges}
    outgoing = defaultdict(set)
    incoming = {}
    for p, c, _ in base.edges:
        outgoing[p].add(c)
        incoming[c] = p
    old_forks = {p: cs.copy() for p, cs in outgoing.items() if len(cs) == 2}
    for action in actions:
        p, a, b, added, displaced = [int(action[k]) for k in ('par', 'c1', 'c2', 'added', 'displaced')]
        if outgoing[p] != ({a, b} - {added}) or incoming.get(added, -1) != displaced:
            raise RuntimeError('stale or interacting contested action')
        if displaced >= 0:
            if outgoing[displaced] != {added}:
                raise RuntimeError('cannot displace a fork')
            del edges[displaced, added]
            outgoing[displaced].remove(added)
        edges[p, added] = .30
        outgoing[p].add(added)
        incoming[added] = p
    if any(outgoing[p] != cs for p, cs in old_forks.items()):
        raise RuntimeError('old fork changed')
    # Preserve original insertion order and confidence, then append new edges.
    result = DatasetGraph(base.dataset, dict(base.nodes),
                          [(p, c, conf) for (p, c), conf in edges.items()], base.shape)
    audit = validate_graph(result)
    audit.update(actions=len(actions), contested_actions=sum(a['displaced'] >= 0 for a in actions),
                 root_actions=sum(a['displaced'] < 0 for a in actions), original_nodes_exact=result.nodes == base.nodes,
                 original_forks_preserved=True)
    return result, audit
