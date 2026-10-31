"""Complete missing fork arms from agreement between existing graph witnesses."""
from collections import Counter, defaultdict

import numpy as np

from biohub.io.submission import DatasetGraph, validate_graph
from biohub.metrics.gt import SCALE_ARR
from biohub.metrics.matching import match_sequence
from biohub.tracker.postpass33 import changed_hop_guard

POLICIES = {"root_consensus2": (2, False), "rewire_consensus3": (3, True)}


def children(graph):
    result = defaultdict(list)
    for parent, child, _ in graph.edges:
        result[parent].append(child)
    return result


def fork_votes(base, witnesses):
    """One vote per bank and exact mapped triple; no annotations are used."""
    ids = np.asarray(list(base.nodes), np.int64)
    coords = np.asarray(list(base.nodes.values()), np.float64).reshape(-1, 4)
    votes = defaultdict(set)
    audit = {}
    for bank, witness in sorted(witnesses.items()):
        donor_ids = np.asarray(list(witness.nodes), np.int64)
        donor_coords = np.asarray(list(witness.nodes.values()), np.float64).reshape(-1, 4)
        mapping = match_sequence(coords[:, 0].astype(int), coords[:, 1:],
            donor_coords[:, 0].astype(int), donor_coords[:, 1:], max_distance=3.0)
        matched = np.flatnonzero(mapping >= 0)
        distances = np.linalg.norm((coords[mapping[matched], 1:] - donor_coords[matched, 1:]) * SCALE_ARR, axis=1)
        rejected = matched[distances > 3.0]
        mapping[rejected] = -1
        pairs = {int(donor_ids[i]): int(ids[j]) for i, j in enumerate(mapping) if j >= 0}
        if len(set(pairs.values())) != len(pairs):
            raise RuntimeError("witness mapping is not one-to-one")
        supported = set()
        donor_children = children(witness)
        for parent, offspring in donor_children.items():
            if len(offspring) != 2 or any(n not in pairs for n in [parent, *offspring]):
                continue
            p, a, b = (pairs[n] for n in [parent, *offspring])
            if len({p, a, b}) != 3:
                continue
            if base.nodes[a][0] != base.nodes[p][0] + 1 or base.nodes[b][0] != base.nodes[p][0] + 1:
                continue
            triple = (p, *sorted((a, b)))
            supported.add(triple)
            votes[triple].add(bank)
        audit[bank] = {"mapped_nodes": len(pairs), "radius_rejections": len(rejected),
            "donor_forks": sum(len(v) == 2 for v in donor_children.values()), "mapped_forks": len(supported)}
    return dict(votes), audit


def select_actions(base, votes, policy, guard_hops=2):
    minimum, allow_rewire = POLICIES[policy]
    outgoing = children(base)
    incoming = {child: parent for parent, child, _ in base.edges}
    pi = np.asarray([p for p, _, _ in base.edges], np.int64)
    ci = np.asarray([c for _, c, _ in base.edges], np.int64)
    guard = changed_hop_guard(max(base.nodes, default=-1) + 1, pi, ci, np.arange(len(pi)), guard_hops)
    proposals = []
    for (parent, a, b), banks in sorted(votes.items()):
        if len(banks) < minimum or len(outgoing[parent]) != 1 or guard[parent]:
            continue
        first = outgoing[parent][0]
        if first not in (a, b):
            continue
        second = b if first == a else a
        old_parent = incoming.get(second)
        if old_parent is not None and (not allow_rewire or len(outgoing[old_parent]) != 1):
            continue
        participants = {parent, first, second}
        if old_parent is not None:
            participants.add(old_parent)
        proposals.append({"parent": parent, "first": first, "second": second,
            "old_parent": old_parent, "votes": sorted(banks), "participants": sorted(participants)})
    counts = Counter(n for p in proposals for n in p["participants"])
    actions = [p for p in proposals if all(counts[n] == 1 for n in p["participants"])]
    return actions, {"qualified_before_conflicts": len(proposals), "conflict_rejections": len(proposals) - len(actions)}


def materialize(base, actions):
    removed = {(a["old_parent"], a["second"]) for a in actions if a["old_parent"] is not None}
    result = DatasetGraph(base.dataset, dict(base.nodes),
        [edge for edge in base.edges if edge[:2] not in removed]
        + [(a["parent"], a["second"], 1.0) for a in actions], base.shape)
    audit = validate_graph(result)
    original_children, final_children = children(base), children(result)
    old_forks = {p for p, cs in original_children.items() if len(cs) == 2}
    new_forks = {p for p, cs in final_children.items() if len(cs) == 2}
    expected = {a["parent"] for a in actions}
    if new_forks != old_forks | expected or old_forks & expected:
        raise RuntimeError("unexpected division changes")
    for parent in old_forks:
        if sorted(original_children[parent]) != sorted(final_children[parent]):
            raise RuntimeError("existing fork changed")
    edge_set = set(result.edges)
    first_arms = {(a["parent"], a["first"]) for a in actions}
    protected = [e for e in base.edges if e[0] in old_forks or e[:2] in first_arms]
    if any(e not in edge_set for e in protected) or result.nodes != base.nodes:
        raise RuntimeError("protected arm or coordinate changed")
    audit.update(new_forks=len(actions), rewired_arms=len(removed), nodes_exact=True, old_forks_exact=True)
    return result, audit
