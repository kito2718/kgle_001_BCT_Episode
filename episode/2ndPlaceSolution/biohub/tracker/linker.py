"""The claims linker: winner-take-all proposals, out-degree 2 by competition.

Input is one live edge per child — its argmax parent at the group-calibrated
probability (`scripts/rescore_groups.py`) — so in-degree <= 1 holds by
construction and a fork can only form when two children INDEPENDENTLY claim
the same parent. That structural difference is what separates this from the
naive out-degree-2 flag fact 35 killed: the linker never assigns a second
child, it only ratifies a second claim that clears its own bar.

    first claim   emitted when p >= tau     (the continuation decision)
    second claim  emitted when p >= tau2    (the division decision)
    later claims  never emitted             (the scorer caps out-degree at 2)

Emission is per-parent rank by probability, so the set is vectorised and
deterministic; edges written in descending p keep the scorer's insertion-order
truncation aligned with our own confidence (fact 5).
"""

from __future__ import annotations

import numpy as np


def emit_claims(
    pi: np.ndarray,
    ci: np.ndarray,
    p: np.ndarray,
    tau: float,
    tau2: float = 2.0,
) -> np.ndarray:
    """Row indices of the emitted edges, given one claim per child.

    `tau2 > 1` disables forks (the od1 linker). Rows where `p` is 0 are dead
    claims (the winner-take-all convention for a child's non-argmax edges)
    and can never emit because `tau` > 0.
    """
    keep = np.flatnonzero(p >= tau)
    if len(keep) == 0:
        return keep

    order = keep[np.lexsort((-p[keep], pi[keep]))]
    pis = pi[order]
    idx = np.arange(len(order))
    new_parent = np.r_[True, pis[1:] != pis[:-1]]
    start = np.maximum.accumulate(np.where(new_parent, idx, 0))
    rank = idx - start

    take = (rank == 0) | ((rank == 1) & (p[order] >= tau2))
    return order[take]


def emit_termination_continuations(
    pi: np.ndarray,
    ci: np.ndarray,
    p: np.ndarray,
    tau: float,
    tau_cont: float,
) -> np.ndarray:
    """Continuation claims with a lower bar only for terminating parents.

    Start with the ordinary rank-0 claims at ``tau``.  A below-threshold
    argmax claim may then emit only when its parent has no ordinary child;
    among several such children, the parent keeps the highest-confidence
    one.  Inputs follow the winner-take-all convention (one non-zero edge per
    child), so indegree remains at most one.  The result is continuation-only
    (outdegree at most one); an independent division rule may add a second
    child later.
    """
    base = emit_claims(pi, ci, p, tau, 2.0)
    if tau_cont >= tau:
        return base

    n_nodes = int(max(pi.max(initial=-1), ci.max(initial=-1)) + 1)
    has_child = np.zeros(n_nodes, dtype=bool)
    has_child[pi[base]] = True
    cand = np.flatnonzero((p >= tau_cont) & (p < tau) & ~has_child[pi])
    if len(cand) == 0:
        return base

    extra: list[int] = []
    for row in cand[np.argsort(-p[cand], kind="stable")]:
        parent = int(pi[row])
        if has_child[parent]:
            continue
        has_child[parent] = True
        extra.append(int(row))
    if not extra:
        return base
    return np.concatenate([base, np.asarray(extra, dtype=base.dtype)])
