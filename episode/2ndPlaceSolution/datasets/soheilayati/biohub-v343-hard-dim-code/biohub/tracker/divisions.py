"""Fast division TP/FP/FN over an emitted edge set — the sweep-side counter.

The official rule (`tracking_cellmot.division_metrics`) is generous on the TP
side and specific on the FP side. This mirrors it from the GLOBAL matching:

  TP (strict)   a predicted fork that is the stand-in of a GT divider and
                whose two children are the stand-ins of the two GT daughters
  FP            evaluable forks (fork's node matched to an annotated GT node
                with out-degree >= 1) that are not TPs, plus cross-component
                forks (two child branches whose nearest matched evidence lies
                in distinct GT weakly-connected components — the annotation-
                bridging exploit the July metric patch closed)
  FN            GT divisions minus TP

Two official mechanisms are deliberately not replicated, in opposite
directions, so this is a bracket rather than an estimate:

  * per-division LOCAL re-matching plus parent/grandchild slack — official
    TP >= strict TP (a fork one frame early, or a daughter evidenced only by
    its own child, still credits officially);
  * "considered but rejected" forks near real divisions — official FP >= the
    FP counted here.

Malformed forks (a local branch with a second predecessor) cannot occur under
the linker's in-degree <= 1, so they are not computed.
`tests/test_divisions.py` verifies the bracket against the official
`evaluate_divisions` on real forked predictions, and the promoted operating
point is always confirmed end-to-end through the official scorer.
"""

from __future__ import annotations

import numpy as np

from biohub.metrics.gt import GroundTruth


def gt_components(gt: GroundTruth) -> np.ndarray:
    """Weakly-connected component id per GT node row (union-find)."""
    parent = np.arange(gt.n_nodes, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, d in gt.edges:
        ra, rb = find(int(s)), find(int(d))
        if ra != rb:
            parent[rb] = ra
    return np.array([find(i) for i in range(gt.n_nodes)], dtype=np.int64)


def division_counts(
    pi: np.ndarray,
    ci: np.ndarray,
    gt_row: np.ndarray,
    gt: GroundTruth,
) -> dict:
    """Strict-TP / faithful-FP division counts for one emitted edge set.

    `pi`/`ci` are the emitted edges over the selected node set; `gt_row` maps
    each selected node to its official GT row (-1 unmatched).
    """
    children: dict[int, list[int]] = {}
    for a, b in zip(pi.tolist(), ci.tolist()):
        children.setdefault(a, []).append(b)
    forks = {a: cs for a, cs in children.items() if len(cs) >= 2}

    gt_out = np.zeros(gt.n_nodes, dtype=np.int64)
    if gt.n_edges:
        np.add.at(gt_out, gt.edges[:, 0], 1)
    gt_daughters: dict[int, set[int]] = {}
    for s, d in gt.edges:
        if gt_out[s] == 2:
            gt_daughters.setdefault(int(s), set()).add(int(d))
    n_gt_div = len(gt_daughters)

    if not forks:
        return {"division_tp": 0, "division_fp": 0, "division_fn": n_gt_div}

    comp = gt_components(gt)

    tp_forks: set[int] = set()
    credited: set[int] = set()
    fp_forks: set[int] = set()
    for f, cs in forks.items():
        g = int(gt_row[f])
        c_rows = [int(gt_row[c]) for c in cs]
        is_tp = (
            g >= 0
            and g in gt_daughters
            and g not in credited
            and all(r >= 0 for r in c_rows)
            and set(c_rows) == gt_daughters[g]
        )
        if is_tp:
            tp_forks.add(f)
            credited.add(g)
            continue

        if g >= 0 and gt_out[g] >= 1:
            fp_forks.add(f)  # evaluable spurious fork
            continue

        # Cross-component evidence: matched child, else unanimous matched
        # grandchildren. (in-degree <= 1 upstream, so branches cannot merge.)
        evidence: list[int] = []
        for c in cs:
            r = int(gt_row[c])
            if r >= 0:
                evidence.append(int(comp[r]))
                continue
            gcs = [int(gt_row[gc]) for gc in children.get(c, []) if gt_row[gc] >= 0]
            comps = {int(comp[r]) for r in gcs}
            if len(comps) == 1:
                evidence.append(comps.pop())
        if len(set(evidence)) >= 2:
            fp_forks.add(f)

    return {
        "division_tp": len(tp_forks),
        "division_fp": len(fp_forks),
        "division_fn": n_gt_div - len(tp_forks),
    }
