"""The official node matching, reimplemented for speed — and checked against it.

`tracksdata.metrics.DistanceMatching` matches predicted to ground-truth nodes
per frame, one-to-one, with weight `1/(1+d)` inside a 7 um radius. That is
exactly right and far too slow to call inside a sweep over operating points,
where the same 199 samples are matched dozens of times.

**This is a faithful replication, not a cleaner equivalent.** The obvious
implementation — `linear_sum_assignment(maximize=True)` over the gated pairs —
disagrees with tracksdata on a handful of nodes per sample, because tracksdata
asks for a *full* bipartite matching (`min_weight_full_bipartite_matching`) and
falls back twice when none exists:

  1. try the sparse full matching over the gated pairs;
  2. on failure, fill empty rows *and columns* with -1 and retry;
  3. on failure again, go dense with the same fill and use `linear_sum_assignment`;
  4. drop pairs whose weight came from the fill.

Forcing every ground-truth node to be matched is a different optimum from
maximising total weight freely, so the two disagree exactly where a node has no
candidate of its own — which is the case the oracle exists to count. Steps 2-4
also carry a quirk worth naming: a dense-fallback pair landing on a position
that was neither stored nor filled reads back as 0 rather than -1 and survives
step 4. It is reproduced here because the official behaviour is what scores us,
not what we would have written.

`tests/test_matching.py` asserts identical assignments on real samples.
"""

from __future__ import annotations

import warnings

import numpy as np

from biohub.metrics.gt import MATCH_UM, SCALE_ARR

_FILL = -1.0


def _fill_empty(weights, fill_value: float) -> None:
    """Mirror of `tracksdata.metrics._ctc_metrics._fill_empty` (mutates)."""
    empty_rows = weights.sum(axis=1) == 0
    empty_cols = weights.sum(axis=0) == 0
    if empty_rows.any():
        weights[empty_rows, :] = fill_value
    if empty_cols.any():
        weights[:, empty_cols] = fill_value


def match_frame(
    pred_um: np.ndarray,
    gt_um: np.ndarray,
    max_distance: float = MATCH_UM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one matching within one frame. Returns `(pred_idx, gt_idx, distance)`."""
    import scipy.sparse as sp
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    if len(pred_um) == 0 or len(gt_um) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)

    # Gate first. The official code forms the full distance matrix and keeps the
    # entries within `max_distance`; gating produces the same set of pairs.
    neighbours = cKDTree(pred_um).query_ball_point(gt_um, r=max_distance)
    rows = np.repeat(np.arange(len(gt_um)), [len(n) for n in neighbours])
    cols = np.fromiter((x for n in neighbours for x in n), dtype=int, count=len(rows))
    if len(rows) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)

    d = np.linalg.norm(pred_um[cols] - gt_um[rows], axis=1)
    # Shape is inferred from the largest index that appears in a pair, exactly
    # as `sp.csr_array((data, (row, col)))` does in the official code.
    weights = sp.csr_array((1.0 / (1.0 + d), (rows, cols)), dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.SparseEfficiencyWarning)
        try:
            ri, ci = sp.csgraph.min_weight_full_bipartite_matching(weights, maximize=True)
        except ValueError:
            _fill_empty(weights, _FILL)
            try:
                ri, ci = sp.csgraph.min_weight_full_bipartite_matching(weights, maximize=True)
            except ValueError:
                coo = weights.tocoo()
                dense = np.full(weights.shape, _FILL, dtype=np.float32)
                dense[coo.row, coo.col] = coo.data
                ri, ci = linear_sum_assignment(dense, maximize=True)
            keep = ~np.isclose(weights[ri, ci], _FILL)
            ri, ci = ri[keep], ci[keep]

    return ci, ri, np.linalg.norm(pred_um[ci] - gt_um[ri], axis=1)


def match_sequence(
    pred_t: np.ndarray,
    pred_zyx: np.ndarray,
    gt_t: np.ndarray,
    gt_zyx: np.ndarray,
    max_distance: float = MATCH_UM,
) -> np.ndarray:
    """Match a whole sample. Returns `gt_to_pred`, `-1` where unmatched.

    Coordinates are voxels in, micrometres inside — the scale is applied here so
    no caller has to remember to.
    """
    gt_to_pred = np.full(len(gt_t), -1, dtype=np.int64)
    if len(pred_t) == 0:
        return gt_to_pred

    order_p = np.argsort(pred_t, kind="stable")
    bounds_p = np.searchsorted(pred_t[order_p], np.arange(int(pred_t.max()) + 2))

    for t in np.unique(gt_t):
        gsel = np.flatnonzero(gt_t == t)
        if t + 1 >= len(bounds_p):
            continue
        psel = order_p[bounds_p[t] : bounds_p[t + 1]]
        if len(psel) == 0:
            continue
        pi, gi, _ = match_frame(
            pred_zyx[psel] * SCALE_ARR, gt_zyx[gsel] * SCALE_ARR, max_distance
        )
        gt_to_pred[gsel[gi]] = psel[pi]
    return gt_to_pred
