"""Frame-pair association under the metric's own degree constraints.

The scorer accepts in-degree <= 1 and out-degree <= 2 (a division). Anything
beyond that it silently repairs rather than penalises, so the constraint belongs
here, where it is deliberate, rather than being discovered downstream
(`reports/metric_verification.md` §5).

Two solvers, same interface:

  hungarian  optimal, O(n^3); used when a frame holds few points
  greedy     cheapest-first, O(n log n); used at candidate density

Both are gated in **micrometres** before anything else, which is also what keeps
the cost matrix from becoming all-pairs (PLAN.md §4.2).
"""

from __future__ import annotations

import numpy as np

# Above this many points per frame the dense Hungarian cost matrix stops being
# worth it; greedy on the gated pairs is within a fraction of a percent and
# orders of magnitude faster.
HUNGARIAN_MAX_POINTS = 700


def gated_pairs(
    src_um: np.ndarray,
    dst_um: np.ndarray,
    gate_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(i, j, distance)` for every pair within `gate_um`. Sparse by construction."""
    from scipy.spatial import cKDTree

    if len(src_um) == 0 or len(dst_um) == 0:
        return (np.zeros(0, int),) * 2 + (np.zeros(0),)

    tree = cKDTree(dst_um)
    neighbours = tree.query_ball_point(src_um, r=gate_um)
    i = np.repeat(np.arange(len(src_um)), [len(n) for n in neighbours])
    j = np.fromiter((x for n in neighbours for x in n), dtype=int, count=len(i))
    if len(i) == 0:
        return i, j, np.zeros(0)
    d = np.linalg.norm(src_um[i] - dst_um[j], axis=1)
    return i, j, d


def _greedy(
    i: np.ndarray, j: np.ndarray, cost: np.ndarray, n_src: int, max_out: int
) -> list[tuple[int, int, float]]:
    order = np.argsort(cost, kind="stable")
    out_used = np.zeros(n_src, dtype=np.int8)
    in_used: set[int] = set()
    picked = []
    for k in order:
        a, b = int(i[k]), int(j[k])
        if out_used[a] >= max_out or b in in_used:
            continue
        out_used[a] += 1
        in_used.add(b)
        picked.append((a, b, float(cost[k])))
    return picked


def _hungarian(
    i: np.ndarray, j: np.ndarray, cost: np.ndarray, n_src: int, n_dst: int, max_out: int
) -> list[tuple[int, int, float]]:
    from scipy.optimize import linear_sum_assignment

    big = float(cost.max()) * 10 + 1.0 if len(cost) else 1.0
    # Out-degree <= 2 is expressed by giving each source `max_out` rows: the
    # assignment is then one-to-one on the expanded matrix and the constraint
    # holds by construction.
    dense = np.full((n_src * max_out, n_dst), big, dtype=np.float64)
    for rep in range(max_out):
        dense[rep * n_src + i, j] = cost
    ri, ci = linear_sum_assignment(dense)

    picked = []
    for r, c in zip(ri, ci):
        v = dense[r, c]
        if v < big:
            picked.append((int(r % n_src), int(c), float(v)))
    return picked


def associate(
    src_um: np.ndarray,
    dst_um: np.ndarray,
    gate_um: float = 7.0,
    max_out_degree: int = 1,
    predicted_dst_um: np.ndarray | None = None,
    solver: str = "auto",
) -> list[tuple[int, int, float]]:
    """Link points at `t` to points at `t+1`. Returns `(i, j, cost_um)`.

    `predicted_dst_um` is where each source is *expected* to appear — the motion
    model's output. Costs are measured from there, so a cell that moves with the
    tissue is cheap even when its raw displacement is large. Gating still uses
    the predicted position, which is the point of having one.
    """
    query = src_um if predicted_dst_um is None else predicted_dst_um
    i, j, d = gated_pairs(query, dst_um, gate_um)
    if len(i) == 0:
        return []

    if solver == "auto":
        solver = (
            "hungarian"
            if max(len(src_um), len(dst_um)) <= HUNGARIAN_MAX_POINTS
            else "greedy"
        )
    if solver == "hungarian":
        return _hungarian(i, j, d, len(src_um), len(dst_um), max_out_degree)
    if solver == "greedy":
        return _greedy(i, j, d, len(src_um), max_out_degree)
    raise ValueError(f"unknown solver {solver!r}")


def link_sequence(
    points_um: dict[int, np.ndarray],
    gate_um: float = 7.0,
    max_out_degree: int = 1,
    motion: str = "none",
    context_um: dict[int, np.ndarray] | None = None,
) -> dict[int, list[tuple[int, int, float]]]:
    """Associate every consecutive frame pair present in `points_um`.

    `motion` is one of `none`, `global`, `local`.

    `context_um` supplies a *denser* point cloud per frame for estimating motion
    only — never for linking. Motion estimated from nine annotated cells is
    mostly noise; the same estimate from a few thousand detected candidates is
    usable, and using the candidates for that involves no ground truth.
    """
    from biohub.motion.flow import estimate_global_translation, estimate_local_flow

    out: dict[int, list[tuple[int, int, float]]] = {}
    times = sorted(points_um)
    for t in times:
        if t + 1 not in points_um:
            continue
        a, b = points_um[t], points_um[t + 1]
        if len(a) == 0 or len(b) == 0:
            out[t] = []
            continue

        if motion == "none":
            predicted = None
        else:
            ca = a if context_um is None else context_um.get(t, a)
            cb = b if context_um is None else context_um.get(t + 1, b)
            shift = estimate_global_translation(ca, cb)
            if motion == "global":
                predicted = a + shift
            elif motion == "local":
                # Fit the field on the context cloud, evaluate it at `a`.
                field_src = np.concatenate([ca, a]) if context_um is not None else a
                flow = estimate_local_flow(field_src, cb, shift=shift)
                predicted = a + flow[-len(a):]
            else:
                raise ValueError(f"unknown motion model {motion!r}")

        out[t] = associate(a, b, gate_um, max_out_degree, predicted)
    return out
