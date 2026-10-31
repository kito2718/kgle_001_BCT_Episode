"""Exact constrained lineage decoder used as the V34 correctness reference.

This is not a post-hoc intervention selector.  It jointly decides cell
occupancy, transitions, births, deaths, and divisions over the complete scored
hypothesis graph.  The initial SciPy/HiGHS backend is a correctness and
feasibility reference; a faster deployment backend may replace it only after
parity tests are available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biohub.transition.graph import TransitionGraph


@dataclass(frozen=True)
class DecodeConfig:
    birth_penalty: float = 1.0
    death_penalty: float = 1.0
    division_penalty: float = 1.0
    count_deviation_penalty: float = 2.0
    time_limit_seconds: float | None = 60.0
    mip_rel_gap: float = 0.0
    tie_break_epsilon: float = 1e-9
    allow_divisions: bool = True


@dataclass(frozen=True)
class DecodeResult:
    selected_nodes: np.ndarray
    selected_edge_rows: np.ndarray
    selected_edges: np.ndarray
    birth_nodes: np.ndarray
    death_nodes: np.ndarray
    division_nodes: np.ndarray
    objective: float
    status: int
    message: str
    optimal: bool


def decode_graph(
    graph: TransitionGraph,
    cfg: DecodeConfig | None = None,
) -> DecodeResult:
    """Solve the complete binary lineage problem.

    For node ``i`` and edge ``e`` the binary variables are ``x_i`` and ``y_e``.
    Birth ``b_i``, death ``d_i``, and division ``q_i`` satisfy::

        incoming(i) + b_i = x_i
        outgoing(i) + d_i = x_i + q_i
        d_i + q_i <= 1
        q_i <= x_i

    Consequently a selected node has exactly one parent or is a birth, then
    continues once, dies, or divides into exactly two children.
    """

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_array

    cfg = cfg or DecodeConfig()
    n = len(graph.t)
    m = len(graph.edges)
    frames = sorted((graph.target_counts or {}).keys())
    k_count = len(frames)

    x0 = 0
    y0 = x0 + n
    b0 = y0 + m
    d0 = b0 + n
    q0 = d0 + n
    over0 = q0 + n
    under0 = over0 + k_count
    n_var = under0 + k_count

    c = np.zeros(n_var, dtype=np.float64)
    c[x0:x0 + n] = -graph.node_score
    c[y0:y0 + m] = -graph.edge_score
    c[b0:b0 + n] = cfg.birth_penalty - graph.birth_score
    c[d0:d0 + n] = cfg.death_penalty - graph.death_score
    c[q0:q0 + n] = cfg.division_penalty - graph.division_score
    if k_count:
        c[over0:under0 + k_count] = cfg.count_deviation_penalty

    # Deterministic preference among otherwise identical solutions.  This is
    # several orders below any calibrated score and only resolves exact ties.
    if n and cfg.tie_break_epsilon:
        c[x0:x0 + n] += cfg.tie_break_epsilon * np.arange(n) / max(n, 1)
    if m and cfg.tie_break_epsilon:
        c[y0:y0 + m] += cfg.tie_break_epsilon * np.arange(m) / max(m, 1)

    lower = np.zeros(n_var, dtype=np.float64)
    upper = np.ones(n_var, dtype=np.float64)
    if not cfg.allow_divisions:
        upper[q0:q0 + n] = 0.0
    if k_count:
        upper[over0:under0 + k_count] = np.inf
    integrality = np.ones(n_var, dtype=np.int8)
    if k_count:
        integrality[over0:under0 + k_count] = 0

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lb: list[float] = []
    ub: list[float] = []

    def constraint(entries: list[tuple[int, float]], lo: float, hi: float) -> None:
        row = len(lb)
        rows.extend([row] * len(entries))
        cols.extend(index for index, _ in entries)
        data.extend(value for _, value in entries)
        lb.append(lo)
        ub.append(hi)

    incoming: list[list[int]] = [[] for _ in range(n)]
    outgoing: list[list[int]] = [[] for _ in range(n)]
    for edge_index, (source, target) in enumerate(graph.edges):
        outgoing[int(source)].append(edge_index)
        incoming[int(target)].append(edge_index)

    for i in range(n):
        constraint(
            [(x0 + i, -1.0), (b0 + i, 1.0)]
            + [(y0 + e, 1.0) for e in incoming[i]],
            0.0,
            0.0,
        )
        constraint(
            [(x0 + i, -1.0), (d0 + i, 1.0), (q0 + i, -1.0)]
            + [(y0 + e, 1.0) for e in outgoing[i]],
            0.0,
            0.0,
        )
        constraint([(q0 + i, 1.0), (x0 + i, -1.0)], -np.inf, 0.0)
        constraint([(d0 + i, 1.0), (q0 + i, 1.0)], -np.inf, 1.0)

    for a, b in graph.duplicate_pairs:
        constraint([(x0 + int(a), 1.0), (x0 + int(b), 1.0)], -np.inf, 1.0)

    for j, frame in enumerate(frames):
        members = np.flatnonzero(graph.t == frame)
        target = float(graph.target_counts[frame])
        constraint(
            [(x0 + int(i), 1.0) for i in members]
            + [(over0 + j, -1.0), (under0 + j, 1.0)],
            target,
            target,
        )

    matrix = coo_array(
        (np.asarray(data), (np.asarray(rows), np.asarray(cols))),
        shape=(len(lb), n_var),
    ).tocsr()
    options: dict[str, float | bool] = {
        "presolve": True,
        "mip_rel_gap": float(cfg.mip_rel_gap),
    }
    if cfg.time_limit_seconds is not None:
        options["time_limit"] = float(cfg.time_limit_seconds)
    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, np.asarray(lb), np.asarray(ub)),
        options=options,
    )
    if result.x is None:
        raise RuntimeError(f"global transition decode failed: {result.message}")

    solution = np.asarray(result.x)
    selected_nodes = np.flatnonzero(solution[x0:x0 + n] > 0.5)
    selected_edge_rows = np.flatnonzero(solution[y0:y0 + m] > 0.5)
    births = np.flatnonzero(solution[b0:b0 + n] > 0.5)
    deaths = np.flatnonzero(solution[d0:d0 + n] > 0.5)
    divisions = np.flatnonzero(solution[q0:q0 + n] > 0.5)
    return DecodeResult(
        selected_nodes=selected_nodes.astype(np.int64),
        selected_edge_rows=selected_edge_rows.astype(np.int64),
        selected_edges=graph.edges[selected_edge_rows],
        birth_nodes=births.astype(np.int64),
        death_nodes=deaths.astype(np.int64),
        division_nodes=divisions.astype(np.int64),
        objective=float(-result.fun),
        status=int(result.status),
        message=str(result.message),
        optimal=bool(result.status == 0),
    )


def decode_ordinary_graph(
    graph: TransitionGraph,
    cfg: DecodeConfig | None = None,
) -> DecodeResult:
    """Exact reduced formulation when divisions are forbidden.

    Birth and death variables are algebraically eliminated. Edge variables are
    continuous because, once node admissions are binary, they form a bipartite
    matching polytope and therefore have an integral optimum. This preserves
    the complete objective while reducing the binary variables from
    ``4 * nodes + edges`` to ``nodes``.
    """

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_array

    cfg = cfg or DecodeConfig(allow_divisions=False)
    if cfg.allow_divisions:
        raise ValueError("decode_ordinary_graph requires allow_divisions=False")
    n = len(graph.t)
    m = len(graph.edges)
    frames = sorted((graph.target_counts or {}).keys())
    k_count = len(frames)

    x0 = 0
    y0 = n
    over0 = y0 + m
    under0 = over0 + k_count
    n_var = under0 + k_count

    birth_bonus = graph.birth_score - cfg.birth_penalty
    death_bonus = graph.death_score - cfg.death_penalty
    node_utility = graph.node_score + birth_bonus + death_bonus
    edge_utility = graph.edge_score.copy()
    if m:
        source, target = graph.edges[:, 0], graph.edges[:, 1]
        edge_utility = (
            edge_utility - birth_bonus[target] - death_bonus[source]
        )

    c = np.zeros(n_var, dtype=np.float64)
    c[x0:x0 + n] = -node_utility
    c[y0:y0 + m] = -edge_utility
    if k_count:
        c[over0:under0 + k_count] = cfg.count_deviation_penalty
    if n and cfg.tie_break_epsilon:
        c[x0:x0 + n] += cfg.tie_break_epsilon * np.arange(n) / max(n, 1)
    if m and cfg.tie_break_epsilon:
        c[y0:y0 + m] += cfg.tie_break_epsilon * np.arange(m) / max(m, 1)

    lower = np.zeros(n_var, dtype=np.float64)
    upper = np.ones(n_var, dtype=np.float64)
    if k_count:
        upper[over0:under0 + k_count] = np.inf
    integrality = np.zeros(n_var, dtype=np.int8)
    integrality[x0:x0 + n] = 1

    incoming: list[list[int]] = [[] for _ in range(n)]
    outgoing: list[list[int]] = [[] for _ in range(n)]
    for edge_index, (source, target) in enumerate(graph.edges):
        outgoing[int(source)].append(edge_index)
        incoming[int(target)].append(edge_index)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lb: list[float] = []
    ub: list[float] = []

    def constraint(entries: list[tuple[int, float]], lo: float, hi: float) -> None:
        row = len(lb)
        rows.extend([row] * len(entries))
        cols.extend(index for index, _ in entries)
        data.extend(value for _, value in entries)
        lb.append(lo)
        ub.append(hi)

    for i in range(n):
        constraint(
            [(x0 + i, -1.0)] + [(y0 + edge, 1.0) for edge in incoming[i]],
            -np.inf,
            0.0,
        )
        constraint(
            [(x0 + i, -1.0)] + [(y0 + edge, 1.0) for edge in outgoing[i]],
            -np.inf,
            0.0,
        )
    for a, b in graph.duplicate_pairs:
        constraint([(x0 + int(a), 1.0), (x0 + int(b), 1.0)], -np.inf, 1.0)
    for j, frame in enumerate(frames):
        members = np.flatnonzero(graph.t == frame)
        target = float(graph.target_counts[frame])
        constraint(
            [(x0 + int(i), 1.0) for i in members]
            + [(over0 + j, -1.0), (under0 + j, 1.0)],
            target,
            target,
        )

    matrix = coo_array(
        (np.asarray(data), (np.asarray(rows), np.asarray(cols))),
        shape=(len(lb), n_var),
    ).tocsr()
    options: dict[str, float | bool] = {
        "presolve": True,
        "mip_rel_gap": float(cfg.mip_rel_gap),
    }
    if cfg.time_limit_seconds is not None:
        options["time_limit"] = float(cfg.time_limit_seconds)
    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, np.asarray(lb), np.asarray(ub)),
        options=options,
    )
    if result.x is None:
        raise RuntimeError(f"reduced ordinary decode failed: {result.message}")

    solution = np.asarray(result.x)
    edge_solution = solution[y0:y0 + m]
    fractionality = np.minimum(edge_solution, 1.0 - edge_solution)
    if len(fractionality) and float(fractionality.max()) > 1e-5:
        raise RuntimeError(
            "ordinary matching relaxation returned a fractional edge solution"
        )
    selected_nodes = np.flatnonzero(solution[x0:x0 + n] > 0.5)
    selected_edge_rows = np.flatnonzero(edge_solution > 0.5)
    incoming_sum = np.zeros(n, dtype=np.float64)
    outgoing_sum = np.zeros(n, dtype=np.float64)
    if len(selected_edge_rows):
        chosen = graph.edges[selected_edge_rows]
        np.add.at(outgoing_sum, chosen[:, 0], 1.0)
        np.add.at(incoming_sum, chosen[:, 1], 1.0)
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[selected_nodes] = True
    births = np.flatnonzero(selected_mask & (incoming_sum < 0.5))
    deaths = np.flatnonzero(selected_mask & (outgoing_sum < 0.5))
    return DecodeResult(
        selected_nodes=selected_nodes.astype(np.int64),
        selected_edge_rows=selected_edge_rows.astype(np.int64),
        selected_edges=graph.edges[selected_edge_rows],
        birth_nodes=births.astype(np.int64),
        death_nodes=deaths.astype(np.int64),
        division_nodes=np.zeros(0, dtype=np.int64),
        objective=float(-result.fun),
        status=int(result.status),
        message=str(result.message),
        optimal=bool(result.status == 0),
    )
