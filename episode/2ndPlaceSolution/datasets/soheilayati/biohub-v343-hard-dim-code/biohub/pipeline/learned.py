"""The test-time pipeline: a zarr volume in, a predicted graph out.

Two operating points, each frozen as module constants so the notebook cannot
quietly run a configuration different from the one that was measured:

**Stage 2** (`track_dataset_learned`, engineering OOF 0.8170, public 0.763):
`verify_deployable.py`'s operating point made deployable â€”

    suppression   5.0 um     fact 15 â€” above ~5 um divisions start merging
    link gate     7.0 um     fact 13 â€” a wider RAW gate lowers the score
    out-degree    1          the Stage 4 lever, not cashed in this path
    motion        local      tissue flow, estimated from the selected cloud

**Stage 4** (`track_dataset_learned_v2`, engineering OOF 0.8471, stage-04
Â§7â€“Â§8): the same detector, then the claims linker â€” 1.25x predicted node
budget (fact 41), flow-gated k=8/14 um edge universe scored by the exported
GBM ensemble with group calibration (`biohub.tracker.edge_model`), and the
division decision as a second independent claim at tau2 (fact 40). Node
selection is `biohub.tracker.edges.select_nodes`, the measurement rig's own
function.

**Node count comes from the learned count model, not from a fixed budget.**
Fact 9 prices that difference at 0.137, and it is the single largest
non-modelling lever available. What the count model does NOT decide any more
is how many nodes to submit: fact 41 measured that the optimal submission
budget is ~1.25x the predicted count.

Edge confidence: the Stage 2 path carries `1 / (1 + d)`, the Stage 4 path the
group-calibrated probability. Either way `build_submission` writes descending,
so the scorer's insertion-order truncation of out-degree > 2 drops the edge we
trust least (fact 5).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biohub.io.submission import DatasetGraph

NMS_UM = 5.0
GATE_UM = 7.0
MAX_OUT_DEGREE = 1
MOTION = "local"

# ---- the Stage 4 operating point (stage-04 Â§7â€“Â§8), same discipline: every
# constant is the one the 0.8471 measurement was taken at.
#
#   budget        1.25 x the predicted count   fact 41 â€” submitting the true
#                 count answers the wrong question; node recall at a budget is
#                 imperfect, so the optimum sits above the count (+0.0239).
#                 1.25 chosen over 1.35 on the tie-break: P = 0.762, and the
#                 multiplier is charged against the UNSEEN true count.
#   edge universe k=8 within 14 um, FLOW-centred  stage-04 Â§6.1 â€” covers
#                 0.9987 of found GT edges, 228/229 findable division edges
#   tau           0.47                         fact 4 â€” emit-vs-abstain
#                 break-even, pre-registered
#   tau2          0.80                         fact 40 â€” the division claim's
#                 bar, swept and confirmed through the official evaluator
#                 (16/124/135 at 1.0x; 26/258/125 at 1.25x)
BUDGET_MULT = 1.25
# V40b registered treatment (configs/experiment_v40b_selection_score_calibration.json):
# per-frame median-logit alignment of the solver's cross-frame ranking vector.
# Off by default; the default path is byte-identical to every historical run.
SCORE_FRAME_ALIGN = False
EDGE_K = 8
EDGE_GATE_UM = 14.0
TAU = 0.47
TAU2 = 0.80

# ---- the solver v1.1 operating point (stage-04 Â§11, facts 47â€“48, official
# 0.8748): node selection becomes a joint decision over a 2x superset â€”
# support-ranked, selection-consistent, WTA-aware, ONE ranking cut per sample
# instead of the per-frame quota. Constants live in `biohub.tracker.solver`
# beside the implementation the rig measured.

# ---- the pair-model fork decision (stage-04 Â§13, fact 50, official 0.8810):
# the Ï„2 dial is replaced by the independent division score at its measured
# operating threshold. 0.10 was chosen over the score-tied 0.01 because it
# reaches the same score with 2.5x fewer false forks â€” the risk-asymmetric
# choice on unseen embryos, the same logic that fixed the budget at 1.25x.
TAU_DIV_PAIRS = 0.10

# ---- the 0.9021 chain (facts 54/59/60/61): pass-2 context continuation,
# temporal division stack on faithful labels, fork-guarded smoothing, and
# score-aligned graph cleanup. Every constant is the one the measurement
# selected; none is tunable here.
TAU_DIV_STACK = 0.98    # fact 61 â€” the faithful stack's swept threshold
SMOOTH_WEIGHT = 0.75    # fact 61 â€” 1.0 over-smooths (fact 60's downturn)
SMOOTH_RADIUS = 2       # fact 60 â€” Â±2-frame window


@dataclass
class TrackResult:
    graph: DatasetGraph
    n_candidates: int
    budget_per_frame: int
    predicted_per_frame: float
    n_selected: int
    n_edgeless: int
    n_forks: int = 0
    # Optional deploy-only state for downstream annotation-free post-passes.
    # Ordinary callers keep the historical lightweight result because every
    # tracker entry point leaves this as ``None`` unless explicitly requested.
    inference_state: dict | None = None


def image_quantiles(zarr_path: Path) -> dict:
    """`image_statistics.quantiles` from the zarr group attributes, or {}."""
    import zarr

    attrs = dict(zarr.open_group(str(zarr_path), mode="r").attrs)
    return (attrs.get("image_statistics") or {}).get("quantiles", {}) or {}


def track_dataset_learned(
    zarr_path: Path,
    models,
    count_model,
    device,
    amp_dtype=None,
    batch: int = 8,
    drop_edgeless: bool = False,
) -> TrackResult:
    """Detect, size, select and link one dataset."""
    import zarr

    from biohub.detector.infer import detect, volumes_from_zarr

    zarr_path = Path(zarr_path)
    name = zarr_path.stem
    shape = tuple(zarr.open_array(str(zarr_path / "0"), mode="r").shape)

    vols = volumes_from_zarr(zarr_path)
    cand = detect(models, vols, device, amp_dtype=amp_dtype, batch=batch)
    del vols
    gc.collect()

    return size_select_link(name, shape, cand, count_model,
                            image_quantiles(zarr_path), drop_edgeless=drop_edgeless)


def size_select_link(
    name: str,
    shape: tuple[int, int, int, int],
    cand,
    count_model,
    quantiles: dict,
    drop_edgeless: bool = False,
) -> TrackResult:
    """Everything downstream of the detector: size, suppress, select, link.

    Factored out so the classical control (`pipeline.classical`) and the learned
    path share it verbatim â€” the public A/B is an experiment on the detector,
    and it is only an experiment if the detector is the single variable.
    """
    from biohub.candidates.nms import select_frame
    from biohub.counts import features_from
    from biohub.metrics.gt import SCALE_ARR
    from biohub.tracker.associate import link_sequence

    feats = features_from(name, quantiles, cand)
    per_frame = float(count_model.predict_per_frame(feats)[0])
    budget = max(1, int(round(per_frame)))

    c_t = cand["t"].to_numpy()
    # Round to integer source voxels and clip: the offset head refines a peak
    # inside its voxel and can push one just past the volume bound.
    c_zyx = np.round(cand[["z", "y", "x"]].to_numpy(dtype=np.float64)).astype(np.int64)
    c_zyx = np.clip(c_zyx, 0, np.array(shape[1:], dtype=np.int64) - 1)
    c_s = cand["score"].to_numpy(dtype=np.float64)

    keep = []
    for t in np.unique(c_t):
        rows = np.flatnonzero(c_t == t)
        keep.append(rows[select_frame(c_zyx[rows], c_s[rows], NMS_UM)][:budget])
    sel = np.concatenate(keep) if keep else np.zeros(0, dtype=np.int64)
    pred_t, pred_zyx = c_t[sel], c_zyx[sel]

    nodes = {int(i): (int(pred_t[i]), int(pred_zyx[i, 0]), int(pred_zyx[i, 1]),
                      int(pred_zyx[i, 2])) for i in range(len(sel))}

    index = {int(t): np.flatnonzero(pred_t == t) for t in np.unique(pred_t)}
    points = {t: pred_zyx[i] * SCALE_ARR for t, i in index.items()}
    links = link_sequence(points, gate_um=GATE_UM, max_out_degree=MAX_OUT_DEGREE,
                          motion=MOTION)

    edges: list[tuple[int, int, float]] = []
    for t, ls in links.items():
        a, b = points[t], points[t + 1]
        for i, j, _cost in ls:
            # Recompute the raw separation for the confidence. `associate`
            # returns a cost measured from the MOTION-PREDICTED position, which
            # is the right thing to link on and the wrong thing to rank by --
            # two links of equal residual are not equally trustworthy if one
            # moved 1 um and the other 6.
            d = float(np.linalg.norm(a[i] - b[j]))
            edges.append((int(index[t][i]), int(index[t + 1][j]), 1.0 / (1.0 + d)))

    graph = DatasetGraph(dataset=name, nodes=nodes, edges=edges, shape=shape)
    linked = {s for s, _, _ in edges} | {d for _, d, _ in edges}
    n_edgeless = len(nodes) - len(linked)

    if drop_edgeless:
        from biohub.io.submission import drop_edgeless_nodes

        graph = drop_edgeless_nodes(graph)

    return TrackResult(
        graph=graph, n_candidates=len(cand), budget_per_frame=budget,
        predicted_per_frame=per_frame, n_selected=len(sel), n_edgeless=n_edgeless,
    )


def track_dataset_learned_v2(
    zarr_path: Path,
    models,
    count_model,
    edge_model,
    device,
    amp_dtype=None,
    batch: int = 8,
) -> TrackResult:
    """The Stage 4 operating point: detect, size at 1.25x, claims-link.

    Same detector as `track_dataset_learned`; everything downstream is the
    configuration measured at engineering OOF 0.8471 (stage-04 Â§8.3):
    1.25x predicted budget, flow-gated k=8/14 um edge universe, GBM ensemble +
    group calibration, winner-take-all claims with the division decision at
    tau2 (in-degree <= 1 by construction, out-degree 2 only when two children
    independently claim one parent).
    """
    import zarr

    from biohub.detector.infer import detect, volumes_from_zarr

    zarr_path = Path(zarr_path)
    name = zarr_path.stem
    shape = tuple(zarr.open_array(str(zarr_path / "0"), mode="r").shape)

    vols = volumes_from_zarr(zarr_path)
    cand = detect(models, vols, device, amp_dtype=amp_dtype, batch=batch)
    del vols
    gc.collect()

    return size_select_claims(name, shape, cand, count_model, edge_model,
                              image_quantiles(zarr_path), device=device)


def size_select_claims(
    name: str,
    shape: tuple[int, int, int, int],
    cand,
    count_model,
    edge_model,
    quantiles: dict,
    device=None,
) -> TrackResult:
    """Everything downstream of the detector, Stage 4 configuration.

    Node selection is `biohub.tracker.edges.select_nodes` â€” the SAME function
    the measurement rig runs â€” so the deployed node set and the measured node
    set cannot drift apart. Edges carry the group-calibrated probability as
    confidence: `build_submission` writes descending, which keeps each
    parent's first claim ahead of its second under the scorer's
    insertion-order truncation (fact 5).
    """
    from biohub.counts import features_from
    from biohub.tracker.edges import build_edges, select_nodes
    from biohub.tracker.linker import emit_claims

    feats = features_from(name, quantiles, cand)
    per_frame = float(count_model.predict_per_frame(feats)[0])
    budget = max(1, int(round(per_frame * BUDGET_MULT)))

    sel = select_nodes(cand, budget, nms_um=NMS_UM)
    sel_t = cand["t"].to_numpy()[sel]
    sel_zyx = np.round(
        cand[["z", "y", "x"]].to_numpy(dtype=np.float64)[sel]
    ).astype(np.int64)
    # The offset head refines a peak inside its voxel and can push one just
    # past the volume bound; the rig matches unclipped (a no-op difference at
    # matching radius), the submission must be in-bounds.
    sel_zyx = np.clip(sel_zyx, 0, np.array(shape[1:], dtype=np.int64) - 1)

    edges = build_edges(cand, sel, k=EDGE_K, gate_um=EDGE_GATE_UM)

    from biohub.tracker.edges import FEATURE_COLUMNS

    gbm_device = device if getattr(device, "type", None) == "cuda" else None
    p = edge_model.predict_p(
        edges[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32), device=gbm_device)
    p_adj = edge_model.rescore_argmax(edges, p)

    pi = edges["pi"].to_numpy()
    ci = edges["ci"].to_numpy()
    take = emit_claims(pi, ci, p_adj, TAU, TAU2)

    nodes = {int(i): (int(sel_t[i]), int(sel_zyx[i, 0]), int(sel_zyx[i, 1]),
                      int(sel_zyx[i, 2])) for i in range(len(sel))}
    out_edges = [(int(pi[r]), int(ci[r]), float(p_adj[r])) for r in take]

    n_forks = 0
    if len(take):
        _, counts = np.unique(pi[take], return_counts=True)
        n_forks = int((counts == 2).sum())

    graph = DatasetGraph(dataset=name, nodes=nodes, edges=out_edges, shape=shape)
    linked = {s for s, _, _ in out_edges} | {d for _, d, _ in out_edges}

    return TrackResult(
        graph=graph, n_candidates=len(cand), budget_per_frame=budget,
        predicted_per_frame=per_frame, n_selected=len(sel),
        n_edgeless=len(nodes) - len(linked), n_forks=n_forks,
    )


def track_dataset_learned_v3(
    zarr_path: Path,
    models,
    count_model,
    edge_model,
    device,
    amp_dtype=None,
    batch: int = 8,
) -> TrackResult:
    """The solver v1.1 operating point (official 0.8748, stage-04 Â§11).

    Same detector and same claims linker as `track_dataset_learned_v2`; the
    single change is NODE SELECTION, which becomes the measured joint
    decision: a 2x superset is edge-scored, support is iterated to
    selection-consistency under winner-take-all semantics, and the budget is
    spent through one ranking cut per sample instead of a per-frame quota
    (facts 47â€“48: +0.0177 over the reselection point, +0.0277 over v2's
    0.8471 configuration, engineering official).
    """
    import zarr

    from biohub.detector.infer import detect, volumes_from_zarr

    zarr_path = Path(zarr_path)
    name = zarr_path.stem
    shape = tuple(zarr.open_array(str(zarr_path / "0"), mode="r").shape)

    vols = volumes_from_zarr(zarr_path)
    cand = detect(models, vols, device, amp_dtype=amp_dtype, batch=batch)
    del vols
    gc.collect()

    return size_select_solve(name, shape, cand, count_model, edge_model,
                             image_quantiles(zarr_path), device=device)


def _candidate_score_vectors(cand):
    """Return detector, superset-ranking, and WTA-ranking score vectors."""
    c_score = cand["score"].to_numpy(dtype=np.float64)
    selection_score = (cand["selection_score"].to_numpy(dtype=np.float64)
                       if "selection_score" in cand else c_score)
    solver_selection_score = (
        cand["solver_selection_score"].to_numpy(dtype=np.float64)
        if "solver_selection_score" in cand else selection_score
    )
    if (selection_score.shape != c_score.shape
            or solver_selection_score.shape != c_score.shape
            or not np.isfinite(selection_score).all()
            or not np.isfinite(solver_selection_score).all()):
        raise ValueError("invalid specialist selection scores")
    return c_score, selection_score, solver_selection_score


def _solve_and_score(name, shape, cand, count_model, edge_model, quantiles,
                     device=None, superset_override=None,
                     count_candidate_override=None):
    """The shared v1.1 front half: solver selection + scored final edges.

    Returns everything the v4/v5 linkers consume. Byte-identical to the body
    of `size_select_solve` up to the claims emission â€” kept as one function so
    the three pipelines cannot drift apart on selection.
    """
    from biohub.counts import features_from
    from biohub.tracker.edges import FEATURE_COLUMNS, build_edges, select_nodes
    from biohub.tracker.solver import SUPERSET_MULT, solve_wta

    # Research-only expanded-solver replay can score a larger candidate union
    # while keeping the deployed count estimate anchored to the original
    # detector view. Defaults are byte-identical to every historical caller.
    count_cand = cand if count_candidate_override is None else count_candidate_override
    feats = features_from(name, quantiles, count_cand)
    per_frame = float(count_model.predict_per_frame(feats)[0])
    budget = max(1, int(round(per_frame * BUDGET_MULT)))

    gbm_device = device if getattr(device, "type", None) == "cuda" else None

    c_t = cand["t"].to_numpy()
    c_zyx = np.round(cand[["z", "y", "x"]].to_numpy(dtype=np.float64)).astype(np.int64)
    c_score, selection_score, solver_selection_score = _candidate_score_vectors(cand)

    if superset_override is None:
        sup = select_nodes(cand, max(1, int(round(budget * SUPERSET_MULT))),
                           nms_um=NMS_UM, score_values=selection_score)
    else:
        sup = np.asarray(superset_override, dtype=np.int64)
        if (
            sup.ndim != 1 or len(np.unique(sup)) != len(sup)
            or (len(sup) and (sup.min() < 0 or sup.max() >= len(cand)))
        ):
            raise ValueError("invalid superset_override")
    e_sup = build_edges(cand, sup, k=EDGE_K, gate_um=EDGE_GATE_UM)
    p_sup = edge_model.predict_p(
        e_sup[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32), device=gbm_device)
    solver_score = solver_selection_score[sup]
    if SCORE_FRAME_ALIGN:
        # V40b (registered): per-frame median-logit alignment of the ranking
        # vector only. Monotone within every frame, so the per-frame control
        # cut and NMS ordering are untouched; edge features, the superset,
        # and every other consumer keep the raw scores.
        clipped = np.clip(solver_score, 1e-6, 1 - 1e-6)
        aligned = np.log(clipped) - np.log1p(-clipped)
        sup_t = c_t[sup]
        for frame in np.unique(sup_t):
            mask = sup_t == frame
            aligned[mask] -= np.median(aligned[mask])
        solver_score = aligned
    rows = solve_wta(solver_score, c_t[sup], c_zyx[sup],
                     e_sup["pi"].to_numpy(), e_sup["ci"].to_numpy(),
                     p_sup, budget)
    sel = sup[rows]
    del e_sup, p_sup

    edges = build_edges(cand, sel, k=EDGE_K, gate_um=EDGE_GATE_UM)
    p = edge_model.predict_p(
        edges[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32), device=gbm_device)
    p_adj = edge_model.rescore_argmax(edges, p)
    return per_frame, budget, gbm_device, c_t, c_zyx, sel, edges, p, p_adj


def _pair_inputs(cand, sel, edges, p_adj, c_t):
    """Node-side arrays for `enumerate_pairs` + the continuation velocity.

    `vel` comes from the continuation-only claims pass (fork arm disabled),
    exactly as every pair measurement computed it.
    """
    from biohub.metrics.gt import SCALE_ARR
    from biohub.tracker.edges import EMB_COLS
    from biohub.tracker.linker import emit_claims

    node_um = cand[["z", "y", "x"]].to_numpy(np.float64)[sel] * SCALE_ARR
    node_scalars = {
        c: cand[c].to_numpy(np.float32)[sel]
        for c in ("score", "peak_width_um", "near_um", "density_10um",
                  "boundary_um")
    }
    node_emb = cand[list(EMB_COLS)].to_numpy(np.float32)[sel]
    pi = edges["pi"].to_numpy()
    ci = edges["ci"].to_numpy()
    base = emit_claims(pi, ci, p_adj, TAU, 2.0)
    vel = np.full((len(sel), 3), np.nan)
    vel[ci[base]] = node_um[ci[base]] - node_um[pi[base]]
    return node_um, node_scalars, node_emb, pi, ci, base, vel


def _graph_result(name, shape, cand, sel, c_t, c_zyx, pi, ci, take, conf,
                  per_frame, budget, node_zyx=None, keep=None) -> TrackResult:
    """Assemble the DatasetGraph from an emitted edge set.

    `node_zyx` overrides the rounded coordinates (smoothing); `keep` drops
    nodes outside it (edgeless cleanup) â€” dropped ids appear in no edge, so
    edge endpoints always resolve.
    """
    sel_t = c_t[sel]
    zyx = c_zyx[sel] if node_zyx is None else node_zyx
    zyx = np.clip(zyx, 0, np.array(shape[1:], dtype=np.int64) - 1)

    nodes = {int(i): (int(sel_t[i]), int(zyx[i, 0]), int(zyx[i, 1]),
                      int(zyx[i, 2]))
             for i in range(len(sel)) if keep is None or i in keep}
    out_edges = [(int(pi[r]), int(ci[r]), float(c))
                 for r, c in zip(take, conf)]

    n_forks = 0
    if len(take):
        _, counts = np.unique(pi[take], return_counts=True)
        n_forks = int((counts == 2).sum())

    graph = DatasetGraph(dataset=name, nodes=nodes, edges=out_edges, shape=shape)
    linked = {s for s, _, _ in out_edges} | {d for _, d, _ in out_edges}
    return TrackResult(
        graph=graph, n_candidates=len(cand), budget_per_frame=budget,
        predicted_per_frame=per_frame, n_selected=len(sel),
        n_edgeless=len(nodes) - len(linked), n_forks=n_forks,
    )


def track_dataset_learned_v4(
    zarr_path: Path,
    models,
    count_model,
    edge_model,
    pair_model,
    device,
    amp_dtype=None,
    batch: int = 8,
) -> TrackResult:
    """The 0.8810 operating point (stage-04 Â§13, fact 50).

    Solver v1.1 selection and the claims continuation are unchanged; the fork
    decision moves from the Ï„2 dial to the independent division score â€”
    `emit_second_by_pairs` at `TAU_DIV_PAIRS`, claimants only.
    """
    import zarr

    from biohub.detector.infer import detect, volumes_from_zarr

    zarr_path = Path(zarr_path)
    name = zarr_path.stem
    shape = tuple(zarr.open_array(str(zarr_path / "0"), mode="r").shape)

    vols = volumes_from_zarr(zarr_path)
    cand = detect(models, vols, device, amp_dtype=amp_dtype, batch=batch)
    del vols
    gc.collect()

    from biohub.tracker.pairs import (PAIR_FEATURES, emit_second_by_pairs,
                                      enumerate_pairs)

    per_frame, budget, _gbm_device, c_t, c_zyx, sel, edges, p, p_adj = \
        _solve_and_score(name, shape, cand, count_model, edge_model,
                         image_quantiles(zarr_path), device=device)
    node_um, node_scalars, node_emb, pi, ci, _base, vel = _pair_inputs(
        cand, sel, edges, p_adj, c_t)

    pairs = enumerate_pairs(edges, p, p_adj, c_t[sel], node_um,
                            node_scalars, node_emb, vel)
    pair_p = pair_model.predict_p(
        pairs[list(PAIR_FEATURES)].to_numpy(np.float32))
    take, conf = emit_second_by_pairs(pi, ci, p_adj, TAU, pairs, pair_p,
                                      TAU_DIV_PAIRS, recruit_orphans=False)
    return _graph_result(name, shape, cand, sel, c_t, c_zyx, pi, ci, take,
                         conf, per_frame, budget)


def track_dataset_learned_v5(
    zarr_path: Path,
    models,
    count_model,
    edge_model,
    ctx_model,
    pair_model,
    pair_stack,
    device,
    amp_dtype=None,
    batch: int = 8,
) -> TrackResult:
    """The 0.9021 operating point (stage-04 Â§13+, facts 54/59/60/61).

    On top of v4's selection: pass-2 context continuation (`ctx_model`, the
    retrained arm scored over `build_context` of a continuation-only first
    pass), the temporal division stack on the v0 pair prior at
    `TAU_DIV_STACK`, fork-guarded line smoothing at `SMOOTH_WEIGHT`, and the
    edgeless-node drop (pure count cost under the official metric, fact 61).
    """
    import zarr

    from biohub.detector.infer import detect, volumes_from_zarr

    zarr_path = Path(zarr_path)
    name = zarr_path.stem
    shape = tuple(zarr.open_array(str(zarr_path / "0"), mode="r").shape)

    vols = volumes_from_zarr(zarr_path)
    cand = detect(models, vols, device, amp_dtype=amp_dtype, batch=batch)
    del vols
    gc.collect()

    st = _stack_state(name, shape, cand, count_model, edge_model, ctx_model,
                      pair_model, pair_stack, image_quantiles(zarr_path),
                      device)
    take = st["take"]
    pi, ci = st["pi"], st["ci"]
    linked = set(pi[take].tolist()) | set(ci[take].tolist())
    return _graph_result(name, shape, cand, st["sel"], st["c_t"], st["c_zyx"],
                         pi, ci, take, st["conf"], st["per_frame"],
                         st["budget"], node_zyx=st["node_zyx"], keep=linked)


def _stack_state(name, shape, cand, count_model, edge_model, ctx_model,
                 pair_model, pair_stack, quantiles, device,
                 superset_override=None, count_candidate_override=None) -> dict:
    """Everything the v5 chain computes, kept for downstream post-passes.

    The operation order is byte-identical to the v5 body it was factored
    from; `track_dataset_learned_v5` assembles its graph from this state and
    `track_dataset_learned_v6` composes the 0.9208 post-pass on it.
    """
    import pandas as pd

    from biohub.metrics.gt import SCALE_ARR
    from biohub.tracker.context import build_context
    from biohub.tracker.edge_model import _logit
    from biohub.tracker.pairs import (PAIR_FEATURES, emit_second_by_pairs,
                                      enumerate_pairs)
    from biohub.tracker.smoothing import smooth_tracks

    per_frame, budget, gbm_device, c_t, c_zyx, sel, edges, p, p_adj = \
        _solve_and_score(
            name, shape, cand, count_model, edge_model, quantiles,
            device=device, superset_override=superset_override,
            count_candidate_override=count_candidate_override,
        )
    node_um, node_scalars, node_emb, pi, ci, take_cont, vel = _pair_inputs(
        cand, sel, edges, p_adj, c_t)

    # pass-2 continuation: context features from the continuation-only first
    # pass, the retrained arm's probability, its own group calibration
    ctx = build_context(edges, c_t[sel], node_um, p, p_adj, take_cont)
    full = pd.concat([edges, ctx], axis=1)
    p2 = ctx_model.predict_p(
        full[list(ctx_model.features)].to_numpy(np.float32), device=gbm_device)
    p_emit = ctx_model.rescore_argmax(full, p2)

    # temporal pairs + the faithful stack on the v0 prior
    pairs = enumerate_pairs(edges, p, p_adj, c_t[sel], node_um,
                            node_scalars, node_emb, vel, temporal=True)
    base_p = pair_model.predict_p(
        pairs[list(PAIR_FEATURES)].to_numpy(np.float32))
    pairs["base_pair_logit"] = _logit(
        np.asarray(base_p, dtype=np.float64)).astype(np.float32)
    pair_p = pair_stack.predict_p(
        pairs[list(pair_stack.features)].to_numpy(np.float32))
    # Preserve the exact fold-routed score as an ordinary deployable feature
    # for decision-matched research.  Existing models select their manifest
    # columns explicitly, so this does not change any frozen pipeline output.
    pairs["stack_pair_p"] = np.asarray(pair_p, dtype=np.float32)
    take, conf = emit_second_by_pairs(pi, ci, p_emit, TAU, pairs, pair_p,
                                      TAU_DIV_STACK, recruit_orphans=False)

    # fork-guarded smoothing over the ROUNDED submission coordinates â€” the
    # convention the 0.9021 sweep was measured at
    base_um = c_zyx[sel].astype(np.float64) * SCALE_ARR
    fitted_um = smooth_tracks(c_t[sel], base_um, pi[take], ci[take],
                              weight=1.0, radius=SMOOTH_RADIUS)
    smoothed = base_um + (fitted_um - base_um) * SMOOTH_WEIGHT
    node_zyx = np.rint(smoothed / SCALE_ARR).astype(np.int64)

    return {
        "per_frame": per_frame, "budget": budget, "c_t": c_t, "c_zyx": c_zyx,
        "sel": sel, "edges": edges, "full": full, "p": p, "p2": p2,
        "p_adj": p_adj, "p_emit": p_emit, "pi": pi, "ci": ci,
        "pairs": pairs, "take": take, "conf": conf, "node_zyx": node_zyx,
    }


# ---- the 0.9208 post-pass (stage-04 Â§23 / checkpoint-v4c-092): constants of
# the NEUTRALISED deployable variant. The checkpoint's stratum bars (0.10 /
# 0.30 by training embryo) cannot apply to a hidden embryo; the deployed rule
# forces the component model's is_6bba feature to 1.0 â€” the embryo carrying
# 84.6 % of the score weight â€” and prunes at one global bar. Measured OOF on
# the strict rig: 0.915239 against the frozen stratum config's 0.915174, so
# the neutralisation costs nothing locally. The frontier peaks at 0.30 and
# falls on both sides (0.20 = 0.9132, 0.35 = 0.9140, 0.40 = 0.9100).
PP_PAIR_ALPHA = 0.4    # blend weight on the long-track arm (logit_extra_a4)
PP_TAU_PAIR = 0.80     # blended second-daughter bar
PP_ASSIGN_TAU = 0.30   # greedy one-to-one bar on RAW pass-2 context p
PP_COMP_BAR = 0.30     # global component bar on the neutralised score
PP_IS_6BBA = 1.0       # the forced embryo-identity constant


def track_dataset_learned_v6(
    zarr_path: Path,
    models,
    count_model,
    edge_model,
    ctx_model,
    pair_model,
    pair_stack,
    pu_model,
    extra_model,
    comp_model,
    device,
    amp_dtype=None,
    batch: int = 8,
) -> TrackResult:
    """The 0.9208-checkpoint operating point, neutralised for hidden embryos.

    The full v5 chain runs unchanged, then the three post-pass actions of
    `Progress/checkpoint-v4c-092.md` compose on its state: the PU/long-track
    division blend at `PP_TAU_PAIR`, greedy one-to-one continuation
    assignment on the raw context probability at `PP_ASSIGN_TAU`, and whole-
    component pruning at `PP_COMP_BAR` with `is_6bba` forced to `PP_IS_6BBA`.
    Nodes keep v5's smoothed coordinates; only edge-incident nodes are
    submitted; edges carry uniform confidence exactly as the checkpoint
    scored them (degree constraints hold by construction, so confidence
    cannot affect the metric).
    """
    import zarr

    from biohub.detector.infer import detect, volumes_from_zarr

    zarr_path = Path(zarr_path)
    name = zarr_path.stem
    shape = tuple(zarr.open_array(str(zarr_path / "0"), mode="r").shape)

    vols = volumes_from_zarr(zarr_path)
    cand = detect(models, vols, device, amp_dtype=amp_dtype, batch=batch)
    del vols
    gc.collect()

    from biohub.tracker.postpass import (add_divisions, blend_pair_scores,
                                         component_features, greedy_assign,
                                         tracklet_features)

    st = _stack_state(name, shape, cand, count_model, edge_model, ctx_model,
                      pair_model, pair_stack, image_quantiles(zarr_path),
                      device)
    pi, ci, sel = st["pi"], st["ci"], st["sel"]
    node_t = st["c_t"][sel]
    cand_sel = cand.iloc[sel].reset_index(drop=True)

    # 1. division blend on the claims pairs
    claims = st["pairs"][st["pairs"].is_claims_pair].reset_index(drop=True)
    p_pu = pu_model.predict_p(
        claims[list(pu_model.features)].to_numpy(np.float32))
    lt = tracklet_features(node_t, st["node_zyx"], pi, ci, st["take"],
                          st["conf"], cand_sel, claims)
    import pandas as pd

    x_long = pd.concat([claims, lt], axis=1)
    p_long = extra_model.predict_p(
        x_long[list(extra_model.features)].to_numpy(np.float32))
    blend = blend_pair_scores(p_pu, p_long, PP_PAIR_ALPHA)

    # 2. greedy assignment base + blended divisions
    base = greedy_assign(pi, ci, st["p2"], PP_ASSIGN_TAU)
    take6 = add_divisions(base, pi, ci, st["p_emit"], claims, blend,
                          PP_TAU_PAIR)

    # 3. whole-component pruning, mask defined on the ORIGINAL v5 graph
    comp_x, root, roots = component_features(
        node_t, st["c_zyx"][sel], st["node_zyx"], pi, ci, st["take"],
        st["conf"], st["p_emit"], cand_sel, st["full"], is_6bba=PP_IS_6BBA)
    comp_p = comp_model.predict_p(
        comp_x[list(comp_model.features)].to_numpy(np.float64))
    comp_score = np.full(len(sel), -np.inf, dtype=np.float32)
    comp_score[roots] = comp_p
    node_keep = comp_score[root] >= PP_COMP_BAR
    take6 = take6[node_keep[pi[take6]] & node_keep[ci[take6]]]

    linked = set(pi[take6].tolist()) | set(ci[take6].tolist())
    conf6 = np.ones(len(take6), dtype=np.float32)
    return _graph_result(name, shape, cand, sel, st["c_t"], st["c_zyx"],
                         pi, ci, take6, conf6, st["per_frame"], st["budget"],
                         node_zyx=st["node_zyx"], keep=linked)


# ---- the 0.9289 chain (checkpoint-v4c-0929), neutralised: every constant is
# the measured one. Component neutralisation re-measured for the current-graph
# model: is_6bba forced to 1.0 + ONE global bar 0.30 reads 0.920726 strict vs
# the frozen stratum config's 0.920711; the stub pruner evaluated LIVE on the
# final topology (deploy has no transfer table) reads 0.920768 â€” parity or
# better on every axis. Official confirmation of the deployable variant is in
# `reports/tp_neutral_livestub_official_rows.parquet`.
PP7_HR_GATE = 0.50     # extract source-res features only above this base score
PP7_HR_WEIGHT = 0.40   # high-res arm's logit weight inside the gate
PP7_TAU_PAIR = 0.65    # refined second-daughter bar
PP7_ASSIGN_TAU = 0.30  # greedy one-to-one bar on RAW pass-2 context p
PP7_COMP_BAR = 0.30    # global bar on the neutralised current-graph component
PP7_LEAF_BAR = 0.725   # premature-leaf raw-context bar (tail arm)
PP7_HEAD_BAR = 0.50    # premature-root live-stub bar (head arm)


def _division_base_scores(zarr_path, st, cand_sel, node_t, sel, pi, ci, vols,
                          allimg_model, full_model, hardneg_model,
                          highres_model):
    """The 0.9289 division base: three-model blend + high-res refinement.

    Returns (claims, lt, refined) â€” the claims-pair frame, its tracklet
    features, and the refined pair score. Shared by pipelines v7 and v8.
    """
    import pandas as pd
    import zarr

    from biohub.detector.targets import src_to_grid
    from biohub.tracker.edges import EMB_COLS
    from biohub.tracker.postpass import (checkpoint_base_blend,
                                         highres_pair_features, next_map,
                                         pair_embedding_features,
                                         pair_image_features,
                                         refine_with_highres,
                                         tracklet_features)

    claims = st["pairs"][st["pairs"].is_claims_pair].reset_index(drop=True)
    lt = tracklet_features(node_t, st["node_zyx"], pi, ci, st["take"],
                           st["conf"], cand_sel, claims)
    nxt = next_map(len(sel), pi, ci, st["take"], st["conf"])
    node_grid = src_to_grid(
        cand_sel[["z", "y", "x"]].to_numpy(np.float32)).astype(np.float32)
    img = pair_image_features(vols, node_t, node_grid, nxt, claims)
    ev = pair_embedding_features(
        node_t, cand_sel[list(EMB_COLS)].to_numpy(np.float32), claims)
    base_frame = pd.concat([claims, lt, img, ev], axis=1)
    p_allimg = allimg_model.predict_p(
        base_frame[list(allimg_model.features)].to_numpy(np.float32))
    p_full = full_model.predict_p(
        base_frame[list(full_model.features)].to_numpy(np.float32))
    p_hard = hardneg_model.predict_p(
        base_frame[list(hardneg_model.features)].to_numpy(np.float32))
    base = checkpoint_base_blend(p_allimg, p_full, p_hard)

    gate = base >= PP7_HR_GATE
    refined = base.copy()
    if gate.any():
        arr = zarr.open_array(str(Path(zarr_path) / "0"), mode="r")
        node_src = cand_sel[["z", "y", "x"]].to_numpy(np.float32)
        hr = highres_pair_features(arr, node_t, node_src,
                                   claims[gate].reset_index(drop=True))
        p_hr = highres_model.predict_p(
            hr[list(highres_model.features)].to_numpy(np.float32))
        hr_full = np.full(len(claims), 0.5, dtype=np.float32)
        hr_full[np.flatnonzero(gate)] = p_hr
        refined = refine_with_highres(base, hr_full, gate, PP7_HR_WEIGHT)
    return claims, lt, refined


def track_dataset_learned_v7(
    zarr_path: Path,
    models,
    count_model,
    edge_model,
    ctx_model,
    pair_model,
    pair_stack,
    allimg_model,
    full_model,
    hardneg_model,
    highres_model,
    comp_model,
    stub_model,
    device,
    amp_dtype=None,
    batch: int = 8,
) -> TrackResult:
    """The 0.9289-checkpoint operating point, neutralised for hidden embryos.

    On the v5 state: the three-model division base (grid-image morphology,
    full embeddings, hard-negative arm; 0.72/0.18/0.10 in logit space),
    source-resolution refinement inside the `PP7_HR_GATE` (weight
    `PP7_HR_WEIGHT`), greedy assignment + divisions at `PP7_TAU_PAIR`,
    current-graph component pruning at `PP7_COMP_BAR` with is_6bba forced,
    and one live-stub terminal-pruning pass.
    """
    import zarr

    from biohub.detector.infer import detect, volumes_from_zarr

    zarr_path = Path(zarr_path)
    name = zarr_path.stem
    shape = tuple(zarr.open_array(str(zarr_path / "0"), mode="r").shape)

    vols = volumes_from_zarr(zarr_path)
    cand = detect(models, vols, device, amp_dtype=amp_dtype, batch=batch)
    # v7 keeps the detector-grid volume: the division base reads morphology
    # from it (~100 MB, freed at return).
    gc.collect()

    import pandas as pd

    from biohub.detector.targets import src_to_grid
    from biohub.tracker.postpass import (add_divisions, checkpoint_base_blend,
                                         component_features, greedy_assign,
                                         highres_pair_features, next_map,
                                         pair_embedding_features,
                                         pair_image_features,
                                         refine_with_highres, stub_features,
                                         tail_prune, tracklet_features)

    st = _stack_state(name, shape, cand, count_model, edge_model, ctx_model,
                      pair_model, pair_stack, image_quantiles(zarr_path),
                      device)
    pi, ci, sel = st["pi"], st["ci"], st["sel"]
    node_t = st["c_t"][sel]
    cand_sel = cand.iloc[sel].reset_index(drop=True)

    claims, lt, refined = _division_base_scores(
        zarr_path, st, cand_sel, node_t, sel, pi, ci, vols,
        allimg_model, full_model, hardneg_model, highres_model)
    del vols
    gc.collect()

    # ---- assignment + divisions + current-graph component pruning ----------
    take7 = greedy_assign(pi, ci, st["p2"], PP7_ASSIGN_TAU)
    take7 = add_divisions(take7, pi, ci, st["p_emit"], claims, refined,
                          PP7_TAU_PAIR)
    comp_x, root, roots = component_features(
        node_t, st["c_zyx"][sel], st["node_zyx"], pi, ci, take7,
        st["p2"][take7], st["p_emit"], cand_sel, st["full"],
        is_6bba=PP_IS_6BBA)
    comp_p = comp_model.predict_p(
        comp_x[list(comp_model.features)].to_numpy(np.float64))
    comp_score = np.full(len(sel), -np.inf, dtype=np.float32)
    comp_score[roots] = comp_p
    node_keep = comp_score[root] >= PP7_COMP_BAR
    take7 = take7[node_keep[pi[take7]] & node_keep[ci[take7]]]

    # ---- one live-stub terminal-pruning pass -------------------------------
    sf, chosen = stub_features(node_t, pi, ci, take7, st["p2"], st["p_emit"],
                               st["full"])
    head_conf = np.full(len(pi), np.inf, dtype=np.float32)
    if len(chosen):
        head_conf[chosen] = stub_model.predict_p(
            sf[list(stub_model.features)].to_numpy(np.float32))
    take7 = tail_prune(take7, node_t, pi, ci, st["p2"], head_conf,
                       PP7_LEAF_BAR, PP7_HEAD_BAR)

    linked = set(pi[take7].tolist()) | set(ci[take7].tolist())
    conf7 = np.ones(len(take7), dtype=np.float32)
    return _graph_result(name, shape, cand, sel, st["c_t"], st["c_zyx"],
                         pi, ci, take7, conf7, st["per_frame"], st["budget"],
                         node_zyx=st["node_zyx"], keep=linked)


# ---- the 0.933 chain (checkpoint-v4c-0933), neutralised: multiview
# component mask (ep02-only persistence â€” the ep05 view is degenerate at
# deploy since every node is its own nearest ep05-ensemble candidate),
# fork confirmation, live-stub terminal pruning, decision-matched edge veto,
# radius-4 fractional smoothing. The neutral variant's exact local score is
# recorded in reports/v4c_0933_neutral_official_rows.parquet.
PP8_COMP_BAR = 0.30      # global bar inside the geometric blend, both models
PP8_GEOM_NEW_W = 0.2     # log-space weight of the multiview arm
PP8_CONFIRM_TAU = 0.325  # fork-confirmation bar (reference forks only)
PP8_VETO_TAU = 0.045     # post_lgb_l31 continuation veto; fork arms protected
PP8_SMOOTH_R = 4         # radius-4 line fit on FRACTIONAL detector centres
PP8_SMOOTH_W = 0.700     # fractional -> fitted blend
PP8_SMOOTH_A = 0.575     # old-coordinate -> fresh blend
PP8_GUARD_HOPS = 2       # fork neighbourhood that keeps old coordinates


def track_dataset_learned_v8(
    zarr_path: Path,
    models,
    models_ep02,
    count_model,
    edge_model,
    ctx_model,
    pair_model,
    pair_stack,
    allimg_model,
    full_model,
    hardneg_model,
    highres_model,
    stub_model,
    comp_old_model,
    comp_mv_model,
    confirm_model,
    track_model,
    post_model,
    device,
    amp_dtype=None,
    batch: int = 8,
    return_state: bool = False,
    include_volumes_in_state: bool = False,
    runtime_view_models: dict | None = None,
    component_is_6bba: float | None = None,
    component_bar: float | None = None,
    smoothing_weight: float | None = None,
    smoothing_alpha: float | None = None,
    heat_alignment: str = "probability_mean",
    detector_vector_model_count: int | None = None,
    candidate_override=None,
    superset_override=None,
    count_candidate_override=None,
    volumes_override=None,
    image_quantiles_override: dict | None = None,
    collect_garbage: bool = True,
    return_correspondence_state: bool = False,
) -> TrackResult:
    """Run the 0.933 family with neutral or research component semantics.

    Defaults preserve the deployed v14 policy.  The optional component and
    smoothing arguments expose the original annotation-free research policy
    when the hidden dataset name supplies its acquisition-domain prefix.
    """
    import zarr

    from biohub.detector.infer import detect, volumes_from_zarr

    zarr_path = Path(zarr_path)
    name = zarr_path.stem
    shape = tuple(zarr.open_array(str(zarr_path / "0"), mode="r").shape)

    # Deployment compositions may evaluate several detector/solver branches
    # for the same immutable movie.  Accepting the already-normalised volume
    # avoids reading, normalising, and downsampling the Zarr once per branch.
    # The historical path remains the default for every existing caller.
    vols = (
        volumes_from_zarr(zarr_path)
        if volumes_override is None
        else volumes_override
    )
    if candidate_override is None:
        cand = detect(
            models, vols, device, amp_dtype=amp_dtype, batch=batch,
            heat_alignment=heat_alignment,
            vector_model_count=detector_vector_model_count,
        )
    else:
        # Research replay path: a detector candidate table persisted by an
        # earlier annotation-free invocation can be passed back through the
        # exact downstream stack.  Defaults and hidden inference are unchanged.
        # Copy the frame rather than mutating the persisted evidence in place.
        cand = candidate_override.copy()
    runtime_views = {}
    if runtime_view_models:
        for tag, view_models in runtime_view_models.items():
            runtime_views[tag] = detect(
                view_models, vols, device, amp_dtype=amp_dtype, batch=batch,
            )[["t", "z", "y", "x", "score", "rank_in_frame"]]
    # the ep02 ensemble is an independent detector view for persistence â€”
    # only needed when the multiview component arm deploys
    pool02 = None
    if comp_mv_model is not None:
        if "ep02" in runtime_views:
            pool02 = runtime_views["ep02"]
        else:
            pool02 = detect(models_ep02, vols, device, amp_dtype=amp_dtype,
                            batch=batch)[["t", "z", "y", "x", "score",
                                          "rank_in_frame"]]
    if collect_garbage:
        gc.collect()

    import pandas as pd

    from biohub.metrics.gt import SCALE_ARR
    from biohub.tracker.edge_model import _logit
    from biohub.tracker.postpass import (add_divisions, component_features,
                                         graph_roots, greedy_assign,
                                         stub_features, tail_prune)
    from biohub.tracker.postpass33 import (
        changed_hop_guard, edge_tracklet_features,
        intensity_component_features, persistence_component_features,
        persistence_component_features_multi, postselect_features)
    from biohub.tracker.smoothing import smooth_tracks

    quantiles = (
        image_quantiles(zarr_path)
        if image_quantiles_override is None
        else image_quantiles_override
    )
    st = _stack_state(
        name, shape, cand, count_model, edge_model, ctx_model,
        pair_model, pair_stack, quantiles, device,
        superset_override=superset_override,
        count_candidate_override=count_candidate_override,
    )
    pi, ci, sel = st["pi"], st["ci"], st["sel"]
    n = len(sel)
    node_t = st["c_t"][sel]
    cand_sel = cand.iloc[sel].reset_index(drop=True)

    claims, lt, refined = _division_base_scores(
        zarr_path, st, cand_sel, node_t, sel, pi, ci, vols,
        allimg_model, full_model, hardneg_model, highres_model)

    # ---- reference graph + multiview component mask ------------------------
    assignment = greedy_assign(pi, ci, st["p2"], PP7_ASSIGN_TAU)
    take_ref = add_divisions(assignment, pi, ci, st["p_emit"], claims,
                             refined, PP7_TAU_PAIR)
    actual_is_6bba = (PP_IS_6BBA if component_is_6bba is None
                      else float(component_is_6bba))
    actual_component_bar = (PP8_COMP_BAR if component_bar is None
                            else float(component_bar))
    comp_x, root, roots = component_features(
        node_t, st["c_zyx"][sel], st["node_zyx"], pi, ci, take_ref,
        st["p2"][take_ref], st["p_emit"], cand_sel, st["full"],
        is_6bba=actual_is_6bba)
    old_p = comp_old_model.predict_p(
        comp_x[list(comp_old_model.features)].to_numpy(np.float64))
    if comp_mv_model is None:
        # the measured deployable rule: multiview arm dropped, v13's
        # component policy (old model, global 0.30 bar)
        geom = old_p / actual_component_bar
    else:
        _, nodes, _ = graph_roots(node_t, pi, ci, take_ref)
        int_x = intensity_component_features(vols, node_t, cand_sel, root,
                                             nodes, roots)
        if runtime_views:
            per_x = persistence_component_features_multi(
                node_t, st["c_zyx"][sel], runtime_views, root, nodes, roots)
        else:
            per_x = persistence_component_features(
                node_t, st["c_zyx"][sel], pool02, root, nodes, roots)
        merged = pd.concat([comp_x.reset_index(drop=True),
                            int_x.reset_index(drop=True),
                            per_x.reset_index(drop=True)], axis=1)
        mv_x = pd.DataFrame({
            c: (merged[c].to_numpy(np.float32) if c in merged.columns
                else np.full(len(merged), np.nan, dtype=np.float32))
            for c in comp_mv_model.features
        })
        new_p = comp_mv_model.predict_p(mv_x.to_numpy(np.float64))
        geom = np.exp(0.8 * np.log(np.maximum(
                          old_p / actual_component_bar, 1e-6))
                      + PP8_GEOM_NEW_W * np.log(np.maximum(
                          new_p / actual_component_bar, 1e-6)))
    if not (return_state and include_volumes_in_state):
        del vols
    del pool02
    if collect_garbage:
        gc.collect()
    comp_score = np.full(n, -np.inf, dtype=np.float32)
    comp_score[roots] = geom.astype(np.float32)
    node_keep = comp_score[root] >= 1.0

    # ---- fork confirmation on the reference-emitted forks ------------------
    ref_forks = np.setdiff1d(take_ref, assignment)
    conf_score = np.zeros(len(claims), dtype=np.float32)
    e1 = claims.e1.to_numpy(np.int64)
    e2 = claims.e2.to_numpy(np.int64)
    emitted_pair = np.isin(e1, ref_forks) | np.isin(e2, ref_forks)
    if emitted_pair.any():
        cf = pd.concat([claims[emitted_pair].reset_index(drop=True),
                        lt[emitted_pair].reset_index(drop=True)], axis=1)
        cf["logit_current"] = _logit(
            refined[emitted_pair].astype(np.float64)).astype(np.float32)
        conf_score[np.flatnonzero(emitted_pair)] = confirm_model.predict_p(
            cf[list(confirm_model.features)].to_numpy(np.float32))
    take = add_divisions(assignment, pi, ci, st["p_emit"], claims,
                         conf_score, PP8_CONFIRM_TAU)
    take = take[node_keep[pi[take]] & node_keep[ci[take]]]

    # ---- live-stub terminal pruning (confirmed side) -----------------------
    def _prune(edges_take):
        sf, chosen = stub_features(node_t, pi, ci, edges_take, st["p2"],
                                   st["p_emit"], st["full"])
        head_conf = np.full(len(pi), np.inf, dtype=np.float32)
        if len(chosen):
            head_conf[chosen] = stub_model.predict_p(
                sf[list(stub_model.features)].to_numpy(np.float32))
        return tail_prune(edges_take, node_t, pi, ci, st["p2"], head_conf,
                          PP7_LEAF_BAR, PP7_HEAD_BAR)

    take = _prune(take)

    # ---- decision-matched edge veto ----------------------------------------
    # features on the REFERENCE-side chain (mask + prune), exactly as trained;
    # scores transfer by edge index to the confirmed graph
    take_ps = take_ref[node_keep[pi[take_ref]] & node_keep[ci[take_ref]]]
    take_ps = _prune(take_ps)
    tracklet, trows = edge_tracklet_features(
        node_t, st["c_zyx"][sel], pi, ci, st["p2"], st["p_emit"], assignment,
        cand_sel)
    # the trainers derive raw_logit from raw_p before fitting
    tracklet["raw_logit"] = _logit(
        tracklet["raw_p"].to_numpy(np.float64)).astype(np.float32)
    # track_model is a dict of the three score columns the postselect
    # features were trained with; each model reads its own manifest order
    track_scores = {
        name_: m.predict_p(tracklet[list(m.features)].to_numpy(np.float64))
        for name_, m in track_model.items()
    }
    ps_x, sel_edges = postselect_features(
        node_t, pi, ci, st["p2"], take_ps, root, comp_score, tracklet, trows,
        track_scores, st["full"])
    post_p = post_model.predict_p(
        ps_x[list(post_model.features)].to_numpy(np.float64))
    veto = np.full(len(pi), np.inf, dtype=np.float32)
    veto[sel_edges] = post_p
    out_deg = np.bincount(pi[take], minlength=n)
    fork_edge = out_deg[pi[take]] >= 2
    take = take[(veto[take] >= PP8_VETO_TAU) | fork_edge]

    # ---- radius-4 fractional smoothing -------------------------------------
    subvox_um = cand_sel[["z", "y", "x"]].to_numpy(np.float64) * SCALE_ARR
    old_um = st["node_zyx"].astype(np.float64) * SCALE_ARR
    fit = smooth_tracks(node_t, subvox_um, pi[take], ci[take], weight=1.0,
                        radius=PP8_SMOOTH_R)
    smooth_weight = (PP8_SMOOTH_W if smoothing_weight is None
                     else float(smoothing_weight))
    smooth_alpha = (PP8_SMOOTH_A if smoothing_alpha is None
                    else float(smoothing_alpha))
    fresh = subvox_um + smooth_weight * (fit - subvox_um)
    final_um = old_um + smooth_alpha * (fresh - old_um)
    guard = changed_hop_guard(n, pi, ci, take, hops=PP8_GUARD_HOPS)
    final_um[guard] = old_um[guard]
    node_zyx = np.rint(final_um / SCALE_ARR).astype(np.int64)

    linked = set(pi[take].tolist()) | set(ci[take].tolist())
    conf8 = np.ones(len(take), dtype=np.float32)
    result = _graph_result(
        name, shape, cand, sel, st["c_t"], st["c_zyx"], pi, ci, take, conf8,
        st["per_frame"], st["budget"], node_zyx=node_zyx, keep=linked,
    )
    if return_correspondence_state:
        result.correspondence_state = {key: st[key].copy() for key in ("pi", "ci", "p", "p2")}
    if return_state:
        # Everything below is produced by hidden-test inference itself.  It
        # deliberately excludes folds, labels, GT matches, and training-only
        # tables.  Edge indices remain the stable row indices of ``st['full']``
        # and are shared by every downstream materializer.
        pair_state = claims[[
            column for column in (
                "par", "c1", "c2", "e1", "e2", "is_claims_pair"
            ) if column in claims
        ]].copy()
        pair_state.insert(0, "dataset", name)
        pair_state["hr_lr_c0p01_w4"] = refined.astype(np.float32)
        pair_state["confirm_compact_score_c0p03"] = conf_score.astype(
            np.float32)
        # V21 historically exposes only claims pairs to the high-resolution
        # division classifier.  V28 needs the complete, still label-free pair
        # universe to learn whether a contested second daughter should replace
        # an ordinary continuation.  Frozen V21 scores are attached where they
        # exist; non-claims pairs remain explicit missing values, not zeros.
        pair_hypotheses = st["pairs"].copy()
        pair_hypotheses.insert(0, "dataset", name)
        pair_hypotheses = pair_hypotheses.merge(
            pair_state[[
                "dataset", "par", "c1", "c2", "hr_lr_c0p01_w4",
                "confirm_compact_score_c0p03",
            ]],
            on=["dataset", "par", "c1", "c2"], how="left",
            validate="one_to_one",
        )
        final_active = np.zeros(len(pi), dtype=bool)
        final_active[take] = True
        pair_hypotheses["v21_final_fork"] = (
            final_active[pair_hypotheses.e1.to_numpy(np.int64)]
            & final_active[pair_hypotheses.e2.to_numpy(np.int64)]
        )
        pair_tracklet = pd.concat([
            pair_state[["dataset", "par", "c1", "c2"]].reset_index(drop=True),
            lt.reset_index(drop=True),
        ], axis=1)
        pair_tracklet["logit_current"] = _logit(
            refined.astype(np.float64)).astype(np.float32)
        post_state = pd.DataFrame({
            "dataset": name,
            "edge_idx": sel_edges.astype(np.int32),
            "post_lgb_l31": post_p.astype(np.float32),
        })
        # Preserve the exact label-free matrix consumed by the post-selection
        # ensemble.  The portable chain also evaluates the saved l7/l15
        # variants on these same rows for downstream neighborhood features.
        post_features_state = ps_x.reset_index(drop=True).copy()
        post_features_state["edge_idx"] = sel_edges.astype(np.int32)
        if "dataset" in post_features_state:
            post_features_state["dataset"] = name
        else:
            post_features_state.insert(0, "dataset", name)
        cache = {
            "sel": sel.astype(np.int32),
            "node_t": node_t.astype(np.int32),
            "base_zyx": st["c_zyx"][sel].astype(np.float32),
            "final_zyx": node_zyx.astype(np.int32),
            "pi": pi.astype(np.int32),
            "ci": ci.astype(np.int32),
            "p_emit": st["p_emit"].astype(np.float32),
            "take": take.astype(np.int32),
            "conf": conf8,
        }
        inference_state = {
            "cache": cache,
            # These tables are immutable after this point.  Reusing them here
            # avoids two full hidden-sample copies immediately before export.
            "candidates": cand,
            "edges": st["full"],
            "raw_p": st["p2"].astype(np.float32),
            "pairs": pair_state,
            "pair_hypotheses": pair_hypotheses,
            "pair_tracklet": pair_tracklet,
            "post_scores": post_state,
            "post_features": post_features_state,
            "checkpoint_candidates": runtime_views,
        }
        if include_volumes_in_state:
            inference_state["volumes"] = vols
        result.inference_state = inference_state
    return result


def size_select_solve(
    name: str,
    shape: tuple[int, int, int, int],
    cand,
    count_model,
    edge_model,
    quantiles: dict,
    device=None,
) -> TrackResult:
    """Everything downstream of the detector, solver v1.1 configuration.

    Selection runs through `biohub.tracker.solver.solve_wta` â€” the SAME
    function the measurement rig runs (the `select_nodes` principle), fed by
    the edge ensemble's raw mean probability over the 2x-superset edge
    universe. The final relink is byte-identical to `size_select_claims`:
    group-calibrated argmax claims at tau 0.47 / tau2 0.80.
    """
    from biohub.counts import features_from
    from biohub.tracker.edges import FEATURE_COLUMNS, build_edges, select_nodes
    from biohub.tracker.linker import emit_claims
    from biohub.tracker.solver import SUPERSET_MULT, solve_wta

    feats = features_from(name, quantiles, cand)
    per_frame = float(count_model.predict_per_frame(feats)[0])
    budget = max(1, int(round(per_frame * BUDGET_MULT)))

    gbm_device = device if getattr(device, "type", None) == "cuda" else None

    c_t = cand["t"].to_numpy()
    c_zyx = np.round(cand[["z", "y", "x"]].to_numpy(dtype=np.float64)).astype(np.int64)
    c_score = cand["score"].to_numpy(dtype=np.float64)

    sup = select_nodes(cand, max(1, int(round(budget * SUPERSET_MULT))),
                       nms_um=NMS_UM)
    e_sup = build_edges(cand, sup, k=EDGE_K, gate_um=EDGE_GATE_UM)
    p_sup = edge_model.predict_p(
        e_sup[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32), device=gbm_device)
    solver_score = c_score[sup]
    if SCORE_FRAME_ALIGN:
        # V40b (registered): per-frame median-logit alignment of the ranking
        # vector only. Monotone within every frame, so the per-frame control
        # cut and NMS ordering are untouched; edge features, the superset,
        # and every other consumer keep the raw scores.
        clipped = np.clip(solver_score, 1e-6, 1 - 1e-6)
        aligned = np.log(clipped) - np.log1p(-clipped)
        sup_t = c_t[sup]
        for frame in np.unique(sup_t):
            mask = sup_t == frame
            aligned[mask] -= np.median(aligned[mask])
        solver_score = aligned
    rows = solve_wta(solver_score, c_t[sup], c_zyx[sup],
                     e_sup["pi"].to_numpy(), e_sup["ci"].to_numpy(),
                     p_sup, budget)
    sel = sup[rows]
    del e_sup, p_sup

    sel_t = c_t[sel]
    # In-bounds for the submission only; selection and matching ran unclipped
    # (a no-op difference at the matching radius â€” same note as v2).
    sel_zyx = np.clip(c_zyx[sel], 0, np.array(shape[1:], dtype=np.int64) - 1)

    edges = build_edges(cand, sel, k=EDGE_K, gate_um=EDGE_GATE_UM)
    p = edge_model.predict_p(
        edges[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32), device=gbm_device)
    p_adj = edge_model.rescore_argmax(edges, p)

    pi = edges["pi"].to_numpy()
    ci = edges["ci"].to_numpy()
    take = emit_claims(pi, ci, p_adj, TAU, TAU2)

    nodes = {int(i): (int(sel_t[i]), int(sel_zyx[i, 0]), int(sel_zyx[i, 1]),
                      int(sel_zyx[i, 2])) for i in range(len(sel))}
    out_edges = [(int(pi[r]), int(ci[r]), float(p_adj[r])) for r in take]

    n_forks = 0
    if len(take):
        _, counts = np.unique(pi[take], return_counts=True)
        n_forks = int((counts == 2).sum())

    graph = DatasetGraph(dataset=name, nodes=nodes, edges=out_edges, shape=shape)
    linked = {s for s, _, _ in out_edges} | {d for _, d, _ in out_edges}

    return TrackResult(
        graph=graph, n_candidates=len(cand), budget_per_frame=budget,
        predicted_per_frame=per_frame, n_selected=len(sel),
        n_edgeless=len(nodes) - len(linked), n_forks=n_forks,
    )
