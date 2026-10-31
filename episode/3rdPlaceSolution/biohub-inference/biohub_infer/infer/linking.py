"""Matcher scoring and the production linking cell: candidate scores between consecutive frames
(build_matcher_scores), their bidirectional / reverse / symmetric probabilities, and the drift-compensated
Hungarian on cost = distance + lambda * (1 - p) (link_embed).
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


from biohub_infer.core.matcher import NEG_INF
from biohub_infer.core.metric import (SCALE)


@torch.no_grad()
def build_matcher_scores(per_t, model, cfg, device, candidate_drifts=None):
    """Raw pair/null logits for each frame boundary.

    Unlike the old probability-only bridge, this preserves both null heads so
    inference can make an explicit birth/death decision. candidate_drifts, if
    supplied, is a list of physical (µm) frame-a offsets and changes both the
    matcher candidate ball and its relative-position features.
    """
    radius = float(cfg.loss.embed_radius_um)
    scale = torch.tensor(SCALE, dtype=torch.float32, device=device)
    scores = []
    for t in range(len(per_t) - 1):
        pa, ea = per_t[t]
        pb, eb = per_t[t + 1]
        if len(pa) == 0 or len(pb) == 0:
            scores.append(dict(
                logits=np.full((len(pa), len(pb)), NEG_INF, np.float32),
                null_a=np.zeros(len(pa), np.float32),
                null_b=np.zeros(len(pb), np.float32)))
            continue
        fa = torch.from_numpy(ea).to(device)
        fb = torch.from_numpy(eb).to(device)
        ga = torch.from_numpy(pa.astype(np.float32)).to(device) * scale
        gb = torch.from_numpy(pb.astype(np.float32)).to(device) * scale
        offset = None
        if candidate_drifts is not None:
            offset = torch.as_tensor(candidate_drifts[t], device=device,
                                     dtype=torch.float32).expand(len(pa), 3)
        # the matcher stays fp32 even under test.amp: autocast casts per op,
        # so MHA would run its QK^T in fp16 and overflow to NaN whatever dtype
        # the inputs are
        with torch.autocast(fa.device.type, enabled=False):
            logits, null_a, null_b = model.matcher(
                fa.float(), ga, fb.float(), gb, radius,
                candidate_offset_a=offset)
        scores.append(dict(
            logits=logits.cpu().numpy().astype(np.float32),
            null_a=null_a.cpu().numpy().astype(np.float32),
            null_b=null_b.cpu().numpy().astype(np.float32)))
    return scores


def matcher_bidir_probs(scores):
    """Direction-averaged link probability: mean of the a->b row softmax
    (competing against a's candidates + null_a) and the b->a column softmax
    (competing against b's candidates + null_b).  Unlike matcher_sym_probs
    this KEEPS the competition normalization in both directions and just
    fuses the two views — a pair is confident only if it wins its race from
    BOTH endpoints."""
    sims = []
    for s in scores:
        logits = torch.from_numpy(s["logits"])
        if not logits.numel():
            sims.append(np.zeros_like(s["logits"], dtype=np.float32))
            continue
        na = torch.from_numpy(s["null_a"])[:, None]
        nb = torch.from_numpy(s["null_b"])[None, :]
        p_ab = torch.softmax(torch.cat([logits, na], 1), 1)[:, :-1]
        p_ba = torch.softmax(torch.cat([logits, nb], 0), 0)[:-1, :]
        sims.append(((p_ab + p_ba) / 2).numpy().astype(np.float32))
    return sims


def matcher_rev_probs(scores):
    """Column softmax only: for each frame-b node, its distribution over
    candidate PARENTS (competing with null_b).

    The a->b row softmax is the wrong direction for divisions — a dividing
    parent's two daughters split its row mass, so the second child is
    structurally suppressed (measured: p50 0.18 on the parent->daughter pairs
    the one-to-one wiring had to drop). Each daughter, however, has exactly one
    parent, so the reverse direction does not split.
    """
    sims = []
    for s in scores:
        logits = torch.from_numpy(s["logits"])
        if not logits.numel():
            sims.append(np.zeros_like(s["logits"], dtype=np.float32))
            continue
        nb = torch.from_numpy(s["null_b"])[None, :]
        sims.append(torch.softmax(torch.cat([logits, nb], 0), 0)[:-1, :]
                    .numpy().astype(np.float32))
    return sims


def matcher_sym_probs(scores):
    """Symmetric two-direction link probability (issue: gate uses the a->b
    row-softmax only, although training supervises BOTH directions).  The
    pair scalar combined with both null heads gives the direction-symmetric
    log-odds `s = 2*pair_logit - null_a - null_b` (same quantity
    partial_pairs assigns on); sigmoid(s/2) maps it to a [0,1] score whose
    0.5 point is "pair logit beats the average null" — row-softmax
    normalizers (candidate-count dependence) drop out entirely."""
    sims = []
    for s in scores:
        logits = s["logits"].astype(np.float64)
        if not logits.size:
            sims.append(np.zeros_like(s["logits"], dtype=np.float32))
            continue
        sym = 2.0 * logits - s["null_a"][:, None] - s["null_b"][None, :]
        sym = np.clip(sym / 2.0, -60.0, 60.0)
        sims.append((1.0 / (1.0 + np.exp(-sym))).astype(np.float32))
    return sims


def partial_pairs(ap, bp, score, max_link_um, motion_alpha, link_bias, drift,
                  sim=None, ext_um=0.0, p_th=1.0):
    """One-to-one partial assignment with individual birth/death dummies.

    The pair scalar is supervised once in each direction.  Combining those
    two directional log-odds gives
    `2*pair_logit - source_null - target_null`; hence the -2 multiplier below.
    This is invariant to the row-softmax normalizers and uses both null heads.
    """
    na, nb = len(ap), len(bp)
    if not na or not nb:
        return []
    d = np.linalg.norm((ap + drift)[:, None, :] - bp[None, :, :], axis=2)
    logits = score["logits"].astype(np.float64)
    # NEG_INF/2 recognizes the matcher's out-of-radius sentinel (real logits
    # are tanh-bounded to |30|). NOTE: with raw-frame scores the sentinel was
    # set on RAW distances, so pairs with drift-residual <= max_link_um but
    # raw distance > embed_radius_um stay unreachable (see main()'s warning).
    valid = (d <= max_link_um) & (logits > NEG_INF / 2)
    if ext_um > max_link_um and sim is not None:
        # confidence-gated long links: without this, partial is capped at the 7 um
        # gate and cannot reach the band where the matcher's exclusive wins are
        valid |= (d <= ext_um) & (sim >= p_th) & (logits > NEG_INF / 2)
    huge = 1e6
    cost = np.full((na + nb, nb + na), huge, np.float64)
    cost[:na, :nb] = np.where(
        valid, -2.0 * logits - link_bias + motion_alpha * d, huge)
    # One private death slot per source and one private birth slot per target.
    cost[np.arange(na), nb + np.arange(na)] = -score["null_a"]
    cost[na + np.arange(nb), np.arange(nb)] = -score["null_b"]
    # Dummy birth rows may absorb unused death columns at zero cost.
    cost[na:, nb:] = 0.0
    ri, ci = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(ri, ci)
            if i < na and j < nb and valid[i, j]]


def build_nodes(pos):
    """(t, z, y, x) int32 node table + per-frame offsets from float zyx
    frames (submission coords are rounded; distances elsewhere stay float)."""
    nodes, offsets = [], [0]
    for t, coords in enumerate(pos):
        for zyx in np.rint(coords).astype(np.int32):
            nodes.append((t, int(zyx[0]), int(zyx[1]), int(zyx[2])))
        offsets.append(len(nodes))
    nodes = np.array(nodes, dtype=np.int32) if nodes else np.zeros((0, 4), np.int32)
    return nodes, offsets


def link_embed(per_t, max_link_um, lam, drift_iters, sims=None,
               ext_um=0.0, p_th=1.0, flow=None, flow_clip_um=None):
    """Production linking with cost = drift-compensated distance +
    lam * (1 - sim); the GATE stays on the distance term (floor-preserving:
    lam = 0 is exactly build_graph). sim = normalized-embedding cosine, or
    matcher link probability when `sims` (from build_sims) is given.
    ext_um/p_th (confidence-gated long links): pairs with
    max_link_um < d <= ext_um are additionally admitted when sim >= p_th —
    the matcher is trained/scored out to 16 um but the 7 um acceptance gate
    would discard that band.
    ext_um <= max_link_um (or a scalar sim) keeps the original gate exactly.

    flow: optional list over boundaries of (M,3) per-node displacements in MICRONS
    for the frame-t detections.  Applied to the
    frame-a positions BEFORE the drift loop, so the existing collective
    estimator keeps running on top and absorbs whatever global bias the field
    has left; flow=None reproduces the previous graph exactly.  Unlike
    build_graph's flow argument this is NOT deviation-only: the raw field is used,
    because the weak term is the collective estimate, not the local one.
    flow_clip_um zeroes vectors longer than the clip — a guard for the field's far tail.
    """
    pos = [np.asarray(p, np.float64).reshape(-1, 3) for p, _ in per_t]
    embs = [e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
            for _, e in per_t]
    nodes, offsets = build_nodes(pos)

    def assign(ap, bp, sim):
        d = np.linalg.norm(ap[:, None, :] - bp[None, :, :], axis=2)
        ok = d <= max_link_um
        if ext_um > max_link_um and isinstance(sim, np.ndarray):
            ok |= (d <= ext_um) & (sim >= p_th)
        cost = np.where(ok, d + lam * (1.0 - sim), 1e6)
        ri, ci = linear_sum_assignment(cost)
        return [(i, j) for i, j in zip(ri, ci) if ok[i, j]]

    edges = []
    for t in range(len(pos) - 1):
        a0, b0 = offsets[t], offsets[t + 1]
        if not len(pos[t]) or not len(pos[t + 1]):
            continue
        ap = pos[t] * SCALE
        bp = pos[t + 1] * SCALE
        if flow is not None:
            fa = np.asarray(flow[t], np.float64)
            # a silent length mismatch would misalign every offset and still
            # produce a plausible score — fail loudly instead
            assert fa.shape == ap.shape, \
                f"flow[{t}] {fa.shape} does not match detections {ap.shape}"
            if flow_clip_um:
                fa = np.where(
                    (np.linalg.norm(fa, axis=1) > float(flow_clip_um))[:, None],
                    0.0, fa)
            ap = ap + fa
        if sims is not None:
            sim = sims[t]              # needed for the gate even at lam = 0
        elif lam <= 0:
            sim = 0.0
        else:
            sim = embs[t] @ embs[t + 1].T
        pairs = assign(ap, bp, sim)
        for _ in range(int(drift_iters)):
            if len(pairs) < 5:
                break
            drift = np.median(np.stack([bp[j] - ap[i] for i, j in pairs]), axis=0)
            pairs = assign(ap + drift, bp, sim)
        edges += [(a0 + i, b0 + j) for i, j in pairs]
    return nodes, edges


