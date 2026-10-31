"""Flow-gated sparse candidate edges over the deployed node set — Stage 4 §5.2.

Two jobs, deliberately split from `labels.py` so the deploy path never imports
anything that touches ground truth:

  select_nodes   the deployed operating point's node set, byte-identical to
                 `verify_deployable.py` / `pipeline.learned` (round to int
                 voxels, NMS 5 um, predicted per-frame budget). The associator
                 experiment holds this fixed; the edge universe is the variable.
  build_edges    for each child at t, the k nearest parents at t-1 measured
                 from the FLOW-CENTRED position (child + predicted backward
                 flow), gated in micrometres.

Why flow-centred: Stage 1 measured that widening a *raw* gate raises coverage
and lowers the score (fact 13). The flow head moves the search centre instead
of widening the circle: parent coverage at 3 um goes 0.6323 -> 0.9602, and
top-4 flow-centred reaches 0.9975 at a 10 um gate (fact 26, stage-02 §10.2).
The gate default is that measured 10 um, not the 7 um matching radius — GT
displacement q99.9 is 13.84 um and division edges move ~3x further than
continuations (fact 14), which is exactly what the flow centre absorbs.

Geometry runs on the *unrounded* refined positions (the same convention as
`scripts/flow_parent_coverage.py`, whose numbers this builder is validated
against); only node selection and GT matching use the rounded submission
coordinates.

Embeddings enter as pairwise similarity only — cosine and L2 between the two
endpoints. Fact 38: the emb channels are not offset-free across samples
(drift/within q50 0.75 on the budget slice), so differences, which cancel
per-sample offsets to first order, are the only safe consumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from biohub.metrics.gt import SCALE_ARR

NMS_UM = 5.0  # the deployed suppression radius (fact 15)
EMB_COLS = tuple(f"emb_{i}" for i in range(48))

# Feature order is part of the edge-model contract: training and deployment
# must slice the same columns in the same order.
FEATURE_COLUMNS = (
    # geometry, all micrometres
    "resid_um",        # | parent - (child + flow) |  — the flow-centred residual
    "resid_z_um",      # |z component| of the residual (anisotropy is real, fact 7)
    "resid_xy_um",     # planar component of the residual
    "dist_um",         # | parent - child |            — raw displacement
    "dz_um",
    "dxy_um",
    "flow_mag_um",     # |child backward flow|: continuations ~1.8, divisions ~5.8 (fact 14)
    "resid_over_sigma",  # residual / child's own motion-error estimate
    # per-endpoint scalars, parent then child
    "score_p", "score_c",
    "rank_p", "rank_c",
    "sigma_p", "sigma_c",          # flow_sigma_um
    "near_p", "near_c",            # near_um, the division-lobe flag
    "dens_p", "dens_c",            # density_10um
    "width_p", "width_c",          # peak_width_um
    "bnd_p", "bnd_c",              # boundary_um
    # pairwise embedding similarity (fact 38: pairwise only)
    "emb_cos", "emb_l2",
    # competition among the child's parent hypotheses (tree.query is sorted,
    # so rank/margin fall out of construction)
    "k_in_gate",       # how many of the k slots landed inside the gate
    "resid_rank",      # 0 = this is the child's nearest flow-centred parent
    "resid_margin",    # residual minus the child's best residual
    # competition for the parent: how many children claim it, and where this
    # edge ranks among the claims. A parent claimed once is a continuation;
    # claimed twice, a division candidate; claimed by many, a crowded region.
    "n_claims_p",
    "claim_rank",
    "claim_margin",
    # context, added after the v0 tie (stage-04 session 1): the missed
    # positives are far-parent mispicks (resid q50 8.3 um) where geometry
    # carries no signal, so these channels carry what geometry cannot:
    "grandparent_um",     # |(parent + parent's flow) - nearest node at t-2|:
                          # a real cell has a history, a junk peak does not.
                          # NaN in the first linkable frame pair (no t-2).
    "parent_flow_mag",    # |parent's own backward flow|
    "flow_coherence_um",  # |child flow - median flow of 5 same-frame
                          # neighbours|: flags exactly the anomalous-motion
                          # children whose resid_um is misleading
    # full signed embedding difference, the schema's prescribed pairwise use
    # (fact 38). cos/L2 summaries measured ~zero importance in v0; the per-
    # channel differences are the stronger test of appearance signal.
    *(f"embd_{i}" for i in range(48)),
)


def select_nodes(cand: pd.DataFrame, budget: int, nms_um: float = NMS_UM,
                 score_values: np.ndarray | None = None) -> np.ndarray:
    """Pool row indices of the deployed node set, in the deployed order.

    Replicates `scripts/verify_deployable.py` exactly — including rounding to
    integer voxels *before* suppression — so the selected set is the one the
    0.8170 operating point scored, and a linker A/B on top of it is a
    single-variable comparison.
    """
    from biohub.candidates.nms import select_frame

    c_t = cand["t"].to_numpy()
    c_zyx = np.round(cand[["z", "y", "x"]].to_numpy(dtype=np.float64)).astype(np.int64)
    c_s = (cand["score"].to_numpy(dtype=np.float64) if score_values is None
           else np.asarray(score_values, dtype=np.float64))
    if c_s.shape != (len(cand),) or not np.isfinite(c_s).all():
        raise ValueError("invalid selection score values")

    keep = []
    for t in np.unique(c_t):
        rows = np.flatnonzero(c_t == t)
        keep.append(rows[select_frame(c_zyx[rows], c_s[rows], nms_um)][:budget])
    return np.concatenate(keep) if keep else np.zeros(0, dtype=np.int64)


def build_edges(
    cand: pd.DataFrame,
    sel: np.ndarray,
    k: int = 4,
    gate_um: float = 10.0,
) -> pd.DataFrame:
    """One row per (parent, child) hypothesis with the `FEATURE_COLUMNS`.

    `pi` / `ci` index into `sel` (positions in the selected-node array, which
    is also the node id space of the submitted graph); `parent_rank` /
    `child_rank` carry the stable pool key `(t, rank_in_frame)` alongside.
    """
    from scipy.spatial import cKDTree

    sel_t = cand["t"].to_numpy()[sel]
    um = cand[["z", "y", "x"]].to_numpy(dtype=np.float64)[sel] * SCALE_ARR
    flow = cand[["flow_z", "flow_y", "flow_x"]].to_numpy(dtype=np.float64)[sel]
    emb = cand[list(EMB_COLS)].to_numpy(dtype=np.float32)[sel]
    scalars = {
        c: cand[c].to_numpy(dtype=np.float32)[sel]
        for c in ("score", "rank_in_frame", "flow_sigma_um", "near_um",
                  "density_10um", "peak_width_um", "boundary_um")
    }

    frames = np.unique(sel_t)
    at = {int(t): np.flatnonzero(sel_t == t) for t in frames}

    # Per-node context, computed once per frame:
    #   grandparent_um   distance from (node + its backward flow) to the
    #                    nearest node in the PREVIOUS frame — a track-history
    #                    check applied to the parent side of each edge
    #   coherence_um     |own flow − median flow of the 5 nearest same-frame
    #                    neighbours| — the tissue moves together, so a flow
    #                    that disagrees with its neighbourhood is suspect
    grandparent = np.full(len(sel), np.nan, dtype=np.float32)
    coherence = np.full(len(sel), np.nan, dtype=np.float32)
    for t in frames:
        t = int(t)
        rows = at[t]
        if t - 1 in at and len(at[t - 1]):
            back = cKDTree(um[at[t - 1]])
            d, _ = back.query(um[rows] + flow[rows], k=1)
            grandparent[rows] = d
        if len(rows) > 1:
            kk = min(6, len(rows))  # self + 5 neighbours
            _, nn = cKDTree(um[rows]).query(um[rows], k=kk)
            nbr = np.atleast_2d(nn)[:, 1:]
            med = np.median(flow[rows][nbr], axis=1)
            coherence[rows] = np.linalg.norm(flow[rows] - med, axis=1)

    parts: list[pd.DataFrame] = []
    for t in frames:
        t = int(t)
        if t - 1 not in at:
            continue
        p_rows, c_rows = at[t - 1], at[t]
        if not len(p_rows) or not len(c_rows):
            continue

        tree = cKDTree(um[p_rows])
        centres = um[c_rows] + flow[c_rows]
        kk = min(k, len(p_rows))
        dist, nn = tree.query(centres, k=kk, distance_upper_bound=gate_um)
        dist = np.atleast_2d(dist.reshape(len(c_rows), kk))
        nn = np.atleast_2d(nn.reshape(len(c_rows), kk))

        valid = np.isfinite(dist)
        if not valid.any():
            continue
        c_local, slot = np.nonzero(valid)
        ci = c_rows[c_local]
        pi = p_rows[nn[valid]]
        resid = dist[valid]

        rvec = um[pi] - centres[c_local]
        delta = um[pi] - um[ci]
        best = np.where(valid[:, 0], dist[:, 0], np.inf)  # column 0 is the nearest

        en_p, en_c = emb[pi], emb[ci]
        norms = np.linalg.norm(en_p, axis=1) * np.linalg.norm(en_c, axis=1)
        emb_cos = np.einsum("ij,ij->i", en_p, en_c) / np.maximum(norms, 1e-6)
        embd = (en_p - en_c).astype(np.float16)

        parts.append(pd.DataFrame({
            "t": np.full(len(ci), t, dtype=np.int32),
            "pi": pi.astype(np.int32),
            "ci": ci.astype(np.int32),
            "parent_rank": scalars["rank_in_frame"][pi].astype(np.int32),
            "child_rank": scalars["rank_in_frame"][ci].astype(np.int32),
            "resid_um": resid.astype(np.float32),
            "resid_z_um": np.abs(rvec[:, 0]).astype(np.float32),
            "resid_xy_um": np.hypot(rvec[:, 1], rvec[:, 2]).astype(np.float32),
            "dist_um": np.linalg.norm(delta, axis=1).astype(np.float32),
            "dz_um": np.abs(delta[:, 0]).astype(np.float32),
            "dxy_um": np.hypot(delta[:, 1], delta[:, 2]).astype(np.float32),
            "flow_mag_um": np.linalg.norm(flow[ci], axis=1).astype(np.float32),
            "resid_over_sigma": (
                resid / np.maximum(scalars["flow_sigma_um"][ci], 0.1)
            ).astype(np.float32),
            "score_p": scalars["score"][pi],
            "score_c": scalars["score"][ci],
            "rank_p": scalars["rank_in_frame"][pi],
            "rank_c": scalars["rank_in_frame"][ci],
            "sigma_p": scalars["flow_sigma_um"][pi],
            "sigma_c": scalars["flow_sigma_um"][ci],
            "near_p": scalars["near_um"][pi],
            "near_c": scalars["near_um"][ci],
            "dens_p": scalars["density_10um"][pi],
            "dens_c": scalars["density_10um"][ci],
            "width_p": scalars["peak_width_um"][pi],
            "width_c": scalars["peak_width_um"][ci],
            "bnd_p": scalars["boundary_um"][pi],
            "bnd_c": scalars["boundary_um"][ci],
            "emb_cos": emb_cos.astype(np.float32),
            "emb_l2": np.linalg.norm(en_p - en_c, axis=1).astype(np.float32),
            "k_in_gate": valid.sum(axis=1)[c_local].astype(np.float32),
            "resid_rank": slot.astype(np.float32),
            "resid_margin": (resid - best[c_local]).astype(np.float32),
            "grandparent_um": grandparent[pi],
            "parent_flow_mag": np.linalg.norm(flow[pi], axis=1).astype(np.float32),
            "flow_coherence_um": coherence[ci],
            **{f"embd_{i}": embd[:, i] for i in range(48)},
        }))

    if not parts:
        cols = ["t", "pi", "ci", "parent_rank", "child_rank", *FEATURE_COLUMNS]
        return pd.DataFrame({c: pd.Series(dtype=np.float32) for c in cols})

    df = pd.concat(parts, ignore_index=True)

    # Parent-side competition over the whole sample: groupby is per (frame,
    # parent) because `pi` already encodes the frame.
    g = df.groupby("pi")["resid_um"]
    df["n_claims_p"] = g.transform("size").astype(np.float32)
    df["claim_rank"] = g.rank(method="first").astype(np.float32) - 1.0
    df["claim_margin"] = (df["resid_um"] - g.transform("min")).astype(np.float32)
    return df
