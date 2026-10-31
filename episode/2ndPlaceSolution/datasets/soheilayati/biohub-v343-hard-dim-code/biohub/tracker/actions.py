"""Post-WTA local reassignment actions (stage-04 §27.5).

The census of the 0.933 graph located the recoverable loss in two classes
with one shape: a submitted node whose WTA parent is wrong (mispicks, 95 %
junk competitors) and a division daughter whose claim was stolen (56/61
missing second arms already in the gated universe).  Both are fixed by the
same move — activate an unused universe edge (P -> C), evicting whatever
occupies C's in-slot, and either replacing P's out-edge (swap) or keeping
it (second-arm fork).

This module is the single source of truth for the action space and its
features.  The dataset builder and the apply-time verifier both call it, so
train and deploy features are byte-identical.  Features are computed for
EVERY universe edge (taken and unused); the training set covers both so the
model can also judge an existing arm when the fork/swap mode is decided.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biohub.metrics.gt import SCALE_ARR

NODE_STATS = ("score", "near_um", "density_10um", "peak_width_um", "boundary_um")
EMB_DIM = 48
CHAIN_STEPS = 3


class ActionState:
    """Arrays of one sample's final graph, ready for feature computation."""

    def __init__(self, name: str, state_dir: Path, graph_dir: Path,
                 scores_dir: Path, cand_dir: Path):
        z = np.load(Path(state_dir) / f"{name}.npz")
        g = np.load(Path(graph_dir) / f"{name}.npz")
        self.name = name
        self.node_t = z["node_t"].astype(np.int64)
        self.pi = z["pi"].astype(np.int64)
        self.ci = z["ci"].astype(np.int64)
        self.n = len(self.node_t)
        self.p_emit = z["p_emit"].astype(np.float32)
        self.take = g["take"].astype(np.int64)
        self.coords = g["final_zyx"].astype(np.int64)
        self.raw = pd.read_parquet(
            Path(scores_dir) / f"{name}.parquet").p.to_numpy(np.float32)

        cand = pd.read_parquet(
            Path(cand_dir) / f"{name}.parquet",
            columns=["z", "y", "x", *NODE_STATS,
                     *[f"emb_{i}" for i in range(EMB_DIM)]])
        sel = z["sel"].astype(np.int64)
        sub = cand.iloc[sel]
        self.um = sub[["z", "y", "x"]].to_numpy(np.float64) * SCALE_ARR
        self.stats = {c: sub[c].to_numpy(np.float32) for c in NODE_STATS}
        emb = sub[[f"emb_{i}" for i in range(EMB_DIM)]].to_numpy(np.float32)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        self.emb = emb / np.maximum(norm, 1e-6)

        self._build_maps()

    # -- graph maps over the final topology --------------------------------
    def _build_maps(self) -> None:
        n, take = self.n, self.take
        fpi, fci = self.pi[take], self.ci[take]
        self.taken = np.zeros(len(self.pi), dtype=bool)
        self.taken[take] = True
        self.out_deg = np.bincount(fpi, minlength=n)
        self.in_deg = np.bincount(fci, minlength=n)
        self.in_S = np.zeros(n, dtype=bool)
        if len(take):
            self.in_S[fpi] = True
            self.in_S[fci] = True

        # primary out-edge per node = max-raw taken arm; unique in-edge
        self.out_e = np.full(n, -1, dtype=np.int64)
        order = take[np.argsort(self.raw[take], kind="stable")]
        self.out_e[self.pi[order]] = order          # max raw wins (last write)
        self.in_e = np.full(n, -1, dtype=np.int64)
        self.in_e[fci] = take

        oe, ie = self.out_e, self.in_e
        self.out_tgt = np.where(oe >= 0, self.ci[np.maximum(oe, 0)], -1)
        self.out_p = np.where(oe >= 0, self.raw[np.maximum(oe, 0)], np.nan)
        self.in_src = np.where(ie >= 0, self.pi[np.maximum(ie, 0)], -1)
        self.in_p = np.where(ie >= 0, self.raw[np.maximum(ie, 0)], np.nan)

        # forward / backward chains (up to CHAIN_STEPS), continuation-only
        self.fwd_len, self.fwd_p = self._chain(self.out_tgt, self.out_p,
                                               stop_fork=self.out_deg)
        self.bwd_len, self.bwd_p = self._chain(self.in_src, self.in_p)

        # rank of each universe edge among its child's / parent's edges by raw
        self.child_rank = _group_rank(self.ci, self.raw)
        self.parent_rank = _group_rank(self.pi, self.raw)

    def _chain(self, nxt: np.ndarray, step_p: np.ndarray,
               stop_fork: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        length = np.zeros(n, dtype=np.float32)
        psum = np.zeros(n, dtype=np.float32)
        cur = np.arange(n)
        alive = np.ones(n, dtype=bool)
        for _ in range(CHAIN_STEPS):
            if stop_fork is not None:
                alive &= stop_fork[cur] == 1
            step = np.where(alive, nxt[cur], -1)
            alive &= step >= 0
            p = np.where(alive, step_p[cur], 0.0)
            length += alive
            psum += np.where(alive, np.nan_to_num(p), 0.0)
            cur = np.where(alive, np.maximum(step, 0), cur)
        mean_p = np.where(length > 0, psum / np.maximum(length, 1), np.nan)
        return length, mean_p.astype(np.float32)

    # -- the action space --------------------------------------------------
    def candidate_actions(self) -> np.ndarray:
        """Unused universe edges, both endpoints submitted, parent not a fork."""
        return np.flatnonzero(~self.taken & self.in_S[self.pi]
                              & self.in_S[self.ci] & (self.out_deg[self.pi] <= 1))

    # -- features for any edge set ----------------------------------------
    def edge_features(self, idx: np.ndarray) -> pd.DataFrame:
        st, P, C = self, self.pi[idx], self.ci[idx]
        um, raw = st.um, st.raw
        d = um[C] - um[P]
        dist = np.linalg.norm(d, axis=1)
        dz = np.abs(d[:, 0])
        dxy = np.linalg.norm(d[:, 1:], axis=1)

        # parent history and constant-velocity residual
        Pprev = st.in_src[P]
        has_hist = Pprev >= 0
        vel = um[P] - um[np.maximum(Pprev, 0)]
        pred = um[P] + vel
        cv = np.linalg.norm(um[C] - pred, axis=1)
        cv = np.where(has_hist, cv, np.nan)

        # child future coherence with the proposed step
        Cnext = st.out_tgt[C]
        has_fut = Cnext >= 0
        fut_step = um[np.maximum(Cnext, 0)] - um[C]
        fut_coh = np.linalg.norm(fut_step - d, axis=1)
        fut_coh = np.where(has_fut, fut_coh, np.nan)

        # the competitor J currently holding P's out-slot
        J = st.out_tgt[P]
        has_J = J >= 0
        Js = np.maximum(J, 0)
        sepJ = np.linalg.norm(um[C] - um[Js], axis=1)
        armJ = um[Js] - um[P]
        armcos = (d * armJ).sum(1) / np.maximum(dist * np.linalg.norm(armJ, axis=1),
                                                1e-6)
        # the incumbent X currently holding C's in-slot
        X = st.in_src[C]
        has_X = X >= 0
        Xs = np.maximum(X, 0)

        emb_pc = (st.emb[P] * st.emb[C]).sum(1)
        emb_pj = np.where(has_J, (st.emb[P] * st.emb[Js]).sum(1), np.nan)

        t_span = max(int(st.node_t.max()) - int(st.node_t.min()), 1)
        out = {
            "raw_p": raw[idx],
            "p_emit": st.p_emit[idx],
            "dist_um": dist, "dist_z_um": dz, "dist_xy_um": dxy,
            "child_rank": st.child_rank[idx].astype(np.float32),
            "parent_rank": st.parent_rank[idx].astype(np.float32),
            "cv_resid_um": cv,
            "fut_coherence_um": fut_coh,
            "t_frac": ((st.node_t[P] - st.node_t.min()) / t_span).astype(np.float32),
            # parent side
            "par_has_hist": has_hist.astype(np.float32),
            "par_in_p": st.in_p[P],
            "par_bwd_len": st.bwd_len[P], "par_bwd_p": st.bwd_p[P],
            "par_out_deg": st.out_deg[P].astype(np.float32),
            "par_out_p": st.out_p[P],
            # child side
            "child_has_inc": has_X.astype(np.float32),
            "inc_p": np.where(has_X, st.in_p[C], np.nan),
            "inc_dist_um": np.where(
                has_X, np.linalg.norm(um[C] - um[Xs], axis=1), np.nan),
            "inc_bwd_len": np.where(has_X, st.bwd_len[Xs], np.nan),
            "inc_bwd_p": np.where(has_X, st.bwd_p[Xs], np.nan),
            "child_fwd_len": st.fwd_len[C], "child_fwd_p": st.fwd_p[C],
            "child_out_p": st.out_p[C],
            # competitor side
            "has_J": has_J.astype(np.float32),
            "J_fwd_len": np.where(has_J, st.fwd_len[Js], np.nan),
            "J_fwd_p": np.where(has_J, st.fwd_p[Js], np.nan),
            "sep_CJ_um": np.where(has_J, sepJ, np.nan),
            "armcos_CJ": np.where(has_J, armcos, np.nan),
            "emb_cos_PC": emb_pc,
            "emb_cos_PJ": emb_pj,
            "emb_margin": emb_pc - np.where(has_J, emb_pj, 0.0),
            "p_margin_inc": raw[idx] - np.where(has_X, st.in_p[C], 0.0),
        }
        for s in NODE_STATS:
            out[f"par_{s}"] = st.stats[s][P]
            out[f"child_{s}"] = st.stats[s][C]
            out[f"J_{s}"] = np.where(has_J, st.stats[s][Js], np.nan)
        out["score_C_minus_J"] = out["child_score"] - np.where(
            has_J, out["J_score"], np.nan)
        out["fwdlen_C_minus_J"] = st.fwd_len[C] - np.where(
            has_J, st.fwd_len[Js], np.nan)
        df = pd.DataFrame({k: np.asarray(v, dtype=np.float32)
                           for k, v in out.items()})
        df.insert(0, "edge_idx", idx.astype(np.int64))
        df.insert(1, "pi", P.astype(np.int64))
        df.insert(2, "ci", C.astype(np.int64))
        df.insert(3, "is_taken", self.taken[idx].astype(np.int8))
        return df


def _group_rank(group: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Rank (0 = best) of each element among its group, by descending key."""
    order = np.lexsort((-key, group))
    r = np.empty(len(group), dtype=np.int64)
    go = group[order]
    newg = np.r_[True, go[1:] != go[:-1]]
    pos = np.arange(len(order))
    start = np.where(newg, pos, 0)
    np.maximum.accumulate(start, out=start)
    r[order] = pos - start
    return r
