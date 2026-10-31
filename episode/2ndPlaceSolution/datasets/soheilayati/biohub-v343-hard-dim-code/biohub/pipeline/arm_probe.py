"""V42 division-arm probe: shared mechanism for the local rig and the kernel.

Frozen in configs/experiment_v42_stage4_probe.json. Everything here is
graph- and image-derived; ground truth never enters. The same functions
build the probe locally (verification) and inside the Kaggle kernel
(deployment), so parity holds by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SCALE = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)

GATE_UM = 14.0
# Amended 2026-08-30 pre-kernel (recorded in the stage-4 registration):
# the frozen provenance is "calibrated p of local universe rank 200"; the
# original value 0.030112245021373128 was taken under single-fold OOF
# scoring, and deployment's five-model mean shrinks scores (25/42 approved
# arms landed at 0.022-0.030). Same quantity re-derived under
# deployment-mode scoring on the same frozen universe:
TEMPORAL_BAR = 0.027950235588257977
IMAGE_BAR = 0.3044258654117584
PER_MOVIE_CAP = 4
GLOBAL_RATE = 0.22
HORIZON = 4

GEOMETRY_FEATURES = [
    "first_arm_len_um", "second_arm_len_um", "arm_len_ratio",
    "daughter_sep_um", "cos_arms", "midpoint_offset_um",
    "parent_motion_um", "has_grandparent", "frame_pos",
]
TEMPORAL_FEATURES = [
    "d_depth", "c1_depth", "sep2_um", "sep3_um", "sep4_um",
    "sep_growth_um_per_step", "sep_monotone_frac", "step_cos_dc1",
    "step_mag_diff_um", "d_speed_um", "c1_speed_um", "mid_step_um",
    "mid_cos_motion", "mid_resid_um", "arm_opp_cos2",
]
FEATURES = GEOMETRY_FEATURES + TEMPORAL_FEATURES


def graph_dicts(nodes: dict, edges: list) -> dict:
    """Node/edge lookups; ``nodes`` maps id -> (t, z, y, x) voxels."""
    pos = {int(n): np.asarray(v[1:], np.float64) * SCALE
           for n, v in nodes.items()}
    node_t = {int(n): int(v[0]) for n, v in nodes.items()}
    children: dict[int, list[int]] = {}
    parent_of: dict[int, int] = {}
    for source, target, *_ in edges:
        children.setdefault(int(source), []).append(int(target))
        parent_of[int(target)] = int(source)
    return {"pos": pos, "t": node_t, "children": children,
            "parent_of": parent_of}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm > 0 else np.nan


def _forward_chain(node: int, children: dict, pos: dict) -> list[np.ndarray]:
    out = [pos[node]]
    cur = node
    for _ in range(HORIZON):
        kids = children.get(cur, [])
        if len(kids) != 1:
            break
        cur = kids[0]
        out.append(pos[cur])
    return out


def triple_features(p: int, c1: int, d: int, g: dict,
                    n_frames: float = 100.0) -> dict:
    pos, children, parent_of, node_t = (
        g["pos"], g["children"], g["parent_of"], g["t"])
    p_um, c1_um, d_um = pos[p], pos[c1], pos[d]
    v_d = d_um - p_um
    v_c1 = c1_um - p_um
    gp = parent_of.get(p)
    motion = (p_um - pos[gp]) if gp is not None else None

    chain_d = _forward_chain(d, children, pos)
    chain_c = _forward_chain(c1, children, pos)
    depth = min(len(chain_d), len(chain_c))
    seps = [float(np.linalg.norm(chain_d[k] - chain_c[k]))
            for k in range(depth)]

    def sep_at(k: int) -> float:
        return seps[k] if depth > k else np.nan

    sd = chain_d[1] - chain_d[0] if len(chain_d) > 1 else None
    sc = chain_c[1] - chain_c[0] if len(chain_c) > 1 else None
    mid_step = None
    if depth >= 2:
        mid0 = (chain_d[0] + chain_c[0]) / 2.0
        mid1 = (chain_d[1] + chain_c[1]) / 2.0
        mid_step = mid1 - mid0

    def speed(chain: list[np.ndarray]) -> float:
        if len(chain) < 2:
            return np.nan
        return float(np.mean([np.linalg.norm(chain[k + 1] - chain[k])
                              for k in range(len(chain) - 1)]))

    cos_arms = _cos(v_d, v_c1)
    return {
        "first_arm_len_um": float(np.linalg.norm(v_c1)),
        "second_arm_len_um": float(np.linalg.norm(v_d)),
        "arm_len_ratio": float(
            np.linalg.norm(v_d) / max(np.linalg.norm(v_c1), 1e-6)),
        "daughter_sep_um": float(np.linalg.norm(d_um - c1_um)),
        "cos_arms": cos_arms if np.isfinite(cos_arms) else 0.0,
        "midpoint_offset_um": float(
            np.linalg.norm((d_um + c1_um) / 2.0 - p_um)),
        "parent_motion_um": float(np.linalg.norm(motion))
        if motion is not None else 0.0,
        "has_grandparent": 1.0 if gp is not None else 0.0,
        "frame_pos": float(node_t[d]) / n_frames,
        "d_depth": float(len(chain_d) - 1),
        "c1_depth": float(len(chain_c) - 1),
        "sep2_um": sep_at(1),
        "sep3_um": sep_at(2),
        "sep4_um": sep_at(3),
        "sep_growth_um_per_step": (
            (seps[depth - 1] - seps[0]) / (depth - 1)
            if depth >= 2 else np.nan),
        "sep_monotone_frac": (
            float(np.mean([seps[k + 1] > seps[k] for k in range(depth - 1)]))
            if depth >= 2 else np.nan),
        "step_cos_dc1": _cos(sd, sc)
        if sd is not None and sc is not None else np.nan,
        "step_mag_diff_um": (
            abs(float(np.linalg.norm(sd)) - float(np.linalg.norm(sc)))
            if sd is not None and sc is not None else np.nan),
        "d_speed_um": speed(chain_d),
        "c1_speed_um": speed(chain_c),
        "mid_step_um": float(np.linalg.norm(mid_step))
        if mid_step is not None else np.nan,
        "mid_cos_motion": (
            _cos(mid_step, motion)
            if mid_step is not None and motion is not None else np.nan),
        "mid_resid_um": (
            float(np.linalg.norm(mid_step - motion))
            if mid_step is not None and motion is not None else np.nan),
        "arm_opp_cos2": (
            _cos(chain_d[1] - p_um, chain_c[1] - p_um)
            if len(chain_d) > 1 and len(chain_c) > 1 else np.nan),
    }


def enumerate_proposals(g: dict) -> list[tuple[int, int, int]]:
    """(parent, c1, daughter) triples under the frozen rule.

    Per root daughter: the 3 nearest one-child parents within 14 um
    (amended 2026-08-30 pre-kernel, recorded in the stage-4 registration:
    mirrors the measured arm1c universe's top-3-parents-by-edge-p, valid
    because the deployed edge model is a calibrated distance function).
    """
    pos, children, parent_of, node_t = (
        g["pos"], g["children"], g["parent_of"], g["t"])
    by_frame: dict[int, list[int]] = {}
    for node, t in node_t.items():
        by_frame.setdefault(t, []).append(node)
    trees = {}
    for t, members in by_frame.items():
        pts = np.asarray([pos[n] for n in members], np.float64)
        trees[t] = (members, cKDTree(pts))
    proposals = []
    for d in sorted(node_t):
        t = node_t[d]
        if t < 1 or d in parent_of:
            continue
        entry = trees.get(t - 1)
        if entry is None:
            continue
        members, tree = entry
        candidates = []
        for j in tree.query_ball_point(pos[d], GATE_UM):
            p = members[j]
            kids = children.get(p, [])
            if len(kids) == 1 and kids[0] != d:
                dist = float(np.linalg.norm(pos[p] - pos[d]))
                candidates.append((dist, p, kids[0]))
        candidates.sort()
        for _, p, c1 in candidates[:3]:
            proposals.append((p, c1, d))
    return proposals


class ProbeModels:
    """The five stage-2 LGBM boosters with their Platt sigmoids."""

    def __init__(self, directory: Path):
        import lightgbm as lgb

        directory = Path(directory)
        platt = json.loads((directory / "platt.json").read_text())
        self.folds = []
        for fold in sorted(platt, key=int):
            booster = lgb.Booster(
                model_file=str(directory / f"fold{fold}.txt"))
            self.folds.append(
                (booster, float(platt[fold]["coef"]),
                 float(platt[fold]["intercept"])))

    def fold_p(self, x: np.ndarray, fold: int) -> np.ndarray:
        booster, coef, intercept = self.folds[fold]
        raw = booster.predict(x)
        return 1.0 / (1.0 + np.exp(-(coef * raw + intercept)))

    def mean_p(self, x: np.ndarray) -> np.ndarray:
        return np.mean(
            [self.fold_p(x, fold) for fold in range(len(self.folds))],
            axis=0)


def score_movie(nodes: dict, edges: list, models: ProbeModels,
                source_array, pair_model_path: Path,
                fold: int | None = None) -> pd.DataFrame:
    """Enumerate, score, and bar-filter one movie's arm candidates.

    ``fold`` None uses the deployed five-model mean; an integer uses that
    single fold's model (the local OOF read). ``source_array`` is the raw
    zarr array for image scoring (only temporal-bar survivors are scored).
    Returns candidate rows with selection flags per the frozen rule; the
    global cap is applied by the caller across movies.
    """
    from biohub.tracker.edge_model import PairStack
    from biohub.tracker.postpass import highres_pair_features

    g = graph_dicts(nodes, edges)
    triples = enumerate_proposals(g)
    if not triples:
        return pd.DataFrame()
    feats = pd.DataFrame(
        [triple_features(p, c1, d, g) for p, c1, d in triples])
    x = feats[FEATURES].to_numpy(np.float64)
    temporal_p = (models.mean_p(x) if fold is None
                  else models.fold_p(x, fold))
    frame = pd.DataFrame(
        triples, columns=["p_node", "c1_node", "d_node"])
    frame["temporal_p"] = temporal_p
    survivors = frame[frame.temporal_p >= TEMPORAL_BAR].copy()
    if not len(survivors):
        return frame.assign(image_p=np.nan, selected=False)

    node_ids = np.asarray(sorted(nodes), np.int64)
    to_row = {int(n): i for i, n in enumerate(node_ids)}
    node_t = np.asarray([nodes[int(n)][0] for n in node_ids], np.int64)
    node_zyx = np.asarray([nodes[int(n)][1:] for n in node_ids], np.float32)
    native = pd.DataFrame({
        "par": survivors.p_node.map(to_row).to_numpy(np.int64),
        "c1": survivors.c1_node.map(to_row).to_numpy(np.int64),
        "c2": survivors.d_node.map(to_row).to_numpy(np.int64),
    })
    swapped = native.rename(columns={"c1": "c2", "c2": "c1"})[
        ["par", "c1", "c2"]]
    both = pd.concat([native, swapped], ignore_index=True)
    pair_model = PairStack.load(pair_model_path)
    hi = highres_pair_features(source_array, node_t, node_zyx, both)
    if list(hi.columns) != pair_model.features:
        raise RuntimeError("image feature schema drift")
    n = len(survivors)

    def fold_probability(fold_params, values):
        v = np.asarray(values, np.float64)
        imputed = np.where(np.isnan(v), fold_params["impute"][None, :], v)
        standardized = (imputed - fold_params["mean"]) / fold_params["scale"]
        logit = standardized @ fold_params["coef"] + float(
            fold_params["intercept"])
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))

    x_native = hi.iloc[:n].to_numpy(np.float32)
    x_swapped = hi.iloc[n:].to_numpy(np.float32)
    image_p = np.mean([
        0.5 * (fold_probability(f, x_native) + fold_probability(f, x_swapped))
        for f in pair_model.folds
    ], axis=0)
    survivors["image_p"] = image_p

    passing = survivors[survivors.image_p >= IMAGE_BAR].sort_values(
        "temporal_p", ascending=False)
    used_p: set[int] = set()
    used_d: set[int] = set()
    chosen: list[int] = []
    for idx, row in passing.iterrows():
        if len(chosen) == PER_MOVIE_CAP:
            break
        if int(row.p_node) in used_p or int(row.d_node) in used_d:
            continue
        used_p.add(int(row.p_node))
        used_d.add(int(row.d_node))
        chosen.append(idx)
    frame = frame.merge(
        survivors[["p_node", "c1_node", "d_node", "image_p"]],
        on=["p_node", "c1_node", "d_node"], how="left")
    picked = survivors.loc[chosen, ["p_node", "d_node"]]
    key = set(zip(picked.p_node.astype(int), picked.d_node.astype(int)))
    pairs = list(zip(frame.p_node.astype(int), frame.d_node.astype(int)))
    frame["selected"] = [pair in key for pair in pairs]
    return frame


def apply_global_cap(selected: pd.DataFrame, n_movies: int) -> pd.DataFrame:
    """Keep the top round(GLOBAL_RATE * n_movies) arms by temporal p."""
    cap = max(1, int(round(GLOBAL_RATE * n_movies)))
    return selected.sort_values(
        "temporal_p", ascending=False).head(cap).copy()
