"""Keep the rotated heat peaks and recover spatially unmatched original peaks."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from biohub.candidates.nms import select_frame
from biohub.detector.infer import TOP_K_PER_FRAME, SCALE_ARR, _neighbour_features

# Existing cross-bank matching tolerance, fixed before this experiment.
MATCH_UM = 3.0
CONTEXT_FIELDS = {"rank_in_frame", "near_um", "density_10um"}


def retain_original_peaks(heat, original):
    """Retain every heat row; fill unused slots with original peaks >3um away.

    Candidate positions, confidence, motion, uncertainty and appearance stay
    native to their source view. Only rank and neighbourhood fields are rebuilt
    for the combined pool. The original count view remains a separate input.
    """
    if list(heat.columns) != list(original.columns):
        raise ValueError("candidate columns differ")
    if heat.empty:
        raise ValueError("heat candidate bank must be nonempty")
    pieces, additions = [], []
    protected = [c for c in heat.columns if c not in CONTEXT_FIELDS]
    heat_times, old_times = heat.t.to_numpy(), original.t.to_numpy()
    for t in sorted(set(heat_times) | set(old_times)):
        hi, oi = np.flatnonzero(heat_times == t), np.flatnonzero(old_times == t)
        if len(hi) > TOP_K_PER_FRAME:
            raise ValueError("heat pool already exceeds the deployed cap")
        slots = TOP_K_PER_FRAME - len(hi)
        selected = np.zeros(0, dtype=int)
        if slots and len(oi):
            old_pos = original.iloc[oi][["z", "y", "x"]].to_numpy(np.float64)
            if len(hi):
                heat_pos = heat.iloc[hi][["z", "y", "x"]].to_numpy(np.float64)
                distance, _ = cKDTree(heat_pos * SCALE_ARR).query(old_pos * SCALE_ARR)
                available = np.flatnonzero(distance > MATCH_UM)
            else:
                available = np.arange(len(oi))
            if len(available):
                score = original.score.to_numpy()[oi[available]]
                keep = select_frame(old_pos[available], score, MATCH_UM)[:slots]
                selected = oi[available[keep]]
        if not len(selected):
            pieces.append(heat.iloc[hi].copy())
            continue
        frame = pd.concat([heat.iloc[hi], original.iloc[selected]], ignore_index=True)
        # Source-row markers are transient verification data, never model inputs.
        frame["_heat_row"] = np.r_[hi,np.full(len(selected), -1)]
        frame["_original_row"] = np.r_[np.full(len(hi), -1), selected]
        frame = frame.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
        near, density = _neighbour_features(
            frame[["z", "y", "x"]].to_numpy(np.float64) * SCALE_ARR,
            frame.score.to_numpy(np.float64))
        frame["near_um"] = np.minimum(near, 1e4).astype(np.float32)
        frame["density_10um"] = density
        frame["rank_in_frame"] = np.arange(len(frame), dtype=np.int32)
        h = frame.loc[frame._heat_row >= 0].sort_values("_heat_row")
        o = frame.loc[frame._original_row >= 0].sort_values("_original_row")
        if not h[protected].reset_index(drop=True).equals(heat.iloc[hi][protected].reset_index(drop=True)):
            raise RuntimeError("retained heat fields changed")
        if not o[protected].reset_index(drop=True).equals(original.iloc[np.sort(selected)][protected].reset_index(drop=True)):
            raise RuntimeError("recovered original fields changed")
        additions.extend(selected.tolist())
        pieces.append(frame.drop(columns=["_heat_row", "_original_row"]))
    result = pd.concat(pieces, ignore_index=True)
    if not additions:
        if not result.equals(heat.reset_index(drop=True)):
            raise RuntimeError("zero-recovery control changed")
    if len(result) != len(heat) + len(additions):
        raise RuntimeError("candidate retention count differs")
    return result, {"heat_rows": len(heat), "recovered_original_rows": len(additions),
        "rows": len(result), "heat_fields_preserved": True, "original_fields_preserved": True,
        "maximum_frame_rows": int(result.groupby("t").size().max())}
