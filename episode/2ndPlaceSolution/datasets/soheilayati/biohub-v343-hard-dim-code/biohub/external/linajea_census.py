"""Align dense Linajea predicted forks to the manual division census.

The manual census supplies event locations. Predicted tracks supply candidate
daughters and trajectory context, so these rows remain pseudo-supervision and
are explicitly marked as such.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .divisions import FEATURE_COLUMNS, features_for_hypotheses


PREDICTED_FILES = {
    "predicted_side_1": "zebrafish_side_1_tracks_071621.txt",
    "predicted_side_2": "zebrafish_side_2_tracks_071621.txt",
}


def _scan_prediction(path: Path) -> pl.LazyFrame:
    raw = pl.scan_csv(path)
    return raw.select([
        pl.col("t").cast(pl.Int32),
        pl.col(" z").str.strip_chars().cast(pl.Float64).alias("z"),
        pl.col(" y").str.strip_chars().cast(pl.Float64).alias("y"),
        pl.col(" x").str.strip_chars().cast(pl.Float64).alias("x"),
        pl.col(" cell_id").str.strip_chars().cast(pl.Int64).alias("cell_id"),
        pl.col(" parent_id").str.strip_chars().cast(pl.Int64).alias("parent_id"),
    ])


def _fork_candidates(rows: pl.DataFrame, child_frames: list[int]) -> pd.DataFrame:
    forks = (rows.lazy()
             .filter((pl.col("t").is_in(child_frames)) & (pl.col("parent_id") != -1))
             .group_by(["t", "parent_id"])
             .agg([
                 pl.len().alias("out_degree"),
                 pl.col("cell_id").sort().alias("children"),
             ])
             .filter(pl.col("out_degree") == 2)
             .collect())
    if not len(forks):
        return pd.DataFrame(columns=[
            "parent_id", "child1_id", "child2_id", "t"])
    candidate = forks.to_pandas()
    candidate["child1_id"] = candidate.children.map(lambda values: int(values[0]))
    candidate["child2_id"] = candidate.children.map(lambda values: int(values[1]))
    candidate["t"] = candidate.t.astype(np.int32) - 1
    return candidate[["parent_id", "child1_id", "child2_id", "t"]]


def _selected_lineage(
    rows: pl.DataFrame,
    hypotheses: pd.DataFrame,
    depth: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected: list[pl.DataFrame] = []
    parent_ids = hypotheses.parent_id.astype(np.int64).unique().tolist()
    child_ids = np.unique(hypotheses[["child1_id", "child2_id"]].to_numpy(np.int64)).tolist()

    current = parent_ids
    for _ in range(depth + 1):
        layer = rows.filter(pl.col("cell_id").is_in(current))
        if not len(layer):
            break
        selected.append(layer)
        current = layer.filter(pl.col("parent_id") != -1)["parent_id"].unique().to_list()
        if not current:
            break

    current = child_ids
    first = rows.filter(pl.col("cell_id").is_in(current))
    if len(first):
        selected.append(first)
    for _ in range(depth):
        layer = rows.filter(pl.col("parent_id").is_in(current))
        if not len(layer):
            break
        single_parent = (layer.group_by("parent_id").len()
                         .filter(pl.col("len") == 1)["parent_id"])
        layer = layer.filter(pl.col("parent_id").is_in(single_parent))
        if not len(layer):
            break
        selected.append(layer)
        current = layer["cell_id"].unique().to_list()

    compact = pl.concat(selected).unique(subset="cell_id", keep="first").to_pandas()
    compact["node_id"] = compact.cell_id.astype(np.int64)
    compact["source"] = "linajea"
    node_ids = set(compact.node_id.tolist())
    linked = compact.parent_id.isin(node_ids)
    edges = compact.loc[linked, ["parent_id", "node_id"]].rename(
        columns={"parent_id": "source_id", "node_id": "target_id"})
    return compact, edges


def _assign_manual_census(
    candidates: pd.DataFrame,
    census: pd.DataFrame,
    positive_radius: float,
    ambiguous_radius: float,
) -> pd.DataFrame:
    labeled = candidates.copy()
    labeled["label"] = np.int8(0)
    labeled["nearest_manual_distance"] = np.float32(np.inf)
    labeled["matched_manual_id"] = np.int64(-1)
    for time, index in labeled.groupby("t", sort=True).groups.items():
        candidate_index = np.asarray(list(index), dtype=np.int64)
        manual = census[census.t == time]
        if manual.empty:
            continue
        left = labeled.loc[candidate_index, ["z", "y", "x"]].to_numpy(np.float64)
        right = manual[["z", "y", "x"]].to_numpy(np.float64)
        distance = cdist(left, right)
        nearest = distance.min(axis=1)
        labeled.loc[candidate_index, "nearest_manual_distance"] = nearest.astype(np.float32)
        ambiguous = candidate_index[nearest <= ambiguous_radius]
        labeled.loc[ambiguous, "label"] = np.int8(-1)
        row, col = linear_sum_assignment(distance)
        accepted = distance[row, col] <= positive_radius
        positive_index = candidate_index[row[accepted]]
        labeled.loc[positive_index, "label"] = np.int8(1)
        manual_ids = manual.node_id.to_numpy(np.int64)[col[accepted]]
        labeled.loc[positive_index, "matched_manual_id"] = manual_ids
    return labeled


def build_linajea_census_hypotheses(
    root: str | Path,
    census: pd.DataFrame,
    *,
    positive_radius: float = 20.0,
    ambiguous_radius: float = 40.0,
) -> pd.DataFrame:
    """Build dense fork hypotheses with manual-event/pseudo-daughter labels."""
    root = Path(root)
    prediction_root = root / "predicted_tracks"
    census_times = sorted(int(value) for value in census.t.unique())
    relevant_frames = sorted({
        frame
        for time in census_times
        for frame in range(max(0, time - 8), time + 10)
    })
    child_frames = [time + 1 for time in census_times]
    sequence_parts: list[pd.DataFrame] = []
    for sequence, filename in PREDICTED_FILES.items():
        rows = (_scan_prediction(prediction_root / filename)
                .filter(pl.col("t").is_in(relevant_frames)).collect())
        hypotheses = _fork_candidates(rows, child_frames)
        if hypotheses.empty:
            continue
        parent_positions = (rows.filter(pl.col("cell_id").is_in(
            hypotheses.parent_id.astype(np.int64).tolist()))
            .select(["cell_id", "z", "y", "x"]).unique(subset="cell_id")
            .rename({"cell_id": "parent_id"}).to_pandas())
        hypotheses = hypotheses.merge(
            parent_positions, on="parent_id", validate="one_to_one")
        compact, compact_edges = _selected_lineage(rows, hypotheses)
        compact["sequence"] = sequence
        compact_edges["source"] = "linajea"
        compact_edges["sequence"] = sequence
        node_columns = ["source", "sequence", "node_id", "t", "z", "y", "x"]
        features = features_for_hypotheses(
            compact[node_columns],
            compact_edges[["source", "sequence", "source_id", "target_id"]],
            hypotheses)
        hypotheses["source"] = "linajea"
        hypotheses["sequence"] = sequence
        sequence_parts.append(pd.concat([
            hypotheses.reset_index(drop=True), features.reset_index(drop=True)], axis=1))

    candidates = pd.concat(sequence_parts, ignore_index=True)
    candidates = _assign_manual_census(
        candidates, census, positive_radius, ambiguous_radius)
    candidates["parent_out_degree"] = np.int8(2)
    candidates["label_provenance"] = "manual_event_pseudo_daughters"
    return candidates[[
        "source", "sequence", "parent_id", "child1_id", "child2_id", "t",
        "label", "parent_out_degree", "nearest_manual_distance",
        "matched_manual_id", "label_provenance", *FEATURE_COLUMNS,
    ]]
