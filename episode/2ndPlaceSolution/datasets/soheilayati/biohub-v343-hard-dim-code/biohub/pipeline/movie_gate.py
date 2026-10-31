"""Name-free movie-level features and version-proof V26 ridge inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biohub.io.submission import DatasetGraph

SCALE_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)


def _quantiles(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        values = np.zeros(1, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_q05": float(np.quantile(values, 0.05)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q50": float(np.quantile(values, 0.50)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
    }


def _component_features(times: np.ndarray, source_row: np.ndarray,
                        target_row: np.ndarray) -> dict[str, float]:
    n = len(times)
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for source, target in zip(source_row, target_row):
        union(int(source), int(target))
    roots = np.array([find(index) for index in range(n)], dtype=np.int64)
    _, inverse = np.unique(roots, return_inverse=True)
    sizes = np.bincount(inverse)
    min_t = np.full(len(sizes), np.iinfo(np.int64).max, dtype=np.int64)
    max_t = np.full(len(sizes), np.iinfo(np.int64).min, dtype=np.int64)
    np.minimum.at(min_t, inverse, times)
    np.maximum.at(max_t, inverse, times)
    durations = max_t - min_t + 1
    result = {"components": float(len(sizes))}
    result.update(_quantiles("component_nodes", sizes))
    result.update(_quantiles("component_frames", durations))
    return result


def graph_features(graph: DatasetGraph, n_candidates: int,
                   budget_per_frame: int) -> dict[str, float]:
    """Return the exact ordered 52-feature V26 Screen-A vector."""
    ids = np.array(list(graph.nodes), dtype=np.int64)
    coordinates = np.array(list(graph.nodes.values()), dtype=np.float64)
    n_nodes, n_edges = len(ids), len(graph.edges)
    if n_nodes == 0:
        raise ValueError("V26 movie gate requires at least one V21 node")
    times = coordinates[:, 0].astype(np.int64)
    frames = int(times.max()) + 1
    per_frame = np.bincount(times, minlength=frames)
    row_of = {int(node): index for index, node in enumerate(ids)}
    source_row = np.array(
        [row_of[int(source)] for source, _, _ in graph.edges], dtype=np.int64
    )
    target_row = np.array(
        [row_of[int(target)] for _, target, _ in graph.edges], dtype=np.int64
    )
    indegree = np.bincount(target_row, minlength=n_nodes)
    outdegree = np.bincount(source_row, minlength=n_nodes)
    positions = coordinates[:, 1:]
    displacement = (
        np.linalg.norm(
            (positions[target_row] - positions[source_row]) * SCALE_UM, axis=1
        )
        if n_edges else np.zeros(1, dtype=np.float64)
    )
    confidence = np.array(
        [float(conf) for _, _, conf in graph.edges], dtype=np.float64
    )
    roughness = np.abs(np.diff(per_frame)) if frames > 1 else np.zeros(1)
    features = {
        "detector_candidates": float(n_candidates),
        "selected_budget_per_frame": float(budget_per_frame),
        "nodes": float(n_nodes),
        "edges": float(n_edges),
        "frames": float(frames),
        "nodes_per_frame": float(n_nodes / frames),
        "edge_node_ratio": float(n_edges / max(n_nodes, 1)),
        "candidate_node_ratio": float(n_candidates / max(n_nodes, 1)),
        "roots": float(np.sum(indegree == 0)),
        "terminals": float(np.sum(outdegree == 0)),
        "forks": float(np.sum(outdegree == 2)),
        "root_fraction": float(np.mean(indegree == 0)),
        "terminal_fraction": float(np.mean(outdegree == 0)),
        "fork_fraction": float(np.mean(outdegree == 2)),
        "mean_abs_frame_count_change": float(np.mean(roughness)),
        "max_abs_frame_count_change": float(np.max(roughness)),
    }
    features.update(_quantiles("frame_nodes", per_frame))
    features.update(_quantiles("edge_conf", confidence))
    features.update(_quantiles("edge_displacement_um", displacement))
    features.update(_component_features(times, source_row, target_row))
    values = np.fromiter(features.values(), dtype=np.float64)
    if len(features) != 52 or not np.isfinite(values).all():
        raise ValueError("invalid V26 movie-gate feature vector")
    return features


@dataclass(frozen=True)
class MovieGate:
    feature_names: tuple[str, ...]
    imputer_median: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    ridge_coef: np.ndarray
    ridge_intercept: float
    threshold: float

    @classmethod
    def load(cls, path: str | Path) -> "MovieGate":
        with np.load(path, allow_pickle=False) as data:
            gate = cls(
                feature_names=tuple(str(value) for value in data["feature_names"]),
                imputer_median=data["imputer_median"].astype(np.float64),
                scaler_mean=data["scaler_mean"].astype(np.float64),
                scaler_scale=data["scaler_scale"].astype(np.float64),
                ridge_coef=data["ridge_coef"].astype(np.float64),
                ridge_intercept=float(data["ridge_intercept"]),
                threshold=float(data["threshold"]),
            )
        size = len(gate.feature_names)
        arrays = (
            gate.imputer_median, gate.scaler_mean, gate.scaler_scale,
            gate.ridge_coef,
        )
        if size != 52 or any(array.shape != (size,) for array in arrays):
            raise ValueError("invalid V26 movie-gate artifact shape")
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("non-finite V26 movie-gate artifact")
        if not np.isfinite(gate.ridge_intercept) or not np.isfinite(gate.threshold):
            raise ValueError("non-finite V26 movie-gate scalar")
        return gate

    def predict(self, features: dict[str, float]) -> float:
        if tuple(features) != self.feature_names:
            raise ValueError("V26 movie-gate feature order mismatch")
        values = np.array([features[name] for name in self.feature_names], dtype=np.float64)
        values = np.where(np.isfinite(values), values, self.imputer_median)
        scaled = (values - self.scaler_mean) / self.scaler_scale
        return float(scaled @ self.ridge_coef + self.ridge_intercept)

    def use_v25(self, features: dict[str, float]) -> tuple[bool, float]:
        prediction = self.predict(features)
        return prediction > self.threshold, prediction
