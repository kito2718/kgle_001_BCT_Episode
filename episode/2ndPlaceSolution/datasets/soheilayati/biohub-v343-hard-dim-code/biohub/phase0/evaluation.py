"""Strict 199-sample scoring, OOF provenance audit, and stratified summaries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from biohub.io.submission import COLUMNS, DatasetGraph, validate_graph
from biohub.metrics import load_gt, score_graph, score_run
from biohub.metrics.gt import discover_gt
from biohub.phase0.provenance import verify_official_scorer


def _integer_values(frame: pd.DataFrame, column: str, context: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{context}: {column} contains missing/non-numeric values")
    if not (values == np.round(values)).all():
        raise ValueError(f"{context}: {column} contains non-integer values")
    return values.astype(np.int64)


def graphs_from_submission(frame: pd.DataFrame) -> dict[str, DatasetGraph]:
    """Parse and structurally validate the exact graph encoded by a CSV."""
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"submission is missing columns: {missing}")
    if frame.empty:
        raise ValueError("submission is empty")
    if frame["dataset"].isna().any():
        raise ValueError("submission contains a missing dataset")
    row_type = frame["row_type"].astype(str)
    unknown = sorted(set(row_type) - {"node", "edge"})
    if unknown:
        raise ValueError(f"unknown row_type values: {unknown}")

    graphs: dict[str, DatasetGraph] = {}
    work = frame.assign(dataset=frame["dataset"].astype(str), row_type=row_type)
    for dataset, part in work.groupby("dataset", sort=True):
        nodes = part[part.row_type == "node"]
        edges = part[part.row_type == "edge"]
        node_ids = _integer_values(nodes, "node_id", dataset)
        if len(np.unique(node_ids)) != len(node_ids):
            raise ValueError(f"{dataset}: duplicate node_id")
        times = _integer_values(nodes, "t", dataset)
        z_values = _integer_values(nodes, "z", dataset)
        y_values = _integer_values(nodes, "y", dataset)
        x_values = _integer_values(nodes, "x", dataset)
        sources = _integer_values(edges, "source_id", dataset)
        targets = _integer_values(edges, "target_id", dataset)
        graph = DatasetGraph(
            dataset=dataset,
            nodes={
                int(node_id): (int(t), int(z), int(y), int(x))
                for node_id, t, z, y, x in zip(
                    node_ids, times, z_values, y_values, x_values, strict=True
                )
            },
            # CSV order is the only available confidence/tie-break order.
            edges=[
                (int(source), int(target), float(-index))
                for index, (source, target) in enumerate(
                    zip(sources, targets, strict=True)
                )
            ],
        )
        validate_graph(graph)
        graphs[dataset] = graph
    return graphs


def load_fold_map(path: str | Path) -> dict[str, int]:
    """Load the engineering-fold JSON and require a one-to-one sample mapping."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    folds = raw.get("folds")
    if not isinstance(folds, dict):
        raise ValueError(f"{path}: expected an object at 'folds'")
    result: dict[str, int] = {}
    for fold, samples in folds.items():
        for sample in samples:
            sample = str(sample)
            if sample in result:
                raise ValueError(f"{path}: {sample} occurs in multiple folds")
            result[sample] = int(fold)
    return result


def audit_oof_manifest(
    path: str | Path | None,
    fold_map: dict[str, int],
    expected_samples: set[str],
) -> dict:
    """Validate per-sample held-out-fold claims against the frozen fold map.

    Supported records use ``prediction_fold`` (or ``fold``), and may provide
    ``training_samples`` and/or ``trained_folds`` for stronger negative checks.
    This audit checks documentary provenance; it cannot infer training leakage
    from model bytes alone.
    """
    if path is None:
        return {
            "verified": False,
            "manifest": None,
            "samples_expected": len(expected_samples),
            "samples_verified": 0,
            "errors": ["OOF manifest was not provided"],
        }
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("samples")
    if isinstance(records, list):
        records = {str(item["dataset"]): item for item in records}
    if not isinstance(records, dict):
        raise ValueError(f"{path}: expected 'samples' mapping or list")

    errors: list[str] = []
    verified = 0
    actual_samples = set(map(str, records))
    missing = sorted(expected_samples - actual_samples)
    extra = sorted(actual_samples - expected_samples)
    if missing:
        errors.append(f"missing {len(missing)} samples: {missing[:8]}")
    if extra:
        errors.append(f"unexpected {len(extra)} samples: {extra[:8]}")
    for dataset in sorted(expected_samples & actual_samples):
        record = records[dataset]
        if not isinstance(record, dict):
            errors.append(f"{dataset}: record is not an object")
            continue
        actual_fold = record.get("prediction_fold", record.get("fold"))
        expected_fold = fold_map.get(dataset)
        if expected_fold is None:
            errors.append(f"{dataset}: absent from engineering fold map")
            continue
        if actual_fold is None or int(actual_fold) != expected_fold:
            errors.append(
                f"{dataset}: prediction_fold={actual_fold}, expected {expected_fold}"
            )
            continue
        training_samples = set(map(str, record.get("training_samples", [])))
        if dataset in training_samples:
            errors.append(f"{dataset}: appears in its own training_samples")
            continue
        trained_folds = {int(value) for value in record.get("trained_folds", [])}
        if expected_fold in trained_folds:
            errors.append(f"{dataset}: held-out fold appears in trained_folds")
            continue
        verified += 1
    return {
        "verified": not errors and verified == len(expected_samples),
        "manifest": str(path.resolve()),
        "samples_expected": len(expected_samples),
        "samples_verified": verified,
        "errors": errors,
    }


def score_submission(
    submission: str | Path,
    gt_dir: str | Path,
    *,
    strict_coverage: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Score a CSV end to end with the pinned official scorer."""
    verify_official_scorer()
    submission = Path(submission)
    frame = pd.read_csv(submission)
    graphs = graphs_from_submission(frame)
    gt_paths = {path.stem: path for path in discover_gt(gt_dir)}
    predicted = set(graphs)
    truth = set(gt_paths)
    unknown = sorted(predicted - truth)
    missing = sorted(truth - predicted)
    if unknown:
        raise ValueError(f"submission has {len(unknown)} datasets without GT: {unknown[:8]}")
    if strict_coverage and missing:
        raise ValueError(f"submission misses {len(missing)} GT datasets: {missing[:8]}")
    selected = sorted(predicted & truth)
    if not selected:
        raise ValueError("submission and ground truth have no datasets in common")

    rows = []
    total = len(selected)
    for index, dataset in enumerate(selected, start=1):
        gt = load_gt(gt_paths[dataset])
        graph = graphs[dataset]
        row = score_graph(graph, gt)
        topology = validate_graph(graph)
        row.update(
            predicted_edges=topology["edges"],
            predicted_forks=topology["forks"],
            predicted_births=topology["births"],
            predicted_deaths=topology["deaths"],
            edgeless_nodes=topology["edgeless_nodes"],
        )
        rows.append(row)
        if progress is not None:
            progress(index, total, dataset)
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def enrich_summary(rows: pd.DataFrame) -> dict:
    """Official run summary plus explicit count-adjustment diagnostics."""
    summary = dict(score_run(rows))
    for column in (
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "num_pred_nodes",
        "gt_nodes",
        "gt_edges",
    ):
        if column in rows:
            summary[column] = int(rows[column].sum())
    if "n_total" in rows:
        estimated = float(rows["n_total"].sum())
        predicted = int(rows["num_pred_nodes"].sum())
        summary.update(
            estimated_true_nodes=estimated,
            predicted_nodes=predicted,
            predicted_true_node_ratio=(predicted / estimated if estimated > 0 else math.nan),
            excess_predicted_nodes=predicted - estimated,
        )
    raw = float(summary["edge_jaccard"])
    adjusted = float(summary["adj_edge_jaccard"])
    summary["count_adjustment_delta"] = adjusted - raw
    summary["count_adjustment_factor"] = adjusted / raw if raw > 0 else math.nan
    summary["edge_weight"] = int(
        (rows["edge_tp"] + rows["edge_fp"] + rows["edge_fn"]).sum()
    )
    return summary


def summarise_groups(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    """Compute the exact official aggregate independently in each stratum."""
    if column not in rows:
        raise ValueError(f"missing stratum column {column!r}")
    output = []
    total_weight = float(
        (rows["edge_tp"] + rows["edge_fp"] + rows["edge_fn"]).sum()
    )
    for value, subset in rows.groupby(column, sort=True, dropna=False):
        summary = enrich_summary(subset)
        summary[column] = value
        summary["weight_share"] = (
            summary["edge_weight"] / total_weight if total_weight else math.nan
        )
        output.append(summary)
    result = pd.DataFrame(output)
    first = [column, "n", "score", "edge_jaccard", "adj_edge_jaccard"]
    columns = first + [name for name in result.columns if name not in first]
    return result[columns]


def json_ready(value):
    """Recursively convert numpy/pandas scalars and non-finite floats for JSON."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
