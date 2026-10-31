"""Runtime-state writer for the deployable 0.940341 post-processing chain.

The v8 tracker already computes every expensive detector, edge, image and
pair signal used by the incremental chain.  This module persists that state in
the same compact directory contracts used by the local feature builders.  It
never accepts or writes annotations, fold assignments, or OOF predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


STATE_DIRS = (
    "base_graph", "candidates", "edges", "raw_scores", "pairs",
    "pair_tracklet", "post_scores", "post_features", "volumes",
)


def write_inference_state(result, root: Path) -> None:
    """Persist one ``track_dataset_learned_v8(return_state=True)`` result."""
    root = Path(root)
    state = result.inference_state
    if state is None:
        raise ValueError("TrackResult has no inference_state; pass return_state=True")
    name = result.graph.dataset
    for directory in STATE_DIRS:
        (root / directory).mkdir(parents=True, exist_ok=True)

    cache = state["cache"]
    forbidden = {"pred_to_gt", "edge_labels", "fold", "label"}
    overlap = forbidden.intersection(cache)
    if overlap:
        raise ValueError(f"annotation-derived cache keys are forbidden: {sorted(overlap)}")
    np.savez_compressed(root / "base_graph" / f"{name}.npz", **cache)
    state["candidates"].to_parquet(
        root / "candidates" / f"{name}.parquet", index=False)
    state["edges"].to_parquet(root / "edges" / f"{name}.parquet", index=False)
    pd.DataFrame({"p": state["raw_p"]}).to_parquet(
        root / "raw_scores" / f"{name}.parquet", index=False)
    state["pairs"].to_parquet(root / "pairs" / f"{name}.parquet", index=False)
    state["pair_tracklet"].to_parquet(
        root / "pair_tracklet" / f"{name}.parquet", index=False)
    state["post_scores"].to_parquet(
        root / "post_scores" / f"{name}.parquet", index=False)
    state["post_features"].to_parquet(
        root / "post_features" / f"{name}.parquet", index=False)
    if "volumes" in state:
        np.save(root / "volumes" / f"{name}.npy", state["volumes"])
    for tag, candidates in state.get("checkpoint_candidates", {}).items():
        directory = root / f"checkpoint_{tag}"
        directory.mkdir(parents=True, exist_ok=True)
        candidates.to_parquet(directory / f"{name}.parquet", index=False)


def finalize_inference_state(root: Path) -> dict[str, Path]:
    """Stream per-sample pair/post shards and write a leakage audit.

    A hidden test can contain substantially more samples than the public test.
    Concatenating every shard through pandas held both all inputs and the final
    table in RAM.  The Parquet writer below keeps only one bounded record batch
    resident and produces the same row order and column types.
    """
    root = Path(root)
    outputs = {}
    for source, filename in (
        ("pairs", "pairs.parquet"),
        ("pair_tracklet", "pair_tracklet.parquet"),
        ("post_scores", "post_scores.parquet"),
        ("post_features", "post_features.parquet"),
    ):
        paths = sorted((root / source).glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no {source} shards under {root}")
        schema = pq.ParquetFile(paths[0]).schema_arrow
        forbidden = {
            column for column in schema.names
            if column in {"label", "exact_label", "pred_to_gt", "edge_labels"}
            or column.endswith("_label")
        }
        if forbidden:
            raise ValueError(f"forbidden columns in {source}: {sorted(forbidden)}")
        target = root / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        writer = pq.ParquetWriter(temporary, schema)
        try:
            for path in paths:
                parquet = pq.ParquetFile(path)
                shard_schema = parquet.schema_arrow
                shard_forbidden = {
                    column for column in shard_schema.names
                    if column in {"label", "exact_label", "pred_to_gt", "edge_labels"}
                    or column.endswith("_label")
                }
                if shard_forbidden:
                    raise ValueError(
                        f"forbidden columns in {source}: {sorted(shard_forbidden)}")
                for batch in parquet.iter_batches(batch_size=65_536):
                    table = pa.Table.from_batches([batch])
                    if not table.schema.equals(schema, check_metadata=False):
                        table = table.cast(schema, safe=False)
                    writer.write_table(table)
        except Exception:
            writer.close()
            temporary.unlink(missing_ok=True)
            raise
        else:
            writer.close()
            temporary.replace(target)
        outputs[source] = target

    samples = sorted(path.stem for path in (root / "base_graph").glob("*.npz"))
    audit = {
        "format": "biohub-current940-inference-state-v1",
        "samples": samples,
        "sample_count": len(samples),
        "annotation_inputs": [],
        "fold_inputs": [],
        "tables": {key: str(value) for key, value in outputs.items()},
    }
    audit_path = root / "inference_state_manifest.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    outputs["manifest"] = audit_path
    return outputs
