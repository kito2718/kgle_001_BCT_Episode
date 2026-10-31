"""Deterministic acquisition clusters from OME-Zarr metadata/image statistics."""

from __future__ import annotations

import warnings
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


CLUSTER_SEED = 20260815
QUANTILES = ("0.0", "0.001", "0.01", "0.1", "0.9", "0.99", "0.999", "1.0")


def _one_image(path: Path) -> dict[str, Any]:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    multiscales = root.attrs.get("multiscales")
    if not multiscales or not multiscales[0].get("datasets"):
        raise ValueError(f"{path.name}: missing OME-Zarr multiscales metadata")
    dataset_meta = multiscales[0]["datasets"][0]
    array = root[dataset_meta["path"]]
    if len(array.shape) != 4:
        raise ValueError(f"{path.name}: expected TZYX, got {array.shape}")
    transformations = dataset_meta.get("coordinateTransformations", [])
    scale = next((item.get("scale") for item in transformations if item.get("type") == "scale"), None)
    if scale is None or len(scale) != 4:
        raise ValueError(f"{path.name}: missing TZYX scale metadata")
    quantiles = root.attrs.get("image_statistics", {}).get("quantiles", {})
    missing = [key for key in QUANTILES if key not in quantiles]
    if missing:
        raise ValueError(f"{path.name}: missing image quantiles {missing}")

    q = {key: float(quantiles[key]) for key in QUANTILES}
    row: dict[str, Any] = {
        "dataset": path.stem,
        "embryo": path.stem.split("_")[0],
        "frames": int(array.shape[0]),
        "depth": int(array.shape[1]),
        "height": int(array.shape[2]),
        "width": int(array.shape[3]),
        "scale_z": float(scale[1]),
        "scale_y": float(scale[2]),
        "scale_x": float(scale[3]),
        "dtype_bits": int(np.dtype(array.dtype).itemsize * 8),
    }
    for key, value in q.items():
        row[f"log1p_q{key.replace('.', 'p')}"] = float(np.log1p(max(value, 0.0)))
    row.update(
        log1p_range_q99_q01=float(np.log1p(max(q["0.99"] - q["0.01"], 0.0))),
        log1p_range_q999_q001=float(
            np.log1p(max(q["0.999"] - q["0.001"], 0.0))
        ),
        contrast_q90_q10=float((q["0.9"] + 1.0) / (q["0.1"] + 1.0)),
        tail_q999_q99=float((q["0.999"] + 1.0) / (q["0.99"] + 1.0)),
    )
    return row


def build_acquisition_clusters(
    image_dir: str | Path,
    n_clusters: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return per-image feature/cluster rows plus deterministic provenance."""
    paths = sorted(Path(image_dir).glob("*.zarr"))
    if len(paths) < 2:
        raise ValueError(f"need at least two images under {image_dir}")
    frame = pd.DataFrame([_one_image(path) for path in paths]).sort_values("dataset")
    frame = frame.reset_index(drop=True)
    feature_columns = [c for c in frame.columns if c not in {"dataset", "embryo"}]
    values = frame[feature_columns].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("acquisition features contain NaN or infinity")
    scaled = StandardScaler().fit_transform(values)

    if n_clusters < 2 or n_clusters >= len(frame):
        raise ValueError(f"n_clusters must be in [2, {len(frame) - 1}]")
    model = KMeans(n_clusters=n_clusters, random_state=CLUSTER_SEED, n_init=50)
    previous_cpu_limit = os.environ.get("LOKY_MAX_CPU_COUNT")
    # A single deterministic KMeans fit does not need process-level parallelism.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    try:
        with warnings.catch_warnings():
            # Some stripped-down Windows environments do not ship ``wmic``.
            warnings.filterwarnings(
                "ignore", message="Could not find the number of physical cores.*"
            )
            labels = model.fit_predict(scaled)
    finally:
        if previous_cpu_limit is None:
            os.environ.pop("LOKY_MAX_CPU_COUNT", None)
        else:
            os.environ["LOKY_MAX_CPU_COUNT"] = previous_cpu_limit

    # KMeans label numbers are nominal. Canonicalize them by sorted centroid so
    # cluster IDs remain stable even if an implementation permutes labels.
    order = sorted(
        range(n_clusters),
        key=lambda label: tuple(np.round(model.cluster_centers_[label], 12)),
    )
    remap = {old: new for new, old in enumerate(order)}
    canonical = np.array([remap[int(label)] for label in labels], dtype=np.int64)
    frame.insert(2, "acquisition_cluster", [f"acq_{value:02d}" for value in canonical])

    cluster_counts = frame["acquisition_cluster"].value_counts().sort_index()
    manifest = {
        "version": 1,
        "method": "StandardScaler + KMeans; labels canonicalized by centroid",
        "statistics_source": "OME-Zarr image_statistics.quantiles and multiscales metadata",
        "seed": CLUSTER_SEED,
        "n_samples": len(frame),
        "n_clusters": n_clusters,
        "feature_columns": feature_columns,
        "inertia": float(model.inertia_),
        "silhouette": float(silhouette_score(scaled, labels)),
        "cluster_counts": {key: int(value) for key, value in cluster_counts.items()},
    }
    return frame, manifest
