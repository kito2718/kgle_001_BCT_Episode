"""Trajectory-aligned dual-scale crop rendering for V31."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from biohub.pipeline.baseline import DOWNSAMPLE


OFFSETS = tuple(range(-3, 4))
SUFFIXES = ("m3", "m2", "m1", "p0", "p1", "p2", "p3")
LOCAL_SHAPE = (9, 17, 17)
CONTEXT_SHAPE = (17, 33, 33)


def source_to_grid(source_zyx: np.ndarray) -> np.ndarray:
    source = np.asarray(source_zyx, dtype=np.float64)
    return (source - (DOWNSAMPLE - 1) / 2.0) / DOWNSAMPLE


def centered_crop(
    frame: np.ndarray,
    source_zyx: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Integer-centred, zero-padded crop from a normalized grid frame."""
    output = np.zeros(shape, dtype=np.float32)
    coordinate = np.asarray(source_zyx, dtype=np.float64)
    if not np.isfinite(coordinate).all():
        return output
    center = np.rint(source_to_grid(coordinate)).astype(np.int64)
    half = np.asarray(shape, dtype=np.int64) // 2
    start = center - half
    stop = start + np.asarray(shape, dtype=np.int64)
    spatial = np.asarray(frame.shape, dtype=np.int64)
    source_start = np.maximum(start, 0)
    source_stop = np.minimum(stop, spatial)
    if np.any(source_start >= source_stop):
        return output
    target_start = source_start - start
    target_stop = target_start + (source_stop - source_start)
    output[
        target_start[0]:target_stop[0],
        target_start[1]:target_stop[1],
        target_start[2]:target_stop[2],
    ] = np.asarray(frame[
        source_start[0]:source_stop[0],
        source_start[1]:source_stop[1],
        source_start[2]:source_stop[2],
    ], dtype=np.float32)
    return output


def coordinates_from_row(
    row: Mapping,
    prefix: str,
) -> np.ndarray:
    return np.asarray([
        [row[f"{prefix}_{suffix}_{axis}"] for axis in "zyx"]
        for suffix in SUFFIXES
    ], dtype=np.float64)


def real_coordinates_from_row(row: Mapping) -> np.ndarray:
    return np.asarray([
        [row[f"trajectory_{suffix}_{axis}"] for axis in "zyx"]
        for suffix in SUFFIXES
    ], dtype=np.float64)


def temporal_permutation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    side = np.asarray([0, 1, 2, 4, 5, 6], dtype=np.int64)
    shuffled = side.copy()
    while np.array_equal(shuffled, side):
        shuffled = rng.permutation(side)
    permutation = np.arange(7, dtype=np.int64)
    permutation[side] = shuffled
    return permutation


def extract_dual_sequence(
    frame_sources: list,
    center_t: int,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = np.zeros((7, *LOCAL_SHAPE), dtype=np.float32)
    context = np.zeros((7, *CONTEXT_SHAPE), dtype=np.float32)
    valid = np.zeros(7, dtype=bool)
    for index, offset in enumerate(OFFSETS):
        frame_t = int(center_t + offset)
        source = frame_sources[index]
        if (
            source is None
            or frame_t < 0
            or frame_t >= len(source)
            or not np.isfinite(coordinates[index]).all()
        ):
            continue
        frame = source[frame_t]
        local[index] = centered_crop(frame, coordinates[index], LOCAL_SHAPE)
        context[index] = centered_crop(frame, coordinates[index], CONTEXT_SHAPE)
        valid[index] = True
    return local, context, valid


def render_real_recipe(volume, row: Mapping):
    coordinates = real_coordinates_from_row(row)
    return extract_dual_sequence(
        [volume] * 7, int(row["t"]), coordinates
    )


def render_synthetic_recipe(base, composed, row: Mapping):
    source = str(row["recipe_source"])
    a = coordinates_from_row(row, "a")
    coordinates = a.copy()
    frame_sources = [composed] * 7
    if source == "normal_tracklet":
        active = int(row["active_frame_mask"])
        frame_sources = [
            composed if active & (1 << index) else base
            for index in range(7)
        ]
    elif source == "short_lived_artifact":
        frame_sources = [base] * 7
        frame_sources[3] = composed
    elif source == "temporal_shuffle":
        coordinates = a[temporal_permutation(int(row["corruption_seed"]))]
    elif source == "wrong_path_decoy":
        b = coordinates_from_row(row, "b")
        coordinates[4:] = b[4:]
    else:
        raise ValueError(f"unknown V31 synthetic recipe source {source!r}")
    return extract_dual_sequence(
        frame_sources, int(row["center_t"]), coordinates
    )


def render_graph_dropout_recipe(volume, row: Mapping):
    """Render a real-identity graph-dropout or its explicit corruption."""
    source = str(row["recipe_source"])
    a = coordinates_from_row(row, "a")
    coordinates = a.copy()
    if source == "normal_tracklet":
        active = int(row["active_frame_mask"])
        for index in range(7):
            if not active & (1 << index):
                coordinates[index] = np.nan
    elif source == "short_lived_artifact":
        offset = np.asarray([
            row["artifact_offset_z"],
            row["artifact_offset_y"],
            row["artifact_offset_x"],
        ], dtype=np.float64)
        coordinates = a + offset
        coordinates[3] = a[3]
    elif source == "temporal_shuffle":
        coordinates = a[temporal_permutation(int(row["corruption_seed"]))]
    elif source == "wrong_path_decoy":
        b = coordinates_from_row(row, "b")
        coordinates[4:] = b[4:]
    else:
        raise ValueError(f"unknown V31 graph-dropout source {source!r}")
    return extract_dual_sequence(
        [volume] * 7, int(row["center_t"]), coordinates
    )


def summarize_dual_sequence(
    local: np.ndarray,
    context: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Fixed image-only discriminator features for the fidelity gate."""
    features: list[float] = []
    for index in range(7):
        if not valid[index]:
            features.extend([0.0] * 14)
            continue
        small = local[index].astype(np.float32)
        large = context[index].astype(np.float32)
        core = small[3:6, 6:11, 6:11]
        shell = np.concatenate([
            large[:, :4, :].ravel(),
            large[:, -4:, :].ravel(),
            large[:, 4:-4, :4].ravel(),
            large[:, 4:-4, -4:].ravel(),
        ])
        median = float(np.median(large))
        mad = float(np.median(np.abs(large - median))) + 1e-6
        projection = small.max(axis=0)
        peak = np.unravel_index(int(np.argmax(projection)), projection.shape)
        peak_offset = np.asarray(peak, dtype=np.float32) - np.asarray(
            projection.shape, dtype=np.float32
        ) // 2
        features.extend([
            float(small.mean()),
            float(small.std()),
            float(small.max()),
            float(np.quantile(small, 0.90)),
            float(np.quantile(small, 0.99)),
            float(large.mean()),
            float(large.std()),
            float(np.quantile(large, 0.99)),
            float(core.mean()),
            float(core.max()),
            float((core.max() - median) / mad),
            float(shell.mean()),
            float(peak_offset[0]),
            float(peak_offset[1]),
        ])
    frame_features = np.asarray(features, dtype=np.float32).reshape(7, 14)
    temporal = np.concatenate([
        frame_features.mean(axis=0),
        frame_features.std(axis=0),
        np.abs(np.diff(frame_features, axis=0)).mean(axis=0),
    ])
    return np.concatenate([
        frame_features.reshape(-1), temporal, valid.astype(np.float32)
    ]).astype(np.float32)
