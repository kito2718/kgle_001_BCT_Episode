#!/usr/bin/env python3
"""Rasterize OOF detector peaks into aligned T3 heatmap crops."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as image_shift

from biohub_division_crop_classifier import SPACING_ZYX
from biohub_division_detected_geometry_classifier import load_peaks
from biohub_division_flow_aligned_diff_classifier import neighbor_case_id, read_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--flow-alignment", type=Path, required=True)
    parser.add_argument("--detector-peaks", type=Path, nargs="+", required=True)
    parser.add_argument("--crop-shape", type=int, nargs=3, default=(12, 48, 48))
    parser.add_argument("--sigma-um", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rasterize(peaks, center: np.ndarray, shape: tuple[int, int, int], sigma_um: float) -> np.ndarray:
    output = np.zeros(shape, dtype=np.float32)
    half = np.asarray(shape) // 2
    spacing = np.asarray(SPACING_ZYX)
    radius = np.ceil(3.0 * sigma_um / spacing).astype(int)
    for peak in peaks:
        relative = np.asarray(peak.coordinate, dtype=np.float64) - center + half
        lower = np.maximum(np.floor(relative).astype(int) - radius, 0)
        upper = np.minimum(np.floor(relative).astype(int) + radius + 1, shape)
        if np.any(lower >= upper):
            continue
        grids = np.meshgrid(*[np.arange(lower[a], upper[a]) for a in range(3)], indexing="ij")
        squared = sum(np.square((grids[a] - relative[a]) * spacing[a] / sigma_um) for a in range(3))
        patch = float(peak.score) * np.exp(-0.5 * squared)
        slices = tuple(slice(lower[a], upper[a]) for a in range(3))
        output[slices] = np.maximum(output[slices], patch)
    return output


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = read_samples(args.samples)
    with args.flow_alignment.open(newline="") as handle:
        flow_rows = list(csv.DictReader(handle))
    if len(flow_rows) != len(samples):
        raise ValueError("flow/sample length mismatch")
    allowed = {
        case for sample in samples
        for case in (neighbor_case_id(sample, -1), sample.case_id, neighbor_case_id(sample, 1)) if case
    }
    peaks = load_peaks(args.detector_peaks, allowed)
    shape = tuple(args.crop_shape)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    cache = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=(len(samples), 3, *shape))
    spacing = np.asarray(SPACING_ZYX)
    for index, (sample, row) in enumerate(zip(samples, flow_rows)):
        center = np.asarray((sample.z, sample.y, sample.x), dtype=np.float64)
        cases = (neighbor_case_id(sample, -1), sample.case_id, neighbor_case_id(sample, 1))
        frames = [rasterize(peaks.get(case, []), center, shape, args.sigma_um) for case in cases]
        prev = np.asarray([float(row[f"prev_local_d{axis}_um"]) for axis in ("z", "y", "x")])
        nxt = np.asarray([float(row[f"next_local_d{axis}_um"]) for axis in ("z", "y", "x")])
        cache[index, 0] = image_shift(frames[0], prev / spacing, order=1, mode="constant", cval=0.0, prefilter=False)
        cache[index, 1] = frames[1]
        cache[index, 2] = image_shift(frames[2], -nxt / spacing, order=1, mode="constant", cval=0.0, prefilter=False)
    cache.flush(); del cache
    os.replace(temporary, args.output)
    summary = {"schema_version": 1, "samples": len(samples), "shape": [len(samples), 3, *shape], "sigma_um": args.sigma_um}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
