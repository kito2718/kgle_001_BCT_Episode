#!/usr/bin/env python3
"""Direct full-volume CV for the Biohub nnU-Net heatmap detector.

This script deliberately does *not* call nnU-Net's sliding-window predictor.
Each already-preprocessed validation case is loaded as one tensor and passed to
the fully convolutional network exactly once::

    logits = network(full_volume[None])

The current Biohub cases have shape CxZxYxX = 1x64x256x256.  The script keeps
nnU-Net's preprocessing, checkpoint restoration, and architecture creation, but
does not create tiles, Gaussian importance maps, or overlap accumulators.

Outputs are resumable and include:

* full-volume masked MSE and voxel pseudo-Dice;
* threshold-wise 7 um one-to-one node matching metrics;
* exported peaks for one selected threshold;
* direct-forward latency and CUDA peak-memory measurements.

The node metrics evaluate the detector only.  Edge/division metrics require a
separate association stage and are intentionally not reported here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2


IGNORE_LABEL = 2
DIRECT_CV_SCHEMA_VERSION = 2
CASE_FRAME_RE = re.compile(r"^(?P<stem>.+)_t(?P<frame>\d+)$")


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not math.isfinite(v) or v < 0 or v > 1 for v in values):
        raise argparse.ArgumentTypeError("thresholds must be comma-separated values in [0, 1]")
    return tuple(sorted(set(values)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one direct CNN forward per selected full volume; no sliding-window inference."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Trainer output directory containing plans.json, dataset.json and fold_N/.",
    )
    parser.add_argument(
        "--preprocessed-dataset-dir",
        type=Path,
        required=True,
        help="DatasetXXX directory under nnUNet_preprocessed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--case-partition", choices=("train", "val", "all"), default="val",
        help="Which fold partition to predict. Default val preserves historical CV behavior.",
    )
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--amp", default="bf16", choices=("none", "fp16", "bf16"))
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile via nnUNet_compile=true.")
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--peak-thresholds", type=parse_float_list, default=parse_float_list("0.1,0.2,0.3,0.4,0.5"))
    parser.add_argument("--export-threshold", type=float, default=0.3)
    parser.add_argument("--nms-radius-um", type=float, default=2.0)
    parser.add_argument("--match-radius-um", type=float, default=7.0)
    parser.add_argument("--max-peaks-per-case", type=int, default=2048)
    parser.add_argument("--voxel-threshold", type=float, default=0.5)
    parser.add_argument("--sigma-um", type=float, default=2.0)
    parser.add_argument("--truncate-sigma", type=float, default=3.0)
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL)
    parser.add_argument("--save-probabilities", action="store_true")
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def case_stem_frame(case_id: str) -> tuple[str, int]:
    match = CASE_FRAME_RE.match(case_id)
    if match is None:
        return case_id, -1
    return match.group("stem"), int(match.group("frame"))


def direct_cv_config(args: argparse.Namespace, cases: Sequence[str], data_identifier: str) -> dict[str, Any]:
    checkpoint_path = args.model_dir / f"fold_{args.fold}" / args.checkpoint
    checkpoint_stat = checkpoint_path.stat()
    return {
        "schema_version": DIRECT_CV_SCHEMA_VERSION,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mode": "direct_full_volume_single_forward_no_sliding_window",
        "model_dir": str(args.model_dir.resolve()),
        "preprocessed_dataset_dir": str(args.preprocessed_dataset_dir.resolve()),
        "data_identifier": data_identifier,
        "fold": args.fold,
        "case_partition": args.case_partition,
        "checkpoint": args.checkpoint,
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "device": args.device,
        "amp": args.amp,
        "compile": bool(args.compile),
        "case_offset": args.case_offset,
        "limit": args.limit,
        "case_count": len(cases),
        "first_case": cases[0] if cases else None,
        "last_case": cases[-1] if cases else None,
        "peak_thresholds": list(args.peak_thresholds),
        "export_threshold": args.export_threshold,
        "nms_radius_um": args.nms_radius_um,
        "match_radius_um": args.match_radius_um,
        "max_peaks_per_case": args.max_peaks_per_case,
        "voxel_threshold": args.voxel_threshold,
        "sigma_um": args.sigma_um,
        "truncate_sigma": args.truncate_sigma,
        "ignore_label": args.ignore_label,
        "save_probabilities": bool(args.save_probabilities),
    }


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_output_dir(args: argparse.Namespace, config: dict[str, Any]) -> str:
    output = args.output_dir
    if args.overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "run_config.json"
    digest = config_hash(config)
    payload = {"config_hash": digest, **config}
    if existing.exists():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if previous.get("config_hash") != digest:
            raise RuntimeError(
                f"output directory contains a different run_config: {existing}. "
                "Use a new --output-dir or pass --overwrite explicitly."
            )
        if not args.resume:
            raise RuntimeError(f"output already exists: {output}; pass --resume or --overwrite")
    else:
        atomic_write_json(existing, payload)
    (output / "cases").mkdir(exist_ok=True)
    (output / "predicted_nodes_by_case").mkdir(exist_ok=True)
    if args.save_probabilities:
        (output / "probabilities").mkdir(exist_ok=True)
    return digest


def cumulative_stride(configuration_manager: Any) -> tuple[int, int, int]:
    strides = np.asarray(configuration_manager.pool_op_kernel_sizes, dtype=np.int64)
    if strides.ndim != 2 or strides.shape[1] != 3:
        raise RuntimeError(f"expected 3D network strides, got {strides.tolist()}")
    return tuple(int(v) for v in np.prod(strides, axis=0))


def pad_full_volume_to_stride(
    tensor: torch.Tensor, required_stride: Sequence[int]
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    spatial = tensor.shape[-3:]
    pad_after = tuple((stride - size % stride) % stride for size, stride in zip(spatial, required_stride))
    if any(pad_after):
        # torch pad order is X-left, X-right, Y-left, Y-right, Z-left, Z-right.
        tensor = F.pad(tensor, (0, pad_after[2], 0, pad_after[1], 0, pad_after[0]), value=0.0)
    return tensor, pad_after


def amp_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def direct_full_volume_forward(
    network: torch.nn.Module,
    data: np.ndarray,
    device: torch.device,
    amp: str,
    required_stride: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    if data.ndim != 4:
        raise ValueError(f"expected preprocessed CZYX data, got {data.shape}")
    original_shape = tuple(int(v) for v in data.shape[-3:])
    x = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).unsqueeze(0)
    x, pad_after = pad_full_volume_to_stride(x, required_stride)
    padded_shape = tuple(int(v) for v in x.shape[-3:])
    x = x.to(device, non_blocking=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode(), amp_context(device, amp):
        output = network(x)
        if isinstance(output, (tuple, list)):
            if len(output) != 1:
                raise RuntimeError(f"direct CV expects one full-resolution output, got {len(output)} outputs")
            output = output[0]
        if not torch.is_tensor(output) or output.ndim != 5:
            raise RuntimeError(f"unexpected network output: {type(output)} shape={getattr(output, 'shape', None)}")
        if tuple(output.shape[-3:]) != padded_shape:
            output = F.interpolate(output.float(), size=padded_shape, mode="trilinear", align_corners=False)
        output = output[..., : original_shape[0], : original_shape[1], : original_shape[2]]
        # Joint Biohub models append auxiliary regression channels after the
        # two detector logits. They must not participate in detector softmax.
        probability = torch.softmax(output[:, :2].float(), dim=1)[0, 1]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    elapsed = time.perf_counter() - started
    probability_np = probability.cpu().numpy().astype(np.float32, copy=False)
    del probability, output, x
    return probability_np, {
        "input_shape_czyx": [int(v) for v in data.shape],
        "network_input_shape_zyx": list(padded_shape),
        "pad_after_zyx": list(pad_after),
        "forward_seconds": elapsed,
        "cuda_peak_memory_bytes": peak_memory,
    }


def gaussian_target_from_sparse_seg(
    seg_zyx: np.ndarray,
    spacing_zyx_um: Sequence[float],
    sigma_um: float,
    truncate_sigma: float,
    ignore_label: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if seg_zyx.ndim != 3:
        raise ValueError(f"expected ZYX segmentation, got {seg_zyx.shape}")
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    radii = np.maximum(1, np.ceil(truncate_sigma * sigma_um / spacing).astype(int))
    kernel_axes = [np.arange(-r, r + 1, dtype=np.float32) * s for r, s in zip(radii, spacing)]
    zz, yy, xx = np.meshgrid(*kernel_axes, indexing="ij")
    kernel = np.exp(-(zz**2 + yy**2 + xx**2) / (2 * sigma_um**2)).astype(np.float32)
    kernel /= max(float(kernel.max()), 1e-12)

    target = np.zeros(seg_zyx.shape, dtype=np.float32)
    centres = np.argwhere(seg_zyx == 1).astype(np.int32)
    shape = np.asarray(seg_zyx.shape, dtype=int)
    for centre in centres:
        lo = np.maximum(centre - radii, 0)
        hi = np.minimum(centre + radii + 1, shape)
        kernel_lo = lo - (centre - radii)
        kernel_hi = kernel_lo + (hi - lo)
        target_slice = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        kernel_slice = tuple(slice(int(a), int(b)) for a, b in zip(kernel_lo, kernel_hi))
        target[target_slice] += kernel[kernel_slice]
    np.clip(target, 0.0, 1.0, out=target)
    valid = seg_zyx != ignore_label
    target *= valid
    return target, valid, centres


def voxel_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    if not np.any(valid):
        return {"valid_voxels": 0, "positive_voxels": 0, "masked_mse": None,
                "positive_region_mse": None, "tp": 0, "fp": 0, "fn": 0, "pseudo_dice": None}
    squared_error = np.square(probability - target, dtype=np.float32)
    masked_mse = float(squared_error[valid].mean())
    predicted = (probability > threshold) & valid
    expected = (target > threshold) & valid
    positive_region_mse = float(squared_error[expected].mean()) if np.any(expected) else None
    tp = int(np.count_nonzero(predicted & expected))
    fp = int(np.count_nonzero(predicted & ~expected & valid))
    fn = int(np.count_nonzero(~predicted & expected & valid))
    denominator = 2 * tp + fp + fn
    return {
        "valid_voxels": int(np.count_nonzero(valid)),
        "positive_voxels": int(np.count_nonzero(expected)),
        "masked_mse": masked_mse,
        "positive_region_mse": positive_region_mse,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pseudo_dice": (2 * tp / denominator) if denominator else 1.0,
    }


def local_maximum_candidates(probability: np.ndarray, minimum_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    neighborhood_max = maximum_filter(probability, size=3, mode="nearest")
    mask = (probability >= minimum_threshold) & (probability == neighborhood_max)
    coords = np.argwhere(mask).astype(np.int32)
    scores = probability[tuple(coords.T)] if len(coords) else np.empty((0,), dtype=np.float32)
    order = np.argsort(scores)[::-1]
    return coords[order], scores[order]


def physical_greedy_nms(
    candidate_coords: np.ndarray,
    candidate_scores: np.ndarray,
    spacing_zyx_um: Sequence[float],
    threshold: float,
    radius_um: float,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(candidate_coords) == 0:
        return candidate_coords, candidate_scores
    spacing = np.asarray(spacing_zyx_um, dtype=np.float32)
    accepted_coords: list[np.ndarray] = []
    # Exact spatial hash for the accepted points. With cell width=radius_um,
    # every point closer than radius lies in the same or one of 26 adjacent
    # cells, so only those cells need distance checks. Candidate order and the
    # strict `< radius_sq` acceptance rule remain unchanged.
    accepted_by_cell: dict[tuple[int, int, int], list[np.ndarray]] = {}
    accepted_scores: list[float] = []
    radius_sq = float(radius_um**2)
    for coord, score in zip(candidate_coords, candidate_scores):
        if float(score) < threshold:
            break
        physical = coord.astype(np.float32) * spacing
        cell_array = np.floor(physical / float(radius_um)).astype(np.int64)
        cell = (int(cell_array[0]), int(cell_array[1]), int(cell_array[2]))
        nearby: list[np.ndarray] = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nearby.extend(accepted_by_cell.get((cell[0] + dz, cell[1] + dy, cell[2] + dx), ()))
        if nearby:
            prior = np.asarray(nearby, dtype=np.float32)
            if np.any(np.sum(np.square(prior - physical), axis=1) < radius_sq):
                continue
        accepted_coords.append(coord)
        accepted_by_cell.setdefault(cell, []).append(physical)
        accepted_scores.append(float(score))
        if len(accepted_coords) >= maximum:
            break
    if not accepted_coords:
        return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.float32)
    return np.asarray(accepted_coords, dtype=np.int32), np.asarray(accepted_scores, dtype=np.float32)


def one_to_one_radius_match(
    predicted_zyx: np.ndarray,
    gt_zyx: np.ndarray,
    spacing_zyx_um: Sequence[float],
    radius_um: float,
) -> tuple[dict[str, Any], np.ndarray]:
    n_pred, n_gt = len(predicted_zyx), len(gt_zyx)
    if n_pred == 0 or n_gt == 0:
        return {
            "tp": 0,
            "fn": n_gt,
            "matched_distance_sum_um": 0.0,
            "matched_distance_max_um": None,
        }, np.empty((0,), dtype=np.int32)
    spacing = np.asarray(spacing_zyx_um, dtype=np.float32)
    pred_phys = predicted_zyx.astype(np.float32) * spacing
    gt_phys = gt_zyx.astype(np.float32) * spacing
    distances = np.sqrt(np.sum(np.square(pred_phys[:, None, :] - gt_phys[None, :, :]), axis=2))

    # A prediction farther than radius_um from every GT can never participate
    # in a valid match. Removing those rows before the augmented Hungarian
    # assignment is exact: they contribute a constant unmatched prediction and
    # cannot alter either maximum cardinality or the valid-pair distance sum.
    eligible_prediction_indices = np.flatnonzero(np.any(distances <= radius_um, axis=1))
    if len(eligible_prediction_indices) == 0:
        return {
            "tp": 0,
            "fn": n_gt,
            "matched_distance_sum_um": 0.0,
            "matched_distance_max_um": None,
        }, np.empty((0,), dtype=np.int32)
    eligible_distances = distances[eligible_prediction_indices]
    n_eligible = len(eligible_prediction_indices)

    # Augmented assignment: matching a valid pair costs <= radius, while leaving
    # both vertices unmatched costs 2 * (radius + epsilon). This prioritizes
    # maximum cardinality and then shorter distances without accepting >radius.
    size = n_eligible + n_gt
    unmatched_cost = float(radius_um + 1e-3)
    invalid_cost = float(max(radius_um * 1000.0, 1e6))
    cost = np.zeros((size, size), dtype=np.float32)
    cost[:n_eligible, :n_gt] = np.where(
        eligible_distances <= radius_um, eligible_distances, invalid_cost
    )
    cost[:n_eligible, n_gt:] = unmatched_cost
    cost[n_eligible:, :n_gt] = unmatched_cost
    rows, columns = linear_sum_assignment(cost)
    matched_pairs = [
        (int(eligible_prediction_indices[row]), float(eligible_distances[row, column]))
        for row, column in zip(rows, columns)
        if row < n_eligible and column < n_gt and eligible_distances[row, column] <= radius_um
    ]
    matched_distances = [distance for _, distance in matched_pairs]
    tp = len(matched_pairs)
    return {
        "tp": tp,
        "fn": n_gt - tp,
        "matched_distance_sum_um": float(sum(matched_distances)),
        "matched_distance_max_um": max(matched_distances) if matched_distances else None,
    }, np.asarray([index for index, _ in matched_pairs], dtype=np.int32)


def safe_threshold_key(threshold: float) -> str:
    return f"{threshold:.6f}".rstrip("0").rstrip(".")


def process_case(
    case_id: str,
    dataset: nnUNetDatasetBlosc2,
    network: torch.nn.Module,
    device: torch.device,
    required_stride: Sequence[int],
    spacing_zyx_um: Sequence[float],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    loaded_started = time.perf_counter()
    data_obj, seg_obj, _, properties = dataset.load_case(case_id)
    data = np.asarray(data_obj, dtype=np.float32)
    seg = np.asarray(seg_obj)
    load_seconds = time.perf_counter() - loaded_started
    if seg.ndim != 4 or seg.shape[0] < 1:
        raise ValueError(f"expected CZYX seg with a detector-label channel, got {seg.shape}")
    # Auxiliary temporal trainers append correspondence/motion supervision
    # channels; detector CV always evaluates channel 0 only.
    seg = seg[:1]

    probability, runtime = direct_full_volume_forward(network, data, device, args.amp, required_stride)
    if probability.shape != tuple(seg.shape[1:]):
        raise RuntimeError(f"prediction/label shape mismatch: {probability.shape} vs {seg.shape[1:]}")
    target, valid, gt_centres = gaussian_target_from_sparse_seg(
        seg[0], spacing_zyx_um, args.sigma_um, args.truncate_sigma, args.ignore_label
    )
    voxels = voxel_metrics(probability, target, valid, args.voxel_threshold)

    minimum_nms_threshold = min(min(args.peak_thresholds), args.export_threshold)
    candidate_coords, candidate_scores = local_maximum_candidates(probability, minimum_nms_threshold)
    # Greedy NMS visits candidates in descending score order. Therefore NMS at
    # the lowest requested threshold is a prefix-compatible superset of NMS at
    # every higher threshold; score-filtering this result is exactly equivalent
    # to rerunning NMS independently for each threshold.
    nms_coords, nms_scores = physical_greedy_nms(
        candidate_coords,
        candidate_scores,
        spacing_zyx_um,
        minimum_nms_threshold,
        args.nms_radius_um,
        args.max_peaks_per_case,
    )
    threshold_metrics: dict[str, Any] = {}
    export_coords = np.empty((0, 3), dtype=np.int32)
    export_scores = np.empty((0,), dtype=np.float32)
    for threshold in args.peak_thresholds:
        keep = nms_scores >= threshold
        peaks, scores = nms_coords[keep], nms_scores[keep]
        match, matched_prediction_indices = one_to_one_radius_match(
            peaks, gt_centres, spacing_zyx_um, args.match_radius_um
        )
        matched_mask = np.zeros(len(peaks), dtype=bool)
        matched_mask[matched_prediction_indices] = True
        if len(peaks):
            in_known_region = valid[tuple(peaks.T)]
        else:
            in_known_region = np.empty((0,), dtype=bool)
        # Sparse GT means predictions outside locally supervised balls are
        # unknown, not false positives. A matched prediction may lie between
        # valid_radius (6 um) and evaluation_radius (7 um), and remains a TP.
        known_fp = int(np.count_nonzero(~matched_mask & in_known_region))
        unknown_predictions = int(np.count_nonzero(~matched_mask & ~in_known_region))
        threshold_metrics[safe_threshold_key(threshold)] = {
            "predicted_nodes": len(peaks),
            "scored_predictions": int(match["tp"] + known_fp),
            "fp": known_fp,
            "unknown_predictions": unknown_predictions,
            **match,
        }
        if math.isclose(threshold, args.export_threshold, rel_tol=0.0, abs_tol=1e-9):
            export_coords, export_scores = peaks, scores

    if not any(math.isclose(v, args.export_threshold, abs_tol=1e-9) for v in args.peak_thresholds):
        keep = nms_scores >= args.export_threshold
        export_coords, export_scores = nms_coords[keep], nms_scores[keep]

    stem, frame = case_stem_frame(case_id)
    prediction_rows = []
    spacing = np.asarray(spacing_zyx_um, dtype=np.float32)
    original_spacing = np.asarray(properties.get("spacing", spacing_zyx_um), dtype=np.float32)
    crop_bbox = properties.get("bbox_used_for_cropping")
    if crop_bbox is None:
        crop_offset = np.zeros(3, dtype=np.float32)
    else:
        crop_offset = np.asarray([axis[0] for axis in crop_bbox], dtype=np.float32)
    for node_index, (coord, score) in enumerate(zip(export_coords, export_scores)):
        # Network outputs live in cropped/resampled nnU-Net coordinates. Export
        # detections in the original Zarr/GEFF voxel space for downstream tracking.
        original_coord = coord.astype(np.float32) * spacing / original_spacing + crop_offset
        physical = original_coord * original_spacing
        prediction_rows.append(
            {
                "case_id": case_id,
                "stem": stem,
                "frame": frame,
                "node_index": node_index,
                "z": float(original_coord[0]),
                "y": float(original_coord[1]),
                "x": float(original_coord[2]),
                "z_um": float(physical[0]),
                "y_um": float(physical[1]),
                "x_um": float(physical[2]),
                "score": float(score),
                "threshold": args.export_threshold,
            }
        )

    result = {
        "case_id": case_id,
        "stem": stem,
        "frame": frame,
        "load_seconds": load_seconds,
        **runtime,
        "spacing_zyx_um": [float(v) for v in spacing_zyx_um],
        "original_spacing_zyx_um": [float(v) for v in original_spacing],
        "crop_offset_zyx": [float(v) for v in crop_offset],
        "gt_nodes": len(gt_centres),
        "candidate_local_maxima": len(candidate_coords),
        "voxel": voxels,
        "threshold_metrics": threshold_metrics,
    }
    return result, prediction_rows, probability


def aggregate_results(
    output_dir: Path,
    config_digest: str,
    thresholds: Sequence[float],
    export_threshold: float,
) -> dict[str, Any]:
    case_results = []
    for path in sorted((output_dir / "cases").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("config_hash") == config_digest:
            case_results.append(payload)
    if not case_results:
        raise RuntimeError("no completed case results to aggregate")

    voxel_tp = sum(int(row["voxel"]["tp"]) for row in case_results)
    voxel_fp = sum(int(row["voxel"]["fp"]) for row in case_results)
    voxel_fn = sum(int(row["voxel"]["fn"]) for row in case_results)
    voxel_denominator = 2 * voxel_tp + voxel_fp + voxel_fn
    valid_total = sum(int(row["voxel"]["valid_voxels"]) for row in case_results)
    weighted_mse = sum(
        float(row["voxel"]["masked_mse"]) * int(row["voxel"]["valid_voxels"])
        for row in case_results
        if row["voxel"]["masked_mse"] is not None
    )
    positive_total = sum(int(row["voxel"].get("positive_voxels", 0)) for row in case_results)
    weighted_positive_mse = sum(
        float(row["voxel"]["positive_region_mse"]) * int(row["voxel"]["positive_voxels"])
        for row in case_results
        if row["voxel"].get("positive_region_mse") is not None
    )
    threshold_summary = {}
    for threshold in thresholds:
        key = safe_threshold_key(threshold)
        rows = [row["threshold_metrics"][key] for row in case_results]
        tp = sum(int(row["tp"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        distance_sum = sum(float(row["matched_distance_sum_um"]) for row in rows)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        threshold_summary[key] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "known_region_precision": precision,
            "sparse_gt_recall": recall,
            "known_region_f1": f1,
            "mean_matched_distance_um": distance_sum / tp if tp else None,
            "predicted_nodes": sum(int(row["predicted_nodes"]) for row in rows),
            "scored_predictions": sum(int(row["scored_predictions"]) for row in rows),
            "unknown_predictions": sum(int(row["unknown_predictions"]) for row in rows),
        }
    best_threshold = max(threshold_summary, key=lambda key: threshold_summary[key]["known_region_f1"])
    return {
        "mode": "direct_full_volume_single_forward_no_sliding_window",
        "completed_cases": len(case_results),
        "gt_nodes": sum(int(row["gt_nodes"]) for row in case_results),
        "mean_forward_seconds": float(np.mean([row["forward_seconds"] for row in case_results])),
        "total_forward_seconds": float(sum(row["forward_seconds"] for row in case_results)),
        "max_cuda_peak_memory_bytes": max(int(row["cuda_peak_memory_bytes"]) for row in case_results),
        "voxel": {
            "valid_voxels": valid_total,
            "positive_voxels": positive_total,
            "masked_mse": weighted_mse / valid_total if valid_total else None,
            "positive_region_mse": weighted_positive_mse / positive_total if positive_total else None,
            "tp": voxel_tp,
            "fp": voxel_fp,
            "fn": voxel_fn,
            "pseudo_dice": (2 * voxel_tp / voxel_denominator) if voxel_denominator else 1.0,
        },
        "node_by_threshold": threshold_summary,
        "best_known_region_f1_threshold": best_threshold,
        "export_threshold": export_threshold,
        "scope_note": (
            "Detector node CV only. Because GT is sparse, unmatched predictions outside locally supervised "
            "regions are unknown_predictions, not FP; global precision is not identifiable. Edge and division "
            "metrics require association."
        ),
    }


def flatten_case_result(result: dict[str, Any], thresholds: Sequence[float]) -> dict[str, Any]:
    row = {
        "case_id": result["case_id"],
        "stem": result["stem"],
        "frame": result["frame"],
        "gt_nodes": result["gt_nodes"],
        "forward_seconds": result["forward_seconds"],
        "cuda_peak_memory_bytes": result["cuda_peak_memory_bytes"],
        "voxel_masked_mse": result["voxel"]["masked_mse"],
        "voxel_positive_region_mse": result["voxel"].get("positive_region_mse"),
        "voxel_pseudo_dice": result["voxel"]["pseudo_dice"],
    }
    for threshold in thresholds:
        key = safe_threshold_key(threshold)
        metric = result["threshold_metrics"][key]
        prefix = "node_thr_" + key.replace(".", "p")
        for name in (
            "predicted_nodes", "scored_predictions", "unknown_predictions", "tp", "fp", "fn",
            "matched_distance_sum_um", "matched_distance_max_um",
        ):
            row[f"{prefix}_{name}"] = metric[name]
    return row


def combine_case_outputs(
    output_dir: Path,
    config_digest: str,
    thresholds: Sequence[float],
) -> None:
    results = []
    for path in sorted((output_dir / "cases").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("config_hash") == config_digest:
            results.append(payload)
    flat = [flatten_case_result(row, thresholds) for row in results]
    if flat:
        atomic_write_csv(output_dir / "per_case.csv", list(flat[0]), flat)

    prediction_files = sorted((output_dir / "predicted_nodes_by_case").glob("*.csv"))
    prediction_rows: list[dict[str, Any]] = []
    prediction_fields = [
        "case_id", "stem", "frame", "node_index", "z", "y", "x",
        "z_um", "y_um", "x_um", "score", "threshold",
    ]
    for path in prediction_files:
        with path.open(newline="", encoding="utf-8") as handle:
            prediction_rows.extend(csv.DictReader(handle))
    atomic_write_csv(output_dir / "predicted_nodes.csv", prediction_fields, prediction_rows)


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.case_offset < 0:
        raise SystemExit("--case-offset must be non-negative")
    if args.nms_radius_um <= 0 or args.match_radius_um <= 0:
        raise SystemExit("radii must be positive")
    if args.max_peaks_per_case <= 0:
        raise SystemExit("--max-peaks-per-case must be positive")
    if not 0 <= args.export_threshold <= 1 or not 0 <= args.voxel_threshold <= 1:
        raise SystemExit("thresholds must lie in [0, 1]")

    split_path = args.preprocessed_dataset_dir / "splits_final.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    if args.fold >= len(splits):
        raise SystemExit(f"fold {args.fold} is outside splits_final.json ({len(splits)} folds)")
    if args.case_partition == "all":
        all_cases = list(dict.fromkeys([*splits[args.fold]["train"], *splits[args.fold]["val"]]))
    else:
        all_cases = list(splits[args.fold][args.case_partition])
    cases = all_cases[args.case_offset :]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit(f"no {args.case_partition} cases selected")

    os.environ["nnUNet_compile"] = "true" if args.compile else "false"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=False,
        use_mirroring=False,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(args.model_dir), use_folds=(args.fold,), checkpoint_name=args.checkpoint
    )
    network = predictor.network.to(device)
    network.eval()
    data_identifier = predictor.configuration_manager.data_identifier
    data_dir = args.preprocessed_dataset_dir / data_identifier
    if not data_dir.is_dir():
        raise SystemExit(f"preprocessed data directory not found: {data_dir}")
    dataset = nnUNetDatasetBlosc2(str(data_dir), identifiers=cases)
    temporal_dilated_motion = "TemporalDilatedBackwardMotion" in args.model_dir.name
    temporal_dilated_no_motion = "TemporalDilatedNoMotion" in args.model_dir.name
    temporal_dilated_input = temporal_dilated_motion or temporal_dilated_no_motion
    temporal_three_frame = "Temporal3Frame" in args.model_dir.name or temporal_dilated_input
    if temporal_dilated_input:
        from nnunetv2.training.nnUNetTrainer.variants.biohub.nnUNetTrainerBiohubTemporalDilatedBackwardMotion import (
            BiohubTemporalDilatedMotionValidationDataset,
        )

        dataset = BiohubTemporalDilatedMotionValidationDataset(
            str(data_dir), identifiers=cases
        )
    elif temporal_three_frame:
        from nnunetv2.training.nnUNetTrainer.variants.biohub.nnUNetTrainerBiohubTemporalThreeFrame import (
            BiohubTemporalThreeFrameDataset,
        )

        dataset = BiohubTemporalThreeFrameDataset(str(data_dir), identifiers=cases)
    spacing = tuple(float(v) for v in predictor.configuration_manager.spacing)
    required_stride = cumulative_stride(predictor.configuration_manager)

    config = direct_cv_config(args, cases, data_identifier)
    config.update(
        {
            "selected_partition_cases_in_fold": len(all_cases),
            "spacing_zyx_um": list(spacing),
            "required_stride_zyx": list(required_stride),
            "network_class": network.__class__.__name__,
            "temporal_three_frame": temporal_three_frame,
            "temporal_dilated_input": temporal_dilated_input,
            "temporal_dilated_motion": temporal_dilated_motion,
            "temporal_dilated_no_motion": temporal_dilated_no_motion,
            "temporal_input_order": ["previous", "centre", "next"] if temporal_three_frame else None,
        }
    )
    digest = prepare_output_dir(args, config)
    print(json.dumps({"config_hash": digest, **config}, indent=2), flush=True)

    prediction_fields = [
        "case_id", "stem", "frame", "node_index", "z", "y", "x",
        "z_um", "y_um", "x_um", "score", "threshold",
    ]
    completed = 0
    for index, case_id in enumerate(cases, start=1):
        case_json = args.output_dir / "cases" / f"{case_id}.json"
        case_predictions = args.output_dir / "predicted_nodes_by_case" / f"{case_id}.csv"
        if args.resume and case_json.exists() and case_predictions.exists():
            previous = json.loads(case_json.read_text(encoding="utf-8"))
            if previous.get("config_hash") == digest:
                print(f"[skip {index}/{len(cases)}] {case_id}", flush=True)
                continue
        result, prediction_rows, probability = process_case(
            case_id, dataset, network, device, required_stride, spacing, args
        )
        result["config_hash"] = digest
        atomic_write_json(case_json, result)
        atomic_write_csv(
            case_predictions,
            prediction_fields,
            prediction_rows,
        )
        if args.save_probabilities:
            destination = args.output_dir / "probabilities" / f"{case_id}.npz"
            temporary = destination.with_name(destination.name + ".tmp.npz")
            np.savez_compressed(temporary, probability=probability.astype(np.float16))
            os.replace(temporary, destination)
        completed += 1
        print(
            f"[case {index}/{len(cases)}] {case_id} shape={result['input_shape_czyx']} "
            f"forward={result['forward_seconds']:.3f}s "
            f"peak_mem={result['cuda_peak_memory_bytes'] / 2**30:.2f}GiB "
            f"gt={result['gt_nodes']}",
            flush=True,
        )

    summary = aggregate_results(args.output_dir, digest, args.peak_thresholds, args.export_threshold)
    summary["config_hash"] = digest
    summary["newly_completed_cases"] = completed
    atomic_write_json(args.output_dir / "summary.json", summary)
    combine_case_outputs(args.output_dir, digest, args.peak_thresholds)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
