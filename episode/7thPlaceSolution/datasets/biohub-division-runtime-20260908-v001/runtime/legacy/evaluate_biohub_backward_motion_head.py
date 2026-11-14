#!/usr/bin/env python3
"""Evaluate the Biohub backward-motion head at GT cell centres."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import geff
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scipy.optimize import linear_sum_assignment

from biohub_nnunet_direct_full_volume_cv import (
    amp_context,
    atomic_write_csv,
    atomic_write_json,
    case_stem_frame,
    cumulative_stride,
    pad_full_volume_to_stride,
)
from biohub_nnunet_output_contract import split_detector_motion_logits


SCALE_UM = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--preprocessed-dataset-dir", type=Path, required=True)
    parser.add_argument("--geff-dir", type=Path, required=True)
    parser.add_argument("--movement-csv", type=Path, required=True)
    parser.add_argument("--motion-lookup", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--dilations", default="1,2")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--amp", default="bf16", choices=("none", "fp16", "bf16"))
    parser.add_argument("--motion-scale-um", type=float, default=8.0)
    parser.add_argument("--sample-radius-um", type=float, default=2.0)
    parser.add_argument("--sample-sigma-um", type=float, default=1.0)
    parser.add_argument("--predicted-nodes-dir", type=Path)
    parser.add_argument("--detector-match-radius-um", type=float, default=7.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def rounded_zyx(attrs: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(round(float(attrs[axis])) for axis in ("z", "y", "x"))


def movement_lookup(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[(row["stem"], int(row["source_id"]), int(row["target_id"]))] = row
    return result


class GraphInfo:
    def __init__(self, path: Path):
        self.graph, _ = geff.read(
            str(path), node_props=["t", "z", "y", "x"], structure_validation=False
        )
        self.nodes_by_t: dict[int, list[int]] = defaultdict(list)
        self.nodes_by_t_coord: dict[tuple[int, tuple[int, int, int]], list[int]] = defaultdict(list)
        for node_raw, attrs in self.graph.nodes(data=True):
            node = int(node_raw)
            frame = int(attrs["t"])
            self.nodes_by_t[frame].append(node)
            self.nodes_by_t_coord[(frame, rounded_zyx(attrs))].append(node)

    def resolve_pair(
        self,
        current_t: int,
        dilation: int,
        current_coord: tuple[int, int, int],
        ancestor_coord: tuple[int, int, int],
    ) -> tuple[int | None, int | None]:
        ancestors = set(self.nodes_by_t_coord.get((current_t - dilation, ancestor_coord), ()))
        for current in self.nodes_by_t_coord.get((current_t, current_coord), ()):
            for ancestor in ancestors:
                try:
                    path = nx.shortest_path(self.graph, ancestor, current)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if int(self.graph.nodes[path[0]]["t"]) == current_t - dilation:
                    return int(current), int(ancestor)
        return None, None

    def path(self, ancestor: int, current: int) -> list[int]:
        try:
            return [int(value) for value in nx.shortest_path(self.graph, ancestor, current)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def physical(self, node: int) -> np.ndarray:
        attrs = self.graph.nodes[node]
        return np.asarray([attrs["z"], attrs["y"], attrs["x"]], dtype=np.float64) * SCALE_UM


def direct_motion_forward(
    network: torch.nn.Module,
    data: np.ndarray,
    device: torch.device,
    amp: str,
    required_stride,
    motion_scale_um: float,
    output_contract: str = "auto",
) -> np.ndarray:
    original_shape = tuple(int(value) for value in data.shape[-3:])
    tensor = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).unsqueeze(0)
    tensor, _ = pad_full_volume_to_stride(tensor, required_stride)
    padded_shape = tuple(int(value) for value in tensor.shape[-3:])
    tensor = tensor.to(device, non_blocking=True)
    with torch.inference_mode(), amp_context(device, amp):
        output = network(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        _, motion_logits, resolved_contract = split_detector_motion_logits(
            output, output_contract
        )
        if motion_logits is None:
            raise RuntimeError(
                f"output contract {resolved_contract} does not contain motion channels"
            )
        if tuple(output.shape[-3:]) != padded_shape:
            output = F.interpolate(output.float(), size=padded_shape, mode="trilinear", align_corners=False)
        motion = motion_logits[
            0, :, : original_shape[0], : original_shape[1], : original_shape[2]
        ]
        motion = motion.float() * float(motion_scale_um)
    result = motion.cpu().numpy().astype(np.float32, copy=False)
    del motion, output, tensor
    return result


def weighted_sample(
    motion: np.ndarray,
    center: np.ndarray,
    spacing: np.ndarray,
    radius_um: float,
    sigma_um: float,
) -> np.ndarray:
    shape = np.asarray(motion.shape[1:], dtype=int)
    radii = np.maximum(1, np.ceil(radius_um / spacing).astype(int))
    rounded = np.rint(center).astype(int)
    if np.any(rounded < 0) or np.any(rounded >= shape):
        raise ValueError(
            "motion sample center is outside the motion tensor: "
            f"center={np.asarray(center).tolist()}, rounded={rounded.tolist()}, "
            f"shape={shape.tolist()}"
        )
    lower = np.maximum(rounded - radii, 0)
    upper = np.minimum(rounded + radii + 1, shape)
    axes = [np.arange(lo, hi, dtype=np.float64) for lo, hi in zip(lower, upper)]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    coords = np.stack([zz, yy, xx], axis=0)
    distance = np.sqrt(np.sum(np.square((coords - center[:, None, None, None]) * spacing[:, None, None, None]), axis=0))
    valid = distance <= radius_um
    weights = np.exp(-np.square(distance) / (2 * sigma_um**2)) * valid
    patch = motion[
        :,
        lower[0] : upper[0],
        lower[1] : upper[1],
        lower[2] : upper[2],
    ]
    denominator = float(weights.sum())
    if denominator <= 0:
        return motion[:, rounded[0], rounded[1], rounded[2]].astype(np.float64)
    return (patch * weights[None]).sum(axis=(1, 2, 3), dtype=np.float64) / denominator


def original_to_preprocessed_coordinates(
    original_zyx: np.ndarray,
    properties: dict[str, Any],
    preprocessed_spacing_zyx: np.ndarray,
) -> np.ndarray:
    """Map exported original-volume coordinates back into network coordinates."""

    coords = np.asarray(original_zyx, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"expected Nx3 original coordinates, got {coords.shape}")
    crop_bbox = properties.get("bbox_used_for_cropping")
    if crop_bbox is None or len(crop_bbox) != 3:
        raise ValueError("bbox_used_for_cropping is required for motion sampling")
    crop_offset = np.asarray([axis[0] for axis in crop_bbox], dtype=np.float64)
    original_spacing = np.asarray(
        properties.get("spacing", preprocessed_spacing_zyx), dtype=np.float64
    )
    preprocessed_spacing = np.asarray(
        preprocessed_spacing_zyx, dtype=np.float64
    )
    if np.any(original_spacing <= 0) or np.any(preprocessed_spacing <= 0):
        raise ValueError(
            "spacing values must be positive: "
            f"original={original_spacing.tolist()}, "
            f"preprocessed={preprocessed_spacing.tolist()}"
        )
    return (
        (coords - crop_offset[None])
        * original_spacing[None]
        / preprocessed_spacing[None]
    )


def load_predicted_nodes(path: Path | None, case_id: str) -> np.ndarray:
    if path is None:
        return np.empty((0, 4), dtype=np.float64)
    case_path = path / f"{case_id}.csv"
    if not case_path.exists():
        return np.empty((0, 4), dtype=np.float64)
    rows = []
    with case_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                [
                    float(row["z"]),
                    float(row["y"]),
                    float(row["x"]),
                    float(row["score"]),
                ]
            )
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def match_predicted_centres(
    predicted_zyx: np.ndarray,
    gt_zyx: np.ndarray,
    spacing: np.ndarray,
    radius_um: float,
) -> dict[int, tuple[int, float]]:
    """Maximum-cardinality, minimum-distance one-to-one radius matching."""

    n_pred, n_gt = len(predicted_zyx), len(gt_zyx)
    if n_pred == 0 or n_gt == 0:
        return {}
    distances = np.linalg.norm(
        (predicted_zyx[:, None] - gt_zyx[None]) * spacing[None, None], axis=2
    )
    eligible = np.flatnonzero(np.any(distances <= radius_um, axis=1))
    if len(eligible) == 0:
        return {}
    eligible_distances = distances[eligible]
    size = len(eligible) + n_gt
    unmatched_cost = radius_um + 1e-3
    cost = np.zeros((size, size), dtype=np.float64)
    cost[: len(eligible), :n_gt] = np.where(
        eligible_distances <= radius_um, eligible_distances, 1e6
    )
    cost[: len(eligible), n_gt:] = unmatched_cost
    cost[len(eligible) :, :n_gt] = unmatched_cost
    rows, columns = linear_sum_assignment(cost)
    return {
        int(column): (int(eligible[row]), float(eligible_distances[row, column]))
        for row, column in zip(rows, columns)
        if row < len(eligible)
        and column < n_gt
        and eligible_distances[row, column] <= radius_um
    }


def category_for_path(
    stem: str,
    graph_info: GraphInfo,
    ancestor: int | None,
    current: int | None,
    movements: dict[tuple[str, int, int], dict[str, str]],
) -> tuple[str, str]:
    if ancestor is None or current is None:
        return "unknown", ""
    path = graph_info.path(ancestor, current)
    edge_rows = [
        movements.get((stem, int(source), int(target)), {})
        for source, target in zip(path[:-1], path[1:])
    ]
    edge_types = [row.get("edge_type", "") for row in edge_rows]
    transitions = [row.get("transition_class", "") for row in edge_rows]
    if "division" in edge_types:
        category = "division"
    elif "freeze" in transitions:
        category = "freeze"
    elif "post_freeze" in transitions:
        category = "post_freeze"
    elif edge_rows:
        category = "normal"
    else:
        category = "unknown"
    return category, "|".join(transitions)


def evaluate_case_dilation(
    case_id: str,
    dilation: int,
    dataset,
    network,
    device,
    amp,
    required_stride,
    spacing: np.ndarray,
    motion_scale_um: float,
    sample_radius_um: float,
    sample_sigma_um: float,
    lookup_rows: list[list[int]],
    graph_info: GraphInfo,
    movements,
    predicted_nodes: np.ndarray | None = None,
    detector_match_radius_um: float = 7.0,
) -> list[dict[str, Any]]:
    dataset.dilation_two_probability = 1.0 if dilation == 2 else 0.0
    data_obj, seg_obj, _, _ = dataset.load_case(case_id)
    data = np.asarray(data_obj, dtype=np.float32)
    seg = np.asarray(seg_obj)
    if seg.shape[0] < 4:
        raise RuntimeError(f"motion target channels missing for {case_id}: {seg.shape}")
    current_ids, ancestor_ids = seg[1], seg[2]
    ids = [int(value) for value in np.unique(current_ids) if int(value) > 0]
    if not ids:
        return []
    motion = direct_motion_forward(
        network, data, device, amp, required_stride, motion_scale_um
    )
    current_centres = np.asarray(
        [
            np.argwhere(current_ids == correspondence_id)
            .astype(np.float64)
            .mean(axis=0)
            for correspondence_id in ids
        ]
    )
    predicted_nodes = (
        np.empty((0, 4), dtype=np.float64)
        if predicted_nodes is None
        else predicted_nodes
    )
    detector_matches = match_predicted_centres(
        predicted_nodes[:, :3],
        current_centres,
        spacing,
        detector_match_radius_um,
    )
    stem, frame = case_stem_frame(case_id)
    rows: list[dict[str, Any]] = []
    past_nodes = graph_info.nodes_by_t.get(frame - dilation, [])
    past_positions = (
        np.asarray([graph_info.physical(node) for node in past_nodes])
        if past_nodes
        else np.empty((0, 3), dtype=np.float64)
    )
    for current_index, correspondence_id in enumerate(ids):
        if correspondence_id > len(lookup_rows):
            continue
        current_pixels = np.argwhere(current_ids == correspondence_id)
        ancestor_pixels = np.argwhere(ancestor_ids == correspondence_id)
        if len(current_pixels) == 0 or len(ancestor_pixels) == 0:
            continue
        current_center = current_pixels.astype(np.float64).mean(axis=0)
        ancestor_center = ancestor_pixels.astype(np.float64).mean(axis=0)
        target_velocity = (ancestor_center - current_center) * spacing / float(dilation)
        prediction = weighted_sample(
            motion, current_center, spacing, sample_radius_um, sample_sigma_um
        )
        epe = float(np.linalg.norm((prediction - target_velocity) * dilation))
        detector_match = detector_matches.get(current_index)
        predicted_center = None
        predicted_score = None
        detector_localization_error_um = None
        predicted_center_vector_epe_um = None
        end_to_end_ancestor_epe_um = None
        if detector_match is not None:
            predicted_index, detector_localization_error_um = detector_match
            predicted_center = predicted_nodes[predicted_index, :3]
            predicted_score = float(predicted_nodes[predicted_index, 3])
            predicted_center_motion = weighted_sample(
                motion,
                predicted_center,
                spacing,
                sample_radius_um,
                sample_sigma_um,
            )
            predicted_center_vector_epe_um = float(
                np.linalg.norm(
                    (predicted_center_motion - target_velocity) * dilation
                )
            )
            projected_ancestor = (
                predicted_center
                + predicted_center_motion * dilation / spacing
            )
            end_to_end_ancestor_epe_um = float(
                np.linalg.norm((projected_ancestor - ancestor_center) * spacing)
            )
        zero_epe = float(np.linalg.norm(target_velocity * dilation))
        raw_row = lookup_rows[correspondence_id - 1]
        current_coord = tuple(int(value) for value in raw_row[:3])
        ancestor_coord = tuple(int(value) for value in raw_row[3:6])
        current_node, ancestor_node = graph_info.resolve_pair(
            frame, dilation, current_coord, ancestor_coord
        )
        category, transition_path = category_for_path(
            stem, graph_info, ancestor_node, current_node, movements
        )
        learned_top1 = None
        zero_top1 = None
        nearest_other_um = None
        if current_node is not None and ancestor_node is not None and len(past_nodes):
            current_physical = graph_info.physical(current_node)
            projected = current_physical + prediction * dilation
            learned_distances = np.linalg.norm(past_positions - projected[None], axis=1)
            zero_distances = np.linalg.norm(past_positions - current_physical[None], axis=1)
            correct_index = past_nodes.index(ancestor_node)
            learned_top1 = bool(learned_distances[correct_index] <= learned_distances.min() + 1e-7)
            zero_top1 = bool(zero_distances[correct_index] <= zero_distances.min() + 1e-7)
            if len(past_nodes) > 1:
                other = np.delete(
                    np.linalg.norm(past_positions - past_positions[correct_index][None], axis=1),
                    correct_index,
                )
                nearest_other_um = float(other.min())
        rows.append(
            {
                "case_id": case_id,
                "stem": stem,
                "frame": frame,
                "dilation": dilation,
                "correspondence_id": correspondence_id,
                "current_node": current_node,
                "ancestor_node": ancestor_node,
                "category": category,
                "transition_path": transition_path,
                "target_vz_um_per_frame": float(target_velocity[0]),
                "target_vy_um_per_frame": float(target_velocity[1]),
                "target_vx_um_per_frame": float(target_velocity[2]),
                "pred_vz_um_per_frame": float(prediction[0]),
                "pred_vy_um_per_frame": float(prediction[1]),
                "pred_vx_um_per_frame": float(prediction[2]),
                "target_displacement_um": zero_epe,
                "predicted_displacement_um": float(np.linalg.norm(prediction * dilation)),
                "learned_epe_um": epe,
                "detector_matched": detector_match is not None,
                "detector_score": predicted_score,
                "detector_localization_error_um": detector_localization_error_um,
                "predicted_center_vector_epe_um": predicted_center_vector_epe_um,
                "end_to_end_ancestor_epe_um": end_to_end_ancestor_epe_um,
                "zero_motion_epe_um": zero_epe,
                "learned_hit_2um": epe <= 2.0,
                "learned_hit_4um": epe <= 4.0,
                "learned_hit_7um": epe <= 7.0,
                "zero_hit_2um": zero_epe <= 2.0,
                "zero_hit_4um": zero_epe <= 4.0,
                "zero_hit_7um": zero_epe <= 7.0,
                "learned_top1": learned_top1,
                "zero_top1": zero_top1,
                "learned_top1_gate7": bool(learned_top1 and epe <= 7.0) if learned_top1 is not None else None,
                "zero_top1_gate7": bool(zero_top1 and zero_epe <= 7.0) if zero_top1 is not None else None,
                "nearest_other_ancestor_um": nearest_other_um,
                "dense_lt6um": nearest_other_um is not None and nearest_other_um < 6.0,
            }
        )
    return rows


def numeric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    learned = np.asarray([float(row["learned_epe_um"]) for row in rows])
    zero = np.asarray([float(row["zero_motion_epe_um"]) for row in rows])
    resolved = [row for row in rows if row["learned_top1"] is not None]
    result = {
        "n": len(rows),
        "learned_epe_mean_um": float(learned.mean()),
        "learned_epe_median_um": float(np.median(learned)),
        "learned_epe_p90_um": float(np.quantile(learned, 0.90)),
        "learned_epe_p95_um": float(np.quantile(learned, 0.95)),
        "zero_epe_mean_um": float(zero.mean()),
        "zero_epe_median_um": float(np.median(zero)),
        "zero_epe_p90_um": float(np.quantile(zero, 0.90)),
        "zero_epe_p95_um": float(np.quantile(zero, 0.95)),
    }
    for radius in (2, 4, 7):
        result[f"learned_hit_{radius}um"] = float(np.mean(learned <= radius))
        result[f"zero_hit_{radius}um"] = float(np.mean(zero <= radius))
    if resolved:
        result.update(
            {
                "resolved_nodes": len(resolved),
                "learned_top1": float(np.mean([bool(row["learned_top1"]) for row in resolved])),
                "zero_top1": float(np.mean([bool(row["zero_top1"]) for row in resolved])),
                "learned_top1_gate7": float(np.mean([bool(row["learned_top1_gate7"]) for row in resolved])),
                "zero_top1_gate7": float(np.mean([bool(row["zero_top1_gate7"]) for row in resolved])),
            }
        )
    return result


def aggregate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    slices: list[tuple[str, list[dict[str, Any]]]] = [("all", rows)]
    for dilation in (1, 2):
        subset = [row for row in rows if int(row["dilation"]) == dilation]
        slices.append((f"d{dilation}", subset))
        for category in ("normal", "freeze", "post_freeze", "division", "unknown"):
            slices.append((f"d{dilation}_{category}", [row for row in subset if row["category"] == category]))
    slices.extend(
        [
            ("target_lt3um", [row for row in rows if float(row["target_displacement_um"]) < 3]),
            ("target_3to7um", [row for row in rows if 3 <= float(row["target_displacement_um"]) < 7]),
            ("target_ge7um", [row for row in rows if float(row["target_displacement_um"]) >= 7]),
            ("dense_lt6um", [row for row in rows if bool(row["dense_lt6um"])]),
        ]
    )
    summary_rows = []
    for name, subset in slices:
        if not subset:
            continue
        summary_rows.append({"slice": name, **numeric_summary(subset)})
    headline = numeric_summary(rows)
    headline["cases_with_rows"] = len({row["case_id"] for row in rows})
    headline["dilations"] = sorted({int(row["dilation"]) for row in rows})
    return headline, summary_rows


def deployment_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [row for row in rows if bool(row.get("detector_matched"))]
    endpoint_errors = np.asarray(
        [float(row["end_to_end_ancestor_epe_um"]) for row in matched],
        dtype=np.float64,
    )
    vector_errors = np.asarray(
        [float(row["predicted_center_vector_epe_um"]) for row in matched],
        dtype=np.float64,
    )
    result = {
        "total_gt_correspondences": len(rows),
        "detector_matched_correspondences": len(matched),
        "coverage": len(matched) / len(rows) if rows else 0.0,
        "unconditional_endpoint_hit_7um": (
            float(np.sum(endpoint_errors <= 7.0) / len(rows)) if rows else 0.0
        ),
    }
    if len(matched):
        result.update(
            {
                "conditional_vector_epe_median_um": float(
                    np.median(vector_errors)
                ),
                "conditional_endpoint_epe_median_um": float(
                    np.median(endpoint_errors)
                ),
                "conditional_endpoint_epe_p90_um": float(
                    np.quantile(endpoint_errors, 0.90)
                ),
                "conditional_endpoint_hit_7um": float(
                    np.mean(endpoint_errors <= 7.0)
                ),
            }
        )
    return result


def write_report(output: Path, summary: dict[str, Any], slices: list[dict[str, Any]]) -> None:
    lines = [
        "# Backward motion head GT-centred evaluation",
        "",
        "This is a GT-centred motion-head upper-bound, not end-to-end tracking.",
        "",
        "## Headline",
        "",
        f"- correspondences: {summary['n']:,}",
        f"- learned median EPE: {summary['learned_epe_median_um']:.4f} um",
        f"- zero-motion median EPE: {summary['zero_epe_median_um']:.4f} um",
        f"- learned <=7 um: {summary['learned_hit_7um']:.4%}",
        f"- zero-motion <=7 um: {summary['zero_hit_7um']:.4%}",
        f"- learned nearest-correct: {summary.get('learned_top1', float('nan')):.4%}",
        f"- zero-motion nearest-correct: {summary.get('zero_top1', float('nan')):.4%}",
        "",
        "## Slice summary",
        "",
    ]
    for row in slices:
        lines.append(
            f"- {row['slice']}: n={row['n']:,}, learned/zero median EPE "
            f"{row['learned_epe_median_um']:.3f}/{row['zero_epe_median_um']:.3f} um, "
            f"<=7 um {row['learned_hit_7um']:.2%}/{row['zero_hit_7um']:.2%}"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    dilations = tuple(int(value) for value in args.dilations.replace(",", " ").split())
    os.environ["BIOHUB_BACKWARD_MOTION_LOOKUP"] = str(args.motion_lookup)
    os.environ["nnUNet_compile"] = "false"
    lookup_payload = json.loads(args.motion_lookup.read_text())
    lookup_cases = lookup_payload.get("cases", lookup_payload)
    movements = movement_lookup(args.movement_csv)

    splits = json.loads((args.preprocessed_dataset_dir / "splits_final.json").read_text())
    cases = list(splits[args.fold]["val"])
    if args.limit is not None:
        cases = cases[: args.limit]
    device = torch.device(args.device)
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
    network = predictor.network.to(device).eval()
    spacing = np.asarray(predictor.configuration_manager.spacing, dtype=np.float64)
    required_stride = cumulative_stride(predictor.configuration_manager)
    data_dir = args.preprocessed_dataset_dir / predictor.configuration_manager.data_identifier
    from nnunetv2.training.nnUNetTrainer.variants.biohub.nnUNetTrainerBiohubTemporalDilatedBackwardMotion import (
        BiohubTemporalDilatedMotionDataset,
    )
    dataset = BiohubTemporalDilatedMotionDataset(str(data_dir), identifiers=cases)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_dir = args.output_dir / "cases"
    case_dir.mkdir(exist_ok=True)
    graph_cache: dict[str, GraphInfo] = {}
    for case_index, case_id in enumerate(cases, start=1):
        stem, _ = case_stem_frame(case_id)
        if stem not in graph_cache:
            graph_cache[stem] = GraphInfo(args.geff_dir / f"{stem}.geff")
        for dilation in dilations:
            lookup_rows = lookup_cases.get(case_id, {}).get(str(dilation), [])
            if not lookup_rows:
                continue
            output_path = case_dir / f"d{dilation}_{case_id}.json"
            if args.resume and output_path.exists():
                continue
            rows = evaluate_case_dilation(
                case_id,
                dilation,
                dataset,
                network,
                device,
                args.amp,
                required_stride,
                spacing,
                args.motion_scale_um,
                args.sample_radius_um,
                args.sample_sigma_um,
                lookup_rows,
                graph_cache[stem],
                movements,
                load_predicted_nodes(args.predicted_nodes_dir, case_id),
                args.detector_match_radius_um,
            )
            atomic_write_json(output_path, {"case_id": case_id, "dilation": dilation, "rows": rows})
        if case_index % 50 == 0 or case_index == len(cases):
            print(f"processed={case_index}/{len(cases)}", flush=True)

    all_rows = []
    for path in sorted(case_dir.glob("d*_*.json")):
        all_rows.extend(json.loads(path.read_text())["rows"])
    if not all_rows:
        raise RuntimeError("no correspondence rows evaluated")
    summary, slice_rows = aggregate(all_rows)
    if args.predicted_nodes_dir is not None:
        summary["deployment"] = deployment_summary(all_rows)
    summary.update(
        {
            "model_dir": str(args.model_dir),
            "checkpoint": args.checkpoint,
            "validation_cases": len(cases),
            "motion_scale_um": args.motion_scale_um,
            "sample_radius_um": args.sample_radius_um,
            "sample_sigma_um": args.sample_sigma_um,
            "scope_note": (
                "Includes detector coverage and predicted-centre endpoint error; "
                "full graph assignment remains excluded."
                if args.predicted_nodes_dir is not None
                else "GT-centred motion-head upper-bound; detector misses and "
                "end-to-end edge assignment are excluded."
            ),
            "predicted_nodes_dir": (
                str(args.predicted_nodes_dir)
                if args.predicted_nodes_dir is not None
                else None
            ),
            "detector_match_radius_um": args.detector_match_radius_um,
        }
    )
    atomic_write_json(args.output_dir / "summary.json", summary)
    atomic_write_csv(args.output_dir / "per_edge.csv", list(all_rows[0]), all_rows)
    atomic_write_csv(args.output_dir / "slice_summary.csv", list(slice_rows[0]), slice_rows)
    write_report(args.output_dir, summary, slice_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
