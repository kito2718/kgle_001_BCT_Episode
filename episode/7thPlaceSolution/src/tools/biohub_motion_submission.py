#!/usr/bin/env python3
"""Run raw-Zarr motion-head nnU-Net inference and per-video ILP tracking.

This is the shared runtime used by the Kaggle submission notebook. It mirrors
the fold evaluator: direct full-volume inference, physical NMS, backward-motion
candidate generation, and tracksdata ILP. No ground-truth data are read.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tracksdata as td
import zarr
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from biohub_motion_ilp_fold import (
    SPACING_ZYX_UM,
    add_candidate_edges,
    build_graph,
    save_graph,
    suppress_solver_output,
)
from biohub_nnunet_direct_full_volume_cv import (
    amp_context,
    cumulative_stride,
    local_maximum_candidates,
    pad_full_volume_to_stride,
    physical_greedy_nms,
)
from evaluate_biohub_backward_motion_head import weighted_sample
def parse_dilations(value: str) -> tuple[int, ...]:
    result = tuple(
        sorted({int(item) for item in value.replace(",", " ").split()})
    )
    if not result or result[0] != 1 or any(item not in (1, 2) for item in result):
        raise argparse.ArgumentTypeError(
            "dilations must contain 1 and may optionally contain 2"
        )
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--dilations", type=parse_dilations, default=(1,))
    parser.add_argument("--det-threshold", type=float, default=0.3)
    parser.add_argument("--nms-radius-um", type=float, default=2.0)
    parser.add_argument("--max-peaks-per-frame", type=int, default=2048)
    parser.add_argument("--motion-scale-um", type=float, default=8.0)
    parser.add_argument("--sample-radius-um", type=float, default=2.0)
    parser.add_argument("--sample-sigma-um", type=float, default=1.0)
    parser.add_argument("--gate-um", type=float, default=7.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--distance-sigma-um", type=float, default=3.0)
    parser.add_argument("--confidence-power", type=float, default=0.5)
    parser.add_argument("--edge-weight", type=float, default=-1.0)
    parser.add_argument("--appearance-weight", type=float, default=0.1)
    parser.add_argument("--disappearance-weight", type=float, default=0.1)
    parser.add_argument("--division-weight", type=float, default=1.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def zscore_frame(frame: np.ndarray) -> np.ndarray:
    """Match nnU-Net's ZScoreNormalization(use_mask_for_norm=False)."""

    result = np.asarray(frame, dtype=np.float32)
    mean = result.mean()
    std = result.std()
    return (result - mean) / max(float(std), 1e-8)


def joint_forward(
    network: torch.nn.Module,
    data: np.ndarray,
    device: torch.device,
    amp: str,
    required_stride: tuple[int, int, int],
    motion_scale_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    original_shape = tuple(int(value) for value in data.shape[-3:])
    tensor = torch.from_numpy(
        np.ascontiguousarray(data, dtype=np.float32)
    ).unsqueeze(0)
    tensor, _ = pad_full_volume_to_stride(tensor, required_stride)
    padded_shape = tuple(int(value) for value in tensor.shape[-3:])
    tensor = tensor.to(device, non_blocking=True)
    with torch.inference_mode(), amp_context(device, amp):
        output = network(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim != 5 or output.shape[1] < 5:
            raise RuntimeError(
                f"expected Bx5xZxYxX network output, got {tuple(output.shape)}"
            )
        if tuple(output.shape[-3:]) != padded_shape:
            output = F.interpolate(
                output.float(),
                size=padded_shape,
                mode="trilinear",
                align_corners=False,
            )
        cropped = output[
            0,
            :5,
            : original_shape[0],
            : original_shape[1],
            : original_shape[2],
        ].float()
        probability = torch.sigmoid(cropped[1] - cropped[0])
        motion = cropped[2:5] * float(motion_scale_um)
    probability_np = probability.cpu().numpy().astype(np.float32, copy=False)
    motion_np = motion.cpu().numpy().astype(np.float32, copy=False)
    del cropped, probability, motion, output, tensor
    return probability_np, motion_np


def peaks_from_probability(
    probability: np.ndarray,
    spacing: np.ndarray,
    threshold: float,
    nms_radius_um: float,
    max_peaks: int,
) -> tuple[np.ndarray, np.ndarray]:
    coords, scores = local_maximum_candidates(probability, threshold)
    return physical_greedy_nms(
        coords,
        scores,
        spacing,
        threshold,
        nms_radius_um,
        max_peaks,
    )


def temporal_input(
    normalized_frame,
    frame: int,
    frame_count: int,
    dilation: int,
) -> np.ndarray:
    def resolved(index: int) -> int:
        return index if 0 <= index < frame_count else frame

    current = normalized_frame(frame)
    return np.stack(
        (
            normalized_frame(resolved(frame - dilation)),
            current,
            normalized_frame(resolved(frame + dilation)),
            np.full_like(current, float(dilation) / 2.0),
        ),
        axis=0,
    )


def infer_video(
    stem: str,
    zarr_path: Path,
    graph_path: Path,
    network: torch.nn.Module,
    device: torch.device,
    required_stride: tuple[int, int, int],
    spacing: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    started = time.perf_counter()
    array = zarr.open_group(str(zarr_path), mode="r")["0"]
    frame_count = int(array.shape[0])
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)

    @lru_cache(maxsize=7)
    def normalized_frame(frame: int) -> np.ndarray:
        return zscore_frame(np.asarray(array[frame]))

    frame_payload: dict[int, dict[str, object]] = {}
    forward_count = 0
    peak_count = 0
    for frame in range(frame_count):
        data = temporal_input(normalized_frame, frame, frame_count, 1)
        probability, motion = joint_forward(
            network,
            data,
            device,
            args.amp,
            required_stride,
            args.motion_scale_um,
        )
        forward_count += 1
        coords, scores = peaks_from_probability(
            probability,
            spacing,
            args.det_threshold,
            args.nms_radius_um,
            args.max_peaks_per_frame,
        )
        vectors_by_dilation: dict[int, np.ndarray] = {
            1: np.asarray(
                [
                    weighted_sample(
                        motion,
                        coord,
                        spacing,
                        args.sample_radius_um,
                        args.sample_sigma_um,
                    )
                    for coord in coords
                ],
                dtype=np.float32,
            ).reshape(-1, 3)
        }
        del data, probability, motion

        for dilation in args.dilations:
            if dilation == 1:
                continue
            data = temporal_input(
                normalized_frame, frame, frame_count, dilation
            )
            _, dilated_motion = joint_forward(
                network,
                data,
                device,
                args.amp,
                required_stride,
                args.motion_scale_um,
            )
            forward_count += 1
            vectors_by_dilation[dilation] = np.asarray(
                [
                    weighted_sample(
                        dilated_motion,
                        coord,
                        spacing,
                        args.sample_radius_um,
                        args.sample_sigma_um,
                    )
                    for coord in coords
                ],
                dtype=np.float32,
            ).reshape(-1, 3)
            del data, dilated_motion

        frame_payload[frame] = {
            "coords": coords.astype(np.float64, copy=False),
            "scores": scores.astype(np.float64, copy=False),
            "motion": vectors_by_dilation,
        }
        peak_count += len(coords)
        if frame % 10 == 0 or frame + 1 == frame_count:
            print(
                f"{stem}: frame={frame + 1}/{frame_count} "
                f"peaks={peak_count}",
                flush=True,
            )

    node_rows: list[dict[str, float | int]] = []
    for frame in range(frame_count):
        payload = frame_payload[frame]
        coords = payload["coords"]
        scores = payload["scores"]
        indices = np.arange(
            len(node_rows), len(node_rows) + len(coords), dtype=np.int64
        )
        payload["indices"] = indices
        node_rows.extend(
            {
                "t": frame,
                "z": float(coord[0]),
                "y": float(coord[1]),
                "x": float(coord[2]),
                "node_score": float(score),
            }
            for coord, score in zip(coords, scores)
        )

    edges: dict[tuple[int, int], tuple[float, float, int]] = {}
    for frame in range(frame_count):
        target = frame_payload[frame]
        for dilation in args.dilations:
            if frame < dilation:
                continue
            source = frame_payload[frame - dilation]
            add_candidate_edges(
                edges,
                source["indices"],
                source["coords"],
                source["scores"],
                target["indices"],
                target["coords"],
                target["scores"],
                target["motion"][dilation],
                dilation,
                "learned",
                args.gate_um,
                args.top_k,
                args.distance_sigma_um,
                args.confidence_power,
            )

    graph = build_graph(node_rows, edges)
    candidate_edges = int(graph.num_edges())
    solve_seconds = 0.0
    if candidate_edges:
        solver = td.solvers.ILPSolver(
            edge_weight=args.edge_weight * td.EdgeAttr("edge_prob"),
            appearance_weight=args.appearance_weight,
            disappearance_weight=args.disappearance_weight,
            division_weight=args.division_weight,
            num_threads=1,
        )
        solve_started = time.perf_counter()
        with suppress_solver_output():
            graph = solver.solve(graph)
        solve_seconds = time.perf_counter() - solve_started
    save_graph(graph, graph_path)
    return {
        "stem": stem,
        "frames": frame_count,
        "forwards": forward_count,
        "nodes": int(graph.num_nodes()),
        "candidate_edges": candidate_edges,
        "selected_edges": int(graph.num_edges()),
        "solve_seconds": solve_seconds,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = arguments()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    if not 0 < args.det_threshold <= 1:
        raise ValueError("det-threshold must be in (0, 1]")
    if args.top_k < 1 or args.max_peaks_per_frame < 1:
        raise ValueError("top-k and max-peaks-per-frame must be positive")

    all_paths = sorted(args.test_dir.glob("*.zarr"))
    if not all_paths:
        raise FileNotFoundError(f"no test .zarr directories in {args.test_dir}")
    paths = all_paths[args.shard_index :: args.num_shards]
    if args.limit_videos is not None:
        paths = paths[: args.limit_videos]
    if not paths:
        raise RuntimeError(
            f"shard {args.shard_index}/{args.num_shards} has no videos"
        )

    os.environ["nnUNet_compile"] = "false"
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
        str(args.model_dir),
        use_folds=(args.fold,),
        checkpoint_name=args.checkpoint,
    )
    network = predictor.network.to(device).eval()
    spacing = np.asarray(
        predictor.configuration_manager.spacing, dtype=np.float64
    )
    if not np.allclose(spacing, SPACING_ZYX_UM):
        raise RuntimeError(
            f"unexpected model spacing {spacing}; expected {SPACING_ZYX_UM}"
        )
    required_stride = cumulative_stride(predictor.configuration_manager)

    graph_dir = args.output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(paths, start=1):
        stem = path.name.removesuffix(".zarr")
        graph_path = graph_dir / f"{stem}.geff"
        if args.resume and graph_path.exists():
            print(f"[{index}/{len(paths)}] resume skip {stem}", flush=True)
            continue
        row = infer_video(
            stem,
            path,
            graph_path,
            network,
            device,
            required_stride,
            spacing,
            args,
        )
        rows.append(row)
        print(
            f"[{index}/{len(paths)}] {stem}: "
            f"nodes={row['nodes']} edges={row['candidate_edges']}"
            f"->{row['selected_edges']} elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "schema_version": 1,
        "scope": "raw Zarr inference without GT",
        "model_dir": str(args.model_dir),
        "fold": args.fold,
        "checkpoint": args.checkpoint,
        "all_test_videos": len(all_paths),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "assigned_videos": [path.name.removesuffix(".zarr") for path in paths],
        "completed_now": rows,
        "config": {
            "dilations": list(args.dilations),
            "det_threshold": args.det_threshold,
            "nms_radius_um": args.nms_radius_um,
            "motion_scale_um": args.motion_scale_um,
            "gate_um": args.gate_um,
            "top_k": args.top_k,
            "distance_sigma_um": args.distance_sigma_um,
            "edge_weight": args.edge_weight,
            "appearance_weight": args.appearance_weight,
            "disappearance_weight": args.disappearance_weight,
            "division_weight": args.division_weight,
        },
    }
    (args.output_dir / f"summary_shard_{args.shard_index}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
