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
import tracksdata as td
import zarr
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from biohub_planar_d4_tta import (
    PLANAR_D4_VIEWS,
    detector_logit_mean_and_identity_motion,
)
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
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable cuDNN autotuning for repeated fixed-shape inference.",
    )
    parser.add_argument("--dilations", type=parse_dilations, default=(1,))
    parser.add_argument("--det-threshold", type=float, default=0.3)
    parser.add_argument(
        "--detector-tta-mode",
        choices=("none", "d4-detector"),
        default="none",
        help=(
            "Average the eight inverse-transformed planar D4 detector logit "
            "differences before sigmoid. Motion remains the identity-view output."
        ),
    )
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
    parser.add_argument("--batch-size", type=int, default=1)
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


def joint_forward_batch(
    network: torch.nn.Module,
    data: np.ndarray,
    device: torch.device,
    amp: str,
    required_stride: tuple[int, int, int],
    motion_scale_um: float,
    detector_tta_mode: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    if data.ndim != 5:
        raise ValueError(
            f"expected BxCxZxYxX input, got shape {tuple(data.shape)}"
        )
    original_shape = tuple(int(value) for value in data.shape[-3:])
    tensor = torch.from_numpy(
        np.ascontiguousarray(data, dtype=np.float32)
    )
    tensor, _ = pad_full_volume_to_stride(tensor, required_stride)
    tensor = tensor.to(device, non_blocking=True)
    views = (
        PLANAR_D4_VIEWS
        if detector_tta_mode == "d4-detector"
        else PLANAR_D4_VIEWS[:1]
    )
    with torch.inference_mode(), amp_context(device, amp):
        mean_logit, identity_motion = detector_logit_mean_and_identity_motion(
            network, tensor, views
        )
        mean_logit = mean_logit[
            ...,
            : original_shape[0],
            : original_shape[1],
            : original_shape[2],
        ]
        identity_motion = identity_motion[
            ...,
            : original_shape[0],
            : original_shape[1],
            : original_shape[2],
        ]
        probability = torch.sigmoid(mean_logit[:, 0])
        motion = identity_motion * float(motion_scale_um)
    probability_np = probability.cpu().numpy().astype(np.float32, copy=False)
    motion_np = motion.cpu().numpy().astype(np.float32, copy=False)
    del (
        probability,
        motion,
        tensor,
        mean_logit,
        identity_motion,
    )
    return probability_np, motion_np


def joint_forward(
    network: torch.nn.Module,
    data: np.ndarray,
    device: torch.device,
    amp: str,
    required_stride: tuple[int, int, int],
    motion_scale_um: float,
    detector_tta_mode: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    probability, motion = joint_forward_batch(
        network,
        np.expand_dims(data, axis=0),
        device,
        amp,
        required_stride,
        motion_scale_um,
        detector_tta_mode,
    )
    return probability[0], motion[0]


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
    phase_seconds = {
        "zarr_open": 0.0,
        "normalize": 0.0,
        "temporal_stack": 0.0,
        "forward": 0.0,
        "peak_nms": 0.0,
        "motion_sample": 0.0,
        "node_table": 0.0,
        "candidate_edges": 0.0,
        "graph_build": 0.0,
        "graph_save": 0.0,
    }
    phase_started = time.perf_counter()
    array = zarr.open_group(str(zarr_path), mode="r")["0"]
    phase_seconds["zarr_open"] += time.perf_counter() - phase_started
    frame_count = int(array.shape[0])
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)

    @lru_cache(maxsize=7)
    def normalized_frame(frame: int) -> np.ndarray:
        phase_started = time.perf_counter()
        result = zscore_frame(np.asarray(array[frame]))
        phase_seconds["normalize"] += time.perf_counter() - phase_started
        return result

    frame_payload: dict[int, dict[str, object]] = {}
    forward_count = 0
    network_call_count = 0
    peak_count = 0
    for batch_start in range(0, frame_count, args.batch_size):
        batch_frames = list(
            range(
                batch_start,
                min(frame_count, batch_start + args.batch_size),
            )
        )
        phase_started = time.perf_counter()
        data = np.stack(
            [
                temporal_input(normalized_frame, frame, frame_count, 1)
                for frame in batch_frames
            ],
            axis=0,
        )
        phase_seconds["temporal_stack"] += (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        probabilities, motions = joint_forward_batch(
            network,
            data,
            device,
            args.amp,
            required_stride,
            args.motion_scale_um,
            args.detector_tta_mode,
        )
        phase_seconds["forward"] += time.perf_counter() - phase_started
        detector_views = (
            len(PLANAR_D4_VIEWS)
            if args.detector_tta_mode == "d4-detector"
            else 1
        )
        forward_count += len(batch_frames) * detector_views
        network_call_count += detector_views

        for batch_index, frame in enumerate(batch_frames):
            probability = probabilities[batch_index]
            motion = motions[batch_index]
            phase_started = time.perf_counter()
            coords, scores = peaks_from_probability(
                probability,
                spacing,
                args.det_threshold,
                args.nms_radius_um,
                args.max_peaks_per_frame,
            )
            phase_seconds["peak_nms"] += (
                time.perf_counter() - phase_started
            )
            phase_started = time.perf_counter()
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
            phase_seconds["motion_sample"] += (
                time.perf_counter() - phase_started
            )
            frame_payload[frame] = {
                "coords": coords.astype(np.float64, copy=False),
                "scores": scores.astype(np.float64, copy=False),
                "motion": vectors_by_dilation,
            }
            peak_count += len(coords)
        del data, probabilities, motions

        for dilation in args.dilations:
            if dilation == 1:
                continue
            phase_started = time.perf_counter()
            data = np.stack(
                [
                    temporal_input(
                        normalized_frame,
                        frame,
                        frame_count,
                        dilation,
                    )
                    for frame in batch_frames
                ],
                axis=0,
            )
            phase_seconds["temporal_stack"] += (
                time.perf_counter() - phase_started
            )
            phase_started = time.perf_counter()
            _, dilated_motions = joint_forward_batch(
                network,
                data,
                device,
                args.amp,
                required_stride,
                args.motion_scale_um,
                "none",
            )
            phase_seconds["forward"] += time.perf_counter() - phase_started
            forward_count += len(batch_frames)
            network_call_count += 1
            for batch_index, frame in enumerate(batch_frames):
                payload = frame_payload[frame]
                phase_started = time.perf_counter()
                payload["motion"][dilation] = np.asarray(
                    [
                        weighted_sample(
                            dilated_motions[batch_index],
                            coord,
                            spacing,
                            args.sample_radius_um,
                            args.sample_sigma_um,
                        )
                        for coord in payload["coords"]
                    ],
                    dtype=np.float32,
                ).reshape(-1, 3)
                phase_seconds["motion_sample"] += (
                    time.perf_counter() - phase_started
                )
            del data, dilated_motions

        batch_end = batch_frames[-1] + 1
        if (
            batch_start == 0
            or batch_end == frame_count
            or batch_end % 10 == 0
        ):
            print(
                f"{stem}: frame={batch_end}/{frame_count} "
                f"peaks={peak_count}",
                flush=True,
            )

    phase_started = time.perf_counter()
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
    phase_seconds["node_table"] += time.perf_counter() - phase_started

    phase_started = time.perf_counter()
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
    phase_seconds["candidate_edges"] += (
        time.perf_counter() - phase_started
    )

    phase_started = time.perf_counter()
    graph = build_graph(node_rows, edges)
    phase_seconds["graph_build"] += time.perf_counter() - phase_started
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
    phase_started = time.perf_counter()
    save_graph(graph, graph_path)
    phase_seconds["graph_save"] += time.perf_counter() - phase_started
    return {
        "stem": stem,
        "frames": frame_count,
        "forwards": forward_count,
        "network_calls": network_call_count,
        "nodes": int(graph.num_nodes()),
        "candidate_edges": candidate_edges,
        "selected_edges": int(graph.num_edges()),
        "solve_seconds": solve_seconds,
        "phase_seconds": {
            key: round(value, 6) for key, value in phase_seconds.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    main_started = time.perf_counter()
    args = arguments()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    if not 0 < args.det_threshold <= 1:
        raise ValueError("det-threshold must be in (0, 1]")
    if args.top_k < 1 or args.max_peaks_per_frame < 1:
        raise ValueError("top-k and max-peaks-per-frame must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=False,
        use_mirroring=False,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        allow_tqdm=False,
    )
    model_init_started = time.perf_counter()
    predictor.initialize_from_trained_model_folder(
        str(args.model_dir),
        use_folds=(args.fold,),
        checkpoint_name=args.checkpoint,
    )
    network = predictor.network.to(device).eval()
    model_init_seconds = time.perf_counter() - model_init_started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
        "timing": {
            "model_init_seconds": model_init_seconds,
            "main_elapsed_seconds": time.perf_counter() - main_started,
        },
        "device": {
            "type": device.type,
            "name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "config": {
            "dilations": list(args.dilations),
            "det_threshold": args.det_threshold,
            "detector_tta_mode": args.detector_tta_mode,
            "detector_tta_views": (
                [view.name for view in PLANAR_D4_VIEWS]
                if args.detector_tta_mode == "d4-detector"
                else [PLANAR_D4_VIEWS[0].name]
            ),
            "nms_radius_um": args.nms_radius_um,
            "motion_scale_um": args.motion_scale_um,
            "gate_um": args.gate_um,
            "top_k": args.top_k,
            "distance_sigma_um": args.distance_sigma_um,
            "edge_weight": args.edge_weight,
            "appearance_weight": args.appearance_weight,
            "disappearance_weight": args.disappearance_weight,
            "division_weight": args.division_weight,
            "batch_size": args.batch_size,
            "amp": args.amp,
            "cudnn_benchmark": args.cudnn_benchmark,
        },
    }
    (args.output_dir / f"summary_shard_{args.shard_index}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
