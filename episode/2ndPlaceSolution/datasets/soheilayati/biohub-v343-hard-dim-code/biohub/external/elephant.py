"""Frozen ELEPHANT feature extraction for Biohub division patches.

This adapter runs the public ELEPHANT ``versatile3d`` segmentation network on
one aligned 3D frame at a time and summarizes its three probability maps.  It
does not modify or fine-tune the external checkpoint.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


TIME_OFFSETS = tuple(range(-3, 9))


def load_versatile3d(
    checkpoint: str | Path,
    elephant_root: str | Path,
    device: str = "cuda",
):
    """Load ELEPHANT's public 3D segmentation U-Net from vendored source."""
    import torch

    core = str(Path(elephant_root) / "elephant-core")
    if core not in sys.path:
        sys.path.insert(0, core)
    from elephant.models import UNet

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return UNet.three_class_segmentation(
        is_eval=True, device=device, state_dict=state)


def load_fluo_n3dh_ce_flow(
    checkpoint: str | Path,
    elephant_root: str | Path,
    device: str = "cuda",
):
    """Load ELEPHANT's public Fluo-N3DH-CE 3D flow network."""
    import torch

    core = str(Path(elephant_root) / "elephant-core")
    if core not in sys.path:
        sys.path.insert(0, core)
    from elephant.models import FlowResNet

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return FlowResNet.three_dimensional_flow(
        is_eval=True, device=device, state_dict=state)


def probability_summary(probability):
    """Summarize a batch of ELEPHANT probability maps.

    Input shape is ``(batch, 3, z, y, x)``.  Channel 0 is background,
    channel 1 is the outer nucleus, and channel 2 is the nucleus center.
    """
    import torch
    from torch.nn import functional as functional

    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)
    batch, _, depth, height, width = probability.shape
    flat = probability.flatten(2)
    values = []
    names = []
    channel_names = ("background", "outer", "center")
    for channel, name in enumerate(channel_names):
        data = flat[:, channel]
        values.extend([data.mean(1), data.std(1), data.amax(1)])
        names.extend([f"{name}_mean", f"{name}_std", f"{name}_max"])

    center = probability[:, 2]
    foreground = 1.0 - probability[:, 0]
    voxel = (depth // 2, height // 2, width // 2)
    top4 = center.flatten(1).topk(min(4, depth * height * width), dim=1).values.mean(1)
    top16 = center.flatten(1).topk(min(16, depth * height * width), dim=1).values.mean(1)
    values.extend([
        center[:, voxel[0], voxel[1], voxel[2]],
        foreground[:, voxel[0], voxel[1], voxel[2]],
        top4, top16,
        center[:, :, :, :width // 2].mean((1, 2, 3)),
        center[:, :, :, width // 2 + 1:].mean((1, 2, 3)),
        center[:, :, :, :width // 2].amax((1, 2, 3)),
        center[:, :, :, width // 2 + 1:].amax((1, 2, 3)),
        center[:, :, :, width // 2 - 1:width // 2 + 2].mean((1, 2, 3)),
        center[:, :, height // 2 - 1:height // 2 + 2, :].mean((1, 2, 3)),
    ])
    names.extend([
        "center_voxel", "foreground_voxel", "center_top4_mean",
        "center_top16_mean", "center_left_mean", "center_right_mean",
        "center_left_max", "center_right_max", "center_mid_x_mean",
        "center_mid_y_mean",
    ])

    local_max = center == functional.max_pool3d(center[:, None], 3, 1, 1)[:, 0]
    for threshold in (0.3, 0.5, 0.7):
        values.append((local_max & (center >= threshold)).sum((1, 2, 3)).float())
        names.append(f"center_peaks_ge_{str(threshold).replace('.', 'p')}")

    z_grid = torch.linspace(-1, 1, depth, device=center.device).view(1, depth, 1, 1)
    y_grid = torch.linspace(-1, 1, height, device=center.device).view(1, 1, height, 1)
    x_grid = torch.linspace(-1, 1, width, device=center.device).view(1, 1, 1, width)
    mass = center.sum((1, 2, 3)).clamp_min(1e-8)
    means = [
        (center * grid).sum((1, 2, 3)) / mass
        for grid in (z_grid, y_grid, x_grid)
    ]
    variances = [
        (center * (grid - mean[:, None, None, None]).square()).sum((1, 2, 3)) / mass
        for grid, mean in zip((z_grid, y_grid, x_grid), means)
    ]
    values.extend([mass, *means, *variances])
    names.extend([
        "center_mass", "center_centroid_z", "center_centroid_y",
        "center_centroid_x", "center_variance_z", "center_variance_y",
        "center_variance_x",
    ])
    return torch.stack(values, dim=1), names


def extract_event_features(
    patches: np.ndarray,
    model,
    *,
    device: str = "cuda",
    batch_size: int = 256,
) -> tuple[np.ndarray, list[str]]:
    """Extract per-offset features from ``(N, Z, T, Y, X)`` uint8 patches."""
    import torch

    if patches.ndim != 5 or patches.shape[2] != len(TIME_OFFSETS):
        raise ValueError(
            f"expected (N, Z, {len(TIME_OFFSETS)}, Y, X), got {patches.shape}")
    frames = patches.transpose(0, 2, 1, 3, 4).reshape(
        -1, patches.shape[1], patches.shape[3], patches.shape[4])
    result = None
    metric_names = None
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            stop = min(start + batch_size, len(frames))
            image = torch.from_numpy(frames[start:stop]).to(
                device=device, dtype=torch.float32).div_(255.0)[:, None]
            keep_axials = torch.ones(
                len(image), 4, dtype=torch.bool, device="cpu")
            probability = torch.exp(model(image, keep_axials))
            summary, names = probability_summary(probability)
            if result is None:
                result = np.empty((len(frames), len(names)), dtype=np.float32)
                metric_names = names
            result[start:stop] = summary.cpu().numpy()
    assert result is not None and metric_names is not None
    event = result.reshape(len(patches), len(TIME_OFFSETS), len(metric_names))
    names = [
        f"el_t{offset:+d}_{metric}"
        for offset in TIME_OFFSETS
        for metric in metric_names
    ]
    return event.reshape(len(patches), -1), names


def flow_summary(flow, first, second):
    """Summarize normalized XYZ flow for aligned consecutive 3D frames."""
    import torch

    batch, _, depth, height, width = flow.shape
    values = []
    names = []
    for channel, name in enumerate(("x", "y", "z")):
        data = flow[:, channel]
        values.extend([
            data.mean((1, 2, 3)), data.std((1, 2, 3)),
            data.amin((1, 2, 3)), data.amax((1, 2, 3)),
            data[:, depth // 2,
                 height // 2 - 2:height // 2 + 3,
                 width // 2 - 2:width // 2 + 3].mean((1, 2)),
        ])
        names.extend([
            f"flow_{name}_mean", f"flow_{name}_std",
            f"flow_{name}_min", f"flow_{name}_max",
            f"flow_{name}_center_mean",
        ])
    magnitude = torch.linalg.vector_norm(flow, dim=1)
    values.extend([
        magnitude.mean((1, 2, 3)), magnitude.std((1, 2, 3)),
        magnitude.amax((1, 2, 3)),
    ])
    names.extend(["flow_mag_mean", "flow_mag_std", "flow_mag_max"])

    x_flow = flow[:, 0]
    left = x_flow[:, :, :, :width // 2]
    right = x_flow[:, :, :, width // 2 + 1:]
    values.extend([
        left.mean((1, 2, 3)), right.mean((1, 2, 3)),
        right.mean((1, 2, 3)) - left.mean((1, 2, 3)),
        (x_flow[:, :, :, 1:] - x_flow[:, :, :, :-1]).mean((1, 2, 3)),
    ])
    names.extend([
        "flow_x_left_mean", "flow_x_right_mean", "flow_x_split",
        "flow_x_divergence",
    ])

    intensity = 0.5 * (first + second)
    weight = intensity / intensity.sum((1, 2, 3), keepdim=True).clamp_min(1e-8)
    for channel, name in enumerate(("x", "y", "z")):
        values.append((flow[:, channel] * weight).sum((1, 2, 3)))
        names.append(f"flow_{name}_intensity_weighted")
    values.extend([
        (first - second).abs().mean((1, 2, 3)),
        (first - second).square().mean((1, 2, 3)).sqrt(),
    ])
    names.extend(["frame_abs_difference", "frame_rms_difference"])
    return torch.stack(values, dim=1), names


def extract_event_flow_features(
    patches: np.ndarray,
    model,
    *,
    device: str = "cuda",
    batch_size: int = 256,
) -> tuple[np.ndarray, list[str]]:
    """Extract ELEPHANT flow summaries between the 11 adjacent patch frames."""
    import torch

    if patches.ndim != 5 or patches.shape[2] != len(TIME_OFFSETS):
        raise ValueError(
            f"expected (N, Z, {len(TIME_OFFSETS)}, Y, X), got {patches.shape}")
    frames = patches.transpose(0, 2, 1, 3, 4).astype(np.float32) / 255.0
    pairs = np.stack([frames[:, :-1], frames[:, 1:]], axis=2).reshape(
        -1, 2, patches.shape[1], patches.shape[3], patches.shape[4])
    result = None
    metric_names = None
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            stop = min(start + batch_size, len(pairs))
            image = torch.from_numpy(pairs[start:stop]).to(device=device)
            keep_axials = torch.ones(
                len(image), 4, dtype=torch.bool, device="cpu")
            flow = model(image, keep_axials)
            summary, names = flow_summary(flow, image[:, 0], image[:, 1])
            if result is None:
                result = np.empty((len(pairs), len(names)), dtype=np.float32)
                metric_names = names
            result[start:stop] = summary.cpu().numpy()
    assert result is not None and metric_names is not None
    transition_offsets = TIME_OFFSETS[:-1]
    event = result.reshape(len(patches), len(transition_offsets), len(metric_names))
    names = [
        f"elf_t{offset:+d}_to_t{offset + 1:+d}_{metric}"
        for offset in transition_offsets
        for metric in metric_names
    ]
    return event.reshape(len(patches), -1), names
