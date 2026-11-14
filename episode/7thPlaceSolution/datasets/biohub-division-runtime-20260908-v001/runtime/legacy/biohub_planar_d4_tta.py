#!/usr/bin/env python3
"""Planar D4 transforms for Biohub scalar heatmaps and ZYX motion fields."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PlanarD4View:
    k: int
    reflect_x: bool

    @property
    def name(self) -> str:
        return f"r{90 * self.k}{'_fx' if self.reflect_x else ''}"


PLANAR_D4_VIEWS = tuple(
    PlanarD4View(k=k, reflect_x=reflect_x)
    for reflect_x in (False, True)
    for k in range(4)
)


def apply_planar_view(tensor: torch.Tensor, view: PlanarD4View) -> torch.Tensor:
    result = torch.flip(tensor, dims=(-1,)) if view.reflect_x else tensor
    return torch.rot90(result, k=view.k, dims=(-2, -1))


def invert_planar_view_spatial(tensor: torch.Tensor, view: PlanarD4View) -> torch.Tensor:
    result = torch.rot90(tensor, k=-view.k, dims=(-2, -1))
    return torch.flip(result, dims=(-1,)) if view.reflect_x else result


def planar_vector_matrix(
    view: PlanarD4View,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return A such that ``(dy_view, dx_view) = A @ (dy, dx)``."""
    rotations = (
        ((1.0, 0.0), (0.0, 1.0)),
        ((0.0, -1.0), (1.0, 0.0)),
        ((-1.0, 0.0), (0.0, -1.0)),
        ((0.0, 1.0), (-1.0, 0.0)),
    )
    rotation = torch.tensor(rotations[view.k], device=device, dtype=dtype)
    reflection = torch.tensor(
        ((1.0, 0.0), (0.0, -1.0 if view.reflect_x else 1.0)),
        device=device,
        dtype=dtype,
    )
    return rotation @ reflection


def transform_motion_components(
    motion_zyx: torch.Tensor,
    view: PlanarD4View,
    *,
    inverse: bool,
    component_dim: int,
) -> torch.Tensor:
    dim = component_dim if component_dim >= 0 else motion_zyx.ndim + component_dim
    if motion_zyx.shape[dim] != 3:
        raise ValueError(
            f"expected three ZYX components at dim {component_dim}, "
            f"got shape={tuple(motion_zyx.shape)}"
        )
    matrix = planar_vector_matrix(view, device=motion_zyx.device, dtype=motion_zyx.dtype)
    if inverse:
        matrix = matrix.transpose(0, 1)
    moved = motion_zyx.movedim(dim, 0)
    transformed_yx = torch.einsum("ij,j...->i...", matrix, moved[1:3])
    result = moved.clone()
    result[1:3] = transformed_yx
    return result.movedim(0, dim)


def invert_motion_view(
    motion_zyx: torch.Tensor,
    view: PlanarD4View,
    *,
    component_dim: int,
) -> torch.Tensor:
    spatial = invert_planar_view_spatial(motion_zyx, view)
    return transform_motion_components(
        spatial, view, inverse=True, component_dim=component_dim
    )


def detector_logit_mean_and_identity_motion(
    network: torch.nn.Module,
    tensor: torch.Tensor,
    views: tuple[PlanarD4View, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse detector logits across views while retaining identity-view motion.

    The input must already be padded to the network stride. Returned tensors
    use the untransformed padded spatial shape. The caller remains responsible
    for cropping back to the original image shape and scaling motion units.
    """

    if tensor.ndim != 5:
        raise ValueError(f"expected BxCxZxYxX tensor, got {tuple(tensor.shape)}")
    if not views:
        raise ValueError("at least one planar view is required")
    detector_logit_sum = None
    identity_motion = None
    for view in views:
        view_tensor = apply_planar_view(tensor, view)
        output = network(view_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output) or output.ndim != 5 or output.shape[1] < 5:
            raise RuntimeError(
                "expected Bx5xZxYxX network output, "
                f"got {type(output)} shape={getattr(output, 'shape', None)}"
            )
        if tuple(output.shape[-3:]) != tuple(view_tensor.shape[-3:]):
            output = F.interpolate(
                output.float(),
                size=view_tensor.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        detector_logit = output[:, 1:2].float() - output[:, 0:1].float()
        detector_logit = invert_planar_view_spatial(detector_logit, view)
        detector_logit_sum = (
            detector_logit
            if detector_logit_sum is None
            else detector_logit_sum + detector_logit
        )
        if view.k == 0 and not view.reflect_x:
            identity_motion = output[:, 2:5].float()
    if identity_motion is None:
        raise RuntimeError("views must include the identity transform for motion")
    return detector_logit_sum / float(len(views)), identity_motion
