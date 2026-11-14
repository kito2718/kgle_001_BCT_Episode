"""Two-network D4 detector/Motion forward with one shared input and one CPU copy.

This is an adapter, not yet a deployed candidate-generation replacement.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from biohub_planar_d4_tta import PLANAR_D4_VIEWS, apply_planar_view, invert_planar_view_spatial


def shared_d4_forward(network_a, network_b, tensor):
    """Return mean detector logit and mean identity Motion on the GPU.

    Each model keeps its own 8-view logit mean. The two model means and
    identity-view Motion outputs are averaged only after inverse TTA. The
    input tensor is already normalized, padded, and resident on the GPU.
    """
    if tensor.ndim != 5:
        raise ValueError(f'Expected B,C,Z,Y,X, got {tuple(tensor.shape)}')
    sums = [None, None]
    motions = [None, None]
    for view in PLANAR_D4_VIEWS:
        transformed = apply_planar_view(tensor, view)
        for idx, model in enumerate((network_a, network_b)):
            output = model(transformed)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim != 5 or output.shape[1] < 5:
                raise ValueError(f'Bad model output {tuple(output.shape)}')
            if tuple(output.shape[-3:]) != tuple(transformed.shape[-3:]):
                output = F.interpolate(output.float(), size=transformed.shape[-3:],
                                       mode='trilinear', align_corners=False)
            logit = invert_planar_view_spatial(output[:, 1:2].float() - output[:, 0:1].float(), view)
            sums[idx] = logit if sums[idx] is None else sums[idx] + logit
            if view.k == 0 and not view.reflect_x:
                motions[idx] = output[:, 2:5].float()
    if any(m is None for m in motions):
        raise RuntimeError('Missing identity Motion output')
    return (sums[0] + sums[1]) / 16.0, (motions[0] + motions[1]) / 2.0


def cpu_outputs_once(logit, motion, original_shape, motion_scale_um):
    crop = tuple(slice(0, n) for n in original_shape)
    fused = torch.cat((torch.sigmoid(logit[(slice(None), slice(None), *crop)]),
                       motion[(slice(None), slice(None), *crop)] * motion_scale_um), dim=1)
    output = fused.cpu().numpy()
    return output[:, 0], output[:, 1:4]
