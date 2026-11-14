#!/usr/bin/env python3
"""Shared output-channel contract for Biohub detector/MotionHead checkpoints."""

from __future__ import annotations

import torch


def split_detector_motion_logits(
    output: torch.Tensor,
    contract: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor | None, str]:
    """Decode both supported Biohub detector/motion output layouts."""

    if not torch.is_tensor(output) or output.ndim != 5:
        raise ValueError(
            f"expected BxCxZxYxX network output, got "
            f"{type(output).__name__} shape={getattr(output, 'shape', None)}"
        )
    channels = int(output.shape[1])
    if contract == "auto":
        if channels == 4:
            contract = "single_logit_motion"
        elif channels >= 5:
            contract = "two_class_motion"
        elif channels == 1:
            contract = "single_logit"
        elif channels == 2:
            contract = "two_class"
        else:
            raise ValueError(f"cannot infer detector output contract from {channels} channels")

    if contract in {"single_logit", "single_logit_motion"}:
        detector_logit = output[:, :1].float()
        motion = output[:, 1:4].float() if contract == "single_logit_motion" else None
        if contract == "single_logit_motion" and channels != 4:
            raise ValueError(
                f"single_logit_motion requires exactly 4 channels, got {channels}"
            )
    elif contract in {"two_class", "two_class_motion"}:
        if channels < 2:
            raise ValueError(f"two-class detector requires at least 2 channels, got {channels}")
        detector_logit = output[:, 1:2].float() - output[:, 0:1].float()
        motion = output[:, 2:5].float() if contract == "two_class_motion" else None
        if contract == "two_class_motion" and channels < 5:
            raise ValueError(
                f"two_class_motion requires at least 5 channels, got {channels}"
            )
    else:
        raise ValueError(f"unknown detector output contract: {contract}")
    return detector_logit, motion, contract

