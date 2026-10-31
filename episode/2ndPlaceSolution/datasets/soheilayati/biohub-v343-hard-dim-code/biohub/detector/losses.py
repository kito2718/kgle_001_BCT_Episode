"""Detector losses — PLAN.md §9.1.1.

    L = 1.0 * heatmap + 0.5 * flow + 0.2 * offset + 0.05 * uncertainty

kept as four separable terms because the plan calls those weights "starting
points, not fixed truths", and separable terms are what make an ablation cheap.

**The heatmap term balances by zone, not by voxel.** A frame carries roughly 157
positive voxels against 51,000 reliable-background ones
(`scripts/audit_supervision_zones.py`), so a plain masked mean would be 325:1
background and the model would learn to output zero everywhere. Each zone is
averaged within itself and then combined.

**Unknown voxels contribute exactly zero.** Not a small weight -- zero. With
44b6 annotating 0.77 % of its cells, a `neg_weight=1e-2` over unknown territory
(what the official baseline does) trains ~99 % of real cells toward background.
`tests/test_detector_targets.py` asserts the gradient is literally unaffected by
what the model predicts there.

**The uncertainty head is trained against a detached error.** It learns to
predict how wrong the flow head is without being able to reduce its own loss by
making the flow head's gradient smaller, which is the usual way an
uncertainty-weighted regression quietly stops fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# Measured magnitude of each raw term, from 200 steps on real fold-0 data with
# every weight set to 1.0. Each term is divided by its own scale before the
# weights below apply, so `LossWeights` means what PLAN.md §9.1.1 says it means:
# a weight of 1.0 contributes 1.0.
#
# Without this the plan's weights are badly misleading here, because the terms
# are not in comparable units -- the heatmap is a squared error in [0, 1] while
# flow and offset are smooth-L1 in micrometres. Measured raw:
#
#     heatmap 0.0393   flow 1.1415   offset 0.7271   uncertainty 1.7214
#
# so `heatmap 1.0, flow 0.5` gave the heatmap 1/26th of the gradient share it
# was supposed to have -- on the term that the error taxonomy prices at +0.164,
# against a flow head whose value is a Stage 4 ceiling.
#
# Fixed constants, deliberately: an adaptive normaliser would be a moving target
# and would make two runs incomparable.
TERM_SCALE = {
    "heatmap": 0.04,
    "flow": 1.14,
    "offset": 0.73,
    "uncertainty": 1.72,
}


@dataclass(frozen=True)
class LossWeights:
    """Contributions, after each term is divided by its `TERM_SCALE`."""

    heatmap: float = 1.0
    flow: float = 0.5
    offset: float = 0.2
    uncertainty: float = 0.05
    # Relative weight of the reliable-background zone against the positive zone,
    # after each is averaged within itself.
    background: float = 1.0


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    total = mask.sum()
    if total == 0:
        return x.sum() * 0.0
    return (x * mask).sum() / total


def heatmap_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    positive: torch.Tensor,
    background: torch.Tensor,
    w: LossWeights,
) -> torch.Tensor:
    """Zone-balanced squared error on the Gaussian target.

    `pred` is a logit; the Gaussian lives in [0, 1], so it is squashed here
    rather than in the model, keeping the model's output a single unbounded
    score that peak extraction can rank directly.
    """
    p = torch.sigmoid(pred)
    err = (p - target) ** 2
    return _masked_mean(err, positive) + w.background * _masked_mean(err, background)


def vector_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Smooth L1 over a (B, 3, Z, Y, X) field, in micrometres.

    `beta = 1.0` is one micrometre, which is where the measured error
    distribution turns over: matched candidates sit at a median of 2.19 um, so
    the quadratic region covers the good cases and the linear region keeps a
    12 um outlier from dominating the batch.
    """
    err = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0).sum(dim=1)
    return _masked_mean(err, mask)


def uncertainty_loss(
    logvar: torch.Tensor,
    flow_pred: torch.Tensor,
    flow_target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL for the flow head's own error, with the error detached."""
    sq = ((flow_pred - flow_target) ** 2).sum(dim=1, keepdim=True).detach()
    lv = logvar.clamp(-8.0, 8.0)
    nll = 0.5 * (torch.exp(-lv) * sq + lv)
    return _masked_mean(nll.squeeze(1), mask)


def detector_loss(
    out: dict[str, torch.Tensor],
    tgt: dict[str, torch.Tensor],
    w: LossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total loss and its parts. `out` is the model's dict, `tgt` the targets."""
    w = w or LossWeights()
    positive = tgt["positive"]
    background = tgt["background"]

    h = heatmap_loss(out["heatmap"].squeeze(1), tgt["heatmap"], positive, background, w)
    f = vector_loss(out["flow"], tgt["flow"], tgt["flow_mask"])
    o = vector_loss(out["offset"], tgt["offset"], tgt["offset_mask"])
    u = uncertainty_loss(out["flow_logvar"], out["flow"], tgt["flow"], tgt["flow_mask"])

    # Divide by the measured scale so the weights carry the plan's meaning.
    hs, fs, os_, us = (h / TERM_SCALE["heatmap"], f / TERM_SCALE["flow"],
                       o / TERM_SCALE["offset"], u / TERM_SCALE["uncertainty"])
    total = w.heatmap * hs + w.flow * fs + w.offset * os_ + w.uncertainty * us
    return total, {
        "heatmap": float(h.detach()),
        "flow": float(f.detach()),
        "offset": float(o.detach()),
        "uncertainty": float(u.detach()),
        # Raw terms above, in their own units, so a log stays comparable with
        # the measured scales. `total`/`monitor` are the normalised objective.
        "total": float(total.detach()),
        # `total` is the training objective but a poor progress signal: a
        # Gaussian NLL is unbounded below, so once the flow head is accurate the
        # uncertainty term goes negative and dominates the sign. Measured on the
        # overfit test, `total` runs 0.98 -> -0.09, which reads as an enormous
        # improvement no matter what the other three terms did. `monitor` is the
        # sum of the non-negative terms and is what gates and plots should use.
        "monitor": float((w.heatmap * hs + w.flow * fs + w.offset * os_).detach()),
    }
