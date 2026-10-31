"""LightGlue-style dual-softmax assignment with per-node matchability."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from cell_tracking.linker.support import masked_log_softmax


def dual_softmax_log_assignment(
    scores: torch.Tensor,
    logit_match_t: torch.Tensor | None = None,
    logit_match_t1: torch.Tensor | None = None,
    support_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``log P`` for an ``(..., n_t, n_t1)`` score matrix."""
    if support_mask is not None:
        support_mask = torch.as_tensor(support_mask, dtype=torch.bool, device=scores.device)
        if support_mask.shape != scores.shape:
            raise ValueError(
                f"support mask shape {tuple(support_mask.shape)} does not match scores {tuple(scores.shape)}"
            )
    log_p = masked_log_softmax(scores, support_mask, dim=-1) + masked_log_softmax(scores, support_mask, dim=-2)
    if logit_match_t is not None:
        log_p = log_p + F.logsigmoid(logit_match_t).unsqueeze(-1)
    if logit_match_t1 is not None:
        log_p = log_p + F.logsigmoid(logit_match_t1).unsqueeze(-2)
    if support_mask is not None:
        log_p = log_p.masked_fill(~support_mask.to(dtype=torch.bool, device=scores.device), float("-inf"))
    return log_p


def unbalanced_log_assignment(
    scores: torch.Tensor,
    logit_match_t: torch.Tensor | None = None,
    logit_match_t1: torch.Tensor | None = None,
    support_mask: torch.Tensor | None = None,
    *,
    n_iter: int = 0,
    eps: float = 1.0,
    rho_row: float = float("inf"),
    rho_col: float = float("inf"),
) -> torch.Tensor:
    """``log P`` with the marginals relaxed instead of the division price hand-set."""
    if n_iter <= 0:
        return dual_softmax_log_assignment(scores, logit_match_t, logit_match_t1, support_mask=support_mask)
    if support_mask is not None:
        support_mask = torch.as_tensor(support_mask, dtype=torch.bool, device=scores.device)
        if support_mask.shape != scores.shape:
            raise ValueError(
                f"support mask shape {tuple(support_mask.shape)} does not match scores {tuple(scores.shape)}"
            )
        masked_scores = scores.masked_fill(~support_mask, float("-inf"))
    else:
        masked_scores = scores

    def _masked_logsumexp(x: torch.Tensor, mask: torch.Tensor | None, dim: int):
        if mask is None:
            return torch.logsumexp(x, dim=dim)
        valid = mask.any(dim=dim)
        safe_x = torch.where(valid.unsqueeze(dim), x, torch.zeros_like(x))
        lse = torch.logsumexp(safe_x, dim=dim)
        return torch.where(valid, lse, torch.zeros_like(lse))

    tau_row = 1.0 if rho_row == float("inf") else rho_row / (rho_row + eps)
    tau_col = 1.0 if rho_col == float("inf") else rho_col / (rho_col + eps)
    f = torch.zeros_like(scores[..., :, 0])
    g = torch.zeros_like(scores[..., 0, :])
    for _ in range(int(n_iter)):
        row_lse = _masked_logsumexp((masked_scores + g.unsqueeze(-2)) / eps, support_mask, dim=-1)
        f = -tau_row * eps * row_lse
        col_lse = _masked_logsumexp((masked_scores + f.unsqueeze(-1)) / eps, support_mask, dim=-2)
        g = -tau_col * eps * col_lse
    log_p = (masked_scores + f.unsqueeze(-1) + g.unsqueeze(-2)) / eps
    if logit_match_t is not None:
        log_p = log_p + F.logsigmoid(logit_match_t).unsqueeze(-1)
    if logit_match_t1 is not None:
        log_p = log_p + F.logsigmoid(logit_match_t1).unsqueeze(-2)
    if support_mask is not None:
        log_p = log_p.masked_fill(~support_mask, float("-inf"))
    return log_p
