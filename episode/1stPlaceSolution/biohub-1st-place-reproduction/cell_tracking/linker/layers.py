"""Transformer operating on nodes to predict edges between nodes."""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt


class CrossAttentionBlock(nn.Module):
    """A single cross-attention block with MLP and residual connections."""

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=dropout)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attention with residual."""
        key_padding_mask = None
        merged = None
        if kv_mask is not None:
            kv_mask = kv_mask.to(dtype=torch.bool)
            sample_empty = ~kv_mask.any(dim=-1, keepdim=True)
            kv_mask = kv_mask | sample_empty
        if attn_mask is not None:
            allowed = attn_mask.to(dtype=torch.bool)
            if kv_mask is not None:
                allowed = allowed & kv_mask.unsqueeze(1)
            empty = ~allowed.any(dim=-1, keepdim=True)
            fallback = kv_mask.unsqueeze(1) if kv_mask is not None else torch.ones_like(allowed)
            allowed = torch.where(empty, fallback, allowed)
            h = self.cross_attn.num_heads
            merged = (~allowed).repeat_interleave(h, dim=0)
        elif kv_mask is not None:
            key_padding_mask = ~kv_mask
        attn_out, _ = self.cross_attn(
            self.norm1(q), self.norm1(kv), self.norm1(kv), key_padding_mask=key_padding_mask, attn_mask=merged
        )
        q = q + attn_out
        q = q + self.mlp(self.norm2(q))
        return q


def _attend(
    block: CrossAttentionBlock,
    x: torch.Tensor,
    kv: torch.Tensor,
    mask: torch.Tensor | None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One attention block, grad-checkpointed while training."""
    if torch.is_grad_enabled():
        return grad_ckpt(
            lambda a, b, m, am, _b=block: _b(a, b, kv_mask=m, attn_mask=am), x, kv, mask, attn_mask, use_reentrant=False
        )
    return block(x, kv, kv_mask=mask, attn_mask=attn_mask)


class SimpleNodeTransformer(nn.Module):
    """Transformer for predicting edges between cell detections."""

    def __init__(
        self,
        feat_dim: int = 33,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.3,
        pair_chunk_size: int | None = 32,
        self_attention: bool = False,
    ):
        super().__init__()
        self.pair_chunk_size = pair_chunk_size
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout) for _ in range(n_blocks)]
        )
        self.self_attention = bool(self_attention)
        self.self_blocks = (
            nn.ModuleList([CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout) for _ in range(n_blocks)])
            if self.self_attention
            else None
        )
        self.symmetric_update = False
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_frames(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
        *,
        attn_t: torch.Tensor | None = None,
        attn_t1: torch.Tensor | None = None,
        self_attn_t: torch.Tensor | None = None,
        self_attn_t1: torch.Tensor | None = None,
        extra_t: torch.Tensor | None = None,
        extra_t1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(q, k)`` node embeddings after the attention stack."""
        unbatched = feat_t.ndim == 2
        if unbatched:
            feat_t, feat_t1 = (feat_t.unsqueeze(0), feat_t1.unsqueeze(0))
            if extra_t is not None:
                extra_t, extra_t1 = (extra_t.unsqueeze(0), extra_t1.unsqueeze(0))

            def _b(m):
                return None if m is None else m.unsqueeze(0) if m.ndim == 2 else m

            attn_t, attn_t1 = (_b(attn_t), _b(attn_t1))
            self_attn_t, self_attn_t1 = (_b(self_attn_t), _b(self_attn_t1))
        q = self.proj(feat_t)
        k = self.proj(feat_t1)
        if extra_t is not None:
            q = q + extra_t
            k = k + extra_t1
        q = self.norm_in(q)
        k = self.norm_in(k)
        for i, block in enumerate(self.blocks):
            if self.self_blocks is not None:
                q = _attend(self.self_blocks[i], q, q, mask_t, self_attn_t)
                k = _attend(self.self_blocks[i], k, k, mask_t1, self_attn_t1)
            q_new = _attend(block, q, k, mask_t1, attn_t)
            k = _attend(block, k, q if self.symmetric_update else q_new, mask_t, attn_t1)
            q = q_new
        q, k = (self.norm_out(q), self.norm_out(k))
        return (q.squeeze(0), k.squeeze(0)) if unbatched else (q, k)

    def forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict edge logits between detections at consecutive frames."""
        unbatched = feat_t.ndim == 2
        if unbatched:
            feat_t = feat_t.unsqueeze(0)
            feat_t1 = feat_t1.unsqueeze(0)
            coords_t = coords_t.unsqueeze(0)
            coords_t1 = coords_t1.unsqueeze(0)
        q, k = self.encode_frames(feat_t, feat_t1, mask_t, mask_t1)
        N_t = q.shape[1]
        chunk = self.pair_chunk_size or N_t
        chunks = []
        pair_mlp = self.pair_mlp
        for i in range(0, N_t, chunk):
            q_c = q[:, i : i + chunk, :]
            coords_c = coords_t[:, i : i + chunk, :]

            def _chunk_fn(
                qc: torch.Tensor, kk: torch.Tensor, cc: torch.Tensor, cc1: torch.Tensor, _pm: nn.Module = pair_mlp
            ) -> torch.Tensor:
                nc_i = qc.shape[1]
                n1 = kk.shape[1]
                qe = qc.unsqueeze(2).expand(-1, -1, n1, -1)
                ke = kk.unsqueeze(1).expand(-1, nc_i, -1, -1)
                rel = (cc.unsqueeze(2) - cc1.unsqueeze(1)) / 100.0
                return _pm(torch.cat([qe, ke, rel], dim=-1)).squeeze(-1)

            if torch.is_grad_enabled():
                out = grad_ckpt(_chunk_fn, q_c, k, coords_c, coords_t1, use_reentrant=False)
            else:
                out = _chunk_fn(q_c, k, coords_c, coords_t1)
            chunks.append(out)
        logits = torch.cat(chunks, dim=1)
        if unbatched:
            logits = logits.squeeze(0)
        return logits
