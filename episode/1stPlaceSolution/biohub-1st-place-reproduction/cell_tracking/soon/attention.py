from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def timm_3d_rope_dim(d_head):
    """``RotaryEmbeddingCat`` size for a 3-axis head."""
    d_head = int(d_head)
    rope_dim = d_head * 2 / 3
    if rope_dim != int(rope_dim) or int(rope_dim) % 4 != 0:
        raise ValueError(
            f"timm RotaryEmbeddingCat 3D-RoPE needs d_head divisible by 6 (rope_dim = d_head*2/3 integer and % 4 == 0); got d_head={d_head}. Use feat_dim=72 nheads=3 or feat_dim=48 nheads=2."
        )
    return int(rope_dim)


def build_3d_rope_cat(grid, n_frames=3, d_head=24, theta=32.0, device="cpu", dtype=torch.float32):
    """Spatial RoPE table, tiled across frames."""
    from timm.layers import build_rotary_pos_embed

    rope_dim = timm_3d_rope_dim(d_head)
    grid = tuple((int(v) for v in grid))
    sin_emb, cos_emb = build_rotary_pos_embed(
        feat_shape=grid,
        dim=rope_dim,
        temperature=float(theta),
        in_pixels=False,
        ref_feat_shape=grid,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    spatial = torch.cat((sin_emb, cos_emb), -1)
    return spatial.repeat(int(n_frames), 1).to(device=device, dtype=dtype)


class EvaRoPESelfAttention(nn.Module):
    """Scaled dot-product attention with spatial rotary embeddings."""

    def __init__(self, feat_dim: int, nheads: int, dropout: float = 0.2, num_prefix_tokens: int = 0):
        super().__init__()
        assert feat_dim % nheads == 0, f"feat_dim {feat_dim} must be divisible by nheads {nheads}"
        self.feat_dim = feat_dim
        self.nheads = nheads
        self.d_head = feat_dim // nheads
        self.dropout_p = float(dropout)
        self.use_rope = bool(True)
        self.num_prefix_tokens = int(num_prefix_tokens)
        timm_3d_rope_dim(self.d_head)
        self.qkv_proj = nn.Linear(feat_dim, 3 * feat_dim, bias=True)
        self.out_proj = nn.Linear(feat_dim, feat_dim, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, rope: torch.Tensor = None) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, S, 3, self.nheads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        if rope is not None:
            from timm.layers import apply_rot_embed_cat

            npt = self.num_prefix_tokens
            if npt:
                q = torch.cat([q[:, :, :npt], apply_rot_embed_cat(q[:, :, npt:], rope)], dim=2)
                k = torch.cat([k[:, :, :npt], apply_rot_embed_cat(k[:, :, npt:], rope)], dim=2)
            else:
                q = apply_rot_embed_cat(q, rope)
                k = apply_rot_embed_cat(k, rope)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout_p if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(attn_out)


class EvaSpaceTimeBlock(nn.Module):
    """Attention and SwiGLU residual block with pre-normalization."""

    def __init__(self, feat_dim, nheads, dim_feedforward=None, dropout=0.2, num_prefix_tokens=0):
        super().__init__()
        self.feat_dim = feat_dim
        self.nheads = nheads
        self.use_swiglu = True
        self.layerscale_init = float(0.0)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.self_attn = EvaRoPESelfAttention(feat_dim, nheads, dropout=dropout, num_prefix_tokens=num_prefix_tokens)
        self.norm2 = nn.LayerNorm(feat_dim)
        dim_ff = dim_feedforward or feat_dim * 4
        hidden_dim = int(dim_ff * 2 / 3)
        self.w1 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, feat_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, rope=None):
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, mask=mask, rope=rope)
        x = x + self.dropout(attn_out)
        norm_x = self.norm2(x)
        ffn_out = self.w3(F.silu(self.w1(norm_x)) * self.w2(norm_x))
        x = x + self.dropout(ffn_out)
        return x


class EvaTransformerEncoder(nn.Module):
    def __init__(self, feat_dim, nheads, num_layers=4, dropout=0.2, num_prefix_tokens=0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EvaSpaceTimeBlock(feat_dim, nheads, dropout=dropout, num_prefix_tokens=num_prefix_tokens)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None, rope=None):
        for layer in self.layers:
            x = layer(x, mask=mask, rope=rope)
        return x
