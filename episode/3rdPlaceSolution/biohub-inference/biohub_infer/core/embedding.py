"""Trilinear sampling of a matcher embedding map at detection coordinates."""
import torch
import torch.nn.functional as F


def sample_embeddings(emb, pos, crop_shape):
    """Trilinear-sample embedding vectors at model-input-grid coords.

    emb: (C, z', y', x') map on a strided grid covering the same crop window;
    pos: (N, 3) float zyx on the SAME grid as crop_shape — the model-input
    grid, i.e. the ds grid when data.downsample is active (orig-voxel coords
    must be to_ds-converted first); crop_shape: (Z, Y, X) of that grid.
    align_corners=False + voxel-center normalization against the crop_shape
    extent maps both grids onto the same physical window."""
    if len(pos) == 0:
        return emb.new_zeros((0, emb.shape[0]))
    sz = torch.tensor(crop_shape, device=emb.device, dtype=torch.float32)
    g = (2.0 * (pos.to(emb.device).float() + 0.5) / sz - 1.0).flip(-1)  # zyx->xyz
    out = F.grid_sample(emb[None].float(), g[None, :, None, None, :],
                        align_corners=False)                # (1,C,N,1,1)
    return out[0, :, :, 0, 0].T                             # (N, C)


