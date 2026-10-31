"""Point sampling on 3-D volumes; coordinates and flow channels use (z, y, x) voxel order."""
import torch.nn.functional as F


def sample_volume(volume, points_zyx):
    """Trilinearly sample B,C,Z,Y,X at B,N,3 (or N,3 for B=1) points."""
    if points_zyx.ndim == 2:
        points_zyx = points_zyx[None]
    b, n, _ = points_zyx.shape
    if b != volume.shape[0]:
        raise ValueError("point and volume batch sizes differ")
    shape = volume.shape[-3:]
    scale = points_zyx.new_tensor([max(v - 1, 1) for v in shape])
    grid = (2.0 * points_zyx / scale - 1.0)[..., [2, 1, 0]]
    grid = grid.reshape(b, n, 1, 1, 3)
    out = F.grid_sample(volume, grid, mode="bilinear", padding_mode="border",
                        align_corners=True)
    return out[:, :, :, 0, 0].transpose(1, 2)


