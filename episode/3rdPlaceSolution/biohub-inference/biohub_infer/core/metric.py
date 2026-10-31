"""Detector peak extraction (max-pool NMS) and the voxel size."""
import numpy as np
import torch.nn.functional as F

SCALE = np.array([1.625, 0.40625, 0.40625])  # (z, y, x) µm/voxel


def extract_peaks(hm, threshold, kernel):
    """Local maxima of a (D,H,W) heatmap tensor -> (N,3) int64 zyx tensor."""
    k = tuple(int(v) for v in kernel)
    pad = tuple(v // 2 for v in k)
    mp = F.max_pool3d(hm[None, None], kernel_size=k, stride=1, padding=pad)[0, 0]
    return ((hm == mp) & (hm > threshold)).nonzero()


                            # centre IS the index; strided yx: +0.5) — the
                            # encode side (dataset.build_coarse_targets)
                            # imports both constants from here


