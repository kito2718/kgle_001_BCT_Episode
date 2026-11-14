import numpy as np
from numba import njit, types


@njit
def intersects(bbox1: types.int64[:], bbox2: types.int64[:]) -> bool:
    """Checks if bounding box intersects by checking if their coordinates are within the
    range of the other along each axis.

    Parameters
    ----------
    bbox1 : ArrayLike
        Bounding box as (min_0, min_1, ..., max_0, max_1, ...).
    bbox2 : ArrayLike
        Bounding box as (min_0, min_1, ..., max_0, max_1, ...).
    Returns
    -------
    bool
        Boolean indicating intersection.
    """
    n_dim = len(bbox1) // 2
    intersects = True
    for i in range(n_dim):
        k = i + n_dim
        intersects &= (
            bbox1[i] <= bbox2[i] < bbox1[k]
            or bbox1[i] < bbox2[k] <= bbox1[k]
            or bbox2[i] <= bbox1[i] < bbox2[k]
            or bbox2[i] < bbox1[k] <= bbox2[k]
        )
        if not intersects:
            break

    return intersects


@njit
def intersection_with_bbox_2d(bbox1: np.ndarray, bbox2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> float:
    y_min = max(bbox1[0], bbox2[0])
    x_min = max(bbox1[1], bbox2[1])
    y_max = min(bbox1[2], bbox2[2])
    x_max = min(bbox1[3], bbox2[3])

    aligned_mask1 = mask1[
        y_min - bbox1[0] : mask1.shape[0] + y_max - bbox1[2],
        x_min - bbox1[1] : mask1.shape[1] + x_max - bbox1[3],
    ]

    aligned_mask2 = mask2[
        y_min - bbox2[0] : mask2.shape[0] + y_max - bbox2[2],
        x_min - bbox2[1] : mask2.shape[1] + x_max - bbox2[3],
    ]

    return np.logical_and(aligned_mask1, aligned_mask2).sum()


@njit
def intersection_with_bbox_3d(bbox1: np.ndarray, bbox2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> float:
    z_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_min = max(bbox1[2], bbox2[2])
    z_max = min(bbox1[3], bbox2[3])
    y_max = min(bbox1[4], bbox2[4])
    x_max = min(bbox1[5], bbox2[5])

    aligned_mask1 = mask1[
        z_min - bbox1[0] : mask1.shape[0] + z_max - bbox1[3],
        y_min - bbox1[1] : mask1.shape[1] + y_max - bbox1[4],
        x_min - bbox1[2] : mask1.shape[2] + x_max - bbox1[5],
    ]

    aligned_mask2 = mask2[
        z_min - bbox2[0] : mask2.shape[0] + z_max - bbox2[3],
        y_min - bbox2[1] : mask2.shape[1] + y_max - bbox2[4],
        x_min - bbox2[2] : mask2.shape[2] + x_max - bbox2[5],
    ]

    return np.logical_and(aligned_mask1, aligned_mask2).sum()


@njit
def fast_intersection_with_bbox(bbox1: np.ndarray, bbox2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> float:
    if not intersects(bbox1, bbox2):
        return 0.0
    if mask1.ndim == 2:
        inter = intersection_with_bbox_2d(bbox1, bbox2, mask1, mask2)
    elif mask1.ndim == 3:
        inter = intersection_with_bbox_3d(bbox1, bbox2, mask1, mask2)
    else:
        raise NotImplementedError("Masks with more than 3 dimensions are not supported")
    return inter


@njit
def fast_iou_with_bbox(bbox1: np.ndarray, bbox2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = fast_intersection_with_bbox(bbox1, bbox2, mask1, mask2)
    if inter == 0.0:
        return 0.0
    union = mask1.sum() + mask2.sum() - inter
    return (inter / union).item()
