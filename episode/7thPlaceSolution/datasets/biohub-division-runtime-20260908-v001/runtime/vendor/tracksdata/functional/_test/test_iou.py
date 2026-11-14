import numpy as np
import pytest

from tracksdata.functional._iou import (
    fast_iou_with_bbox,
    intersection_with_bbox_2d,
    intersection_with_bbox_3d,
    intersects,
)


def test_intersects_2d_overlapping() -> None:
    """Test intersects function with overlapping 2D bounding boxes."""
    bbox1 = np.array([0, 0, 4, 4], dtype=np.int64)  # (y_min, x_min, y_max, x_max)
    bbox2 = np.array([2, 2, 6, 6], dtype=np.int64)

    assert intersects(bbox1, bbox2) is True
    assert intersects(bbox2, bbox1) is True


def test_intersects_2d_non_overlapping() -> None:
    """Test intersects function with non-overlapping 2D bounding boxes."""
    bbox1 = np.array([0, 0, 2, 2], dtype=np.int64)
    bbox2 = np.array([3, 3, 5, 5], dtype=np.int64)

    assert intersects(bbox1, bbox2) is False
    assert intersects(bbox2, bbox1) is False


def test_intersects_2d_touching() -> None:
    """Test intersects function with touching 2D bounding boxes."""
    bbox1 = np.array([0, 0, 2, 2], dtype=np.int64)
    bbox2 = np.array([2, 2, 4, 4], dtype=np.int64)

    # Touching at corner should not intersect (< not <=)
    assert intersects(bbox1, bbox2) is False
    assert intersects(bbox2, bbox1) is False


def test_intersects_2d_edge_touching() -> None:
    """Test intersects function with edge-touching 2D bounding boxes."""
    bbox1 = np.array([0, 0, 3, 2], dtype=np.int64)
    bbox2 = np.array([1, 2, 4, 4], dtype=np.int64)

    # Touching along edge should not intersect
    assert intersects(bbox1, bbox2) is False
    assert intersects(bbox2, bbox1) is False


def test_intersects_2d_contained() -> None:
    """Test intersects function with one bbox contained in another."""
    bbox1 = np.array([0, 0, 6, 6], dtype=np.int64)  # Larger box
    bbox2 = np.array([2, 2, 4, 4], dtype=np.int64)  # Smaller box inside

    assert intersects(bbox1, bbox2) is True
    assert intersects(bbox2, bbox1) is True  # Should be symmetric


def test_intersects_3d_overlapping() -> None:
    """Test intersects function with overlapping 3D bounding boxes."""
    bbox1 = np.array([0, 0, 0, 4, 4, 4], dtype=np.int64)  # (z_min, y_min, x_min, z_max, y_max, x_max)
    bbox2 = np.array([2, 2, 2, 6, 6, 6], dtype=np.int64)

    assert intersects(bbox1, bbox2) is True
    assert intersects(bbox2, bbox1) is True


def test_intersects_3d_non_overlapping() -> None:
    """Test intersects function with non-overlapping 3D bounding boxes."""
    bbox1 = np.array([0, 0, 0, 2, 2, 2], dtype=np.int64)
    bbox2 = np.array([3, 3, 3, 5, 5, 5], dtype=np.int64)

    assert intersects(bbox1, bbox2) is False
    assert intersects(bbox2, bbox1) is False


def test_intersects_3d_partial_overlap() -> None:
    """Test intersects function with partial overlap in 3D."""
    bbox1 = np.array([0, 0, 0, 3, 3, 3], dtype=np.int64)
    bbox2 = np.array([2, 0, 0, 5, 2, 2], dtype=np.int64)  # Overlap in z and y, but not full x

    assert intersects(bbox1, bbox2) is True
    assert intersects(bbox2, bbox1) is True


def test_intersection_with_bbox_2d_basic() -> None:
    """Test 2D intersection calculation with basic overlap."""
    bbox1 = np.array([0, 0, 4, 4], dtype=np.int64)
    bbox2 = np.array([2, 2, 6, 6], dtype=np.int64)

    mask1 = np.ones((4, 4), dtype=bool)  # 4x4 all True
    mask2 = np.ones((4, 4), dtype=bool)  # 4x4 all True

    intersection = intersection_with_bbox_2d(bbox1, bbox2, mask1, mask2)

    # Intersection should be 2x2 = 4 pixels
    assert intersection == 4.0


def test_intersection_with_bbox_2d_no_mask_overlap() -> None:
    """Test 2D intersection with overlapping bboxes but non-overlapping masks."""
    bbox1 = np.array([0, 0, 4, 4], dtype=np.int64)
    bbox2 = np.array([2, 2, 6, 6], dtype=np.int64)

    # Create masks that don't overlap in the intersection region
    mask1 = np.zeros((4, 4), dtype=bool)
    mask1[0:2, 0:2] = True  # Top-left corner

    mask2 = np.zeros((4, 4), dtype=bool)
    mask2[2:4, 2:4] = True  # Bottom-right corner

    intersection = intersection_with_bbox_2d(bbox1, bbox2, mask1, mask2)

    # No actual mask overlap in intersection region
    assert intersection == 0.0


def test_intersection_with_bbox_2d_partial_mask_overlap() -> None:
    """Test 2D intersection with partial mask overlap."""
    bbox1 = np.array([0, 0, 4, 4], dtype=np.int64)
    bbox2 = np.array([2, 2, 6, 6], dtype=np.int64)

    # Create masks with partial overlap
    mask1 = np.ones((4, 4), dtype=bool)

    mask2 = np.zeros((4, 4), dtype=bool)
    mask2[0:2, 0:2] = True  # Only part of the intersection region

    intersection = intersection_with_bbox_2d(bbox1, bbox2, mask1, mask2)

    # The intersection region is from (2,2) to (4,4) in global coordinates
    # In mask1 coordinates: [2:4, 2:4]
    # In mask2 coordinates: [0:2, 0:2]
    # So we're comparing mask1[2:4, 2:4] (all True) with mask2[0:2, 0:2] (all True)
    # This should give us 4 pixels of intersection
    assert intersection == 4.0


def test_intersection_with_bbox_3d_basic() -> None:
    """Test 3D intersection calculation with basic overlap."""
    bbox1 = np.array([0, 0, 0, 4, 4, 4], dtype=np.int64)
    bbox2 = np.array([2, 2, 2, 6, 6, 6], dtype=np.int64)

    mask1 = np.ones((4, 4, 4), dtype=bool)  # 4x4x4 all True
    mask2 = np.ones((4, 4, 4), dtype=bool)  # 4x4x4 all True

    intersection = intersection_with_bbox_3d(bbox1, bbox2, mask1, mask2)

    # Intersection should be 2x2x2 = 8 voxels
    assert intersection == 8.0


def test_intersection_with_bbox_3d_no_mask_overlap() -> None:
    """Test 3D intersection with overlapping bboxes but non-overlapping masks."""
    bbox1 = np.array([0, 0, 0, 4, 4, 4], dtype=np.int64)
    bbox2 = np.array([2, 2, 2, 6, 6, 6], dtype=np.int64)

    # Create masks that don't overlap in the intersection region
    mask1 = np.zeros((4, 4, 4), dtype=bool)
    mask1[0:2, 0:2, 0:2] = True  # One corner

    mask2 = np.zeros((4, 4, 4), dtype=bool)
    mask2[2:4, 2:4, 2:4] = True  # Opposite corner

    intersection = intersection_with_bbox_3d(bbox1, bbox2, mask1, mask2)

    # No actual mask overlap in intersection region
    assert intersection == 0.0


def test_fast_iou_with_bbox_2d_no_intersection() -> None:
    """Test fast IoU with non-intersecting 2D bounding boxes."""
    bbox1 = np.array([0, 0, 2, 2], dtype=np.int64)
    bbox2 = np.array([3, 3, 5, 5], dtype=np.int64)

    mask1 = np.ones((2, 2), dtype=bool)
    mask2 = np.ones((2, 2), dtype=bool)

    iou = fast_iou_with_bbox(bbox1, bbox2, mask1, mask2)

    assert iou == 0.0


def test_fast_iou_with_bbox_2d_perfect_overlap() -> None:
    """Test fast IoU with perfectly overlapping 2D masks."""
    bbox1 = np.array([0, 0, 3, 3], dtype=np.int64)
    bbox2 = np.array([0, 0, 3, 3], dtype=np.int64)

    mask1 = np.ones((3, 3), dtype=bool)
    mask2 = np.ones((3, 3), dtype=bool)

    iou = fast_iou_with_bbox(bbox1, bbox2, mask1, mask2)

    assert iou == 1.0


def test_fast_iou_with_bbox_2d_partial_overlap() -> None:
    """Test fast IoU with partial overlap in 2D."""
    bbox1 = np.array([0, 0, 4, 4], dtype=np.int64)
    bbox2 = np.array([2, 2, 6, 6], dtype=np.int64)

    mask1 = np.ones((4, 4), dtype=bool)  # 16 pixels
    mask2 = np.ones((4, 4), dtype=bool)  # 16 pixels

    iou = fast_iou_with_bbox(bbox1, bbox2, mask1, mask2)

    # Intersection: 2x2 = 4 pixels
    # Union: 16 + 16 - 4 = 28 pixels
    # IoU: 4/28 = 1/7 ≈ 0.142857
    expected_iou = 4.0 / 28.0
    assert abs(iou - expected_iou) < 1e-6


def test_fast_iou_with_bbox_3d_basic() -> None:
    """Test fast IoU with 3D masks."""
    bbox1 = np.array([0, 0, 0, 3, 3, 3], dtype=np.int64)
    bbox2 = np.array([1, 1, 1, 4, 4, 4], dtype=np.int64)

    mask1 = np.ones((3, 3, 3), dtype=bool)  # 27 voxels
    mask2 = np.ones((3, 3, 3), dtype=bool)  # 27 voxels

    iou = fast_iou_with_bbox(bbox1, bbox2, mask1, mask2)

    # Intersection: 2x2x2 = 8 voxels
    # Union: 27 + 27 - 8 = 46 voxels
    # IoU: 8/46 ≈ 0.173913
    expected_iou = 8.0 / 46.0
    assert abs(iou - expected_iou) < 1e-6


def test_fast_iou_with_bbox_zero_intersection() -> None:
    """Test fast IoU when masks don't actually overlap despite bbox intersection."""
    bbox1 = np.array([0, 0, 4, 4], dtype=np.int64)
    bbox2 = np.array([2, 2, 6, 6], dtype=np.int64)

    # Masks that don't overlap in the intersection region
    mask1 = np.zeros((4, 4), dtype=bool)
    mask1[0:2, 0:2] = True  # Top-left

    mask2 = np.zeros((4, 4), dtype=bool)
    mask2[2:4, 2:4] = True  # Bottom-right (in bbox2 coordinates)

    iou = fast_iou_with_bbox(bbox1, bbox2, mask1, mask2)

    assert iou == 0.0


def test_fast_iou_with_bbox_unsupported_dimensions() -> None:
    """Test fast IoU with unsupported mask dimensions."""
    bbox1 = np.array([0, 0, 0, 0, 2, 2, 2, 2], dtype=np.int64)  # 4D bbox
    bbox2 = np.array([1, 1, 1, 1, 3, 3, 3, 3], dtype=np.int64)

    mask1 = np.ones((2, 2, 2, 2), dtype=bool)  # 4D mask
    mask2 = np.ones((2, 2, 2, 2), dtype=bool)

    with pytest.raises(NotImplementedError, match="Masks with more than 3 dimensions are not supported"):
        fast_iou_with_bbox(bbox1, bbox2, mask1, mask2)


def test_fast_iou_with_bbox_empty_masks() -> None:
    """Test fast IoU with empty masks."""
    bbox1 = np.array([0, 0, 3, 3], dtype=np.int64)
    bbox2 = np.array([1, 1, 4, 4], dtype=np.int64)

    mask1 = np.zeros((3, 3), dtype=bool)  # All False
    mask2 = np.ones((3, 3), dtype=bool)  # All True

    iou = fast_iou_with_bbox(bbox1, bbox2, mask1, mask2)

    assert iou == 0.0


def test_fast_iou_with_bbox_identical_masks() -> None:
    """Test fast IoU with identical masks and bboxes."""
    bbox = np.array([1, 1, 4, 4], dtype=np.int64)

    # Create identical masks with some pattern
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[2, 2] = True  # Diagonal pattern

    iou = fast_iou_with_bbox(bbox, bbox, mask, mask)

    assert iou == 1.0


def test_intersects_edge_cases() -> None:
    """Test intersects function with various edge cases."""
    # Same bbox
    bbox = np.array([1, 1, 3, 3], dtype=np.int64)
    assert intersects(bbox, bbox) is True

    # Zero-area bbox - the intersects function uses < and <= comparisons
    # A zero-area bbox [1,1,1,1] can still intersect with [0,0,2,2]
    # because 0 <= 1 < 2 is True for both dimensions
    zero_bbox = np.array([1, 1, 1, 1], dtype=np.int64)
    normal_bbox = np.array([0, 0, 2, 2], dtype=np.int64)
    assert intersects(zero_bbox, normal_bbox) is True  # This is actually correct

    # Negative coordinates
    neg_bbox1 = np.array([-2, -2, 0, 0], dtype=np.int64)
    neg_bbox2 = np.array([-1, -1, 1, 1], dtype=np.int64)
    assert intersects(neg_bbox1, neg_bbox2) is True
