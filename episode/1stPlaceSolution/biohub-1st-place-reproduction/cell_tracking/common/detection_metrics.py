"""Detection / segmentation metrics."""

import numpy as np
from cell_tracking.common.io import pooled_voxel_um


def centroids_vox(labels):
    """Voxel-space centroids (pooled-voxel units) of each nonzero instance, shape (k,3), ordered by ascending label id."""
    vox = np.argwhere(labels > 0)
    if len(vox) == 0:
        return np.zeros((0, 3))
    labs = labels[vox[:, 0], vox[:, 1], vox[:, 2]]
    _, inv = np.unique(labs, return_inverse=True)
    inv = inv.reshape(-1)
    counts = np.bincount(inv)
    sums = np.stack([np.bincount(inv, weights=vox[:, k]) for k in range(3)], axis=1)
    return sums / counts[:, None]


def centroids_um(labels):
    """Physical-µm centroids of each nonzero instance, shape (k,3) ordered by label id."""
    return centroids_vox(labels) * pooled_voxel_um()


def instance_mean_prob(labels, prob_logit):
    """(ids, mean(sigmoid(logit)) per instance) -- the ranking the shipped stack prunes on."""
    n = int(labels.max()) + 1 if labels.size else 1
    flat = labels.ravel()
    cnt = np.bincount(flat, minlength=n)
    ids = np.nonzero(cnt)[0]
    ids = ids[ids > 0]
    if len(ids) == 0:
        return (ids, np.zeros(0, np.float64))
    p = 1.0 / (1.0 + np.exp(-np.asarray(prob_logit, dtype=np.float64)))
    tot = np.bincount(flat, weights=p.ravel(), minlength=n)
    return (ids, tot[ids] / cnt[ids])


def keep_top_instances(labels, prob_logit, n_keep):
    """Label volume holding only the `n_keep` most confident instances."""
    ids, mean_p = instance_mean_prob(labels, prob_logit)
    if len(ids) == 0 or n_keep >= len(ids):
        return labels
    if n_keep <= 0:
        return np.zeros_like(labels)
    keep = ids[np.argsort(-mean_p, kind="stable")[: int(n_keep)]]
    lut = np.zeros(int(labels.max()) + 1, dtype=labels.dtype)
    lut[keep] = keep
    return lut[labels]


def detection_prf(pred_um, gt_um, tol_um):
    """Greedy-optimal (Hungarian) centroid match within tol_um -> (precision, recall, f1)."""
    from scipy.optimize import linear_sum_assignment

    npred, ngt = (len(pred_um), len(gt_um))
    if ngt == 0 and npred == 0:
        return (1.0, 1.0, 1.0)
    if ngt == 0 or npred == 0:
        return (0.0, 0.0, 0.0)
    d = np.linalg.norm(pred_um[:, None, :] - gt_um[None, :, :], axis=-1)
    ri, ci = linear_sum_assignment(d)
    tp = int((d[ri, ci] <= tol_um).sum())
    prec = tp / npred
    rec = tp / ngt
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return (prec, rec, f1)


def division_prf(pred_um, gt_um, tol_um):
    """(tp, fp, fn) for division events, one-to-one matched within tol_um."""
    from scipy.optimize import linear_sum_assignment

    pred_um = np.asarray(pred_um, np.float64).reshape(-1, 3)
    gt_um = np.asarray(gt_um, np.float64).reshape(-1, 3)
    if len(pred_um) == 0 or len(gt_um) == 0:
        return (0, len(pred_um), len(gt_um))
    d = np.linalg.norm(pred_um[:, None, :] - gt_um[None, :, :], axis=-1)
    ri, ci = linear_sum_assignment(d)
    tp = int((d[ri, ci] <= tol_um).sum())
    return (tp, len(pred_um) - tp, len(gt_um) - tp)


def gt_division_centres_um(gt_lbl, observed, ttd):
    """Physical-µm centroid of every GT instance dividing AT THE CURRENT FRAME."""
    gt_lbl = np.asarray(gt_lbl)
    div_vox = np.asarray(observed, bool) & (np.asarray(ttd) == 0.0)
    ids = np.unique(gt_lbl)
    ids = ids[ids > 0]
    keep = [i for i in ids if div_vox[gt_lbl == i].any()]
    if not keep:
        return np.zeros((0, 3))
    masked = np.where(np.isin(gt_lbl, keep), gt_lbl, 0)
    return centroids_um(masked)


def division_parent_centres_um(divisions, instances):
    """Physical-µm centroid of each division event's PARENT instance -- its OWN current-frame centroid, not `children_um` (the daughters' predicted future landing points, which can sit many µm away from the parent)."""
    if not divisions:
        return np.zeros((0, 3))
    instances = np.asarray(instances)
    ids = np.unique(instances)
    ids = ids[ids > 0]
    cents = {int(i): c for i, c in zip(ids, centroids_um(instances))}
    return np.array([cents[int(d["label"])] for d in divisions], dtype=np.float64)


def endpoint_error(pred_track, track_tgt, mask, spacing=None, track_scale=None):
    """Median endpoint error in um between predicted and target displacement, over `mask`."""
    from cell_tracking.common.config import TRACK_SCALE

    ts = TRACK_SCALE if track_scale is None else float(track_scale)
    m = np.asarray(mask, bool)
    if not m.any():
        return float("nan")
    d = (np.asarray(pred_track)[:, m] - np.asarray(track_tgt)[:, m]) * ts
    return float(np.median(np.linalg.norm(d, axis=0)))


def instance_ious(pred, gt):
    """Best IoU with any predicted instance, one entry per GT instance (0 if unmatched)."""
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]
    ious = []
    for g in gt_ids:
        gm = gt == g
        overlap = pred[gm]
        cand = np.unique(overlap)
        cand = cand[cand > 0]
        best = 0.0
        for c in cand:
            pm = pred == c
            inter = int((gm & pm).sum())
            union = int((gm | pm).sum())
            best = max(best, inter / union if union else 0.0)
        ious.append(best)
    return np.asarray(ious, dtype=np.float64)


def mean_instance_iou(pred, gt):
    """Mean over GT instances of the best IoU with any predicted instance (0 if unmatched)."""
    ious = instance_ious(pred, gt)
    if not ious.size:
        return 1.0 if (pred > 0).sum() == 0 else 0.0
    return float(ious.mean())
