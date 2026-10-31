"""Sample original backward and reversed-context forward motion at fixed nodes."""
import numpy as np
import torch
from scipy.ndimage import map_coordinates

from biohub.detector.infer import _stacks
from biohub.detector.targets import src_to_grid


def sample_field(field, source_zyx):
    """Trilinear, border-replicated sampling of physical vector fields."""
    values = np.asarray(field, dtype=np.float32)
    points = np.asarray(source_zyx, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] != 3 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('expected (3,Z,Y,X) field and (N,3) source positions')
    grid = src_to_grid(points).T
    return np.column_stack([map_coordinates(v, grid, order=1, mode='nearest',
                                           prefilter=False) for v in values]).astype(np.float32)


def sample_temporal_motion(models, vols, times, source_zyx, device, amp_dtype=None, batch=8, backward_fields=None):
    """Use each held-fold detector's temporal channels in both orders.

    Reversed backward motion already has the desired forward sign. No spatial
    reflection, sign correction, offset correction or coordinate motion occurs.
    Ensemble means retain model order, with field arithmetic in float32.
    """
    times = np.asarray(times, dtype=np.int64)
    points = np.asarray(source_zyx, dtype=np.float64)
    if not models or batch < 1 or len(times) != len(points):
        raise ValueError('nonempty models, positive batch and aligned nodes required')
    context = int(getattr(models[0], 'in_frames', 3))
    if context < 3 or context % 2 != 1 or any(int(getattr(m, 'in_frames', 3)) != context for m in models):
        raise ValueError('all models must share an odd temporal context >=3')
    if len(times) and (times.min() < 0 or times.max() >= len(vols)):
        raise ValueError('node frame outside movie')
    result = [np.empty((len(times), 3), np.float32) for _ in range(2)]
    if backward_fields is not None:
        if backward_fields.shape != (len(vols), 3, *vols.shape[1:]) or backward_fields.dtype != np.float32:
            raise ValueError('cached original field shape/dtype differs')
        for t in np.unique(times):
            selected = np.flatnonzero(times == t)
            result[0][selected] = sample_field(backward_fields[t], points[selected])
    for start in range(0, len(vols), batch):
        ts = list(range(start, min(start + batch, len(vols))))
        x = torch.from_numpy(_stacks(vols, ts, context)).to(device)
        for view in (range(2) if backward_fields is None else (1,)):
            inp = x if view == 0 else torch.flip(x, dims=(1,))
            total = None
            with torch.no_grad():
                for model in models:
                    with torch.autocast('cuda', dtype=amp_dtype,
                                        enabled=device.type == 'cuda' and amp_dtype is not None):
                        output = model(inp)
                    field = output['flow'].float().cpu().numpy()
                    total = field if total is None else total + field
                    del output
            total /= float(len(models))
            for i, t in enumerate(ts):
                selected = np.flatnonzero(times == t)
                result[view][selected] = sample_field(total[i], points[selected])
    if not all(np.isfinite(r).all() for r in result):
        raise RuntimeError('nonfinite sampled temporal motion')
    return tuple(result)
