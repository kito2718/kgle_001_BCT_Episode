import numpy as np
SCALE=np.array([1.625,.40625,.40625])
def features(pair, source, targets, parent_index, builder, stem, t, score):
    names = sorted((k for k in pair if k.startswith(('geom_', 'traj_'))))
    f = {k: pair[k] for k in names}
    shift, _ = builder.cumulative_shift(stem, t, t + 1, 3)
    other = np.delete(source, parent_index, axis=0) + shift
    children = targets[[pair['first'], pair['second']]]
    alt = np.linalg.norm(other[:, None, :] - children[None, :, :], axis=2).min(axis=0) if len(other) else np.full(2, 30.0)
    own = np.linalg.norm(children - (source[parent_index] + shift), axis=1)
    continuation = min(own[0] + min(alt[1], 15.0), own[1] + min(alt[0], 15.0))
    f.update(alt_origin_min=float(min(alt)), alt_origin_max=float(max(alt)), alt_margin_min=float(min(alt - own)), alt_margin_max=float(max(alt - own)), alt_continuation_cost=float(continuation), alt_split_cost=float(sum(own)), alt_split_minus_continue=float(sum(own) - continuation), parent_detector_score=float(score))
    return f
