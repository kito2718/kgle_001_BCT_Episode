"""Predicting how many cells a sample holds, without ground truth.

`estimated_number_of_nodes` is the single largest non-modelling lever in this
competition — fact 9, worth 0.124 of score — and it lives in the ground-truth
GEFF. The hidden test ships `.zarr` only, so it must be predicted from
inference-available signals.

`scripts/count_predictor.py` measured that this is possible (leave-one-embryo-out
R^2 0.9575) but only ever fit-and-predicted out-of-fold on training samples; it
never persisted a model that could size an unseen submission. This package is
that missing piece.
"""

from biohub.counts.predictor import (
    CANDIDATE_OCCUPANCY_FEATURES,
    CountModel,
    FEATURE_EXCLUDE,
    features_from,
    fit_count_model,
)

__all__ = [
    "CANDIDATE_OCCUPANCY_FEATURES",
    "CountModel",
    "FEATURE_EXCLUDE",
    "features_from",
    "fit_count_model",
]
