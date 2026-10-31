"""A fitted, serialisable node-count predictor.

`scripts/count_predictor.py` established that the count is predictable from
inference-available features — zarr `image_statistics` quantiles plus candidate
statistics — at leave-one-embryo-out R^2 0.9575. But it only ever ran
fit-and-predict *out of fold on training samples*: there was no saved model, so
nothing could size a submission for a sample with no ground truth. Without one
the notebook has to fall back on a fixed per-frame budget, which fact 9 prices at
0.137 of score.

Two deliberate choices here:

**Features are computed from a DataFrame, not a parquet directory.** At training
time the pool is on disk; at inference it is in memory and never written. The
feature function therefore takes the candidate table itself, and
`scripts/count_predictor.py`'s definitions are reproduced exactly so a model
fitted from cached pools is valid on freshly detected ones.

**The fitted model serialises to plain JSON — scaler moments and ridge
coefficients — and predicts with numpy alone.** Pickling an sklearn pipeline
would couple the notebook to a library version we do not control on Kaggle, and
the model is a linear map on standardised features, so there is nothing to lose
by writing it out explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Kept identical to scripts/count_predictor.py.
QUANTILE_KEYS = ("0.0", "0.001", "0.01", "0.1", "0.9", "0.99", "0.999", "1.0")
NMS_RADII = (3.5, 5.0, 7.0)
FEATURE_EXCLUDE = ("dataset", "embryo", "n_total", "true_per_frame")

# Deliberately domain-reduced count profile. These features describe how many
# geometrically distinct peaks the deployed detector produced; unlike the full
# v1 profile they do not read absolute image intensities or detector-score
# calibration. The profile is fixed before the upstream robustness experiment
# in Progress/stage-05-upstream-generalization.md.
CANDIDATE_OCCUPANCY_FEATURES = (
    "nms3.5_per_frame",
    "nms5.0_per_frame",
    "nms7.0_per_frame",
    "pool_per_frame",
)


def features_from(name: str, image_quantiles: dict, cand: pd.DataFrame) -> dict:
    """Inference-available features for one sample.

    `image_quantiles` is `attrs['image_statistics']['quantiles']` from the zarr
    group; `cand` is that sample's candidate table with `t, z, y, x, score`.
    """
    from biohub.candidates.nms import select_frame

    t = cand["t"].to_numpy()
    zyx = cand[["z", "y", "x"]].to_numpy(dtype=np.float64)
    score = cand["score"].to_numpy(dtype=np.float64)
    frames = np.unique(t)

    out: dict[str, float] = {"dataset": name, "embryo": name.split("_")[0]}
    # Candidates surviving physical suppression, per frame, at a few radii. This
    # is the density signal; the pool is capped per frame so raw pool size
    # saturates on dense samples.
    strong_cut = float(np.quantile(score, 0.95)) if len(score) else 0.0
    for radius in NMS_RADII:
        counts, strong = [], []
        for f in frames:
            rows = np.flatnonzero(t == f)
            kept = select_frame(zyx[rows], score[rows], radius)
            counts.append(len(kept))
            s = score[rows][kept]
            strong.append(float((s >= strong_cut).sum()) if len(s) else 0.0)
        out[f"nms{radius}_per_frame"] = float(np.mean(counts)) if counts else 0.0
        out[f"nms{radius}_strong_per_frame"] = float(np.mean(strong)) if strong else 0.0

    out["pool_per_frame"] = len(cand) / max(1, len(frames))
    for p in (10, 50, 90, 99):
        out[f"score_p{p}"] = float(np.percentile(score, p)) if len(score) else 0.0
    for k in QUANTILE_KEYS:
        out[f"imq_{k}"] = float((image_quantiles or {}).get(k, np.nan))
    out["n_frames"] = float(len(frames))
    return out


def _design(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    """`[X, log1p(X)]`, NaN-safe — the design matrix the script fits on."""
    X = df.reindex(columns=feat_cols).to_numpy(dtype=np.float64)
    X = np.column_stack([X, np.log1p(np.clip(X, 0, None))])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class CountModel:
    """Standardise, then a linear map, then exponentiate. Predicts cells/frame."""

    feat_cols: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    alpha: float
    n_train: int
    pool: str

    def predict_per_frame(self, feats: dict | pd.DataFrame) -> np.ndarray:
        df = pd.DataFrame([feats]) if isinstance(feats, dict) else feats
        X = _design(df, self.feat_cols)
        if X.shape[1] != len(self.coef):
            raise ValueError(f"expected {len(self.coef)} design columns, got "
                             f"{X.shape[1]} — feature set has drifted")
        z = (X - self.mean) / np.where(self.scale == 0, 1.0, self.scale)
        return np.exp(z @ self.coef + self.intercept)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "feat_cols": list(self.feat_cols),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": float(self.intercept),
            "alpha": float(self.alpha),
            "n_train": int(self.n_train),
            "pool": self.pool,
            "note": "log-space ridge on [X, log1p(X)]; predict_per_frame returns "
                    "cells per frame, so multiply by n_frames for the node count",
        }, indent=1))

    @classmethod
    def load(cls, path) -> "CountModel":
        d = json.loads(Path(path).read_text())
        return cls(feat_cols=d["feat_cols"], mean=np.asarray(d["mean"]),
                   scale=np.asarray(d["scale"]), coef=np.asarray(d["coef"]),
                   intercept=d["intercept"], alpha=d["alpha"],
                   n_train=d["n_train"], pool=d.get("pool", "?"))


def fit_count_model(
    df: pd.DataFrame,
    pool: str = "?",
    feat_cols: list[str] | tuple[str, ...] | None = None,
) -> CountModel:
    """Fit on every row of `df`, which must carry `true_per_frame`.

    This is the *deployment* fit — all training samples, no held-out group. Its
    honesty is established separately by the leave-one-embryo-out run in
    `scripts/count_predictor.py`; refitting on everything afterwards is standard
    and is what the hidden test should face.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if feat_cols is None:
        feat_cols = [c for c in df.columns if c not in FEATURE_EXCLUDE]
    else:
        feat_cols = list(feat_cols)
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            raise ValueError(f"count-model features missing from table: {missing}")
        if len(set(feat_cols)) != len(feat_cols):
            raise ValueError("count-model feature list contains duplicates")
    X = _design(df, feat_cols)
    y = np.log(df["true_per_frame"].to_numpy(dtype=np.float64))

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 40)))
    model.fit(X, y)
    scaler, ridge = model[0], model[1]
    return CountModel(
        feat_cols=feat_cols, mean=scaler.mean_, scale=scaler.scale_,
        coef=ridge.coef_, intercept=float(ridge.intercept_),
        alpha=float(ridge.alpha_), n_train=len(df), pool=pool,
    )
