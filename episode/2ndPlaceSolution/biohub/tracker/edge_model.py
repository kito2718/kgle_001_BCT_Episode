"""Deploy-side edge scoring: the trained GBM ensemble + group calibration.

Why this exists instead of shipping the joblib pickles: the edge model trains
locally under sklearn 1.8.0 / Python 3.14 and must predict inside Kaggle's
image, whose sklearn we do not control. A HistGradientBoosting pickle is not a
cross-version artifact. So `scripts/export_edge_model.py` dumps the trees to
plain arrays in one `.npz` and this module re-implements inference — the same
choice `CountModel` made with JSON, for the same reason. Parity with sklearn's
`predict_proba` is asserted at export time and in `tests/test_edge_model.py`
(float64 thresholds, NaN routed by `missing_go_to_left`, sigmoid link).

Two consumers:

  predict_p          mean probability over the fold models — at deployment all
                     five are clean (none trained on the hidden test), the same
                     argument as heatmap averaging (stage-02)
  rescore_argmax     PLAN §9.6 group calibration, winner-take-all: per child,
                     the argmax edge re-read from competition evidence (top-1
                     logit, top-1/top-2 margin, child context) through a
                     logistic model fitted at export on all known-truth groups;
                     every non-argmax edge drops to 0. Mirrors
                     `scripts/rescore_groups.py` — the pandas ops are copied
                     verbatim so the tie-breaks cannot diverge.

Tree traversal is vectorised over (trees x edges) and runs on torch when a
CUDA device is passed (the T4 does it in well under a second; a numpy fallback
covers CPU-only runs). Both backends share one code path via `_xp`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Child-context columns for the group calibrator, in the fitted order.
# Must match scripts/rescore_groups.py CHILD_COLS.
CHILD_COLS = ("score_c", "sigma_c", "near_c", "dens_c", "bnd_c",
              "flow_mag_um", "k_in_gate", "flow_coherence_um")

TREE_CHUNK = 32  # trees traversed per vectorised pass; bounds peak memory


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class _Fold:
    """One HistGBM fold, flattened: every tree's nodes concatenated, with
    `off[k]` the first node of tree k (so `left`/`right` are absolute)."""

    value: np.ndarray      # f8, leaf value (0 for internal nodes)
    feat: np.ndarray       # i4, split feature index
    thr: np.ndarray        # f8, split threshold (x <= thr goes left)
    miss: np.ndarray       # bool, NaN goes left
    left: np.ndarray       # i4, absolute node index
    right: np.ndarray      # i4
    leaf: np.ndarray       # bool
    off: np.ndarray        # i8, (n_trees + 1,)
    base: float            # baseline raw prediction


def flatten_sklearn_fold(model) -> dict:
    """A fitted HistGradientBoostingClassifier's trees as flat arrays.

    Training-side only (the export script and tests call it); it reads sklearn
    internals but imports nothing, so this module stays importable on Kaggle.
    The arrays are exactly `_Fold`'s fields.
    """
    values, feats, thrs, miss, left, right, leaf = [], [], [], [], [], [], []
    off = [0]
    for (pred,) in model._predictors:
        nodes = pred.nodes
        base = off[-1]
        values.append(nodes["value"].astype(np.float64))
        feats.append(nodes["feature_idx"].astype(np.int32))
        thrs.append(nodes["num_threshold"].astype(np.float64))
        miss.append(nodes["missing_go_to_left"].astype(bool))
        left.append(nodes["left"].astype(np.int64) + base)
        right.append(nodes["right"].astype(np.int64) + base)
        leaf.append(nodes["is_leaf"].astype(bool))
        off.append(base + len(nodes))
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits present; numeric only")
    return {
        "value": np.concatenate(values),
        "feat": np.concatenate(feats),
        "thr": np.concatenate(thrs),
        "miss": np.concatenate(miss),
        "left": np.concatenate(left).astype(np.int32),
        "right": np.concatenate(right).astype(np.int32),
        "leaf": np.concatenate(leaf),
        "off": np.asarray(off, dtype=np.int64),
        "base": np.float64(model._baseline_prediction.ravel()[0]),
    }


def _load_folds(z) -> list[_Fold]:
    return [
        _Fold(*(z[f"f{k}_{a}"] for a in
                ("value", "feat", "thr", "miss", "left", "right", "leaf", "off")),
              base=float(z[f"f{k}_base"]))
        for k in range(int(z["n_folds"]))
    ]


class _FoldEnsemble:
    """Shared npz GBM machinery: flat-tree traversal + mean-sigmoid ensemble.

    `EdgeModel` adds the group calibrator on top; `PairModel` is this plus a
    feature list — the division pair GBM has no per-child competition
    calibration, its threshold is swept directly on the pair score.
    """

    def __init__(self, folds: list[_Fold], features: list[str]):
        self.folds = folds
        self.features = features

    def _raw_one_fold(self, f: _Fold, X: np.ndarray, device=None) -> np.ndarray:
        """Sum of leaf values over the fold's trees, float64. `X` is (N, F)
        float32; comparisons happen in float64 exactly as sklearn's predict."""
        if device is not None:
            return self._raw_torch(f, X, device)
        return self._raw_numpy(f, X)

    def _raw_numpy(self, f: _Fold, X: np.ndarray) -> np.ndarray:
        n = len(X)
        raw = np.full(n, f.base, dtype=np.float64)
        if n == 0:
            return raw
        X64 = X.astype(np.float64)
        n_trees = len(f.off) - 1
        for c0 in range(0, n_trees, TREE_CHUNK):
            roots = f.off[c0:min(c0 + TREE_CHUNK, n_trees)].astype(np.int64)
            node = np.broadcast_to(roots[:, None], (len(roots), n)).copy()
            while True:
                is_leaf = f.leaf[node]
                if is_leaf.all():
                    break
                x = X64[np.arange(n)[None, :], f.feat[node]]
                go_left = np.where(np.isnan(x), f.miss[node], x <= f.thr[node])
                nxt = np.where(go_left, f.left[node], f.right[node])
                node = np.where(is_leaf, node, nxt)
            raw += f.value[node].sum(axis=0)
        return raw

    def _raw_torch(self, f: _Fold, X: np.ndarray, device) -> np.ndarray:
        import torch

        n = len(X)
        raw = torch.full((n,), f.base, dtype=torch.float64, device=device)
        if n == 0:
            return raw.cpu().numpy()
        X64 = torch.from_numpy(np.ascontiguousarray(X)).to(device).double()
        value = torch.from_numpy(f.value).to(device)
        feat = torch.from_numpy(f.feat.astype(np.int64)).to(device)
        thr = torch.from_numpy(f.thr).to(device)
        miss = torch.from_numpy(f.miss).to(device)
        left = torch.from_numpy(f.left.astype(np.int64)).to(device)
        right = torch.from_numpy(f.right.astype(np.int64)).to(device)
        leaf = torch.from_numpy(f.leaf).to(device)
        cols = torch.arange(n, device=device)
        n_trees = len(f.off) - 1
        for c0 in range(0, n_trees, TREE_CHUNK):
            roots = torch.from_numpy(
                f.off[c0:min(c0 + TREE_CHUNK, n_trees)].astype(np.int64)).to(device)
            node = roots[:, None].expand(-1, n).contiguous()
            while True:
                is_leaf = leaf[node]
                if bool(is_leaf.all()):
                    break
                x = X64[cols[None, :], feat[node]]
                go_left = torch.where(torch.isnan(x), miss[node], x <= thr[node])
                nxt = torch.where(go_left, left[node], right[node])
                node = torch.where(is_leaf, node, nxt)
            raw += value[node].sum(dim=0)
        return raw.cpu().numpy()

    def predict_p(self, X: np.ndarray, device=None) -> np.ndarray:
        """Mean sigmoid(raw) over folds — the deployment ensemble, float32."""
        p = np.zeros(len(X), dtype=np.float64)
        for f in self.folds:
            raw = self._raw_one_fold(f, X, device=device)
            p += 1.0 / (1.0 + np.exp(-raw))
        return (p / len(self.folds)).astype(np.float32)


class EdgeModel(_FoldEnsemble):
    def __init__(self, folds: list[_Fold], features: list[str], cal: dict):
        super().__init__(folds, features)
        self.cal = cal  # mean, scale, coef, intercept, features

    @classmethod
    def load(cls, path: Path) -> "EdgeModel":
        z = np.load(str(path))
        folds = _load_folds(z)
        cal = {a: z[f"cal_{a}"] for a in ("mean", "scale", "coef", "intercept")}
        cal_feats = [str(s) for s in z["cal_features"]]
        expect = ["logit1", "margin", *CHILD_COLS]
        if cal_feats != expect:
            raise ValueError(f"calibrator features {cal_feats} != {expect}; "
                             "the npz was exported by an incompatible version")
        model = cls(folds, [str(s) for s in z["features"]], cal)
        return model

    # ---- group calibration -------------------------------------------------

    def rescore_argmax(self, edges: pd.DataFrame, p: np.ndarray) -> np.ndarray:
        """Winner-take-all calibrated scores, one live edge per child.

        The frame construction (stable sort by (ci, -p), first = argmax,
        second's p defaulting to 0.0) is copied from scripts/rescore_groups.py
        so the two cannot disagree on tie-breaks.
        """
        e = edges[["ci", *CHILD_COLS]].copy()
        e["p"] = np.asarray(p, dtype=np.float32)
        e["row"] = np.arange(len(e))

        e = e.sort_values(["ci", "p"], ascending=[True, False], kind="stable")
        first = e.drop_duplicates("ci", keep="first").set_index("ci")
        rest = e[e.duplicated("ci", keep="first")]
        second = rest.drop_duplicates("ci", keep="first").set_index("ci")
        p2 = pd.Series(0.0, index=first.index, dtype=float)
        p2.loc[second.index] = second["p"].to_numpy()

        logit1 = _logit(first["p"].to_numpy())
        X = np.column_stack([
            logit1,
            logit1 - _logit(p2.to_numpy()),
            *(first[c].to_numpy(dtype=np.float64) for c in CHILD_COLS),
        ])
        X = np.nan_to_num(X, nan=0.0)
        z = (X - self.cal["mean"]) / self.cal["scale"]
        p_adj = 1.0 / (1.0 + np.exp(-(z @ self.cal["coef"] + self.cal["intercept"])))

        out = np.zeros(len(edges), dtype=np.float32)
        out[first["row"].to_numpy()] = p_adj.astype(np.float32)
        return out


class PairModel(_FoldEnsemble):
    """The division pair GBM (fact 50) as a deployable npz ensemble.

    No calibrator: the pair decision is a swept threshold on the mean fold
    probability (`emit_second_by_pairs`), and it doubles as the temporal
    stack's fixed prior (`base_pair_logit`).
    """

    @classmethod
    def load(cls, path: Path) -> "PairModel":
        z = np.load(str(path))
        return cls(_load_folds(z), [str(s) for s in z["features"]])


def flatten_sklearn_forest(model) -> dict:
    """A fitted ExtraTrees/RandomForest classifier as `_Fold`-shaped arrays.

    Leaf `value` is the positive-class probability (sklearn normalises tree
    values per node); internal nodes carry 0 and `base` is 0, so the shared
    traversal returns SUM over trees of leaf p — the caller divides by the
    tree count instead of applying a sigmoid. NaN routing uses the trees' own
    `missing_go_to_left` (bare forests in sklearn >= 1.4 handle NaN natively;
    this model ships without an imputer).
    """
    values, feats, thrs, miss, left, right, leaf = [], [], [], [], [], [], []
    off = [0]
    classes = list(getattr(model, "classes_", [0, 1]))
    if classes != [0, 1]:
        raise ValueError(f"expected binary classes [0, 1], got {classes}")
    for est in model.estimators_:
        tr = est.tree_
        base = off[-1]
        is_leaf = tr.children_left == -1
        val = tr.value[:, 0, :]
        val = val / np.maximum(val.sum(axis=1, keepdims=True), 1e-12)
        values.append(np.where(is_leaf, val[:, 1], 0.0).astype(np.float64))
        feats.append(np.maximum(tr.feature, 0).astype(np.int32))
        thrs.append(tr.threshold.astype(np.float64))
        mgl = getattr(tr, "missing_go_to_left", None)
        miss.append(np.asarray(mgl, dtype=bool) if mgl is not None
                    else np.zeros(tr.node_count, dtype=bool))
        left.append(np.where(is_leaf, np.arange(tr.node_count),
                             tr.children_left).astype(np.int64) + base)
        right.append(np.where(is_leaf, np.arange(tr.node_count),
                              tr.children_right).astype(np.int64) + base)
        leaf.append(is_leaf)
        off.append(base + tr.node_count)
    return {
        "value": np.concatenate(values),
        "feat": np.concatenate(feats),
        "thr": np.concatenate(thrs),
        "miss": np.concatenate(miss),
        "left": np.concatenate(left).astype(np.int32),
        "right": np.concatenate(right).astype(np.int32),
        "leaf": np.concatenate(leaf),
        "off": np.asarray(off, dtype=np.int64),
        "base": np.float64(0.0),
    }


class ForestModel(_FoldEnsemble):
    """Averaged-tree classifier folds (ExtraTrees) from a npz — the long-track
    division arm. Same flat arrays and traversal as the GBMs; the link
    function differs (mean of leaf probabilities, no sigmoid)."""

    @classmethod
    def load(cls, path: Path) -> "ForestModel":
        z = np.load(str(path))
        return cls(_load_folds(z), [str(s) for s in z["features"]])

    def predict_p(self, X: np.ndarray, device=None) -> np.ndarray:
        p = np.zeros(len(X), dtype=np.float64)
        for f in self.folds:
            n_trees = len(f.off) - 1
            p += self._raw_one_fold(f, X, device=device) / max(n_trees, 1)
        return (p / len(self.folds)).astype(np.float32)


class BoosterEnsemble:
    """LightGBM booster folds from version-portable model text.

    lightgbm's text format survives library upgrades where sklearn-API
    pickles do not; the Kaggle image ships lightgbm (and the wheels Dataset
    carries a fallback). Loaded lazily so this module imports without it.
    """

    def __init__(self, boosters: list, features: list[str]):
        self.boosters = boosters
        self.features = features

    @classmethod
    def load(cls, texts: list[str], features: list[str]) -> "BoosterEnsemble":
        import lightgbm

        return cls([lightgbm.Booster(model_str=t) for t in texts], features)

    def predict_p(self, X: np.ndarray) -> np.ndarray:
        p = np.zeros(len(X), dtype=np.float64)
        for b in self.boosters:
            p += b.predict(X)
        return (p / len(self.boosters)).astype(np.float32)


class PairStack:
    """The temporal division stack (facts 59/61), sklearn-free.

    Each fold is the exported `SimpleImputer(median) -> StandardScaler ->
    LogisticRegression` pipeline as plain arrays; deployment averages the five
    fold sigmoids, the same all-folds-are-clean argument as every other
    ensemble in the notebook. NaN handling is the imputer's: per-column
    training medians, applied before standardisation.
    """

    def __init__(self, folds: list[dict], features: list[str], base_model: str):
        self.folds = folds  # each: impute, mean, scale, coef, intercept
        self.features = features
        self.base_model = base_model

    @classmethod
    def load(cls, path: Path) -> "PairStack":
        z = np.load(str(path))
        folds = [
            {a: z[f"f{k}_{a}"] for a in
             ("impute", "mean", "scale", "coef", "intercept")}
            for k in range(int(z["n_folds"]))
        ]
        return cls(folds, [str(s) for s in z["features"]],
                   str(z["base_model"]))

    def predict_p(self, X: np.ndarray) -> np.ndarray:
        """Mean sigmoid over folds, float32. `X` is (N, F) in `features` order."""
        X64 = np.asarray(X, dtype=np.float64)
        p = np.zeros(len(X64), dtype=np.float64)
        for f in self.folds:
            Xi = np.where(np.isnan(X64), f["impute"][None, :], X64)
            zsc = (Xi - f["mean"]) / f["scale"]
            p += 1.0 / (1.0 + np.exp(-(zsc @ f["coef"] + float(f["intercept"]))))
        return (p / len(self.folds)).astype(np.float32)
