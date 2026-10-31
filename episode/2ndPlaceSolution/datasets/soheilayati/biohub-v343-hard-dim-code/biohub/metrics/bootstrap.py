"""Paired group bootstrap — PLAN.md §4.5.

A fixed delta such as `+0.0025` is not a result on its own. Two systems are
compared on the *same* held-out groups, and the resampling unit is the group,
never the individual edge: edges inside one lineage and one field of view are
strongly correlated, so bootstrapping edges would understate the interval by a
large factor.

`reports/fov_overlap_audit.md` established that no two samples overlap, so the
independent group here is the sample. That is a measured fact, not an
assumption — if a later audit finds overlap, `group` must change accordingly.

Resampling is stratified by embryo. With only two embryos and 84.6 % of the
score weight in 6bba, unstratified resampling would sometimes draw a replicate
with almost no 6bba, and the resulting interval would describe a split we will
never face.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from biohub.metrics.score import score_run


@dataclass
class BootstrapResult:
    observed: float  # exact delta on the real groups, not a bootstrap statistic
    median: float
    p_gt_0: float
    p_gt_practical: float
    practical_threshold: float
    ci80: tuple[float, float]
    ci90: tuple[float, float]
    ci95: tuple[float, float]
    n_boot: int
    per_stratum: dict[str, float]
    score_a: float
    score_b: float

    def __str__(self) -> str:
        strata = "  ".join(f"{k}:{v:+.4f}" for k, v in sorted(self.per_stratum.items()))
        return (
            f"delta = {self.observed:+.4f}  (A={self.score_a:.4f} -> B={self.score_b:.4f})\n"
            f"  bootstrap median {self.median:+.4f}   "
            f"P(delta>0) = {self.p_gt_0:.3f}   "
            f"P(delta>{self.practical_threshold:g}) = {self.p_gt_practical:.3f}\n"
            f"  80% [{self.ci80[0]:+.4f}, {self.ci80[1]:+.4f}]  "
            f"90% [{self.ci90[0]:+.4f}, {self.ci90[1]:+.4f}]  "
            f"95% [{self.ci95[0]:+.4f}, {self.ci95[1]:+.4f}]\n"
            f"  by stratum: {strata}"
        )


def _aligned(a: pd.DataFrame, b: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = sorted(set(a[key]) & set(b[key]))
    if not common:
        raise ValueError("systems share no groups; a paired comparison is undefined")
    if len(common) < len(a) or len(common) < len(b):
        raise ValueError(
            f"systems cover different groups ({len(a)} vs {len(b)}, {len(common)} shared); "
            "pass matching subsets explicitly rather than silently intersecting"
        )
    a = a.set_index(key).loc[common].reset_index()
    b = b.set_index(key).loc[common].reset_index()
    return a, b


def paired_bootstrap(
    rows_a: pd.DataFrame,
    rows_b: pd.DataFrame,
    *,
    key: str = "dataset",
    stratify_by: str | None = "embryo",
    n_boot: int = 2000,
    practical_threshold: float = 0.0025,
    seed: int = 0,
) -> BootstrapResult:
    """Compare system B against system A over shared groups.

    Both frames are per-group sufficient-statistics rows from
    `biohub.metrics.score`. Each replicate resamples groups with replacement and
    recomputes the **exact** official aggregate for both systems on the same
    draw — the pairing is what removes the shared group-difficulty variance.
    """
    a, b = _aligned(rows_a, rows_b, key)
    rng = np.random.default_rng(seed)

    score_a = score_run(a)["score"]
    score_b = score_run(b)["score"]
    observed = score_b - score_a

    if stratify_by is None:
        blocks = [np.arange(len(a))]
    else:
        blocks = [
            np.flatnonzero(a[stratify_by].to_numpy() == v)
            for v in pd.unique(a[stratify_by])
        ]

    recs_a = a.to_dict("records")
    recs_b = b.to_dict("records")

    deltas = np.empty(n_boot)
    for k in range(n_boot):
        pick = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in blocks])
        deltas[k] = (
            score_run([recs_b[i] for i in pick])["score"]
            - score_run([recs_a[i] for i in pick])["score"]
        )

    per_stratum: dict[str, float] = {}
    if stratify_by is not None:
        for v in pd.unique(a[stratify_by]):
            m = a[stratify_by].to_numpy() == v
            per_stratum[str(v)] = (
                score_run(b[m])["score"] - score_run(a[m])["score"]
            )

    q = lambda lo, hi: (float(np.quantile(deltas, lo)), float(np.quantile(deltas, hi)))  # noqa: E731
    return BootstrapResult(
        observed=float(observed),
        median=float(np.median(deltas)),
        p_gt_0=float((deltas > 0).mean()),
        p_gt_practical=float((deltas > practical_threshold).mean()),
        practical_threshold=practical_threshold,
        ci80=q(0.10, 0.90),
        ci90=q(0.05, 0.95),
        ci95=q(0.025, 0.975),
        n_boot=n_boot,
        per_stratum=per_stratum,
        score_a=float(score_a),
        score_b=float(score_b),
    )
