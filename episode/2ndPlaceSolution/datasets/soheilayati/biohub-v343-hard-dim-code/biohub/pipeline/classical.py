"""The Stage 1 classical pipeline, deployable — the public-leaderboard CONTROL.

The learned kernel scored **0.763** public. Alone that number cannot be
attributed: the only local→public calibration is one Stage 0 datapoint
(0.4344 → 0.527), which brackets the learned detector's surviving gain anywhere
between +0.05 and +0.18 over classical. This pipeline exists to collapse that
range to a measurement: same sizing, same suppression, same linking, same
confidence — the **only** variable against the learned kernel is the detector.

Candidates come from `biohub.candidates.extract.extract_dataset` with its
default `ExtractConfig`, which is byte-for-byte how `candidates_v1` was built
(`scripts/cache_candidates.py`). Everything downstream is
`pipeline.learned.size_select_link`, shared code, not a copy. Locally this
operating point scores **0.5866** over the 199 training samples
(`reports/deployable_rows.parquet`, predicted budget).

The count model must be one fitted on *classical* pool features
(`artifacts/count_model_classical.json`): the nms/score feature distributions
differ between pools, and reading v2-fitted coefficients on v1 features would
quietly mis-size every sample.
"""

from __future__ import annotations

import gc
from pathlib import Path

from biohub.pipeline.learned import TrackResult, image_quantiles, size_select_link


def track_dataset_classical(
    zarr_path: Path,
    count_model,
    drop_edgeless: bool = False,
) -> TrackResult:
    """LoG-extract, size, select and link one dataset. CPU-only by design."""
    from biohub.candidates.extract import ExtractConfig, extract_dataset

    zarr_path = Path(zarr_path)
    cand, shape = extract_dataset(zarr_path, ExtractConfig())
    result = size_select_link(zarr_path.stem, shape, cand, count_model,
                              image_quantiles(zarr_path),
                              drop_edgeless=drop_edgeless)
    gc.collect()
    return result
