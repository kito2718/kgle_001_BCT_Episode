"""Training examples for the temporal detector — PLAN.md §2.3, §9.1.

One example is a `[t-1, t, t+1]` stack on the `[1, 4, 4]` grid, with targets for
the centre frame.

**Augmentation transforms coordinates, then rebuilds the targets** — it never
transforms a target array. Flipping a heatmap is easy and flipping the flow and
offset *fields* that accompany it is where sign errors live: each vector's
components have to be permuted and negated in step with the axes. Transforming
the annotated centres and their parents instead and calling `make_targets` again
makes every vector correct by construction, and costs nothing because building a
target is microseconds.

**No z flips** (PLAN.md §2.3). The grid is isotropic at 1.625 um so a z flip is
geometrically legal, but the *appearance* is not: cells measure HWHM 5.18 um in
z against 3.80 in y/x (`reports/cell_scale.parquet`), which is the microscope's
axial PSF. y/x are interchangeable; z is not.

**Normalisation is per frame**, matching `scripts/audit_supervision_zones.py`.
That matters: `TargetConfig.background_tau = 0.02` was measured against
*per-frame* quantiles, and switching to per-video statistics would silently
invalidate it. Per-video is a planned ablation and has to carry its own tau
measurement (PLAN.md §11.2, one variable at a time).

**Order matters, and getting it wrong is silent.** The supervision zones are
decided from the *clean* normalised volume; photometric augmentation is applied
afterwards, to the model input only:

    normalise -> downsample -> geometric aug (volumes AND coordinates)
              -> DECIDE ZONES from the clean volume
              -> photometric aug -> model input

The first draft of this file built the background mask from the augmented volume.
A gamma of 1.6 with a negative background shift would then push previously-
*unknown* voxels — dim real cells among them — below tau and supervise them as
negatives, which is precisely the PLAN.md §2.6 failure this whole design exists
to avoid. It is also meaningless quantitatively: tau = 0.02 was measured against
un-augmented volumes, so a threshold applied to an augmented one is calibrated
against nothing. `tests/test_detector_dataset.py` pins the ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from biohub.detector.targets import TargetConfig, make_targets, parents_for_frame
from biohub.pipeline.baseline import (
    DOWNSAMPLE,
    GRID_UM,
    BaselineConfig,
    downsample,
    normalise,
)

# Quantiles are estimated from every Nth voxel. A frame holds 4.2 M voxels and
# an exact quantile over all of them costs more than the model step it feeds;
# a stride of 3 is ~27x cheaper and agrees to well under the 0.02 threshold.
QUANTILE_STRIDE = 3


@dataclass
class AugmentConfig:
    """Ranges are deliberately wide in intensity.

    `reports/annotation_regime_audit.md` §2: 44b6 is ~6x brighter than 6bba at
    the median, and the gap varies by quantile (2.0x at p1, 6.0x at p50, 2.1x at
    p99.9). Quantile normalisation removes the scale but not the *shape*, so
    gamma — which changes the shape — carries most of the load here.
    """

    geometric: bool = True
    brightness: tuple[float, float] = (0.75, 1.35)
    contrast: tuple[float, float] = (0.80, 1.30)
    gamma: tuple[float, float] = (0.60, 1.60)
    background_shift: tuple[float, float] = (-0.03, 0.08)
    noise_sigma: float = 0.02
    blur_sigma_um: tuple[float, float] = (0.0, 1.2)
    p_noise: float = 0.5
    p_blur: float = 0.3


@dataclass
class DatasetConfig:
    data_root: Path = Path("data/train")
    target: TargetConfig = field(default_factory=TargetConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: bool = True
    # Odd temporal context presented as channels, centred on the target frame.
    # Historical checkpoints use 3.  V38 preregisters 5 as an independent
    # temporal-context witness while leaving every other detector input fixed.
    in_frames: int = 3
    norm_lo_q: float = 0.02
    norm_hi_q: float = 0.999
    # Directory of `<name>.npy` float16 volumes already normalised and
    # downsampled by `scripts/cache_volumes.py`. Memory-mapped, so a worker
    # reads three 1 MB slices instead of normalising and mean-pooling three
    # 4.2 M-voxel source frames. Falls back to computing from the zarr when a
    # file is absent, which keeps the inference path identical to training.
    volume_cache: Path | None = None
    # `FILE:ARM` naming a label subset from `scripts/make_label_subsets.py`, e.g.
    # `data/label_subsets_v1.json:censored`. Applied ONLY when `train` is True —
    # the whole point of the ablation is to thin what the model learns from while
    # judging it against complete annotation, so a subset that leaked into
    # validation would hide exactly the effect being measured. Stored as a string
    # rather than a loaded mapping so the config stays small and picklable across
    # Windows `spawn`; each worker reads the file once, lazily.
    label_subset: str | None = None
    # Reproduce the pre-2026-08-07 augmentation seeding, which drew from
    # `hash((name, t, train))`. That has no epoch term, so within a worker
    # process an example received the IDENTICAL augmentation every epoch — and
    # because `hash()` on a str is salted per process, an example saw at most
    # `num_workers` distinct augmentations across an entire run rather than a
    # fresh one each epoch. Kept only so the original fold-0 trajectory can be
    # approximated; never use it for a real run.
    legacy_seed: bool = False


def normalise_fast(frame: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    """`baseline.normalise`, with the quantiles estimated on a strided subsample."""
    flat = frame.reshape(-1)[::QUANTILE_STRIDE]
    lo, hi = np.quantile(flat, [lo_q, hi_q])
    if hi <= lo:
        return np.zeros(frame.shape, dtype=np.float32)
    return np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0.0, None)


def flip_axis_src(coords: np.ndarray, axis: int, src_extent: int) -> np.ndarray:
    """Mirror source-voxel coordinates along one axis.

    `c -> (S - 1) - c` is exactly consistent with `np.flip` on the downsampled
    grid: with `grid = (src - 1.5) / 4`, mirroring the source gives
    `(253.5 - src) / 4`, and mirroring the grid gives `63 - (src - 1.5) / 4`,
    which is the same number. `tests/test_detector_dataset.py` pins it.
    """
    out = coords.copy()
    out[:, axis] = (src_extent - 1) - out[:, axis]
    return out


def augment_geometric(
    vols: np.ndarray,
    centres: np.ndarray,
    parents: np.ndarray,
    src_shape: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """XY flips and a y/x transpose, applied to volumes and coordinates alike.

    `vols` is (F, Z, Y, X) on the grid; coordinates are source voxels.
    """
    # Axis 1 of the volume is y, axis 2 is x; in coordinates those are 1 and 2.
    if rng.random() < 0.5:
        vols = vols[:, :, ::-1, :]
        centres = flip_axis_src(centres, 1, src_shape[1])
        parents = flip_axis_src(parents, 1, src_shape[1])
    if rng.random() < 0.5:
        vols = vols[:, :, :, ::-1]
        centres = flip_axis_src(centres, 2, src_shape[2])
        parents = flip_axis_src(parents, 2, src_shape[2])
    if rng.random() < 0.5:
        # y and x share an extent and a voxel size, so swapping them is a valid
        # rotation. Combined with the flips this generates the dihedral group.
        vols = vols.swapaxes(2, 3)
        centres = centres[:, [0, 2, 1]]
        parents = parents[:, [0, 2, 1]]
    return np.ascontiguousarray(vols), centres, parents


def augment_intensity(
    vols: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator
) -> np.ndarray:
    """Brightness, contrast, gamma, background level, noise and blur.

    Applied identically to all frames of the stack: the temporal head must see a
    consistent scene, not one whose exposure jitters between t-1 and t.
    """
    v = vols.astype(np.float32, copy=True)

    gamma = rng.uniform(*cfg.gamma)
    v = np.power(np.clip(v, 0.0, None), gamma)

    v = v * rng.uniform(*cfg.brightness)
    mean = float(v.mean())
    v = (v - mean) * rng.uniform(*cfg.contrast) + mean
    v = v + rng.uniform(*cfg.background_shift)

    if rng.random() < cfg.p_blur and cfg.blur_sigma_um[1] > 0:
        from scipy import ndimage

        s = rng.uniform(*cfg.blur_sigma_um) / float(GRID_UM[0])
        if s > 1e-3:
            v = ndimage.gaussian_filter(v, sigma=(0, s, s, s))
    if rng.random() < cfg.p_noise and cfg.noise_sigma > 0:
        v = v + rng.normal(0.0, cfg.noise_sigma, size=v.shape).astype(np.float32)

    return np.clip(v, 0.0, None)


class DetectorDataset:
    """Indexed by `(dataset, t)` over frames that carry at least one annotation.

    Frames with no annotated cell would contribute background supervision only,
    which is available in abundance from the annotated frames — so compute goes
    where the positive signal is.
    """

    def __init__(self, datasets: list[str], cfg: DatasetConfig | None = None):
        self.cfg = cfg or DatasetConfig()
        if self.cfg.in_frames < 1 or self.cfg.in_frames % 2 != 1:
            raise ValueError("in_frames must be a positive odd integer")
        self.datasets = list(datasets)
        self._gt: dict[str, object] = {}
        self._arr: dict[str, object] = {}
        self._keep: dict[str, np.ndarray] | None = None
        self.index: list[tuple[str, int]] = []
        for name in self.datasets:
            gt = self._gt_for(name)
            frames = np.unique(gt.t)
            radius = self.cfg.in_frames // 2
            # The complete context must exist. All volumes are 100 frames.
            frames = frames[(frames >= radius) & (frames <= 99 - radius)]
            self.index.extend((name, int(t)) for t in frames)

    def __len__(self) -> int:
        return len(self.index)

    def _label_keep(self, name: str) -> np.ndarray | None:
        """Retained node rows for `name`, or None when no subset is configured."""
        if not (self.cfg.train and self.cfg.label_subset):
            return None
        if self._keep is None:
            import json

            path, arm = self.cfg.label_subset.rsplit(":", 1)
            spec = json.loads(Path(path).read_text())
            if arm not in spec["arms"]:
                raise SystemExit(f"{path}: no arm {arm!r} "
                                 f"(have {sorted(spec['arms'])})")
            self._keep = {k: np.asarray(v, dtype=np.int64)
                          for k, v in spec["arms"][arm].items()}
        # A configured subset that does not mention a training sample would
        # silently train on its full labels, which is the failure this raises on.
        if name not in self._keep:
            raise SystemExit(f"label subset {self.cfg.label_subset} has no entry "
                             f"for training sample {name}")
        return self._keep[name]

    def _gt_for(self, name: str):
        if name not in self._gt:
            from biohub.metrics import load_gt
            from biohub.metrics.gt import subset_gt

            gt = load_gt(Path(self.cfg.data_root) / f"{name}.geff")
            keep = self._label_keep(name)
            self._gt[name] = gt if keep is None else subset_gt(gt, keep)
        return self._gt[name]

    def _arr_for(self, name: str):
        # Opened lazily so the dataset stays picklable: on Windows a DataLoader
        # worker is spawned, not forked, and a live zarr handle would not survive.
        if name not in self._arr:
            cached = self._cache_path(name)
            if cached is not None:
                self._arr[name] = ("cache", np.load(cached, mmap_mode="r"))
            else:
                import zarr

                self._arr[name] = ("zarr", zarr.open_array(
                    str(Path(self.cfg.data_root) / f"{name}.zarr" / "0"), mode="r"
                ))
        return self._arr[name]

    def _rng(self, i: int, name: str, t: int) -> np.random.Generator:
        """Augmentation RNG: fresh every epoch, reproducible from the run seed.

        `torch.initial_seed()` inside a DataLoader worker is re-derived by the
        loader from its generator on every epoch, so it varies epoch to epoch and
        is determined by the top-level `torch.manual_seed`. Mixing the dataset
        index in keeps two examples in one batch independent.

        The alternative it replaces — `hash((name, t, train))` — had no epoch
        term at all, so an example was augmented identically every epoch within a
        worker, and `hash()`'s per-process salt meant the whole run drew from at
        most `num_workers` variants. Augmentation that does not vary is close to
        no augmentation, which matters here because sparse positives make the
        heatmap head the first thing to overfit.
        """
        if self.cfg.legacy_seed:
            return np.random.default_rng(abs(hash((name, t, self.cfg.train))) % (2**32))
        try:
            import torch

            base = int(torch.initial_seed())
        except Exception:  # pragma: no cover - torch is always present in practice
            base = 0
        if not self.cfg.train:
            # Validation must not vary between epochs, or the held-out row would
            # move for reasons unrelated to the model.
            base = 0
        return np.random.default_rng((base + 2654435761 * (i + 1)) % (2**63))

    def _cache_path(self, name: str) -> Path | None:
        if self.cfg.volume_cache is None:
            return None
        p = Path(self.cfg.volume_cache) / f"{name}.npy"
        return p if p.exists() else None

    def __getitem__(self, i: int) -> dict:
        name, t = self.index[i]
        gt = self._gt_for(name)
        kind, arr = self._arr_for(name)
        rng = self._rng(i, name, t)

        if kind == "cache":
            # Already normalised and downsampled, in that order.
            radius = self.cfg.in_frames // 2
            vols = np.asarray(arr[t - radius : t + radius + 1], dtype=np.float32)
            src_shape = tuple(s * d for s, d in zip(arr.shape[1:], DOWNSAMPLE))
        else:
            # Exact quantiles, matching both the cache and
            # scripts/audit_supervision_zones.py. `normalise_fast` differs by at
            # most 0.0016 in value, which sounds negligible -- but the
            # background zone is a *threshold* at 0.02, and that difference
            # flips 0.02 % of background voxels. An approximation that is
            # harmless in magnitude is not automatically harmless through a
            # threshold, and once the cache exists there is nothing to buy.
            src_shape = tuple(arr.shape[1:])
            cfg = BaselineConfig(norm_lo_q=self.cfg.norm_lo_q,
                                 norm_hi_q=self.cfg.norm_hi_q)
            radius = self.cfg.in_frames // 2
            vols = np.stack([
                downsample(normalise(np.asarray(arr[t + dt]), cfg))
                for dt in range(-radius, radius + 1)
            ]).astype(np.float32)

        centres, parents = parents_for_frame(gt, t)
        if self.cfg.train and self.cfg.augment.geometric:
            vols, centres, parents = augment_geometric(
                vols, centres, parents, src_shape, rng
            )

        # Zones come from the CLEAN volume, after geometry and before photometry.
        # See the module docstring: deciding them after a photometric augment
        # supervises darkened unknown territory as background.
        tg = make_targets(
            centres,
            tuple(vols.shape[1:]),
            vol_norm=vols[self.cfg.in_frames // 2],
            parents_src=parents,
            cfg=self.cfg.target,
        )

        image = augment_intensity(vols, self.cfg.augment, rng) if self.cfg.train else vols
        return {
            "image": image,
            "heatmap": tg["heatmap"],
            "positive": tg["positive"].astype(np.float32),
            "background": tg["background"].astype(np.float32),
            "offset": tg["offset"],
            "offset_mask": tg["offset_mask"],
            "flow": tg["flow"],
            "flow_mask": tg["flow_mask"],
            "dataset": name,
            "t": t,
        }


def collate(batch: list[dict]) -> dict:
    """Stack the array fields; keep the identifiers as plain lists."""
    import torch

    keys = ("image", "heatmap", "positive", "background",
            "offset", "offset_mask", "flow", "flow_mask")
    out = {
        k: torch.from_numpy(np.stack([b[k] for b in batch])).float() for k in keys
    }
    out["dataset"] = [b["dataset"] for b in batch]
    out["t"] = [b["t"] for b in batch]
    return out


def grid_shape(src_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(s // d) for s, d in zip(src_shape, DOWNSAMPLE))
