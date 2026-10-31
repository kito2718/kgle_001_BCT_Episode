"""Validated source-mixing contracts and shared image-patch batching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import ExternalLineageSplit


@dataclass(frozen=True)
class MixSource:
    """One source in a batch-level mixing profile."""

    name: str
    percent: int
    loss_weight: float
    quality: str
    scope: str


@dataclass(frozen=True)
class ExternalMixProfile:
    """A deterministic, validated batch source composition."""

    name: str
    batch_multiple: int
    sources: tuple[MixSource, ...]

    @classmethod
    def open(cls, path: str | Path, name: str) -> "ExternalMixProfile":
        contract = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(contract.get("schema_version", -1)) != 1:
            raise ValueError("unsupported external mix schema")
        try:
            raw = contract["profiles"][name]
        except KeyError as error:
            raise KeyError(f"unknown external mix profile {name!r}") from error
        sources = tuple(
            MixSource(
                name=str(item["name"]),
                percent=int(item["percent"]),
                loss_weight=float(item["loss_weight"]),
                quality=str(item["quality"]),
                scope=str(item["scope"]),
            )
            for item in raw["sources"]
        )
        profile = cls(
            name=name,
            batch_multiple=int(raw["batch_multiple"]),
            sources=sources,
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.batch_multiple < 1:
            raise ValueError("batch_multiple must be positive")
        names = [source.name for source in self.sources]
        if not names or len(names) != len(set(names)):
            raise ValueError("mix source names must be non-empty and unique")
        if sum(source.percent for source in self.sources) != 100:
            raise ValueError("mix percentages must sum to 100")
        for source in self.sources:
            if source.percent <= 0:
                raise ValueError(f"{source.name} has a non-positive percentage")
            if not 0.0 < source.loss_weight <= 1.0:
                raise ValueError(f"{source.name} has an invalid loss weight")
        by_name = {source.name: source for source in self.sources}
        if "zebrahub_aux" in by_name:
            pseudo = by_name["zebrahub_aux"]
            if "pseudo" not in pseudo.quality and "representation" not in pseudo.scope:
                raise ValueError("ZebraHub must remain pseudo or representation-only")
            if pseudo.loss_weight > 0.5:
                raise ValueError("ZebraHub auxiliary loss weight must not exceed 0.5")
        if "celegans_train" in by_name:
            scope = by_name["celegans_train"].scope.lower()
            if "train only" not in scope or "external_test" in scope:
                raise ValueError("C. elegans mixing must use the frozen train split only")
        if self.name == "final_calibration" and (
            len(self.sources) != 1
            or self.sources[0].name != "biohub"
            or self.sources[0].percent != 100
        ):
            raise ValueError("final calibration must be 100% Biohub")

    def batch_counts(self, batch_size: int) -> dict[str, int]:
        """Return deterministic largest-remainder counts for one batch."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        exact = np.asarray(
            [batch_size * source.percent / 100.0 for source in self.sources])
        counts = np.floor(exact).astype(np.int64)
        remainder = int(batch_size - counts.sum())
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:remainder]] += 1
        return {
            source.name: int(count)
            for source, count in zip(self.sources, counts)
        }

    def loss_weights(self) -> dict[str, float]:
        return {source.name: source.loss_weight for source in self.sources}

    def open_celegans_training_split(
        self,
        prepared_root: str | Path,
        split_contract: str | Path,
    ) -> ExternalLineageSplit:
        """Open only the frozen C. elegans training embryos for this profile."""
        if not any(source.name == "celegans_train" for source in self.sources):
            raise ValueError(f"profile {self.name!r} does not include C. elegans")
        return ExternalLineageSplit.open_celegans(
            prepared_root, split_contract, "train")


def stack_normalized_tzyx_patches(patches: list[np.ndarray]) -> np.ndarray:
    """Robust-normalize equal TZYX patches and stack them as N,T,Z,Y,X.

    Normalization is per patch, so source-specific uint8/uint16 intensity ranges
    never leak into a shared global statistic. Spatial resampling belongs in the
    source cropper; this function deliberately refuses mismatched shapes.
    """
    if not patches:
        raise ValueError("cannot stack an empty patch list")
    arrays = [np.asarray(patch) for patch in patches]
    shape = arrays[0].shape
    if len(shape) != 4 or any(array.shape != shape for array in arrays):
        raise ValueError("mixed image patches must share one TZYX shape")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("mixed image patches contain non-finite values")
    batch = np.stack(arrays).astype(np.float32)
    flat = batch.reshape(len(batch), -1)
    low = np.quantile(flat, 0.05, axis=1).reshape(-1, 1, 1, 1, 1)
    high = np.quantile(flat, 0.995, axis=1).reshape(-1, 1, 1, 1, 1)
    return np.clip((batch - low) / np.maximum(high - low, 1.0), 0, 1).astype(
        np.float32)
