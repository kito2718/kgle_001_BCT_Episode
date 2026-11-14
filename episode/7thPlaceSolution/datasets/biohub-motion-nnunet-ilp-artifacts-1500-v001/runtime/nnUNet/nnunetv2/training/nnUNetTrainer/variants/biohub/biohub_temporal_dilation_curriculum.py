"""Shared-epoch temporal dilation curriculum for Biohub data-loader workers."""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import random
from dataclasses import dataclass
from typing import Protocol, Tuple


@dataclass(frozen=True)
class DilationProbabilities:
    d1: float
    d2: float
    d3: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return self.d1, self.d2, self.d3


class RandomSource(Protocol):
    def random(self) -> float:
        ...


def curriculum_probabilities(epoch: int) -> DilationProbabilities:
    """Return the preregistered d1/d2/d3 distribution for a zero-based epoch."""
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}")
    if epoch < 100:
        return DilationProbabilities(1.0, 0.0, 0.0)
    if epoch < 700:
        return DilationProbabilities(0.5, 0.5, 0.0)
    return DilationProbabilities(0.5, 0.25, 0.25)


class SharedEpochDilationSampler:
    """Sample d1/d2/d3 while reading an epoch shared by forked workers.

    ``RawValue`` avoids filesystem reads and lock acquisition in the per-sample
    hot path. The main process must only call ``set_epoch`` at an epoch
    boundary. A signed 32-bit aligned value is sufficient for training epochs.
    """

    def __init__(self, initial_epoch: int = 0) -> None:
        if initial_epoch < 0:
            raise ValueError(
                f"initial_epoch must be non-negative, got {initial_epoch}"
            )
        self._epoch = mp.RawValue(ctypes.c_int32, int(initial_epoch))

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self._epoch.value = int(epoch)

    def get_epoch(self) -> int:
        return int(self._epoch.value)

    def probabilities(self) -> DilationProbabilities:
        return curriculum_probabilities(self.get_epoch())

    def sample(self, rng: RandomSource | None = None) -> int:
        random_value = random.random() if rng is None else float(rng.random())
        probabilities = self.probabilities()
        if random_value < probabilities.d1:
            return 1
        if random_value < probabilities.d1 + probabilities.d2:
            return 2
        return 3
