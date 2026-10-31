"""Share detector forward passes across banks inside one kernel run.

The composed V35 kernel needs the V25 consensus (the mean over the V21 and
V24 banks' ten checkpoints) but cannot afford ten extra forward passes per
batch. ``BankTap`` wraps each raw model: it returns the model's output
unchanged — the V21/V24 chains stay bitwise identical — while accumulating
per-batch bank sums (sigmoid heat in float32; vector heads finalized to
float16 on CPU). ``ConsensusReplay`` then serves the ten-model mean heads
from the two bank stores, so the V25 chain's ``detect`` runs with zero new
model forwards.

Numerics: the replayed consensus equals the naive ten-model ``detect`` up to
float16 storage of the vector heads (~1e-3 relative) and the
logit/sigmoid round-trip on the heat mean (float32 eps). Verified end to end
by ``scripts/verify_v35_heatshare_equivalence.py``.
"""

from __future__ import annotations

import torch


class BankStore:
    """Per-batch running sums for one detector bank."""

    def __init__(self, n_models: int):
        self.n_models = n_models
        self.prob: dict[int, torch.Tensor] = {}
        self.vec: dict[int, dict[str, torch.Tensor]] = {}
        self._hot_index: int | None = None
        self._hot_prob: torch.Tensor | None = None
        self._hot_vec: dict[str, torch.Tensor] | None = None

    def _finalize_hot(self) -> None:
        if self._hot_index is None:
            return
        self.prob[self._hot_index] = self._hot_prob
        # The offset head sets sub-voxel candidate coordinates; keep it at
        # float32 so replayed coordinates match the naive path to ~1e-6.
        # Flow/logvar/features tolerate float16 (edge-feature precision).
        self.vec[self._hot_index] = {
            key: (value if key == "offset" else value.half())
            for key, value in self._hot_vec.items()
        }
        self._hot_index = None
        self._hot_prob = None
        self._hot_vec = None

    def add(self, index: int, out: dict) -> None:
        if index != self._hot_index:
            self._finalize_hot()
            self._hot_index = index
            self._hot_prob = torch.zeros_like(
                out["heatmap"][:, 0], dtype=torch.float32, device="cpu")
            self._hot_vec = {
                key: torch.zeros(out[key].shape, dtype=torch.float32, device="cpu")
                for key in ("flow", "flow_logvar", "offset", "features")
            }
        self._hot_prob += torch.sigmoid(out["heatmap"][:, 0].float()).cpu()
        for key in ("flow", "flow_logvar", "offset", "features"):
            self._hot_vec[key] += out[key].float().cpu()

    def finish(self) -> None:
        self._finalize_hot()

    def clear(self) -> None:
        self._finalize_hot()
        self.prob.clear()
        self.vec.clear()


class BankTap:
    """Bitwise pass-through model wrapper that feeds a BankStore."""

    def __init__(self, model, store: BankStore):
        self.model = model
        self.store = store
        self.feature = model.feature
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def __call__(self, x):
        out = self.model(x)
        self.store.add(self.calls, out)
        self.calls += 1
        return out


class ConsensusReplay:
    """Ten-model mean heads served from two finished BankStores."""

    def __init__(self, store_a: BankStore, store_b: BankStore, feature):
        self.store_a = store_a
        self.store_b = store_b
        self.feature = feature
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def __call__(self, x):
        index = self.calls
        self.calls += 1
        total = float(self.store_a.n_models + self.store_b.n_models)
        prob = (self.store_a.prob[index] + self.store_b.prob[index]) / total
        prob = prob.clamp(1e-6, 1.0 - 1e-6).to(x.device)
        out = {"heatmap": torch.logit(prob)[:, None]}
        for key in ("flow", "flow_logvar", "offset", "features"):
            mean = (
                self.store_a.vec[index][key].float()
                + self.store_b.vec[index][key].float()
            ) / total
            out[key] = mean.to(x.device)
        return out


def wrap_banks(v21_models, v24_models):
    """Return (tapped_v21, tapped_v24, replay, stores) for one kernel."""
    store21 = BankStore(len(v21_models))
    store24 = BankStore(len(v24_models))
    tapped21 = [BankTap(m, store21) for m in v21_models]
    tapped24 = [BankTap(m, store24) for m in v24_models]
    replay = ConsensusReplay(store21, store24, v21_models[0].feature)
    return tapped21, tapped24, replay, (store21, store24)


def reset_banks(tapped21, tapped24, replay, stores) -> None:
    for tap in (*tapped21, *tapped24):
        tap.reset()
    replay.reset()
    for store in stores:
        store.clear()


def finish_banks(stores) -> None:
    for store in stores:
        store.finish()
