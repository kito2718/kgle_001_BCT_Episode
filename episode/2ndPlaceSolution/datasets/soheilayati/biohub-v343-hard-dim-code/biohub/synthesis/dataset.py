"""Training-side view of a synthetic variant — composed frames + merged GT.

`SynthVariantData` is what a detector-training run consumes: for a host
sample and one variant it serves

    frame(t)      cache frame + this variant's ADD-patches (float32), and
    merged GT     host nodes/edges plus the synthetic nodes/edges, as a
                  GroundTruth-shaped object `parents_for_frame` accepts.

The decisive property (pinned by test): an injected second daughter's
parent resolves to the HOST mother, so `make_targets` writes the large
backward birth vector into the flow supervision — the exact signal the
frozen flow head lacks (points at the thief 53/53). No trainer surgery:
the consumer run swaps datasets, not heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biohub.metrics.gt import GroundTruth, load_gt
from biohub.synthesis.generate import compose_frame, load_variant


@dataclass
class MergedGT:
    """GroundTruth-shaped view over host + synthetic annotations."""

    t: np.ndarray
    zyx: np.ndarray
    edges: np.ndarray
    n_host: int

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_nodes(self) -> int:
        return len(self.t)


def merge_gt(host: GroundTruth, syn: dict) -> MergedGT:
    """Host arrays + synthetic nodes; synthetic parents become edges.

    `syn["parent"]` holds merged indices (host rows < n_host, earlier
    synthetic nodes >= n_host) or -1 for chain heads — exactly the layout
    `Variant.add_node` wrote.
    """
    n_host = int(host.n_nodes)
    t = np.concatenate([host.t, syn["t"].astype(np.int64)])
    zyx = np.concatenate([host.zyx, syn["zyx"]])
    syn_edges = [(int(p), n_host + j) for j, p in enumerate(syn["parent"])
                 if int(p) >= 0]
    edges = (np.concatenate([host.edges,
                             np.asarray(syn_edges, dtype=np.int64)
                             .reshape(-1, 2)])
             if syn_edges else host.edges)
    return MergedGT(t=t, zyx=zyx, edges=edges, n_host=n_host)


def load_variant_nodes(npz_path: Path) -> dict:
    """Only the synthetic node arrays — index building must not hold the
    pixel patches of hundreds of variants in the main process."""
    z = np.load(npz_path)
    return {"t": z["syn_t"], "zyx": z["syn_zyx"], "parent": z["syn_parent"]}


class ComposedVolume:
    """Cache-shaped view whose frames carry the variant's ADD-patches.

    Quacks exactly like the mmap'd cache array `DetectorDataset` slices
    (`arr[t-1:t+2]`, `.shape`), so the parent `__getitem__` — augmentation,
    zone ordering, targets — runs UNCHANGED on synthetic variants.
    """

    def __init__(self, cache_path: Path, variant_npz: Path):
        from biohub.synthesis.generate import load_variant

        self._cache = np.load(cache_path, mmap_mode="r")
        self._per_t, _ = load_variant(variant_npz)
        # Tracker morphology asks for the same composed frame many times via
        # advanced indexing. A movie is only ~105 MB as float32, so lazy
        # per-frame memoisation removes repeated patch composition while
        # keeping detector-training slice semantics unchanged.
        self._frame_cache: dict[int, np.ndarray] = {}
        self.shape = self._cache.shape

    def _frame(self, t: int) -> np.ndarray:
        cache = getattr(self, "_frame_cache", None)
        if cache is None:
            # Backward-compatible with light-weight objects constructed in
            # tests or old pickles before the memoisation field existed.
            return compose_frame(self._cache[t], self._per_t.get(t, []))
        if t not in cache:
            cache[t] = compose_frame(self._cache[t], self._per_t.get(t, []))
        return cache[t]

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, sl):
        # Tracker morphology features use NumPy-style advanced indexing
        # ``vol[t, z, y, x]``.  Support it without materialising the complete
        # composed movie: render each requested frame once, then gather that
        # frame's points.  DetectorDataset's scalar/slice access below remains
        # byte-for-byte unchanged.
        if isinstance(sl, tuple):
            if len(sl) != 4:
                raise IndexError("ComposedVolume expects (t, z, y, x)")
            tt, zz, yy, xx = np.broadcast_arrays(*[
                np.asarray(index) for index in sl
            ])
            flat_t = tt.reshape(-1).astype(np.int64)
            flat_z = zz.reshape(-1).astype(np.int64)
            flat_y = yy.reshape(-1).astype(np.int64)
            flat_x = xx.reshape(-1).astype(np.int64)
            out = np.empty(len(flat_t), dtype=np.float32)
            for t in np.unique(flat_t):
                rows = flat_t == t
                frame = self._frame(int(t))
                out[rows] = frame[flat_z[rows], flat_y[rows], flat_x[rows]]
            return out.reshape(tt.shape)
        if isinstance(sl, slice):
            return np.stack([self._frame(t)
                             for t in range(*sl.indices(self.shape[0]))])
        return self._frame(int(sl))


from biohub.detector.dataset import DetectorDataset  # noqa: E402  (no cycle:
# detector never imports synthesis; module-level so DataLoader workers can
# pickle the dataset across Windows spawn)


class SynthDetectorDataset(DetectorDataset):
    """`DetectorDataset` over a mixed list of real names and variant keys.

    A variant key is `"{sample}::{tag}"` (e.g. `6bba_09961292::v000`). Real
    names behave byte-identically to the parent class; variant keys serve
    composited frames and MERGED ground truth through the parent's own
    augmentation/target path — so an injected daughter's flip-consistent
    backward birth vector is built by the same `make_targets` call that
    builds every real target.
    """

    SEP = "::"

    def __init__(self, names, cfg, birth_oversample: int = 1):
        if cfg.label_subset and any(self.SEP in n for n in names):
            raise SystemExit(
                "label_subset with synthetic variants is unsupported")
        if cfg.volume_cache is None:
            raise SystemExit("synthetic variants require the volume cache "
                             "(they are composited onto it)")
        self._variants_root = (Path(cfg.data_root).parent.parent
                               / "experiments/synthesis_engine/variants")
        super().__init__(names, cfg)
        if birth_oversample > 1:
            index_set = set(self.index)
            extra = []
            for name in names:
                if self.SEP not in name:
                    continue
                gt = self._gt_for(name)
                deg = np.zeros(len(gt.t), dtype=np.int64)
                np.add.at(deg, gt.edges[:, 0], 1)
                daughters = gt.edges[deg[gt.edges[:, 0]] >= 2, 1]
                for t in sorted({int(v) for v in np.unique(gt.t[daughters])}):
                    if (name, t) in index_set:
                        extra.extend([(name, t)] * (birth_oversample - 1))
            self.index.extend(extra)
            print(f"birth oversampling x{birth_oversample}: "
                  f"+{len(extra)} index entries", flush=True)

    def _variant_npz(self, name: str) -> Path:
        host_name, tag = name.split(self.SEP)
        return self._variants_root / host_name / f"{tag}.npz"

    def _gt_for(self, name):
        if self.SEP not in name:
            return super()._gt_for(name)
        if name not in self._gt:
            host_name, _ = name.split(self.SEP)
            host = load_gt(Path(self.cfg.data_root) / f"{host_name}.geff")
            self._gt[name] = merge_gt(host, load_variant_nodes(
                self._variant_npz(name)))
        return self._gt[name]

    def _arr_for(self, name):
        if self.SEP not in name:
            return super()._arr_for(name)
        if name not in self._arr:
            host_name, _ = name.split(self.SEP)
            self._arr[name] = ("cache", ComposedVolume(
                Path(self.cfg.volume_cache) / f"{host_name}.npy",
                self._variant_npz(name)))
        return self._arr[name]


class SynthVariantData:
    """One (host, variant): composited frames + merged GT, training-ready."""

    def __init__(self, name: str, tag: str, repo: Path):
        self.name, self.tag = name, tag
        self.cache = np.load(repo / "artifacts/volumes_v1" / f"{name}.npy",
                             mmap_mode="r")
        vdir = repo / "experiments/synthesis_engine/variants" / name
        self.per_t, syn = load_variant(vdir / f"{tag}.npz")
        self.gt = merge_gt(load_gt(repo / "data/train" / f"{name}.geff"), syn)
        self.n_frames = int(self.cache.shape[0])

    def frame(self, t: int) -> np.ndarray:
        return compose_frame(self.cache[t], self.per_t.get(t, []))

    def stack(self, t: int) -> np.ndarray:
        """The `[t-1, t, t+1]` training stack, composited."""
        return np.stack([self.frame(t - 1), self.frame(t), self.frame(t + 1)])

    def annotated_frames(self) -> np.ndarray:
        """Frames carrying any (host or synthetic) annotation, interior."""
        fr = np.unique(self.gt.t)
        return fr[(fr >= 1) & (fr <= self.n_frames - 2)]
