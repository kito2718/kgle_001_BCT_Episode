# Dataset loading shared by training and inference.

import csv
import glob
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
import zarr

from config import BaseConfig
from utils import *

sys.path.append(BaseConfig.vendor_src)
sys.path.append(BaseConfig.vendor_scripts)

from tracking_cellmot.io import open_dataset  # noqa: E402
from train_unet_transformer import (  # noqa: E402
    extract_pos_features,
    load_dataset_windows,
)


def read_human_events(manifest):
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        events = list(csv.DictReader(f))
    if not events:
        raise ValueError("Human division manifest is empty")
    for event in events:
        event["t"] = event.get("division_t", event.get("t"))
    return events


def add_human_divisions_to_graph(graph, events, scale, max_distance=7.0):
    """Return a graph copy containing the reviewed division events."""
    augmented = graph.copy()
    existing = {}
    for row in graph.node_attrs().to_dicts():
        existing.setdefault(int(row["t"]), []).append((
            int(row["node_id"]),
            np.asarray([row["z"], row["y"], row["x"]], dtype=np.float32),
        ))

    next_id = max(augmented.node_ids(), default=0) + 1
    scale = np.asarray(scale, dtype=np.float32)
    for event in events:
        t = int(event["t"])
        node_ids = []
        for name, frame in (
            ("parent", t), ("daughter_1", t + 1), ("daughter_2", t + 1)
        ):
            point = np.asarray(
                [float(event[f"{name}_{axis}"]) for axis in "zyx"],
                dtype=np.float32,
            )
            node_id = None
            candidates = existing.get(frame, [])
            if event.get(f"{name}_has_gt_node") == "True" and candidates:
                distances = [
                    float(np.linalg.norm((candidate - point) * scale))
                    for _, candidate in candidates
                ]
                nearest = int(np.argmin(distances))
                if distances[nearest] <= max_distance:
                    node_id = candidates[nearest][0]
            if node_id is None:
                node_id = next_id
                next_id += 1
                z, y, x = np.rint(point).astype(int)
                augmented.add_node(
                    {"t": frame, "z": int(z), "y": int(y), "x": int(x)},
                    index=node_id,
                )
            node_ids.append(node_id)

        for daughter_id in node_ids[1:]:
            if not augmented.has_edge(node_ids[0], daughter_id):
                augmented.add_edge(node_ids[0], daughter_id, {})
    return augmented


class Frame(NamedTuple):
    coords: torch.Tensor
    pos: torch.Tensor
    real: torch.Tensor
    matched: list[torch.Tensor]
    feat: torch.Tensor


class CompetitionDataset(torch.utils.data.Dataset):
    """Two-frame windows from the competition data."""

    def __init__(self, cfg, stems, desc, augmentations=None):
        print(f"loading {desc}: {len(stems)} clips", flush=True)
        clips = [
            load_dataset_windows(
                Path(cfg.train_dir) / f"{stem}.zarr",
                window_size=cfg.time_length,
                downsample=calculate_downsample(cfg),
            )
            for stem in stems
        ]
        print(f"  {sum(len(windows) for _, windows in clips)} windows", flush=True)
        self.pos_per_dim = cfg.pos_per_dim
        self.max_nodes = max(
            max(window.node_counts)
            for _, windows in clips
            for window in windows
        )
        self.augmentations = augmentations or []
        self.samples = [
            (self._pad(window, video), video)
            for video, windows in clips
            for window in windows
        ]

    def _pad(self, window, video):
        """Pad variable cell counts so samples can be batched."""
        m = self.max_nodes
        coords = torch.zeros(2, m, 3, dtype=torch.float32)
        masks = torch.zeros(2, m, dtype=torch.bool)
        pos_feats = torch.zeros(2, m, window.pos_feats[0].shape[1])
        targets = torch.zeros(1, m, m, dtype=torch.float32)

        for i, n in enumerate(window.node_counts):
            # The vendor loader uses raw/downsample coordinates, which align with
            # point sampling. Average pooling aligns each output voxel to the
            # centre of its raw-image pooling block instead.
            raw_coords = window.coords[i] * window.coords[i].new_tensor(
                video.downsample
            )
            network_coords = image_to_network_coordinates(
                raw_coords, video.downsample
            )
            coords[i, :n] = network_coords
            masks[i, :n] = True
            full_coords = np.column_stack([
                np.full(n, window.t_start + i, dtype=np.float32),
                network_coords.numpy(),
            ])
            pos_feats[i, :n] = torch.from_numpy(
                extract_pos_features(
                    full_coords, video.image_shape, self.pos_per_dim
                )
            )
        targets[0, :window.node_counts[0], :window.node_counts[1]] = window.targets[0]

        return {
            "t_start": window.t_start,
            "n_frames": 2,
            "pos_feats": pos_feats,
            "coords": coords,
            "masks": masks,
            "targets": targets,
            "node_counts": torch.tensor(window.node_counts),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample, video = self.samples[index]
        raw = zarr.open_group(str(video.zarr_path), mode="r")["0"][
            sample["t_start"]:sample["t_start"] + 2,
        ].astype(np.float32)
        imgs = downsample_volume(torch.from_numpy(raw), video.downsample)
        imgs = (
            (imgs - video.q_low) / (video.q_high - video.q_low + 1e-6)
        ).clamp(min=0)
        if tuple(imgs.shape[1:]) != tuple(video.image_shape[1:]):
            imgs = F.interpolate(
                imgs[:, None], size=video.image_shape[1:],
                mode="trilinear", align_corners=False,
            )[:, 0]

        coords, masks = sample["coords"], sample["masks"]
        rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        for aug in self.augmentations:
            imgs, coords, masks = aug(imgs, coords, masks, rng=rng)

        return {
            **sample,
            "imgs": imgs.half(),
            "coords": coords,
            "masks": masks,
            "image_shape": torch.tensor(video.image_shape),
            "voxel_size": torch.tensor(video.voxel_size, dtype=torch.float32),
            "downsample": torch.tensor(video.downsample, dtype=torch.float32),
        }


def human_division_samples(cfg, manifest, source_dataset, max_nodes):
    videos = {
        Path(video.zarr_path).stem: video
        for _, video in source_dataset.samples
    }
    events = [
        event for event in read_human_events(manifest)
        if event["clip"] in videos
    ]
    samples = []
    for event in events:
        t = int(event["t"])
        raw = np.asarray([
            [float(event[f"{name}_{axis}"]) for axis in "zyx"]
            for name in ("parent", "daughter_1", "daughter_2")
        ], dtype=np.float32)
        video = videos[event["clip"]]
        points = image_to_network_coordinates(raw, video.downsample)
        features = extract_pos_features(
            np.column_stack((
                np.asarray([t, t + 1, t + 1], dtype=np.float32), points,
            )),
            video.image_shape,
            cfg.pos_per_dim,
        )
        coords = torch.zeros(2, max_nodes, 3)
        masks = torch.zeros(2, max_nodes, dtype=torch.bool)
        pos_feats = torch.zeros(2, max_nodes, 4 * cfg.pos_per_dim)
        targets = torch.zeros(1, max_nodes, max_nodes)
        coords[0, 0] = torch.from_numpy(points[0])
        coords[1, :2] = torch.from_numpy(points[1:])
        masks[0, 0] = True
        masks[1, :2] = True
        pos_feats[0, 0] = torch.from_numpy(features[0])
        pos_feats[1, :2] = torch.from_numpy(features[1:])
        targets[0, 0, :2] = 1.0
        samples.append(({
            "t_start": t,
            "n_frames": 2,
            "pos_feats": pos_feats,
            "coords": coords,
            "masks": masks,
            "targets": targets,
            "node_counts": torch.tensor([1, 2]),
        }, video))
    return samples


class HumanDivisionDataset(CompetitionDataset):
    """Standalone human-positive division samples for evaluation."""

    def __init__(self, cfg, manifest, source_dataset):
        self.augmentations = []
        self.max_nodes = 2
        self.samples = human_division_samples(
            cfg, manifest, source_dataset, self.max_nodes)
        print(f"  loaded {len(self.samples)} human-positive divisions",
              flush=True)


class ZebraHubSource(NamedTuple):
    name: str
    zarr_path: Path
    image_shape: np.ndarray
    voxel_size: np.ndarray
    q_low: float
    q_high: float
    frames: list[np.ndarray]
    frame_ids: list[np.ndarray]
    train_pairs: np.ndarray
    val_pairs: np.ndarray
    split_frame: int
    edge_source_ids: np.ndarray
    edge_target_ids: np.ndarray


def split_frame_pairs(frame_pairs, n_frames, val_fraction):
    """Split by time and omit the one pair that crosses the boundary."""
    if not 0 < val_fraction < 1:
        raise ValueError("zebrahub_val_fraction must be between 0 and 1")
    split_frame = int(n_frames * (1 - val_fraction))
    frame_pairs = np.asarray(frame_pairs, dtype=np.int32)
    train_pairs = frame_pairs[frame_pairs + 1 < split_frame]
    val_pairs = frame_pairs[frame_pairs >= split_frame]
    if not len(train_pairs) or not len(val_pairs):
        raise ValueError("ZebraHub train/validation time split is empty")
    return train_pairs, val_pairs, split_frame


def load_zebrahub_sources(cfg):
    """Load shared metadata for all configured ZebraHub recordings once."""
    downsample = np.asarray(calculate_downsample(cfg), dtype=np.int64)
    sources = []

    for name in cfg.zebrahub_ids:
        zarr_path = Path(cfg.zebrahub_dir) / f"{name}_resampled.zarr"
        geff_path = Path(cfg.zebrahub_dir) / f"{name}_resampled.geff"
        ds = open_dataset(
            zarr_path, normalize=False, load_image=False, require_tracks=False
        )
        image_shape = np.asarray(ds.image_shape, dtype=np.int64)

        # Loading in the parent process lets DataLoader workers share these large,
        # read-only arrays instead of loading one copy per worker.
        geff = zarr.open_group(str(geff_path), mode="r")
        ids = np.asarray(geff["nodes/ids"][:], dtype=np.uint64)
        times = np.asarray(geff["nodes/props/t/values"][:], dtype=np.int32)
        coords = np.column_stack([
            np.asarray(geff[f"nodes/props/{axis}/values"][:], dtype=np.int32)
            for axis in "zyx"
        ])
        if np.any(times[1:] < times[:-1]):
            order = np.argsort(times, kind="stable")
            times, ids, coords = times[order], ids[order], coords[order]

        counts = np.bincount(times, minlength=int(image_shape[0]))
        boundaries = np.cumsum(counts)[:-1]
        frames = np.split(coords, boundaries)
        frame_ids = np.split(ids, boundaries)
        frame_pairs = np.flatnonzero((counts[:-1] > 0) & (counts[1:] > 0))
        train_pairs, val_pairs, split_frame = split_frame_pairs(
            frame_pairs, image_shape[0], cfg.zebrahub_val_fraction
        )

        edges = np.asarray(geff["edges/ids"][:], dtype=np.uint64)
        if np.any(edges[1:, 1] < edges[:-1, 1]):
            edges = edges[np.argsort(edges[:, 1], kind="stable")]

        source = ZebraHubSource(
            name=name,
            zarr_path=zarr_path,
            image_shape=image_shape,
            voxel_size=np.asarray(ds.scale, dtype=np.float32) * downsample,
            q_low=float(ds.quantiles[cfg.q_low_key]),
            q_high=float(ds.quantiles[cfg.q_high_key]),
            frames=frames,
            frame_ids=frame_ids,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            split_frame=split_frame,
            edge_source_ids=edges[:, 0],
            edge_target_ids=edges[:, 1],
        )
        sources.append(source)
        print(
            f"  {name}: {len(times):,} nodes | "
            f"pairs={len(train_pairs)} train + {len(val_pairs)} val | "
            f"split_frame={split_frame}",
            flush=True,
        )

    print(f"loaded ZebraHub sources: {[source.name for source in sources]}", flush=True)
    return sources


class ZebraHubDataset(torch.utils.data.Dataset):
    """Random competition-sized crops from multiple ZebraHub recordings."""

    max_nodes = 1024

    def __init__(self, cfg, sources, split, num_samples, augmentations=None):
        if split not in {"train", "val"}:
            raise ValueError("ZebraHub split must be 'train' or 'val'")
        if cfg.time_length != 2:
            raise ValueError("ZebraHub pretraining currently requires time_length=2")
        if not sources:
            raise ValueError("At least one ZebraHub source is required")

        self.sources = sources
        self.split = split
        self.num_samples = num_samples
        self.seed = cfg.seed
        self.downsample = np.asarray(calculate_downsample(cfg), dtype=np.int64)
        self.output_shape = np.asarray(cfg.volume_shape, dtype=np.int64)
        self.crop_shape = self.output_shape * self.downsample
        self.pos_per_dim = cfg.pos_per_dim
        self.augmentations = augmentations or []

    def __len__(self):
        return self.num_samples

    def _choose_source(self, index, rng):
        if self.split == "val":
            return self.sources[index % len(self.sources)]
        return self.sources[int(rng.integers(len(self.sources)))]

    def _random_crop(self, source, rng):
        frame_pairs = (
            source.train_pairs if self.split == "train" else source.val_pairs
        )
        for _ in range(20):
            t = int(rng.choice(frame_pairs))
            anchor_frame = source.frames[t + int(rng.integers(2))]
            anchor = anchor_frame[int(rng.integers(len(anchor_frame)))]

            # Randomly place one known cell inside the crop. Near an image boundary,
            # move the crop inward; the image is never padded.
            origin = anchor - rng.uniform(0.25, 0.75, 3) * self.crop_shape
            origin = np.clip(
                origin.astype(np.int64),
                0,
                source.image_shape[1:] - self.crop_shape,
            )

            point_ids, points = [], []
            for frame in (t, t + 1):
                coords = source.frames[frame]
                inside = (
                    (coords >= origin) & (coords < origin + self.crop_shape)
                ).all(1)
                point_ids.append(source.frame_ids[frame][inside])
                points.append(coords[inside] - origin)

            if max(map(len, points)) <= self.max_nodes:
                return t, origin, point_ids, points

        raise RuntimeError(
            f"Could not find a {source.name} crop with at most {self.max_nodes} cells"
        )

    def _build_edge_targets(self, source, source_ids, target_ids):
        """Build the source-by-target edge matrix for one crop."""
        target = torch.zeros(self.max_nodes, self.max_nodes, dtype=torch.float32)
        source_rows = {int(node_id): row for row, node_id in enumerate(source_ids)}

        positions = np.searchsorted(source.edge_target_ids, target_ids)
        found = positions < len(source.edge_target_ids)
        found[found] &= (
            source.edge_target_ids[positions[found]] == target_ids[found]
        )
        for col, edge_index in zip(np.flatnonzero(found), positions[found]):
            row = source_rows.get(int(source.edge_source_ids[edge_index]))
            if row is not None:
                target[row, col] = 1.0
        return target

    def __getitem__(self, index):
        if self.split == "val":
            rng = np.random.default_rng(self.seed + index)
        else:
            rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))

        source = self._choose_source(index, rng)
        t, origin, point_ids, points = self._random_crop(source, rng)
        stop = origin + self.crop_shape
        raw = zarr.open_group(str(source.zarr_path), mode="r")["0"][
            t:t + 2,
            origin[0]:stop[0],
            origin[1]:stop[1],
            origin[2]:stop[2],
        ].astype(np.float32)
        imgs = downsample_volume(torch.from_numpy(raw), self.downsample)
        imgs = (
            (imgs - source.q_low) / (source.q_high - source.q_low + 1e-6)
        ).clamp(min=0)

        coords = torch.zeros(2, self.max_nodes, 3, dtype=torch.float32)
        masks = torch.zeros(2, self.max_nodes, dtype=torch.bool)
        pos_feats = torch.zeros(
            2, self.max_nodes, 4 * self.pos_per_dim, dtype=torch.float32
        )
        node_counts = torch.tensor([len(p) for p in points], dtype=torch.long)
        image_shape = (source.image_shape[0], *self.output_shape)
        for i, frame_points in enumerate(points):
            n = len(frame_points)
            scaled = image_to_network_coordinates(
                frame_points.astype(np.float32), self.downsample
            )
            coords[i, :n] = torch.from_numpy(scaled)
            masks[i, :n] = True
            full_coords = np.column_stack([
                np.full(n, t + i, dtype=np.float32), scaled,
            ])
            pos_feats[i, :n] = torch.from_numpy(
                extract_pos_features(full_coords, image_shape, self.pos_per_dim)
            )

        targets = self._build_edge_targets(
            source, point_ids[0], point_ids[1]
        ).unsqueeze(0)

        for aug in self.augmentations:
            imgs, coords, masks = aug(imgs, coords, masks, rng=rng)

        return {
            "source": source.name,
            "t_start": t,
            "n_frames": 2,
            "imgs": imgs.half(),
            "pos_feats": pos_feats,
            "coords": coords,
            "masks": masks,
            "targets": targets,
            "node_counts": node_counts,
            "image_shape": torch.tensor(image_shape, dtype=torch.long),
            "voxel_size": torch.tensor(source.voxel_size, dtype=torch.float32),
            "downsample": torch.tensor(self.downsample, dtype=torch.float32),
        }


def split_dir(cfg, split=None):
    return cfg.train_dir if (split or cfg.split) == "train" else cfg.test_dir


def clip_ids(cfg, split=None):
    # Every clip in the split, or the explicit list in the config.
    if cfg.clip_ids:
        return list(cfg.clip_ids)
    if (split or cfg.split) == "train" and cfg.valid_ids:
        return list(cfg.valid_ids)
    d = split_dir(cfg, split)
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(d, "*.zarr")))


def load_volume(cfg, sample_id, split=None):
    # (T, Z, Y, X) float32, subsampled to an isotropic grid and quantile-normalised.
    # Returns the array plus the physical voxel size after subsampling.
    zarr_file = os.path.join(split_dir(cfg, split), f"{sample_id}.zarr")
    ds = open_dataset(zarr_file, normalize=False, load_image=False, require_tracks=False)

    arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low = float(ds.quantiles[cfg.q_low_key])
    q_high = float(ds.quantiles[cfg.q_high_key])

    downsample = calculate_downsample(cfg, arr.shape[1:])
    # Process a few time points at once so inference never materialises the
    # entire full-resolution clip in RAM.
    parts = []
    for start in range(0, arr.shape[0], 4):
        raw = torch.from_numpy(arr[start:start + 4].astype(np.float32))
        parts.append(downsample_volume(raw, downsample).numpy())
    small = np.concatenate(parts)
    assert small.shape[1:] == tuple(cfg.volume_shape), small.shape

    small = (small - q_low) / (q_high - q_low + 1e-6)
    small = np.clip(small, 0.0, None)

    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    return small, {
        "voxel_size": voxel_size,
        "scale": ds.scale,
        "downsample": downsample,
        "zarr_path": str(ds.zarr_path),
    }


def load_truth(cfg, sample_id):
    # Ground-truth graph plus the clip's estimated total cell count. The count is the
    # denominator of the node-count penalty and covers unannotated cells too, which is
    # why under-predicting can push the multiplier above 1.
    from geff import GeffMetadata

    zarr_file = os.path.join(cfg.train_dir, f"{sample_id}.zarr")
    ds = open_dataset(zarr_file, normalize=False, load_image=False, require_tracks=True)
    meta = GeffMetadata.read(zarr_file.replace(".zarr", ".geff"))
    return ds.tracks, ds.scale, float(meta.extra["estimated_number_of_nodes"])


def has_truth(cfg, sample_id):
    return os.path.exists(os.path.join(cfg.train_dir, f"{sample_id}.geff"))


def all_stems(cfg):
    return sorted(
        os.path.basename(p)[:-5] for p in glob.glob(os.path.join(cfg.train_dir, "*.zarr"))
        if os.path.exists(os.path.join(cfg.train_dir, os.path.basename(p)[:-5] + ".geff"))
    )


def build_fold(cfg):
    """Return one deterministic train/validation split."""
    stems = all_stems(cfg)
    if cfg.fold_mode not in {"stratified", "all"}:
        raise ValueError(f"unknown fold mode: {cfg.fold_mode}")
    if not 0 <= cfg.fold < cfg.n_folds:
        raise ValueError(f"fold must be between 0 and {cfg.n_folds - 1}")

    from sklearn.model_selection import StratifiedKFold

    strata = []
    for stem in stems:
        graph, _, _ = load_truth(cfg, stem)
        divisions = sum(
            graph.out_degree(node_id) == 2 for node_id in graph.node_ids()
        )
        division_bin = "2+" if divisions >= 2 else str(divisions)
        strata.append(f"{stem.split('_')[0]}_{division_bin}")
    splits = StratifiedKFold(
        cfg.n_folds, shuffle=True, random_state=cfg.seed
    ).split(stems, strata)
    train_idx, val_idx = list(splits)[cfg.fold]
    train = stems if cfg.fold_mode == "all" else [stems[i] for i in train_idx]
    val = [stems[i] for i in val_idx]

    if cfg.fold_mode != "all":
        assert set(train).isdisjoint(val)
        assert set(train) | set(val) == set(stems)

    return {
        "train": sorted(train),
        "test": sorted(val),
        "val_in_train": cfg.fold_mode == "all",
    }
