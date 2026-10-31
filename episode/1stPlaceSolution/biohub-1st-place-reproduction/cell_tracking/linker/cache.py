"""Precomputed per-frame edge-head node caches."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from cell_tracking.detector.inference import UnetCellPoseModel, decode_instances
from cell_tracking.common.io import load_volume, read_array_meta
from cell_tracking.linker.features import instance_node_features, node_feat_dim


def _parse_movie_lines(text: str) -> list[str]:
    """Non-blank movie names; ``#`` comments and duplicates stripped (order kept)."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and line not in out:
            out.append(line)
    return out


def load_movie_list(path: Path | str) -> set[str]:
    p = Path(path)
    if not p.is_file():
        return set()
    return set(_parse_movie_lines(p.read_text()))


def frame_cache_path(out_dir: Path | str, movie: str, frame: int) -> Path:
    return Path(out_dir) / movie / f"frame_{int(frame):04d}.npz"


def write_frame_cache(
    path: Path | str, coords_native: np.ndarray, feats: np.ndarray, volumes: np.ndarray, frame: int
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coords_native=np.asarray(coords_native, dtype=np.float64),
        feats=np.asarray(feats, dtype=np.float32),
        volumes=np.asarray(volumes, dtype=np.float64),
        frame=np.int32(frame),
    )


def read_frame_cache(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def list_cached_frames(cache_dir: Path | str, movie: str) -> list[Path]:
    """Sorted per-frame NPZ paths for one movie."""
    d = Path(cache_dir) / movie
    if not d.is_dir():
        return []
    return sorted(d.glob("frame_*.npz"))


def extract_frame_cache(
    fsm: UnetCellPoseModel | None,
    vol: np.ndarray | None,
    *,
    device: str = "cpu",
    frame_at=None,
    t_rel: float = 0.0,
    base: int | None = None,
    include_time: bool = True,
    include_appearance: bool = True,
    include_geom: bool = False,
    cached_decode: tuple[np.ndarray, np.ndarray] | None = None,
    pool: int | None = None,
    native_shape: tuple[int, int, int] | None = None,
    return_inst: bool = False,
    **decode_overrides,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode one native volume and return ``(coords_native, feats, volumes)``."""
    if cached_decode is not None:
        if include_appearance:
            raise ValueError("cached_decode has no d1 appearance block; pass include_appearance=False")
        if pool is None or base is None or native_shape is None:
            raise ValueError("cached_decode needs pool, base and native_shape")
        inst, prob_logit = cached_decode
        d1 = None
    else:
        base = int(base if base is not None else fsm.meta.get("base") or 24)
        pool = fsm.pool
        native_shape = tuple(np.asarray(vol).shape)
        if include_appearance:
            inst, pred, d1 = decode_instances(
                fsm, vol, device=device, frame_at=frame_at, _return_pred=True, _return_d1=True, **decode_overrides
            )
        else:
            inst, pred = decode_instances(
                fsm, vol, device=device, frame_at=frame_at, _return_pred=True, **decode_overrides
            )
            d1 = None
        prob_logit = pred[fsm.ch.prob]
    window_shape = (2, *tuple(native_shape))
    _, coords_native, feats, volumes = instance_node_features(
        inst,
        d1,
        prob_logit,
        pool=int(pool),
        t_rel=t_rel,
        window_shape=window_shape,
        base=int(base),
        include_time=include_time,
        include_appearance=include_appearance,
        include_geom=include_geom,
    )
    if return_inst:
        return (coords_native, feats, volumes, inst)
    return (coords_native, feats, volumes)


def precompute_movie(
    fsm: UnetCellPoseModel | None,
    data_dir: Path | str,
    movie: str,
    out_dir: Path | str,
    *,
    device: str = "cpu",
    limit_t: int = 0,
    skip_existing: bool = True,
    frozen_context: bool = False,
    include_time: bool = True,
    include_appearance: bool = True,
    include_geom: bool = False,
    decode_cache: Path | str | None = None,
    intensity_mask_dir: Path | str | None = None,
    intensity_bg_percentile: float = 10.0,
    **decode_overrides,
) -> int:
    """Write per-frame NPZ caches for one movie; returns frames written."""
    if decode_cache is not None:
        raise ValueError("only checkpoint-based cache generation is supported")
    data_dir = Path(data_dir)
    zarr_path = data_dir / f"{movie}.zarr"
    meta = read_array_meta(zarr_path)
    n_t = int(meta["shape"][0])
    if limit_t:
        n_t = min(n_t, int(limit_t))

    def frame_at_for(t: int):
        if frozen_context:
            return None

        def frame_at(dt: int):
            u = t + dt
            if 0 <= u < int(meta["shape"][0]):
                return load_volume(zarr_path, u, meta=meta)
            return None

        return frame_at

    expected_dim = node_feat_dim(
        int(fsm.meta.get("base") or 24), include_time, include_appearance, include_geom=include_geom
    )
    mask_intensity_path_for = None
    if intensity_mask_dir is not None:
        from cell_tracking.linker.intensity import intensity_frame_path

        mask_intensity_path_for = lambda t: intensity_frame_path(intensity_mask_dir, movie, t)
    written = 0
    for t in range(n_t):
        out_path = frame_cache_path(out_dir, movie, t)
        mask_out_path = mask_intensity_path_for(t) if mask_intensity_path_for else None
        need_frame = not (skip_existing and out_path.is_file())
        need_mask = mask_out_path is not None and (not (skip_existing and mask_out_path.is_file()))
        if not need_frame and (not need_mask):
            continue
        vol = load_volume(zarr_path, t, meta=meta)
        coords, feats, volumes, inst = extract_frame_cache(
            fsm,
            vol,
            device=device,
            frame_at=frame_at_for(t),
            include_time=include_time,
            include_appearance=include_appearance,
            include_geom=include_geom,
            return_inst=True,
            **decode_overrides,
        )
        if feats.shape != (len(coords), expected_dim):
            raise RuntimeError(
                f"{movie} frame {t}: feats shape {feats.shape}, expected ({len(coords)}, {expected_dim})"
            )
        if need_frame:
            write_frame_cache(out_path, coords, feats, volumes, t)
            written += 1
        if need_mask:
            from cell_tracking.linker.intensity import write_instance_mask_intensity_sidecar

            write_instance_mask_intensity_sidecar(
                mask_out_path,
                vol,
                inst,
                coords_native=coords,
                pool=int(fsm.pool),
                frame=t,
                bg_percentile=intensity_bg_percentile,
            )
    return written


def movie_prob_percentile(cache_dir: Path | str, movie: str, pct: float, *, include_geom: bool = False) -> float:
    """Return the ``pct``-th percentile of a movie's cached detector confidences."""
    from cell_tracking.linker.support import cache_mean_prob

    if not 0.0 < float(pct) < 100.0:
        raise ValueError(f"percentile must be in (0, 100), got {pct}")
    chunks = []
    for path in list_cached_frames(cache_dir, movie):
        probs = cache_mean_prob(read_frame_cache(path), include_geom=include_geom)
        if len(probs):
            chunks.append(probs)
    if not chunks:
        return 0.0
    return float(np.percentile(np.concatenate(chunks), float(pct)))


def cache_shape_stats(
    cache_dir: Path | str,
    movies: list[str] | None = None,
    *,
    base: int = 24,
    include_appearance: bool = False,
    max_frames_per_movie: int = 0,
) -> dict[str, float]:
    """Mean of the two decode-derived feature dims, and of the raw volumes."""
    from cell_tracking.linker.features import shape_slice

    cache_dir = Path(cache_dir)
    names = movies if movies is not None else sorted((p.name for p in cache_dir.iterdir() if p.is_dir()))
    sl = shape_slice(base, include_appearance)
    n = 0
    log_volume = 0.0
    mean_prob = 0.0
    volume = 0.0
    for movie in names:
        paths = list_cached_frames(cache_dir, movie)
        if max_frames_per_movie:
            paths = paths[:max_frames_per_movie]
        for path in paths:
            c = read_frame_cache(path)
            feats = np.asarray(c["feats"], np.float64)
            if not len(feats):
                continue
            n += len(feats)
            log_volume += float(feats[:, sl][:, 0].sum())
            mean_prob += float(feats[:, sl][:, 1].sum())
            volume += float(np.asarray(c["volumes"], np.float64).sum())
    if n == 0:
        raise ValueError(f"no cached nodes under {cache_dir}")
    return {"n_nodes": n, "log_volume": log_volume / n, "mean_prob": mean_prob / n, "volume": volume / n}
