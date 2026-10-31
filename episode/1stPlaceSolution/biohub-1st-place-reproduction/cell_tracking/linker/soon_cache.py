"""Per-detection soon-to-divide scores, stored beside the edge feature cache."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SOON_KEY = "soon"
SOON_DAUGHTER_KEY = "soon_daughter"
SOON_ALL_KEY = "soon_all"
SOON_HIDDEN_KEY = "soon_hidden"
SOON_HIDDEN_PRED_KEY = "soon_hidden_pred"
INTERPHASE_COL, MOTHER_COL, DAUGHTER_COL = (0, 1, 2)
SOON_KEYS = (SOON_KEY, SOON_DAUGHTER_KEY, SOON_ALL_KEY, SOON_HIDDEN_KEY, SOON_HIDDEN_PRED_KEY)
SOON_META_NAME = "soon_meta.json"


def soon_frame_path(soon_dir: Path | str, movie: str, frame: int) -> Path:
    return Path(soon_dir) / movie / f"frame_{int(frame):04d}.npy"


def soon_hidden_frame_path(soon_dir: Path | str, movie: str, frame: int) -> Path:
    return Path(soon_dir) / movie / f"frame_{int(frame):04d}_hidden.npy"


def soon_hidden_pred_frame_path(soon_dir: Path | str, movie: str, frame: int) -> Path:
    return Path(soon_dir) / movie / f"frame_{int(frame):04d}_hidden_pred.npy"


def soon_meta_path(soon_dir: Path | str) -> Path:
    return Path(soon_dir) / SOON_META_NAME


def write_soon_meta(soon_dir: Path | str, **fields) -> Path:
    """Write the directory-level sidecar metadata, merging into any existing file."""
    path = soon_meta_path(soon_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = read_soon_meta(soon_dir)
    meta.update(fields)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return path


def read_soon_meta(soon_dir: Path | str) -> dict:
    path = soon_meta_path(soon_dir)
    return {} if not path.is_file() else json.loads(path.read_text())


def soon_hidden_dim(soon_dir: Path | str) -> int | None:
    """``D`` for a sidecar built with ``--emit-hidden``, or None if it carries none."""
    got = read_soon_meta(soon_dir).get("hidden_dim")
    return None if got is None else int(got)


def soon_hidden_pred_dim(soon_dir: Path | str) -> int | None:
    """``D`` of the predicted-``t+1`` array, or None for a single-array sidecar."""
    got = read_soon_meta(soon_dir).get("hidden_pred_dim")
    return None if got is None else int(got)


def soon_hidden_two_role(soon_dir: Path | str) -> bool:
    """Whether this sidecar carries the predicted array beside the observed one."""
    return bool(read_soon_meta(soon_dir).get("hidden_two_role", False))


def write_soon_hidden_frame(path: Path | str, hidden: np.ndarray) -> None:
    """Write an ``(N, D)`` penultimate-activation sidecar frame."""
    h = np.asarray(hidden, dtype=np.float32)
    if h.ndim != 2:
        raise ValueError(f"hidden sidecar must be (N, D), got {h.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, h)


def read_soon_hidden_frame(path: Path | str) -> np.ndarray | None:
    path = Path(path)
    return None if not path.is_file() else np.load(path)


def write_soon_hidden_pred_frame(path: Path | str, hidden: np.ndarray) -> None:
    """Write an ``(N, D)`` predicted-``t+1`` frame."""
    write_soon_hidden_frame(path, hidden)


def read_soon_hidden_pred_frame(path: Path | str) -> np.ndarray | None:
    return read_soon_hidden_frame(path)


def write_soon_frame(path: Path | str, scores: np.ndarray) -> None:
    """Write a ``(N,)`` binary or ``(N, 3)`` 3-class sidecar frame."""
    s = np.asarray(scores, dtype=np.float32)
    if s.ndim == 2 and s.shape[1] != 3:
        raise ValueError(f"multiclass sidecar must be (N, 3) [interphase, mother, daughter], got {s.shape}")
    if s.ndim not in (1, 2):
        raise ValueError(f"sidecar must be (N,) or (N, 3), got {s.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, s)


def read_soon_frame(path: Path | str) -> np.ndarray | None:
    path = Path(path)
    return None if not path.is_file() else np.load(path)


def attach_soon(
    cache: dict, scores: np.ndarray, hidden: np.ndarray | None = None, hidden_pred: np.ndarray | None = None
) -> dict:
    """Return *cache* carrying per-detection scores under ``soon`` (and ``soon_daughter``)."""
    s = np.asarray(scores, dtype=np.float32)
    n = int(np.asarray(cache["coords_native"]).shape[0])
    out = dict(cache)
    if s.ndim == 3:
        if s.shape[1] != n or s.shape[2] != 3:
            raise ValueError(f"soon scores {s.shape} do not match {n} cached detections")
        mean = s.mean(axis=0)
        out[SOON_KEY] = np.ascontiguousarray(mean[:, MOTHER_COL])
        out[SOON_DAUGHTER_KEY] = np.ascontiguousarray(mean[:, DAUGHTER_COL])
        out[SOON_ALL_KEY] = np.ascontiguousarray(s)
    elif s.shape == (n,):
        out[SOON_KEY] = s
    elif s.shape == (n, 3):
        out[SOON_KEY] = np.ascontiguousarray(s[:, MOTHER_COL])
        out[SOON_DAUGHTER_KEY] = np.ascontiguousarray(s[:, DAUGHTER_COL])
        out[SOON_ALL_KEY] = np.ascontiguousarray(s)
    else:
        raise ValueError(f"soon scores {s.shape} do not match {n} cached detections")
    for key, arr in ((SOON_HIDDEN_KEY, hidden), (SOON_HIDDEN_PRED_KEY, hidden_pred)):
        if arr is None:
            continue
        h = np.asarray(arr, dtype=np.float32)
        if h.ndim != 2 or h.shape[0] != n:
            raise ValueError(f"soon hidden states {h.shape} do not match {n} cached detections")
        out[key] = np.ascontiguousarray(h)
    return out


def attach_soon_from_dir(
    cache: dict,
    soon_dir: Path | str | list | tuple,
    movie: str,
    with_hidden: bool = False,
    with_hidden_pred: bool = False,
    routing: str = "shared",
) -> dict:
    """Attach the sidecar frame matching *cache*'s own frame index."""
    dirs = list(soon_dir) if isinstance(soon_dir, (list, tuple)) else [soon_dir]
    if not dirs:
        raise ValueError("soon_dir is empty")
    routing = str(routing)
    if routing not in ("shared", "matched"):
        raise ValueError(f"routing must be 'shared' or 'matched', got {routing!r}")
    frame = int(np.asarray(cache["frame"]))
    if len(dirs) == 1:
        path = soon_frame_path(dirs[0], movie, frame)
        scores = read_soon_frame(path)
        if scores is None:
            raise FileNotFoundError(f"no soon sidecar at {path}")
        hidden = None
        if with_hidden:
            hpath = soon_hidden_frame_path(dirs[0], movie, frame)
            hidden = read_soon_hidden_frame(hpath)
            if hidden is None:
                raise FileNotFoundError(f"no soon hidden sidecar at {hpath}; rebuild the sidecar with --emit-hidden")
        hidden_pred = None
        if with_hidden_pred:
            ppath = soon_hidden_pred_frame_path(dirs[0], movie, frame)
            hidden_pred = read_soon_hidden_pred_frame(ppath)
            if hidden_pred is None:
                raise FileNotFoundError(
                    f"no predicted soon hidden sidecar at {ppath}; rebuild the sidecar with build_lejepa_sidecar.py --two-role"
                )
        return attach_soon(cache, scores, hidden=hidden, hidden_pred=hidden_pred)
    if with_hidden or with_hidden_pred:
        raise ValueError("hidden soon attach is single-dir only")
    parts = []
    for d in dirs:
        path = soon_frame_path(d, movie, frame)
        sc = read_soon_frame(path)
        if sc is None:
            raise FileNotFoundError(f"no soon sidecar at {path}")
        parts.append(np.asarray(sc, dtype=np.float32))
    stacked = np.stack(parts, axis=0)
    if routing == "shared":
        return attach_soon(cache, stacked.mean(axis=0))
    return attach_soon(cache, stacked)


def soon_role(cache: dict, role: str) -> np.ndarray | None:
    """The per-detection score a gate should read for *role*."""
    if role == "mother":
        return cache.get(SOON_KEY)
    if role == "daughter":
        got = cache.get(SOON_DAUGHTER_KEY)
        return cache.get(SOON_KEY) if got is None else got
    raise ValueError(f"role must be 'mother' or 'daughter', got {role!r}")


def align_soon_nn(
    query_coords: np.ndarray,
    query_soon: np.ndarray,
    key_coords: np.ndarray | None,
    key_soon: np.ndarray | None,
    *,
    voxel_size=None,
    max_um: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Copy *key_soon* onto query nodes via spatial NN; hold *query_soon* otherwise."""
    from cell_tracking.common.config import SCALE as NATIVE_VOXEL_UM
    from cell_tracking.linker.motion import DEFAULT_MAX_LINK_UM, nearest_neighbor_index

    q = np.asarray(query_soon, np.float32)
    if q.ndim == 1:
        q = q[:, None]
    out = np.array(q, copy=True)
    n = len(out)
    has = np.zeros(n, bool)
    if key_coords is None or key_soon is None or n == 0:
        return (out, has)
    key_s = np.asarray(key_soon, np.float32)
    if key_s.ndim == 1:
        key_s = key_s[:, None]
    if len(key_s) == 0:
        return (out, has)
    if key_s.shape[-1] != out.shape[-1]:
        raise ValueError(f"query soon width {out.shape[-1]} does not match key width {key_s.shape[-1]}")
    vs = NATIVE_VOXEL_UM if voxel_size is None else voxel_size
    radius = DEFAULT_MAX_LINK_UM if max_um is None else float(max_um)
    idx, has = nearest_neighbor_index(query_coords, key_coords, voxel_size=vs, max_um=radius)
    out[has] = key_s[idx[has]]
    return (out, has)


def temporal_soon_from_caches(
    caches: list[dict], t: int, *, gap: int = 1, voxel_size=None, max_um: float | None = None
) -> dict[str, np.ndarray]:
    """NN-aligned mother history / daughter lookahead for frame pair ``t -> t+gap``."""
    c0, c1 = (caches[t], caches[t + gap])
    soon_t = c0.get(SOON_ALL_KEY)
    soon_t1 = c1.get(SOON_ALL_KEY)
    if soon_t is None or soon_t1 is None:
        raise ValueError("temporal_soon_from_caches needs soon_all on both pair frames")
    coords_t = c0["coords_native"]
    coords_t1 = c1["coords_native"]

    def _nb(i: int):
        if i < 0 or i >= len(caches):
            return (None, None)
        c = caches[i]
        return (c.get("coords_native"), c.get(SOON_ALL_KEY))

    tm2_c, tm2_s = _nb(t - 2)
    tm1_c, tm1_s = _nb(t - 1)
    tp2_c, tp2_s = _nb(t + gap + 1)
    kw = dict(voxel_size=voxel_size, max_um=max_um)
    soon_tm2, has_tm2 = align_soon_nn(coords_t, soon_t, tm2_c, tm2_s, **kw)
    soon_tm1, has_tm1 = align_soon_nn(coords_t, soon_t, tm1_c, tm1_s, **kw)
    soon_tp2, has_tp2 = align_soon_nn(coords_t1, soon_t1, tp2_c, tp2_s, **kw)
    return {
        "soon_tm2": soon_tm2,
        "soon_tm1": soon_tm1,
        "soon_tp2": soon_tp2,
        "soon_has_tm2": has_tm2.astype(np.float32),
        "soon_has_tm1": has_tm1.astype(np.float32),
        "soon_has_tp2": has_tp2.astype(np.float32),
    }


SISTER_SHIFT = 20
SISTER_PAD_KEY = np.int64(1) << np.int64(62)


def sister_frame_path(sister_dir: Path | str, movie: str, frame: int) -> Path:
    return Path(sister_dir) / movie / f"frame_{int(frame):04d}_sister.npz"


def sister_keys(triples: np.ndarray) -> np.ndarray:
    """``(M,)`` packed keys for ``(M, 3)`` ``(parent, child_a, child_b)`` rows."""
    t = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    lo = np.minimum(t[:, 1], t[:, 2])
    hi = np.maximum(t[:, 1], t[:, 2])
    return t[:, 0] << 2 * SISTER_SHIFT | lo << SISTER_SHIFT | hi


def write_sister_frame(path: Path | str, triples: np.ndarray, logits: np.ndarray) -> None:
    """Write one frame's ``(parent, child_a, child_b) -> sister logit`` table."""
    t = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    z = np.asarray(logits, dtype=np.float32).reshape(-1)
    if len(t) != len(z):
        raise ValueError(f"{len(t)} sister triples for {len(z)} logits")
    if len(t) and (t < 0).any():
        raise ValueError("sister triples must be non-negative detection indices")
    if len(t) and (t[:, 1] >= t[:, 2]).any():
        raise ValueError("sister triples must be ordered child_a < child_b")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, triples=t, logits=z)


def read_sister_frame(path: Path | str) -> tuple[np.ndarray, np.ndarray] | None:
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path) as data:
        return (np.asarray(data["triples"], dtype=np.int64), np.asarray(data["logits"], dtype=np.float32))


def sister_lookup_arrays(triples: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Key-sorted ``(keys, logits)`` ready for a searchsorted lookup."""
    keys = sister_keys(triples)
    order = np.argsort(keys, kind="stable")
    return (keys[order], np.asarray(logits, dtype=np.float32).reshape(-1)[order])


def read_sister_lookup(sister_dir: Path | str, movie: str, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Key-sorted sister table for one frame; empty arrays when the frame has none."""
    got = read_sister_frame(sister_frame_path(sister_dir, movie, frame))
    if got is None:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))
    return sister_lookup_arrays(*got)
