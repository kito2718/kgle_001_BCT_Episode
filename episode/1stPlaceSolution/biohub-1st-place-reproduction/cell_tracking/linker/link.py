"""Movie linking from cached frames and edge checkpoint loading."""

from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from cell_tracking.linker.cache import list_cached_frames, read_frame_cache
from cell_tracking.linker.intensity import INTENSITY_NORM_KEY, INTENSITY_RAW_SUM_KEY, attach_intensity_from_dir
from cell_tracking.linker.occupancy import (
    attach_occupancy_from_dir,
    occupancy_forward_kw,
    occupancy_frame_path,
    read_occupancy_frame,
)
from cell_tracking.linker.model import UnetCellPoseEdgeTransformer
from cell_tracking.linker.soon_cache import SOON_ALL_KEY, SOON_HIDDEN_KEY, attach_soon_from_dir
from cell_tracking.linker.motion import local_tissue_flow
from cell_tracking.linker.support import candidate_mask
from cell_tracking.linker.support import prune_cache_frame as _support_prune_cache_frame
from cell_tracking.linker.decode import FORK_DECODES, _decode_pair, decode_fork_configurations, resolve_refractory_by_prob
from cell_tracking.linker.scoring import _check_feat_width, _pair_fwd_kw, _tensor_frame, _velocity_tensor


@torch.no_grad()
def prune_cache_frame(cache: dict, min_prob: float = 0.0, include_geom: bool = False) -> dict:
    """Drop nodes below ``min_prob`` using the shared support helper."""
    return _support_prune_cache_frame(cache, min_prob, include_geom=include_geom)


def link_movie_from_cache(
    model: UnetCellPoseEdgeTransformer,
    cache_dir: Path | str,
    movie: str,
    *,
    device: str | torch.device = "cpu",
    threshold: float = 0.1,
    max_children: int = 2,
    min_prob: float = 0.0,
    support_radius_um: float = 30.0,
    support_k: int = 16,
    fork_threshold: float | None = None,
    fork_max_children: int = 32,
    soon_dir: Path | str | list | tuple | None = None,
    intensity_dir: Path | str | None = None,
    occupancy_dir: Path | str | None = None,
    occupancy_division_dir: Path | str | None = None,
    flow_k: int = 20,
) -> tuple:
    """Link a movie from precomputed per-frame caches."""
    if float(support_radius_um) < 0:
        raise ValueError(f"support_radius_um must be >= 0, got {support_radius_um}")
    if int(support_k) < 0:
        raise ValueError(f"support_k must be >= 0, got {support_k}")
    if float(support_radius_um) <= 0 and int(support_k) <= 0:
        raise ValueError("local support needs support_radius_um > 0 or support_k > 0")
    if int(fork_max_children) < 0:
        raise ValueError(f"fork_max_children must be >= 0, got {fork_max_children}")
    if "greedy" not in FORK_DECODES:
        raise ValueError(f"fork_decode must be one of {FORK_DECODES}, got {'greedy'!r}")
    model_has_fork = bool(True)
    fork_enabled = bool(True)
    if not model_has_fork:
        raise ValueError("fork_enabled=True requires a checkpoint with fork_head=True")
    paths = list_cached_frames(cache_dir, movie)
    if len(paths) < 2:
        return ([], [], None, [])
    raw_frames = [read_frame_cache(p) for p in paths]
    soon_two_role = bool(False)
    if soon_dir is not None:
        soon_kw = {"routing": str("shared")}
        raw_frames = [attach_soon_from_dir(c, soon_dir, movie, **soon_kw) for c in raw_frames]
    if intensity_dir is not None:
        raw_frames = [attach_intensity_from_dir(c, intensity_dir, movie) for c in raw_frames]
    else:
        raise ValueError(
            "this checkpoint has intensity_pair=True or intensity_fork=True; scoring it needs intensity_dir"
        )
    if occupancy_dir is not None:
        occ_routing = str("shared") if isinstance(occupancy_dir, (list, tuple)) else "shared"
        raw_frames = [attach_occupancy_from_dir(c, occupancy_dir, movie, routing=occ_routing) for c in raw_frames]
    else:
        raise ValueError(
            "this checkpoint has occupancy_pair=True or occupancy_fork=True; scoring it needs occupancy_dir"
        )
    div_frames = None
    if occupancy_division_dir is not None:
        if isinstance(occupancy_division_dir, (list, tuple)):
            raise ValueError(
                "occupancy_fork_daughter is not defined for matched occupancy ensembles; pass a single --occupancy-division-dir"
            )
        div_frames = []
        for cache in raw_frames:
            path = occupancy_frame_path(occupancy_division_dir, movie, int(np.asarray(cache["frame"])))
            got = read_occupancy_frame(path)
            if got is None:
                raise FileNotFoundError(f"no occupancy division sidecar at {path}")
            div_frames.append(got)
    else:
        raise ValueError(
            "this checkpoint has occupancy_fork_daughter=True; scoring it needs occupancy_division_dir (the --out-division sidecar)"
        )
    caches = [prune_cache_frame(c, min_prob, include_geom=True) for c in raw_frames]
    gt_oracle = None
    frame_kw = {"ablate_appearance": False}
    tensors = [_tensor_frame(c, **frame_kw) for c in caches]
    coords_np = [np.asarray(c["coords_native"], dtype=np.float64) for c in caches]
    for feat, _c, _v in tensors:
        if len(feat):
            _check_feat_width(feat, model, movie)
    frames: list[np.ndarray] = list(coords_np)
    decode_mc = 2
    decode_kw = {
        "threshold": threshold,
        "max_children": decode_mc,
        "birth_offset": 0.0,
        "decode": "dual",
        "dustbin": 0.0,
        "res_max_um": 0.0,
        "res_mask_logits": False,
        "assign_kw": None,
        "mc_passes": 1,
    }

    def _vel(t: int) -> torch.Tensor:
        """Incoming velocity at ``t``, measured over the SAME spacing as the pair."""
        return _velocity_tensor(caches[t], coords_np[t - 1] if t > 0 else None)

    def _fwd(t: int, t1: int) -> dict:
        w0 = w1 = None
        return _pair_fwd_kw(
            reverse_tta=False,
            flow_k=flow_k,
            flow_kind="local_c2f",
            flow_weights_t=w0,
            flow_weights_t1=w1,
            harmonic_w=0.0,
        )

    need_occ = bool(True)

    def _occ(t: int, t1: int, gap: int = 1) -> dict:
        return occupancy_forward_kw(
            caches[t],
            caches[t1],
            coords_np[t],
            coords_np[t1],
            gap=gap,
            fork_occ_t=None if div_frames is None else div_frames[t],
            fork_occ_t1=None if div_frames is None else div_frames[t1],
        )

    def _flow_at(t: int) -> np.ndarray:
        w = None
        return local_tissue_flow(coords_np[t], coords_np[t + 1], k=flow_k, kind="local_c2f", weights=w)

    if fork_threshold is None:
        fork_threshold = 0.0
    keep_matrices = float(0.0) > 0.0
    need_gate = True
    need_flow = None is not None
    pair_links: list[list[tuple[int, int]]] = []
    link_probs: list[list[float]] = []
    second_children_prev: set[int] = set()
    fork_refractory_prev: set[int] = set()
    refractory_by_prob = "prob" == "prob"
    fork_np_by_t: dict[int, dict[str, np.ndarray]] = {}
    gap1: dict[int, np.ndarray] = {}
    for t in range(len(paths) - 1):
        feat_t, coords_t, vol_t = tensors[t]
        feat_t1, coords_t1, vol_t1 = tensors[t + 1]
        if len(feat_t) == 0 or len(feat_t1) == 0:
            pair_links.append([])
            link_probs.append([])
            continue
        fwd = _fwd(t, t + 1)
        support = candidate_mask(coords_np[t], coords_np[t + 1], radius_um=support_radius_um, rank_k=support_k)
        soon_all_t = caches[t].get(SOON_ALL_KEY)
        soon_all_t1 = caches[t + 1].get(SOON_ALL_KEY)
        soon_hidden_t = caches[t].get(SOON_HIDDEN_KEY)
        soon_hidden_t1 = caches[t + 1].get(SOON_HIDDEN_KEY)
        temporal_kw = {}
        sister_kw = {}
        ctx_kw = {}
        decoded = _decode_pair(
            model,
            feat_t,
            feat_t1,
            coords_t,
            coords_t1,
            vol_t,
            vol_t1,
            _vel(t),
            device=device,
            **decode_kw,
            **fwd,
            support_mask=support,
            return_aux=True,
            return_fork=True,
            soon_t=soon_all_t,
            soon_t1=soon_all_t1,
            soon_hidden_t=soon_hidden_t,
            soon_hidden_t1=soon_hidden_t1,
            intensity_t=caches[t].get(INTENSITY_NORM_KEY),
            intensity_t1=caches[t + 1].get(INTENSITY_NORM_KEY),
            intensity_raw_sum_t=caches[t].get(INTENSITY_RAW_SUM_KEY),
            intensity_raw_sum_t1=caches[t + 1].get(INTENSITY_RAW_SUM_KEY),
            **temporal_kw,
            **sister_kw,
            **_occ(t, t + 1),
            **ctx_kw,
        )
        links, probs, logits, fork_output = decoded
        if fork_output is not None:
            fork_np = {name: value.detach().cpu().numpy() for name, value in fork_output.items()}
            fork_np_by_t[t] = fork_np
            links, config_probs = decode_fork_configurations(
                fork_np,
                threshold=float(fork_threshold),
                max_children=max_children,
                div_penalty=float(0.0),
                refractory_parents=None,
            )
            forked_now = {p for p, c in Counter((i for i, _ in links)).items() if c >= 2}
            fork_refractory_prev = {j for i, j in links if i in forked_now}
            link_probs.append(config_probs)
        else:
            link_probs.append([float(probs[i, j]) for i, j in links])
        pair_links.append(links)
    for t, (links, config_probs) in resolve_refractory_by_prob(
        fork_np_by_t, threshold=float(fork_threshold), max_children=max_children, div_penalty=float(0.0)
    ).items():
        pair_links[t] = links
        link_probs[t] = config_probs
    gap = 2
    probs_by_t: dict[int, np.ndarray] = {}
    return (frames, pair_links, None, link_probs)


def load_edge_checkpoint(
    path: Path | str, *, device: str | torch.device = "cpu"
) -> tuple[UnetCellPoseEdgeTransformer, dict]:
    """Load a trained edge head and its saved config."""
    import json

    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_path = path.parent / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    config = json.loads(cfg_path.read_text())
    required = (
        "feat_dim",
        "hidden_dim",
        "n_heads",
        "n_blocks",
        "mlp_ratio",
        "dropout",
        "flow_feats",
        "shape_feats",
        "shape_residual_direction",
        "fov_feats",
        "use_volume",
        "local_attention",
        "attn_radius_um",
        "attn_k",
        "fork_head",
        "fork_hidden_dim",
        "fork_topk",
        "fork_train_topk",
        "soon_fork",
        "fork_vol_conserv",
        "fork_hierarchical",
        "fork_kind_occupancy",
        "fork_kind_geometry",
        "fork_kind_expected",
        "intensity_pair",
        "intensity_fork",
        "occupancy_pair",
        "occupancy_fork",
        "occupancy_fork_daughter",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{path}: missing linker metadata {missing}")
    expected = {
        "feat_dim": 39,
        "hidden_dim": 128,
        "n_heads": 4,
        "n_blocks": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.3,
        "attn_radius_um": 30.0,
        "attn_k": 16,
        "fork_hidden_dim": 64,
        "fork_topk": 16,
        "fork_train_topk": 16,
    }
    for key in required:
        if key not in expected:
            expected[key] = True
    if any((config[key] != value for key, value in expected.items())):
        raise ValueError(f"{path}: linker architecture differs from the packaged recipe")
    if config.get("drop_time_feats", True) is not True:
        raise ValueError(f"{path}: temporal node features are unsupported")
    disabled = (
        "motion_feats",
        "um_dist",
        "no_parent",
        "symmetric_update",
        "self_attention",
        "matchability",
        "edge_attention",
        "edge_rope",
        "soon_node",
        "soon_null",
        "soon_pair",
        "soon_fork_stats",
        "soon_temporal",
        "fork_kind_edge_context",
        "sister_feats",
        "edge_context_pool",
        "drop_shape_feats",
        "soon_hidden_two_role",
    )
    if any((config.get(key, False) for key in disabled)) or config.get("soon_hidden_dim", 0) != 0:
        raise ValueError(f"{path}: unsupported linker features")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"{path}: missing linker model state")
    model = UnetCellPoseEdgeTransformer(
        feat_dim=int(config.get("feat_dim", 82)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        n_heads=int(config.get("n_heads", 4)),
        n_blocks=int(config.get("n_blocks", 4)),
        mlp_ratio=float(config.get("mlp_ratio", 2.0)),
        dropout=float(config.get("dropout", 0.3)),
        pair_chunk_size=config.get("pair_chunk_size", 32),
        fork_hidden_dim=int(config.get("fork_hidden_dim", 64)),
        fork_topk=int(config.get("fork_topk", 12)),
        fork_train_topk=int(config.get("fork_train_topk", 32)),
        attn_radius_um=float(config.get("attn_radius_um", 30.0)),
        attn_k=int(config.get("attn_k", 16)),
    )
    model.soon_hidden_two_role = bool(config.get("soon_hidden_two_role", False))
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return (model, config)
