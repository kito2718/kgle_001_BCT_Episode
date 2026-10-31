"""Pair logits from cached frames, forward/reverse fusion and link confidence."""

from __future__ import annotations
import numpy as np
import torch
from cell_tracking.linker.assign import unbalanced_log_assignment
from cell_tracking.linker.features import appearance_slice
from cell_tracking.linker.motion import incoming_velocity, local_tissue_flow


def mean_impute_appearance(feats: np.ndarray, base: int = 24) -> np.ndarray:
    """Replace the ``d1`` appearance block with its column mean over this frame's nodes."""
    out = np.array(feats, dtype=np.float32, copy=True)
    if not len(out) or out.shape[1] < 2 * int(base):
        return out
    sl = appearance_slice(base)
    out[:, sl] = out[:, sl].mean(axis=0, keepdims=True)
    return out


def _check_feat_width(feats: np.ndarray, model, movie: str) -> None:
    """Fail loudly when a cache's feature width does not match the model's input."""
    want = int(model.proj.in_features)
    got = int(feats.shape[1])
    if got == want:
        return
    slack = 0
    slack += 8
    if slack and got == want + slack:
        return
    raise ValueError(
        f"{movie}: cache has {got} features per node but this head expects {want}. The cache and the checkpoint were built with different --no-appearance-feats / --no-time-feats / --drop-time-feats settings; re-cache or re-train."
    )


def _tensor_frame(
    cache: dict[str, np.ndarray], *, ablate_appearance: bool = False, base: int = 24
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(feats, coords, volumes)`` as float32 tensors for one cached frame."""
    coords = torch.from_numpy(np.asarray(cache["coords_native"], dtype=np.float32))
    feats_np = np.asarray(cache["feats"], dtype=np.float32)
    if ablate_appearance:
        feats_np = mean_impute_appearance(feats_np, base=base)
    feats = torch.from_numpy(feats_np)
    volumes = torch.from_numpy(np.asarray(cache["volumes"], dtype=np.float32))
    return (feats, coords, volumes)


def _velocity_tensor(cache: dict, prev_coords: np.ndarray | None) -> torch.Tensor:
    """``(N, 4)`` ``[vz, vy, vx, has]`` incoming velocity for one cached frame."""
    coords = np.asarray(cache["coords_native"], dtype=np.float64)
    vel, has = incoming_velocity(coords, prev_coords)
    return torch.from_numpy(np.concatenate([vel, has[:, None].astype(np.float64)], axis=1)).float()


def spatial_node_features(feat_t, feat_t1):
    """Slice the 8 constant t_rel dims off a cache that still has them."""
    from cell_tracking.linker.features import time_block_slice

    def _one(f):
        sl = time_block_slice(int(f.shape[-1]))
        if sl is None:
            return f
        start, end = sl
        if isinstance(f, torch.Tensor):
            return torch.cat([f[..., :start], f[..., end:]], dim=-1)
        return np.concatenate([f[..., :start], f[..., end:]], axis=-1)

    return (_one(feat_t), _one(feat_t1))


def _context_to_device(ctx: dict | None, device) -> dict | None:
    if not ctx:
        return ctx
    out = {}
    for key, value in ctx.items():
        if value is None:
            out[key] = None
        elif torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = torch.as_tensor(value).to(device)
    return out


@torch.no_grad()
def _pair_logits(
    model,
    feat_t,
    feat_t1,
    coords_t,
    coords_t1,
    vol_t,
    vol_t1,
    vel_t,
    *,
    device,
    matchability: bool = False,
    reverse_tta: bool = False,
    flow_k: int = 20,
    flow_kind: str = "local",
    flow_weights_t=None,
    flow_weights_t1=None,
    flow_zero: bool = False,
    pair_mask: torch.Tensor | np.ndarray | None = None,
    return_fork: bool = False,
    mc_passes: int = 1,
    soon_t: torch.Tensor | np.ndarray | None = None,
    soon_t1: torch.Tensor | np.ndarray | None = None,
    soon_hidden_t: torch.Tensor | np.ndarray | None = None,
    soon_hidden_t1: torch.Tensor | np.ndarray | None = None,
    soon_tm2: torch.Tensor | np.ndarray | None = None,
    soon_tm1: torch.Tensor | np.ndarray | None = None,
    soon_tp2: torch.Tensor | np.ndarray | None = None,
    soon_has_tm2: torch.Tensor | np.ndarray | None = None,
    soon_has_tm1: torch.Tensor | np.ndarray | None = None,
    soon_has_tp2: torch.Tensor | np.ndarray | None = None,
    intensity_t: torch.Tensor | np.ndarray | None = None,
    intensity_t1: torch.Tensor | np.ndarray | None = None,
    intensity_raw_sum_t: torch.Tensor | np.ndarray | None = None,
    intensity_raw_sum_t1: torch.Tensor | np.ndarray | None = None,
    occ_next: torch.Tensor | np.ndarray | None = None,
    has_next: torch.Tensor | np.ndarray | None = None,
    occ_prev: torch.Tensor | np.ndarray | None = None,
    has_prev: torch.Tensor | np.ndarray | None = None,
    peak_next: torch.Tensor | np.ndarray | None = None,
    next_maps: torch.Tensor | np.ndarray | None = None,
    occ_fork_next: torch.Tensor | np.ndarray | None = None,
    occ_fork_has_next: torch.Tensor | np.ndarray | None = None,
    occ_fork_prev: torch.Tensor | np.ndarray | None = None,
    occ_fork_has_prev: torch.Tensor | np.ndarray | None = None,
    sister_key: np.ndarray | None = None,
    sister_logit: np.ndarray | None = None,
    edge_context: dict | None = None,
):
    """Raw ``(n_t, n_t1)`` logits plus optional side outputs for one frame pair."""
    if mc_passes > 1:
        model.train()
    else:
        model.eval()
    dev = torch.device(device)
    feat_t, feat_t1 = spatial_node_features(feat_t, feat_t1)
    c0 = coords_t.cpu().numpy().astype("float64")
    c1 = coords_t1.cpu().numpy().astype("float64")

    def _flow(a, b, w):
        if flow_zero:
            return torch.zeros((len(a), 3), dtype=torch.float32)
        return torch.from_numpy(local_tissue_flow(a, b, k=int(flow_k), kind=str(flow_kind), weights=w)).float()

    flow_t = _flow(c0, c1, flow_weights_t)
    use_birth = bool(False)
    vel_fwd = vel_t.to(dev) if vel_t is not None and False else None
    soon_fork = bool(True)
    if soon_fork and return_fork and (soon_t is None or soon_t1 is None):
        raise ValueError(
            "this checkpoint has soon_fork=True; fork decoding needs soon_t/soon_t1 (pass --soon-fork-dir so the sidecar is loaded)"
        )
    st = None if soon_t is None else torch.as_tensor(soon_t, dtype=torch.float32)
    st1 = None if soon_t1 is None else torch.as_tensor(soon_t1, dtype=torch.float32)
    sh = None if soon_hidden_t is None else torch.as_tensor(soon_hidden_t, dtype=torch.float32)
    sh1 = None if soon_hidden_t1 is None else torch.as_tensor(soon_hidden_t1, dtype=torch.float32)
    stm2 = None if soon_tm2 is None else torch.as_tensor(soon_tm2, dtype=torch.float32)
    stm1 = None if soon_tm1 is None else torch.as_tensor(soon_tm1, dtype=torch.float32)
    stp2 = None if soon_tp2 is None else torch.as_tensor(soon_tp2, dtype=torch.float32)
    htm2 = None if soon_has_tm2 is None else torch.as_tensor(soon_has_tm2, dtype=torch.float32)
    htm1 = None if soon_has_tm1 is None else torch.as_tensor(soon_has_tm1, dtype=torch.float32)
    htp2 = None if soon_has_tp2 is None else torch.as_tensor(soon_has_tp2, dtype=torch.float32)
    if intensity_t is None or intensity_t1 is None:
        raise ValueError(
            "this checkpoint has intensity_pair=True; scoring needs intensity_t/intensity_t1 (pass --intensity-dir so the sidecar is loaded)"
        )
    if return_fork and (
        intensity_t is None or intensity_t1 is None or intensity_raw_sum_t is None or (intensity_raw_sum_t1 is None)
    ):
        raise ValueError(
            "this checkpoint has intensity_fork=True; fork decoding needs intensity_t/intensity_t1/intensity_raw_sum_t/intensity_raw_sum_t1 (pass --intensity-dir so the sidecar is loaded)"
        )
    it = None if intensity_t is None else torch.as_tensor(intensity_t, dtype=torch.float32)
    it1 = None if intensity_t1 is None else torch.as_tensor(intensity_t1, dtype=torch.float32)
    isum = None if intensity_raw_sum_t is None else torch.as_tensor(intensity_raw_sum_t, dtype=torch.float32)
    isum1 = None if intensity_raw_sum_t1 is None else torch.as_tensor(intensity_raw_sum_t1, dtype=torch.float32)
    if occ_next is None or has_next is None or occ_prev is None or (has_prev is None) or (peak_next is None):
        raise ValueError(
            "this checkpoint has occupancy_pair=True or occupancy_fork=True; scoring needs occ_next/has_next/occ_prev/has_prev/peak_next (pass --occupancy-dir so the sidecar is loaded)"
        )
    if return_fork and (
        occ_fork_next is None or occ_fork_has_next is None or occ_fork_prev is None or (occ_fork_has_prev is None)
    ):
        raise ValueError(
            "this checkpoint has occupancy_fork_daughter=True; fork decoding needs occ_fork_next/occ_fork_has_next/occ_fork_prev/occ_fork_has_prev (pass --occupancy-division-dir so the --out-division sidecar is loaded)"
        )
    if reverse_tta:
        raise ValueError(
            "occupancy_pair is directed (parent next-map vs child prev-map); reverse_tta is not defined for it"
        )
    on = None if occ_next is None else torch.as_tensor(occ_next, dtype=torch.float32)
    hn = None if has_next is None else torch.as_tensor(has_next, dtype=torch.float32)
    op = None if occ_prev is None else torch.as_tensor(occ_prev, dtype=torch.float32)
    hp = None if has_prev is None else torch.as_tensor(has_prev, dtype=torch.float32)
    pn = None if peak_next is None else torch.as_tensor(peak_next, dtype=torch.float32)
    nm = None if next_maps is None else torch.as_tensor(next_maps, dtype=torch.float32)
    ofn = None if occ_fork_next is None else torch.as_tensor(occ_fork_next, dtype=torch.float32)
    ofhn = None if occ_fork_has_next is None else torch.as_tensor(occ_fork_has_next, dtype=torch.float32)
    ofp = None if occ_fork_prev is None else torch.as_tensor(occ_fork_prev, dtype=torch.float32)
    ofhp = None if occ_fork_has_prev is None else torch.as_tensor(occ_fork_has_prev, dtype=torch.float32)
    skey = None if sister_key is None else torch.as_tensor(sister_key, dtype=torch.long)
    slog = None if sister_logit is None else torch.as_tensor(sister_logit, dtype=torch.float32)

    def _run(ft, ft1, ct, ct1, vt, vt1, flow, vel, pair_mask_local=None):
        return model(
            ft.to(dev),
            ft1.to(dev),
            ct.to(dev),
            ct1.to(dev),
            vt.to(dev),
            vt1.to(dev),
            vel_t=vel,
            flow_t=flow.to(dev) if flow is not None else None,
            pair_mask=pair_mask_local,
            return_birth=use_birth,
            return_matchability=matchability,
            return_fork=return_fork,
            soon_t=None if st is None else st.to(dev),
            soon_t1=None if st1 is None else st1.to(dev),
            intensity_t=None if it is None else it.to(dev),
            intensity_t1=None if it1 is None else it1.to(dev),
            intensity_raw_sum_t=None if isum is None else isum.to(dev),
            intensity_raw_sum_t1=None if isum1 is None else isum1.to(dev),
            occ_next=None if on is None else on.to(dev),
            has_next=None if hn is None else hn.to(dev),
            occ_prev=None if op is None else op.to(dev),
            has_prev=None if hp is None else hp.to(dev),
            peak_next=None if pn is None else pn.to(dev),
            next_maps=None if nm is None else nm.to(dev),
            occ_fork_next=None if ofn is None else ofn.to(dev),
            occ_fork_has_next=None if ofhn is None else ofhn.to(dev),
            occ_fork_prev=None if ofp is None else ofp.to(dev),
            occ_fork_has_prev=None if ofhp is None else ofhp.to(dev),
            soon_hidden_t=None if sh is None else sh.to(dev),
            soon_hidden_t1=None if sh1 is None else sh1.to(dev),
            soon_tm2=None if stm2 is None else stm2.to(dev),
            soon_tm1=None if stm1 is None else stm1.to(dev),
            soon_tp2=None if stp2 is None else stp2.to(dev),
            soon_has_tm2=None if htm2 is None else htm2.to(dev),
            soon_has_tm1=None if htm1 is None else htm1.to(dev),
            soon_has_tp2=None if htp2 is None else htp2.to(dev),
            sister_key=None if skey is None else skey.to(dev),
            sister_logit=None if slog is None else slog.to(dev),
            edge_context=_context_to_device(edge_context, dev),
        )

    if return_fork and (not True):
        raise ValueError("return_fork requires a checkpoint with fork_head=True")
    if return_fork and reverse_tta:
        raise ValueError("reverse_tta is not defined for fork configurations")
    if mc_passes > 1 and return_fork:
        raise ValueError("mc_passes > 1 is not supported with return_fork")

    def _unpack(out_single):
        if return_fork:
            return (
                out_single["logits"],
                out_single.get("birth"),
                out_single.get("match_t"),
                out_single.get("match_t1"),
            )
        if matchability:
            return (out_single[0], None, out_single[1], out_single[2])
        if use_birth:
            return (out_single[0], out_single[1], None, None)
        return (out_single, None, None, None)

    out = _run(feat_t, feat_t1, coords_t, coords_t1, vol_t, vol_t1, flow_t, vel_fwd, pair_mask)
    fork_output = None
    if return_fork:
        logits = out["logits"]
        birth = out["birth"]
        match_t = out["match_t"]
        match_t1 = out["match_t1"]
        fork_output = {
            name: out[name]
            for name in (
                "config_logits",
                "config_mask",
                "config_kind",
                "config_parent",
                "config_child1",
                "config_child2",
                "null_logits",
            )
        }
    elif matchability:
        logits, match_t, match_t1 = out
        birth = None
    else:
        logits, birth = out if use_birth else (out, None)
        match_t = match_t1 = None
    if mc_passes > 1:
        for _ in range(mc_passes - 1):
            o = _run(feat_t, feat_t1, coords_t, coords_t1, vol_t, vol_t1, flow_t, vel_fwd, pair_mask)
            lg, bi, mt, mt1 = _unpack(o)
            logits = logits + lg
            if birth is not None and bi is not None:
                birth = birth + bi
            if match_t is not None and mt is not None:
                match_t = match_t + mt
                match_t1 = match_t1 + mt1
        logits = logits / mc_passes
        if birth is not None:
            birth = birth / mc_passes
        if match_t is not None:
            match_t = match_t / mc_passes
            match_t1 = match_t1 / mc_passes
        model.eval()
    if reverse_tta:
        flow_rev = _flow(c1, c0, flow_weights_t1)
        out_rev = _run(
            feat_t1,
            feat_t,
            coords_t1,
            coords_t,
            vol_t1,
            vol_t,
            flow_rev,
            None,
            pair_mask.T if pair_mask is not None else None,
        )
        if matchability:
            logits_rev, match_t_rev, match_t1_rev = out_rev
            logits = 0.5 * (logits + logits_rev.transpose(-1, -2))
            match_t = 0.5 * (match_t + match_t1_rev)
            match_t1 = 0.5 * (match_t1 + match_t_rev)
        else:
            logits_rev = out_rev[0] if use_birth else out_rev
            logits = 0.5 * (logits + logits_rev.transpose(-1, -2))
    if return_fork:
        return (logits, birth, match_t, match_t1, fork_output)
    return (logits, birth, match_t, match_t1)


@torch.no_grad()
def _assignment_from_outputs(
    logits: torch.Tensor,
    birth: torch.Tensor | None,
    match_t: torch.Tensor | None,
    match_t1: torch.Tensor | None,
    *,
    coords_t: torch.Tensor,
    coords_t1: torch.Tensor,
    birth_offset: float,
    decode: str,
    dustbin: float,
    sinkhorn_iters: int,
    res_max_um: float,
    res_mask_logits: bool,
    flow_feats: bool,
    support_mask: torch.Tensor | np.ndarray | None,
    assign_kw: dict | None,
    fwd_kw: dict,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert one raw model output into the probabilities used by decoding."""
    if birth is not None:
        aug = torch.cat([logits, (birth + float(0.0)).unsqueeze(0)], dim=0)
        aug_mask = None
        if support_mask is not None:
            sm = torch.as_tensor(support_mask, dtype=torch.bool, device=aug.device)
            aug_mask = torch.cat([sm, torch.ones((1, aug.shape[1]), dtype=torch.bool, device=aug.device)], dim=0)
        log_p = unbalanced_log_assignment(aug, None, match_t1, support_mask=aug_mask, **{}).exp()
        return (log_p[:-1].cpu().numpy(), log_p[-1].cpu().numpy())
    return (
        unbalanced_log_assignment(logits, match_t, match_t1, support_mask=support_mask, **{}).exp().cpu().numpy(),
        None,
    )


def harmonic_fuse_probs(p_fwd, p_rev, w: float) -> np.ndarray:
    """Weighted harmonic mean ``1 / ((1-w)/p_fwd + w/p_rev)``."""
    a = np.asarray(p_fwd, dtype=np.float64)
    if float(w) <= 0.0:
        return a
    b = np.asarray(p_rev, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"p_fwd shape {a.shape} != p_rev shape {b.shape}")
    if not 0.0 < float(w) <= 1.0:
        raise ValueError(f"harmonic_w must be in (0, 1], got {w}")
    num = a * b
    den = (1.0 - float(w)) * b + float(w) * a
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _split_harmonic(fwd_kw: dict) -> tuple[dict, float]:
    """Peel ``harmonic_w`` off the kwargs ``_pair_logits`` does not accept."""
    kw = dict(fwd_kw)
    w = float(kw.pop("harmonic_w", 0.0) or 0.0)
    if w < 0.0 or w > 1.0:
        raise ValueError(f"harmonic_w must be in [0, 1], got {w}")
    if w > 0.0 and kw.get("reverse_tta"):
        raise ValueError("harmonic_w and reverse_tta both consume the reverse pass; pick one")
    return (kw, w)


def _harmonic_with_reverse(
    model,
    feat_t,
    feat_t1,
    coords_t,
    coords_t1,
    vol_t,
    vol_t1,
    *,
    device,
    probs,
    w,
    birth_offset,
    decode,
    dustbin,
    sinkhorn_iters,
    res_max_um,
    res_mask_logits,
    support_mask,
    assign_kw,
    mc_passes,
    fwd_kw,
) -> np.ndarray:
    """Run the swapped pair, assign, transpose, fuse with *probs*."""
    rev_kw = dict(fwd_kw)
    rev_kw["flow_weights_t"] = fwd_kw.get("flow_weights_t1")
    rev_kw["flow_weights_t1"] = fwd_kw.get("flow_weights_t")
    rev_mask = None
    if support_mask is not None:
        rev_mask = support_mask.T if torch.is_tensor(support_mask) else np.asarray(support_mask).T
    logits_rev, birth_rev, match_t_rev, match_t1_rev = _pair_logits(
        model,
        feat_t1,
        feat_t,
        coords_t1,
        coords_t,
        vol_t1,
        vol_t,
        None,
        device=device,
        matchability=bool(False),
        pair_mask=rev_mask,
        mc_passes=mc_passes,
        **rev_kw,
    )
    p_rev, _ = _assignment_from_outputs(
        logits_rev,
        birth_rev,
        match_t_rev,
        match_t1_rev,
        coords_t=coords_t1,
        coords_t1=coords_t,
        birth_offset=birth_offset,
        decode=decode,
        dustbin=dustbin,
        sinkhorn_iters=sinkhorn_iters,
        res_max_um=res_max_um,
        res_mask_logits=res_mask_logits,
        flow_feats=bool(True),
        support_mask=rev_mask,
        assign_kw=assign_kw,
        fwd_kw=rev_kw,
    )
    return harmonic_fuse_probs(probs, np.asarray(p_rev).T, w)


def _pair_fwd_kw(
    reverse_tta: bool = False,
    flow_k: int = 20,
    flow_kind: str = "local",
    flow_weights_t=None,
    flow_weights_t1=None,
    harmonic_w: float = 0.0,
) -> dict:
    return dict(
        reverse_tta=reverse_tta,
        flow_k=flow_k,
        flow_kind=flow_kind,
        flow_weights_t=flow_weights_t,
        flow_weights_t1=flow_weights_t1,
        harmonic_w=float(harmonic_w),
    )


def link_confidence(
    frames: list[np.ndarray],
    pair_links: list[list[tuple[int, int]]],
    link_probs: list[list[float]],
    *,
    default: float = 0.0,
) -> np.ndarray:
    """Per-node max adjacent link probability, in `graph_from_pair_links`' node order."""
    best = [np.full(len(f), float(default), dtype=np.float64) for f in frames]
    for t, links in enumerate(pair_links):
        probs = link_probs[t] if t < len(link_probs) else []
        for (i, j), p in zip(links, probs, strict=True):
            p = float(p)
            if i < len(best[t]):
                best[t][i] = max(best[t][i], p)
            if t + 1 < len(best) and j < len(best[t + 1]):
                best[t + 1][j] = max(best[t + 1][j], p)
    return np.concatenate(best) if best else np.zeros(0, np.float64)
