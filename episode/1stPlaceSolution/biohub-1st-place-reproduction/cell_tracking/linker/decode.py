"""Link and fork assignment decoders over pair probabilities."""

from __future__ import annotations
import threading
from collections import defaultdict
import numpy as np
import torch
from cell_tracking.linker.scoring import _assignment_from_outputs, _harmonic_with_reverse, _pair_logits, _split_harmonic


DECODE_MODES = ("greedy", "hungarian", "sinkhorn", "dual", "dual_hung")


def hungarian_decode(probs: np.ndarray, *, threshold: float = 0.2, max_children: int = 1) -> list[tuple[int, int]]:
    """Globally optimal decode of the same probabilities ``greedy_decode`` sees."""
    from scipy.optimize import linear_sum_assignment

    probs = np.asarray(probs, np.float64)
    n_t, n_t1 = probs.shape
    mc = max(int(max_children), 1)
    if n_t == 0 or n_t1 == 0:
        return []
    eps = 1e-12
    cost = np.repeat(-np.log(np.maximum(probs, eps)), mc, axis=0)
    reject = -np.log(max(float(threshold), eps))
    dummy = np.full((n_t1, n_t1), reject + 1000000.0)
    np.fill_diagonal(dummy, reject)
    rows, cols = linear_sum_assignment(np.vstack([cost, dummy]))
    return [(int(r // mc), int(c)) for r, c in zip(rows, cols, strict=True) if r < n_t * mc]


def greedy_decode(
    probs: np.ndarray, *, threshold: float = 0.5, max_parents: int = 1, max_children: int = 2
) -> list[tuple[int, int]]:
    """Greedy decode with degree caps (``max_parents=1``, ``max_children=2``)."""
    cand = [(probs[i, j], i, j) for i, j in zip(*np.where(probs > threshold))]
    cand.sort(reverse=True)
    n_ch: dict[int, int] = {}
    n_pa: dict[int, int] = {}
    out: list[tuple[int, int]] = []
    for _, i, j in cand:
        i, j = (int(i), int(j))
        if n_ch.get(i, 0) >= max_children or n_pa.get(j, 0) >= max_parents:
            continue
        out.append((i, j))
        n_ch[i] = n_ch.get(i, 0) + 1
        n_pa[j] = n_pa.get(j, 0) + 1
    return out


def decode_with_birth(
    probs: np.ndarray, p_birth: np.ndarray, *, max_parents: int = 1, max_children: int = 2, floor: float = 0.0
) -> list[tuple[int, int]]:
    """Greedy decode where each child's competitor is its own no-parent probability."""
    n_t1 = probs.shape[1]
    cand = []
    for j in range(n_t1):
        col = probs[:, j]
        keep = np.where((col > p_birth[j]) & (col > floor))[0]
        for i in keep:
            cand.append((float(col[i]), int(i), int(j)))
    cand.sort(reverse=True)
    n_ch: dict[int, int] = {}
    n_pa: dict[int, int] = {}
    out: list[tuple[int, int]] = []
    for _, i, j in cand:
        if n_ch.get(i, 0) >= max_children or n_pa.get(j, 0) >= max_parents:
            continue
        out.append((i, j))
        n_ch[i] = n_ch.get(i, 0) + 1
        n_pa[j] = n_pa.get(j, 0) + 1
    return out


def _hungarian_with_birth(probs: np.ndarray, p_birth: np.ndarray, *, max_children: int = 1) -> list[tuple[int, int]]:
    """:func:`hungarian_decode` with a PER-CHILD reservation price from ``p_birth``."""
    from scipy.optimize import linear_sum_assignment

    probs = np.asarray(probs, np.float64)
    n_t, n_t1 = probs.shape
    mc = max(int(max_children), 1)
    if n_t == 0 or n_t1 == 0:
        return []
    eps = 1e-12
    cost = np.repeat(-np.log(np.maximum(probs, eps)), mc, axis=0)
    reject = -np.log(np.maximum(np.asarray(p_birth, np.float64), eps))
    dummy = np.full((n_t1, n_t1), float(reject.max()) + 1000000.0)
    np.fill_diagonal(dummy, reject)
    rows, cols = linear_sum_assignment(np.vstack([cost, dummy]))
    return [(int(r // mc), int(c)) for r, c in zip(rows, cols, strict=True) if r < n_t * mc]


def _decode_probs(
    probs: np.ndarray, p_birth: np.ndarray | None, *, threshold: float, max_children: int, decode: str
) -> list[tuple[int, int]]:
    """Decode a probability matrix the way *decode* asks for."""
    if decode == "dual":
        if p_birth is not None:
            return decode_with_birth(probs, p_birth, max_children=max_children, floor=threshold)
        return greedy_decode(probs, threshold=threshold, max_children=max_children)
    if decode == "dual_hung":
        if p_birth is not None:
            return _hungarian_with_birth(probs, p_birth, max_children=max_children)
        return hungarian_decode(probs, threshold=threshold, max_children=max_children)
    if decode == "sinkhorn":
        return decode_with_birth(probs, p_birth, max_children=max_children)
    if p_birth is not None:
        if decode == "hungarian":
            return _hungarian_with_birth(probs, p_birth, max_children=max_children)
        return decode_with_birth(probs, p_birth, max_children=max_children)
    if decode == "hungarian":
        return hungarian_decode(probs, threshold=threshold, max_children=max_children)
    return greedy_decode(probs, threshold=threshold, max_children=max_children)


FORK_DECODES = ("greedy", "exact", "exact_dual", "greedy_owner", "kindfirst", "bayes")


def decode_fork_configurations(
    config_output: dict[str, np.ndarray],
    *,
    threshold: float = 0.0,
    max_children: int = 2,
    div_penalty: float = 0.0,
    refractory_parents: set[int] | None = None,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Greedily decode one categorical configuration per parent."""
    required = ("config_logits", "config_mask", "config_kind", "config_child1", "config_child2")
    missing = [name for name in required if name not in config_output]
    if missing:
        raise ValueError(f"categorical fork output is missing {missing}")
    logits = np.asarray(config_output["config_logits"], dtype=np.float64)
    mask = np.asarray(config_output["config_mask"], dtype=bool)
    kind = np.asarray(config_output["config_kind"], dtype=np.int64)
    child1 = np.asarray(config_output["config_child1"], dtype=np.int64)
    child2 = np.asarray(config_output["config_child2"], dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError(f"config logits must be 2-D for one pair, got {logits.shape}")
    if any((value.shape != logits.shape for value in (mask, kind, child1, child2))):
        raise ValueError("categorical fork descriptors must share config_logits shape")
    if int(max_children) < 1:
        return ([], [])
    safe = np.where(mask, logits, -np.inf)
    row_max = np.max(safe, axis=1, keepdims=True)
    exp = np.where(mask, np.exp(safe - row_max), 0.0)
    denom = exp.sum(axis=1, keepdims=True)
    if np.any(denom <= 0):
        raise ValueError("a parent has no valid categorical fork configuration")
    log_probs = np.where(mask, safe - row_max - np.log(denom), -np.inf)
    candidates: list[tuple[float, int, int, float]] = []
    for i in range(logits.shape[0]):
        empty = np.flatnonzero(mask[i] & (kind[i] == 0))
        if len(empty) != 1:
            raise ValueError(f"parent {i} must have exactly one valid empty configuration")
        empty_logp = log_probs[i, int(empty[0])]
        parent_refractory = refractory_parents is not None and i in refractory_parents
        for c in np.flatnonzero(mask[i] & (kind[i] != 0)):
            if parent_refractory and kind[i, c] == 2:
                continue
            pen = float(div_penalty) if kind[i, c] == 2 else 0.0
            margin = float(log_probs[i, c] - empty_logp - pen)
            if np.isfinite(margin) and margin >= float(threshold):
                candidates.append((margin, i, int(c), float(np.exp(log_probs[i, c]))))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_parents: set[int] = set()
    owner: dict[int, int] = {}
    links: list[tuple[int, int]] = []
    link_probs: list[float] = []
    for _margin, parent, slot, probability in candidates:
        if parent in used_parents:
            continue
        children = [int(child1[parent, slot])]
        if int(kind[parent, slot]) == 2:
            children.append(int(child2[parent, slot]))
        if len(children) > int(max_children) or len(set(children)) != len(children):
            continue
        if any((child < 0 or child in owner for child in children)):
            continue
        used_parents.add(parent)
        for child in children:
            owner[child] = parent
            links.append((parent, child))
            link_probs.append(probability)
    return (links, link_probs)


def fork_marginal(config_output: dict[str, np.ndarray]) -> np.ndarray:
    """Per-parent P(fork): the row-softmax mass on the parent's fork configurations."""
    log_probs, mask, kind, _, _ = _fork_log_probs(config_output)
    return np.where(mask & (kind == 2), np.exp(log_probs), 0.0).sum(axis=1)


def resolve_refractory_by_prob(
    fork_np_by_t: dict[int, dict[str, np.ndarray]],
    *,
    threshold: float = 0.0,
    max_children: int = 2,
    div_penalty: float = 0.0,
) -> dict[int, tuple[list[tuple[int, int]], list[float]]]:
    """Greedy fork decode over a movie with the refractory rule resolved by P(fork)."""
    p_fork = {t: fork_marginal(out) for t, out in fork_np_by_t.items()}
    barred: dict[int, set[int]] = {t: set() for t in fork_np_by_t}
    stale = set(fork_np_by_t)
    decoded: dict[int, tuple[list[tuple[int, int]], list[float]]] = {}
    while stale:
        for t in stale:
            decoded[t] = decode_fork_configurations(
                fork_np_by_t[t],
                threshold=threshold,
                max_children=max_children,
                div_penalty=div_penalty,
                refractory_parents=barred[t],
            )
        stale = set()
        forks = []
        for t, (links, _) in decoded.items():
            children: dict[int, list[int]] = defaultdict(list)
            for i, j in links:
                children[i].append(j)
            forks.extend(((float(p_fork[t][i]), t, i, ch) for i, ch in children.items() if len(ch) >= 2))
        kept_parents: dict[int, set[int]] = defaultdict(set)
        kept_children: dict[int, set[int]] = defaultdict(set)
        for _p, t, i, ch in sorted(forks, key=lambda f: (-f[0], f[1], f[2])):
            if i in kept_children[t - 1] or any((c in kept_parents[t + 1] for c in ch)):
                barred[t].add(i)
                stale.add(t)
                continue
            kept_parents[t].add(i)
            kept_children[t].update(ch)
    return decoded


def _fork_log_probs(
    config_output: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-softmax log-probabilities shared by the categorical fork decoders."""
    required = ("config_logits", "config_mask", "config_kind", "config_child1", "config_child2")
    missing = [name for name in required if name not in config_output]
    if missing:
        raise ValueError(f"categorical fork output is missing {missing}")
    logits = np.asarray(config_output["config_logits"], dtype=np.float64)
    mask = np.asarray(config_output["config_mask"], dtype=bool)
    kind = np.asarray(config_output["config_kind"], dtype=np.int64)
    child1 = np.asarray(config_output["config_child1"], dtype=np.int64)
    child2 = np.asarray(config_output["config_child2"], dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError(f"config logits must be 2-D for one pair, got {logits.shape}")
    if any((value.shape != logits.shape for value in (mask, kind, child1, child2))):
        raise ValueError("categorical fork descriptors must share config_logits shape")
    safe = np.where(mask, logits, -np.inf)
    row_max = np.max(safe, axis=1, keepdims=True)
    exp = np.where(mask, np.exp(safe - row_max), 0.0)
    denom = exp.sum(axis=1, keepdims=True)
    if np.any(denom <= 0):
        raise ValueError("a parent has no valid categorical fork configuration")
    log_probs = np.where(mask, safe - row_max - np.log(denom), -np.inf)
    return (log_probs, mask, kind, child1, child2)


_GUROBI_ENV = None


_GUROBI_ENV_LOCK = threading.Lock()


def _decode_pair(
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
    threshold: float,
    max_children: int,
    birth_offset: float = 0.0,
    decode: str = "greedy",
    dustbin: float = 0.0,
    sinkhorn_iters: int = 20,
    res_max_um: float = 0.0,
    res_mask_logits: bool = False,
    support_mask: torch.Tensor | np.ndarray | None = None,
    assign_kw: dict | None = None,
    return_aux: bool = False,
    return_fork: bool = False,
    fork_pairs: torch.Tensor | np.ndarray | None = None,
    fork_pair_mask: torch.Tensor | np.ndarray | None = None,
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
    sister_key: np.ndarray | None = None,
    sister_logit: np.ndarray | None = None,
    **fwd_kw,
):
    """One frame pair -> ``(links, probs)``, using the birth logit when the model has one."""
    want_fork = bool(True)
    matchability = bool(False)
    if matchability and "dual" != "dual":
        raise ValueError(
            "this arm was trained with matchability, so it must be scored with --decode dual; the selected decode would drop sigma"
        )
    kw, w = _split_harmonic(fwd_kw)
    out = _pair_logits(
        model,
        feat_t,
        feat_t1,
        coords_t,
        coords_t1,
        vol_t,
        vol_t1,
        vel_t,
        device=device,
        matchability=matchability,
        pair_mask=support_mask,
        return_fork=want_fork,
        mc_passes=1,
        soon_t=soon_t,
        soon_t1=soon_t1,
        soon_hidden_t=soon_hidden_t,
        soon_hidden_t1=soon_hidden_t1,
        soon_tm2=soon_tm2,
        soon_tm1=soon_tm1,
        soon_tp2=soon_tp2,
        soon_has_tm2=soon_has_tm2,
        soon_has_tm1=soon_has_tm1,
        soon_has_tp2=soon_has_tp2,
        sister_key=sister_key,
        sister_logit=sister_logit,
        **kw,
    )
    if want_fork:
        logits, birth, match_t, match_t1, fork_output = out
    else:
        logits, birth, match_t, match_t1 = out
        fork_output = None
    probs, p_birth = _assignment_from_outputs(
        logits,
        birth,
        match_t,
        match_t1,
        coords_t=coords_t,
        coords_t1=coords_t1,
        birth_offset=0.0,
        decode="dual",
        dustbin=0.0,
        sinkhorn_iters=sinkhorn_iters,
        res_max_um=0.0,
        res_mask_logits=False,
        flow_feats=bool(True),
        support_mask=support_mask,
        assign_kw=None,
        fwd_kw=kw,
    )
    if w > 0.0 and (not want_fork):
        probs = _harmonic_with_reverse(
            model,
            feat_t,
            feat_t1,
            coords_t,
            coords_t1,
            vol_t,
            vol_t1,
            device=device,
            probs=probs,
            w=w,
            birth_offset=0.0,
            decode="dual",
            dustbin=0.0,
            sinkhorn_iters=sinkhorn_iters,
            res_max_um=0.0,
            res_mask_logits=False,
            support_mask=support_mask,
            assign_kw=None,
            mc_passes=1,
            fwd_kw=kw,
        )
    links = _decode_probs(probs, p_birth, threshold=threshold, max_children=max_children, decode="dual")
    return (links, probs, logits, fork_output)
