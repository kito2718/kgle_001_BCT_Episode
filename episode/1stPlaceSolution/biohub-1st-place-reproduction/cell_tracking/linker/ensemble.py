"""Forward-output ensemble of unetcellpose edge heads, for submission-time linker averaging."""

from __future__ import annotations
import torch

_FORK_AVG_KEYS = ("logits", "birth", "null_logits")


def _mean(values):
    """Average a list of tensors, propagating ``None`` (all members agree by construction)."""
    if any((v is None for v in values)):
        if not all((v is None for v in values)):
            raise ValueError("ensemble members disagree on a None output slot")
        return None
    return torch.stack(list(values), dim=0).mean(dim=0)


SHORTLIST_MODES = ("union", "consensus")


class EdgeEnsemble:
    """Average several edge heads at ``forward`` output, proxying attributes to the primary."""

    def __init__(self, models, shortlist="consensus"):
        models = list(models)
        if not models:
            raise ValueError("EdgeEnsemble needs at least one member")
        if shortlist not in SHORTLIST_MODES:
            raise ValueError(f"shortlist must be one of {SHORTLIST_MODES}, got {shortlist!r}")
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "primary", models[0])
        object.__setattr__(self, "shortlist", str(shortlist))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "primary"), name)

    def eval(self):
        for m in self.models:
            m.eval()
        return self

    def train(self, mode: bool = True):
        for m in self.models:
            m.train(mode)
        return self

    def to(self, device):
        moved = [m.to(device) for m in self.models]
        object.__setattr__(self, "models", moved)
        object.__setattr__(self, "primary", moved[0])
        return self

    @property
    def fork_topk(self):
        return self.primary.fork_topk

    @fork_topk.setter
    def fork_topk(self, value):
        for m in self.models:
            m.fork_topk = value

    def member_kwargs(self, k, kwargs):
        """Keyword arguments for member ``k``."""
        return kwargs

    def __call__(self, *args, **kwargs):
        if len(self.models) == 1:
            return self.models[0](*args, **self.member_kwargs(0, kwargs))
        outs = [m(*args, **self.member_kwargs(k, kwargs), _return_fork_state=True) for k, m in enumerate(self.models)]
        first = outs[0]
        if isinstance(first, dict):
            merged = dict(first)
            merged.pop("_fork_state", None)
            merged.pop("kind_logits", None)
            merged.pop("who_logits", None)
            for key in _FORK_AVG_KEYS:
                if key in first:
                    merged[key] = _mean([o[key] for o in outs])
            if "config_logits" in first and kwargs.get("fork_children") is None:
                if self.shortlist == "consensus":
                    children = consensus_shortlist(outs, int(self.primary.fork_topk), kwargs.get("pair_mask"))
                else:
                    children = union_shortlists(outs)
                aligned = [
                    m(
                        *args,
                        **{**self.member_kwargs(k, kwargs), "fork_children": children},
                        _fork_state=o["_fork_state"],
                    )
                    for k, (m, o) in enumerate(zip(self.models, outs, strict=True))
                ]
                merged.update(average_fork_configs(aligned))
            elif "config_logits" in first:
                merged.update(average_fork_configs(outs))
            return merged
        if isinstance(first, tuple):
            return tuple((_mean([o[i] for o in outs]) for i in range(len(first))))
        return _mean(outs)


class MatchedEdgeEnsemble(EdgeEnsemble):
    """``EdgeEnsemble`` that feeds member ``k`` sidecar slice ``soon_t[k]``."""

    def member_kwargs(self, k, kwargs):
        soon_t = kwargs.get("soon_t")
        soon_t1 = kwargs.get("soon_t1")
        n = len(self.models)

        def _is_stacked(s):
            if s is None:
                return False
            shape = getattr(s, "shape", None)
            return shape is not None and len(shape) == 3 and (int(shape[0]) == n)

        def _is_member_stacked(s):
            if s is None:
                return False
            shape = getattr(s, "shape", None)
            return shape is not None and len(shape) >= 3 and (int(shape[0]) == n)

        if not _is_stacked(soon_t):
            return kwargs
        kw = dict(kwargs)
        kw["soon_t"] = soon_t[k]
        kw["soon_t1"] = soon_t1[k] if _is_stacked(soon_t1) else soon_t1
        for key in ("occ_next", "has_next", "occ_prev", "has_prev", "peak_next", "next_maps"):
            val = kwargs.get(key)
            if _is_member_stacked(val):
                kw[key] = val[k]
        return kw


def consensus_shortlist(outs: list[dict], k: int, pair_mask=None) -> torch.Tensor:
    """``(B, n_src, k)`` children ranked by the members' MEAN pair logit inside the shared support, ``-1`` where a parent has fewer than ``k`` supported children."""
    logits = _mean([o["logits"] for o in outs])
    support = None
    state = outs[0].get("_fork_state")
    if state is not None and state[4] is not None:
        support = state[4]
    elif pair_mask is not None:
        support = torch.as_tensor(pair_mask, dtype=torch.bool, device=logits.device)
    if support is not None:
        logits = logits.masked_fill(~support.to(torch.bool), float("-inf"))
    k = max(1, min(int(k), int(logits.shape[-1])))
    scores, children = logits.topk(k, dim=-1)
    return torch.where(torch.isfinite(scores), children, torch.full_like(children, -1))


def union_shortlists(outs: list[dict]) -> torch.Tensor:
    """``(..., n_src, K)`` union of the members' singleton-config children, ``-1`` padded."""
    parts = []
    for o in outs:
        single = (o["config_kind"] == 1) & o["config_mask"]
        child = torch.where(single, o["config_child1"], torch.full_like(o["config_child1"], -1))
        parts.append(child)
    cat = torch.cat(parts, dim=-1)
    big = int(cat.max().item()) + 1 if cat.numel() else 0
    x = torch.where(cat < 0, torch.full_like(cat, big), cat)
    x, _ = x.sort(dim=-1)
    dup = torch.zeros_like(x, dtype=torch.bool)
    dup[..., 1:] = x[..., 1:] == x[..., :-1]
    x = torch.where(dup, torch.full_like(x, big), x)
    x, _ = x.sort(dim=-1)
    x = torch.where(x >= big, torch.full_like(x, -1), x)
    any_real = (x >= 0).flatten(0, -2).any(dim=0)
    keep = int(any_real.nonzero().max().item()) + 1 if any_real.any() else 1
    return x[..., :keep]


def _has_kind_head(out: dict) -> bool:
    return out.get("kind_logits") is not None and out.get("who_logits") is not None


def _nonfork_width(config_kind: torch.Tensor) -> int:
    """Leading empty+singleton slot count."""
    n = (config_kind != 2).sum(dim=-1)
    n_nf = int(n.reshape(-1)[0].item())
    if not torch.equal(n, torch.full_like(n, n_nf)):
        raise ValueError(f"hierarchical config layout is not uniform, non-fork widths {n.unique().tolist()}")
    return n_nf


def _renorm_masked(log_slot: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log-softmax over the unmasked slots; all-masked rows stay -inf."""
    if log_slot.shape[-1] == 0:
        return log_slot
    filled = torch.where(mask, log_slot, float("-inf"))
    has = mask.any(dim=-1, keepdim=True)
    dummy = torch.where(has, filled, torch.zeros_like(filled))
    lse = torch.logsumexp(dummy, dim=-1, keepdim=True)
    return torch.where(has, filled - lse, torch.full_like(log_slot, float("-inf")))


def _log_who_given_kind(who_logits: torch.Tensor, config_mask: torch.Tensor, n_nf: int):
    """Per-kind log-softmax of the who MLP: (not-fork slots, fork slots)."""
    log_slot_nf = _renorm_masked(who_logits[..., :n_nf], config_mask[..., :n_nf])
    log_slot_f = _renorm_masked(who_logits[..., n_nf:], config_mask[..., n_nf:])
    return (log_slot_nf, log_slot_f)


def _compose_kind_who(log_p_kind, log_slot_nf, log_slot_f, config_mask):
    """log P(config) = log P(kind) + log P(who|kind), renormalised over the mask."""
    log_p_c_nf = log_p_kind[..., 0:1] + log_slot_nf
    if log_slot_f.shape[-1] == 0:
        log_p_c_f = log_slot_f
    else:
        log_p_c_f = log_p_kind[..., 1:2] + log_slot_f
    log_p = torch.cat([log_p_c_nf, log_p_c_f], dim=-1)
    log_p_masked = torch.where(config_mask, log_p, float("-inf"))
    row_lse = torch.logsumexp(log_p_masked, dim=-1, keepdim=True)
    log_p = log_p - row_lse
    return log_p.masked_fill(~config_mask, 0.0)


def average_hierarchical_fork_configs(outs: list[dict]) -> dict:
    """Average kind and who|kind, then compose."""
    first = outs[0]
    n_nf = _nonfork_width(first["config_kind"])
    log_p_kind = torch.stack([torch.log_softmax(o["kind_logits"].float(), dim=-1) for o in outs], dim=0).mean(dim=0)
    log_p_kind = log_p_kind - torch.logsumexp(log_p_kind, dim=-1, keepdim=True)
    slots = [_log_who_given_kind(o["who_logits"].float(), first["config_mask"], n_nf) for o in outs]
    log_slot_nf = _renorm_masked(
        torch.stack([s[0] for s in slots], dim=0).mean(dim=0), first["config_mask"][..., :n_nf]
    )
    log_slot_f = _renorm_masked(torch.stack([s[1] for s in slots], dim=0).mean(dim=0), first["config_mask"][..., n_nf:])
    log_p = _compose_kind_who(log_p_kind, log_slot_nf, log_slot_f, first["config_mask"])
    return {
        "config_logits": log_p,
        "config_mask": first["config_mask"],
        "config_kind": first["config_kind"],
        "config_parent": first["config_parent"],
        "config_child1": first["config_child1"],
        "config_child2": first["config_child2"],
    }


def average_fork_configs(outs: list[dict]) -> dict:
    """Average aligned fork outputs."""
    from cell_tracking.linker.support import masked_log_softmax

    first = outs[0]
    for o in outs[1:]:
        for key in ("config_kind", "config_child1", "config_child2", "config_mask"):
            if not torch.equal(o[key], first[key]):
                raise ValueError(f"fork configs are not aligned across members ({key})")
    hier = [_has_kind_head(o) for o in outs]
    if any(hier) and (not all(hier)):
        raise ValueError("cannot ensemble mixed hierarchical and flat fork members")
    if all(hier):
        return average_hierarchical_fork_configs(outs)
    log_p = torch.stack([masked_log_softmax(o["config_logits"], o["config_mask"], dim=-1) for o in outs], dim=0).mean(
        dim=0
    )
    log_p = torch.where(first["config_mask"], log_p, torch.zeros_like(log_p))
    return {
        "config_logits": log_p,
        "config_mask": first["config_mask"],
        "config_kind": first["config_kind"],
        "config_parent": first["config_parent"],
        "config_child1": first["config_child1"],
        "config_child2": first["config_child2"],
    }
