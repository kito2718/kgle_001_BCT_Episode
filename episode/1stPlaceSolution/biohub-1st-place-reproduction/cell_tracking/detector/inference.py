"""Bridge a trained unetcellpose checkpoint to the shared detection contract."""

from __future__ import annotations
import contextlib
import numpy as np
import torch
from cell_tracking.common.config import FLOW_SCALE, POOL_SCHEDULE, input_offsets, parse_pool_schedule
from cell_tracking.detector.decode import flow_follow_decode, track_displacement_um
from cell_tracking.common.io import get_pool, normalize, pool_xy, set_pool
from cell_tracking.common.detection_metrics import centroids_vox
from cell_tracking.detector.model import build_backbone

_DECODE_DEFAULTS = {
    "decode_device": None,
    "fg_thresh": 0.5,
    "decode_step_um": None,
    "sink_merge_um": 3.0,
    "decode_min_size": 100,
    "decode_niter": 200,
    "decode_unit_flow": False,
    "decode_quantile_exit": False,
}


class UnetCellPoseModel:
    """A loaded unetcellpose net plus the decode settings it was trained/validated under."""

    __slots__ = ("net", "pool", "rep", "horizons", "offsets", "decode", "meta")

    def __init__(self, net, pool, rep, horizons, decode, meta, offsets=None):
        self.net = net
        self.pool = int(pool)
        self.rep = str(rep)
        self.horizons = int(horizons)
        self.offsets = tuple(range(int(horizons) + 1)) if offsets is None else tuple((int(o) for o in offsets))
        if 0 not in self.offsets:
            raise ValueError(f"offsets {self.offsets} must contain 0 (the frame being decoded)")
        if len(set(self.offsets)) != len(self.offsets) or list(self.offsets) != sorted(self.offsets):
            raise ValueError(f"offsets {self.offsets} must be sorted and unique")
        self.decode = dict(decode)
        self.meta = dict(meta)

    @property
    def ch(self):
        return self.net.ch


def _decode_from(src) -> dict:
    missing = [key for key in _DECODE_DEFAULTS if key != "decode_device" and key not in src]
    if missing:
        raise ValueError(f"missing detector decode metadata: {missing}")
    return {
        "decode_device": src.get("decode_device"),
        **{key: src[key] for key in _DECODE_DEFAULTS if key != "decode_device"},
    }


def load_unetcellpose(ckpt_path, device="cpu", *, expected_pool=None) -> UnetCellPoseModel:
    """Load either a `checkpoint_payload` ``.pt`` or a Lightning ``.ckpt``."""
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "state_dict" in ck and "hyper_parameters" in ck:
        hp = dict(ck["hyper_parameters"])
        required = ("arch", "base", "rep", "context", "horizons", "pool_schedule", "norm", "act")
        missing = [key for key in required if key not in hp]
        if missing:
            raise ValueError(f"{ckpt_path}: missing detector metadata {missing}")
        if "pool" not in hp:
            if expected_pool is None:
                raise ValueError(f"{ckpt_path}: pool metadata is absent; supply the explicit recipe expected_pool")
            hp["pool"] = int(expected_pool)
        expected = {
            "arch": "plain",
            "base": 24,
            "rep": "flow",
            "context": 0,
            "horizons": 0,
            "norm": "batch",
            "act": "relu",
        }
        if any((hp[key] != value for key, value in expected.items())):
            raise ValueError(f"{ckpt_path}: detector architecture differs from the packaged recipe")
        if parse_pool_schedule(hp["pool_schedule"]) != ((1, 2, 2), (1, 2, 2), (2, 2, 2)):
            raise ValueError(f"{ckpt_path}: incompatible detector pooling schedule")
        sd = {k[len("model.") :]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
        rep = hp.get("rep", "flow")
        horizons = int(hp.get("horizons", 0))
        offsets = hp.get("in_offsets") or input_offsets(int(hp.get("context", 0) or 0), horizons)
        net = build_backbone(
            str(hp.get("arch", "plain") or "plain"),
            in_ch=len(offsets),
            in_offsets=tuple(offsets),
            base=int(hp.get("base", 24)),
            horizons=horizons,
            pool_schedule=parse_pool_schedule(hp.get("pool_schedule")),
            rep=rep,
            n_rays=int(hp.get("n_rays", 96) or 96),
            norm=str(hp.get("norm") or "batch"),
            act=str(hp.get("act") or "relu"),
        )
        pool = int(hp["pool"])
        decode = _decode_from(hp)
        meta = {"format": "ckpt", "base": hp.get("base"), "monitor": hp.get("monitor")}
    else:
        required = (
            "state_dict",
            "arch",
            "base",
            "rep",
            "context",
            "horizons",
            "pool",
            "pool_schedule",
            "trunk_norm",
            "trunk_act",
        )
        missing = [key for key in required if key not in ck]
        if missing:
            raise ValueError(f"{ckpt_path}: missing detector metadata {missing}")
        expected = {
            "arch": "plain",
            "base": 24,
            "rep": "flow",
            "context": 0,
            "horizons": 0,
            "trunk_norm": "batch",
            "trunk_act": "relu",
        }
        if any((ck[key] != value for key, value in expected.items())):
            raise ValueError(f"{ckpt_path}: detector architecture differs from the packaged recipe")
        if tuple((tuple(stage) for stage in ck["pool_schedule"])) != ((1, 2, 2), (1, 2, 2), (2, 2, 2)):
            raise ValueError(f"{ckpt_path}: incompatible detector pooling schedule")
        sd = ck["state_dict"]
        rep = ck.get("rep", "flow")
        horizons = int(ck.get("horizons", 0))
        offsets = ck.get("in_offsets") or input_offsets(int(ck.get("context", 0) or 0), horizons)
        in_ch = int(ck.get("in_ch", len(offsets)))
        if in_ch != len(offsets):
            raise RuntimeError(f"{ckpt_path}: in_ch={in_ch} contradicts in_offsets={list(offsets)}")
        net = build_backbone(
            str(ck.get("arch", "plain") or "plain"),
            in_ch=in_ch,
            in_offsets=tuple(offsets),
            base=int(ck.get("base", 24)),
            horizons=horizons,
            pool_schedule=tuple((tuple(s) for s in ck.get("pool_schedule", POOL_SCHEDULE))),
            rep=rep,
            n_rays=int(ck.get("n_rays", 96) or 96),
            norm=str(ck.get("trunk_norm") or "batch"),
            act=str(ck.get("trunk_act") or "relu"),
        )
        pool = int(ck["pool"])
        decode = _decode_from(ck)
        meta = {
            "format": "pt",
            "base": ck.get("base"),
            "monitor": ck.get("monitor"),
            "best_score": ck.get("best_score"),
            "n_train_frames": ck.get("n_train_frames"),
        }
    if expected_pool is not None and pool != int(expected_pool):
        raise ValueError(f"{ckpt_path}: checkpoint pool {pool} differs from recipe {expected_pool}")
    if rep != "flow":
        raise ValueError(f"{ckpt_path}: expected flow representation, got {rep!r}")
    if "out.weight" in sd:
        raise ValueError(f"{ckpt_path}: incompatible fused output head")
    net.load_state_dict(sd, strict=True)
    net.eval().to(device)
    return UnetCellPoseModel(net, pool, rep, horizons, decode, meta, offsets=offsets)


def pooled_to_native(cent_pooled, pool):
    """Pooled-grid centroids -> native full-resolution voxel coordinates."""
    c = np.asarray(cent_pooled, dtype=np.float64)
    if c.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    out = c.copy()
    out[:, 1] = c[:, 1] * pool + (pool - 1) / 2.0
    out[:, 2] = c[:, 2] * pool + (pool - 1) / 2.0
    return out


def _input_stack(fsm: UnetCellPoseModel, x0, frame_at):
    """The ``(C, Z, Y, X)`` model input for a frame whose pooled/normalised volume is `x0`."""
    if len(fsm.offsets) == 1:
        return x0[None]
    chans = []
    for dt in fsm.offsets:
        v = None if dt == 0 or frame_at is None else frame_at(int(dt))
        chans.append(x0 if v is None else normalize(pool_xy(np.asarray(v), f=fsm.pool)))
    return np.ascontiguousarray(np.stack(chans))


def decode_instances(
    fsm: UnetCellPoseModel, vol, device="cpu", frame_at=None, _return_pred=False, _return_d1=False, **overrides
):
    """One native-resolution volume -> pooled-grid int32 instance labels."""
    d = dict(fsm.decode)
    d.update({k: v for k, v in overrides.items() if v is not None})
    prev_pool = get_pool()
    set_pool(fsm.pool)
    try:
        x = normalize(pool_xy(np.asarray(vol), f=fsm.pool))
        with torch.no_grad():
            t = torch.from_numpy(_input_stack(fsm, x, frame_at))[None].to(device)
            if _return_d1:
                if not hasattr(fsm.net, "forward_with_d1"):
                    raise NotImplementedError(
                        f"{type(fsm.net).__name__} has no forward_with_d1; edge-feature caching needs a single unetcellpose backbone checkpoint"
                    )
                pred_t, d1_t = fsm.net.forward_with_d1(t)
                pred = pred_t[0].float().cpu().numpy()
                d1 = d1_t[0].float().cpu().numpy()
            else:
                pred = fsm.net(t)[0].float().cpu().numpy()
                d1 = None
        ch = fsm.ch
        inst = flow_follow_decode(
            pred[ch.flow],
            pred[ch.prob],
            fg_thresh=d["fg_thresh"],
            step_um=d["decode_step_um"],
            sink_merge_um=d["sink_merge_um"],
            min_size=d["decode_min_size"],
            niter=d["decode_niter"],
            flow_scale=FLOW_SCALE,
            unit_flow=d["decode_unit_flow"],
            device=d["decode_device"],
            quantile_exit=d["decode_quantile_exit"],
            contact_logit=pred[ch.contact] if ch.contact is not None else None,
            contact_cut=d.get("contact_cut", 0.5),
        )
        if _return_pred and _return_d1:
            return (inst, pred, d1)
        if _return_pred:
            return (inst, pred)
        if _return_d1:
            return (inst, d1)
        return inst
    finally:
        set_pool(prev_pool)


def detect_unetcellpose(fsm: UnetCellPoseModel, vol, device="cpu", frame_at=None, **overrides):
    """One native-resolution volume -> ``(n, 3)`` native-voxel centroids ``(Z, Y, X)``."""
    inst = decode_instances(fsm, vol, device=device, frame_at=frame_at, **overrides)
    return pooled_to_native(centroids_vox(inst), fsm.pool)


def detect_unetcellpose_conf(fsm: UnetCellPoseModel, vol, device="cpu", frame_at=None, **overrides):
    """`detect_unetcellpose` plus a per-instance confidence: ``(coords_native, conf)``."""
    inst, pred = decode_instances(fsm, vol, device=device, frame_at=frame_at, _return_pred=True, **overrides)
    coords = pooled_to_native(centroids_vox(inst), fsm.pool)
    ids = np.unique(inst)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return (coords, np.zeros(0, np.float64))
    p = 1.0 / (1.0 + np.exp(-np.asarray(pred[fsm.ch.prob], np.float64)))
    flat = inst.ravel()
    n = int(inst.max()) + 1
    sums = np.bincount(flat, weights=p.ravel(), minlength=n)
    counts = np.bincount(flat, minlength=n)
    return (coords, sums[ids] / np.maximum(counts[ids], 1))


def detect_unetcellpose_motion(fsm: UnetCellPoseModel, vol, device="cpu", frame_at=None, split_um=None, **overrides):
    """`detect_unetcellpose` plus the net's own predicted motion: ``(coords_native, disp_um)``."""
    if int(fsm.horizons) < 1:
        raise ValueError(
            f"detect_unetcellpose_motion needs a horizons>=1 checkpoint (this one has horizons={fsm.horizons}, i.e. no track head). Train with --horizons 1."
        )
    prev_pool = get_pool()
    set_pool(fsm.pool)
    try:
        inst, pred = decode_instances(fsm, vol, device=device, frame_at=frame_at, _return_pred=True, **overrides)
        _, disp = track_displacement_um(inst, pred[fsm.ch.track(1)], split_um=split_um)
    finally:
        set_pool(prev_pool)
    return (pooled_to_native(centroids_vox(inst), fsm.pool), disp)


TTA_SETS = {
    "none": ((),),
    "xy": ((), (1,), (2,), (1, 2)),
    "full": ((), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)),
}


class EnsembleTTANet(torch.nn.Module):
    """Average several unetcellpose nets, optionally over flip-TTA, in FIELD space."""

    def __init__(self, nets, ch, flips=TTA_SETS["none"], autocast="none"):
        super().__init__()
        self.nets = torch.nn.ModuleList(list(nets))
        self.ch = ch
        self.flips = tuple((tuple(f) for f in flips))
        if not self.flips:
            raise ValueError("flips must contain at least the identity ()")
        a = "none" if autocast is None else str(autocast)
        if a not in ("none", "fp16", "bf16"):
            raise ValueError(f"autocast must be one of none/fp16/bf16, got {autocast!r}")
        self.autocast = a

    def _member_ctx(self, device):
        if self.autocast == "none" or device.type != "cuda":
            return contextlib.nullcontext()
        dt = torch.float16 if self.autocast == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dt)

    def forward(self, t):
        inst, prob_i = (self.ch.inst, self.ch.prob)
        acc, n = (None, 0)
        ctx = self._member_ctx(t.device)
        for f in self.flips:
            ti = torch.flip(t, dims=[2 + a for a in f]) if f else t
            for net in self.nets:
                with ctx:
                    p = net(ti)
                p = p.float()
                if f:
                    p = torch.flip(p, dims=[2 + a for a in f]).contiguous()
                    if self.ch.rep == "flow":
                        for a in f:
                            p[:, inst.start + a].neg_()
                p[:, prob_i] = torch.sigmoid(p[:, prob_i])
                acc = p if acc is None else acc.add_(p)
                n += 1
        acc.div_(n)
        pr = acc[:, prob_i].clamp(1e-06, 1 - 1e-06)
        acc[:, prob_i] = torch.log(pr / (1 - pr))
        return acc


def load_unetcellpose_ensemble(
    ckpt_paths, device="cpu", tta="none", autocast="none", *, expected_pool=None
) -> UnetCellPoseModel:
    """Load N checkpoints as one UnetCellPoseModel whose net averages them (optionally with flip TTA)."""
    paths = [str(p) for p in ckpt_paths]
    if not paths:
        raise ValueError("no checkpoints given")
    members = [load_unetcellpose(p, device=device, expected_pool=expected_pool) for p in paths]
    head = members[0]
    for p, m in zip(paths[1:], members[1:]):
        for attr in ("pool", "rep", "offsets"):
            if getattr(m, attr) != getattr(head, attr):
                raise SystemExit(
                    f"ensemble member {p} has {attr}={getattr(m, attr)!r}, first member has {getattr(head, attr)!r} -- refusing to average fields that are not the same quantity on one grid"
                )
        if m.ch.out_ch != head.ch.out_ch:
            raise SystemExit(f"ensemble member {p} has out_ch={m.ch.out_ch}, first has {head.ch.out_ch}")
        if m.decode != head.decode:
            raise ValueError(f"ensemble member {p} has different decode settings")
    if tta not in TTA_SETS:
        raise SystemExit(f"--tta must be one of {sorted(TTA_SETS)}, got {tta!r}")
    net = EnsembleTTANet([m.net for m in members], head.ch, TTA_SETS[tta], autocast=autocast).eval().to(device)
    meta = dict(head.meta)
    meta.update(
        {
            "format": "ensemble",
            "n_members": len(members),
            "tta": tta,
            "autocast": autocast,
            "members": paths,
            "n_forward_per_volume": len(members) * len(TTA_SETS[tta]),
        }
    )
    return UnetCellPoseModel(net, head.pool, head.rep, head.horizons, head.decode, meta, offsets=head.offsets)
