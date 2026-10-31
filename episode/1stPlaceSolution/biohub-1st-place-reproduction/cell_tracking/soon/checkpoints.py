from __future__ import annotations
import os
import torch
from cell_tracking.common.config import SOON_CROP_SIZE as CROP_SIZE
from cell_tracking.soon.model import SiameseSpaceTimeSoonNet

KIND = "soon_crop"
FOURTH_CHANNELS = ("gaussian",)


def deploy_context(meta):
    """Read the required temporal input context."""
    if meta["context"] != 2:
        raise ValueError("soon checkpoints require context=2")
    return int(meta["context"])


def load_soon_crop(path, device="cpu"):
    """Load a teacher or final soon checkpoint with strict recipe metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "kind": KIND,
        "arch": "soon_stcausal",
        "features": "intensity",
        "in_ch": 4,
        "base": 16,
        "feat_dim": 96,
        "n_frames": 3,
        "num_classes": 3,
        "attn": "softmax",
        "cond_encoder": False,
        "aniso_pool": "mid",
        "encoder": "cnn",
        "num_layers": 4,
        "nheads": 4,
        "causal": False,
        "query_tag": True,
        "st_rope": True,
        "st_rope_base": 32.0,
        "st_swiglu": True,
        "st_layerscale": 0.0,
        "xy_pool": 1,
        "context": 2,
        "fourth_channel": "gaussian",
        "aux_class_aware": True,
        "pool": 2,
    }
    for key, value in expected.items():
        if key not in checkpoint or checkpoint[key] != value:
            raise ValueError(f"{path}: expected {key}={value!r}, got {checkpoint.get(key)!r}")
    for key in ("crop_size", "st_registers", "state_dict"):
        if key not in checkpoint:
            raise ValueError(f"{path}: missing checkpoint metadata: {key}")
    if tuple(checkpoint["crop_size"]) != CROP_SIZE:
        raise ValueError(f"{path}: expected crop_size={CROP_SIZE}")
    if checkpoint["st_registers"] not in (0, 4):
        raise ValueError(f"{path}: expected zero teacher registers or four final registers")
    for key in ("hierarchical", "independent_heads", "role_masked"):
        if checkpoint.get(key, False):
            raise ValueError(f"{path}: unsupported {key}")
    if checkpoint.get("encoder_kw"):
        raise ValueError(f"{path}: unsupported encoder options")
    net = SiameseSpaceTimeSoonNet(num_registers=checkpoint["st_registers"], dropout=0.0)
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.to(device).eval()
    meta = {
        key: checkpoint[key]
        for key in (
            "kind",
            "arch",
            "features",
            "pool",
            "in_ch",
            "base",
            "num_classes",
            "n_frames",
            "context",
            "fourth_channel",
            "attn",
            "cond_encoder",
            "aniso_pool",
            "encoder",
        )
    }
    meta.update(crop_size=CROP_SIZE, encoder_kw={}, role_masked=False, role_ball_thresh=0.2)
    return (net, meta)


def save_soon_crop(
    path,
    net,
    *,
    crop_size=CROP_SIZE,
    features="intensity",
    pool=2,
    context=2,
    fourth_channel="gaussian",
    val_frac=0.2,
    val_embryo="6bba",
    train_loss=None,
    val_loss=None,
    best_epoch=None,
    extra=None,
):
    """Every saved checkpoint keeps its full state_dict, ``aux_*`` weights included."""
    if fourth_channel is not None and fourth_channel not in FOURTH_CHANNELS:
        raise ValueError(f"fourth_channel must be one of {FOURTH_CHANNELS} or None, got {fourth_channel!r}")
    sd = net.state_dict()
    payload = {
        "format": "soon_crop_v1",
        "kind": KIND,
        "arch": net.arch,
        "feat_dim": net.feat_dim,
        "n_frames": int(net.n_frames),
        "state_dict": sd,
        "crop_size": tuple((int(s) for s in crop_size)),
        "features": str(features),
        "pool": int(pool),
        "in_ch": int(net.in_ch),
        "base": int(net.base),
        "num_classes": int(net.num_classes),
        "attn": str(net.attn),
        "cond_encoder": bool(net.cond_encoder),
        "aniso_pool": net.aniso_pool,
        "encoder": str(net.encoder_kind),
        "encoder_kw": dict(net.encoder_kw or {}),
        "role_masked": bool(net.role_masked),
        "role_ball_thresh": float(net.role_ball_thresh),
        "num_layers": int(net.num_layers),
        "nheads": int(net.nheads),
        "causal": bool(net.causal),
        "query_tag": bool(net.query_tag),
        "st_rope": bool(net.use_rope),
        "st_swiglu": bool(net.use_swiglu),
        "st_layerscale": float(net.layerscale_init),
        "st_registers": int(net.num_registers),
        "st_rope_base": float(net.rope_base),
        "aux_spheres": bool(net.aux_spheres),
        "xy_pool": int(net.xy_pool),
        "aux_class_aware": bool(net.aux_class_aware),
        "context": None if context is None else int(context),
        "fourth_channel": fourth_channel,
        "val_frac": float(val_frac),
        "val_embryo": str(val_embryo),
        "train_loss": None if train_loss is None else float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        "best_epoch": None if best_epoch is None else int(best_epoch),
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)
    return path
