"""The matcher network: a 2.5D UNet trunk with an embedding tap and the NodeMatcher head (EmbedUnet).

Input (B, 1, Z, Y, X) float. The timm 2D encoder runs per z-slice with depth avg-pooling; the 3D conv
decoder fuses across z. Every spatial dim must be divisible by 2**encoder_depth.
"""
import torch


def get_model_from_cfg(cfg, resume_path=None):
    """EmbedUnet from cfg.model (restored from the checkpoint's hyper-parameters), weights from resume_path."""
    from .unet.embed_unet import EmbedUnet
    m = cfg.model
    if str(m.get('temporal', 'none')) != 'siamese' or str(m.get('arch', 'unet25d')) != 'unet25d':
        raise ValueError(f'not a 2.5D siamese matcher config: temporal={m.get("temporal")} arch={m.get("arch")}')
    if m.get('amp_fp32_modules'):
        raise ValueError('model.amp_fp32_modules is not supported')
    model = EmbedUnet(
        embed_dim=int(m.get("embed_dim", 32)),
        embed_head_arch=str(m.get("embed_head_arch", "linear")),
        freeze_trunk=bool(m.get("freeze_trunk", False)),
        matcher=str(m.get("matcher", "none")),
        matcher_drift_dpos=bool(m.get("matcher_drift_dpos", False)),
        matcher_dpos_noise_um=float(m.get("matcher_dpos_noise_um", 0.0)),
        matcher_attn_window_um=float(m.get("matcher_attn_window_um", 0.0)),
        matcher_drift_frame=bool(m.get("matcher_drift_frame", False)),
        matcher_cls_head=bool(m.get("matcher_cls_head", False)),
        matcher_role_embed=bool(m.get("matcher_role_embed", False)),
        matcher_d=int(m.get("matcher_d", 128)),
        embed_tap=int(m.get("embed_tap", -2)),
        encoder_name=m.encoder_name,
        encoder_depth=int(m.encoder_depth),
        decoder_channels=tuple(m.decoder_channels),
        in_channels=int(m.in_channels),
        slicer_trainable=bool(m.get("slicer_trainable", False)),
        classes=1,
        pretrained=bool(m.pretrained),
        iso_z=bool(m.get("iso_z", False)),
        iso_z_zmix=bool(m.get("iso_z_zmix", True)),
    )
    if resume_path:
        model.load_state_dict(_resume_state(resume_path))
        print(f"loaded weights from {resume_path}")
    return model


def _resume_state(resume_path):
    """Lightning checkpoint -> model state dict; EMA weights win when the checkpoint has them."""
    ckpt = torch.load(str(resume_path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    ema = {k.removeprefix("model_ema.module."): v for k, v in state.items()
           if k.startswith("model_ema.module.")}
    if ema:
        return ema
    return {k.removeprefix("model."): v for k, v in state.items()
            if not k.startswith("model_ema.")}
