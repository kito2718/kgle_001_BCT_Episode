"""The matcher trunk: a 2.5D U-Net whose decoder stops at the tapped block, plus the NodeMatcher head.

forward(x) -> (None, embedding (B, C, z', y', x')): the embedding is the decoder feature of block
`embed_tap` (embed_head_arch "none" = identity head), sampled at the detections by
embedding.sample_embeddings. The stride-1 block and `segmentation_head` (the detection heatmap the trunk
was trained with) are kept only so the checkpoint loads strictly; they never run.
"""
import math

import torch
import torch.nn as nn

from .decoder import UnetDecoder
from .encoder import TimmUniversalEncoder


class EmbedUnet(nn.Module):
    def __init__(self, embed_dim: int = 32, embed_head_arch: str = "linear", freeze_trunk: bool = False,
                 matcher: str = "none", matcher_drift_dpos: bool = False, matcher_dpos_noise_um: float = 0.0,
                 matcher_attn_window_um: float = 0.0, matcher_drift_frame: bool = False,
                 matcher_cls_head: bool = False, matcher_role_embed: bool = False, matcher_d: int = 128,
                 embed_tap: int = -2, encoder_name: str = "resnet34", encoder_depth: int = 5,
                 decoder_channels=(256, 128, 64, 32, 16), in_channels: int = 3, slicer_trainable: bool = False,
                 classes: int = 1, pretrained: bool = True, iso_z: bool = False, iso_z_zmix: bool = True):
        super().__init__()
        if embed_head_arch != "none" or matcher not in ("xattn", "xattn_cos"):
            raise ValueError(f'unsupported matcher trunk: embed_head_arch={embed_head_arch}, matcher={matcher}')
        self.encoder = TimmUniversalEncoder(name=encoder_name, in_channels=in_channels, depth=encoder_depth,
                                            pretrained=pretrained, iso_z=iso_z, slicer_trainable=slicer_trainable)
        self.decoder = UnetDecoder(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels,
                                   n_blocks=encoder_depth)
        self.segmentation_head = nn.Conv3d(decoder_channels[-1], classes, kernel_size=3, padding=1)
        from ..matcher import NodeMatcher
        embed_dim = decoder_channels[int(embed_tap)]          # identity head: the tap feature is the embedding
        self.matcher = NodeMatcher(feat_dim=embed_dim, d=int(matcher_d), cos_prior=(matcher == "xattn_cos"),
                                   drift_dpos=matcher_drift_dpos, dpos_noise_um=matcher_dpos_noise_um,
                                   attn_window_um=matcher_attn_window_um, align_pos_encoding=matcher_drift_frame,
                                   cls_head=matcher_cls_head, role_embed=matcher_role_embed)
        self.matcher_drift_frame = bool(matcher_drift_frame)
        self.embed_tap = int(embed_tap)
        self.embed_head = nn.Identity()
        self.embed_logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.embed_null = nn.Parameter(torch.tensor(0.0))
        self.freeze_trunk = bool(freeze_trunk)
        self.embed_only_forward = True

    def forward(self, x: torch.Tensor):
        return None, self.embed_head(self._forward_to_tap(x))

    def _forward_to_tap(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder + decoder blocks up to and including the tapped one; returns the tap feature."""
        features = self.encoder(x)
        shapes = [f.shape[2:] for f in features][::-1]
        feats = features[1:][::-1]
        h, skips = feats[0], feats[1:]
        stop = self.embed_tap % len(self.decoder.blocks) + 1
        for i, block in enumerate(self.decoder.blocks[:stop]):
            skip = skips[i] if i < len(skips) else None
            h = block(h, *shapes[i + 1], skip_connection=skip)
        return h
