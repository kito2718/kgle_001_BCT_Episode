"""2.5D encoder: a timm EfficientNet run per z-slice, features z-pooled by 2 at every stage.

`in_channels` > 1 stacks that many ADJACENT z-slices as the 2D channels (SlidingSlicer). forward returns
[input, stage1, ..., stage<depth>] as (B, C, D, H, W) tensors.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models import EfficientNetFeatures


class SlidingSlicer(nn.Module):
    """Fixed one-hot (slice_size, 1, 1) conv: plane i of the output is the input shifted by i - slice_size // 2 in z."""

    def __init__(self, slice_size=3, stride=1, trainable=False):
        super().__init__()
        self.conv = nn.Conv3d(1, slice_size, kernel_size=(slice_size, 1, 1), stride=(stride, 1, 1),
                              bias=False, padding=(slice_size // 2, 0, 0))
        with torch.no_grad():
            self.conv.weight.data.fill_(0)
            for i in range(slice_size):
                self.conv.weight.data[i, 0, i] = 1
        if not trainable:
            for param in self.conv.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.conv(x)


class TimmUniversalEncoder(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, in_channels: int = 3, in_planes: int = 1,
                 depth: int = 5, iso_z: bool = False, slicer_trainable: bool = False):
        super().__init__()
        if not 1 <= depth <= 5:
            raise ValueError(f'encoder depth should be in [1, 5], got {depth}')
        if in_planes != 1 or iso_z:
            raise ValueError('only the single-plane, z-pooled 2.5D encoder is supported')
        self.name = name
        self.model = timm.create_model(name, in_chans=in_channels, features_only=True, pretrained=pretrained,
                                       out_indices=tuple(range(depth)))
        if not isinstance(self.model, EfficientNetFeatures) \
                or list(self.model.feature_info.reduction()) != [2 ** (i + 1) for i in range(depth)]:
            raise ValueError(f'{name}: only EfficientNet-style encoders (stride 2 per stage) are supported')
        self._out_channels = [in_channels] + self.model.feature_info.channels()
        self._strides = [2] * depth
        self.in_channels = in_channels
        if in_channels > 1:
            self.sliding_slicer = SlidingSlicer(slice_size=in_channels, stride=1, trainable=slicer_trainable)

    @property
    def out_channels(self) -> list[int]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.in_channels > 1:
            x = self.sliding_slicer(x)
        B = x.shape[0]
        features = [x]
        x = rearrange(x, 'b c d h w -> (b d) c h w')
        x = self.model.bn1(self.model.conv_stem(x))
        if 0 in self.model._stage_out_idx:
            raise ValueError('Unsupported model downsampling pattern.')
        stage = 0
        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i + 1 in self.model._stage_out_idx:
                x = rearrange(x, '(b d) c h w -> b c d h w', b=B)
                stride = self._strides[stage]
                if stride > 1:
                    x = F.avg_pool3d(x, kernel_size=(stride, 1, 1), stride=(stride, 1, 1), padding=0)
                features.append(x)
                x = rearrange(x, 'b c d h w -> (b d) c h w')
                stage += 1
        return features
