"""3D U-Net decoder blocks (upsample to the skip's shape, concat, two conv-BN-ReLU)."""
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         bias=not use_batchnorm)
        bn = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()
        super().__init__(conv, bn, nn.ReLU(inplace=True))


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_batchnorm: bool = True,
                 interpolation_mode: str = "nearest"):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.conv1 = Conv3dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.conv2 = Conv3dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)

    def forward(self, feature_map: torch.Tensor, target_depth: int, target_height: int, target_width: int,
                skip_connection: Optional[torch.Tensor] = None) -> torch.Tensor:
        feature_map = F.interpolate(feature_map, size=(target_depth, target_height, target_width),
                                    mode=self.interpolation_mode)
        if skip_connection is not None:
            feature_map = torch.cat([feature_map, skip_connection], dim=1)
        return self.conv2(self.conv1(feature_map))


class UnetDecoder(nn.Module):
    """`blocks[i]` takes encoder stage depth-i (upsampled) and the skip of stage depth-i-1."""

    def __init__(self, encoder_channels: Sequence[int], decoder_channels: Sequence[int], n_blocks: int = 5,
                 use_batchnorm: bool = True, interpolation_mode: str = "nearest"):
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError(f'Model depth is {n_blocks}, but you provide `decoder_channels` for '
                             f'{len(decoder_channels)} blocks.')
        encoder_channels = encoder_channels[1:][::-1]     # drop the input-resolution entry, deepest first
        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        self.blocks = nn.ModuleList([
            UnetDecoderBlock(i, s, o, use_batchnorm=use_batchnorm, interpolation_mode=interpolation_mode)
            for i, s, o in zip(in_channels, skip_channels, decoder_channels)])
