"""Directly supervised image-pair correspondence on a fixed candidate surface."""
import torch
from torch import nn


class PatchCorrespondence(nn.Module):
    def __init__(self, scalar_features=51):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv3d(2,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv3d(16,24,3,stride=2,padding=1),nn.GroupNorm(4,24),nn.SiLU(),
            nn.Conv3d(24,32,3,stride=2,padding=1),nn.GroupNorm(8,32),nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),nn.Flatten())
        self.scalars=nn.Sequential(nn.Linear(2*scalar_features,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(160,64),nn.SiLU(),nn.Linear(64,1))

    def encode(self, patch):
        patch=patch.float()
        mean=patch.mean(dim=(2,3,4),keepdim=True)
        std=patch.std(dim=(2,3,4),keepdim=True,unbiased=False).clamp_min(.001)
        return self.encoder((patch-mean)/std)

    def pair(self, parent, child, scalar):
        return self.head(torch.cat([parent,child,(parent-child).abs(),parent*child,self.scalars(scalar)],dim=1)).squeeze(-1)

    def forward(self,parent_patch,child_patch,scalar):
        return self.pair(self.encode(parent_patch),self.encode(child_patch),scalar)
