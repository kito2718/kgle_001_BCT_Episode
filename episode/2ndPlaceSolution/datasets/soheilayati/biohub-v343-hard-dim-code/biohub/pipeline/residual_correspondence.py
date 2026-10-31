"""Image residuals around an already trained sparse-safe edge probability."""
import torch
from biohub.pipeline.supervised_patch_correspondence import PatchCorrespondence


def residual_model(state_dict,scalar_features=51):
    model=PatchCorrespondence(scalar_features)
    model.load_state_dict(state_dict)
    torch.nn.init.zeros_(model.head[-1].weight)
    torch.nn.init.zeros_(model.head[-1].bias)
    return model


def combined_logit(base_probability,residual):
    return torch.logit(base_probability.clamp(1e-6,1-1e-6))+residual
