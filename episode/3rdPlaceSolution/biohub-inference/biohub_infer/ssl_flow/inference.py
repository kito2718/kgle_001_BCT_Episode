"""flow_164 checkpoint loading and per-point flow offsets."""
import torch
from omegaconf import OmegaConf

from biohub_infer.core.dataset import to_ds
from .data import downsample_mean, normalize
from .model import build_flow_model
from .ops import sample_volume


def load_model(checkpoint, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = build_flow_model(OmegaConf.to_container(cfg.model)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, cfg, device


@torch.no_grad()
def point_priors(result, points_zyx, downsample=(1, 1, 1)):
    """Return point flow, warped positions, and descriptors in original voxels.

    ``points_zyx`` and returned coordinates are original-volume coordinates;
    model tensors live on the mean-pooled grid.
    """
    device = result["flow"].device
    ds = torch.tensor(downsample, device=device, dtype=torch.float32)
    pts = torch.as_tensor(points_zyx, device=device, dtype=torch.float32)
    grid_pts = torch.as_tensor(to_ds(pts.cpu().numpy(), downsample), device=device,
                               dtype=torch.float32)
    flow_grid = sample_volume(result["flow"], grid_pts)[0]
    descriptors = torch.nn.functional.normalize(
        sample_volume(result["features_a"], grid_pts)[0], dim=1)
    flow_orig = flow_grid * ds
    return {"flow_zyx": flow_orig.cpu().numpy(),
            "warped_zyx": (pts + flow_orig).cpu().numpy(),
            "descriptor": descriptors.cpu().numpy()}


def prepare_zarr_frame(array, t, quantiles, downsample):
    qlo, qhi = float(quantiles["0.001"]), float(quantiles["0.999"])
    return downsample_mean(normalize(array[t], qlo, qhi), downsample)
